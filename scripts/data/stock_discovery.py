import os
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
import numpy as np
from collections import defaultdict
from pathlib import Path
import json
import time
import warnings
warnings.filterwarnings('ignore', message='Failed to fetch')

logging.basicConfig(level=logging.INFO)

# Liquidity thresholds — override via environment variables
_MIN_AVG_DAILY_VOLUME = int(os.getenv('MIN_AVG_DAILY_VOLUME_STOCKS', '1_000_000'.replace('_', '')))
_MIN_MARKET_CAP       = int(os.getenv('MIN_MARKET_CAP',               '500_000_000'.replace('_', '')))

_DELISTED_CACHE_PATH = Path(__file__).resolve().parent.parent / 'logs' / 'delisted_cache.json'
_DISCOVERY_CACHE_PATH = Path(__file__).resolve().parent.parent / 'logs' / 'discovery_cache.json'
_DISCOVERY_CACHE_TTL_HOURS = 4  # Reuse discovery results for up to 4 hours


def passes_liquidity_filter(symbol: str, info: dict) -> bool:
    """Return True if symbol meets minimum volume and market-cap thresholds.

    Checks `averageVolume` and `marketCap` from a yfinance `.info` dict.
    Missing or None values are treated as 0 (fail).
    """
    avg_vol = info.get('averageVolume') or 0
    mkt_cap = info.get('marketCap') or 0
    if avg_vol < _MIN_AVG_DAILY_VOLUME:
        logging.debug(f"{symbol}: liquidity fail — avgVol {avg_vol:,} < {_MIN_AVG_DAILY_VOLUME:,}")
        return False
    if mkt_cap < _MIN_MARKET_CAP:
        logging.debug(f"{symbol}: liquidity fail — mktCap {mkt_cap:,} < {_MIN_MARKET_CAP:,}")
        return False
    return True


def _load_delisted_cache() -> set:
    """Load persisted denylist of delisted/unfound tickers."""
    try:
        if _DELISTED_CACHE_PATH.exists():
            data = json.loads(_DELISTED_CACHE_PATH.read_text())
            return set(data.get('delisted', []))
    except Exception:
        pass
    return set()


def _save_delisted_cache(delisted: set) -> None:
    """Persist denylist to disk."""
    try:
        _DELISTED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _DELISTED_CACHE_PATH.write_text(
            json.dumps({'delisted': sorted(delisted), 'updated': datetime.now().isoformat()}, indent=2)
        )
    except Exception as e:
        logging.warning(f"Could not save delisted cache: {e}")


def _load_discovery_cache() -> dict | None:
    """
    Return cached discovery results if they are younger than the TTL, else None.
    The cache stores the full opportunities dict so signals survive across sessions.
    """
    try:
        if _DISCOVERY_CACHE_PATH.exists():
            data = json.loads(_DISCOVERY_CACHE_PATH.read_text())
            cached_at = datetime.fromisoformat(data.get('timestamp', '2000-01-01'))
            age_hours = (datetime.now() - cached_at).total_seconds() / 3600
            if age_hours <= _DISCOVERY_CACHE_TTL_HOURS:
                logging.info(
                    f"♻️  Using cached discovery results ({age_hours:.1f}h old, "
                    f"TTL={_DISCOVERY_CACHE_TTL_HOURS}h): "
                    f"{data.get('total_discovered', 0)} stocks"
                )
                return data
    except Exception as e:
        logging.debug(f"Could not load discovery cache: {e}")
    return None


def _save_discovery_cache(data: dict) -> None:
    """Persist discovery results to disk so the next session can reuse them."""
    try:
        _DISCOVERY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _DISCOVERY_CACHE_PATH.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logging.warning(f"Could not save discovery cache: {e}")


class StockDiscovery:
    def __init__(self):
        self.discovered_stocks = []
        self.opportunities = defaultdict(list)
        self.full_universe = []
        self._delisted: set = _load_delisted_cache()
        if self._delisted:
            logging.info(f"🚫 Loaded {len(self._delisted)} known delisted/unfound tickers from cache")

    def _mark_delisted(self, symbol: str) -> None:
        """Add a ticker to the denylist and persist it."""
        if symbol not in self._delisted:
            self._delisted.add(symbol)
            logging.info(f"🗑️  Marking {symbol} as delisted/unfound — added to denylist")
            _save_delisted_cache(self._delisted)

    def _is_active(self, symbol: str, period: str = '5d') -> bool:
        """
        Return True if yfinance returns price data for this symbol.
        Marks the symbol as delisted if no data is found.
        """
        if symbol in self._delisted:
            return False
        try:
            hist = yf.Ticker(symbol).history(period=period)
            if hist.empty:
                self._mark_delisted(symbol)
                return False
            return True
        except Exception:
            self._mark_delisted(symbol)
            return False
    
    def get_full_market_universe(self):
        """Get complete list of all tradeable US stocks."""
        logging.info("📡 Fetching full market universe...")
        
        all_stocks = []
        
        # Method 1: Get from NASDAQ FTP
        try:
            # NASDAQ listed stocks
            nasdaq_url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
            nasdaq_df = pd.read_csv(nasdaq_url, sep='|')
            nasdaq_df = nasdaq_df[nasdaq_df['Test Issue'] == 'N']
            nasdaq_symbols = nasdaq_df['Symbol'].astype(str).tolist()
            
            # Other exchanges
            other_url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt"
            other_df = pd.read_csv(other_url, sep='|')
            other_df = other_df[other_df['Test Issue'] == 'N']
            other_symbols = other_df['ACT Symbol'].astype(str).tolist()
            
            all_stocks = nasdaq_symbols + other_symbols
            
            # Clean symbols - remove special chars, keep only valid tickers
            all_stocks = [
                s.strip() for s in all_stocks 
                if isinstance(s, str) and 
                s.strip().replace('.', '').replace('-', '').isalpha() and 
                len(s.strip()) <= 5 and
                s.strip() not in ['nan', 'NaN', '']
            ]
            
            logging.info(f"✅ Fetched {len(all_stocks)} stocks from NASDAQ FTP")
        
        except Exception as e:
            logging.warning(f"NASDAQ FTP failed: {e}, using fallback method")
            all_stocks = self.get_index_constituents()
        
        self.full_universe = sorted(list(set(all_stocks)))
        logging.info(f"📊 Total market universe: {len(self.full_universe)} stocks")
        
        return self.full_universe
    
    def get_index_constituents(self):
        """Fallback: Get stocks from major indices."""
        logging.info("📡 Fetching index constituents...")
        
        constituents = []
        
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; research-bot/1.0)'}

        # S&P 500
        try:
            html = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers=headers, timeout=10).text
            sp500_symbols = pd.read_html(html)[0]['Symbol'].str.replace('.', '-').tolist()
            constituents.extend(sp500_symbols)
            logging.info(f"  ✅ S&P 500: {len(sp500_symbols)} stocks")
        except Exception as e:
            logging.warning(f"Failed to fetch S&P 500: {e}")

        # NASDAQ 100
        try:
            html = requests.get('https://en.wikipedia.org/wiki/Nasdaq-100', headers=headers, timeout=10).text
            tables = pd.read_html(html)
            nasdaq100_df = next((t for t in tables if 'Ticker' in t.columns), None)
            if nasdaq100_df is not None:
                nasdaq100_symbols = nasdaq100_df['Ticker'].dropna().tolist()
                constituents.extend(nasdaq100_symbols)
                logging.info(f"  ✅ NASDAQ 100: {len(nasdaq100_symbols)} stocks")
        except Exception as e:
            logging.warning(f"Failed to fetch NASDAQ 100: {e}")

        # Dow Jones 30
        try:
            html = requests.get('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average', headers=headers, timeout=10).text
            tables = pd.read_html(html)
            dow_df = next((t for t in tables if 'Symbol' in t.columns), None)
            if dow_df is not None:
                dow_symbols = dow_df['Symbol'].dropna().tolist()
                constituents.extend(dow_symbols)
                logging.info(f"  ✅ Dow 30: {len(dow_symbols)} stocks")
        except Exception as e:
            logging.warning(f"Failed to fetch Dow 30: {e}")
        
        # Add popular growth/tech stocks
        popular_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'AMD', 'NFLX', 'DIS',
            'BABA', 'COIN', 'PLTR', 'SNOW', 'CRWD', 'NET', 'DDOG', 'ZS', 'SHOP', 'SQ',
            'PYPL', 'ADBE', 'CRM', 'ORCL', 'INTC', 'QCOM', 'AVGO', 'TXN', 'MU', 'LRCX',
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'SCHW',
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY', 'MPC', 'VLO', 'PSX', 'HAL',
            'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'DHR', 'AMGN',
            'BA', 'CAT', 'GE', 'MMM', 'HON', 'UNP', 'UPS', 'FDX', 'LMT', 'RTX',
            'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'CVS', 'WBA', 'COST',
            'V', 'MA', 'AXP', 'BLK', 'SPGI', 'ICE', 'CME', 'MCO', 'MSCI', 'NDAQ'
        ]
        constituents.extend(popular_stocks)
        logging.info(f"  ✅ Popular stocks: {len(popular_stocks)} added")
        
        return list(set(constituents))
    
    def get_actively_traded_universe(self, min_volume=500000, max_stocks=2000):
        """Filter universe to actively traded stocks."""
        logging.info(f"🔍 Filtering for liquid stocks (min volume: {min_volume:,})...")
        
        if not self.full_universe:
            self.get_full_market_universe()
        
        liquid_stocks = []
        batch_size = 50
        
        # Process in batches
        total_to_scan = min(len(self.full_universe), max_stocks)
        
        for i in range(0, total_to_scan, batch_size):
            batch = self.full_universe[i:i+batch_size]

            for symbol in batch:
                if symbol in self._delisted:
                    continue
                try:
                    stock = yf.Ticker(symbol)
                    hist = stock.history(period='5d')

                    if hist.empty:
                        self._mark_delisted(symbol)
                        continue

                    if len(hist) >= 3:
                        avg_volume = hist['Volume'].mean()
                        current_price = hist['Close'].iloc[-1]

                        if (avg_volume >= min_volume and
                                current_price >= 5.0 and
                                current_price <= 2000.0):
                            liquid_stocks.append(symbol)

                except Exception as e:
                    logging.debug(f"Skipping {symbol} in liquidity filter: {e}")
                    continue
            
            if (i // batch_size) % 10 == 0 and i > 0:
                logging.info(f"  Processed {i}/{total_to_scan}, found {len(liquid_stocks)} liquid stocks")
                time.sleep(0.5)
        
        logging.info(f"✅ Filtered to {len(liquid_stocks)} actively traded stocks")
        return liquid_stocks
    
    def scan_unusual_volume(self, universe, top_n=20):
        """Scan for unusual volume."""
        logging.info(f"🔍 Scanning {len(universe)} stocks for unusual volume...")
        
        unusual_stocks = []
        
        for symbol in universe:
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period='10d')
                
                if len(hist) < 5:
                    continue
                
                current_volume = hist['Volume'].iloc[-1]
                avg_volume = hist['Volume'].iloc[:-1].mean()
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                
                if volume_ratio >= 2.0:
                    try:
                        info = stock.info
                    except Exception:
                        info = {}
                    if not passes_liquidity_filter(symbol, info):
                        continue
                    unusual_stocks.append((symbol, volume_ratio))
                    self.opportunities[symbol].append(f"Unusual volume: {volume_ratio:.1f}x")

            except Exception as e:
                logging.debug(f"Skipping {symbol} in volume scan: {e}")
                continue
        
        unusual_stocks.sort(key=lambda x: x[1], reverse=True)
        top_unusual = [symbol for symbol, _ in unusual_stocks[:top_n]]
        
        logging.info(f"  💥 Found {len(top_unusual)} stocks with unusual volume")
        return top_unusual
    
    def scan_breakouts(self, universe, top_n=20):
        """Scan for breakouts."""
        logging.info(f"🔍 Scanning {len(universe)} stocks for breakouts...")
        
        breakout_stocks = []
        
        for symbol in universe:
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period='1y')
                
                if len(hist) < 50:
                    continue
                
                current_price = hist['Close'].iloc[-1]
                high_52week = hist['High'].max()
                
                if current_price >= high_52week * 0.98:
                    pct_from_high = ((current_price - high_52week) / high_52week) * 100
                    try:
                        info = stock.info
                    except Exception:
                        info = {}
                    if not passes_liquidity_filter(symbol, info):
                        continue
                    breakout_stocks.append((symbol, pct_from_high))
                    self.opportunities[symbol].append(f"52W breakout: {pct_from_high:+.1f}%")

            except Exception as e:
                logging.debug(f"Skipping {symbol} in breakout scan: {e}")
                continue
        
        breakout_stocks.sort(key=lambda x: x[1], reverse=True)
        top_breakouts = [symbol for symbol, _ in breakout_stocks[:top_n]]
        
        logging.info(f"  🚀 Found {len(top_breakouts)} breakout stocks")
        return top_breakouts
    
    def scan_oversold(self, universe, top_n=20):
        """Scan for oversold stocks."""
        logging.info(f"🔍 Scanning {len(universe)} stocks for oversold conditions...")
        
        oversold_stocks = []
        
        for symbol in universe:
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period='1mo')
                
                if len(hist) < 14:
                    continue
                
                delta = hist['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1]
                
                if current_rsi < 30:
                    try:
                        info = stock.info
                    except Exception:
                        info = {}
                    if not passes_liquidity_filter(symbol, info):
                        continue
                    oversold_stocks.append((symbol, current_rsi))
                    self.opportunities[symbol].append(f"Oversold RSI: {current_rsi:.1f}")

            except Exception as e:
                logging.debug(f"Skipping {symbol} in oversold scan: {e}")
                continue
        
        oversold_stocks.sort(key=lambda x: x[1])
        top_oversold = [symbol for symbol, _ in oversold_stocks[:top_n]]
        
        logging.info(f"  📉 Found {len(top_oversold)} oversold stocks")
        return top_oversold
    
    def scan_momentum(self, universe, top_n=20):
        """Scan for momentum."""
        logging.info(f"🔍 Scanning {len(universe)} stocks for momentum...")
        
        momentum_stocks = []
        
        for symbol in universe:
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period='1mo')
                
                if len(hist) < 20:
                    continue
                
                price_20d_ago = hist['Close'].iloc[0]
                current_price = hist['Close'].iloc[-1]
                return_pct = ((current_price - price_20d_ago) / price_20d_ago) * 100
                
                if return_pct >= 5.0:
                    try:
                        info = stock.info
                    except Exception:
                        info = {}
                    if not passes_liquidity_filter(symbol, info):
                        continue
                    momentum_stocks.append((symbol, return_pct))
                    self.opportunities[symbol].append(f"20D momentum: +{return_pct:.1f}%")

            except Exception as e:
                logging.debug(f"Skipping {symbol} in momentum scan: {e}")
                continue
        
        momentum_stocks.sort(key=lambda x: x[1], reverse=True)
        top_momentum = [symbol for symbol, _ in momentum_stocks[:top_n]]
        
        logging.info(f"  📈 Found {len(top_momentum)} momentum stocks")
        return top_momentum
    
    def scan_gap_moves(self, universe, top_n=15):
        """Scan for gap moves."""
        logging.info(f"🔍 Scanning {len(universe)} stocks for gap moves...")
        
        gap_stocks = []
        
        for symbol in universe:
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period='5d')
                
                if len(hist) < 2:
                    continue
                
                prev_close = hist['Close'].iloc[-2]
                current_open = hist['Open'].iloc[-1]
                gap_pct = ((current_open - prev_close) / prev_close) * 100
                
                if abs(gap_pct) >= 3.0:
                    try:
                        info = stock.info
                    except Exception:
                        info = {}
                    if not passes_liquidity_filter(symbol, info):
                        continue
                    gap_stocks.append((symbol, gap_pct))
                    direction = "up" if gap_pct > 0 else "down"
                    self.opportunities[symbol].append(f"Gap {direction}: {gap_pct:+.1f}%")

            except Exception as e:
                logging.debug(f"Skipping {symbol} in gap scan: {e}")
                continue
        
        gap_stocks.sort(key=lambda x: abs(x[1]), reverse=True)
        top_gaps = [symbol for symbol, _ in gap_stocks[:top_n]]
        
        logging.info(f"  ⚡ Found {len(top_gaps)} gap stocks")
        return top_gaps
    
    def scan_mean_reversion(self, universe, top_n=15):
        """Scan for mean reversion."""
        logging.info(f"🔍 Scanning {len(universe)} stocks for mean reversion...")
        
        reversion_stocks = []
        
        for symbol in universe:
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period='3mo')
                
                if len(hist) < 60:
                    continue
                
                current_price = hist['Close'].iloc[-1]
                avg_60d = hist['Close'].iloc[-60:].mean()
                high_60d = hist['High'].iloc[-60:].max()
                
                below_avg_pct = ((current_price - avg_60d) / avg_60d) * 100
                below_high_pct = ((current_price - high_60d) / high_60d) * 100
                
                if below_avg_pct <= -10 and below_high_pct >= -30:
                    try:
                        info = stock.info
                    except Exception:
                        info = {}
                    if not passes_liquidity_filter(symbol, info):
                        continue
                    reversion_stocks.append((symbol, below_avg_pct))
                    self.opportunities[symbol].append(f"Mean reversion: {below_avg_pct:.1f}% below avg")

            except Exception as e:
                logging.debug(f"Skipping {symbol} in mean reversion scan: {e}")
                continue
        
        reversion_stocks.sort(key=lambda x: x[1])
        top_reversion = [symbol for symbol, _ in reversion_stocks[:top_n]]
        
        logging.info(f"  🔄 Found {len(top_reversion)} mean reversion candidates")
        return top_reversion
    
    def get_premarket_movers(self, top_n=10):
        """Get pre-market movers."""
        logging.info("🔍 Fetching pre-market movers...")
        
        try:
            url = "https://finviz.com/screener.ashx?v=111&s=ta_topgainers"
            headers = {'User-Agent': 'Mozilla/5.0'}

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            # Find the table that contains a 'Ticker' column — FinViz layout changes over time
            tables = pd.read_html(response.text)
            df = next((t for t in tables if 'Ticker' in t.columns), None)
            if df is None:
                logging.warning("Failed to get pre-market movers: no Ticker table found in FinViz response")
                return []

            tickers = df['Ticker'].dropna().head(top_n).tolist()

            for ticker in tickers:
                self.opportunities[ticker].append("Pre-market mover")

            logging.info(f"  🔥 Found {len(tickers)} pre-market movers")
            return tickers

        except Exception as e:
            # Truncate error to avoid logging full HTML responses
            logging.warning(f"Failed to get pre-market movers: {str(e)[:200]}")
            return []
    
    def filter_by_liquidity(self, symbols, min_volume=500000, min_price=5.0):
        """Final liquidity filter."""
        liquid_stocks = []

        for symbol in symbols:
            if symbol in self._delisted:
                continue
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period='5d')

                if hist.empty:
                    self._mark_delisted(symbol)
                    continue

                avg_volume = hist['Volume'].mean()
                current_price = hist['Close'].iloc[-1]

                if avg_volume >= min_volume and current_price >= min_price:
                    liquid_stocks.append(symbol)
            except Exception as e:
                logging.debug(f"Skipping {symbol} in final liquidity filter: {e}")
                continue

        return liquid_stocks
    
    def rank_opportunities(self, stocks):
        """Rank by signal count."""
        ranked = []
        
        for symbol in stocks:
            signal_count = len(self.opportunities.get(symbol, []))
            ranked.append((symbol, signal_count))
        
        ranked.sort(key=lambda x: x[1], reverse=True)
        return [symbol for symbol, _ in ranked]
    
    def discover_opportunities(self, max_stocks=30, deep_scan=False, use_cache=True):
        """Main discovery function.

        Args:
            max_stocks:  Maximum number of opportunities to return.
            deep_scan:   If True, scan the full 2000-stock universe (slow).
            use_cache:   If True (default), return cached results when they are
                         younger than _DISCOVERY_CACHE_TTL_HOURS.  Pass False to
                         force a fresh scan regardless of cache age.
        """
        # --- Cache check ---
        if use_cache and not deep_scan:
            cached = _load_discovery_cache()
            if cached is not None:
                self.discovered_stocks = cached.get('stocks', [])[:max_stocks]
                # Restore opportunities dict so callers can read signals
                raw_ops = cached.get('opportunities', {})
                self.opportunities = defaultdict(list, {k: list(v) for k, v in raw_ops.items()})
                return self.discovered_stocks

        logging.info("🚀 Starting FULL MARKET SCAN...")
        logging.info("="*70)

        all_candidates = []
        self.opportunities.clear()
        
        # Get universe
        if deep_scan:
            universe = self.get_actively_traded_universe(min_volume=500000, max_stocks=2000)
        else:
            universe = self.get_index_constituents()
            universe.extend(self.get_actively_traded_universe(min_volume=1000000, max_stocks=500))
            universe = list(set(universe))
        
        logging.info(f"📊 Scanning universe: {len(universe)} stocks")
        logging.info("="*70)
        
        # Run scans
        all_candidates.extend(self.scan_unusual_volume(universe, top_n=15))
        all_candidates.extend(self.scan_breakouts(universe, top_n=15))
        all_candidates.extend(self.scan_oversold(universe, top_n=15))
        all_candidates.extend(self.scan_momentum(universe, top_n=15))
        all_candidates.extend(self.scan_gap_moves(universe, top_n=10))
        all_candidates.extend(self.scan_mean_reversion(universe, top_n=10))
        all_candidates.extend(self.get_premarket_movers(top_n=10))
        
        unique_candidates = list(dict.fromkeys(all_candidates))
        liquid_stocks = self.filter_by_liquidity(unique_candidates, min_volume=500000)
        ranked_stocks = self.rank_opportunities(liquid_stocks)
        self.discovered_stocks = ranked_stocks[:max_stocks]
        
        logging.info(f"\n{'='*70}")
        logging.info(f"✅ DISCOVERY COMPLETE: {len(self.discovered_stocks)} opportunities")
        logging.info(f"{'='*70}\n")
        
        for i, symbol in enumerate(self.discovered_stocks[:15], 1):
            signals = self.opportunities.get(symbol, [])
            logging.info(f"{i:2d}. {symbol:6s} ({len(signals)} signals): {', '.join(signals)}")
        
        self.save_opportunities()
        return self.discovered_stocks
    
    def save_opportunities(self):
        """Save opportunities to both the human-readable log and the TTL cache."""
        output = {
            'timestamp': datetime.now().isoformat(),
            'total_discovered': len(self.discovered_stocks),
            'universe_scanned': len(self.full_universe),
            'stocks': self.discovered_stocks,
            'opportunities': dict(self.opportunities)
        }

        with open('logs/discovered_opportunities.json', 'w') as f:
            json.dump(output, f, indent=2)

        # Also persist as the discovery cache so the next session can reuse results
        _save_discovery_cache(output)

        logging.info(f"\n💾 Saved to logs/discovered_opportunities.json (cache TTL={_DISCOVERY_CACHE_TTL_HOURS}h)")

    def get_exploration_symbols(self, n: int = 2, exclude: set = None) -> list:
        """
        Return n random symbols from full_universe not in exclude or _delisted.
        Used to inject novel symbols into sessions when Sharpe is below target.
        """
        import random
        if not self.full_universe:
            return []
        exclude_set = (exclude or set()) | self._delisted
        candidates = [s for s in self.full_universe if s not in exclude_set]
        if not candidates:
            return []
        return random.sample(candidates, min(n, len(candidates)))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--deep', action='store_true', help='Run deep scan')
    parser.add_argument('--max', type=int, default=25, help='Max stocks')
    args = parser.parse_args()
    
    discovery = StockDiscovery()
    stocks = discovery.discover_opportunities(max_stocks=args.max, deep_scan=args.deep)
    
    print(f"\n{'='*70}")
    print(f"FINAL WATCHLIST ({len(stocks)} stocks):")
    print(f"{'='*70}")
    for i, symbol in enumerate(stocks, 1):
        signals = discovery.opportunities.get(symbol, [])
        print(f"{i:2d}. {symbol:6s} - {len(signals)} signals: {', '.join(signals)}")
