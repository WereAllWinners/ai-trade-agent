import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
import numpy as np
from collections import defaultdict
import json
import time
import warnings
warnings.filterwarnings('ignore', message='Failed to fetch')

logging.basicConfig(level=logging.INFO)

class StockDiscovery:
    def __init__(self):
        self.discovered_stocks = []
        self.opportunities = defaultdict(list)
        self.full_universe = []
    
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
        
        # S&P 500
        try:
            sp500_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            sp500_symbols = sp500_table[0]['Symbol'].str.replace('.', '-').tolist()
            constituents.extend(sp500_symbols)
            logging.info(f"  ✅ S&P 500: {len(sp500_symbols)} stocks")
        except Exception as e:
            logging.warning(f"Failed to fetch S&P 500: {e}")
        
        # NASDAQ 100
        try:
            nasdaq100_table = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
            nasdaq100_symbols = nasdaq100_table[4]['Ticker'].tolist()
            constituents.extend(nasdaq100_symbols)
            logging.info(f"  ✅ NASDAQ 100: {len(nasdaq100_symbols)} stocks")
        except Exception as e:
            logging.warning(f"Failed to fetch NASDAQ 100: {e}")
        
        # Dow Jones 30
        try:
            dow_table = pd.read_html('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')
            dow_symbols = dow_table[1]['Symbol'].tolist()
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
                try:
                    stock = yf.Ticker(symbol)
                    hist = stock.history(period='5d')
                    
                    if not hist.empty and len(hist) >= 3:
                        avg_volume = hist['Volume'].mean()
                        current_price = hist['Close'].iloc[-1]
                        
                        if (avg_volume >= min_volume and 
                            current_price >= 5.0 and 
                            current_price <= 2000.0):
                            liquid_stocks.append(symbol)
                
                except:
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
                    unusual_stocks.append((symbol, volume_ratio))
                    self.opportunities[symbol].append(f"Unusual volume: {volume_ratio:.1f}x")
            
            except:
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
                    breakout_stocks.append((symbol, pct_from_high))
                    self.opportunities[symbol].append(f"52W breakout: {pct_from_high:+.1f}%")
            
            except:
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
                    oversold_stocks.append((symbol, current_rsi))
                    self.opportunities[symbol].append(f"Oversold RSI: {current_rsi:.1f}")
            
            except:
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
                    momentum_stocks.append((symbol, return_pct))
                    self.opportunities[symbol].append(f"20D momentum: +{return_pct:.1f}%")
            
            except:
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
                    gap_stocks.append((symbol, gap_pct))
                    direction = "up" if gap_pct > 0 else "down"
                    self.opportunities[symbol].append(f"Gap {direction}: {gap_pct:+.1f}%")
            
            except:
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
                    reversion_stocks.append((symbol, below_avg_pct))
                    self.opportunities[symbol].append(f"Mean reversion: {below_avg_pct:.1f}% below avg")
            
            except:
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
            df = pd.read_html(response.text)[6]
            
            tickers = df['Ticker'].head(top_n).tolist()
            
            for ticker in tickers:
                self.opportunities[ticker].append("Pre-market mover")
            
            logging.info(f"  🔥 Found {len(tickers)} pre-market movers")
            return tickers
        
        except Exception as e:
            logging.warning(f"Failed to get pre-market movers: {e}")
            return []
    
    def filter_by_liquidity(self, symbols, min_volume=500000, min_price=5.0):
        """Final liquidity filter."""
        liquid_stocks = []
        
        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period='5d')
                
                if not hist.empty:
                    avg_volume = hist['Volume'].mean()
                    current_price = hist['Close'].iloc[-1]
                    
                    if avg_volume >= min_volume and current_price >= min_price:
                        liquid_stocks.append(symbol)
            except:
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
    
    def discover_opportunities(self, max_stocks=30, deep_scan=False):
        """Main discovery function."""
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
        """Save opportunities."""
        output = {
            'timestamp': datetime.now().isoformat(),
            'total_discovered': len(self.discovered_stocks),
            'universe_scanned': len(self.full_universe),
            'stocks': self.discovered_stocks,
            'opportunities': dict(self.opportunities)
        }
        
        with open('logs/discovered_opportunities.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        logging.info(f"\n💾 Saved to logs/discovered_opportunities.json")

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
