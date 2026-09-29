import os
import io
import sys
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/ root
from service_suffix import service_suffix
warnings.filterwarnings('ignore', message='Failed to fetch')

logging.basicConfig(level=logging.INFO)

# Liquidity thresholds — override via environment variables
_MIN_AVG_DAILY_VOLUME = int(os.getenv('MIN_AVG_DAILY_VOLUME_STOCKS', '1_000_000'.replace('_', '')))
_MIN_MARKET_CAP       = int(os.getenv('MIN_MARKET_CAP',               '500_000_000'.replace('_', '')))

_DELISTED_CACHE_PATH = Path(__file__).resolve().parent.parent / 'logs' / 'delisted_cache.json'
_DISCOVERY_CACHE_PATH = Path(__file__).resolve().parent.parent / 'logs' / 'discovery_cache.json'
_SIGNAL_EV_PATH = Path(__file__).resolve().parent.parent / 'logs' / 'signal_expectancy.json'
_DISCOVERY_CACHE_TTL_HOURS = int(os.getenv('DISCOVERY_CACHE_TTL_HOURS', '4'))  # Reuse discovery results for up to N hours
# sprint02 D2: a scan yielding fewer than this many opportunities is not cached —
# an empty/low-yield scan (e.g. from a data-source outage) would otherwise poison
# every session for the full TTL window until the next scheduled scan.
_MIN_CACHEABLE_OPPORTUNITIES = int(os.getenv('MIN_CACHEABLE_OPPORTUNITIES', '5'))
# sprint02 D5a: consecutive no-data fetches required before a symbol is marked
# delisted. A StockDiscovery instance is rebuilt fresh each ~30-min session, so
# the counter must be read from/written to the persisted sidecar (below), not
# held only in memory — an in-memory counter would never see more than 1
# failure before the process/instance is discarded.
_DELISTED_FAILURE_THRESHOLD = int(os.getenv('DELISTED_FAILURE_THRESHOLD', '2'))

# sprint04 F3: yf.download() batch-fetch chunking + chunk-health check.
_BATCH_CHUNK_SIZE = int(os.getenv('BATCH_CHUNK_SIZE', '100'))
# A chunk is only eligible for the infrastructure-failure sentinel when it has
# at least this many symbols — below this, a couple of genuinely dead symbols
# can swing empty_fraction past the threshold on small-sample noise alone, and
# per-symbol classification (the safer default) applies instead.
_BATCH_CHUNK_MIN_SIZE_FOR_HEALTH_CHECK = 10
_BATCH_CHUNK_FAILURE_THRESHOLD = float(os.getenv('BATCH_CHUNK_FAILURE_THRESHOLD', '0.5'))


def _fetch_batch_history(symbols: list[str], period: str,
                          chunk_size: int = None) -> dict:
    """Fetch OHLCV history for many symbols via yf.download(), chunked
    (sprint04 F3). Returns a dict with an entry for EVERY requested symbol —
    a symbol omitted entirely from yf.download's response (this happens, not
    just NaN-filled) is treated identically to an all-NaN column, both
    counting toward the chunk's empty_fraction — mapping to:

      - a real (possibly-empty) single-index OHLCV DataFrame: this symbol's
        own genuine fetch outcome, safe to record via _record_fetch_result.
      - None: a chunk-level sentinel meaning "infrastructure failure,
        unknown — do NOT treat as evidence this symbol is dead." Fires only
        when a chunk of at least _BATCH_CHUNK_MIN_SIZE_FOR_HEALTH_CHECK
        symbols comes back more than _BATCH_CHUNK_FAILURE_THRESHOLD empty —
        collapsing a rate-limited/network-blip chunk into the same signal as
        genuine symbol death would let one bad scan mass-delist a whole
        chunk of otherwise-healthy symbols via D5a's persisted counter.
    """
    def _is_empty(df: pd.DataFrame) -> bool:
        if df is None or df.empty:
            return True
        close = df['Close'] if 'Close' in df.columns else pd.Series(dtype=float)
        return close.empty or close.isna().all()

    chunk_size = chunk_size or _BATCH_CHUNK_SIZE
    result: dict = {}

    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i + chunk_size]
        try:
            raw = yf.download(tickers=chunk, period=period, group_by='ticker',
                               threads=True, progress=False)
            if len(chunk) == 1:
                # yf.download does not return MultiIndex columns for a
                # length-1 tickers list, even with group_by='ticker'.
                chunk_data = {chunk[0]: raw}
            else:
                top_level = (raw.columns.get_level_values(0)
                             if isinstance(raw.columns, pd.MultiIndex) else [])
                chunk_data = {
                    symbol: raw[symbol] if symbol in top_level else pd.DataFrame()
                    for symbol in chunk
                }
        except Exception as e:
            logging.warning(f"⚠️ Batch fetch failed for chunk {i}-{i + len(chunk)}: {e}")
            for symbol in chunk:
                result[symbol] = None
            continue

        empties = [s for s in chunk if _is_empty(chunk_data[s])]
        empty_fraction = len(empties) / len(chunk)

        if (len(chunk) >= _BATCH_CHUNK_MIN_SIZE_FOR_HEALTH_CHECK and
                empty_fraction > _BATCH_CHUNK_FAILURE_THRESHOLD):
            logging.warning(
                f"⚠️ Batch fetch chunk {i}-{i + len(chunk)} returned "
                f"{empty_fraction:.0%} empty — treating as infrastructure "
                f"failure, not symbol death"
            )
            try:
                from alerts import alert_yfinance_batch_chunk_failure
                alert_yfinance_batch_chunk_failure(empty_fraction, len(chunk))
            except Exception as alert_err:
                logging.debug(f"Could not send yfinance_batch_chunk_failure alert: {alert_err}")
            for symbol in chunk:
                result[symbol] = None
            continue

        for symbol in chunk:
            df = chunk_data[symbol]
            # Normalize all-NaN / omitted-from-response to a real empty
            # DataFrame so every downstream call site's plain `hist.empty`
            # check catches it uniformly, regardless of which of the three
            # shapes ("truly no rows", "all-NaN", "absent from response")
            # produced it.
            result[symbol] = pd.DataFrame() if _is_empty(df) else df

    return result


def _load_signal_ev() -> dict[str, float]:
    """Load per-signal expected value map from logs/signal_expectancy.json.

    Returns {} when the file does not exist or cannot be parsed so that
    rank_opportunities falls back to count-based ranking without raising.
    """
    try:
        data = json.loads(_SIGNAL_EV_PATH.read_text())
        return data.get('signal_ev', {})
    except Exception:
        return {}


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


def _load_fail_counts() -> dict:
    """Load persisted per-symbol consecutive-failure counters (sprint02 D5a).

    Stored as a sidecar section inside delisted_cache.json rather than a
    separate file, keeping the two pieces of denylist state together.
    """
    try:
        if _DELISTED_CACHE_PATH.exists():
            data = json.loads(_DELISTED_CACHE_PATH.read_text())
            return dict(data.get('fail_counts', {}))
    except Exception:
        pass
    return {}


def _save_delisted_cache(delisted: set, fail_counts: dict = None) -> None:
    """Persist denylist to disk, alongside the fail-count sidecar.

    If fail_counts isn't given, the sidecar already on disk is preserved
    rather than clobbered — callers that only touch the delisted set (e.g.
    _mark_delisted) don't need to know about counter state to avoid losing it.
    """
    try:
        _DELISTED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if fail_counts is None:
            fail_counts = _load_fail_counts()
        _DELISTED_CACHE_PATH.write_text(
            json.dumps({
                'delisted': sorted(delisted),
                'fail_counts': fail_counts,
                'updated': datetime.now().isoformat(),
            }, indent=2)
        )
    except Exception as e:
        logging.warning(f"Could not save delisted cache: {e}")


def _load_discovery_cache() -> dict | None:
    """
    Return cached discovery results if they are younger than the TTL, else None.
    The cache stores the full opportunities dict so signals survive across sessions.
    """
    # sprint02 D4.1: this is the actual functional cache driving discovery
    # decisions (unlike the human-readable log) — paper and live stock services
    # sharing one unsuffixed path here is what caused the historical incident
    # where live's session read paper's stale/differing stock count.
    from service_suffix import suffixed_path
    _path = suffixed_path(_DISCOVERY_CACHE_PATH)
    try:
        if _path.exists():
            data = json.loads(_path.read_text())
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
    from service_suffix import suffixed_path
    _path = suffixed_path(_DISCOVERY_CACHE_PATH)
    try:
        _path.parent.mkdir(parents=True, exist_ok=True)
        _path.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logging.warning(f"Could not save discovery cache: {e}")


class StockDiscovery:
    def __init__(self):
        self.discovered_stocks = []
        self.opportunities = defaultdict(list)
        self.full_universe = []
        self._delisted: set = _load_delisted_cache()
        self._fail_counts: dict = _load_fail_counts()
        self._funnel: dict = {}  # sprint04 F2.1 — per-stage survivor stats for one scan
        self._full_universe_source: str | None = None  # sprint04 F2.1 — 'ftp' | 'curated_fallback'
        self._batch_history_cache: dict = {}  # sprint04 F3 — keyed by (period, sorted symbols)
        if self._delisted:
            logging.info(f"🚫 Loaded {len(self._delisted)} known delisted/unfound tickers from cache")

    def _mark_delisted(self, symbol: str) -> None:
        """Add a ticker to the denylist and persist it."""
        if symbol not in self._delisted:
            self._delisted.add(symbol)
            logging.info(f"🗑️  Marking {symbol} as delisted/unfound — added to denylist")
            _save_delisted_cache(self._delisted, self._fail_counts)

    def _record_fetch_result(self, symbol: str, success: bool) -> None:
        """Track consecutive yfinance no-data failures per symbol (sprint02
        D5a), persisted across process restarts so a single transient blip
        doesn't get treated the same as genuine delisting — see
        _DELISTED_FAILURE_THRESHOLD. Only marks delisted once the persisted
        counter reaches threshold; any successful fetch resets it to 0.
        """
        if success:
            if self._fail_counts.get(symbol, 0):
                self._fail_counts[symbol] = 0
                _save_delisted_cache(self._delisted, self._fail_counts)
            return

        count = self._fail_counts.get(symbol, 0) + 1
        self._fail_counts[symbol] = count
        _save_delisted_cache(self._delisted, self._fail_counts)

        if count >= _DELISTED_FAILURE_THRESHOLD:
            self._mark_delisted(symbol)
        else:
            logging.debug(
                f"⚠️  {symbol}: fetch failure {count}/{_DELISTED_FAILURE_THRESHOLD} — not yet delisted"
            )

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

    def _get_batch_history(self, symbols: list[str], period: str) -> dict:
        """Memoized wrapper around _fetch_batch_history (sprint04 F3), keyed
        by (period, sorted symbol tuple) — sorted, not just tuple(symbols),
        so two callers requesting the same symbol set in a different order
        still hit the cache instead of silently double-downloading."""
        key = (period, tuple(sorted(symbols)))
        if key not in self._batch_history_cache:
            self._batch_history_cache[key] = _fetch_batch_history(symbols, period)
        return self._batch_history_cache[key]

    def _record_funnel_stage(self, stage: str, universe_in: int,
                              evaluated: list, passed: list,
                              higher_is_better: bool = True) -> None:
        """Record survivor stats + nearest-miss symbols for one funnel stage
        (sprint04 F2.1). `evaluated` is every (symbol, metric_value) pair that
        survived the fetch and got a metric computed — not just the ones that
        passed the threshold — so near-misses can be drawn from
        evaluated-minus-passed."""
        passed_set = set(passed)
        misses = [(sym, val) for sym, val in evaluated if sym not in passed_set]
        misses.sort(key=lambda t: t[1], reverse=higher_is_better)
        self._funnel[stage] = {
            'universe_in': universe_in,
            'evaluated': len(evaluated),
            'passed': len(passed),
            'near_misses': misses[:10],
        }

    def _log_funnel_summary(self) -> None:
        """One consolidated INFO line at scan completion, plus a DEBUG
        near-miss dump for the stage where the survivor count first
        collapses (sprint04 F2.1)."""
        order = ['unusual_volume', 'breakout', 'oversold', 'momentum',
                 'gap_moves', 'mean_reversion', 'premarket',
                 'dedup', 'final_liquidity', 'ranked']
        present = [s for s in order if s in self._funnel]
        if not present:
            return
        parts = [f"{s}={self._funnel[s]['passed']}" for s in present]
        universe_in = self._funnel[present[0]]['universe_in']
        source = self._full_universe_source or 'unknown'
        logging.info("FUNNEL: universe=%s (full_universe=%d via %s) -> %s",
                     universe_in, len(self.full_universe), source, ' -> '.join(parts))

        collapse_stage = min(present, key=lambda s: self._funnel[s]['passed'])
        if self._funnel[collapse_stage].get('near_misses'):
            logging.debug("FUNNEL nearest-miss @ %s: %s",
                           collapse_stage, self._funnel[collapse_stage]['near_misses'])

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
            self._full_universe_source = 'ftp'  # sprint04 F2.1

        except Exception as e:
            logging.warning(f"NASDAQ FTP failed: {e}, using fallback method")
            all_stocks = self.get_index_constituents()
            self._full_universe_source = 'curated_fallback'  # sprint04 F2.1

        self.full_universe = sorted(list(set(all_stocks)))
        logging.info(f"📊 Total market universe: {len(self.full_universe)} stocks")

        return self.full_universe
    
    def get_index_constituents(self):
        """Fallback: Get stocks from major indices."""
        logging.info("📡 Fetching index constituents...")
        
        constituents = []
        
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; research-bot/1.0)'}

        _SP500_URL   = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        _NASDAQ_URL  = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        _DOW_URL     = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'

        # S&P 500
        _resp = None
        try:
            _resp = requests.get(_SP500_URL, headers=headers, timeout=10)
            sp500_symbols = pd.read_html(io.StringIO(_resp.text))[0]['Symbol'].str.replace('.', '-').tolist()
            constituents.extend(sp500_symbols)
            logging.info(f"  ✅ S&P 500: {len(sp500_symbols)} stocks")
        except Exception as e:
            _status = _resp.status_code if _resp is not None else 'N/A'
            _body   = (_resp.text[:200] if _resp is not None else '')
            logging.warning("Failed to fetch S&P 500 (url=%s status=%s body_head=%.200r): %s",
                            _SP500_URL, _status, _body, str(e)[:100])

        # NASDAQ 100
        _resp = None
        try:
            _resp = requests.get(_NASDAQ_URL, headers=headers, timeout=10)
            tables = pd.read_html(io.StringIO(_resp.text))
            nasdaq100_df = next((t for t in tables if 'Ticker' in t.columns), None)
            if nasdaq100_df is not None:
                nasdaq100_symbols = nasdaq100_df['Ticker'].dropna().tolist()
                constituents.extend(nasdaq100_symbols)
                logging.info(f"  ✅ NASDAQ 100: {len(nasdaq100_symbols)} stocks")
        except Exception as e:
            _status = _resp.status_code if _resp is not None else 'N/A'
            _body   = (_resp.text[:200] if _resp is not None else '')
            logging.warning("Failed to fetch NASDAQ 100 (url=%s status=%s body_head=%.200r): %s",
                            _NASDAQ_URL, _status, _body, str(e)[:100])

        # Dow Jones 30
        _resp = None
        try:
            _resp = requests.get(_DOW_URL, headers=headers, timeout=10)
            tables = pd.read_html(io.StringIO(_resp.text))
            dow_df = next((t for t in tables if 'Symbol' in t.columns), None)
            if dow_df is not None:
                dow_symbols = dow_df['Symbol'].dropna().tolist()
                constituents.extend(dow_symbols)
                logging.info(f"  ✅ Dow 30: {len(dow_symbols)} stocks")
        except Exception as e:
            _status = _resp.status_code if _resp is not None else 'N/A'
            _body   = (_resp.text[:200] if _resp is not None else '')
            logging.warning("Failed to fetch Dow 30 (url=%s status=%s body_head=%.200r): %s",
                            _DOW_URL, _status, _body, str(e)[:100])
        
        # sprint02 D6a: all three curated fetches failed — the universe is
        # about to fall back to only the hardcoded popular-stocks list below.
        if not constituents:
            try:
                from alerts import alert_discovery_curated_fetch_failed
                alert_discovery_curated_fetch_failed()
            except Exception as e:
                logging.debug(f"Could not send discovery_curated_fetch_failed alert: {e}")

        # Add popular growth/tech stocks
        # sprint04 F4: this list needs occasional manual review — D5a's
        # persisted fail-counter is the safety net for *future* deaths
        # between reviews, but known-dead entries shouldn't have to burn
        # through the fail-threshold at all. Verified live via yfinance
        # 2026-07-08: SQ was renamed to XYZ (Block, Inc.); WBA (Walgreens
        # Boots Alliance) went private and has no replacement ticker.
        popular_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'AMD', 'NFLX', 'DIS',
            'BABA', 'COIN', 'PLTR', 'SNOW', 'CRWD', 'NET', 'DDOG', 'ZS', 'SHOP', 'XYZ',
            'PYPL', 'ADBE', 'CRM', 'ORCL', 'INTC', 'QCOM', 'AVGO', 'TXN', 'MU', 'LRCX',
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'SCHW',
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY', 'MPC', 'VLO', 'PSX', 'HAL',
            'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'DHR', 'AMGN',
            'BA', 'CAT', 'GE', 'MMM', 'HON', 'UNP', 'UPS', 'FDX', 'LMT', 'RTX',
            'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'CVS', 'COST',
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

        total_to_scan = min(len(self.full_universe), max_stocks)
        candidates = [s for s in self.full_universe[:total_to_scan] if s not in self._delisted]

        # sprint04 F3: batched via _get_batch_history — the old manual
        # batch_size=50/time.sleep throttle is superseded by
        # _fetch_batch_history's own per-chunk batching.
        history_map = self._get_batch_history(candidates, period='5d')

        liquid_stocks = []
        for symbol in candidates:
            hist = history_map.get(symbol)
            if hist is None:
                continue  # chunk-level infrastructure failure — skip, no counter change
            try:
                if hist.empty:
                    self._record_fetch_result(symbol, success=False)
                    continue
                self._record_fetch_result(symbol, success=True)

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

        logging.info(f"✅ Filtered to {len(liquid_stocks)} actively traded stocks")
        return liquid_stocks
    
    def scan_unusual_volume(self, universe, top_n=20):
        """Scan for unusual volume."""
        logging.info(f"🔍 Scanning {len(universe)} stocks for unusual volume...")

        unusual_stocks = []
        evaluated = []  # sprint04 F2.1
        candidates = [s for s in universe if s not in self._delisted]  # sprint04 F4
        history_map = self._get_batch_history(candidates, period='10d')  # sprint04 F3

        for symbol in candidates:
            try:
                hist = history_map.get(symbol)
                if hist is None:
                    continue  # sprint04 F3 — chunk-level infrastructure failure, no counter change

                if hist.empty:
                    self._record_fetch_result(symbol, success=False)  # sprint04 F4
                    continue
                self._record_fetch_result(symbol, success=True)  # sprint04 F4

                if len(hist) < 5:
                    continue

                current_volume = hist['Volume'].iloc[-1]
                avg_volume = hist['Volume'].iloc[:-1].mean()
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                evaluated.append((symbol, volume_ratio))  # sprint04 F2.1

                if volume_ratio >= 2.0:
                    try:
                        info = yf.Ticker(symbol).info
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
        self._record_funnel_stage('unusual_volume', len(universe), evaluated,
                                   [s for s, _ in unusual_stocks], higher_is_better=True)  # sprint04 F2.1

        logging.info(f"  💥 Found {len(top_unusual)} stocks with unusual volume")
        return top_unusual
    
    def scan_breakouts(self, universe, top_n=20):
        """Scan for breakouts."""
        logging.info(f"🔍 Scanning {len(universe)} stocks for breakouts...")

        breakout_stocks = []
        evaluated = []  # sprint04 F2.1
        candidates = [s for s in universe if s not in self._delisted]  # sprint04 F4
        history_map = self._get_batch_history(candidates, period='1y')  # sprint04 F3

        for symbol in candidates:
            try:
                hist = history_map.get(symbol)
                if hist is None:
                    continue  # sprint04 F3 — chunk-level infrastructure failure, no counter change

                if hist.empty:
                    self._record_fetch_result(symbol, success=False)  # sprint04 F4
                    continue
                self._record_fetch_result(symbol, success=True)  # sprint04 F4

                if len(hist) < 50:
                    continue

                current_price = hist['Close'].iloc[-1]
                high_52week = hist['High'].max()
                pct_from_high = ((current_price - high_52week) / high_52week) * 100
                evaluated.append((symbol, pct_from_high))  # sprint04 F2.1

                if current_price >= high_52week * 0.98:
                    try:
                        info = yf.Ticker(symbol).info
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
        self._record_funnel_stage('breakout', len(universe), evaluated,
                                   [s for s, _ in breakout_stocks], higher_is_better=True)  # sprint04 F2.1

        logging.info(f"  🚀 Found {len(top_breakouts)} breakout stocks")
        return top_breakouts
    
    def scan_oversold(self, universe, top_n=20):
        """Scan for oversold stocks."""
        logging.info(f"🔍 Scanning {len(universe)} stocks for oversold conditions...")

        oversold_stocks = []
        evaluated = []  # sprint04 F2.1
        candidates = [s for s in universe if s not in self._delisted]  # sprint04 F4
        history_map = self._get_batch_history(candidates, period='1mo')  # sprint04 F3

        for symbol in candidates:
            try:
                hist = history_map.get(symbol)
                if hist is None:
                    continue  # sprint04 F3 — chunk-level infrastructure failure, no counter change

                if hist.empty:
                    self._record_fetch_result(symbol, success=False)  # sprint04 F4
                    continue
                self._record_fetch_result(symbol, success=True)  # sprint04 F4

                if len(hist) < 14:
                    continue

                delta = hist['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1]
                evaluated.append((symbol, current_rsi))  # sprint04 F2.1

                if current_rsi < 30:
                    try:
                        info = yf.Ticker(symbol).info
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
        self._record_funnel_stage('oversold', len(universe), evaluated,
                                   [s for s, _ in oversold_stocks], higher_is_better=False)  # sprint04 F2.1

        logging.info(f"  📉 Found {len(top_oversold)} oversold stocks")
        return top_oversold
    
    def scan_momentum(self, universe, top_n=20):
        """Scan for momentum."""
        logging.info(f"🔍 Scanning {len(universe)} stocks for momentum...")

        momentum_stocks = []
        evaluated = []  # sprint04 F2.1
        candidates = [s for s in universe if s not in self._delisted]  # sprint04 F4
        history_map = self._get_batch_history(candidates, period='1mo')  # sprint04 F3

        for symbol in candidates:
            try:
                hist = history_map.get(symbol)
                if hist is None:
                    continue  # sprint04 F3 — chunk-level infrastructure failure, no counter change

                if hist.empty:
                    self._record_fetch_result(symbol, success=False)  # sprint04 F4
                    continue
                self._record_fetch_result(symbol, success=True)  # sprint04 F4

                if len(hist) < 20:
                    continue

                price_20d_ago = hist['Close'].iloc[0]
                current_price = hist['Close'].iloc[-1]
                return_pct = ((current_price - price_20d_ago) / price_20d_ago) * 100
                evaluated.append((symbol, return_pct))  # sprint04 F2.1

                if return_pct >= 5.0:
                    try:
                        info = yf.Ticker(symbol).info
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
        self._record_funnel_stage('momentum', len(universe), evaluated,
                                   [s for s, _ in momentum_stocks], higher_is_better=True)  # sprint04 F2.1

        logging.info(f"  📈 Found {len(top_momentum)} momentum stocks")
        return top_momentum
    
    def scan_gap_moves(self, universe, top_n=15):
        """Scan for gap moves."""
        logging.info(f"🔍 Scanning {len(universe)} stocks for gap moves...")

        gap_stocks = []
        evaluated = []  # sprint04 F2.1
        candidates = [s for s in universe if s not in self._delisted]  # sprint04 F4
        history_map = self._get_batch_history(candidates, period='5d')  # sprint04 F3

        for symbol in candidates:
            try:
                hist = history_map.get(symbol)
                if hist is None:
                    continue  # sprint04 F3 — chunk-level infrastructure failure, no counter change

                if hist.empty:
                    self._record_fetch_result(symbol, success=False)  # sprint04 F4
                    continue
                self._record_fetch_result(symbol, success=True)  # sprint04 F4

                if len(hist) < 2:
                    continue

                prev_close = hist['Close'].iloc[-2]
                current_open = hist['Open'].iloc[-1]
                gap_pct = ((current_open - prev_close) / prev_close) * 100
                evaluated.append((symbol, abs(gap_pct)))  # sprint04 F2.1

                if abs(gap_pct) >= 3.0:
                    try:
                        info = yf.Ticker(symbol).info
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
        self._record_funnel_stage('gap_moves', len(universe), evaluated,
                                   [s for s, _ in gap_stocks], higher_is_better=True)  # sprint04 F2.1

        logging.info(f"  ⚡ Found {len(top_gaps)} gap stocks")
        return top_gaps
    
    def scan_mean_reversion(self, universe, top_n=15):
        """Scan for mean reversion."""
        logging.info(f"🔍 Scanning {len(universe)} stocks for mean reversion...")

        reversion_stocks = []
        evaluated = []  # sprint04 F2.1
        candidates = [s for s in universe if s not in self._delisted]  # sprint04 F4
        history_map = self._get_batch_history(candidates, period='3mo')  # sprint04 F3

        for symbol in candidates:
            try:
                hist = history_map.get(symbol)
                if hist is None:
                    continue  # sprint04 F3 — chunk-level infrastructure failure, no counter change

                if hist.empty:
                    self._record_fetch_result(symbol, success=False)  # sprint04 F4
                    continue
                self._record_fetch_result(symbol, success=True)  # sprint04 F4

                if len(hist) < 60:
                    continue

                current_price = hist['Close'].iloc[-1]
                avg_60d = hist['Close'].iloc[-60:].mean()
                high_60d = hist['High'].iloc[-60:].max()

                below_avg_pct = ((current_price - avg_60d) / avg_60d) * 100
                below_high_pct = ((current_price - high_60d) / high_60d) * 100
                evaluated.append((symbol, below_avg_pct))  # sprint04 F2.1

                if below_avg_pct <= -10 and below_high_pct >= -30:
                    try:
                        info = yf.Ticker(symbol).info
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
        self._record_funnel_stage('mean_reversion', len(universe), evaluated,
                                   [s for s, _ in reversion_stocks], higher_is_better=False)  # sprint04 F2.1

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
            tables = pd.read_html(io.StringIO(response.text))
            df = next((t for t in tables if 'Ticker' in t.columns), None)
            if df is None:
                logging.warning("Failed to get pre-market movers: no Ticker table found in FinViz response")
                return []

            tickers = df['Ticker'].dropna().head(top_n).tolist()

            for ticker in tickers:
                self.opportunities[ticker].append("Pre-market mover")

            self._funnel['premarket'] = {'universe_in': None, 'evaluated': None,
                                          'passed': len(tickers), 'near_misses': []}  # sprint04 F2.1
            logging.info(f"  🔥 Found {len(tickers)} pre-market movers")
            return tickers

        except Exception as e:
            # Truncate error to avoid logging full HTML responses
            logging.warning(f"Failed to get pre-market movers: {str(e)[:200]}")
            self._funnel['premarket'] = {'universe_in': None, 'evaluated': None,
                                          'passed': 0, 'near_misses': []}  # sprint04 F2.1
            return []
    
    def filter_by_liquidity(self, symbols, min_volume=500000, min_price=5.0):
        """Final liquidity filter."""
        candidates = [s for s in symbols if s not in self._delisted]
        history_map = self._get_batch_history(candidates, period='5d')  # sprint04 F3

        liquid_stocks = []
        for symbol in candidates:
            hist = history_map.get(symbol)
            if hist is None:
                continue  # chunk-level infrastructure failure — skip, no counter change
            try:
                if hist.empty:
                    self._record_fetch_result(symbol, success=False)
                    continue
                self._record_fetch_result(symbol, success=True)

                avg_volume = hist['Volume'].mean()
                current_price = hist['Close'].iloc[-1]

                if avg_volume >= min_volume and current_price >= min_price:
                    liquid_stocks.append(symbol)
            except Exception as e:
                logging.debug(f"Skipping {symbol} in final liquidity filter: {e}")
                continue

        return liquid_stocks
    
    def rank_opportunities(self, stocks):
        """Rank by signal EV sum; falls back to signal count when EV map is absent."""
        ev_map = _load_signal_ev()
        ranked = []

        for symbol in stocks:
            signals = self.opportunities.get(symbol, [])
            if ev_map:
                # Use EV for known signals; unknown signals get a small count-based bonus
                score = sum(
                    ev_map[sig] if sig in ev_map else len(signals) * 0.001
                    for sig in signals
                )
            else:
                score = float(len(signals))
            ranked.append((symbol, score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return [symbol for symbol, _ in ranked]
    
    def build_scan_universe(self, deep_scan: bool = False) -> list[str]:
        """Pure universe construction, no scanning (sprint04 F2.2). Extracted
        from discover_opportunities so both the live session and the offline
        replay tool (scripts/tools/replay_discovery_funnel.py) build the
        identical universe from the identical code path."""
        if deep_scan:
            return self.get_actively_traded_universe(min_volume=500000, max_stocks=2000)
        universe = self.get_index_constituents()
        universe.extend(self.get_actively_traded_universe(min_volume=1000000, max_stocks=500))
        return list(set(universe))

    def run_scan_pipeline(self, universe: list[str]) -> dict:
        """Run every discovery scan against `universe` with CURRENT
        thresholds and return ranked candidates + full funnel stats
        (sprint04 F2.2). No cache read/write, no alerts, no
        save_opportunities() — this is the single code path both the live
        session (discover_opportunities) and the offline replay tool call;
        do not reimplement scan logic anywhere else."""
        self._funnel = {}
        self.opportunities.clear()
        self._batch_history_cache = {}  # sprint04 F3 — fresh per pipeline run

        all_candidates = []
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

        self._funnel['dedup'] = {'universe_in': len(all_candidates), 'evaluated': None,
                                  'passed': len(unique_candidates), 'near_misses': []}
        self._funnel['final_liquidity'] = {'universe_in': len(unique_candidates), 'evaluated': None,
                                            'passed': len(liquid_stocks), 'near_misses': []}
        self._funnel['ranked'] = {'universe_in': len(liquid_stocks), 'evaluated': None,
                                   'passed': len(ranked_stocks), 'near_misses': []}
        self._log_funnel_summary()

        return {'universe_size': len(universe), 'ranked_stocks': ranked_stocks,
                'opportunities': dict(self.opportunities), 'funnel': self._funnel}

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

        universe = self.build_scan_universe(deep_scan=deep_scan)
        logging.info(f"📊 Scanning universe: {len(universe)} stocks")
        logging.info("="*70)

        result = self.run_scan_pipeline(universe)
        self.discovered_stocks = result['ranked_stocks'][:max_stocks]

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

        # sprint02 D4.1: paper and live stock services share this WorkingDirectory —
        # suffix the filename so they no longer race on the same file.
        _opp_path = f'logs/discovered_opportunities{service_suffix()}.json'
        with open(_opp_path, 'w') as f:
            json.dump(output, f, indent=2)

        # sprint02 D2: don't cache a low-yield scan — an empty/near-empty result
        # (e.g. from a data-source outage) would otherwise poison every session
        # for the full TTL window until the next scheduled scan. The human-readable
        # log above is written unconditionally regardless (observability artifact,
        # not the TTL cache); only the cache write is gated.
        if len(self.discovered_stocks) >= _MIN_CACHEABLE_OPPORTUNITIES:
            _save_discovery_cache(output)
            logging.info(f"\n💾 Saved to {_opp_path} (cache TTL={_DISCOVERY_CACHE_TTL_HOURS}h)")
        else:
            logging.warning(
                f"⚠️ Low-yield scan (N={len(self.discovered_stocks)} < "
                f"{_MIN_CACHEABLE_OPPORTUNITIES}) — result NOT cached; next session will rescan."
            )
            try:
                from alerts import alert_discovery_low_yield
                alert_discovery_low_yield(len(self.discovered_stocks), _MIN_CACHEABLE_OPPORTUNITIES)
            except Exception as e:
                logging.debug(f"Could not send discovery_low_yield alert: {e}")

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
