#!/usr/bin/env python3
"""
Market Researcher - Lightweight nightly research for both trading bots.

Writes a rolling-window JSON context file (30-day retention) that the agents
read at startup and that is used as training context for fine-tuning.

Usage:
    python market_researcher.py --bot trading
    python market_researcher.py --bot options
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

_SCRIPTS_DIR = Path(__file__).resolve().parent
_LOGS_DIR = _SCRIPTS_DIR.parent / 'logs'
_LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sys.path.append(str(_SCRIPTS_DIR))
import unusual_flow_scanner

CONTEXT_FILES = {
    'trading': _LOGS_DIR / 'market_context_trading.json',
    'options': _LOGS_DIR / 'market_context_options.json',
}
OPTIONS_PARAMS_FILE = _LOGS_DIR / 'monday_params_options.json'
ROLLING_DAYS = 30

OPTIONS_WATCHLIST = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMD']


# ── Rolling JSON writer ───────────────────────────────────────────────────────

def write_rolling(path: Path, entry: dict, max_days: int = ROLLING_DAYS) -> None:
    """Append entry to rolling JSON file, pruning entries older than max_days."""
    records = []
    if path.exists():
        try:
            with open(path) as f:
                records = json.load(f)
        except Exception:
            records = []
    cutoff = datetime.now() - timedelta(days=max_days)
    records = [
        r for r in records
        if datetime.fromisoformat(r['timestamp']) > cutoff
    ]
    records.append(entry)
    with open(path, 'w') as f:
        json.dump(records, f, indent=2)
    logging.info(f"💾 {path.name}: {len(records)} entries retained (window: {max_days}d)")


# ── Shared data fetchers ──────────────────────────────────────────────────────

def fetch_macro_pulse() -> dict:
    """Fetch core macro indicators: VIX, VIX3M, SPY, QQQ, TLT."""
    result = {}
    indicators = {
        '^VIX':  'vix',
        '^VIX3M': 'vix3m',
        'SPY':   'spy',
        'QQQ':   'qqq',
        'TLT':   'tlt',
    }
    for sym, key in indicators.items():
        try:
            hist = yf.Ticker(sym).history(period='3mo')
            if hist.empty or len(hist) < 5:
                continue
            close = hist['Close']
            current = float(close.iloc[-1])
            d = {'current': round(current, 2)}
            if len(close) >= 21:
                d['return_1m_pct'] = round(
                    float((close.iloc[-1] - close.iloc[-21]) / close.iloc[-21] * 100), 2
                )
            if key == 'vix':
                d['avg_30d'] = round(float(close.tail(30).mean()), 2)
                d['regime'] = 'high' if current > 25 else ('low' if current < 15 else 'normal')
            elif key in ('spy', 'qqq'):
                if len(close) >= 50:
                    ma50 = float(close.rolling(50).mean().iloc[-1])
                    d['above_50ma'] = bool(current > ma50)
                    d['trend'] = 'bullish' if current > ma50 else 'bearish'
            elif key == 'tlt':
                d['flight_to_safety'] = bool(d.get('return_1m_pct', 0) > 2)
            result[key] = d
            time.sleep(0.15)
        except Exception as e:
            logging.warning(f"  ⚠️  {sym}: {e}")

    # VIX term structure: backwardation = fear spike, contango = calm
    if 'vix' in result and 'vix3m' in result:
        spot = result['vix']['current']
        v3m  = result['vix3m']['current']
        result['vix_term_structure'] = {
            'spot': spot,
            'three_month': v3m,
            'shape': 'contango' if v3m > spot else 'backwardation',
        }
    return result


# ETFs never have earnings — skip them to avoid noisy 404s from yfinance
_ETF_SYMBOLS = {
    'SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'SLV', 'TLT', 'LQD', 'HYG',
    'XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLI', 'XLU', 'XLB', 'XLC', 'XLRE',
    'VXX', 'UVXY', 'SQQQ', 'TQQQ', 'SPX',
}


def get_earnings_days(symbol: str) -> Optional[int]:
    """Return days until next earnings for symbol, or None if unavailable/ETF."""
    if symbol.upper() in _ETF_SYMBOLS:
        return None
    # Suppress yfinance's own HTTP error logging — we handle the exception ourselves
    yf_logger = logging.getLogger('yfinance')
    prev_level = yf_logger.level
    yf_logger.setLevel(logging.CRITICAL)
    try:
        cal = yf.Ticker(symbol).calendar
        if not cal:
            return None
        if isinstance(cal, dict):
            ed = cal.get('Earnings Date')
            if ed:
                date = pd.Timestamp(ed[0] if isinstance(ed, list) else ed)
                return int((date - pd.Timestamp.now()).days)
        elif hasattr(cal, 'columns') and len(cal.columns) > 0:
            date = pd.Timestamp(cal.columns[0])
            return int((date - pd.Timestamp.now()).days)
    except Exception:
        pass
    finally:
        yf_logger.setLevel(prev_level)
    return None


def get_atm_iv(symbol: str) -> Optional[float]:
    """ATM implied volatility (%) from nearest options expiry via yfinance."""
    try:
        t = yf.Ticker(symbol)
        exps = t.options
        if not exps:
            return None
        chain = t.option_chain(exps[0])
        hist = t.history(period='5d')
        if hist.empty:
            return None
        current_price = float(hist['Close'].iloc[-1])
        calls = chain.calls.copy()
        calls['dist'] = abs(calls['strike'] - current_price)
        atm = calls.nsmallest(1, 'dist')
        if atm.empty:
            return None
        return round(float(atm['impliedVolatility'].iloc[0]) * 100, 1)
    except Exception:
        return None


# ── Options-specific helpers ──────────────────────────────────────────────────

def options_iv_snapshot(watchlist: list) -> dict:
    """HV, ATM IV, and earnings risk for each symbol in the watchlist."""
    snapshot = {}
    for symbol in watchlist:
        try:
            hist = yf.Ticker(symbol).history(period='90d')
            if hist.empty or len(hist) < 20:
                continue
            returns = hist['Close'].pct_change().dropna()
            hv_20 = round(float(returns.tail(20).std() * np.sqrt(252) * 100), 1)
            hv_60 = round(float(returns.tail(60).std() * np.sqrt(252) * 100), 1) if len(returns) >= 60 else hv_20
            atm_iv = get_atm_iv(symbol)
            days_to_earnings = get_earnings_days(symbol)
            snapshot[symbol] = {
                'hv_20d_pct':       hv_20,
                'hv_60d_pct':       hv_60,
                'atm_iv_pct':       atm_iv,
                'vol_regime':       (
                    'elevated'   if hv_20 > hv_60 * 1.2 else
                    'suppressed' if hv_20 < hv_60 * 0.8 else
                    'normal'
                ),
                'days_to_earnings': days_to_earnings,
                'earnings_risk':    days_to_earnings is not None and 0 <= days_to_earnings <= 3,
            }
            time.sleep(0.4)
        except Exception as e:
            logging.warning(f"  ⚠️  IV snapshot {symbol}: {e}")
    return snapshot


def build_options_params(macro: dict, iv_snap: dict) -> dict:
    """Derive actionable trading params for options_agent from research."""
    vix_regime  = macro.get('vix', {}).get('regime', 'normal')
    term_shape  = macro.get('vix_term_structure', {}).get('shape', 'contango')
    spy_trend   = macro.get('spy', {}).get('trend', 'bullish')
    tlt_safety  = macro.get('tlt', {}).get('flight_to_safety', False)

    # Size down when options are expensive (high VIX), size up when cheap (low VIX)
    size_mult = {'high': 0.6, 'low': 1.2, 'normal': 1.0}.get(vix_regime, 1.0)

    # Overall directional bias for the session
    if tlt_safety or term_shape == 'backwardation':
        bias = 'defensive'   # fear spike — lean puts or stay flat
    elif spy_trend == 'bullish':
        bias = 'bullish'
    elif spy_trend == 'bearish':
        bias = 'bearish'
    else:
        bias = 'neutral'

    # Tickers to skip due to imminent earnings (IV crush risk)
    avoid_earnings = [s for s, v in iv_snap.items() if v.get('earnings_risk')]
    # Tickers where buying options is extra expensive right now
    iv_elevated = [s for s, v in iv_snap.items() if v.get('vol_regime') == 'elevated']

    return {
        'updated_at':              datetime.now().isoformat(),
        'source':                  'market_researcher (weeknight)',
        'vix_regime':              vix_regime,
        'market_bias':             bias,
        'position_size_multiplier': size_mult,
        'min_confidence_override': 0.80 if vix_regime == 'high' else 0.75,
        'avoid_earnings_risk':     avoid_earnings,
        'iv_elevated_tickers':     iv_elevated,
    }


# ── Bot-specific runners ──────────────────────────────────────────────────────

def run_trading_research() -> None:
    """Weeknight light research for the stock trading bot."""
    logging.info("=" * 60)
    logging.info("📊 TRADING BOT — NIGHTLY MARKET RESEARCH")
    logging.info("=" * 60)

    macro = fetch_macro_pulse()
    logging.info(f"  VIX: {macro.get('vix', {}).get('current')} "
                 f"({macro.get('vix', {}).get('regime')})")
    logging.info(f"  SPY: {macro.get('spy', {}).get('trend')} | "
                 f"TLT safety: {macro.get('tlt', {}).get('flight_to_safety')}")

    from stock_discovery import StockDiscovery
    disc    = StockDiscovery()
    universe = disc.get_index_constituents()
    top_volume   = disc.scan_unusual_volume(universe, top_n=10)
    top_momentum = disc.scan_momentum(universe, top_n=10)
    top_oversold = disc.scan_oversold(universe, top_n=5)

    entry = {
        'timestamp':            datetime.now().isoformat(),
        'type':                 'weeknight',
        'macro':                macro,
        'top_volume_tickers':   top_volume,
        'top_momentum_tickers': top_momentum,
        'oversold_tickers':     top_oversold,
    }
    write_rolling(CONTEXT_FILES['trading'], entry)
    logging.info("✅ Trading nightly research complete")


def run_options_research() -> None:
    """Weeknight light research for the options trading bot."""
    logging.info("=" * 60)
    logging.info("💰 OPTIONS BOT — NIGHTLY MARKET RESEARCH")
    logging.info("=" * 60)

    macro   = fetch_macro_pulse()
    iv_snap = options_iv_snapshot(OPTIONS_WATCHLIST)
    params  = build_options_params(macro, iv_snap)

    logging.info(f"  VIX: {macro.get('vix', {}).get('current')} "
                 f"({macro.get('vix', {}).get('regime')})")
    logging.info(f"  Term structure: {macro.get('vix_term_structure', {}).get('shape')}")
    logging.info(f"  Bias: {params['market_bias']} | "
                 f"Size mult: {params['position_size_multiplier']}")
    logging.info(f"  Earnings risk (next 3d): {params['avoid_earnings_risk']}")
    logging.info(f"  IV elevated: {params['iv_elevated_tickers']}")

    # Unusual options flow scan — augments symbol_bias in params
    logging.info("🌊 Scanning unusual options flow...")
    try:
        flow = unusual_flow_scanner.scan_watchlist(OPTIONS_WATCHLIST)
        unusual = flow.get('unusual_symbols', [])
        if unusual:
            logging.info(f"  Unusual flow detected on {len(unusual)} symbols:")
            for r in unusual:
                logging.info(f"    {r['symbol']:<6} {r['signal']:<28} P/C={r['pcr_volume']:.2f}")
        # Inject flow signals into params for options_agent to consume
        params['flow_call_signals'] = [
            r['symbol'] for r in unusual
            if r.get('signal') in ('unusual_call_buying', 'heavy_call_flow')
        ]
        params['flow_put_signals'] = [
            r['symbol'] for r in unusual
            if r.get('signal') in ('unusual_put_buying', 'heavy_put_flow')
        ]
    except Exception as e:
        logging.warning(f"  ⚠️  Flow scan failed: {e}")
        flow = {}
        params.setdefault('flow_call_signals', [])
        params.setdefault('flow_put_signals', [])

    entry = {
        'timestamp':          datetime.now().isoformat(),
        'type':               'weeknight',
        'macro':              macro,
        'iv_snapshot':        iv_snap,
        'unusual_flow':       flow,
        'recommended_params': params,
    }
    write_rolling(CONTEXT_FILES['options'], entry)

    with open(OPTIONS_PARAMS_FILE, 'w') as f:
        json.dump(params, f, indent=2)
    logging.info(f"💾 Options params written → {OPTIONS_PARAMS_FILE.name}")
    logging.info("✅ Options nightly research complete")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nightly market researcher')
    parser.add_argument('--bot', choices=['trading', 'options'], required=True,
                        help='Which bot to run research for')
    args = parser.parse_args()
    if args.bot == 'trading':
        run_trading_research()
    else:
        run_options_research()
