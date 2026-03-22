#!/usr/bin/env python3
"""
Options Weekend Strategist - Deep weekly research for the options trading bot.

Runs Saturday morning. Writes:
  - logs/weekend_context_options.json  (12-week rolling history for training)
  - logs/monday_params_options.json    (actionable params for options_agent)

Covers: macro pulse, VIX term structure, sector trends, full IV analysis
per watchlist symbol (ATM IV, put/call ratio, HV regime), and earnings
calendar for the entire upcoming week.
"""
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
sys.path.append(str(_SCRIPTS_DIR))

from market_researcher import (
    fetch_macro_pulse,
    get_earnings_days,
    write_rolling,
    OPTIONS_PARAMS_FILE,
)
import congressional_tracker
import unusual_flow_scanner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

WEEKEND_CONTEXT_FILE = _LOGS_DIR / 'weekend_context_options.json'
WEEKEND_ROLLING_DAYS = 12 * 7  # 12 weeks

OPTIONS_WATCHLIST = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMD']

SECTOR_ETFS = {
    'Technology':             'XLK',
    'Financials':             'XLF',
    'Energy':                 'XLE',
    'Healthcare':             'XLV',
    'Consumer Discretionary': 'XLY',
    'Industrials':            'XLI',
    'Utilities':              'XLU',
    'Materials':              'XLB',
}

# Extended universe for earnings calendar scan
EARNINGS_SCAN_UNIVERSE = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AMD', 'INTC', 'QCOM',
    'JPM',  'BAC',  'GS',   'MS',   'WFC',   'C',
    'XOM',  'CVX',
    'UNH',  'JNJ',  'PFE',  'ABBV', 'MRK',  'LLY',
    'WMT',  'HD',   'MCD',  'COST',
]


# ── Research components ───────────────────────────────────────────────────────

def sector_analysis() -> dict:
    """1-month and 3-month returns per sector ETF with trend."""
    logging.info("🔄 Sector analysis...")
    results = {}
    for sector, etf in SECTOR_ETFS.items():
        try:
            hist = yf.Ticker(etf).history(period='3mo')
            if hist.empty or len(hist) < 20:
                continue
            close = hist['Close']
            ret_1m = float((close.iloc[-1] - close.iloc[-21]) / close.iloc[-21] * 100)
            ret_3m = float((close.iloc[-1] - close.iloc[0])   / close.iloc[0]   * 100)
            ma20   = float(close.rolling(20).mean().iloc[-1])
            results[sector] = {
                'etf':            etf,
                'return_1m_pct':  round(ret_1m, 2),
                'return_3m_pct':  round(ret_3m, 2),
                'above_20ma':     bool(close.iloc[-1] > ma20),
                'trend':          'bullish' if close.iloc[-1] > ma20 else 'bearish',
            }
            time.sleep(0.2)
        except Exception as e:
            logging.warning(f"  ⚠️  {etf}: {e}")

    ranked = sorted(results.items(), key=lambda x: x[1]['return_1m_pct'], reverse=True)
    if ranked:
        logging.info(f"  📈 Top sector:  {ranked[0][0]}  ({ranked[0][1]['return_1m_pct']:+.1f}%)")
        logging.info(f"  📉 Weak sector: {ranked[-1][0]} ({ranked[-1][1]['return_1m_pct']:+.1f}%)")
    return dict(ranked)


def earnings_next_week(universe: list) -> list:
    """All earnings events in the next 7 days from the scan universe."""
    logging.info("📅 Scanning for next-week earnings...")
    events = []
    for symbol in universe:
        days = get_earnings_days(symbol)
        if days is not None and 0 <= days <= 7:
            events.append({'symbol': symbol, 'earnings_in_days': days})
        time.sleep(0.2)
    events.sort(key=lambda x: x['earnings_in_days'])
    logging.info(f"  Found {len(events)} events: "
                 f"{[e['symbol'] for e in events]}")
    return events


def full_iv_analysis(watchlist: list) -> dict:
    """
    Per-symbol IV analysis: ATM IV on near + far expiry, put/call ratio,
    historical vol regime, and an IV term structure shape.
    """
    logging.info("📊 Full IV analysis...")
    analysis = {}

    for symbol in watchlist:
        try:
            t    = yf.Ticker(symbol)
            hist = t.history(period='252d')
            if hist.empty or len(hist) < 60:
                continue

            returns  = hist['Close'].pct_change().dropna()
            hv_20    = round(float(returns.tail(20).std()  * np.sqrt(252) * 100), 1)
            hv_60    = round(float(returns.tail(60).std()  * np.sqrt(252) * 100), 1)
            hv_252   = round(float(returns.std()           * np.sqrt(252) * 100), 1)
            cur_price = float(hist['Close'].iloc[-1])

            exps = t.options
            if not exps:
                continue

            near_iv = far_iv = put_call_ratio = None

            # Near expiry: ATM IV + put/call ratio
            try:
                near = t.option_chain(exps[0])
                calls = near.calls.copy()
                puts  = near.puts.copy()
                calls['dist'] = abs(calls['strike'] - cur_price)
                atm = calls.nsmallest(1, 'dist')
                if not atm.empty:
                    near_iv = round(float(atm['impliedVolatility'].iloc[0]) * 100, 1)
                call_vol = calls['volume'].fillna(0).sum()
                put_vol  = puts['volume'].fillna(0).sum()
                if call_vol > 0:
                    put_call_ratio = round(float(put_vol / call_vol), 2)
                time.sleep(0.3)
            except Exception:
                pass

            # Far expiry: ATM IV for term structure shape
            try:
                if len(exps) > 1:
                    far   = t.option_chain(exps[1])
                    fcalls = far.calls.copy()
                    fcalls['dist'] = abs(fcalls['strike'] - cur_price)
                    fatm = fcalls.nsmallest(1, 'dist')
                    if not fatm.empty:
                        far_iv = round(float(fatm['impliedVolatility'].iloc[0]) * 100, 1)
                    time.sleep(0.3)
            except Exception:
                pass

            # HV regime: is short-term vol elevated vs annual baseline?
            hv_regime = (
                'elevated'   if hv_20 > hv_252 * 1.15 else
                'suppressed' if hv_20 < hv_252 * 0.85 else
                'normal'
            )

            # IV term structure (near vs far expiry)
            if near_iv and far_iv:
                iv_ts = 'contango' if far_iv > near_iv else 'backwardation'
            else:
                iv_ts = 'unknown'

            # PCR signal: >1.2 bearish sentiment, <0.7 bullish sentiment
            if put_call_ratio is not None:
                pcr_signal = (
                    'bearish' if put_call_ratio > 1.2 else
                    'bullish' if put_call_ratio < 0.7 else
                    'neutral'
                )
            else:
                pcr_signal = 'neutral'

            analysis[symbol] = {
                'current_price':     round(cur_price, 2),
                'hv_20d_pct':        hv_20,
                'hv_60d_pct':        hv_60,
                'hv_252d_pct':       hv_252,
                'near_atm_iv_pct':   near_iv,
                'far_atm_iv_pct':    far_iv,
                'iv_term_structure': iv_ts,
                'put_call_ratio':    put_call_ratio,
                'pcr_signal':        pcr_signal,
                'hv_regime':         hv_regime,
            }
            logging.info(f"  {symbol:5s}: near_IV={near_iv}%  "
                         f"P/C={put_call_ratio}  regime={hv_regime}")
            time.sleep(0.4)

        except Exception as e:
            logging.warning(f"  ⚠️  {symbol}: {e}")

    return analysis


# ── Param builder ─────────────────────────────────────────────────────────────

def build_weekly_options_params(
    macro: dict,
    iv_analysis: dict,
    sectors: dict,
    earnings_events: list,
    congressional: dict | None = None,
    flow: dict | None = None,
) -> dict:
    """Comprehensive weekly params for options_agent."""
    vix_regime = macro.get('vix', {}).get('regime', 'normal')
    term_shape = macro.get('vix_term_structure', {}).get('shape', 'contango')
    spy_trend  = macro.get('spy', {}).get('trend', 'bullish')
    tlt_safety = macro.get('tlt', {}).get('flight_to_safety', False)

    # Size multiplier based on vol environment
    size_mult = {'high': 0.6, 'low': 1.2, 'normal': 1.0}.get(vix_regime, 1.0)

    # Overall market bias
    if tlt_safety or term_shape == 'backwardation':
        bias = 'defensive'
    elif spy_trend == 'bullish':
        bias = 'bullish'
    elif spy_trend == 'bearish':
        bias = 'bearish'
    else:
        bias = 'neutral'

    # Earnings avoidance: skip tickers with earnings ≤ 4 days away (IV crush risk)
    avoid_earnings = [e['symbol'] for e in earnings_events if e['earnings_in_days'] <= 4]

    # Per-symbol directional lean combining market bias + put/call signal
    symbol_bias = {}
    for sym, data in iv_analysis.items():
        pcr_sig  = data.get('pcr_signal', 'neutral')
        hv_reg   = data.get('hv_regime',  'normal')
        if bias == 'defensive':
            sym_bias = 'avoid' if hv_reg == 'elevated' else 'lean_put'
        elif bias == 'bullish' and pcr_sig != 'bearish':
            sym_bias = 'lean_call'
        elif bias == 'bearish' or pcr_sig == 'bearish':
            sym_bias = 'lean_put'
        else:
            sym_bias = 'neutral'
        symbol_bias[sym] = sym_bias

    # Override symbol bias with unusual options flow signals
    if flow:
        for r in flow.get('unusual_symbols', []):
            sym = r['symbol']
            if sym in symbol_bias:
                sig = r.get('signal', '')
                if sig in ('unusual_call_buying', 'heavy_call_flow'):
                    symbol_bias[sym] = 'lean_call'
                elif sig in ('unusual_put_buying', 'heavy_put_flow'):
                    symbol_bias[sym] = 'lean_put'

    # Congressional signals as supplemental context
    cong_bullish = (congressional or {}).get('bullish', [])
    cong_bearish = (congressional or {}).get('bearish', [])

    # Top sectors (bullish, ranked by 1-month return)
    top_sectors  = [s for s, d in list(sectors.items())[:3] if d.get('trend') == 'bullish']
    weak_sectors = [s for s, d in list(sectors.items())[-2:]]

    return {
        'updated_at':              datetime.now().isoformat(),
        'source':                  'options_weekend_strategist',
        'vix_regime':              vix_regime,
        'market_bias':             bias,
        'position_size_multiplier': size_mult,
        'min_confidence_override': 0.80 if vix_regime == 'high' else 0.75,
        'avoid_earnings_risk':     avoid_earnings,
        'earnings_this_week':      earnings_events,
        'symbol_bias':             symbol_bias,
        'top_sectors':             top_sectors,
        'weak_sectors':            weak_sectors,
        'congressional_bullish':   cong_bullish,
        'congressional_bearish':   cong_bearish,
    }


# ── Main orchestrator ─────────────────────────────────────────────────────────

def run_weekend_analysis() -> dict:
    logging.info("=" * 70)
    logging.info("🏖️  OPTIONS WEEKEND STRATEGIST")
    logging.info(f"📅 {datetime.now().strftime('%A, %B %d, %Y')}")
    logging.info("=" * 70)

    macro           = fetch_macro_pulse()
    sectors         = sector_analysis()
    earnings_events = earnings_next_week(EARNINGS_SCAN_UNIVERSE)
    iv_analysis     = full_iv_analysis(OPTIONS_WATCHLIST)

    # Congressional trades — use cached data (refreshed only if >7 days old)
    logging.info("🏛️  Loading congressional trade data...")
    try:
        cong_data    = congressional_tracker.load_cached()
        cong_summary = {
            'total_trades': cong_data.get('total_trades', 0),
            'top_symbols':  cong_data.get('top_symbols', [])[:20],
            'bullish': [
                s for s in cong_data.get('top_symbols', [])[:20]
                if (cong_data.get('symbol_map', {}).get(s) or {}).get('signal') == 'bullish'
            ],
            'bearish': [
                s for s in cong_data.get('top_symbols', [])[:20]
                if (cong_data.get('symbol_map', {}).get(s) or {}).get('signal') == 'bearish'
            ],
        }
        logging.info(f"  Congressional: {cong_summary['total_trades']} trades | "
                     f"bullish: {cong_summary['bullish'][:5]}")
    except Exception as e:
        logging.warning(f"  ⚠️  Congressional data unavailable: {e}")
        cong_summary = {}

    # Unusual options flow — full watchlist scan
    logging.info("🌊 Scanning unusual options flow...")
    try:
        flow_data = unusual_flow_scanner.scan_watchlist(OPTIONS_WATCHLIST)
        logging.info(f"  Unusual flow: {flow_data.get('unusual_count', 0)}/"
                     f"{flow_data.get('total_scanned', 0)} symbols")
        for r in flow_data.get('unusual_symbols', []):
            logging.info(f"    {r['symbol']:<6} {r['signal']:<28} P/C={r['pcr_volume']:.2f}")
    except Exception as e:
        logging.warning(f"  ⚠️  Flow scan unavailable: {e}")
        flow_data = {}

    params = build_weekly_options_params(
        macro, iv_analysis, sectors, earnings_events,
        congressional=cong_summary,
        flow=flow_data,
    )

    entry = {
        'timestamp':            datetime.now().isoformat(),
        'type':                 'weekend',
        'macro':                macro,
        'sector_analysis':      sectors,
        'earnings_next_7_days': earnings_events,
        'iv_analysis':          iv_analysis,
        'congressional':        cong_summary,
        'unusual_flow':         flow_data,
        'recommended_params':   params,
    }

    # 12-week rolling context for training
    write_rolling(WEEKEND_CONTEXT_FILE, entry, max_days=WEEKEND_ROLLING_DAYS)

    # Overwrite current actionable params for options_agent
    with open(OPTIONS_PARAMS_FILE, 'w') as f:
        json.dump(params, f, indent=2)

    logging.info("=" * 70)
    logging.info(f"  Market bias:        {params['market_bias']}")
    logging.info(f"  VIX regime:         {params['vix_regime']}")
    logging.info(f"  Size multiplier:    {params['position_size_multiplier']}")
    logging.info(f"  Avoid (earnings):   {params['avoid_earnings_risk']}")
    logging.info(f"  Top sectors:        {params['top_sectors']}")
    logging.info(f"  Symbol biases:      {params['symbol_bias']}")
    logging.info(f"  Cong. bullish:      {params['congressional_bullish'][:5]}")
    logging.info(f"  Cong. bearish:      {params['congressional_bearish'][:5]}")
    logging.info("✅ OPTIONS WEEKEND ANALYSIS COMPLETE")
    logging.info("=" * 70)

    return entry


if __name__ == '__main__':
    run_weekend_analysis()
