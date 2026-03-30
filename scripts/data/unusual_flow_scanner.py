#!/usr/bin/env python3
"""
Unusual Options Flow Scanner

Scans yfinance option chains for unusual activity — no paid API required.

Signals detected:
  - High volume/OI ratio on specific strikes (new money entering, not just rolls)
  - Significant OTM option buying (speculative directional bets)
  - Put/call ratio divergence from a neutral baseline (0.7–1.3)

Used by:
  - options_weekend_strategist.py  (deep weekly scan, full watchlist)
  - market_researcher.py           (nightly light scan, watchlist only)
  - options_agent.py               (per-symbol check before trade prompt)
"""
import logging
import time
from datetime import datetime
from typing import Optional

import numpy as np
import yfinance as yf

log = logging.getLogger(__name__)

_DEFAULT_WATCHLIST = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMD']

# Thresholds
_VOL_OI_THRESHOLD   = 1.0    # volume > open interest = strong new-money signal
_OTM_CALL_PCT       = 0.05   # strike > price * (1 + this) = OTM call
_OTM_PUT_PCT        = 0.05   # strike < price * (1 - this) = OTM put
_MIN_CONTRACT_VOL   = 50     # ignore illiquid strikes below this volume
_PCR_BULLISH_CAP    = 0.55   # P/C below this = heavy call flow
_PCR_BEARISH_FLOOR  = 1.45   # P/C above this = heavy put flow


# ── Per-symbol scanner ────────────────────────────────────────────────────────

def scan_symbol(symbol: str) -> Optional[dict]:
    """
    Scan the nearest expiry option chain for unusual activity.
    Returns a result dict, or None on data failure.
    """
    try:
        t    = yf.Ticker(symbol)
        exps = t.options
        if not exps:
            return None

        hist = t.history(period='5d')
        if hist.empty:
            return None
        current_price = float(hist['Close'].iloc[-1])

        near_exp = exps[0]
        chain    = t.option_chain(near_exp)
        calls    = chain.calls.copy()
        puts     = chain.puts.copy()

        # Aggregate volume and OI
        call_vol = float(calls['volume'].fillna(0).sum())
        put_vol  = float(puts['volume'].fillna(0).sum())
        call_oi  = float(calls['openInterest'].fillna(0).sum())
        put_oi   = float(puts['openInterest'].fillna(0).sum())

        pcr_volume = put_vol / call_vol if call_vol > 0 else 1.0
        pcr_oi     = put_oi  / call_oi  if call_oi  > 0 else 1.0

        # Unusual OTM strikes: high vol/OI + meaningfully OTM + minimum liquidity
        calls['vol_oi'] = (calls['volume'].fillna(0)
                           / calls['openInterest'].replace(0, np.nan))
        puts['vol_oi']  = (puts['volume'].fillna(0)
                           / puts['openInterest'].replace(0, np.nan))

        otm_call_mask = (
            (calls['strike']       > current_price * (1 + _OTM_CALL_PCT)) &
            (calls['vol_oi']       > _VOL_OI_THRESHOLD) &
            (calls['volume'].fillna(0) > _MIN_CONTRACT_VOL)
        )
        otm_put_mask = (
            (puts['strike']        < current_price * (1 - _OTM_PUT_PCT)) &
            (puts['vol_oi']        > _VOL_OI_THRESHOLD) &
            (puts['volume'].fillna(0) > _MIN_CONTRACT_VOL)
        )

        unusual_calls = (
            calls[otm_call_mask]
            .nlargest(3, 'vol_oi')
            [['strike', 'volume', 'openInterest', 'vol_oi', 'impliedVolatility']]
            .round(3)
            .to_dict('records')
        )
        unusual_puts = (
            puts[otm_put_mask]
            .nlargest(3, 'vol_oi')
            [['strike', 'volume', 'openInterest', 'vol_oi', 'impliedVolatility']]
            .round(3)
            .to_dict('records')
        )

        has_otm_calls = len(unusual_calls) > 0
        has_otm_puts  = len(unusual_puts)  > 0

        if   has_otm_calls and not has_otm_puts:   signal = 'unusual_call_buying'
        elif has_otm_puts  and not has_otm_calls:   signal = 'unusual_put_buying'
        elif has_otm_calls and has_otm_puts:        signal = 'unusual_both_sides'
        elif pcr_volume < _PCR_BULLISH_CAP:         signal = 'heavy_call_flow'
        elif pcr_volume > _PCR_BEARISH_FLOOR:       signal = 'heavy_put_flow'
        else:                                        signal = 'normal'

        return {
            'symbol':          symbol,
            'current_price':   round(current_price, 2),
            'expiry_scanned':  near_exp,
            'call_volume':     int(call_vol),
            'put_volume':      int(put_vol),
            'pcr_volume':      round(pcr_volume, 2),
            'pcr_oi':          round(pcr_oi, 2),
            'unusual_calls':   unusual_calls,
            'unusual_puts':    unusual_puts,
            'signal':          signal,
            'is_unusual':      signal != 'normal',
        }

    except Exception as e:
        log.warning(f"  Flow scan failed for {symbol}: {e}")
        return None


# ── Watchlist scanner ─────────────────────────────────────────────────────────

def scan_watchlist(watchlist: list) -> dict:
    """
    Scan the full watchlist. Returns a summary dict including a ranked list
    of unusual symbols (most extreme P/C divergence first).
    """
    results  = {}
    unusual  = []

    for symbol in watchlist:
        log.info(f"  Scanning {symbol} options flow...")
        r = scan_symbol(symbol)
        if r:
            results[symbol] = r
            if r['is_unusual']:
                unusual.append(r)
                log.info(f"    🚨 {symbol}: {r['signal']}  P/C={r['pcr_volume']:.2f}")
        time.sleep(0.5)   # gentle on yfinance

    # Rank unusual symbols by divergence from 1.0 (neutral)
    unusual.sort(key=lambda x: abs(x['pcr_volume'] - 1.0), reverse=True)

    return {
        'scanned_at':      datetime.now().isoformat(),
        'total_scanned':   len(results),
        'unusual_count':   len(unusual),
        'unusual_symbols': unusual,
        'all_results':     results,
    }


# ── Prompt formatter ──────────────────────────────────────────────────────────

def format_for_prompt(symbol: str, scan_result: Optional[dict]) -> str:
    """
    Compact prompt snippet for a symbol.
    Returns '' if result is None or signal is normal.
    """
    if not scan_result or not scan_result.get('is_unusual'):
        return ''
    sig = scan_result['signal'].replace('_', ' ').upper()
    pcr = scan_result['pcr_volume']
    cv  = scan_result['call_volume']
    pv  = scan_result['put_volume']
    return (
        f"\nOptions Flow ({symbol}): {sig}  "
        f"(P/C ratio: {pcr:.2f} | call vol: {cv:,} | put vol: {pv:,})"
    )


if __name__ == '__main__':
    import json
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    results = scan_watchlist(_DEFAULT_WATCHLIST)
    print(f"\nUnusual flow: {results['unusual_count']}/{results['total_scanned']} symbols")
    for r in results['unusual_symbols']:
        print(f"  {r['symbol']:<6}  {r['signal']:<28}  P/C={r['pcr_volume']:.2f}")
    print(json.dumps(results, indent=2, default=str))
