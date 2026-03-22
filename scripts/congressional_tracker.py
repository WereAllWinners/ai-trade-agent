#!/usr/bin/env python3
"""
Congressional Trade Tracker

Pulls stock trades by US House and Senate members from the public
House/Senate Stock Watcher datasets — no API key required.

Data sources (free S3 buckets maintained by stockwatcher.dev):
  House:  https://house-stock-watcher-data.s3-us-east-2.amazonaws.com/data/all_transactions.json
  Senate: https://senate-stock-watcher-data.s3-us-east-2.amazonaws.com/aggregate/all_transactions.json

Typical uses:
  - Weekend strategist: add congressional sentiment to weekly context
  - Prompt injection: flag if congress members recently bought/sold a symbol
  - Fine-tuning data: saved to logs/congressional_trades.json each Saturday
"""
import json
import logging
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger(__name__)

_HOUSE_URL  = ('https://house-stock-watcher-data.s3-us-east-2.amazonaws.com'
               '/data/all_transactions.json')
_SENATE_URL = ('https://senate-stock-watcher-data.s3-us-east-2.amazonaws.com'
               '/aggregate/all_transactions.json')
_TIMEOUT    = 30
_LOGS_DIR   = Path(__file__).resolve().parent.parent / 'logs'


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get(url: str) -> list:
    try:
        r = requests.get(url, timeout=_TIMEOUT,
                         headers={'User-Agent': 'ai-trade-agent/1.0'})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning(f"Failed to fetch {url}: {e}")
        return []


def _parse_date(s: str) -> Optional[datetime]:
    for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%Y-%m-%dT%H:%M:%S', '%m/%d/%y'):
        try:
            return datetime.strptime(s.strip(), fmt)
        except Exception:
            pass
    return None


def _normalize_type(raw: str) -> str:
    r = raw.lower()
    if any(w in r for w in ('purchase', 'buy', 'bought')):
        return 'purchase'
    if any(w in r for w in ('sale', 'sell', 'sold', 'exchange')):
        return 'sale'
    return raw.lower()


# ── Core fetch ────────────────────────────────────────────────────────────────

def fetch_congressional_trades(days_back: int = 45) -> list:
    """
    Fetch and normalize House + Senate trades from the last `days_back` days.
    Returns list of dicts: symbol, chamber, rep_name, trade_type, amount, date.
    Sorted by date descending.
    """
    cutoff = datetime.now() - timedelta(days=days_back)
    trades = []

    log.info("  Fetching House trades...")
    for t in _get(_HOUSE_URL):
        ticker = str(t.get('ticker', '')).strip().upper()
        if not ticker or ticker in ('--', 'N/A', 'NONE', ''):
            continue
        date = _parse_date(t.get('transaction_date') or t.get('disclosure_date') or '')
        if not date or date < cutoff:
            continue
        trades.append({
            'symbol':     ticker,
            'chamber':    'House',
            'rep_name':   t.get('representative', 'Unknown'),
            'trade_type': _normalize_type(t.get('type', '')),
            'amount':     t.get('amount', ''),
            'date':       date.strftime('%Y-%m-%d'),
        })

    log.info("  Fetching Senate trades...")
    for t in _get(_SENATE_URL):
        ticker = str(t.get('ticker', '')).strip().upper()
        if not ticker or ticker in ('--', 'N/A', 'NONE', ''):
            continue
        date = _parse_date(
            t.get('transaction_date') or t.get('date') or ''
        )
        if not date or date < cutoff:
            continue
        first = t.get('first_name', '')
        last  = t.get('last_name', '')
        name  = t.get('senator', f"{first} {last}").strip()
        trades.append({
            'symbol':     ticker,
            'chamber':    'Senate',
            'rep_name':   name,
            'trade_type': _normalize_type(t.get('type', '')),
            'amount':     t.get('amount', ''),
            'date':       date.strftime('%Y-%m-%d'),
        })

    trades.sort(key=lambda x: x['date'], reverse=True)
    log.info(f"  {len(trades)} congressional trades in last {days_back} days")
    return trades


# ── Analysis helpers ──────────────────────────────────────────────────────────

def summarize_for_symbol(symbol: str, trades: list) -> Optional[dict]:
    """Buy/sell summary + signal for a single symbol."""
    sym_trades = [t for t in trades if t['symbol'] == symbol]
    if not sym_trades:
        return None
    buys  = [t for t in sym_trades if t['trade_type'] == 'purchase']
    sells = [t for t in sym_trades if t['trade_type'] == 'sale']
    signal = (
        'bullish' if len(buys) > len(sells) else
        'bearish' if len(sells) > len(buys) else
        'neutral'
    )
    return {
        'symbol':       symbol,
        'total_trades': len(sym_trades),
        'buys':         len(buys),
        'sells':        len(sells),
        'signal':       signal,
        'recent':       sym_trades[:3],
    }


def get_top_symbols(trades: list, top_n: int = 25) -> list:
    counts = Counter(t['symbol'] for t in trades)
    return [s for s, _ in counts.most_common(top_n)]


def build_symbol_map(trades: list) -> dict:
    """Return {symbol: summary_dict} for all traded symbols."""
    symbols = set(t['symbol'] for t in trades)
    return {s: summarize_for_symbol(s, trades) for s in symbols}


# ── Prompt formatter ──────────────────────────────────────────────────────────

def format_for_prompt(symbol: str, summary: Optional[dict]) -> str:
    """Compact prompt snippet for a symbol. Returns '' if no activity."""
    if not summary:
        return ''
    signal = summary['signal'].upper()
    lines  = [f"\nCongressional Activity ({symbol}, last 45d): {signal}"]
    lines.append(f"  {summary['buys']} purchase(s), {summary['sells']} sale(s)")
    for t in summary['recent'][:2]:
        lines.append(
            f"  • {t['rep_name']} ({t['chamber']}): "
            f"{t['trade_type']} {t['amount']}  [{t['date']}]"
        )
    return '\n'.join(lines)


# ── Save to disk ──────────────────────────────────────────────────────────────

def run_and_save(days_back: int = 45) -> dict:
    """Fetch trades, compute summary, write to logs/congressional_trades.json."""
    trades      = fetch_congressional_trades(days_back=days_back)
    top_symbols = get_top_symbols(trades, top_n=30)
    symbol_map  = build_symbol_map(trades)

    output = {
        'fetched_at':   datetime.now().isoformat(),
        'days_back':    days_back,
        'total_trades': len(trades),
        'top_symbols':  top_symbols,
        'symbol_map':   symbol_map,
        'trades':       trades,
    }

    _LOGS_DIR.mkdir(exist_ok=True)
    out_path = _LOGS_DIR / 'congressional_trades.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"  Saved {len(trades)} trades → {out_path.name}")
    return output


def load_cached() -> dict:
    """Load the most recently saved congressional trades file, or fetch fresh."""
    path = _LOGS_DIR / 'congressional_trades.json'
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            # Refresh if older than 7 days
            fetched = datetime.fromisoformat(data.get('fetched_at', '2000-01-01'))
            if (datetime.now() - fetched).days < 7:
                return data
        except Exception:
            pass
    return run_and_save()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    result = run_and_save()
    print(f"\nTop symbols by congressional activity ({result['days_back']}d):")
    for sym in result['top_symbols'][:15]:
        s = result['symbol_map'].get(sym)
        if s:
            print(f"  {sym:<6}  {s['buys']}B / {s['sells']}S  → {s['signal']}")
