#!/usr/bin/env python3
"""
Economic Calendar Guard — Finnhub free-tier API.

Fetches today's high-impact US economic events and provides halt logic
for FOMC / NFP / CPI releases that cause >1% intraday whipsaw.

Env flags:
  MACRO_GUARD_ENABLED=false  — disable guard entirely (never halts)
  FINNHUB_API_KEY=<key>      — free tier at finnhub.io (60 calls/min)

File cache: logs/econ_calendar_YYYY-MM-DD.json (one per calendar day).
API is called at most once per day — negligible against the 60/min limit.
"""
import json
import logging
import os
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

MACRO_GUARD_ENABLED: bool = os.getenv('MACRO_GUARD_ENABLED', 'true').lower() != 'false'
_FINNHUB_KEY = os.getenv('FINNHUB_API_KEY', '')

# Resolve logs/ relative to the repo root (two levels up from scripts/data/)
_LOGS_DIR = Path(__file__).resolve().parent.parent.parent / 'logs'

# Events that reliably cause >1% intraday whipsaw — halt within ±30/60 min window
_HALT_KEYWORDS = frozenset({
    'fomc', 'federal open market committee', 'federal reserve',
    'nonfarm payroll', 'non-farm payroll', 'nfp', 'nonfarm',
    'cpi', 'consumer price index',
})


# ── File-based cache ──────────────────────────────────────────────────────────

def _cache_path() -> Path:
    today = datetime.now().strftime('%Y-%m-%d')
    return _LOGS_DIR / f'econ_calendar_{today}.json'


# ── ET timezone helper ────────────────────────────────────────────────────────

def _to_et_str(utc_str: str) -> str:
    """Convert a UTC datetime string to a human-readable ET time string."""
    try:
        try:
            from zoneinfo import ZoneInfo
            _et = ZoneInfo('America/New_York')
            dt_utc = datetime.fromisoformat(utc_str.replace('Z', '+00:00'))
            dt_et  = dt_utc.astimezone(_et)
        except ImportError:
            # Python < 3.9 fallback: assume EDT (UTC-4) during market hours
            dt_utc = datetime.fromisoformat(utc_str.replace('Z', '+00:00'))
            dt_et  = dt_utc.replace(tzinfo=None) - timedelta(hours=4)
        return dt_et.strftime('%-I:%M %p')
    except Exception:
        return utc_str  # show raw string rather than crashing


# ── Public API ────────────────────────────────────────────────────────────────

def get_todays_high_impact_events() -> list[dict]:
    """Return today's high-impact US economic events.

    Results are cached to logs/econ_calendar_YYYY-MM-DD.json so the
    Finnhub API is called at most once per calendar day.
    Returns [] on any failure — the guard must never block trading
    due to an API outage.

    Each event dict: {time, event, impact, country}
    """
    if not MACRO_GUARD_ENABLED:
        return []

    # File-based cache hit (valid for the whole calendar day)
    cache = _cache_path()
    if cache.exists():
        try:
            with open(cache) as f:
                return json.load(f)
        except Exception:
            pass  # corrupted cache — re-fetch

    if not _FINNHUB_KEY:
        log.debug("FINNHUB_API_KEY not set — skipping economic calendar")
        return []

    today = datetime.now().strftime('%Y-%m-%d')
    try:
        resp = requests.get(
            'https://finnhub.io/api/v1/calendar/economic',
            params={'from': today, 'to': today, 'token': _FINNHUB_KEY},
            timeout=10,
        )
        resp.raise_for_status()
        raw = resp.json().get('economicCalendar', [])
    except Exception as e:
        log.warning(f"Finnhub economic calendar fetch failed: {e}")
        return []

    events = []
    for item in raw:
        # Normalize impact: Finnhub uses "3"=high / "2"=medium / "1"=low
        # (or sometimes the word strings directly)
        raw_impact = str(item.get('impact', '')).lower().strip()
        if raw_impact in ('3', 'high'):
            impact = 'high'
        elif raw_impact in ('2', 'medium'):
            impact = 'medium'
        else:
            impact = 'low'

        country = str(item.get('country', '')).upper()
        if country != 'US' or impact != 'high':
            continue

        events.append({
            'time':    item.get('time', ''),
            'event':   item.get('event', ''),
            'impact':  impact,
            'country': country,
        })

    # Write cache
    _LOGS_DIR.mkdir(exist_ok=True)
    try:
        with open(cache, 'w') as f:
            json.dump(events, f)
    except Exception as e:
        log.warning(f"Could not write economic calendar cache: {e}")

    return events


def get_earnings_today(symbols: list) -> list[str]:
    """Return the subset of symbols with earnings today or tomorrow.

    Uses yfinance .calendar — best-effort, skips any symbol that fails.
    Returns [] on any error so callers always get a list.
    """
    if not symbols:
        return []

    today    = date.today()
    tomorrow = today + timedelta(days=1)
    result   = []

    for sym in symbols:
        try:
            import yfinance as yf
            import pandas as pd
            cal = yf.Ticker(sym).calendar
            if not cal:
                continue
            # yfinance returns either a dict or a DataFrame
            if isinstance(cal, dict):
                raw_dates = cal.get('Earnings Date', [])
                if not isinstance(raw_dates, list):
                    raw_dates = [raw_dates]
            elif hasattr(cal, 'columns'):
                raw_dates = list(cal.columns)
            else:
                continue

            for d in raw_dates:
                try:
                    dt = pd.Timestamp(d).date()
                    if dt in (today, tomorrow):
                        result.append(sym)
                        break
                except Exception:
                    continue
        except Exception as e:
            log.debug(f"Earnings check failed for {sym}: {e}")

    return result


def format_macro_guard_block(events: list, earnings_today: list) -> str:
    """Format a prompt-ready macro guard block.

    Returns '' if there are no events or earnings to report.
    """
    lines = []
    if events:
        parts = []
        for e in events[:3]:
            time_et = _to_et_str(e['time'])
            parts.append(f"{e['event']} @ {time_et} ET [{e['impact'].upper()} IMPACT]")
        lines.append(f"⚠️  Macro Events Today: {' | '.join(parts)}")
    if earnings_today:
        lines.append(
            f"⚠️  Earnings Today/Tomorrow: {', '.join(earnings_today)}"
            f" — elevated IV crush risk"
        )
    return '\n'.join(lines)


def should_halt_trading(events: list) -> tuple[bool, str]:
    """Return (True, reason) if any FOMC/NFP/CPI event is within the halt window.

    Halt window: event time is within the next 30 minutes OR occurred within
    the last 60 minutes of the current moment.  These three event types
    reliably cause >1% intraday whipsaw.

    Returns (False, '') when the list is empty, guard is disabled, or no
    halt-eligible event falls in the window.
    """
    if not events or not MACRO_GUARD_ENABLED:
        return False, ''

    now_utc = datetime.now(timezone.utc)

    for event in events:
        name = event.get('event', '').lower()
        if not any(kw in name for kw in _HALT_KEYWORDS):
            continue

        time_str = event.get('time', '')
        if not time_str:
            continue

        try:
            # Finnhub times are UTC ("2024-01-31 08:30:00" or ISO format)
            normalized = time_str.replace(' ', 'T')
            if '+' not in normalized and not normalized.endswith('Z'):
                normalized += '+00:00'
            event_utc = datetime.fromisoformat(normalized)
        except Exception:
            log.debug(f"Could not parse event time '{time_str}' for halt check")
            continue

        delta_min = (event_utc - now_utc).total_seconds() / 60
        if -60 <= delta_min <= 30:
            direction = f"in {int(delta_min)}min" if delta_min > 0 else f"{int(-delta_min)}min ago"
            reason = f"{event['event']} — {direction}"
            return True, reason

    return False, ''
