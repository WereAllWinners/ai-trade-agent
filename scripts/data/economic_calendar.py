#!/usr/bin/env python3
"""
Economic Calendar Guard — static committed schedule.

Fetches today's high-impact US economic events and provides halt logic
for FOMC / NFP / CPI releases that cause >1% intraday whipsaw.

sprint04 F1: the previous Finnhub free-tier integration 403'd on every
call (entitlement limit, not a bug) and never actually functioned. It's
been replaced with a static, committed schedule extracted from official
Fed/BLS pages via scripts/tools/build_macro_calendar.py — see CLAUDE.md
for the refresh procedure. Zero network calls, zero API key.

Env flags:
  MACRO_GUARD_ENABLED=false  — disable guard entirely (never halts)

Data file: data/macro_event_calendar.json (committed, refreshed manually —
see CLAUDE.md). check_calendar_staleness() warns/alerts once per day if
the file has run out of future-dated events.
"""
import json
import logging
import os
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

MACRO_GUARD_ENABLED: bool = os.getenv('MACRO_GUARD_ENABLED', 'true').lower() != 'false'

# Resolve repo-root-relative paths (two levels up from scripts/data/)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CALENDAR_PATH = _REPO_ROOT / 'data' / 'macro_event_calendar.json'

_ET = ZoneInfo('America/New_York')

# Events that reliably cause >1% intraday whipsaw — halt within ±30/60 min window
_HALT_KEYWORDS = frozenset({
    'fomc', 'federal open market committee', 'federal reserve',
    'nonfarm payroll', 'non-farm payroll', 'nfp', 'nonfarm',
    'cpi', 'consumer price index',
})


# ── Static calendar loading ────────────────────────────────────────────────────

def _load_static_calendar() -> list[dict]:
    """Read data/macro_event_calendar.json. Returns [] on any read/parse
    failure — same fail-open contract the old cache-read path had."""
    try:
        with open(_CALENDAR_PATH) as f:
            data = json.load(f)
        return data.get('events', [])
    except Exception as e:
        log.debug(f"Could not load {_CALENDAR_PATH}: {e}")
        return []


# ── ET timezone helper ────────────────────────────────────────────────────────

def _to_et_str(utc_str: str) -> str:
    """Convert a UTC datetime string to a human-readable ET time string."""
    try:
        dt_utc = datetime.fromisoformat(utc_str.replace('Z', '+00:00'))
        dt_et = dt_utc.astimezone(_ET)
        return dt_et.strftime('%-I:%M %p')
    except Exception:
        return utc_str  # show raw string rather than crashing


# ── Public API ────────────────────────────────────────────────────────────────

def get_todays_high_impact_events() -> list[dict]:
    """Return today's high-impact US economic events.

    Reads the static, committed data/macro_event_calendar.json — no
    network calls. Returns [] on any failure — the guard must never block
    trading due to a missing/malformed data file.

    Each event dict: {time, event, impact, country} — 'time' is a UTC
    string in 'YYYY-MM-DD HH:MM:SS' format, matching the shape
    should_halt_trading() already expects.
    """
    if not MACRO_GUARD_ENABLED:
        return []

    events = _load_static_calendar()
    if not events:
        return []

    today_et = datetime.now(_ET).date()

    result = []
    for item in events:
        try:
            event_date = date.fromisoformat(item['date'])
        except Exception:
            continue
        if event_date != today_et:
            continue

        try:
            hour, minute = (int(p) for p in item['time_et'].split(':'))
            dt_et = datetime(
                event_date.year, event_date.month, event_date.day,
                hour, minute, tzinfo=_ET,
            )
            dt_utc = dt_et.astimezone(timezone.utc)
        except Exception:
            continue

        result.append({
            'time':    dt_utc.strftime('%Y-%m-%d %H:%M:%S'),
            'event':   item.get('event', ''),
            'impact':  'high',
            'country': 'US',
        })

    return result


def check_calendar_staleness() -> None:
    """Warn (and alert once/day) if the static macro calendar has no
    future-dated events left (sprint04 F1.3). Call once per session.

    Fails open: never raises, never blocks trading.
    """
    try:
        events = _load_static_calendar()
        today = date.today()

        has_future = False
        for item in events:
            try:
                if date.fromisoformat(item['date']) >= today:
                    has_future = True
                    break
            except Exception:
                continue

        if not has_future:
            reason = (
                "data/macro_event_calendar.json has no future-dated events — "
                "run scripts/tools/build_macro_calendar.py to refresh (see CLAUDE.md)"
            )
            log.warning(f"⚠️  {reason}")
            try:
                import sys
                sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
                from alerts import alert_macro_calendar_stale
                alert_macro_calendar_stale(reason)
            except Exception as alert_err:
                log.debug(f"Could not send macro_calendar_stale alert: {alert_err}")
    except Exception as e:
        log.debug(f"check_calendar_staleness failed: {e}")


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
            # Times are UTC ("2024-01-31 08:30:00" or ISO format)
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
