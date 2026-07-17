#!/usr/bin/env python3
"""
expiry_sentinel.py — R1 WI-3: alert-only detect-and-remediate backstop for
long option positions approaching expiration, regardless of whether the
trading daemon that would normally close them is running. Auto-exercise
cannot be pre-gated (it's the broker's own behavior, not an order our code
submits) — this is the visibility layer that was completely missing during
the 2026-07 incident: the exact exposure ($177,500 across 4 contracts) was
invisible until it had already happened.

Deliberately does NOT import options_agent.py — a monitoring tool's
reliability shouldn't depend on the trading agent's full, heavy import
graph (Ollama/vLLM clients, etc.) succeeding. Own small OCC parser instead,
same slicing scheme as options_agent.py::parse_dte_from_symbol().

Usage:
    python scripts/tools/expiry_sentinel.py            # checks + alerts
    python scripts/tools/expiry_sentinel.py --dry-run   # checks, prints only
"""
import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from dotenv import dotenv_values

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _pathfix  # noqa: F401

from alpaca.trading.client import TradingClient
import alerts as _alerts

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_WARN_CALENDAR_DAYS = 5
_CRITICAL_TRADING_DAYS = 2  # aligned in spirit with the agent's exit_dte_threshold


def parse_occ_symbol(symbol: str) -> dict | None:
    """Parse an OCC option symbol into its components.

    Format: {underlying}{YYMMDD}{C|P}{8-digit strike, x1000}. Same slicing
    scheme as options_agent.py::parse_dte_from_symbol() (expiry always at
    symbol[-15:-9]). Returns None for anything that doesn't parse cleanly
    (e.g. a plain stock ticker) rather than raising.
    """
    if len(symbol) < 15:
        return None
    try:
        expiry = datetime.strptime(symbol[-15:-9], '%y%m%d').date()
        option_type = symbol[-9]
        if option_type not in ('C', 'P'):
            return None
        strike = int(symbol[-8:]) / 1000
        underlying = symbol[:-15]
        if not underlying:
            return None
        return {'underlying': underlying, 'expiry': expiry, 'option_type': option_type, 'strike': strike}
    except (ValueError, IndexError):
        return None


def calendar_days_to_expiry(expiry: date, today: date | None = None) -> int:
    today = today or date.today()
    return (expiry - today).days


def trading_days_to_expiry(expiry: date, today: date | None = None) -> int:
    """Weekday-only (Mon-Fri) count of trading sessions remaining, strictly
    after `today` up to and including `expiry`. No NYSE holiday calendar
    dependency exists anywhere in this repo today (checked requirements.txt
    and every is_market_open() call site) -- accepted, documented
    limitation: this can be off by one around a holiday week (Thanksgiving,
    Christmas, July 4th), not a fix target for this brief.
    """
    today = today or date.today()
    if expiry <= today:
        return 0
    count = 0
    d = today
    while d < expiry:
        d += timedelta(days=1)
        if d.weekday() < 5:
            count += 1
    return count


def classify_tier(calendar_days: int, trading_days: int) -> str | None:
    """CRITICAL takes priority over WARN. Always shows worst-case regardless
    of moneyness -- matches the brief's literal wording and is the safer
    default for a monitor (never under-warns an OTM contract that could
    still move ITM before expiry)."""
    if trading_days <= _CRITICAL_TRADING_DAYS:
        return 'critical'
    if calendar_days <= _WARN_CALENDAR_DAYS:
        return 'warn'
    return None


def compute_exposure(qty: float, strike: float) -> float:
    """Worst-case cash exposure if exercised: contracts x 100 x strike --
    the exact number that was invisible during the incident."""
    return qty * 100 * strike


def _build_clients() -> dict:
    clients = {}
    for label, paper, env_name in (('paper', True, '.env'), ('live', False, '.env.live')):
        env = dotenv_values(_PROJECT_ROOT / env_name)
        key = env.get('ALPACA_API_KEY') or env.get('APCA_API_KEY_ID')
        secret = env.get('ALPACA_SECRET_KEY') or env.get('APCA_API_SECRET_KEY')
        if key and secret:
            clients[label] = TradingClient(key, secret, paper=paper)
        else:
            logging.info(f"No credentials for {label} account ({env_name}) — skipping.")
    return clients


def scan_account(client, account_label: str) -> list[dict]:
    """Read-only scan of all option positions on one account. Returns a
    list of {'symbol','underlying','expiry','strike','qty','calendar_days',
    'trading_days','tier','exposure'} for every position with a non-None
    tier (i.e. WARN or CRITICAL). Silent (empty list) if no options held or
    none are within either tier."""
    positions = client.get_all_positions()
    flagged = []
    for p in positions:
        parsed = parse_occ_symbol(p.symbol)
        if parsed is None:
            continue  # not an option position
        cal_days = calendar_days_to_expiry(parsed['expiry'])
        trd_days = trading_days_to_expiry(parsed['expiry'])
        tier = classify_tier(cal_days, trd_days)
        if tier is None:
            continue
        qty = float(p.qty)
        flagged.append({
            'symbol': p.symbol, 'underlying': parsed['underlying'],
            'expiry': parsed['expiry'], 'strike': parsed['strike'],
            'option_type': parsed['option_type'], 'qty': qty,
            'calendar_days': cal_days, 'trading_days': trd_days,
            'tier': tier, 'exposure': compute_exposure(qty, parsed['strike']),
        })
    return flagged


def run(dry_run: bool = False) -> int:
    clients = _build_clients()
    alerted = False

    for account_label, client in clients.items():
        paper = account_label == 'paper'
        try:
            flagged = scan_account(client, account_label)
        except Exception as e:
            logging.error(f"{account_label}: could not scan account — {e}")
            continue

        if not flagged:
            logging.info(f"{account_label}: no options within WARN/CRITICAL window. Clean.")
            continue

        cash = None
        try:
            cash = float(client.get_account().cash)
        except Exception as e:
            logging.warning(f"{account_label}: could not fetch cash for exposure context — {e}")

        total_exposure = sum(f['exposure'] for f in flagged)
        for f in flagged:
            logging.warning(
                f"{account_label}: {f['tier'].upper()} — {f['symbol']} "
                f"({f['calendar_days']}cd/{f['trading_days']}td to expiry), "
                f"qty={f['qty']}, exposure=${f['exposure']:,.2f}"
            )
            level = _alerts.AlertLevel.CRITICAL if f['tier'] == 'critical' else _alerts.AlertLevel.WARNING
            cash_note = f" Account cash: ${cash:,.2f}." if cash is not None else ""
            message = (
                f"{account_label} {f['symbol']}: {f['tier'].upper()} — "
                f"{f['calendar_days']} calendar / {f['trading_days']} trading day(s) to expiry. "
                f"If exercised: {f['qty']:.0f} contract(s) x 100 x ${f['strike']:.2f} strike = "
                f"${f['exposure']:,.2f}. Total flagged exposure this scan: ${total_exposure:,.2f}."
                f"{cash_note}"
            )
            # Account tag must be in the key -- alert_once_per_day's own
            # suffixed_path() reads the sentinel process's ambient env, not
            # per-call account context, and would otherwise let a paper
            # alert suppress a live one for the same symbol/strike/expiry.
            condition_key = f"expiry_{f['tier']}_{account_label}_{f['symbol']}"
            if dry_run:
                logging.info(f"[DRY RUN] Would alert: {condition_key}")
            else:
                _alerts.alert_once_per_day(condition_key, lambda level=level, message=message, f=f: _alerts.send_alert(
                    level, 'expiry_risk', message,
                    data={'account': account_label, 'symbol': f['symbol'], 'tier': f['tier'],
                          'calendar_days': f['calendar_days'], 'trading_days': f['trading_days'],
                          'exposure': f['exposure']},
                ))
            alerted = True

    return 1 if alerted else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true',
                         help='Check and print, but do not send alerts or write dedup state')
    args = parser.parse_args()
    sys.exit(run(dry_run=args.dry_run))


if __name__ == '__main__':
    main()
