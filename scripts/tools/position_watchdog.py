#!/usr/bin/env python3
"""
position_watchdog.py — R1 WI-2/WI-7: detects a trading service that is
stopped while its account still holds open positions or orders in its
instrument class ("service down while it has custody" — the exact condition
that let the 2026-07 auto-exercise incident go unnoticed for 10.5 days), and
(WI-7) a breach of the account cash invariant (cash >= 0 and
non_marginable_buying_power >= 0).

Runs standalone, off a systemd timer — deliberately NOT invoked from inside
any trading daemon, so it keeps working precisely when those daemons are
down. Checks both the paper and live accounts in one process.

Usage:
    python scripts/tools/position_watchdog.py           # checks + alerts
    python scripts/tools/position_watchdog.py --dry-run  # checks, prints only
"""
import argparse
import logging
import sys
from pathlib import Path

from dotenv import dotenv_values

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _pathfix  # noqa: F401

from alpaca.trading.client import TradingClient
from trading_service_registry import TRADING_UNITS, bot_positions, service_state
import alerts as _alerts
import halt_buys as _halt_buys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_BACKOFF_HOURS = 6.0  # brief's own stated cadence: "immediately, then every 6h"


def _build_clients() -> dict:
    """Two separate TradingClients (paper, live), dotenv_values() into
    separate dicts per env file — never load_dotenv() twice, matches
    backfill_trade_log_source.py's established dual-account pattern.
    Returns {} for an account whose credentials aren't configured (live may
    not be set up yet) rather than raising."""
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


def check_all_services() -> dict:
    """Return {unit: state} for every unit in TRADING_UNITS, via service_state()."""
    return {u['unit']: service_state(u['unit']) for u in TRADING_UNITS}


def check_exposure(positions, orders, bot: str) -> dict:
    """Classify one bot's exposure within an already-fetched account snapshot.

    'clean'   — no positions/orders in this bot's instrument class.
    'exposed' — positions and/or orders exist; caller should alert.

    Note: this function assumes the fetch already succeeded — an API error
    fetching positions/orders in the first place is a separate 'unknown'
    state handled by the caller (run()), since "couldn't check" must never
    be silently equivalent to "clean" — that's the whole point of this
    brief. Keeping the states separate here (rather than folding 'unknown'
    into this function) means this function has no failure mode of its own
    to test around.
    """
    bot_pos = bot_positions(positions, bot)
    bot_ord = bot_positions(orders, bot)
    if not bot_pos and not bot_ord:
        return {'state': 'clean'}
    return {'state': 'exposed', 'positions': bot_pos, 'orders': bot_ord}


def check_cash_invariant(client, account_label: str) -> dict:
    """The account's own hard safety invariant: cash >= 0 and
    non_marginable_buying_power >= 0. Breaching either is the exact failure
    mode of the 2026-07 incident (cash went to -$58,736.66 while
    non_marginable_buying_power stayed positive — margin/RegT buying-power
    figures do not reflect a cash breach, which is why this checks `cash`
    directly rather than trusting a "looks fine" buying-power number)."""
    account = client.get_account()
    cash = float(account.cash)
    nmbp = float(account.non_marginable_buying_power)
    if cash >= 0 and nmbp >= 0:
        return {'state': 'ok', 'cash': cash, 'non_marginable_buying_power': nmbp}
    return {'state': 'breach', 'cash': cash, 'non_marginable_buying_power': nmbp}


def run(dry_run: bool = False) -> int:
    """One shot. Returns process exit code (0 = no alerts fired, 1 = at
    least one alert fired or would have under --dry-run)."""
    clients = _build_clients()
    states = check_all_services()
    alerted = False

    for account_label, client in clients.items():
        # R1 WI-7: cash invariant, unconditional per account (unrelated to
        # service state -- checked regardless of whether any unit is down).
        paper = account_label == 'paper'
        try:
            cash_result = check_cash_invariant(client, account_label)
        except Exception as e:
            logging.error(f"{account_label}: could not check cash invariant — {e}")
            cash_result = None

        if cash_result is not None and cash_result['state'] == 'breach':
            logging.critical(
                f"{account_label}: CASH INVARIANT BREACHED — cash=${cash_result['cash']:,.2f} "
                f"non_marginable_buying_power=${cash_result['non_marginable_buying_power']:,.2f}"
            )
            if dry_run:
                logging.info(f"[DRY RUN] Would alert + write HALT_BUYS flag for {account_label}")
            else:
                _alerts.alert_with_backoff(
                    f"cash_invariant_breach_{account_label}",
                    lambda al=account_label, r=cash_result: _alerts.alert_cash_negative(
                        al, r['cash'], r['non_marginable_buying_power'],
                    ),
                    min_interval_hours=_BACKOFF_HOURS,
                )
                _halt_buys.write_halt_flag(
                    _halt_buys.flag_path(paper),
                    reason=f"cash=${cash_result['cash']:,.2f} non_marginable_buying_power="
                           f"${cash_result['non_marginable_buying_power']:,.2f}",
                    data=cash_result,
                )
            alerted = True
        # Healthy: do nothing -- no auto-clear anywhere, clearing is manual only.

        account_units = [u for u in TRADING_UNITS if u['paper'] == (account_label == 'paper')]
        # not_found units are skipped entirely (see trading_service_registry
        # docstring) — only truly 'inactive' (installed, not running) units
        # get an exposure check.
        inactive_units = [u for u in account_units if states[u['unit']] == 'inactive']
        not_found = [u for u in account_units if states[u['unit']] == 'not_found']
        for u in not_found:
            logging.info(f"{u['unit']}: not installed — skipping (N/A, not an alert condition).")

        if not inactive_units:
            continue

        try:
            positions = client.get_all_positions()
            orders = client.get_orders()
        except Exception as e:
            for u in inactive_units:
                logging.error(f"{u['unit']} ({account_label}): could not fetch account state — {e}")
                condition_key = f"watchdog_unknown_{account_label}_{u['bot']}"
                if dry_run:
                    logging.info(f"[DRY RUN] Would alert (unknown/API error): {condition_key}")
                else:
                    _alerts.alert_with_backoff(
                        condition_key,
                        lambda u=u, e=e: _alerts.send_alert(
                            _alerts.AlertLevel.WARNING, 'watchdog_check_failed',
                            f"{u['unit']} ({account_label}) is down and its exposure "
                            f"could not be checked: {e}. Treating as unresolved, not clean.",
                            data={'unit': u['unit'], 'account': account_label, 'error': str(e)},
                        ),
                        min_interval_hours=_BACKOFF_HOURS,
                    )
                alerted = True
            continue

        for u in inactive_units:
            result = check_exposure(positions, orders, u['bot'])
            if result['state'] == 'clean':
                logging.info(f"{u['unit']} ({account_label}): down, but no {u['bot']} exposure. Clean.")
                continue
            logging.warning(
                f"{u['unit']} ({account_label}): down with {len(result['positions'])} "
                f"position(s), {len(result['orders'])} order(s) — ALERTING."
            )
            condition_key = f"watchdog_exposed_{account_label}_{u['bot']}"
            if dry_run:
                logging.info(f"[DRY RUN] Would alert (exposed): {condition_key}")
            else:
                _alerts.alert_with_backoff(
                    condition_key,
                    lambda u=u, r=result: _alerts.alert_service_down_with_positions(
                        u['unit'], account_label, u['bot'], r['positions'], r['orders'],
                    ),
                    min_interval_hours=_BACKOFF_HOURS,
                )
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
