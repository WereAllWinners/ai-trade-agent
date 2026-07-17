#!/usr/bin/env python3
"""
pre_stop_check.py — R1 WI-9a: advisory custody check, intended to be run
before any manual `systemctl stop` on a trading service. Prints the
custody state (open positions/orders for that bot) and whether the
position-watchdog timer is active, and exits nonzero if stopping would
violate the custody rule (docs/operations.md).

Advisory only — does NOT wrap, call, or block `systemctl stop` itself.

Usage:
    python scripts/tools/pre_stop_check.py ai-options-bot.service
"""
import sys
from pathlib import Path
from typing import NamedTuple

from dotenv import dotenv_values

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _pathfix  # noqa: F401

from alpaca.trading.client import TradingClient
from trading_service_registry import TRADING_UNITS, bot_positions, service_state

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_WATCHDOG_TIMER = 'ai-position-watchdog.timer'


class CustodyCheckResult(NamedTuple):
    safe_to_stop: bool
    reasons: list
    info: dict


def _build_client(paper: bool):
    env_file = _PROJECT_ROOT / ('.env' if paper else '.env.live')
    env = dotenv_values(env_file)
    key = env.get('ALPACA_API_KEY') or env.get('APCA_API_KEY_ID')
    secret = env.get('ALPACA_SECRET_KEY') or env.get('APCA_API_SECRET_KEY')
    if not key or not secret:
        return None
    return TradingClient(key, secret, paper=paper)


def check_custody_state(client, bot: str, watchdog_active: bool) -> CustodyCheckResult:
    """No open positions/orders for this bot -> safe regardless of watchdog.
    Positions/orders held + watchdog active -> safe (it keeps monitoring
    through the stop). Positions/orders held + watchdog NOT active -> unsafe.
    """
    positions = bot_positions(client.get_all_positions(), bot)
    orders = bot_positions(client.get_orders(), bot)
    info = {
        'position_count': len(positions), 'order_count': len(orders),
        'symbols': sorted({p.symbol for p in positions} | {o.symbol for o in orders}),
        'watchdog_active': watchdog_active,
    }

    if not positions and not orders:
        return CustodyCheckResult(True, ['No open positions or orders for this bot.'], info)
    if watchdog_active:
        return CustodyCheckResult(
            True,
            [f"{len(positions)} position(s) / {len(orders)} order(s) held, but the "
             f"position watchdog ({_WATCHDOG_TIMER}) is active and will keep monitoring "
             f"them while this service is stopped."],
            info,
        )
    return CustodyCheckResult(
        False,
        [f"UNSAFE: {len(positions)} position(s) / {len(orders)} order(s) held "
         f"({', '.join(info['symbols'])}), and the position watchdog is NOT active. "
         f"Stopping now means nothing will monitor this custody. Flatten/protect first, "
         f"or start the watchdog timer, before stopping."],
        info,
    )


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <service-unit>")
        sys.exit(2)

    unit = sys.argv[1]
    match = next((u for u in TRADING_UNITS if u['unit'] == unit), None)
    if match is None:
        print(f"Unknown unit: {unit}. Known units: {', '.join(u['unit'] for u in TRADING_UNITS)}")
        sys.exit(2)

    client = _build_client(paper=match['paper'])
    if client is None:
        account_label = 'paper' if match['paper'] else 'live'
        print(f"No credentials configured for the {account_label} account — cannot check custody.")
        sys.exit(2)

    watchdog_active = service_state(_WATCHDOG_TIMER) == 'active'
    result = check_custody_state(client, match['bot'], watchdog_active)

    print(f"Custody check: {unit} ({'paper' if match['paper'] else 'live'}/{match['bot']})")
    print(f"  Watchdog ({_WATCHDOG_TIMER}) active: {watchdog_active}")
    print(f"  Positions: {result.info['position_count']}  Orders: {result.info['order_count']}")
    if result.info['symbols']:
        print(f"  Symbols: {', '.join(result.info['symbols'])}")
    print(f"  {'SAFE TO STOP' if result.safe_to_stop else 'NOT SAFE TO STOP'}")
    for reason in result.reasons:
        print(f"  - {reason}")

    sys.exit(0 if result.safe_to_stop else 1)


if __name__ == '__main__':
    main()
