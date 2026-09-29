#!/usr/bin/env python3
"""
liquidate_live_to_cash.py — cancel all open orders and close every position
on the LIVE Alpaca account, converting the account fully to cash.

Deliberately narrow and explicit, unlike the existing close_all_positions.py:
  - requires --execute; preview-only by default
  - requires --live explicitly (no ambient PAPER_TRADING fallback -- this
    script only ever touches the live account)
  - reads .env.live specifically, matching the codebase's established
    dual-account convention (position_watchdog.py::_build_clients() etc.) --
    .env alone holds the PAPER account's credentials, not live's
  - uses cancel_orders() then close_all_positions(cancel_orders=True), the
    same proven, fractional-share-safe Alpaca bulk endpoint call already
    used by close_all_positions.py for paper
  - polls and reports final account state rather than assuming success

Usage:
    python scripts/tools/liquidate_live_to_cash.py --live            # preview
    python scripts/tools/liquidate_live_to_cash.py --live --execute  # execute
"""
import argparse
import logging
import sys
import time
from pathlib import Path

from dotenv import dotenv_values

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _pathfix  # noqa: F401

from alpaca.trading.client import TradingClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _build_live_client() -> TradingClient:
    env = dotenv_values(_PROJECT_ROOT / '.env.live')
    key = env.get('ALPACA_API_KEY') or env.get('APCA_API_KEY_ID')
    secret = env.get('ALPACA_SECRET_KEY') or env.get('APCA_API_SECRET_KEY')
    if not key or not secret:
        logging.error("No live credentials found in .env.live — aborting.")
        sys.exit(1)
    return TradingClient(key, secret, paper=False)


def _print_state(client: TradingClient, label: str) -> list:
    account = client.get_account()
    positions = client.get_all_positions()
    orders = client.get_orders()
    logging.info(f"--- {label} ---")
    logging.info(f"Cash:          ${float(account.cash):>12,.2f}")
    logging.info(f"Equity:        ${float(account.equity):>12,.2f}")
    logging.info(f"Buying power:  ${float(account.buying_power):>12,.2f}")
    logging.info(f"Open positions: {len(positions)}")
    for p in positions:
        logging.info(f"  {p.symbol:<8} qty={p.qty:>16} value=${float(p.market_value):>10,.2f}")
    logging.info(f"Open orders: {len(orders)}")
    for o in orders:
        logging.info(f"  {o.symbol:<8} {o.side} qty={o.qty} status={o.status}")
    return positions


def run(execute: bool) -> None:
    client = _build_live_client()
    positions = _print_state(client, "LIVE account — current state")

    if not positions:
        logging.info("No open positions — nothing to liquidate.")
        return

    if not execute:
        logging.info("\nPreview only — pass --execute to submit orders.")
        return

    logging.info("\nCancelling all open orders...")
    client.cancel_orders()
    time.sleep(2)

    logging.info("Closing all positions (Alpaca submits market sell orders for each)...")
    client.close_all_positions(cancel_orders=True)

    logging.info("Close orders submitted. Waiting for fills...")
    for _ in range(6):
        time.sleep(5)
        remaining = client.get_all_positions()
        if not remaining:
            break
        logging.info(f"  {len(remaining)} position(s) still closing...")

    _print_state(client, "LIVE account — after liquidation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', required=True,
                         help='Required explicit confirmation this targets the LIVE account')
    parser.add_argument('--execute', action='store_true',
                         help='Submit orders (default: preview only)')
    args = parser.parse_args()
    run(execute=args.execute)


if __name__ == '__main__':
    main()
