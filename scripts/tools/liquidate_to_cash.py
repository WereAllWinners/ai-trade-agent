#!/usr/bin/env python3
"""
liquidate_to_cash.py

Sells positions in descending market-value order until cash is positive.
Dry-run by default — pass --execute to actually submit orders.

Usage:
    python3 scripts/liquidate_to_cash.py           # preview only
    python3 scripts/liquidate_to_cash.py --execute  # submit orders
"""
import os
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _pathfix  # noqa: F401
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / '.env')

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run(execute: bool = False) -> None:
    paper = os.getenv('PAPER_TRADING', 'true').lower() != 'false'
    client = TradingClient(
        os.getenv('ALPACA_API_KEY'),
        os.getenv('ALPACA_SECRET_KEY'),
        paper=paper,
    )

    account = client.get_account()
    cash = float(account.cash)
    logging.info(f"Current cash: ${cash:,.2f}")

    if cash >= 0:
        logging.info("✅ Cash is already positive — nothing to do.")
        return

    cash_needed = abs(cash)
    logging.info(f"Need to raise: ${cash_needed:,.2f}")

    # Get all stock positions (skip options — illiquid, small value)
    positions = client.get_all_positions()
    stock_positions = [
        p for p in positions
        if len(p.symbol) <= 10   # options symbols are much longer
    ]
    stock_positions.sort(key=lambda p: float(p.market_value), reverse=True)

    to_sell = []
    running_total = 0.0
    for p in stock_positions:
        if running_total >= cash_needed:
            break
        to_sell.append(p)
        running_total += float(p.market_value)

    if not to_sell:
        logging.warning("⚠️  No stock positions available to sell.")
        return

    logging.info(f"\n{'Symbol':<10} {'Qty':>6} {'Mkt Value':>12}  {'P&L':>10}")
    logging.info("-" * 44)
    for p in to_sell:
        logging.info(
            f"{p.symbol:<10} {float(p.qty):>6.0f} ${float(p.market_value):>11,.2f}  "
            f"${float(p.unrealized_pl):>+9,.2f}"
        )
    logging.info("-" * 44)
    logging.info(f"{'TOTAL':<17} ${running_total:>11,.2f}  (need ${cash_needed:,.2f})")
    logging.info(f"Cash after selling: ~${cash + running_total:,.2f}")

    if not execute:
        logging.info("\n⚠️  DRY RUN — no orders submitted. Pass --execute to sell.")
        return

    # Cancel all open orders first so shares aren't held and unavailable
    logging.info("\n🗑️  Cancelling all open orders to free held shares...")
    client.cancel_orders()
    import time; time.sleep(3)

    logging.info("🚀 Submitting sell orders...")
    for p in to_sell:
        try:
            qty = int(float(p.qty))
            order = client.submit_order(MarketOrderRequest(
                symbol=p.symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            ))
            logging.info(f"✅ SELL {qty} {p.symbol} — order {order.id}")
        except Exception as e:
            logging.error(f"❌ Failed to sell {p.symbol}: {e}")

    logging.info("\nDone. Check Alpaca dashboard for fill confirmations.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true',
                        help='Actually submit sell orders (default is dry run)')
    args = parser.parse_args()
    run(execute=args.execute)
