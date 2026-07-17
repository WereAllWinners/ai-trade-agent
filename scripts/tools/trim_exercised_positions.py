#!/usr/bin/env python3
"""
trim_exercised_positions.py — WI-1 Variant B: sell only the specified stock
lots (default: the auto-exercised AAPL/NVDA/MSFT positions) needed to cover
a cash shortfall, falling back to the smallest additional set of other stock
positions if the named lots aren't enough. Preview by default; --execute to
actually submit orders.

Fixes three bugs found in the existing liquidation scripts during R1
planning: liquidate_to_cash.py/cover_cash_deficit.py both truncate fractional
share quantities via int(float(qty)), silently selling less than the cash
math assumed; cover_cash_deficit.py never cancels open orders first, risking
failure on broker-locked shares; both scripts decide paper/live implicitly
from .env's ambient PAPER_TRADING, with no CLI override, which is risky for a
tool that moves real money.

Usage:
    python scripts/tools/trim_exercised_positions.py --paper                      # preview
    python scripts/tools/trim_exercised_positions.py --paper --execute            # execute
    python scripts/tools/trim_exercised_positions.py --paper --symbols AAPL,NVDA --target-buffer 500
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
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_SYMBOLS = ['AAPL', 'NVDA', 'MSFT']


def _build_client(paper: bool) -> TradingClient:
    env_file = _PROJECT_ROOT / ('.env' if paper else '.env.live')
    env = dotenv_values(env_file)
    key = env.get('ALPACA_API_KEY') or env.get('APCA_API_KEY_ID')
    secret = env.get('ALPACA_SECRET_KEY') or env.get('APCA_API_SECRET_KEY')
    if not key or not secret:
        logging.error(f"Credentials not found in {env_file}")
        sys.exit(1)
    return TradingClient(key, secret, paper=paper)


def compute_sell_plan(positions: list, named_symbols: list, shortfall: float) -> list:
    """Return [(position, qty_to_sell, proceeds), ...] covering `shortfall`.

    Tier 1: named_symbols, sorted by market value descending (minimizes the
    number of symbols touched among the ones this brief specifically
    implicates). Tier 2, only if tier 1 doesn't cover it: remaining
    positions, sorted ascending (cover_cash_deficit.py's smallest-set
    semantics, preserving large "core" positions). The last position needed
    in either tier is partial-sold for the exact remaining amount — no
    truncation, exact float qty.
    """
    if shortfall <= 0:
        return []

    by_symbol = {p.symbol: p for p in positions}
    named = [by_symbol[s] for s in named_symbols if s in by_symbol]
    named.sort(key=lambda p: float(p.market_value), reverse=True)
    remaining_syms = set(by_symbol) - {p.symbol for p in named}
    fallback = sorted((by_symbol[s] for s in remaining_syms), key=lambda p: float(p.market_value))

    plan = []
    covered = 0.0
    _EPS = 1e-6  # float rounding from `need / price` must not leave a phantom remainder
    for p in named + fallback:
        if shortfall - covered <= _EPS:
            break
        mkt_val = float(p.market_value)
        qty = float(p.qty)
        price = mkt_val / qty if qty else 0.0
        need = shortfall - covered
        if mkt_val <= need:
            sell_qty, proceeds = qty, mkt_val
        else:
            sell_qty = round(need / price, 9)  # Alpaca's max fractional precision
            proceeds = sell_qty * price
        plan.append((p, sell_qty, proceeds))
        covered += proceeds
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbols', default=','.join(_DEFAULT_SYMBOLS),
                         help=f'Comma-separated symbols to trim first (default: {",".join(_DEFAULT_SYMBOLS)})')
    parser.add_argument('--target-buffer', type=float, default=0.0,
                         help='Target cash to reach, above $0 (default: 0.0, no hidden buffer)')
    parser.add_argument('--execute', action='store_true', help='Submit sell orders (default: preview only)')
    acct_group = parser.add_mutually_exclusive_group(required=True)
    acct_group.add_argument('--paper', action='store_true', help='Act on the paper account')
    acct_group.add_argument('--live', action='store_true', help='Act on the live account')
    args = parser.parse_args()

    client = _build_client(paper=args.paper)
    named_symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]

    account = client.get_account()
    cash = float(account.cash)
    equity = float(account.equity)
    logging.info(f"Mode:         {'PAPER' if args.paper else 'LIVE'}")
    logging.info(f"Equity:       ${equity:>14,.2f}")
    logging.info(f"Cash:         ${cash:>14,.2f}")
    logging.info(f"Target:       ${args.target_buffer:>14,.2f}")

    shortfall = max(0.0, args.target_buffer - cash)
    if shortfall <= 0:
        logging.info("Cash already meets target — nothing to do.")
        sys.exit(0)
    logging.info(f"Shortfall:    ${shortfall:>14,.2f}\n")

    positions = client.get_all_positions()
    plan = compute_sell_plan(positions, named_symbols, shortfall)

    if not plan:
        logging.error("No positions available to cover the shortfall.")
        sys.exit(1)

    logging.info(f"{'Symbol':<8} {'Sell Qty':>14} {'Held Qty':>14} {'Proceeds':>14}")
    logging.info("-" * 54)
    total_proceeds = 0.0
    for p, qty, proceeds in plan:
        full = qty >= float(p.qty) - 1e-9
        logging.info(f"{p.symbol:<8} {qty:>14.6f} {float(p.qty):>14.6f} ${proceeds:>13,.2f}{'  (full)' if full else '  (partial)'}")
        total_proceeds += proceeds
    logging.info("-" * 54)
    logging.info(f"Projected proceeds: ${total_proceeds:,.2f}")
    logging.info(f"Projected post-sale cash: ${cash + total_proceeds:,.2f}")

    if not args.execute:
        logging.info("\nPreview only — pass --execute to submit orders.")
        sys.exit(0)

    logging.info("\nCancelling all open orders first...")
    client.cancel_orders()
    time.sleep(2)

    submitted = []
    for p, qty, _ in plan:
        order = MarketOrderRequest(
            symbol=p.symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
        )
        try:
            result = client.submit_order(order)
            submitted.append(result)
            logging.info(f"Submitted SELL {qty:.6f} {p.symbol} (order_id={result.id})")
        except Exception as e:
            logging.error(f"Failed to submit SELL {p.symbol}: {e}")

    logging.info("Waiting for fills...")
    time.sleep(10)
    for order in submitted:
        status = client.get_order_by_id(str(order.id))
        logging.info(
            f"  {status.symbol}: status={status.status.value} "
            f"filled_qty={status.filled_qty} avg_price={status.filled_avg_price}"
        )

    account = client.get_account()
    logging.info("\n--- Account after trim ---")
    logging.info(f"Cash:         ${float(account.cash):>14,.2f}")
    logging.info(f"Equity:       ${float(account.equity):>14,.2f}")


if __name__ == '__main__':
    main()
