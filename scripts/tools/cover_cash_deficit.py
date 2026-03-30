#!/usr/bin/env python3
"""
cover_cash_deficit.py — Sell the minimum number of positions needed to
bring cash back to a positive balance (plus a small buffer).

Usage:
    python scripts/cover_cash_deficit.py          # preview only
    python scripts/cover_cash_deficit.py --execute
"""
import os
import sys
import time
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / '.env')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

PAPER = os.getenv('PAPER_TRADING', 'true').lower() != 'false'
client = TradingClient(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_SECRET_KEY'), paper=PAPER)

account = client.get_account()
cash = float(account.cash)
equity = float(account.equity)

logging.info(f"Equity:       ${equity:>12,.2f}")
logging.info(f"Cash:         ${cash:>12,.2f}")

if cash >= 0:
    logging.info("✅ Cash is already positive — nothing to do.")
    sys.exit(0)

# Target: get to +$500 buffer above zero
target_cash = 500.0
deficit = abs(cash) + target_cash
logging.info(f"Need to raise: ${deficit:,.2f} to reach +${target_cash:.0f} buffer\n")

# Get all positions sorted smallest market value first
positions = client.get_all_positions()
positions.sort(key=lambda p: float(p.market_value))

logging.info(f"{'Symbol':<8} {'Qty':>6} {'Market Value':>14} {'Unrealized P&L':>16}")
logging.info("-" * 50)
for p in positions:
    logging.info(f"{p.symbol:<8} {float(p.qty):>6.0f} ${float(p.market_value):>13,.2f}  ${float(p.unrealized_pl):>+13,.2f}")

# Pick smallest positions until we've covered the deficit
to_sell = []
covered = 0.0
for p in positions:
    if covered >= deficit:
        break
    to_sell.append(p)
    covered += float(p.market_value)

logging.info(f"\nPlan: sell {len(to_sell)} position(s) raising ~${covered:,.2f}:")
for p in to_sell:
    logging.info(f"  SELL {int(float(p.qty))} shares of {p.symbol} (~${float(p.market_value):,.2f})")

if '--execute' not in sys.argv:
    logging.info("\nDry run — add --execute to sell these positions.")
    sys.exit(0)

confirm = input("\nExecute these sells? [yes/no]: ").strip().lower()
if confirm != 'yes':
    logging.info("Aborted.")
    sys.exit(0)

for p in to_sell:
    qty = int(float(p.qty))
    try:
        order = MarketOrderRequest(
            symbol=p.symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
        )
        client.submit_order(order)
        logging.info(f"✅ Sell order submitted: {qty} shares of {p.symbol}")
    except Exception as e:
        logging.error(f"❌ Failed to sell {p.symbol}: {e}")

logging.info("\nWaiting 5s for fills...")
time.sleep(5)

account = client.get_account()
logging.info(f"Cash after sells: ${float(account.cash):,.2f}")
