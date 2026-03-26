#!/usr/bin/env python3
"""
close_all_positions.py — Cancel all open orders and liquidate all positions
on the paper trading account to clear the negative cash balance.

Usage:
    python scripts/close_all_positions.py
"""
import os
import sys
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

load_dotenv(Path(__file__).resolve().parent.parent / '.env')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PAPER = os.getenv('PAPER_TRADING', 'true').lower() != 'false'

client = TradingClient(
    os.getenv('ALPACA_API_KEY'),
    os.getenv('ALPACA_SECRET_KEY'),
    paper=PAPER,
)

mode = "PAPER" if PAPER else "LIVE"
logging.info(f"Connected to Alpaca ({mode})")

# Show current state
account = client.get_account()
logging.info(f"Equity:        ${float(account.equity):>14,.2f}")
logging.info(f"Cash:          ${float(account.cash):>14,.2f}")
logging.info(f"Buying Power:  ${float(account.buying_power):>14,.2f}")

positions = client.get_all_positions()
logging.info(f"Open positions: {len(positions)}")
for p in positions:
    logging.info(f"  {p.symbol:>6}  qty={p.qty:>8}  market_value=${float(p.market_value):>12,.2f}")

if not positions:
    logging.info("No open positions — nothing to do.")
    sys.exit(0)

confirm = input("\nClose ALL positions and cancel ALL orders? [yes/no]: ").strip().lower()
if confirm != 'yes':
    logging.info("Aborted.")
    sys.exit(0)

# Step 1: cancel all open orders first
logging.info("Cancelling all open orders...")
client.cancel_orders()
time.sleep(2)

# Step 2: close all positions (Alpaca submits market sell orders for each)
logging.info("Closing all positions...")
client.close_all_positions(cancel_orders=True)

logging.info("Close orders submitted. Waiting 10s for fills...")
time.sleep(10)

# Step 3: show resulting account state
account = client.get_account()
positions = client.get_all_positions()
logging.info("--- Account after liquidation ---")
logging.info(f"Equity:        ${float(account.equity):>14,.2f}")
logging.info(f"Cash:          ${float(account.cash):>14,.2f}")
logging.info(f"Buying Power:  ${float(account.buying_power):>14,.2f}")
logging.info(f"Open positions remaining: {len(positions)}")
if positions:
    logging.warning("Some positions may still be closing — check the Alpaca dashboard.")
else:
    logging.info("✅ All positions closed. Cash balance restored.")
