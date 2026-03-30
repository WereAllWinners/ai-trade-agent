#!/usr/bin/env python3
"""
fix_pdt.py — Show and optionally update Alpaca paper account configurations
to clear the PDT (Pattern Day Trader) day-trading buying power restriction.

Usage:
    python scripts/fix_pdt.py          # show current config
    python scripts/fix_pdt.py --fix    # apply fix
"""
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / '.env')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from alpaca.trading.client import TradingClient

PAPER = os.getenv('PAPER_TRADING', 'true').lower() != 'false'
client = TradingClient(
    os.getenv('ALPACA_API_KEY'),
    os.getenv('ALPACA_SECRET_KEY'),
    paper=PAPER,
)

account = client.get_account()
logging.info(f"Equity:                  ${float(account.equity):>14,.2f}")
logging.info(f"Cash:                    ${float(account.cash):>14,.2f}")
logging.info(f"Buying Power:            ${float(account.buying_power):>14,.2f}")
logging.info(f"Day Trading BP:          ${float(account.daytrading_buying_power or 0):>14,.2f}")
logging.info(f"Day Trade Count:         {account.daytrade_count}")
logging.info(f"Pattern Day Trader flag: {account.pattern_day_trader}")

config = client.get_account_configurations()
logging.info(f"\nCurrent account config:")
logging.info(f"  dtbp_check:            {config.dtbp_check}")
logging.info(f"  pdt_check:             {config.pdt_check}")
logging.info(f"  max_margin_multiplier: {config.max_margin_multiplier}")

if account.pattern_day_trader:
    logging.warning("\n⚠️  Account is flagged as Pattern Day Trader (PDT).")
    logging.warning("   This zeros out daytrading_buying_power in Alpaca paper trading.")
    logging.warning("   Fix: reset the paper account at app.alpaca.markets →")
    logging.warning("   Paper Trading → Account Settings → Reset Account")
else:
    logging.info("\n✅ No PDT flag — account should be able to trade normally.")

if '--fix' not in sys.argv:
    sys.exit(0)

# Attempt to update dtbp_check via direct REST call as a fallback
import requests as req

base = "https://paper-api.alpaca.markets" if PAPER else "https://api.alpaca.markets"
headers = {
    "APCA-API-KEY-ID": os.getenv('ALPACA_API_KEY'),
    "APCA-API-SECRET-KEY": os.getenv('ALPACA_SECRET_KEY'),
}
resp = req.patch(f"{base}/v2/account/configurations", json={"dtbp_check": "entry"}, headers=headers)
if resp.ok:
    data = resp.json()
    logging.info(f"\n✅ dtbp_check updated → {data.get('dtbp_check')}")
else:
    logging.error(f"❌ Failed to update config: {resp.status_code} {resp.text}")
    logging.warning("   If PDT flag is set, the only fix is resetting the paper account.")
