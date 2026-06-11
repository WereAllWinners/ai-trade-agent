#!/usr/bin/env python3
"""
risk_reconciler.py — Count unprotected options positions and write reconcile status.

An "unprotected" options position is an OCC contract (symbol length > 10) that has
no open SELL order of any kind (stop-limit or limit).  Called from the options
agent's session loop after the position-management sweep.

Usage (standalone):
    python3 scripts/risk_reconciler.py
"""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pathfix  # noqa: F401

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_STATUS_PATH = _PROJECT_ROOT / 'logs' / 'reconcile_status.json'


def count_unprotected_positions(trading_client) -> int:
    """Return the number of options positions (OCC symbol > 10 chars) with no open SELL order.

    Returns -1 when the broker API is unavailable so callers can distinguish
    "zero unprotected" from "unknown state".
    """
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import OrderSide

        positions   = trading_client.get_all_positions()
        open_orders = trading_client.get_orders(GetOrdersRequest(status='open'))

        # Any open SELL on the symbol counts as protection (stop-limit or limit).
        protected = {o.symbol for o in open_orders if o.side == OrderSide.SELL}

        return sum(
            1 for p in positions
            if len(p.symbol) > 10 and p.symbol not in protected
        )
    except Exception as e:
        logging.warning("Could not count unprotected positions: %s", e)
        return -1


def write_reconcile_status(trading_client, path: Path = _DEFAULT_STATUS_PATH) -> None:
    """Write unprotected position count and timestamp to *path* as JSON.

    Callers should wrap this in try/except so a broker API hiccup never
    blocks the trading session loop.
    """
    n = count_unprotected_positions(trading_client)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        'unprotected_positions': n,
        'checked_at': datetime.now().isoformat(),
    }))
    if n > 0:
        logging.warning(
            "⚠️  RECONCILE: %d unprotected options position(s) detected — "
            "broker-side stop missing", n
        )
    elif n == 0:
        logging.info("✅ RECONCILE: all options positions are protected")
    else:
        logging.warning("⚠️  RECONCILE: could not determine protection status (API error)")


# ---------------------------------------------------------------------------
# CLI entry point — useful for ad-hoc checks without starting the daemon
# ---------------------------------------------------------------------------

def main():
    import os
    from alpaca.trading.client import TradingClient

    api_key = os.getenv('ALPACA_API_KEY') or os.getenv('APCA_API_KEY_ID', '')
    secret  = os.getenv('ALPACA_SECRET_KEY') or os.getenv('APCA_API_SECRET_KEY', '')
    paper   = os.getenv('PAPER_TRADING', 'true').lower() != 'false'

    if not api_key or not secret:
        print("ALPACA_API_KEY / ALPACA_SECRET_KEY not set in environment")
        sys.exit(1)

    client = TradingClient(api_key, secret, paper=paper)
    write_reconcile_status(client, _DEFAULT_STATUS_PATH)
    data = json.loads(_DEFAULT_STATUS_PATH.read_text())
    print(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
