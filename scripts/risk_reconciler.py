#!/usr/bin/env python3
"""
risk_reconciler.py — Detect and repair unprotected positions.

Three responsibilities:
  1. count_unprotected_positions  — returns {'options': int, 'stocks': int}
  2. write_reconcile_status       — writes JSON snapshot for health monitoring
  3. reprotect_positions          — attaches GTC OCO exits to naked stock positions

Broker API limitation (confirmed 2026-06-12):
  Alpaca supports fractional stop/stop-limit orders only with time_in_force=DAY.
  GTC stop orders on fractional quantities are rejected. Fractional positions are
  counted in 'skipped_fractional' but no broker-side order is submitted.

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


def count_unprotected_positions(trading_client) -> dict:
    """Count positions without any open SELL order, split by asset class.

    Returns {'options': int, 'stocks': int}.
    Each value is -1 independently when the broker API is unavailable so callers
    can distinguish "zero unprotected" from "unknown state."
    """
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import OrderSide

        positions   = trading_client.get_all_positions()
        open_orders = trading_client.get_orders(GetOrdersRequest(status='open'))

        protected = {o.symbol for o in open_orders if o.side == OrderSide.SELL}

        options_unprotected = sum(
            1 for p in positions
            if len(p.symbol) > 10 and p.symbol not in protected
        )
        stocks_unprotected = sum(
            1 for p in positions
            if len(p.symbol) <= 10 and p.symbol not in protected
        )
        return {'options': options_unprotected, 'stocks': stocks_unprotected}
    except Exception as e:
        logging.warning("Could not count unprotected positions: %s", e)
        return {'options': -1, 'stocks': -1}


def write_reconcile_status(trading_client, path: Path = _DEFAULT_STATUS_PATH) -> None:
    """Write unprotected position counts and timestamp to *path* as JSON.

    Schema:
      unprotected_options   — options positions without a SELL order (-1 = API error)
      unprotected_stocks    — stock positions without a SELL order (-1 = API error)
      unprotected_positions — deprecated legacy key; -1 if either count is -1, else sum
      checked_at            — ISO timestamp

    Callers should wrap this in try/except so a broker API hiccup never blocks the
    trading session loop.
    """
    counts = count_unprotected_positions(trading_client)
    n_opts = counts['options']
    n_stk  = counts['stocks']

    # Legacy key: -1 if either individual count is unknown; never sum with -1.
    if n_opts == -1 or n_stk == -1:
        legacy = -1
    else:
        legacy = n_opts + n_stk

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        'unprotected_options':   n_opts,
        'unprotected_stocks':    n_stk,
        'unprotected_positions': legacy,
        'checked_at': datetime.now().isoformat(),
    }))

    for label, n in (('options', n_opts), ('stocks', n_stk)):
        if n > 0:
            logging.warning(
                "⚠️  RECONCILE: %d unprotected %s position(s) — broker-side stop missing", n, label
            )
        elif n == 0:
            logging.info("✅ RECONCILE: all %s positions are protected", label)
        else:
            logging.warning("⚠️  RECONCILE: could not determine %s protection status (API error)", label)


def reprotect_positions(trading_client, params: dict, dry_run: bool = False) -> dict:
    """Attach GTC OCO exit orders to naked stock positions.

    For whole-share quantities: submits a GTC OCO SELL (take_profit limit +
    stop_loss stop) for the full position quantity.

    For fractional quantities: GTC stop orders are not supported by Alpaca
    (API limitation confirmed 2026-06-12 — only DAY TIF for fractional
    stop/stop-limit). These positions are counted in 'skipped_fractional' and
    logged; no broker-side order is submitted.

    Returns a summary dict:
      protected          — new OCO orders submitted
      already_protected  — positions that already had a SELL order (skipped)
      skipped_fractional — fractional positions (no GTC stop available)
      errors             — positions where submit_order raised
    """
    from alpaca.trading.requests import (
        GetOrdersRequest, LimitOrderRequest, StopLossRequest, TakeProfitRequest,
    )
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

    summary = {'protected': 0, 'already_protected': 0, 'skipped_fractional': 0, 'errors': 0}

    try:
        positions   = trading_client.get_all_positions()
        open_orders = trading_client.get_orders(GetOrdersRequest(status='open'))
    except Exception as e:
        logging.warning("reprotect_positions: could not fetch positions/orders: %s", e)
        return summary

    protected_syms = {o.symbol for o in open_orders if o.side == OrderSide.SELL}

    stop_loss_pct   = float(params.get('stop_loss',   -0.07))
    take_profit_pct = float(params.get('take_profit',  0.15))

    for pos in positions:
        symbol = pos.symbol
        if len(symbol) > 10:
            continue  # options — handled by options_agent

        if symbol in protected_syms:
            summary['already_protected'] += 1
            continue

        qty           = float(pos.qty)
        avg_entry     = float(pos.avg_entry_price)
        stop_price    = round(avg_entry * (1 + stop_loss_pct),   2)
        target_price  = round(avg_entry * (1 + take_profit_pct), 2)

        is_fractional = (qty != int(qty))
        if is_fractional:
            # Alpaca does not support GTC stop/stop-limit for fractional quantities.
            logging.warning(
                "⚠️  REPROTECT: %s qty=%.4f is fractional — GTC stop unsupported; "
                "relying on agent SELL decisions for protection.", symbol, qty
            )
            summary['skipped_fractional'] += 1
            continue

        if dry_run:
            logging.info(
                "[DRY RUN] Would submit GTC OCO SELL %d %s — stop=$%.2f target=$%.2f",
                int(qty), symbol, stop_price, target_price,
            )
            summary['protected'] += 1
            continue

        try:
            order = LimitOrderRequest(
                symbol=symbol,
                qty=int(qty),
                side=OrderSide.SELL,
                limit_price=target_price,
                time_in_force=TimeInForce.GTC,
                order_class=OrderClass.OCO,
                stop_loss=StopLossRequest(stop_price=stop_price),
            )
            trading_client.submit_order(order)
            logging.info(
                "✅ REPROTECT: submitted GTC OCO SELL %d %s (stop=$%.2f target=$%.2f)",
                int(qty), symbol, stop_price, target_price,
            )
            summary['protected'] += 1
        except Exception as e:
            logging.warning("⚠️  REPROTECT: could not protect %s: %s", symbol, e)
            summary['errors'] += 1

    return summary


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
