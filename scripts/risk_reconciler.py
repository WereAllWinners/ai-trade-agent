#!/usr/bin/env python3
"""
risk_reconciler.py — Detect and repair unprotected positions.

Three responsibilities:
  1. count_unprotected_positions  — returns {'options': int, 'stocks_whole': int, 'stocks_fractional': int}
  2. write_reconcile_status       — writes JSON snapshot for health monitoring
  3. reprotect_positions          — attaches GTC OCO exits to naked stock positions

Broker API limitation (confirmed 2026-06-12):
  Alpaca supports fractional stop/stop-limit orders only with time_in_force=DAY.
  GTC stop orders on fractional quantities are rejected. Fractional positions are
  counted in 'skipped_fractional' but no broker-side order is submitted.

Usage (standalone):
    python3 scripts/risk_reconciler.py
    python3 scripts/risk_reconciler.py --dry-run   # eyeball report before real run
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

    Returns {'options': int, 'stocks_whole': int, 'stocks_fractional': int}.
    Each value is -1 independently when the broker API is unavailable so callers
    can distinguish "zero unprotected" from "unknown state."
    """
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import OrderSide

        positions   = trading_client.get_all_positions()
        open_orders = trading_client.get_orders(GetOrdersRequest(status='open'))

        protected = {o.symbol for o in open_orders if o.side == OrderSide.SELL}

        options_unprotected = 0
        stocks_whole        = 0
        stocks_fractional   = 0

        for p in positions:
            sym = p.symbol
            qty = float(p.qty)
            if len(sym) > 10:
                if sym not in protected:
                    options_unprotected += 1
            else:
                if sym not in protected:
                    if qty == int(qty) and int(qty) > 0:
                        stocks_whole += 1
                    else:
                        stocks_fractional += 1

        return {
            'options':          options_unprotected,
            'stocks_whole':     stocks_whole,
            'stocks_fractional': stocks_fractional,
        }
    except Exception as e:
        logging.warning("Could not count unprotected positions: %s", e)
        return {'options': -1, 'stocks_whole': -1, 'stocks_fractional': -1}


def write_reconcile_status(trading_client, path: Path = _DEFAULT_STATUS_PATH) -> None:
    """Write unprotected position counts and timestamp to *path* as JSON.

    Schema:
      unprotected_options        — options positions without a SELL order (-1 = API error)
      unprotected_stocks_whole   — whole-share stock positions without a SELL order (-1 = API error)
      unprotected_stocks_fractional — fractional positions (always unprotectable, warn-only)
      unprotected_positions      — deprecated legacy key; -1 if ANY count is -1, else sum of all three
      checked_at                 — ISO timestamp

    Callers should wrap this in try/except so a broker API hiccup never blocks the
    trading session loop.
    """
    counts = count_unprotected_positions(trading_client)
    n_opts  = counts['options']
    n_whole = counts['stocks_whole']
    n_frac  = counts['stocks_fractional']

    # Legacy key: -1 if ANY individual count is unknown; never sum with -1.
    if n_opts == -1 or n_whole == -1 or n_frac == -1:
        legacy = -1
    else:
        legacy = n_opts + n_whole + n_frac

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        'unprotected_options':           n_opts,
        'unprotected_stocks_whole':      n_whole,
        'unprotected_stocks_fractional': n_frac,
        'unprotected_positions':         legacy,  # deprecated: sum of all three
        'checked_at': datetime.now().isoformat(),
    }))

    for label, n in (('options', n_opts), ('stocks_whole', n_whole), ('stocks_fractional', n_frac)):
        if n > 0:
            logging.warning(
                "⚠️  RECONCILE: %d unprotected %s position(s) — broker-side stop missing", n, label
            )
        elif n == 0:
            logging.info("✅ RECONCILE: all %s positions are protected", label)
        else:
            logging.warning("⚠️  RECONCILE: could not determine %s protection status (API error)", label)


def reprotect_positions(
    trading_client,
    params: dict,
    dry_run: bool = False,
    positions_opened_today: 'set | None' = None,
) -> dict:
    """Attach GTC OCO exit orders to naked stock positions.

    For whole-share quantities: submits a GTC OCO SELL (take_profit limit +
    stop_loss stop) for the full position quantity.

    For fractional quantities: GTC stop orders are not supported by Alpaca
    (API limitation confirmed 2026-06-12 — only DAY TIF for fractional
    stop/stop-limit). These positions are counted in 'skipped_fractional' and
    logged; no broker-side order is submitted.

    When params['no_same_day_close'] is True and a symbol appears in
    positions_opened_today, the position is deferred (counted in 'deferred_pdt')
    to avoid same-day round-trip PDT violations.

    When dry_run=True, no orders are submitted but the 'report' list in the
    returned dict is populated with per-position details for human review:
      {symbol, qty, current_price, stop, target, would_fill_immediately: bool}

    Returns:
      protected            — new OCO orders submitted (or would-be in dry_run)
      already_protected    — positions that already had a SELL order (skipped)
      skipped_fractional   — fractional positions (no GTC stop available)
      deferred_pdt         — positions skipped due to same-day PDT deferral
      errors               — positions where submit_order raised
      report               — list of per-position dicts (dry_run=True only)
    """
    from alpaca.trading.requests import (
        GetOrdersRequest, LimitOrderRequest, StopLossRequest, TakeProfitRequest,
    )
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

    summary: dict = {
        'protected': 0, 'already_protected': 0, 'skipped_fractional': 0,
        'deferred_pdt': 0, 'errors': 0, 'report': [],
    }

    try:
        positions   = trading_client.get_all_positions()
        open_orders = trading_client.get_orders(GetOrdersRequest(status='open'))
    except Exception as e:
        logging.warning("reprotect_positions: could not fetch positions/orders: %s", e)
        return summary

    protected_syms = {o.symbol for o in open_orders if o.side == OrderSide.SELL}

    stop_loss_pct   = float(params.get('stop_loss',   -0.07))
    take_profit_pct = float(params.get('take_profit',  0.15))
    no_same_day     = bool(params.get('no_same_day_close', False))

    for pos in positions:
        symbol = pos.symbol
        if len(symbol) > 10:
            continue  # options — handled by options_agent

        if symbol in protected_syms:
            summary['already_protected'] += 1
            continue

        qty          = float(pos.qty)
        avg_entry    = float(pos.avg_entry_price)
        stop_price   = round(avg_entry * (1 + stop_loss_pct),   2)
        target_price = round(avg_entry * (1 + take_profit_pct), 2)

        is_fractional = (qty != int(qty))
        if is_fractional:
            # Alpaca does not support GTC stop/stop-limit for fractional quantities.
            logging.warning(
                "⚠️  REPROTECT: %s qty=%.4f is fractional — GTC stop unsupported; "
                "relying on agent SELL decisions for protection.", symbol, qty
            )
            summary['skipped_fractional'] += 1
            continue

        # PDT deferral: skip positions opened today when same-day close is disallowed.
        if no_same_day and positions_opened_today and symbol in positions_opened_today:
            logging.info(
                "⏭️  REPROTECT: deferring %s (opened today, no_same_day_close active)", symbol
            )
            summary['deferred_pdt'] += 1
            continue

        if dry_run:
            try:
                current_price = float(pos.current_price)
            except Exception:
                current_price = avg_entry
            would_fill = current_price <= stop_price or current_price >= target_price
            summary['report'].append({
                'symbol':               symbol,
                'qty':                  int(qty),
                'current_price':        current_price,
                'stop':                 stop_price,
                'target':               target_price,
                'would_fill_immediately': would_fill,
            })
            logging.info(
                "[DRY RUN] Would submit GTC OCO SELL %d %s — stop=$%.2f target=$%.2f "
                "(current=$%.2f would_fill_immediately=%s)",
                int(qty), symbol, stop_price, target_price, current_price, would_fill,
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
    import argparse
    from alpaca.trading.client import TradingClient

    parser = argparse.ArgumentParser(description='Risk reconciler — count and repair naked positions')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be submitted without placing orders')
    args = parser.parse_args()

    api_key = os.getenv('ALPACA_API_KEY') or os.getenv('APCA_API_KEY_ID', '')
    secret  = os.getenv('ALPACA_SECRET_KEY') or os.getenv('APCA_API_SECRET_KEY', '')
    paper   = os.getenv('PAPER_TRADING', 'true').lower() != 'false'

    if not api_key or not secret:
        print("ALPACA_API_KEY / ALPACA_SECRET_KEY not set in environment")
        sys.exit(1)

    client = TradingClient(api_key, secret, paper=paper)

    if args.dry_run:
        params = {
            'stop_loss':   float(os.getenv('STOP_LOSS_PCT',   '-0.07')),
            'take_profit': float(os.getenv('TAKE_PROFIT_PCT',  '0.15')),
        }
        summary = reprotect_positions(client, params, dry_run=True)
        print(f"\nDry-run report ({len(summary['report'])} positions would be protected):\n")
        for entry in summary['report']:
            flag = ' ⚠️  WOULD FILL' if entry['would_fill_immediately'] else ''
            print(
                f"  {entry['symbol']:12s}  qty={entry['qty']:4d}  "
                f"cur=${entry['current_price']:8.2f}  "
                f"stop=${entry['stop']:8.2f}  target=${entry['target']:8.2f}{flag}"
            )
        print(f"\nSummary: protected={summary['protected']}  "
              f"already_protected={summary['already_protected']}  "
              f"skipped_fractional={summary['skipped_fractional']}  "
              f"deferred_pdt={summary['deferred_pdt']}  "
              f"errors={summary['errors']}")
    else:
        write_reconcile_status(client, _DEFAULT_STATUS_PATH)
        data = json.loads(_DEFAULT_STATUS_PATH.read_text())
        print(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
