#!/usr/bin/env python3
"""
exit_reconciler.py — R1 WI-4/WI-5: fill-state verification.

`manage_existing_positions()` (options) and `execute_trade()`'s SELL branch
(stock) both used to log an exit as complete the instant they submitted the
order, without ever confirming the broker actually filled it — the defect
that fabricated the 2026-07 incident's two false "closed at a profit"
records (NVDA +71.9%, AAPL +179.7%) for positions that in reality rode to
expiry and were exercised. See findings/paper-negcash-addendum-a-2026-07-16.md.

Both agents now record a `pending_exits` row at submit time instead of
writing the trade log immediately. This module is the sole authority that
later resolves each row against the broker's real terminal order status,
and is what actually writes the trade_log/DB/alert records — using
confirmed fill data, not the pre-submission position snapshot the old code
used to trust. One implementation, parameterized by `bot`, so the identical
fix applies to both agents rather than being hand-duplicated.

`reconcile_broker_fills()` (WI-5, options-only) is a separate, complementary
sweep: it catches fills that never went through our own order-submission
code at all — the exact gap that made the MSFT protective-stop fill
invisible (options_agent.py's GTC-stop submission discards the order's own
id, so there was never anything to look up even in principle). Stock
already has an equivalent (`sync_broker_exits.py`, for bracket/OCO legs);
this is options' version, generalized to standalone orders too since the
options protective stop is never wrapped in a bracket/OCO structure.

Neither function writes to `trades`/`outcomes` for the options side, and
neither restructures those tables — the one exception is `bot='stock'`
resolving a fill, which calls `_db.insert_trade()`. That is a *move*, not
a new write: `execute_trade()`'s SELL branch used to call it at submit
time; it now fires here, at resolve time, with confirmed fill data.
Stock's `trades` table has rows today and populates correctly — this
design must not silently stop feeding it.
"""
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import db as _db
from alerts import alert_trade_executed, alert_exit_failed, alert_broker_fill_unaccounted

_TRADE_LOG_PATHS = {
    'options': Path('logs/options_trade_log.jsonl'),
    'stock':   Path('logs/trade_log.jsonl'),
}
_MAX_RECONCILE_RETRIES = 10  # mirrors unreconciled_orders' existing UNRECONCILED_MAX_RETRIES convention

# Orders in these statuses have nothing more to give — no fill happened.
_FAILED_STATUSES = {'canceled', 'expired', 'rejected'}


def _dollar_multiplier(symbol: str) -> int:
    """100 for an OCC option contract (100 shares/contract), 1 for stock —
    same length-based idiom used throughout the codebase."""
    return 100 if len(symbol) > 10 else 1


def _write_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(record, default=str) + '\n')


def _compute_pnl(entry_price, exit_price, qty, symbol):
    if not entry_price or exit_price is None:
        return None, None
    mult = _dollar_multiplier(symbol)
    pnl = (exit_price - entry_price) * qty * mult
    pnl_pct = (exit_price - entry_price) / entry_price
    return pnl, pnl_pct


def _resolve_filled(row: dict, bot: str, source: str, filled_qty: float,
                     avg_price: float, db_path: Path) -> None:
    """Write the real trade_log/DB/alert records for a confirmed fill.

    Per the one-rule policy: a real, broker-confirmed fill always carries
    `action: 'sell'` with real numbers — a truthful record that, once R2
    unfreezes the outcome pipeline, is correctly positioned to be picked up
    as a real outcome.
    """
    entry_price = row['avg_entry_price']
    pnl, pnl_pct = _compute_pnl(entry_price, avg_price, filled_qty, row['symbol'])

    trade_log = {
        'timestamp': datetime.now().isoformat(),
        ('contract' if bot == 'options' else 'symbol'): row['symbol'],
        'action': 'sell',
        'quantity': filled_qty,
        'reason': row['intended_reason'],
        'exit_pl_pct': pnl_pct,
        'order_id': row['order_id'],
        'source': source,
    }
    _write_jsonl(_TRADE_LOG_PATHS[bot], trade_log)

    agent_name = 'OptionsAgent' if bot == 'options' else 'StockAgent'
    alert_trade_executed(
        agent_name, row['symbol'], 'sell', filled_qty, avg_price, row['order_id'],
        pnl=pnl, pnl_pct=pnl_pct, avg_entry_price=entry_price,
    )

    if bot == 'stock':
        # Move, not a new write — see module docstring.
        try:
            _db.insert_trade(
                {'symbol': row['symbol'], 'action': 'sell', 'shares': filled_qty,
                 'order_id': row['order_id'], 'reasoning': row['intended_reason']},
                bot='stock', source=source, db_path=db_path,
            )
        except Exception as e:
            logging.warning(f"⚠️  Could not write resolved stock exit to DB: {e}")

    _db.resolve_pending_exit(row['order_id'], 'filled', db_path=db_path)
    logging.info(
        f"✅ Exit confirmed filled: {row['symbol']} qty={filled_qty} "
        f"@ ${avg_price} ({row['intended_reason']})"
    )


def _write_exit_failed_record(row: dict, bot: str, source: str, broker_status: str,
                               filled_qty: float, unfilled_qty: float) -> None:
    """JSONL + alert only — no DB resolve here, so this can be called
    standalone for a partial-fill remainder without re-resolving a
    pending_exits row that _resolve_filled already resolved. No `action`
    key: nothing was filled for this portion, so there's nothing true to
    claim — this is what makes the FIFO buy/sell matcher ignore the row by
    construction."""
    trade_log = {
        'timestamp': datetime.now().isoformat(),
        ('contract' if bot == 'options' else 'symbol'): row['symbol'],
        'reason': row['intended_reason'],
        'order_id': row['order_id'],
        'broker_status': broker_status,
        'unfilled_qty': unfilled_qty,
        'source': source,
        'event': 'exit_failed',
    }
    _write_jsonl(_TRADE_LOG_PATHS[bot], trade_log)

    agent_name = 'OptionsAgent' if bot == 'options' else 'StockAgent'
    alert_exit_failed(agent_name, row['symbol'], row['order_id'], broker_status,
                       row['intended_reason'], filled_qty=filled_qty)
    logging.warning(
        f"🔴 Exit FAILED to fill: {row['symbol']} status={broker_status} "
        f"unfilled_qty={unfilled_qty} ({row['intended_reason']})"
    )


def _resolve_exit_failed(row: dict, bot: str, source: str, broker_status: str,
                          db_path: Path) -> None:
    """Pure zero-fill failure — the exact NVDA/AAPL incident shape: an exit
    order submitted, never filled, and (before this fix) silently logged as
    a completed win anyway."""
    _write_exit_failed_record(row, bot, source, broker_status, 0.0, row['intended_qty'])
    _db.resolve_pending_exit(row['order_id'], 'exit_failed', db_path=db_path)


def reconcile_pending_exits(trading_client, bot: str, source: str, db_path: Path = None) -> dict:
    """Poll every 'pending' row for (bot, source) via get_order_by_id and
    resolve it against the broker's real terminal status. Returns
    {'checked','filled','exit_failed','still_pending','errors'}."""
    db_path = db_path or _db.DB_PATH
    rows = _db.get_pending_exits(bot, source, status='pending', db_path=db_path)
    stats = {'checked': 0, 'filled': 0, 'exit_failed': 0, 'still_pending': 0, 'errors': 0}

    for row in rows:
        stats['checked'] += 1
        try:
            order = trading_client.get_order_by_id(row['order_id'])
        except Exception as e:
            logging.warning(f"⚠️  Could not check pending exit {row['order_id']} ({row['symbol']}): {e}")
            retry_count = _db.bump_pending_exit_retry(row['order_id'], db_path=db_path)
            if retry_count > _MAX_RECONCILE_RETRIES:
                _db.mark_pending_exit_abandoned(row['order_id'], db_path=db_path)
                logging.error(
                    f"❌ Abandoning pending exit {row['order_id']} ({row['symbol']}) "
                    f"after {retry_count} failed status checks"
                )
            stats['errors'] += 1
            continue

        status = order.status.value if hasattr(order.status, 'value') else str(order.status)
        filled_qty = float(order.filled_qty) if order.filled_qty else 0.0
        avg_price = float(order.filled_avg_price) if order.filled_avg_price else None

        if status == 'filled' and filled_qty > 0:
            _resolve_filled(row, bot, source, filled_qty, avg_price, db_path)
            stats['filled'] += 1
        elif status in _FAILED_STATUSES:
            if filled_qty > 0 and avg_price is not None:
                # Partial-fill convention: one row, one status ('filled', via
                # _resolve_filled below) using the ACTUAL filled_qty; a
                # second, DB-resolve-free JSONL write records the unfilled
                # remainder as exit_failed. The still-open remainder isn't
                # separately tracked — it's simply an open broker position
                # again, and the next session's normal P&L/DTE sweep picks
                # it up fresh, same self-heal as a full failure.
                _resolve_filled(row, bot, source, filled_qty, avg_price, db_path)
                remainder = row['intended_qty'] - filled_qty
                if remainder > 1e-9:
                    _write_exit_failed_record(row, bot, source, status, filled_qty, remainder)
                stats['filled'] += 1
            else:
                _resolve_exit_failed(row, bot, source, status, db_path)
                stats['exit_failed'] += 1
        else:
            # Still resting (new/accepted/partially_filled-and-still-open/
            # etc.) — not terminal yet. Leave pending for the next pass;
            # bump the retry counter so an order that never resolves can
            # eventually be abandoned rather than polled forever.
            retry_count = _db.bump_pending_exit_retry(row['order_id'], db_path=db_path)
            if retry_count > _MAX_RECONCILE_RETRIES:
                _db.mark_pending_exit_abandoned(row['order_id'], db_path=db_path)
                logging.warning(
                    f"⚠️  Pending exit {row['order_id']} ({row['symbol']}) never reached a "
                    f"terminal status after {retry_count} checks — abandoning (still open at broker)."
                )
            stats['still_pending'] += 1

    return stats


# ---------------------------------------------------------------------------
# WI-5: broker-side fills with no app record at all
# ---------------------------------------------------------------------------

def _load_known_option_order_ids(trade_log_path: Path, db_path: Path) -> set:
    """Union of every order_id already in options_trade_log.jsonl (buy or
    sell, any row shape) and every order_id ever recorded in pending_exits
    (any status) — dedups against both normal app-submitted exits and
    whatever reconcile_pending_exits() just resolved in this same pass."""
    known: set = set()
    if trade_log_path.exists():
        with open(trade_log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                oid = rec.get('order_id')
                if oid:
                    known.add(str(oid))
    try:
        with _db.get_conn(db_path) as conn:
            rows = conn.execute("SELECT order_id FROM pending_exits").fetchall()
            known.update(str(r['order_id']) for r in rows)
    except Exception as e:
        logging.warning(f"exit_reconciler: could not read pending_exits order_ids: {e}")
    return known


def _lookup_entry_price(symbol: str, trade_log_path: Path) -> float | None:
    """Best-effort entry-price lookup for P&L enrichment on an orphan fill
    — same FIFO-ish technique options_outcome_tracker.py already uses
    elsewhere. Returns the most recent BUY row's entry_price for this
    contract, or None if genuinely absent (the row is still written, just
    without a P&L figure, rather than dropping the fact of the fill)."""
    if not trade_log_path.exists():
        return None
    last_buy_price = None
    with open(trade_log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get('contract') == symbol and rec.get('action') == 'buy':
                last_buy_price = rec.get('entry_price')
    return float(last_buy_price) if last_buy_price is not None else None


def reconcile_broker_fills(trading_client, source: str, db_path: Path = None,
                            trade_log_path: Path = None, lookback_days: int = 7) -> dict:
    """Options-only (stock has sync_broker_exits.py for its bracket/OCO
    fills; options' protective stop is never wrapped in a bracket/OCO
    structure, so it needs its own, more general sweep). Scans recently-
    closed SELL orders on the account; anything with a real fill and no
    corresponding app record gets a 'source_tag': 'broker_reconcile' JSONL
    row (action: 'sell', real numbers — per the one-rule policy, this is a
    genuine fill) plus a CRITICAL alert. Returns {'scanned', 'orphans_found'}.
    """
    db_path = db_path or _db.DB_PATH
    trade_log_path = trade_log_path or _TRADE_LOG_PATHS['options']

    from sync_broker_exits import _fetch_closed_orders

    after = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    orders = _fetch_closed_orders(trading_client, after)
    known_ids = _load_known_option_order_ids(trade_log_path, db_path)

    stats = {'scanned': 0, 'orphans_found': 0}
    for order in orders:
        if len(order.symbol) <= 10:
            continue  # stock — sync_broker_exits.py already covers this
        side = order.side.value if hasattr(order.side, 'value') else str(order.side)
        if side != 'sell':
            continue
        if not order.filled_qty or float(order.filled_qty) <= 0:
            continue

        stats['scanned'] += 1
        order_id = str(order.id)
        if order_id in known_ids:
            continue

        filled_qty = float(order.filled_qty)
        avg_price = float(order.filled_avg_price) if order.filled_avg_price else 0.0
        entry_price = _lookup_entry_price(order.symbol, trade_log_path)
        pnl, pnl_pct = _compute_pnl(entry_price, avg_price, filled_qty, order.symbol)

        trade_log = {
            'timestamp': datetime.now().isoformat(),
            'contract': order.symbol,
            'action': 'sell',
            'quantity': filled_qty,
            'reason': 'broker_fill_unaccounted',
            'exit_pl_pct': pnl_pct,
            'order_id': order_id,
            'source': source,
            'source_tag': 'broker_reconcile',
        }
        _write_jsonl(trade_log_path, trade_log)
        alert_trade_executed(
            'OptionsAgent', order.symbol, 'sell', filled_qty, avg_price, order_id,
            pnl=pnl, pnl_pct=pnl_pct, avg_entry_price=entry_price,
        )
        alert_broker_fill_unaccounted(order.symbol, order_id, filled_qty, avg_price)
        logging.warning(
            f"🔴 Broker fill with NO app record found: {order.symbol} qty={filled_qty} "
            f"@ ${avg_price} — reconciled as {order_id}."
        )
        stats['orphans_found'] += 1
        known_ids.add(order_id)  # avoid double-counting within this same pass

    return stats
