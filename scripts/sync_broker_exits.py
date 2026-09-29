#!/usr/bin/env python3
"""
sync_broker_exits.py — Detect and synthesize trade_log rows for broker-side
exit fills that bypassed execute_trade (sprint01 C2).

Problem: OutcomeTracker matches buys to sells inside trade_log.jsonl, but
broker-side exits (Alpaca bracket stop/take-profit children, and GTC OCO
exits submitted by risk_reconciler.reprotect_positions) are filled directly
by Alpaca and never written to the trade log. Bracket/OCO exits — the most
common real outcomes — therefore produce no outcome rows; training_data_builder
falls back to N-day forward price change, mislabeling realized stop-outs that
later bounced as winners.

Empirical findings from a live probe against the paper account (sprint01
C2.1, see docs/sprint01-v0-findings.md for context) that shaped this module:

  - Both order_class='bracket' and order_class='oco' parents retain a
    populated `.legs` list even after the parent itself has closed
    (canceled/filled) — the nested-legs approach works for both, no fuzzy
    symbol/timestamp fallback matching is needed.
  - The fill can land on EITHER side of an OCO/bracket pair: sometimes the
    stop leg fills and the parent shows canceled; sometimes the take-profit
    leg is what the API reports as the "parent" order itself (status=filled)
    while its one nested leg shows canceled. Both the parent AND its legs
    must be checked for status == 'filled'.
  - In this account's history, order_class='bracket' fills are rare/unverified
    (only one ever created, never filled via its children — likely because
    Alpaca does not support GTC brackets on the fractional-share quantities
    this bot trades) while order_class='oco' (from reprotect_positions) is
    the dominant real exit-protection path. Both are handled identically here
    since the matching logic is the same either way.

Usage (standalone):
    python3 scripts/sync_broker_exits.py                        # dry-run preview
    python3 scripts/sync_broker_exits.py --lookback 90           # wider preview
    python3 scripts/sync_broker_exits.py --execute               # historical backfill
                                                                   # — NEVER run this
                                                                   #   from automation
    python3 scripts/sync_broker_exits.py --execute-daemon        # incremental write,
                                                                   #   used by trading_daemon.py
"""
import json
import logging
import os
import sys
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pathfix  # noqa: F401
import db as _db

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_TRADE_LOG_PATH = _PROJECT_ROOT / 'logs' / 'trade_log.jsonl'

# Order classes that can carry broker-side exit legs we need to recover.
_SYNC_ORDER_CLASSES = ('bracket', 'oco')
_PAGE_LIMIT = 500


def _fetch_closed_orders(trading_client, after: datetime) -> list:
    """Fetch all closed orders since *after*, paginating past the per-page cap.

    Alpaca's closed-orders endpoint caps each page (max 500). Loop using
    `until` set to the oldest `submitted_at` seen on the previous page,
    stopping once a page comes back shorter than the page limit (exhausted).
    """
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus

    all_orders: list = []
    seen_ids: set = set()
    until = None
    while True:
        req = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            nested=True,
            limit=_PAGE_LIMIT,
            after=after,
            until=until,
        )
        page = trading_client.get_orders(req)
        if not page:
            break
        new_orders = [o for o in page if str(o.id) not in seen_ids]
        if not new_orders:
            break
        all_orders.extend(new_orders)
        seen_ids.update(str(o.id) for o in new_orders)
        if len(page) < _PAGE_LIMIT:
            break
        oldest = min((o.submitted_at for o in page if o.submitted_at), default=None)
        if oldest is None or (until is not None and oldest >= until):
            break  # can't make progress — stop rather than loop forever
        until = oldest
    return all_orders


def _load_known_order_ids(trade_log_path: Path, db_path: Path = None) -> set:
    """Order ids already present in trade_log.jsonl or the trades DB table.

    Both sources are checked: the JSONL append in sync_broker_exits() is not
    deduped by the filesystem, and a prior sync run's DB insert (`INSERT OR
    IGNORE`, unique on order_id) does not by itself prevent re-appending the
    same row to the JSONL file.
    """
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
        with _db.get_conn(db_path or _db.DB_PATH) as conn:
            rows = conn.execute(
                "SELECT order_id FROM trades WHERE order_id IS NOT NULL"
            ).fetchall()
            known.update(str(r['order_id']) for r in rows)
    except Exception as e:
        logging.warning("sync_broker_exits: could not read known order_ids from DB: %s", e)
    return known


def _candidate_fills(order) -> list:
    """Return [(fill_order, exit_type)] for every actually-filled SELL leg of
    a bracket/OCO structure.

    Checks BOTH the parent order itself and each nested leg — either side of
    an OCO/bracket pair can be the one Alpaca reports as filled (confirmed
    empirically, see module docstring).
    """
    from alpaca.trading.enums import OrderSide

    candidates = []
    for o in [order] + list(order.legs or []):
        if (o.side == OrderSide.SELL and o.status and o.status.value == 'filled'
                and o.filled_qty and float(o.filled_qty) > 0):
            exit_type = o.type.value if o.type else 'unknown'
            candidates.append((o, exit_type))
    return candidates


def _scan_broker_exits(trading_client, known_order_ids: set, lookback_days: int):
    """Walk closed bracket/OCO orders and classify each filled SELL leg as
    new or already-known.

    Returns (new_rows, counts) where counts has 'scanned' (candidate SELL
    fills examined) and 'skipped_existing' (already in the trade log/DB).
    """
    after = datetime.now() - timedelta(days=lookback_days)
    orders = _fetch_closed_orders(trading_client, after)

    new_rows = []
    seen_this_run: set = set()
    scanned = 0
    skipped_existing = 0

    for order in orders:
        oc = order.order_class.value if order.order_class else None
        if oc not in _SYNC_ORDER_CLASSES:
            continue
        for fill_order, exit_type in _candidate_fills(order):
            scanned += 1
            oid = str(fill_order.id)
            if oid in known_order_ids or oid in seen_this_run:
                skipped_existing += 1
                continue
            seen_this_run.add(oid)
            fill_time = fill_order.filled_at or fill_order.submitted_at
            if fill_time and fill_time.tzinfo is not None:
                # Alpaca timestamps are UTC-aware; the rest of trade_log.jsonl uses
                # naive datetime.now().isoformat() (execute_trade). Normalize to a
                # naive UTC timestamp so outcome_tracker can subtract entry/exit
                # times without TypeError: can't subtract offset-naive and aware.
                fill_time = fill_time.astimezone(timezone.utc).replace(tzinfo=None)
            new_rows.append({
                'timestamp':      fill_time.isoformat() if fill_time else datetime.now().isoformat(),
                'symbol':         fill_order.symbol,
                'action':         'sell',
                'shares':         float(fill_order.filled_qty),
                'confidence':     None,
                'reasoning':      f'broker_bracket_exit ({exit_type}, {oc})',
                'order_id':       oid,
                'synthesized_by': 'broker_fill_sync',
            })

    return new_rows, {'scanned': scanned, 'skipped_existing': skipped_existing}


def sync_broker_exits(trading_client, trade_log_path: Path = _DEFAULT_TRADE_LOG_PATH,
                       lookback_days: int = 7, dry_run: bool = False,
                       db_path: Path = None, bot: str = 'stock', source: str = 'paper') -> dict:
    """Find filled broker-side exit orders absent from the trade log and
    synthesize sell rows for them.

    Returns {'scanned': int, 'synthesized': int, 'skipped_existing': int,
             'errors': int, 'report': [...]}   # report populated in dry_run
    """
    summary = {'scanned': 0, 'synthesized': 0, 'skipped_existing': 0, 'errors': 0, 'report': []}

    try:
        known_ids = _load_known_order_ids(trade_log_path, db_path)
        new_rows, counts = _scan_broker_exits(trading_client, known_ids, lookback_days)
    except Exception as e:
        logging.warning("sync_broker_exits: could not scan broker orders: %s", e)
        summary['errors'] += 1
        return summary

    summary['scanned'] = counts['scanned']
    summary['skipped_existing'] = counts['skipped_existing']

    if dry_run:
        summary['report'] = new_rows
        summary['synthesized'] = len(new_rows)
        return summary

    if not new_rows:
        return summary

    trade_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trade_log_path, 'a') as f:
        for row in new_rows:
            try:
                # sprint02 D4.3: same paper/live commingling fix as the agents'
                # own trade_log writes — `source` was previously only passed to
                # the DB insert below, never stamped onto the JSONL row itself.
                row = {**row, 'source': source}
                f.write(json.dumps(row) + '\n')
                f.flush()
                _db.insert_trade(row, bot=bot, source=source, db_path=db_path or _db.DB_PATH)
                summary['synthesized'] += 1
            except Exception as e:
                logging.warning(
                    "sync_broker_exits: failed to write row for order_id=%s: %s",
                    row.get('order_id'), e,
                )
                summary['errors'] += 1

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Sync broker-side bracket/OCO exit fills into the trade log'
    )
    parser.add_argument('--lookback', type=int, default=7,
                         help='Lookback window in days (default 7)')
    parser.add_argument('--execute', action='store_true',
                         help='Historical backfill write. NEVER invoke this from automation — '
                              'preview with the default dry-run first and get explicit sign-off.')
    parser.add_argument('--execute-daemon', action='store_true',
                         help='Incremental daemon-safe write (short lookback, idempotent via '
                              'order_id dedup). Used by trading_daemon.py; kept distinct from '
                              '--execute so backfill vs. ongoing-sync writes are distinguishable '
                              'in logs.')
    args = parser.parse_args()

    if args.execute and args.execute_daemon:
        print("Specify only one of --execute / --execute-daemon")
        sys.exit(1)

    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv('ALPACA_API_KEY') or os.getenv('APCA_API_KEY_ID', '')
    secret  = os.getenv('ALPACA_SECRET_KEY') or os.getenv('APCA_API_SECRET_KEY', '')
    paper   = os.getenv('PAPER_TRADING', 'true').lower() != 'false'

    if not api_key or not secret:
        print("ALPACA_API_KEY / ALPACA_SECRET_KEY not set in environment")
        sys.exit(1)

    from alpaca.trading.client import TradingClient
    client = TradingClient(api_key, secret, paper=paper)

    dry_run = not (args.execute or args.execute_daemon)
    mode = 'backfill-execute' if args.execute else ('daemon-execute' if args.execute_daemon else 'dry-run')
    logging.info("🔄 sync_broker_exits running in %s mode (lookback=%dd)", mode, args.lookback)

    # sprint02 D4.3: `source` must match the client just built above — the
    # function's own default ('paper') would silently mistag every live sync
    # once the JSONL row itself carries a source field.
    summary = sync_broker_exits(
        client, lookback_days=args.lookback, dry_run=dry_run,
        source='paper' if paper else 'live',
    )

    if dry_run:
        print(f"\nDry-run report ({len(summary['report'])} exit(s) would be synthesized):\n")
        for row in summary['report']:
            print(
                f"  {row['symbol']:12s}  shares={row['shares']:<14.4f}  "
                f"order_id={row['order_id']}  {row['reasoning']}"
            )

    print(
        f"\nSummary: scanned={summary['scanned']}  synthesized={summary['synthesized']}  "
        f"skipped_existing={summary['skipped_existing']}  errors={summary['errors']}"
    )


if __name__ == '__main__':
    main()
