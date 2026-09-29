"""
tests/test_sync_broker_exits.py — Broker-side bracket/OCO exit sync (sprint01 C2)

Covers:
  - _candidate_fills: parent-fill vs leg-fill detection (either side of an
    OCO/bracket pair can be the one that actually filled)
  - _scan_broker_exits: dedup against known order_ids, partial fills, schema
  - sync_broker_exits: dry_run vs write, dedup against JSONL + DB, pagination
  - Outcome-tracker integration: a synthesized sell row pairs with a prior
    buy row via OutcomeTracker.match_and_calculate_pnl, producing a realized
    P&L outcome instead of a fallback-labeled one.
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / 'scripts'
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_SCRIPTS_DIR / 'analysis') not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR / 'analysis'))

import db as _db
from sync_broker_exits import (
    sync_broker_exits, _scan_broker_exits, _candidate_fills,
    _load_known_order_ids, _fetch_closed_orders,
)


# ---------------------------------------------------------------------------
# Mock Alpaca order helpers
# ---------------------------------------------------------------------------

def _leg(order_id, side='sell', status='filled', filled_qty=1.0, symbol='AAPL',
         order_type='stop', filled_at=None):
    from alpaca.trading.enums import OrderSide
    o = MagicMock()
    o.id = order_id
    o.side = OrderSide.SELL if side == 'sell' else OrderSide.BUY
    o.status = MagicMock(value=status)
    o.filled_qty = filled_qty
    o.symbol = symbol
    o.type = MagicMock(value=order_type)
    o.filled_at = filled_at or datetime(2026, 6, 1, 15, 0, 0)
    o.submitted_at = filled_at or datetime(2026, 6, 1, 14, 0, 0)
    return o


def _parent(order_id, order_class='oco', legs=None, side='sell', status='canceled',
            filled_qty=0.0, symbol='AAPL', order_type='limit', submitted_at=None,
            filled_at=None):
    from alpaca.trading.enums import OrderSide
    o = MagicMock()
    o.id = order_id
    o.order_class = MagicMock(value=order_class)
    o.legs = legs or []
    o.side = OrderSide.SELL if side == 'sell' else OrderSide.BUY
    o.status = MagicMock(value=status)
    o.filled_qty = filled_qty
    o.symbol = symbol
    o.type = MagicMock(value=order_type)
    o.submitted_at = submitted_at or datetime(2026, 6, 1, 13, 0, 0)
    o.filled_at = filled_at
    return o


def _client(pages) -> MagicMock:
    """pages: list of lists — one list of orders per successive get_orders() call."""
    c = MagicMock()
    c.get_orders.side_effect = list(pages) + [[]]
    return c


@pytest.fixture()
def db_path(tmp_path):
    p = tmp_path / 'test_trading.db'
    _db.init_db(p)
    return p


@pytest.fixture()
def trade_log_path(tmp_path):
    return tmp_path / 'trade_log.jsonl'


# ---------------------------------------------------------------------------
# _candidate_fills — parent-fill vs leg-fill detection
# ---------------------------------------------------------------------------

class TestCandidateFills:
    def test_leg_filled_parent_canceled(self):
        """Stop leg fills, OCO parent (take-profit side) shows canceled — the
        common case (4/5 in the real account probe)."""
        leg = _leg('leg-1', status='filled', filled_qty=35.0)
        parent = _parent('parent-1', legs=[leg], status='canceled', filled_qty=0.0)

        candidates = _candidate_fills(parent)
        assert len(candidates) == 1
        assert candidates[0][0].id == 'leg-1'
        assert candidates[0][1] == 'stop'

    def test_parent_filled_leg_canceled(self):
        """Take-profit side reported AS the parent (status=filled); its one
        leg (the stop) shows canceled — the ABM case from the real probe."""
        leg = _leg('leg-2', status='canceled', filled_qty=0.0)
        parent = _parent('parent-2', legs=[leg], status='filled', filled_qty=176.0,
                          order_type='limit')

        candidates = _candidate_fills(parent)
        assert len(candidates) == 1
        assert candidates[0][0].id == 'parent-2'
        assert candidates[0][1] == 'limit'

    def test_neither_side_filled_returns_empty(self):
        leg = _leg('leg-3', status='canceled', filled_qty=0.0)
        parent = _parent('parent-3', legs=[leg], status='canceled', filled_qty=0.0)

        assert _candidate_fills(parent) == []

    def test_buy_side_never_returned(self):
        """A filled BUY leg/parent must never be treated as an exit candidate."""
        leg = _leg('leg-4', side='buy', status='filled', filled_qty=10.0)
        parent = _parent('parent-4', legs=[leg], side='buy', status='filled', filled_qty=10.0)

        assert _candidate_fills(parent) == []


# ---------------------------------------------------------------------------
# _scan_broker_exits — dedup, schema, order_class filtering
# ---------------------------------------------------------------------------

class TestScanBrokerExits:
    def test_filled_bracket_child_sell_absent_from_log_is_synthesized(self):
        leg = _leg('new-order-1', status='filled', filled_qty=35.0, symbol='PFE')
        parent = _parent('parent-oco-1', order_class='oco', legs=[leg])
        client = _client([[parent]])

        rows, counts = _scan_broker_exits(client, known_order_ids=set(), lookback_days=7)

        assert counts['scanned'] == 1
        assert counts['skipped_existing'] == 0
        assert len(rows) == 1
        row = rows[0]
        assert row['order_id'] == 'new-order-1'
        assert row['symbol'] == 'PFE'
        assert row['action'] == 'sell'
        assert row['shares'] == 35.0
        assert row['synthesized_by'] == 'broker_fill_sync'
        assert 'bracket_stop' not in row
        assert 'bracket_take_profit' not in row

    def test_already_known_order_id_is_skipped(self):
        leg = _leg('already-known-1', status='filled', filled_qty=10.0)
        parent = _parent('parent-oco-2', order_class='oco', legs=[leg])
        client = _client([[parent]])

        rows, counts = _scan_broker_exits(
            client, known_order_ids={'already-known-1'}, lookback_days=7
        )

        assert rows == []
        assert counts['scanned'] == 1
        assert counts['skipped_existing'] == 1

    def test_non_bracket_non_oco_order_class_ignored(self):
        """Agent-submitted plain SELLs (order_class='simple') are already
        logged via execute_trade and must not be re-synthesized."""
        parent = _parent('simple-1', order_class='simple', legs=[], status='filled',
                          filled_qty=5.0)
        client = _client([[parent]])

        rows, counts = _scan_broker_exits(client, known_order_ids=set(), lookback_days=7)

        assert rows == []
        assert counts['scanned'] == 0

    def test_partial_fill_synthesized_with_filled_qty(self):
        """A partial fill (filled_qty < original order qty) is synthesized
        using the actual filled quantity."""
        leg = _leg('partial-1', status='filled', filled_qty=12.5, symbol='AVGO')
        parent = _parent('parent-oco-3', order_class='bracket', legs=[leg])
        client = _client([[parent]])

        rows, _ = _scan_broker_exits(client, known_order_ids=set(), lookback_days=7)

        assert len(rows) == 1
        assert rows[0]['shares'] == 12.5

    def test_bracket_and_oco_both_handled(self):
        leg1 = _leg('bracket-fill-1', status='filled', filled_qty=1.0, symbol='AAPL')
        p1 = _parent('bp-1', order_class='bracket', legs=[leg1])
        leg2 = _leg('oco-fill-1', status='filled', filled_qty=2.0, symbol='MSFT')
        p2 = _parent('op-1', order_class='oco', legs=[leg2])
        client = _client([[p1, p2]])

        rows, counts = _scan_broker_exits(client, known_order_ids=set(), lookback_days=7)

        assert counts['scanned'] == 2
        assert {r['order_id'] for r in rows} == {'bracket-fill-1', 'oco-fill-1'}


# ---------------------------------------------------------------------------
# sync_broker_exits — dry_run, write path, dedup against JSONL + DB
# ---------------------------------------------------------------------------

class TestSyncBrokerExits:
    def test_dry_run_writes_nothing_and_populates_report(self, trade_log_path, db_path):
        leg = _leg('dry-1', status='filled', filled_qty=8.0, symbol='TGT')
        parent = _parent('parent-dry-1', order_class='oco', legs=[leg])
        client = _client([[parent]])

        summary = sync_broker_exits(
            client, trade_log_path=trade_log_path, dry_run=True, db_path=db_path
        )

        assert summary['synthesized'] == 1
        assert len(summary['report']) == 1
        assert not trade_log_path.exists()

    def test_execute_appends_to_jsonl_and_db(self, trade_log_path, db_path):
        leg = _leg('write-1', status='filled', filled_qty=4.0, symbol='JPM')
        parent = _parent('parent-write-1', order_class='oco', legs=[leg])
        client = _client([[parent]])

        summary = sync_broker_exits(
            client, trade_log_path=trade_log_path, dry_run=False, db_path=db_path
        )

        assert summary['synthesized'] == 1
        assert summary['errors'] == 0
        lines = trade_log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row['order_id'] == 'write-1'

        with _db.get_conn(db_path) as conn:
            db_row = conn.execute(
                "SELECT * FROM trades WHERE order_id = ?", ('write-1',)
            ).fetchone()
        assert db_row is not None
        assert db_row['symbol'] == 'JPM'
        assert db_row['action'] == 'sell'

    def test_jsonl_row_carries_source_field(self, trade_log_path, db_path):
        """sprint02 D4.3: the JSONL line itself must carry `source`, not just
        the DB row — outcome_tracker reads the JSONL directly."""
        leg = _leg('write-live-1', status='filled', filled_qty=1.0, symbol='TSLA')
        parent = _parent('parent-write-live-1', order_class='oco', legs=[leg])
        client = _client([[parent]])

        sync_broker_exits(
            client, trade_log_path=trade_log_path, dry_run=False, db_path=db_path,
            source='live',
        )

        row = json.loads(trade_log_path.read_text().strip().splitlines()[0])
        assert row['source'] == 'live'

        with _db.get_conn(db_path) as conn:
            db_row = conn.execute(
                "SELECT source FROM trades WHERE order_id = ?", ('write-live-1',)
            ).fetchone()
        assert db_row['source'] == 'live'

    def test_dedups_against_existing_jsonl_content(self, trade_log_path, db_path):
        trade_log_path.write_text(json.dumps({
            'timestamp': '2026-06-01T00:00:00', 'symbol': 'AAPL', 'action': 'sell',
            'shares': 1.0, 'order_id': 'existing-in-jsonl',
        }) + '\n')

        leg = _leg('existing-in-jsonl', status='filled', filled_qty=1.0)
        parent = _parent('parent-dup-1', order_class='oco', legs=[leg])
        client = _client([[parent]])

        summary = sync_broker_exits(
            client, trade_log_path=trade_log_path, dry_run=False, db_path=db_path
        )

        assert summary['synthesized'] == 0
        assert summary['skipped_existing'] == 1
        # File must still contain exactly the original line — nothing appended.
        assert len(trade_log_path.read_text().strip().splitlines()) == 1

    def test_dedups_against_db_even_if_absent_from_jsonl(self, trade_log_path, db_path):
        """Simulates a prior sync run whose DB insert succeeded (INSERT OR
        IGNORE, unique on order_id) — the JSONL append is not auto-deduped by
        the filesystem, so this must be checked explicitly against the DB too."""
        _db.insert_trade(
            {'timestamp': '2026-06-01T00:00:00', 'symbol': 'AAPL', 'action': 'sell',
             'shares': 1.0, 'order_id': 'existing-in-db-only'},
            bot='stock', source='paper', db_path=db_path,
        )

        leg = _leg('existing-in-db-only', status='filled', filled_qty=1.0)
        parent = _parent('parent-dup-2', order_class='oco', legs=[leg])
        client = _client([[parent]])

        summary = sync_broker_exits(
            client, trade_log_path=trade_log_path, dry_run=False, db_path=db_path
        )

        assert summary['synthesized'] == 0
        assert summary['skipped_existing'] == 1
        assert not trade_log_path.exists()


# ---------------------------------------------------------------------------
# Pagination (sprint01 A3 — required, not optional)
# ---------------------------------------------------------------------------

class TestPagination:
    def test_two_page_response_yields_orders_from_both_pages(self):
        page1 = [
            _parent(f'p1-{i}', order_class='oco',
                    legs=[_leg(f'leg1-{i}', status='filled', filled_qty=1.0)],
                    submitted_at=datetime(2026, 6, 1, 12, 0, 0) + timedelta(minutes=i))
            for i in range(500)
        ]
        page2 = [
            _parent('p2-0', order_class='oco',
                     legs=[_leg('leg2-0', status='filled', filled_qty=2.0)],
                     submitted_at=datetime(2026, 5, 1, 12, 0, 0))
        ]
        client = _client([page1, page2])

        orders = _fetch_closed_orders(client, after=datetime(2026, 1, 1))

        assert len(orders) == 501
        assert client.get_orders.call_count == 2

    def test_single_short_page_does_not_trigger_second_call(self):
        page1 = [
            _parent('only-1', order_class='oco',
                    legs=[_leg('leg-only-1', status='filled', filled_qty=1.0)])
        ]
        client = _client([page1])

        orders = _fetch_closed_orders(client, after=datetime(2026, 1, 1))

        assert len(orders) == 1
        assert client.get_orders.call_count == 1


# ---------------------------------------------------------------------------
# Outcome-tracker integration — synced sell pairs with a prior buy
# ---------------------------------------------------------------------------

class TestOutcomeTrackerIntegration:
    def test_synced_sell_pairs_with_prior_buy_for_realized_pnl(self, tmp_path, db_path):
        """A buy in the trade log + a broker-synthesized sell must produce one
        outcome row with realized P&L, not a fallback forward-price label."""
        from outcome_tracker import OutcomeTracker

        trade_log_path = tmp_path / 'trade_log.jsonl'
        # sprint02 D4.3: a current-day buy row carries a 'source' tag (added
        # by execute_trade); the synced sell below must match it or FIFO
        # pairing intentionally won't cross (source, symbol) buckets — see
        # test_outcome_tracker.py for the legacy (untagged) case, which is
        # the accepted gap D4.4's targeted backfill closes separately.
        buy_row = {
            'timestamp': '2026-06-01T09:30:00', 'symbol': 'PFE', 'action': 'buy',
            'shares': 35.0, 'confidence': 0.8, 'reasoning': 'bullish',
            'order_id': 'buy-order-1', 'bracket_stop': 20.0, 'bracket_take_profit': 28.0,
            'source': 'paper',
        }
        with open(trade_log_path, 'w') as f:
            f.write(json.dumps(buy_row) + '\n')

        # Sync a broker-side stop-loss exit for the same symbol/quantity.
        leg = _leg('sell-order-1', status='filled', filled_qty=35.0, symbol='PFE',
                    filled_at=datetime(2026, 6, 2, 10, 0, 0))
        parent = _parent('parent-int-1', order_class='oco', legs=[leg], symbol='PFE')
        client = _client([[parent]])

        summary = sync_broker_exits(
            client, trade_log_path=trade_log_path, dry_run=False, db_path=db_path,
            source='paper',
        )
        assert summary['synthesized'] == 1

        tracker = OutcomeTracker.__new__(OutcomeTracker)
        tracker.trade_log_path = trade_log_path
        tracker.paper = True

        fill_prices = {'buy-order-1': 20.50, 'sell-order-1': 24.55}
        with patch.object(tracker, 'get_fill_price',
                           side_effect=lambda order_id: fill_prices.get(order_id)), \
             patch('outcome_tracker._fetch_spy_return', return_value=None):
            trades = tracker.load_raw_trades()
            outcomes = tracker.match_and_calculate_pnl(trades)

        assert len(outcomes) == 1
        outcome = outcomes[0]
        assert outcome['buy_order_id'] == 'buy-order-1'
        assert outcome['sell_order_id'] == 'sell-order-1'
        assert outcome['entry_price'] == 20.50
        assert outcome['exit_price'] == 24.55
        assert outcome['realized_pnl'] == round((24.55 - 20.50) * 35.0, 2)
        assert outcome['won'] is True
