"""
tests/test_pending_exits.py — R1 WI-4 db.py CRUD for the pending_exits table

All tests run offline against a temp SQLite file (no live API calls).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))


@pytest.fixture()
def db_path(tmp_path):
    import db as _db
    p = tmp_path / 'test_trading.db'
    _db.init_db(p)
    return p


def _rec(order_id='order-1', symbol='NVDA260710C00200000', qty=1.0):
    return {
        'order_id': order_id, 'symbol': symbol, 'intended_qty': qty,
        'intended_reason': 'take_profit_50%', 'avg_entry_price': 2.52,
        'time_in_force': 'GTC', 'submitted_at': '2026-07-17T10:00:00',
    }


class TestInsertAndGet:
    def test_insert_then_get_pending(self, db_path):
        import db as _db
        _db.insert_pending_exit(_rec(), bot='options', source='paper', db_path=db_path)
        rows = _db.get_pending_exits('options', 'paper', 'pending', db_path=db_path)
        assert len(rows) == 1
        assert rows[0]['order_id'] == 'order-1'
        assert rows[0]['status'] == 'pending'
        assert rows[0]['intended_qty'] == 1.0

    def test_duplicate_order_id_is_ignored_not_error(self, db_path):
        import db as _db
        _db.insert_pending_exit(_rec(order_id='dup'), bot='options', db_path=db_path)
        _db.insert_pending_exit(_rec(order_id='dup', qty=99.0), bot='options', db_path=db_path)
        rows = _db.get_pending_exits('options', 'paper', 'pending', db_path=db_path)
        assert len(rows) == 1
        assert rows[0]['intended_qty'] == 1.0  # first insert wins, second silently ignored

    def test_filters_by_bot_and_source(self, db_path):
        import db as _db
        _db.insert_pending_exit(_rec(order_id='o1'), bot='options', source='paper', db_path=db_path)
        _db.insert_pending_exit(_rec(order_id='o2'), bot='stock', source='paper', db_path=db_path)
        _db.insert_pending_exit(_rec(order_id='o3'), bot='options', source='live', db_path=db_path)
        options_paper = _db.get_pending_exits('options', 'paper', 'pending', db_path=db_path)
        assert [r['order_id'] for r in options_paper] == ['o1']

    def test_empty_when_none_pending(self, db_path):
        import db as _db
        assert _db.get_pending_exits('options', 'paper', 'pending', db_path=db_path) == []


class TestResolve:
    def test_resolve_filled_removes_from_pending(self, db_path):
        import db as _db
        _db.insert_pending_exit(_rec(), bot='options', db_path=db_path)
        _db.resolve_pending_exit('order-1', 'filled', db_path=db_path)
        assert _db.get_pending_exits('options', 'paper', 'pending', db_path=db_path) == []
        filled = _db.get_pending_exits('options', 'paper', 'filled', db_path=db_path)
        assert len(filled) == 1
        assert filled[0]['resolved_at'] is not None

    def test_resolve_exit_failed(self, db_path):
        import db as _db
        _db.insert_pending_exit(_rec(), bot='options', db_path=db_path)
        _db.resolve_pending_exit('order-1', 'exit_failed', db_path=db_path)
        rows = _db.get_pending_exits('options', 'paper', 'exit_failed', db_path=db_path)
        assert len(rows) == 1


class TestRetryAndAbandon:
    def test_bump_retry_increments_and_returns_new_count(self, db_path):
        import db as _db
        _db.insert_pending_exit(_rec(), bot='options', db_path=db_path)
        assert _db.bump_pending_exit_retry('order-1', db_path=db_path) == 1
        assert _db.bump_pending_exit_retry('order-1', db_path=db_path) == 2

    def test_mark_abandoned(self, db_path):
        import db as _db
        _db.insert_pending_exit(_rec(), bot='options', db_path=db_path)
        _db.mark_pending_exit_abandoned('order-1', db_path=db_path)
        rows = _db.get_pending_exits('options', 'paper', 'abandoned', db_path=db_path)
        assert len(rows) == 1
        assert rows[0]['resolved_at'] is not None


class TestNoTradesOutcomesTouched:
    """R1's explicit boundary: this table must never write to trades/outcomes."""

    def test_pending_exit_lifecycle_leaves_trades_and_outcomes_empty(self, db_path):
        import db as _db
        _db.insert_pending_exit(_rec(), bot='options', db_path=db_path)
        _db.resolve_pending_exit('order-1', 'filled', db_path=db_path)
        with _db.get_conn(db_path) as conn:
            trades_count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            outcomes_count = conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
        assert trades_count == 0
        assert outcomes_count == 0
