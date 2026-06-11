"""
Unit tests for PR-1 DB migrations and new query helpers.

All tests run offline against a temp SQLite file (no live API calls).
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))


@pytest.fixture()
def db_path(tmp_path):
    import db as _db
    p = tmp_path / 'test_trading.db'
    _db.init_db(p)
    return p


class TestMigrations:
    """New columns are present after init_db()."""

    def test_decisions_has_order_id_column(self, db_path):
        import db as _db
        with _db.get_conn(db_path) as conn:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(decisions)")}
        assert 'order_id' in cols

    def test_decisions_has_regime_column(self, db_path):
        import db as _db
        with _db.get_conn(db_path) as conn:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(decisions)")}
        assert 'regime' in cols

    def test_outcomes_has_spy_return_pct_column(self, db_path):
        import db as _db
        with _db.get_conn(db_path) as conn:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(outcomes)")}
        assert 'spy_return_pct' in cols

    def test_outcomes_has_excess_return_pct_column(self, db_path):
        import db as _db
        with _db.get_conn(db_path) as conn:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(outcomes)")}
        assert 'excess_return_pct' in cols

    def test_outcomes_has_regime_column(self, db_path):
        import db as _db
        with _db.get_conn(db_path) as conn:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(outcomes)")}
        assert 'regime' in cols

    def test_init_db_is_idempotent(self, db_path):
        """Running init_db twice must not raise (idempotent ALTER TABLE guards)."""
        import db as _db
        _db.init_db(db_path)   # second call — must not error


class TestGetOutcomeByOrderId:
    def test_returns_matching_outcome(self, db_path):
        import db as _db
        _db.insert_outcome({
            'symbol':           'AAPL',
            'buy_order_id':     'test-oid-123',
            'sell_order_id':    'test-sell-oid',
            'entry_timestamp':  '2026-01-10T10:00:00',
            'exit_timestamp':   '2026-01-12T15:00:00',
            'entry_price':      180.0,
            'exit_price':       190.0,
            'shares':           10.0,
            'realized_pnl':     100.0,
            'pnl_pct':          0.0556,
            'hold_hours':       53.0,
            'entry_confidence': 0.75,
            'entry_reasoning':  'test',
            'won':              True,
        }, db_path=db_path)
        result = _db.get_outcome_by_order_id('test-oid-123', db_path=db_path)
        assert result is not None
        assert result['symbol'] == 'AAPL'
        assert result['pnl_pct'] == pytest.approx(0.0556)

    def test_returns_none_for_unknown_order(self, db_path):
        import db as _db
        result = _db.get_outcome_by_order_id('nonexistent-oid', db_path=db_path)
        assert result is None

    def test_distinguishes_two_same_symbol_same_day_trades(self, db_path):
        """Two outcomes for the same symbol on the same day are returned by their respective order IDs."""
        import db as _db
        base = {
            'symbol':           'TSLA',
            'entry_timestamp':  '2026-01-15T09:30:00',
            'exit_timestamp':   '2026-01-15T14:00:00',
            'entry_price':      200.0,
            'shares':           5.0,
            'hold_hours':       4.5,
            'entry_confidence': 0.70,
            'entry_reasoning':  'test',
        }
        _db.insert_outcome({**base, 'buy_order_id': 'oid-win', 'sell_order_id': 'oid-win-sell',
                             'exit_price': 210.0, 'realized_pnl': 50.0, 'pnl_pct': 0.05, 'won': True},
                           db_path=db_path)
        _db.insert_outcome({**base, 'buy_order_id': 'oid-loss', 'sell_order_id': 'oid-loss-sell',
                             'exit_price': 190.0, 'realized_pnl': -50.0, 'pnl_pct': -0.05, 'won': False},
                           db_path=db_path)

        win  = _db.get_outcome_by_order_id('oid-win',  db_path=db_path)
        loss = _db.get_outcome_by_order_id('oid-loss', db_path=db_path)
        assert win['won']  == 1
        assert loss['won'] == 0


class TestGetCalibrationMap:
    def test_returns_empty_dict_on_empty_db(self, db_path):
        import db as _db
        result = _db.get_calibration_map('stock', db_path=db_path)
        assert result == {}

    def test_returns_win_rate_for_populated_bin(self, db_path):
        import db as _db
        # Insert 10 outcomes in the 0.70-0.80 bin: 8 wins, 2 losses
        cutoff_ts = (datetime.now() - timedelta(days=30)).isoformat()
        base = {
            'symbol':          'SPY',
            'entry_timestamp': cutoff_ts,
            'exit_timestamp':  datetime.now().isoformat(),
            'entry_price': 400.0, 'exit_price': 410.0, 'shares': 1.0,
            'hold_hours': 24.0, 'entry_reasoning': 'test',
        }
        for i in range(10):
            win = i < 8
            _db.insert_outcome({
                **base,
                'buy_order_id':     f'oid-cal-{i}',
                'realized_pnl':     10.0 if win else -10.0,
                'pnl_pct':          0.025 if win else -0.025,
                'entry_confidence': 0.75,
                'won':              win,
            }, db_path=db_path)

        result = _db.get_calibration_map('stock', db_path=db_path)
        assert '0.70-0.80' in result
        assert result['0.70-0.80']['count'] == 10
        assert result['0.70-0.80']['win_rate'] == pytest.approx(0.8)

    def test_omits_bins_below_minimum_sample_count(self, db_path):
        """Bins with fewer than 10 outcomes should not appear in the result."""
        import db as _db
        cutoff_ts = (datetime.now() - timedelta(days=30)).isoformat()
        # Insert only 5 outcomes in the 0.80-0.90 bin (below min of 10)
        for i in range(5):
            _db.insert_outcome({
                'symbol': 'QQQ',
                'buy_order_id': f'oid-small-{i}',
                'entry_timestamp': cutoff_ts,
                'exit_timestamp':  datetime.now().isoformat(),
                'entry_price': 300.0, 'exit_price': 310.0, 'shares': 1.0,
                'realized_pnl': 10.0, 'pnl_pct': 0.033,
                'hold_hours': 12.0, 'entry_confidence': 0.85,
                'entry_reasoning': 'test', 'won': True,
            }, db_path=db_path)

        result = _db.get_calibration_map('stock', db_path=db_path)
        assert '0.80-0.90' not in result


class TestInsertDecisionNewColumns:
    def test_decision_stores_order_id_and_regime(self, db_path):
        import db as _db
        _db.insert_decision({
            'timestamp':    '2026-01-10T10:00:00',
            'session_id':   'sess-1',
            'bot':          'stock',
            'symbol':       'AAPL',
            'decision':     'buy',
            'confidence':   0.82,
            'reasoning':    'test',
            'executed':     True,
            'order_id':     'order-abc-123',
            'regime':       'bull_midvol',
        }, db_path=db_path)

        with _db.get_conn(db_path) as conn:
            row = conn.execute(
                "SELECT order_id, regime FROM decisions WHERE symbol='AAPL'"
            ).fetchone()
        assert row['order_id'] == 'order-abc-123'
        assert row['regime']   == 'bull_midvol'


class TestInsertOutcomeNewColumns:
    def test_outcome_stores_spy_and_regime(self, db_path):
        import db as _db
        _db.insert_outcome({
            'symbol':            'MSFT',
            'buy_order_id':      'oid-msft-1',
            'entry_timestamp':   '2026-01-10T09:30:00',
            'exit_timestamp':    '2026-01-12T15:00:00',
            'entry_price':       400.0, 'exit_price': 420.0, 'shares': 5.0,
            'realized_pnl':      100.0, 'pnl_pct': 0.05,
            'hold_hours':        53.0, 'entry_confidence': 0.78,
            'entry_reasoning':   'test', 'won': True,
            'spy_return_pct':    0.02,
            'excess_return_pct': 0.03,
            'regime':            'bull_lowvol',
        }, db_path=db_path)

        with _db.get_conn(db_path) as conn:
            row = conn.execute(
                "SELECT spy_return_pct, excess_return_pct, regime FROM outcomes WHERE symbol='MSFT'"
            ).fetchone()
        assert row['spy_return_pct']    == pytest.approx(0.02)
        assert row['excess_return_pct'] == pytest.approx(0.03)
        assert row['regime']            == 'bull_lowvol'
