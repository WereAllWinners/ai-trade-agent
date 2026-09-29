"""
tests/test_options_outcome_tracker.py — sprint03 E2.2 peek-don't-pop retry
handling, mirroring tests/test_outcome_tracker.py's TestUnreconciledRetryHandling
against OptionsOutcomeTracker's contract/quantity shape.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'analysis'))

from options_outcome_tracker import OptionsOutcomeTracker


def _make_tracker(trade_log_path, paper=True):
    tracker = OptionsOutcomeTracker.__new__(OptionsOutcomeTracker)
    tracker.trade_log_path = trade_log_path
    tracker.paper = paper
    return tracker


def _write_rows(path, rows):
    with open(path, 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')


_CONTRACT = 'AAPL260116C00200000'


class TestUnreconciledRetryHandling:
    """Mirror of test_outcome_tracker.py's class — same peek-don't-pop fix,
    same behavioral guarantees, against contract/quantity fields."""

    def test_missing_fill_price_leaves_buy_in_queue_not_popped(self, tmp_path):
        trade_log_path = tmp_path / 'options_trade_log.jsonl'
        _write_rows(trade_log_path, [
            {'timestamp': '2026-06-01T09:00:00', 'contract': _CONTRACT, 'action': 'buy',
             'quantity': 2.0, 'order_id': 'buy-1', 'source': 'paper'},
            {'timestamp': '2026-06-02T09:00:00', 'contract': _CONTRACT, 'action': 'sell',
             'quantity': 2.0, 'order_id': 'sell-1', 'source': 'paper'},
        ])
        tracker = _make_tracker(trade_log_path)

        fill_prices = {'sell-1': 5.0}
        mock_db = MagicMock()
        mock_db.upsert_unreconciled_order.return_value = 1
        with patch.object(tracker, 'get_fill_price', side_effect=lambda oid: fill_prices.get(oid)), \
             patch.object(tracker, 'get_order_status', return_value={'status': 'unknown'}), \
             patch('options_outcome_tracker._fetch_spy_return', return_value=None), \
             patch('options_outcome_tracker._db', mock_db):
            trades = tracker.load_raw_trades()
            outcomes = tracker.match_and_calculate_pnl(trades)

        assert outcomes == []
        mock_db.upsert_unreconciled_order.assert_called_once()
        mock_db.mark_unreconciled_order_abandoned.assert_not_called()

    def test_retryable_on_a_later_pass_once_fill_price_becomes_available(self, tmp_path):
        trade_log_path = tmp_path / 'options_trade_log.jsonl'
        _write_rows(trade_log_path, [
            {'timestamp': '2026-06-01T09:00:00', 'contract': _CONTRACT, 'action': 'buy',
             'quantity': 2.0, 'order_id': 'buy-1', 'source': 'paper'},
            {'timestamp': '2026-06-02T09:00:00', 'contract': _CONTRACT, 'action': 'sell',
             'quantity': 2.0, 'order_id': 'sell-1', 'source': 'paper'},
        ])
        tracker = _make_tracker(trade_log_path)

        fill_prices = {'buy-1': 4.0, 'sell-1': 5.0}
        with patch.object(tracker, 'get_fill_price', side_effect=lambda oid: fill_prices.get(oid)), \
             patch('options_outcome_tracker._fetch_spy_return', return_value=None):
            trades = tracker.load_raw_trades()
            outcomes = tracker.match_and_calculate_pnl(trades)

        assert len(outcomes) == 1
        assert outcomes[0]['buy_order_id'] == 'buy-1'
        assert outcomes[0]['sell_order_id'] == 'sell-1'

    def test_retry_count_increments_across_passes(self, tmp_path):
        trade_log_path = tmp_path / 'options_trade_log.jsonl'
        _write_rows(trade_log_path, [
            {'timestamp': '2026-06-01T09:00:00', 'contract': _CONTRACT, 'action': 'buy',
             'quantity': 2.0, 'order_id': 'buy-1', 'source': 'paper'},
            {'timestamp': '2026-06-02T09:00:00', 'contract': _CONTRACT, 'action': 'sell',
             'quantity': 2.0, 'order_id': 'sell-1', 'source': 'paper'},
        ])
        tracker = _make_tracker(trade_log_path)

        fill_prices = {'sell-1': 5.0}
        mock_db = MagicMock()
        mock_db.upsert_unreconciled_order.side_effect = [1, 2, 3]
        with patch.object(tracker, 'get_fill_price', side_effect=lambda oid: fill_prices.get(oid)), \
             patch.object(tracker, 'get_order_status', return_value={'status': 'unknown'}), \
             patch('options_outcome_tracker._fetch_spy_return', return_value=None), \
             patch('options_outcome_tracker._db', mock_db):
            trades = tracker.load_raw_trades()
            tracker.match_and_calculate_pnl(trades)
            tracker.match_and_calculate_pnl(trades)
            tracker.match_and_calculate_pnl(trades)

        assert mock_db.upsert_unreconciled_order.call_count == 3
        assert mock_db.mark_unreconciled_order_abandoned.call_count == 0

    def test_abandonment_after_max_retries_pops_buy_and_marks_status(self, tmp_path, caplog):
        import logging
        trade_log_path = tmp_path / 'options_trade_log.jsonl'
        _write_rows(trade_log_path, [
            {'timestamp': '2026-06-01T09:00:00', 'contract': _CONTRACT, 'action': 'buy',
             'quantity': 2.0, 'order_id': 'buy-1', 'source': 'paper'},
            {'timestamp': '2026-06-02T09:00:00', 'contract': _CONTRACT, 'action': 'sell',
             'quantity': 2.0, 'order_id': 'sell-1', 'source': 'paper'},
        ])
        tracker = _make_tracker(trade_log_path)

        fill_prices = {'sell-1': 5.0}
        mock_db = MagicMock()
        mock_db.upsert_unreconciled_order.return_value = 10
        with patch.object(tracker, 'get_fill_price', side_effect=lambda oid: fill_prices.get(oid)), \
             patch.object(tracker, 'get_order_status', return_value={'status': 'unknown'}), \
             patch('options_outcome_tracker._fetch_spy_return', return_value=None), \
             patch('options_outcome_tracker._db', mock_db), \
             caplog.at_level(logging.WARNING):
            trades = tracker.load_raw_trades()
            tracker.match_and_calculate_pnl(trades)

        mock_db.mark_unreconciled_order_abandoned.assert_called_once()
        assert any('abandoning' in r.getMessage() for r in caplog.records)

    def test_healthy_pairs_unaffected_by_retry_logic(self, tmp_path):
        trade_log_path = tmp_path / 'options_trade_log.jsonl'
        _write_rows(trade_log_path, [
            {'timestamp': '2026-06-01T09:00:00', 'contract': _CONTRACT, 'action': 'buy',
             'quantity': 2.0, 'order_id': 'buy-1', 'source': 'paper'},
            {'timestamp': '2026-06-02T09:00:00', 'contract': _CONTRACT, 'action': 'sell',
             'quantity': 2.0, 'order_id': 'sell-1', 'source': 'paper'},
        ])
        tracker = _make_tracker(trade_log_path)

        fill_prices = {'buy-1': 4.0, 'sell-1': 5.0}
        mock_db = MagicMock()
        with patch.object(tracker, 'get_fill_price', side_effect=lambda oid: fill_prices.get(oid)), \
             patch('options_outcome_tracker._fetch_spy_return', return_value=None), \
             patch('options_outcome_tracker._db', mock_db):
            trades = tracker.load_raw_trades()
            outcomes = tracker.match_and_calculate_pnl(trades)

        assert len(outcomes) == 1
        mock_db.upsert_unreconciled_order.assert_not_called()

    def test_fifo_order_preserved_when_buy_stuck_behind_multiple_sells(self, tmp_path):
        trade_log_path = tmp_path / 'options_trade_log.jsonl'
        _write_rows(trade_log_path, [
            {'timestamp': '2026-06-01T09:00:00', 'contract': _CONTRACT, 'action': 'buy',
             'quantity': 2.0, 'order_id': 'buy-1', 'source': 'paper'},
            {'timestamp': '2026-06-02T09:00:00', 'contract': _CONTRACT, 'action': 'buy',
             'quantity': 2.0, 'order_id': 'buy-2', 'source': 'paper'},
            {'timestamp': '2026-06-03T09:00:00', 'contract': _CONTRACT, 'action': 'sell',
             'quantity': 2.0, 'order_id': 'sell-1', 'source': 'paper'},
            {'timestamp': '2026-06-04T09:00:00', 'contract': _CONTRACT, 'action': 'sell',
             'quantity': 2.0, 'order_id': 'sell-2', 'source': 'paper'},
        ])
        tracker = _make_tracker(trade_log_path)

        fill_prices = {'buy-2': 4.1, 'sell-1': 5.0, 'sell-2': 5.1}
        mock_db = MagicMock()
        mock_db.upsert_unreconciled_order.return_value = 1
        with patch.object(tracker, 'get_fill_price', side_effect=lambda oid: fill_prices.get(oid)), \
             patch.object(tracker, 'get_order_status', return_value={'status': 'unknown'}), \
             patch('options_outcome_tracker._fetch_spy_return', return_value=None), \
             patch('options_outcome_tracker._db', mock_db):
            trades = tracker.load_raw_trades()
            outcomes = tracker.match_and_calculate_pnl(trades)

        assert outcomes == []
        assert mock_db.upsert_unreconciled_order.call_count == 2

    def test_no_duplicate_unreconciled_rows_across_two_tracker_runs(self, tmp_path):
        import db as _real_db
        db_path = tmp_path / 'test.db'
        _real_db.init_db(db_path)

        trade_log_path = tmp_path / 'options_trade_log.jsonl'
        _write_rows(trade_log_path, [
            {'timestamp': '2026-06-01T09:00:00', 'contract': _CONTRACT, 'action': 'buy',
             'quantity': 2.0, 'order_id': 'buy-1', 'source': 'paper'},
            {'timestamp': '2026-06-02T09:00:00', 'contract': _CONTRACT, 'action': 'sell',
             'quantity': 2.0, 'order_id': 'sell-1', 'source': 'paper'},
        ])
        tracker = _make_tracker(trade_log_path)

        class _DBWithPath:
            def upsert_unreconciled_order(self, rec, bot, source='paper'):
                return _real_db.upsert_unreconciled_order(rec, bot, source=source, db_path=db_path)

            def mark_unreconciled_order_abandoned(self, order_id, reason, bot, source='paper'):
                return _real_db.mark_unreconciled_order_abandoned(
                    order_id, reason, bot, source=source, db_path=db_path)

        fill_prices = {'sell-1': 5.0}
        with patch.object(tracker, 'get_fill_price', side_effect=lambda oid: fill_prices.get(oid)), \
             patch.object(tracker, 'get_order_status', return_value={'status': 'unknown'}), \
             patch('options_outcome_tracker._fetch_spy_return', return_value=None), \
             patch('options_outcome_tracker._db', _DBWithPath()):
            trades = tracker.load_raw_trades()
            tracker.match_and_calculate_pnl(trades)
            tracker.match_and_calculate_pnl(trades)

        with _real_db.get_conn(db_path) as conn:
            rows = conn.execute(
                "SELECT retry_count FROM unreconciled_orders WHERE order_id='buy-1'"
            ).fetchall()
        assert len(rows) == 1
        assert rows[0]['retry_count'] == 1
