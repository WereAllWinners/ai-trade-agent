"""
tests/test_outcome_tracker.py — sprint02 D4.3 paper/live FIFO isolation.

Confirms match_and_calculate_pnl() groups by (symbol, source) rather than
symbol alone, so a paper BUY can never get FIFO-matched against a live SELL
(or vice versa) — the bug empirically confirmed in trade_log.jsonl (2/15
sampled order_ids belonged to the live account despite being commingled with
paper order_ids for the same symbols in the same file).
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'analysis'))

from outcome_tracker import OutcomeTracker


def _make_tracker(trade_log_path, paper=True):
    tracker = OutcomeTracker.__new__(OutcomeTracker)
    tracker.trade_log_path = trade_log_path
    tracker.paper = paper
    return tracker


def _write_rows(path, rows):
    with open(path, 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')


class TestFifoSourceIsolation:
    def test_paper_buy_never_matches_live_sell_for_same_symbol(self, tmp_path):
        """The core bug: a paper BUY and a live SELL for the same symbol,
        interleaved in the shared trade log, must NOT be FIFO-paired."""
        trade_log_path = tmp_path / 'trade_log.jsonl'
        _write_rows(trade_log_path, [
            {'timestamp': '2026-06-01T09:00:00', 'symbol': 'ALKS', 'action': 'buy',
             'shares': 10.0, 'order_id': 'paper-buy-1', 'source': 'paper'},
            {'timestamp': '2026-06-02T09:00:00', 'symbol': 'ALKS', 'action': 'sell',
             'shares': 10.0, 'order_id': 'live-sell-1', 'source': 'live'},
        ])
        tracker = _make_tracker(trade_log_path)

        fill_prices = {'paper-buy-1': 40.0, 'live-sell-1': 50.0}
        with patch.object(tracker, 'get_fill_price', side_effect=lambda oid: fill_prices.get(oid)), \
             patch.object(tracker, 'get_order_status', return_value={'status': 'filled'}), \
             patch('outcome_tracker._fetch_spy_return', return_value=None), \
             patch('outcome_tracker._db'):
            trades = tracker.load_raw_trades()
            outcomes = tracker.match_and_calculate_pnl(trades)

        # No outcome produced — the paper buy and live sell sit in different
        # (symbol, source) buckets and are never paired against each other.
        assert outcomes == []

    def test_paper_buy_matches_paper_sell_normally(self, tmp_path):
        """Same-source pairs still match correctly — this isn't a blanket
        block, only cross-source pairing is prevented."""
        trade_log_path = tmp_path / 'trade_log.jsonl'
        _write_rows(trade_log_path, [
            {'timestamp': '2026-06-01T09:00:00', 'symbol': 'ALKS', 'action': 'buy',
             'shares': 10.0, 'order_id': 'paper-buy-2', 'source': 'paper'},
            {'timestamp': '2026-06-02T09:00:00', 'symbol': 'ALKS', 'action': 'sell',
             'shares': 10.0, 'order_id': 'paper-sell-2', 'source': 'paper'},
        ])
        tracker = _make_tracker(trade_log_path)

        fill_prices = {'paper-buy-2': 40.0, 'paper-sell-2': 44.0}
        with patch.object(tracker, 'get_fill_price', side_effect=lambda oid: fill_prices.get(oid)), \
             patch('outcome_tracker._fetch_spy_return', return_value=None):
            trades = tracker.load_raw_trades()
            outcomes = tracker.match_and_calculate_pnl(trades)

        assert len(outcomes) == 1
        assert outcomes[0]['buy_order_id'] == 'paper-buy-2'
        assert outcomes[0]['sell_order_id'] == 'paper-sell-2'

    def test_legacy_untagged_rows_still_pair_with_each_other(self, tmp_path):
        """Pre-fix historical rows (no 'source' field at all) must continue
        to FIFO-match exactly as before — no regression for existing history.
        Both sides bucket under source=None."""
        trade_log_path = tmp_path / 'trade_log.jsonl'
        _write_rows(trade_log_path, [
            {'timestamp': '2026-03-01T09:00:00', 'symbol': 'MRK', 'action': 'buy',
             'shares': 5.0, 'order_id': 'legacy-buy-1'},  # no 'source' key
            {'timestamp': '2026-03-02T09:00:00', 'symbol': 'MRK', 'action': 'sell',
             'shares': 5.0, 'order_id': 'legacy-sell-1'},  # no 'source' key
        ])
        tracker = _make_tracker(trade_log_path)

        fill_prices = {'legacy-buy-1': 100.0, 'legacy-sell-1': 107.0}
        with patch.object(tracker, 'get_fill_price', side_effect=lambda oid: fill_prices.get(oid)), \
             patch('outcome_tracker._fetch_spy_return', return_value=None):
            trades = tracker.load_raw_trades()
            outcomes = tracker.match_and_calculate_pnl(trades)

        assert len(outcomes) == 1
        assert outcomes[0]['buy_order_id'] == 'legacy-buy-1'
        assert outcomes[0]['sell_order_id'] == 'legacy-sell-1'

    def test_legacy_buy_does_not_match_new_tagged_sell(self, tmp_path):
        """A legacy (untagged) BUY and a new source-tagged SELL for the same
        symbol must NOT pair — this is the expected, accepted gap that D4.4's
        targeted backfill closes for currently-open positions specifically."""
        trade_log_path = tmp_path / 'trade_log.jsonl'
        _write_rows(trade_log_path, [
            {'timestamp': '2026-03-01T09:00:00', 'symbol': 'AEP', 'action': 'buy',
             'shares': 8.0, 'order_id': 'legacy-buy-2'},  # no 'source' key
            {'timestamp': '2026-07-01T09:00:00', 'symbol': 'AEP', 'action': 'sell',
             'shares': 8.0, 'order_id': 'tagged-sell-2', 'source': 'live'},
        ])
        tracker = _make_tracker(trade_log_path)

        fill_prices = {'legacy-buy-2': 130.0, 'tagged-sell-2': 128.0}
        with patch.object(tracker, 'get_fill_price', side_effect=lambda oid: fill_prices.get(oid)), \
             patch.object(tracker, 'get_order_status', return_value={'status': 'filled'}), \
             patch('outcome_tracker._fetch_spy_return', return_value=None), \
             patch('outcome_tracker._db'):
            trades = tracker.load_raw_trades()
            outcomes = tracker.match_and_calculate_pnl(trades)

        assert outcomes == []


class TestUnreconciledRetryHandling:
    """sprint03 E2.2 — peek-don't-pop: a missing fill-price lookup no longer
    permanently destroys the buy/sell pair. It's retried across passes via
    upsert_unreconciled_order's retry_count, until UNRECONCILED_MAX_RETRIES is
    exceeded and the pair is abandoned (only then does the buy leave the FIFO
    queue)."""

    def test_missing_fill_price_leaves_buy_in_queue_not_popped(self, tmp_path):
        trade_log_path = tmp_path / 'trade_log.jsonl'
        _write_rows(trade_log_path, [
            {'timestamp': '2026-06-01T09:00:00', 'symbol': 'GE', 'action': 'buy',
             'shares': 5.0, 'order_id': 'buy-1', 'source': 'paper'},
            {'timestamp': '2026-06-02T09:00:00', 'symbol': 'GE', 'action': 'sell',
             'shares': 5.0, 'order_id': 'sell-1', 'source': 'paper'},
        ])
        tracker = _make_tracker(trade_log_path)

        # buy-1's fill price is unavailable (simulating a transient lookup failure)
        fill_prices = {'sell-1': 50.0}
        mock_db = MagicMock()
        mock_db.upsert_unreconciled_order.return_value = 1  # 1st retry
        with patch.object(tracker, 'get_fill_price', side_effect=lambda oid: fill_prices.get(oid)), \
             patch.object(tracker, 'get_order_status', return_value={'status': 'unknown'}), \
             patch('outcome_tracker._fetch_spy_return', return_value=None), \
             patch('outcome_tracker._db', mock_db):
            trades = tracker.load_raw_trades()
            outcomes = tracker.match_and_calculate_pnl(trades)

        assert outcomes == []
        mock_db.upsert_unreconciled_order.assert_called_once()
        mock_db.mark_unreconciled_order_abandoned.assert_not_called()

    def test_retryable_on_a_later_pass_once_fill_price_becomes_available(self, tmp_path):
        """The same trade_log rows, re-processed once the fill price is
        available — the pair now matches, proving nothing was structurally
        destroyed on the earlier failed pass."""
        trade_log_path = tmp_path / 'trade_log.jsonl'
        _write_rows(trade_log_path, [
            {'timestamp': '2026-06-01T09:00:00', 'symbol': 'GE', 'action': 'buy',
             'shares': 5.0, 'order_id': 'buy-1', 'source': 'paper'},
            {'timestamp': '2026-06-02T09:00:00', 'symbol': 'GE', 'action': 'sell',
             'shares': 5.0, 'order_id': 'sell-1', 'source': 'paper'},
        ])
        tracker = _make_tracker(trade_log_path)

        fill_prices = {'buy-1': 40.0, 'sell-1': 50.0}
        with patch.object(tracker, 'get_fill_price', side_effect=lambda oid: fill_prices.get(oid)), \
             patch('outcome_tracker._fetch_spy_return', return_value=None):
            trades = tracker.load_raw_trades()
            outcomes = tracker.match_and_calculate_pnl(trades)

        assert len(outcomes) == 1
        assert outcomes[0]['buy_order_id'] == 'buy-1'
        assert outcomes[0]['sell_order_id'] == 'sell-1'

    def test_retry_count_increments_across_passes(self, tmp_path):
        trade_log_path = tmp_path / 'trade_log.jsonl'
        _write_rows(trade_log_path, [
            {'timestamp': '2026-06-01T09:00:00', 'symbol': 'GE', 'action': 'buy',
             'shares': 5.0, 'order_id': 'buy-1', 'source': 'paper'},
            {'timestamp': '2026-06-02T09:00:00', 'symbol': 'GE', 'action': 'sell',
             'shares': 5.0, 'order_id': 'sell-1', 'source': 'paper'},
        ])
        tracker = _make_tracker(trade_log_path)

        fill_prices = {'sell-1': 50.0}
        mock_db = MagicMock()
        mock_db.upsert_unreconciled_order.side_effect = [1, 2, 3]
        with patch.object(tracker, 'get_fill_price', side_effect=lambda oid: fill_prices.get(oid)), \
             patch.object(tracker, 'get_order_status', return_value={'status': 'unknown'}), \
             patch('outcome_tracker._fetch_spy_return', return_value=None), \
             patch('outcome_tracker._db', mock_db):
            trades = tracker.load_raw_trades()
            tracker.match_and_calculate_pnl(trades)
            tracker.match_and_calculate_pnl(trades)
            tracker.match_and_calculate_pnl(trades)

        assert mock_db.upsert_unreconciled_order.call_count == 3
        assert mock_db.mark_unreconciled_order_abandoned.call_count == 0

    def test_abandonment_after_max_retries_pops_buy_and_marks_status(self, tmp_path, caplog):
        import logging
        trade_log_path = tmp_path / 'trade_log.jsonl'
        _write_rows(trade_log_path, [
            {'timestamp': '2026-06-01T09:00:00', 'symbol': 'GE', 'action': 'buy',
             'shares': 5.0, 'order_id': 'buy-1', 'source': 'paper'},
            {'timestamp': '2026-06-02T09:00:00', 'symbol': 'GE', 'action': 'sell',
             'shares': 5.0, 'order_id': 'sell-1', 'source': 'paper'},
        ])
        tracker = _make_tracker(trade_log_path)

        fill_prices = {'sell-1': 50.0}
        mock_db = MagicMock()
        mock_db.upsert_unreconciled_order.return_value = 10  # == default UNRECONCILED_MAX_RETRIES
        with patch.object(tracker, 'get_fill_price', side_effect=lambda oid: fill_prices.get(oid)), \
             patch.object(tracker, 'get_order_status', return_value={'status': 'unknown'}), \
             patch('outcome_tracker._fetch_spy_return', return_value=None), \
             patch('outcome_tracker._db', mock_db), \
             caplog.at_level(logging.WARNING):
            trades = tracker.load_raw_trades()
            tracker.match_and_calculate_pnl(trades)

        mock_db.mark_unreconciled_order_abandoned.assert_called_once()
        assert any('abandoning' in r.getMessage() for r in caplog.records)

    def test_healthy_pairs_unaffected_by_retry_logic(self, tmp_path):
        trade_log_path = tmp_path / 'trade_log.jsonl'
        _write_rows(trade_log_path, [
            {'timestamp': '2026-06-01T09:00:00', 'symbol': 'GE', 'action': 'buy',
             'shares': 5.0, 'order_id': 'buy-1', 'source': 'paper'},
            {'timestamp': '2026-06-02T09:00:00', 'symbol': 'GE', 'action': 'sell',
             'shares': 5.0, 'order_id': 'sell-1', 'source': 'paper'},
        ])
        tracker = _make_tracker(trade_log_path)

        fill_prices = {'buy-1': 40.0, 'sell-1': 50.0}
        mock_db = MagicMock()
        with patch.object(tracker, 'get_fill_price', side_effect=lambda oid: fill_prices.get(oid)), \
             patch('outcome_tracker._fetch_spy_return', return_value=None), \
             patch('outcome_tracker._db', mock_db):
            trades = tracker.load_raw_trades()
            outcomes = tracker.match_and_calculate_pnl(trades)

        assert len(outcomes) == 1
        mock_db.upsert_unreconciled_order.assert_not_called()

    def test_fifo_order_preserved_when_buy_stuck_behind_multiple_sells(self, tmp_path):
        """A stuck buy at the head of the queue must not let a later sell
        jump ahead and match a different, later buy out of order."""
        trade_log_path = tmp_path / 'trade_log.jsonl'
        _write_rows(trade_log_path, [
            {'timestamp': '2026-06-01T09:00:00', 'symbol': 'GE', 'action': 'buy',
             'shares': 5.0, 'order_id': 'buy-1', 'source': 'paper'},
            {'timestamp': '2026-06-02T09:00:00', 'symbol': 'GE', 'action': 'buy',
             'shares': 5.0, 'order_id': 'buy-2', 'source': 'paper'},
            {'timestamp': '2026-06-03T09:00:00', 'symbol': 'GE', 'action': 'sell',
             'shares': 5.0, 'order_id': 'sell-1', 'source': 'paper'},
            {'timestamp': '2026-06-04T09:00:00', 'symbol': 'GE', 'action': 'sell',
             'shares': 5.0, 'order_id': 'sell-2', 'source': 'paper'},
        ])
        tracker = _make_tracker(trade_log_path)

        # buy-1's fill price is never available; buy-2 and both sells are fine.
        fill_prices = {'buy-2': 41.0, 'sell-1': 50.0, 'sell-2': 51.0}
        mock_db = MagicMock()
        mock_db.upsert_unreconciled_order.return_value = 1
        with patch.object(tracker, 'get_fill_price', side_effect=lambda oid: fill_prices.get(oid)), \
             patch.object(tracker, 'get_order_status', return_value={'status': 'unknown'}), \
             patch('outcome_tracker._fetch_spy_return', return_value=None), \
             patch('outcome_tracker._db', mock_db):
            trades = tracker.load_raw_trades()
            outcomes = tracker.match_and_calculate_pnl(trades)

        # Neither sell can resolve buy-1 (stuck at head); buy-2 never gets a
        # chance to pair with either sell out of order — both sells report
        # unreconciled against buy-1, and no outcome is produced.
        assert outcomes == []
        assert mock_db.upsert_unreconciled_order.call_count == 2

    def test_no_duplicate_unreconciled_rows_across_two_tracker_runs(self, tmp_path):
        """Real sqlite (not mocked) — two separate match_and_calculate_pnl
        passes for the same unresolved pair must not create duplicate
        unreconciled_orders rows."""
        import db as _real_db
        db_path = tmp_path / 'test.db'
        _real_db.init_db(db_path)

        trade_log_path = tmp_path / 'trade_log.jsonl'
        _write_rows(trade_log_path, [
            {'timestamp': '2026-06-01T09:00:00', 'symbol': 'GE', 'action': 'buy',
             'shares': 5.0, 'order_id': 'buy-1', 'source': 'paper'},
            {'timestamp': '2026-06-02T09:00:00', 'symbol': 'GE', 'action': 'sell',
             'shares': 5.0, 'order_id': 'sell-1', 'source': 'paper'},
        ])
        tracker = _make_tracker(trade_log_path)

        # upsert_unreconciled_order/mark_unreconciled_order_abandoned default
        # db_path to db.DB_PATH bound at function-definition time — patching
        # the module attribute afterward doesn't affect that already-bound
        # default, so route db_path through explicitly via a thin wrapper
        # instead of trying to patch it away.
        class _DBWithPath:
            def upsert_unreconciled_order(self, rec, bot, source='paper'):
                return _real_db.upsert_unreconciled_order(rec, bot, source=source, db_path=db_path)

            def mark_unreconciled_order_abandoned(self, order_id, reason, bot, source='paper'):
                return _real_db.mark_unreconciled_order_abandoned(
                    order_id, reason, bot, source=source, db_path=db_path)

        fill_prices = {'sell-1': 50.0}
        with patch.object(tracker, 'get_fill_price', side_effect=lambda oid: fill_prices.get(oid)), \
             patch.object(tracker, 'get_order_status', return_value={'status': 'unknown'}), \
             patch('outcome_tracker._fetch_spy_return', return_value=None), \
             patch('outcome_tracker._db', _DBWithPath()):
            trades = tracker.load_raw_trades()
            tracker.match_and_calculate_pnl(trades)
            tracker.match_and_calculate_pnl(trades)

        with _real_db.get_conn(db_path) as conn:
            rows = conn.execute(
                "SELECT retry_count FROM unreconciled_orders WHERE order_id='buy-1'"
            ).fetchall()
        assert len(rows) == 1
        assert rows[0]['retry_count'] == 1
