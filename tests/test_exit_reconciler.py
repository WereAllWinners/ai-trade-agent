"""
tests/test_exit_reconciler.py — R1 WI-4/WI-5

Covers:
  - reconcile_pending_exits: submit->filled, submit->expired (zero fill),
    partial-fill convention, still-open stays pending, API-error retry,
    abandon after max retries, restart-with-pending (DB persistence across
    a fresh call), bot='stock' triggers insert_trade / bot='options' doesn't
  - The exact regression scenario reconstructed from the 2026-07 incident:
    NVDA order submitted, expires unfilled -> no false profit record
  - reconcile_broker_fills: matched fill (no duplicate), unmatched orphan
    fill (reconcile row + CRITICAL alert), partial fill handling
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))

import db as _db  # noqa: E402
from exit_reconciler import reconcile_broker_fills, reconcile_pending_exits  # noqa: E402


@pytest.fixture()
def db_path(tmp_path):
    p = tmp_path / 'test_trading.db'
    _db.init_db(p)
    return p


def _seed_pending(db_path, order_id='order-1', bot='options', source='paper',
                   symbol='NVDA260710C00200000', intended_qty=1.0,
                   intended_reason='take_profit_50%', avg_entry_price=2.52):
    _db.insert_pending_exit(
        {'order_id': order_id, 'symbol': symbol, 'intended_qty': intended_qty,
         'intended_reason': intended_reason, 'avg_entry_price': avg_entry_price,
         'time_in_force': 'DAY', 'submitted_at': '2026-07-01T14:06:51'},
        bot=bot, source=source, db_path=db_path,
    )


def _mock_order(status, filled_qty=0, filled_avg_price=None, symbol='NVDA260710C00200000', side='sell'):
    o = MagicMock()
    o.status = MagicMock(value=status)
    o.filled_qty = str(filled_qty) if filled_qty else None
    o.filled_avg_price = str(filled_avg_price) if filled_avg_price is not None else None
    o.symbol = symbol
    o.side = MagicMock(value=side)
    return o


class TestReconcilePendingExitsFilled:
    def test_submit_then_filled_writes_real_record(self, db_path, tmp_path):
        _seed_pending(db_path, order_id='o-filled', avg_entry_price=2.52)
        client = MagicMock()
        client.get_order_by_id.return_value = _mock_order('filled', filled_qty=1, filled_avg_price=4.40)
        trade_log = tmp_path / 'options_trade_log.jsonl'

        with patch('exit_reconciler._TRADE_LOG_PATHS', {'options': trade_log, 'stock': tmp_path / 'x.jsonl'}), \
             patch('exit_reconciler.alert_trade_executed') as mock_alert:
            stats = reconcile_pending_exits(client, bot='options', source='paper', db_path=db_path)

        assert stats == {'checked': 1, 'filled': 1, 'exit_failed': 0, 'still_pending': 0, 'errors': 0}
        mock_alert.assert_called_once()
        args = mock_alert.call_args[0]
        assert args[2] == 'sell'
        assert args[3] == 1.0  # filled_qty
        assert args[4] == 4.40  # avg_price

        rows = [json.loads(l) for l in trade_log.read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]['action'] == 'sell'
        assert rows[0]['order_id'] == 'o-filled'

        assert _db.get_pending_exits('options', 'paper', 'pending', db_path=db_path) == []
        assert len(_db.get_pending_exits('options', 'paper', 'filled', db_path=db_path)) == 1

    def test_stock_fill_calls_insert_trade_options_does_not(self, db_path, tmp_path):
        for bot in ('stock', 'options'):
            order_id = f'o-{bot}'
            _seed_pending(db_path, order_id=order_id, bot=bot)
            client = MagicMock()
            client.get_order_by_id.return_value = _mock_order('filled', filled_qty=1, filled_avg_price=4.40)
            with patch('exit_reconciler._TRADE_LOG_PATHS', {'options': tmp_path / 'o.jsonl', 'stock': tmp_path / 's.jsonl'}), \
                 patch('exit_reconciler.alert_trade_executed'), \
                 patch('exit_reconciler._db.insert_trade') as mock_insert_trade:
                reconcile_pending_exits(client, bot=bot, source='paper', db_path=db_path)
            if bot == 'stock':
                mock_insert_trade.assert_called_once()
                assert mock_insert_trade.call_args[1]['bot'] == 'stock'
            else:
                mock_insert_trade.assert_not_called()


class TestReconcilePendingExitsFailed:
    def test_submit_then_expired_zero_fill_is_exit_failed(self, db_path, tmp_path):
        """The exact NVDA regression shape: order expires with filled_qty=0."""
        _seed_pending(db_path, order_id='4d903428-5cca-4fdd-8f63-c3c275ebd06f',
                      symbol='NVDA260710C00200000', intended_reason='take_profit_50%')
        client = MagicMock()
        client.get_order_by_id.return_value = _mock_order('expired', filled_qty=0)
        trade_log = tmp_path / 'options_trade_log.jsonl'

        with patch('exit_reconciler._TRADE_LOG_PATHS', {'options': trade_log, 'stock': tmp_path / 'x.jsonl'}), \
             patch('exit_reconciler.alert_trade_executed') as mock_trade_alert, \
             patch('exit_reconciler.alert_exit_failed') as mock_exit_failed_alert:
            stats = reconcile_pending_exits(client, bot='options', source='paper', db_path=db_path)

        assert stats['exit_failed'] == 1
        assert stats['filled'] == 0

        # No false "closed at a profit" record -- the exact defect being fixed.
        mock_trade_alert.assert_not_called()
        mock_exit_failed_alert.assert_called_once()

        rows = [json.loads(l) for l in trade_log.read_text().splitlines()]
        assert len(rows) == 1
        assert 'action' not in rows[0]  # one-rule policy: nothing filled, nothing to claim
        assert rows[0]['event'] == 'exit_failed'
        assert 'realized_pnl' not in rows[0]

        assert len(_db.get_pending_exits('options', 'paper', 'exit_failed', db_path=db_path)) == 1

    def test_canceled_zero_fill_also_exit_failed(self, db_path, tmp_path):
        _seed_pending(db_path, order_id='o-canceled')
        client = MagicMock()
        client.get_order_by_id.return_value = _mock_order('canceled', filled_qty=0)
        with patch('exit_reconciler._TRADE_LOG_PATHS', {'options': tmp_path / 'o.jsonl', 'stock': tmp_path / 's.jsonl'}), \
             patch('exit_reconciler.alert_exit_failed') as mock_alert:
            stats = reconcile_pending_exits(client, bot='options', source='paper', db_path=db_path)
        assert stats['exit_failed'] == 1
        mock_alert.assert_called_once()

    def test_rejected_zero_fill_also_exit_failed(self, db_path, tmp_path):
        _seed_pending(db_path, order_id='o-rejected')
        client = MagicMock()
        client.get_order_by_id.return_value = _mock_order('rejected', filled_qty=0)
        with patch('exit_reconciler._TRADE_LOG_PATHS', {'options': tmp_path / 'o.jsonl', 'stock': tmp_path / 's.jsonl'}):
            stats = reconcile_pending_exits(client, bot='options', source='paper', db_path=db_path)
        assert stats['exit_failed'] == 1


class TestPartialFillConvention:
    def test_partial_fill_one_status_two_jsonl_rows(self, db_path, tmp_path):
        """One row, one status ('filled') -- real sell for the filled portion
        AND an exit-failed row for the remainder, per the reviewed convention."""
        _seed_pending(db_path, order_id='o-partial', intended_qty=3.0)
        client = MagicMock()
        # 2 of 3 intended contracts filled, order then expired
        client.get_order_by_id.return_value = _mock_order('expired', filled_qty=2, filled_avg_price=3.00)
        trade_log = tmp_path / 'options_trade_log.jsonl'

        with patch('exit_reconciler._TRADE_LOG_PATHS', {'options': trade_log, 'stock': tmp_path / 'x.jsonl'}), \
             patch('exit_reconciler.alert_trade_executed'), \
             patch('exit_reconciler.alert_exit_failed'):
            stats = reconcile_pending_exits(client, bot='options', source='paper', db_path=db_path)

        assert stats['filled'] == 1
        assert stats['exit_failed'] == 0  # counted under 'filled' per the convention, not double-counted

        # DB: exactly one resolution, status='filled', not a hybrid status, not still pending
        assert _db.get_pending_exits('options', 'paper', 'pending', db_path=db_path) == []
        filled_rows = _db.get_pending_exits('options', 'paper', 'filled', db_path=db_path)
        assert len(filled_rows) == 1

        # JSONL: two records -- one real sell (qty=2), one exit_failed (remainder=1)
        rows = [json.loads(l) for l in trade_log.read_text().splitlines()]
        assert len(rows) == 2
        sell_rows = [r for r in rows if r.get('action') == 'sell']
        failed_rows = [r for r in rows if 'action' not in r]
        assert len(sell_rows) == 1
        assert sell_rows[0]['quantity'] == 2.0
        assert len(failed_rows) == 1
        assert failed_rows[0]['unfilled_qty'] == 1.0

    def test_partial_fill_stock_insert_trade_uses_actual_filled_qty(self, db_path, tmp_path):
        _seed_pending(db_path, order_id='o-partial-stock', bot='stock', intended_qty=10.0)
        client = MagicMock()
        client.get_order_by_id.return_value = _mock_order('canceled', filled_qty=6, filled_avg_price=50.0)
        with patch('exit_reconciler._TRADE_LOG_PATHS', {'options': tmp_path / 'o.jsonl', 'stock': tmp_path / 's.jsonl'}), \
             patch('exit_reconciler.alert_trade_executed'), \
             patch('exit_reconciler.alert_exit_failed'), \
             patch('exit_reconciler._db.insert_trade') as mock_insert_trade:
            reconcile_pending_exits(client, bot='stock', source='paper', db_path=db_path)
        mock_insert_trade.assert_called_once()
        assert mock_insert_trade.call_args[0][0]['shares'] == 6.0  # actual, not intended 10.0


class TestStillOpenAndRetries:
    def test_still_resting_order_stays_pending(self, db_path, tmp_path):
        _seed_pending(db_path, order_id='o-resting')
        client = MagicMock()
        client.get_order_by_id.return_value = _mock_order('new', filled_qty=0)
        with patch('exit_reconciler._TRADE_LOG_PATHS', {'options': tmp_path / 'o.jsonl', 'stock': tmp_path / 's.jsonl'}):
            stats = reconcile_pending_exits(client, bot='options', source='paper', db_path=db_path)
        assert stats['still_pending'] == 1
        assert len(_db.get_pending_exits('options', 'paper', 'pending', db_path=db_path)) == 1

    def test_api_error_bumps_retry_stays_pending(self, db_path, tmp_path):
        _seed_pending(db_path, order_id='o-error')
        client = MagicMock()
        client.get_order_by_id.side_effect = Exception('Alpaca API unavailable')
        with patch('exit_reconciler._TRADE_LOG_PATHS', {'options': tmp_path / 'o.jsonl', 'stock': tmp_path / 's.jsonl'}):
            stats = reconcile_pending_exits(client, bot='options', source='paper', db_path=db_path)
        assert stats['errors'] == 1
        rows = _db.get_pending_exits('options', 'paper', 'pending', db_path=db_path)
        assert len(rows) == 1
        assert rows[0]['retry_count'] == 1

    def test_abandoned_after_max_retries(self, db_path, tmp_path):
        _seed_pending(db_path, order_id='o-abandon')
        client = MagicMock()
        client.get_order_by_id.side_effect = Exception('down')
        with patch('exit_reconciler._TRADE_LOG_PATHS', {'options': tmp_path / 'o.jsonl', 'stock': tmp_path / 's.jsonl'}), \
             patch('exit_reconciler._MAX_RECONCILE_RETRIES', 2):
            for _ in range(4):
                reconcile_pending_exits(client, bot='options', source='paper', db_path=db_path)
        assert _db.get_pending_exits('options', 'paper', 'pending', db_path=db_path) == []
        assert len(_db.get_pending_exits('options', 'paper', 'abandoned', db_path=db_path)) == 1


class TestRestartWithPending:
    def test_pending_state_survives_across_separate_calls(self, db_path, tmp_path):
        """Simulates the real architecture: each session is a brand-new
        process. Seeding via one call and resolving via a completely
        separate, later call proves the DB (not memory) is what carries
        state across that boundary."""
        _seed_pending(db_path, order_id='o-restart')
        # "session 1" ends here -- nothing in memory carries forward.

        # "session 2": fresh call, same db_path, must still find the row.
        client = MagicMock()
        client.get_order_by_id.return_value = _mock_order('filled', filled_qty=1, filled_avg_price=5.0)
        with patch('exit_reconciler._TRADE_LOG_PATHS', {'options': tmp_path / 'o.jsonl', 'stock': tmp_path / 's.jsonl'}), \
             patch('exit_reconciler.alert_trade_executed'):
            stats = reconcile_pending_exits(client, bot='options', source='paper', db_path=db_path)
        assert stats['filled'] == 1


class TestReconcileBrokerFills:
    def _mock_closed_order(self, order_id, symbol, filled_qty, filled_avg_price, side='sell'):
        o = MagicMock()
        o.id = order_id
        o.symbol = symbol
        o.filled_qty = str(filled_qty)
        o.filled_avg_price = str(filled_avg_price)
        o.side = MagicMock(value=side)
        return o

    def test_matched_fill_no_duplicate(self, db_path, tmp_path):
        trade_log = tmp_path / 'options_trade_log.jsonl'
        trade_log.write_text(json.dumps({'order_id': 'known-1', 'action': 'sell'}) + '\n')
        client = MagicMock()
        known_order = self._mock_closed_order('known-1', 'MSFT260710C00382500', 1, 6.05)

        with patch('sync_broker_exits._fetch_closed_orders', return_value=[known_order]), \
             patch('exit_reconciler.alert_trade_executed') as mock_alert, \
             patch('exit_reconciler.alert_broker_fill_unaccounted') as mock_critical:
            stats = reconcile_broker_fills(client, source='paper', db_path=db_path, trade_log_path=trade_log)

        assert stats['orphans_found'] == 0
        mock_alert.assert_not_called()
        mock_critical.assert_not_called()

    def test_unmatched_orphan_fill_reconciled(self, db_path, tmp_path):
        """The exact MSFT shape: a real broker fill with no app record at all."""
        trade_log = tmp_path / 'options_trade_log.jsonl'
        trade_log.write_text('')
        client = MagicMock()
        orphan = self._mock_closed_order('e83926ab-52fc-4ab3-b5cc-3956b34847e4',
                                          'MSFT260710C00382500', 1, 6.05)

        with patch('sync_broker_exits._fetch_closed_orders', return_value=[orphan]), \
             patch('exit_reconciler.alert_trade_executed') as mock_alert, \
             patch('exit_reconciler.alert_broker_fill_unaccounted') as mock_critical:
            stats = reconcile_broker_fills(client, source='paper', db_path=db_path, trade_log_path=trade_log)

        assert stats['orphans_found'] == 1
        mock_alert.assert_called_once()
        mock_critical.assert_called_once()

        rows = [json.loads(l) for l in trade_log.read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]['action'] == 'sell'  # real fill -- per the one-rule policy
        assert rows[0]['source_tag'] == 'broker_reconcile'
        assert rows[0]['order_id'] == 'e83926ab-52fc-4ab3-b5cc-3956b34847e4'

    def test_stock_symbols_skipped(self, db_path, tmp_path):
        trade_log = tmp_path / 'options_trade_log.jsonl'
        trade_log.write_text('')
        client = MagicMock()
        stock_order = self._mock_closed_order('stock-1', 'AAPL', 10, 200.0)
        with patch('sync_broker_exits._fetch_closed_orders', return_value=[stock_order]):
            stats = reconcile_broker_fills(client, source='paper', db_path=db_path, trade_log_path=trade_log)
        assert stats['scanned'] == 0
        assert stats['orphans_found'] == 0

    def test_buy_side_orders_skipped(self, db_path, tmp_path):
        trade_log = tmp_path / 'options_trade_log.jsonl'
        trade_log.write_text('')
        client = MagicMock()
        buy_order = self._mock_closed_order('buy-1', 'NVDA260710C00200000', 1, 2.52, side='buy')
        with patch('sync_broker_exits._fetch_closed_orders', return_value=[buy_order]):
            stats = reconcile_broker_fills(client, source='paper', db_path=db_path, trade_log_path=trade_log)
        assert stats['scanned'] == 0

    def test_partial_fill_orphan_still_reconciled(self, db_path, tmp_path):
        """A partially-filled broker order with no app record is still a
        real, reportable fill for whatever quantity did fill."""
        trade_log = tmp_path / 'options_trade_log.jsonl'
        trade_log.write_text('')
        client = MagicMock()
        partial_orphan = self._mock_closed_order('partial-orphan', 'SPY250328C00560000', 1, 3.5)
        with patch('sync_broker_exits._fetch_closed_orders', return_value=[partial_orphan]), \
             patch('exit_reconciler.alert_trade_executed'), \
             patch('exit_reconciler.alert_broker_fill_unaccounted'):
            stats = reconcile_broker_fills(client, source='paper', db_path=db_path, trade_log_path=trade_log)
        assert stats['orphans_found'] == 1
        rows = [json.loads(l) for l in trade_log.read_text().splitlines()]
        assert rows[0]['quantity'] == 1.0

    def test_known_pending_exit_order_id_not_treated_as_orphan(self, db_path, tmp_path):
        """An order that went through pending_exits (even if not yet in the
        JSONL, e.g. this same reconcile pass already handled it) must not
        also be flagged as an orphan."""
        _seed_pending(db_path, order_id='already-pending-1')
        trade_log = tmp_path / 'options_trade_log.jsonl'
        trade_log.write_text('')
        client = MagicMock()
        same_order = self._mock_closed_order('already-pending-1', 'NVDA260710C00200000', 1, 4.40)
        with patch('sync_broker_exits._fetch_closed_orders', return_value=[same_order]):
            stats = reconcile_broker_fills(client, source='paper', db_path=db_path, trade_log_path=trade_log)
        assert stats['orphans_found'] == 0
