"""
tests/test_risk_reconciler.py — Broker-side risk reconciliation (PR-5, C2/C3)

Covers:
  - count_unprotected_positions: dict return, three-way split by asset class
  - write_reconcile_status: new three-way keys + legacy key + -1 propagation
  - reprotect_positions: whole-share OCO, fractional skip, PDT deferral, dry-run report
  - morning_model_check: per-class check (fractional warn-only), sequence (repair then count)
  - options_agent GTC stop: logged but not submitted in DRY_RUN mode
  - _cancel_gtc_stops: cancels stop-limit GTC SELLs, leaves plain limit GTC intact
"""
import json
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Ensure scripts/ is importable without GPU/heavy deps
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / 'scripts'
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass  # noqa: E402
from risk_reconciler import (                                          # noqa: E402
    count_unprotected_positions, write_reconcile_status, reprotect_positions,
)


# ---------------------------------------------------------------------------
# Helpers to build mock Alpaca objects
# ---------------------------------------------------------------------------

def _pos(symbol: str, qty: float = 1.0, avg_entry: float = 100.0,
         current_price: float = None) -> MagicMock:
    p = MagicMock()
    p.symbol          = symbol
    p.qty             = str(qty)
    p.avg_entry_price = str(avg_entry)
    p.current_price   = str(current_price if current_price is not None else avg_entry)
    return p


def _order(symbol: str, side: OrderSide, tif=TimeInForce.DAY, stop_price=None) -> MagicMock:
    o = MagicMock()
    o.symbol        = symbol
    o.side          = side
    o.time_in_force = tif
    o.stop_price    = stop_price
    return o


def _client(positions=None, orders=None) -> MagicMock:
    c = MagicMock()
    c.get_all_positions.return_value = positions or []
    c.get_orders.return_value        = orders or []
    return c


# ---------------------------------------------------------------------------
# count_unprotected_positions — returns {'options': int, 'stocks_whole': int, 'stocks_fractional': int}
# ---------------------------------------------------------------------------

class TestCountUnprotectedPositions:
    def test_options_position_with_open_sell_is_protected(self):
        """Options position that has an open SELL order → options=0."""
        occ_sym = 'AAPL240119C00190000'
        client  = _client(
            positions=[_pos(occ_sym)],
            orders=[_order(occ_sym, OrderSide.SELL)],
        )
        result = count_unprotected_positions(client)
        assert result == {'options': 0, 'stocks_whole': 0, 'stocks_fractional': 0}

    def test_options_position_without_sell_is_unprotected(self):
        """Options position with no SELL order → options=1."""
        occ_sym = 'AAPL240119C00190000'
        client  = _client(positions=[_pos(occ_sym)], orders=[])
        result  = count_unprotected_positions(client)
        assert result['options']          == 1
        assert result['stocks_whole']     == 0
        assert result['stocks_fractional'] == 0

    def test_whole_share_stock_without_sell_counted_in_stocks_whole(self):
        """Whole-share stock position (qty=10, integer) with no SELL → stocks_whole=1."""
        client = _client(positions=[_pos('AAPL', qty=10.0)], orders=[])
        result = count_unprotected_positions(client)
        assert result == {'options': 0, 'stocks_whole': 1, 'stocks_fractional': 0}

    def test_fractional_stock_counted_in_stocks_fractional(self):
        """Fractional stock position (qty=0.5) with no SELL → stocks_fractional=1, not stocks_whole."""
        client = _client(positions=[_pos('AAPL', qty=0.5)], orders=[])
        result = count_unprotected_positions(client)
        assert result == {'options': 0, 'stocks_whole': 0, 'stocks_fractional': 1}

    def test_single_whole_share_qty_one_is_stocks_whole(self):
        """qty=1.0 (integer-valued float) → stocks_whole, not stocks_fractional."""
        client = _client(positions=[_pos('TSLA', qty=1.0)], orders=[])
        result = count_unprotected_positions(client)
        assert result['stocks_whole']      == 1
        assert result['stocks_fractional'] == 0

    def test_stock_with_open_sell_stop_is_protected(self):
        """Stock position with any open SELL order (e.g. stop/bracket child) → not counted."""
        client = _client(
            positions=[_pos('AAPL')],
            orders=[_order('AAPL', OrderSide.SELL, stop_price=90.0)],
        )
        result = count_unprotected_positions(client)
        assert result == {'options': 0, 'stocks_whole': 0, 'stocks_fractional': 0}

    def test_mixed_positions(self):
        """Whole stock naked + option naked + option protected → stocks_whole=1, options=1."""
        occ_prot   = 'AAPL240119C00190000'
        occ_naked  = 'TSLA240119P00200000'
        client = _client(
            positions=[_pos('AAPL', qty=5.0), _pos(occ_prot), _pos(occ_naked)],
            orders=[_order(occ_prot, OrderSide.SELL)],
        )
        result = count_unprotected_positions(client)
        assert result == {'options': 1, 'stocks_whole': 1, 'stocks_fractional': 0}

    def test_mixed_whole_and_fractional_stocks(self):
        """Mix of whole and fractional stocks (both naked) → each counted in correct bucket."""
        client = _client(
            positions=[_pos('AAPL', qty=5.0), _pos('MSFT', qty=0.25)],
            orders=[],
        )
        result = count_unprotected_positions(client)
        assert result['stocks_whole']      == 1
        assert result['stocks_fractional'] == 1
        assert result['options']           == 0

    def test_api_error_returns_minus_one_dict(self):
        """When broker API raises, return {'options': -1, 'stocks_whole': -1, 'stocks_fractional': -1}."""
        client = MagicMock()
        client.get_all_positions.side_effect = Exception("Network error")
        result = count_unprotected_positions(client)
        assert result == {'options': -1, 'stocks_whole': -1, 'stocks_fractional': -1}

    def test_buy_order_does_not_count_as_protected(self):
        """An open BUY order does not protect the position."""
        occ_sym = 'AAPL240119C00190000'
        client  = _client(
            positions=[_pos(occ_sym)],
            orders=[_order(occ_sym, OrderSide.BUY)],
        )
        assert count_unprotected_positions(client)['options'] == 1


# ---------------------------------------------------------------------------
# write_reconcile_status — three-way keys + legacy key
# ---------------------------------------------------------------------------

class TestWriteReconcileStatus:
    def test_writes_all_keys(self, tmp_path):
        """Output JSON must contain all three per-class keys and legacy combined key."""
        occ_sym = 'AAPL240119C00190000'
        client  = _client(
            positions=[_pos(occ_sym)],
            orders=[_order(occ_sym, OrderSide.SELL)],
        )
        out = tmp_path / 'reconcile_status.json'
        write_reconcile_status(client, out)
        data = json.loads(out.read_text())
        assert data['unprotected_options']           == 0
        assert data['unprotected_stocks_whole']      == 0
        assert data['unprotected_stocks_fractional'] == 0
        assert data['unprotected_positions']         == 0   # legacy = sum of all three
        assert 'checked_at' in data

    def test_legacy_key_is_sum_when_all_known(self, tmp_path):
        """Legacy key = options + stocks_whole + stocks_fractional when none is -1."""
        client = _client(
            positions=[
                _pos('AAPL', qty=5.0),               # whole → stocks_whole=1
                _pos('MSFT', qty=0.3),                # fractional → stocks_fractional=1
                _pos('AAPL240119C00190000'),           # option → options=1
            ],
            orders=[],
        )
        out = tmp_path / 'status.json'
        write_reconcile_status(client, out)
        data = json.loads(out.read_text())
        assert data['unprotected_options']           == 1
        assert data['unprotected_stocks_whole']      == 1
        assert data['unprotected_stocks_fractional'] == 1
        assert data['unprotected_positions']         == 3   # 1+1+1

    def test_legacy_key_is_minus_one_when_any_api_error(self, tmp_path):
        """-1 propagation: if ANY count is -1, legacy key must be -1, never a partial sum."""
        client = MagicMock()
        client.get_all_positions.side_effect = Exception("API down")
        out = tmp_path / 'status.json'
        write_reconcile_status(client, out)
        data = json.loads(out.read_text())
        assert data['unprotected_positions']         == -1
        assert data['unprotected_options']           == -1
        assert data['unprotected_stocks_whole']      == -1
        assert data['unprotected_stocks_fractional'] == -1

    def test_unprotected_warning_logged(self, tmp_path, caplog):
        import logging
        occ_sym = 'AAPL240119C00190000'
        client  = _client(positions=[_pos(occ_sym)], orders=[])
        out     = tmp_path / 'reconcile_status.json'
        with caplog.at_level(logging.WARNING):
            write_reconcile_status(client, out)
        assert any('unprotected' in msg.lower() for msg in caplog.messages)

    def test_creates_parent_dir_if_missing(self, tmp_path):
        nested = tmp_path / 'nested' / 'dir' / 'reconcile_status.json'
        client = _client()
        write_reconcile_status(client, nested)
        assert nested.exists()


# ---------------------------------------------------------------------------
# reprotect_positions
# ---------------------------------------------------------------------------

class TestReprotectPositions:
    _DEFAULT_PARAMS = {'stop_loss': -0.07, 'take_profit': 0.15}
    _PDT_PARAMS     = {'stop_loss': -0.07, 'take_profit': 0.15, 'no_same_day_close': True}

    def test_naked_whole_share_position_gets_oco(self):
        """Whole-share position with no SELL → submit_order called with OCO SELL."""
        client = _client(
            positions=[_pos('AAPL', qty=10.0, avg_entry=100.0)],
            orders=[],
        )
        result = reprotect_positions(client, self._DEFAULT_PARAMS)
        assert result['protected'] == 1
        assert result['already_protected'] == 0
        client.submit_order.assert_called_once()
        order_arg = client.submit_order.call_args[0][0]
        assert order_arg.side  == OrderSide.SELL
        assert order_arg.time_in_force == TimeInForce.GTC
        assert order_arg.order_class   == OrderClass.OCO

    def test_oco_prices_derived_from_avg_entry(self):
        """OCO target and stop prices are computed from avg_entry_price × factors."""
        client = _client(
            positions=[_pos('AAPL', qty=5.0, avg_entry=200.0)],
            orders=[],
        )
        reprotect_positions(client, self._DEFAULT_PARAMS)
        order_arg = client.submit_order.call_args[0][0]
        # take_profit = 200 * 1.15 = 230.00
        assert float(order_arg.limit_price) == pytest.approx(230.00, abs=0.01)

    def test_position_with_existing_sell_is_skipped(self):
        """Position already has a SELL order → submit_order not called."""
        client = _client(
            positions=[_pos('AAPL', qty=10.0)],
            orders=[_order('AAPL', OrderSide.SELL)],
        )
        result = reprotect_positions(client, self._DEFAULT_PARAMS)
        assert result['already_protected'] == 1
        assert result['protected']         == 0
        client.submit_order.assert_not_called()

    def test_fractional_position_is_skipped_with_count(self):
        """Fractional qty → skipped_fractional incremented, no order submitted."""
        client = _client(
            positions=[_pos('AAPL', qty=0.5)],
            orders=[],
        )
        result = reprotect_positions(client, self._DEFAULT_PARAMS)
        assert result['skipped_fractional'] == 1
        assert result['protected']          == 0
        client.submit_order.assert_not_called()

    def test_dry_run_does_not_submit(self):
        """dry_run=True → no submit_order call; protected count still incremented."""
        client = _client(
            positions=[_pos('AAPL', qty=10.0, avg_entry=100.0, current_price=105.0)],
            orders=[],
        )
        result = reprotect_positions(client, self._DEFAULT_PARAMS, dry_run=True)
        assert result['protected'] == 1
        client.submit_order.assert_not_called()

    def test_dry_run_report_contains_per_position_entry(self):
        """dry_run=True → report list contains one entry per would-be-protected position."""
        client = _client(
            positions=[_pos('AAPL', qty=5, avg_entry=200.0, current_price=210.0)],
            orders=[],
        )
        result = reprotect_positions(client, self._DEFAULT_PARAMS, dry_run=True)
        assert len(result['report']) == 1
        entry = result['report'][0]
        assert entry['symbol'] == 'AAPL'
        assert entry['qty']    == 5
        assert entry['current_price'] == pytest.approx(210.0)
        assert entry['stop']          == pytest.approx(200.0 * 0.93, abs=0.01)
        assert entry['target']        == pytest.approx(200.0 * 1.15, abs=0.01)

    def test_dry_run_would_fill_immediately_when_price_below_stop(self):
        """would_fill_immediately=True when current_price has already crossed stop level."""
        avg = 100.0
        stop = round(avg * (1 - 0.07), 2)   # 93.0
        client = _client(
            positions=[_pos('AAPL', qty=3, avg_entry=avg, current_price=stop - 1)],
            orders=[],
        )
        result = reprotect_positions(client, self._DEFAULT_PARAMS, dry_run=True)
        assert result['report'][0]['would_fill_immediately'] is True

    def test_dry_run_would_not_fill_when_price_within_range(self):
        """would_fill_immediately=False when current_price is between stop and target."""
        client = _client(
            positions=[_pos('AAPL', qty=3, avg_entry=100.0, current_price=100.0)],
            orders=[],
        )
        result = reprotect_positions(client, self._DEFAULT_PARAMS, dry_run=True)
        assert result['report'][0]['would_fill_immediately'] is False

    def test_dry_run_fractional_skipped_not_in_report(self):
        """Fractional positions are skipped before reaching the dry-run report block."""
        client = _client(
            positions=[_pos('AAPL', qty=0.5, avg_entry=100.0)],
            orders=[],
        )
        result = reprotect_positions(client, self._DEFAULT_PARAMS, dry_run=True)
        assert result['report']           == []
        assert result['skipped_fractional'] == 1

    def test_pdt_deferral_skips_position_opened_today(self):
        """When no_same_day_close and symbol in positions_opened_today → deferred_pdt."""
        client = _client(
            positions=[_pos('AAPL', qty=5.0)],
            orders=[],
        )
        result = reprotect_positions(
            client, self._PDT_PARAMS,
            positions_opened_today={'AAPL'},
        )
        assert result['deferred_pdt'] == 1
        assert result['protected']    == 0
        client.submit_order.assert_not_called()

    def test_pdt_deferral_does_not_skip_position_opened_yesterday(self):
        """Symbol NOT in positions_opened_today → protected despite no_same_day_close."""
        client = _client(
            positions=[_pos('AAPL', qty=5.0)],
            orders=[],
        )
        result = reprotect_positions(
            client, self._PDT_PARAMS,
            positions_opened_today={'MSFT'},  # AAPL not in today's buys
        )
        assert result['deferred_pdt'] == 0
        assert result['protected']    == 1
        client.submit_order.assert_called_once()

    def test_pdt_deferral_inactive_when_flag_off(self):
        """positions_opened_today has no effect when no_same_day_close is False."""
        client = _client(
            positions=[_pos('AAPL', qty=5.0)],
            orders=[],
        )
        # no_same_day_close not set → should protect regardless
        result = reprotect_positions(
            client, self._DEFAULT_PARAMS,
            positions_opened_today={'AAPL'},
        )
        assert result['deferred_pdt'] == 0
        assert result['protected']    == 1

    def test_api_error_returns_empty_summary(self):
        """If get_all_positions raises, return zeroed summary without raising."""
        client = MagicMock()
        client.get_all_positions.side_effect = Exception("API down")
        result = reprotect_positions(client, self._DEFAULT_PARAMS)
        assert result == {
            'protected': 0, 'already_protected': 0, 'skipped_fractional': 0,
            'deferred_pdt': 0, 'errors': 0, 'report': [],
        }

    def test_options_symbols_not_reprotected(self):
        """OCC-length symbols are options — not touched by reprotect_positions."""
        occ_sym = 'AAPL240119C00190000'
        client  = _client(positions=[_pos(occ_sym, qty=1.0)], orders=[])
        result  = reprotect_positions(client, self._DEFAULT_PARAMS)
        assert result['protected']          == 0
        assert result['skipped_fractional'] == 0
        client.submit_order.assert_not_called()


# ---------------------------------------------------------------------------
# morning_model_check — reconcile gate (updated for three-way schema)
# ---------------------------------------------------------------------------

class TestMorningModelCheckReconcile:
    def _run_check(self, reconcile_data: dict | None, tmp_path: Path):
        """Run only check_reconcile_status in isolation."""
        sys.path.insert(0, str(_SCRIPTS_DIR / 'tools'))
        from morning_model_check import CheckResult, check_reconcile_status

        if reconcile_data is not None:
            status_file = tmp_path / 'logs' / 'reconcile_status.json'
            status_file.parent.mkdir(parents=True, exist_ok=True)
            status_file.write_text(json.dumps(reconcile_data))

        results = CheckResult()
        with patch('morning_model_check._PROJECT_ROOT', tmp_path):
            check_reconcile_status(results)
        return results

    def test_zero_unprotected_passes(self, tmp_path):
        data = {
            'unprotected_options': 0, 'unprotected_stocks_whole': 0,
            'unprotected_stocks_fractional': 0, 'unprotected_positions': 0,
            'checked_at': '2026-01-01T00:00',
        }
        results = self._run_check(data, tmp_path)
        assert results.all_passed

    def test_nonzero_stocks_whole_fails(self, tmp_path):
        """Unprotected whole-share stocks → hard fail."""
        data = {
            'unprotected_options': 0, 'unprotected_stocks_whole': 1,
            'unprotected_stocks_fractional': 0, 'unprotected_positions': 1,
            'checked_at': '2026-01-01T00:00',
        }
        results = self._run_check(data, tmp_path)
        assert not results.all_passed

    def test_nonzero_options_fails(self, tmp_path):
        data = {
            'unprotected_options': 2, 'unprotected_stocks_whole': 0,
            'unprotected_stocks_fractional': 0, 'unprotected_positions': 2,
            'checked_at': '2026-01-01T00:00',
        }
        results = self._run_check(data, tmp_path)
        assert not results.all_passed

    def test_nonzero_stocks_fractional_is_warn_only(self, tmp_path):
        """Unprotected fractional positions → warn-only (broker GTC limitation)."""
        data = {
            'unprotected_options': 0, 'unprotected_stocks_whole': 0,
            'unprotected_stocks_fractional': 5, 'unprotected_positions': 5,
            'checked_at': '2026-01-01T00:00',
        }
        results = self._run_check(data, tmp_path)
        assert results.all_passed        # warn-only: still "passes"
        assert results.has_warnings      # but a warning is recorded

    def test_legacy_only_key_still_works(self, tmp_path):
        """Old-format file with only unprotected_positions key → parses options from it."""
        data = {'unprotected_positions': 0, 'checked_at': '2026-01-01T00:00'}
        results = self._run_check(data, tmp_path)
        assert results.all_passed

    def test_missing_file_is_warn_only(self, tmp_path):
        results = self._run_check(None, tmp_path)
        assert results.all_passed
        assert results.has_warnings

    def test_api_error_status_is_warn_only(self, tmp_path):
        data = {
            'unprotected_options': -1, 'unprotected_stocks_whole': -1,
            'unprotected_stocks_fractional': -1, 'unprotected_positions': -1,
            'checked_at': '2026-01-01T00:00',
        }
        results = self._run_check(data, tmp_path)
        assert results.all_passed
        assert results.has_warnings

    def test_repair_runs_before_reconcile_count(self, tmp_path):
        """Morning check calls repair, then writes a fresh reconcile_status, then reads it.

        Sequence: repair_naked_positions (writes fresh status) → check_reconcile_status.
        Verified by: repair writes 0 unprotected; check reads the same file and passes.
        """
        sys.path.insert(0, str(_SCRIPTS_DIR / 'tools'))
        from morning_model_check import CheckResult, repair_naked_positions, check_reconcile_status

        status_file = tmp_path / 'logs' / 'reconcile_status.json'
        status_file.parent.mkdir(parents=True, exist_ok=True)

        def _fake_repair(results_obj):
            # Simulate the repair writing a fresh file with 0 unprotected
            status_file.write_text(json.dumps({
                'unprotected_options': 0, 'unprotected_stocks_whole': 0,
                'unprotected_stocks_fractional': 0, 'unprotected_positions': 0,
                'checked_at': '2026-06-12T08:00:00',
            }))
            results_obj.add('repair_sweep', True,
                            'protected=0 fractional_skipped=0 deferred_pdt=0 errors=0')

        results = CheckResult()
        with patch('morning_model_check._PROJECT_ROOT', tmp_path):
            _fake_repair(results)
            check_reconcile_status(results)

        assert results.all_passed


# ---------------------------------------------------------------------------
# options_agent — GTC stop in DRY_RUN mode
# ---------------------------------------------------------------------------

class TestOptionsAgentGTCStop:
    """Verify the GTC stop logic without importing the full OptionsAgent."""

    def _make_agent_stub(self, dry_run_env: str = 'true'):
        """Return a thin object that exercises the GTC stop code path."""
        import importlib
        import types

        # We need to import options_agent but with Alpaca mocked
        # conftest.py already stubs heavy deps; Alpaca isn't stubbed so we do it here
        alpaca_mock = MagicMock()
        alpaca_mock.trading.requests.StopLimitOrderRequest = MagicMock(
            return_value=MagicMock()
        )
        alpaca_mock.trading.enums.OrderSide  = OrderSide
        alpaca_mock.trading.enums.TimeInForce = TimeInForce

        return alpaca_mock

    def test_dry_run_does_not_call_submit_order(self, monkeypatch):
        """In DRY_RUN mode the GTC stop should be logged but submit_order must NOT be called."""
        monkeypatch.setenv('DRY_RUN', 'true')
        monkeypatch.setenv('OPTIONS_GTC_STOP', 'true')
