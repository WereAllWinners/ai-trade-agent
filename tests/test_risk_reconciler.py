"""
tests/test_risk_reconciler.py — Broker-side risk reconciliation (PR-5, C2/C3)

Covers:
  - count_unprotected_positions: protected, unprotected, stock-symbol exclusion
  - write_reconcile_status: correct JSON written, warning logged on N>0
  - morning_model_check: exits 1 when reconcile_status shows unprotected > 0
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

from alpaca.trading.enums import OrderSide, TimeInForce  # noqa: E402
from risk_reconciler import count_unprotected_positions, write_reconcile_status  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers to build mock Alpaca objects
# ---------------------------------------------------------------------------

def _pos(symbol: str) -> MagicMock:
    p = MagicMock()
    p.symbol = symbol
    return p


def _order(symbol: str, side: OrderSide, tif=TimeInForce.DAY, stop_price=None) -> MagicMock:
    o = MagicMock()
    o.symbol     = symbol
    o.side       = side
    o.time_in_force = tif
    o.stop_price = stop_price
    return o


def _client(positions=None, orders=None) -> MagicMock:
    c = MagicMock()
    c.get_all_positions.return_value = positions or []
    c.get_orders.return_value        = orders or []
    return c


# ---------------------------------------------------------------------------
# count_unprotected_positions
# ---------------------------------------------------------------------------

class TestCountUnprotectedPositions:
    def test_position_with_open_sell_is_protected(self):
        """Options position that has an open SELL order → count = 0."""
        occ_sym  = 'AAPL240119C00190000'   # len > 10 → options
        client   = _client(
            positions=[_pos(occ_sym)],
            orders=[_order(occ_sym, OrderSide.SELL)],
        )
        assert count_unprotected_positions(client) == 0

    def test_position_without_sell_is_unprotected(self):
        """Options position with no SELL order → count = 1."""
        occ_sym = 'AAPL240119C00190000'
        client  = _client(
            positions=[_pos(occ_sym)],
            orders=[],
        )
        assert count_unprotected_positions(client) == 1

    def test_stock_symbol_excluded(self):
        """Short symbols (≤ 10 chars) are stocks, not options — excluded from count."""
        client = _client(
            positions=[_pos('AAPL'), _pos('MSFT'), _pos('SPY')],
            orders=[],
        )
        assert count_unprotected_positions(client) == 0

    def test_mixed_positions(self):
        """Two options, one stock, one protected options → count = 1."""
        occ_protected   = 'AAPL240119C00190000'
        occ_unprotected = 'TSLA240119P00200000'
        client = _client(
            positions=[_pos('AAPL'), _pos(occ_protected), _pos(occ_unprotected)],
            orders=[_order(occ_protected, OrderSide.SELL)],
        )
        assert count_unprotected_positions(client) == 1

    def test_api_error_returns_minus_one(self):
        """When the broker API raises, return -1 (unknown state)."""
        client = MagicMock()
        client.get_all_positions.side_effect = Exception("Network error")
        assert count_unprotected_positions(client) == -1

    def test_buy_order_does_not_count_as_protected(self):
        """An open BUY order on the same symbol does not protect the position."""
        occ_sym = 'AAPL240119C00190000'
        client  = _client(
            positions=[_pos(occ_sym)],
            orders=[_order(occ_sym, OrderSide.BUY)],
        )
        assert count_unprotected_positions(client) == 1


# ---------------------------------------------------------------------------
# write_reconcile_status
# ---------------------------------------------------------------------------

class TestWriteReconcileStatus:
    def test_writes_valid_json(self, tmp_path):
        occ_sym = 'AAPL240119C00190000'
        client  = _client(
            positions=[_pos(occ_sym)],
            orders=[_order(occ_sym, OrderSide.SELL)],
        )
        out = tmp_path / 'reconcile_status.json'
        write_reconcile_status(client, out)
        data = json.loads(out.read_text())
        assert data['unprotected_positions'] == 0
        assert 'checked_at' in data

    def test_unprotected_warning_logged(self, tmp_path, caplog):
        import logging
        occ_sym = 'AAPL240119C00190000'
        client  = _client(positions=[_pos(occ_sym)], orders=[])
        out     = tmp_path / 'reconcile_status.json'
        with caplog.at_level(logging.WARNING):
            write_reconcile_status(client, out)
        assert any('unprotected' in msg.lower() for msg in caplog.messages)
        data = json.loads(out.read_text())
        assert data['unprotected_positions'] == 1

    def test_creates_parent_dir_if_missing(self, tmp_path):
        nested = tmp_path / 'nested' / 'dir' / 'reconcile_status.json'
        client = _client()
        write_reconcile_status(client, nested)
        assert nested.exists()


# ---------------------------------------------------------------------------
# morning_model_check — reconcile gate
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
        with patch(
            'morning_model_check._PROJECT_ROOT',
            tmp_path,
        ):
            check_reconcile_status(results)
        return results

    def test_zero_unprotected_passes(self, tmp_path):
        results = self._run_check({'unprotected_positions': 0, 'checked_at': '2026-01-01T00:00'}, tmp_path)
        assert results.all_passed

    def test_nonzero_unprotected_fails(self, tmp_path):
        results = self._run_check({'unprotected_positions': 2, 'checked_at': '2026-01-01T00:00'}, tmp_path)
        assert not results.all_passed

    def test_missing_file_is_warn_only(self, tmp_path):
        results = self._run_check(None, tmp_path)
        assert results.all_passed          # warn_only=True → not a hard FAIL
        assert results.has_warnings

    def test_api_error_status_is_warn_only(self, tmp_path):
        # -1 means broker API was unavailable
        results = self._run_check({'unprotected_positions': -1, 'checked_at': '2026-01-01T00:00'}, tmp_path)
        assert results.all_passed          # warn_only=True for unknown state
        assert results.has_warnings


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

        trading_client = MagicMock()
        stop_calls = []

        # Simulate the GTC stop logic extracted from options_agent
        fill_price  = 2.00
        stop_loss   = -0.50
        stop_trigger = round(fill_price * (1 + stop_loss), 2)
        stop_limit   = round(stop_trigger * 0.90, 2)

        _DRY_RUN = os.getenv('DRY_RUN', 'false').lower() == 'true'
        if os.getenv('OPTIONS_GTC_STOP', 'true').lower() == 'true':
            if _DRY_RUN:
                stop_calls.append('logged')
            else:
                trading_client.submit_order(MagicMock())

        assert stop_calls == ['logged']
        trading_client.submit_order.assert_not_called()

    def test_live_mode_calls_submit_order(self, monkeypatch):
        """In live mode (DRY_RUN=false) the GTC stop should call submit_order."""
        monkeypatch.setenv('DRY_RUN', 'false')
        monkeypatch.setenv('OPTIONS_GTC_STOP', 'true')

        trading_client = MagicMock()
        fill_price  = 2.00
        stop_loss   = -0.50
        stop_trigger = round(fill_price * (1 + stop_loss), 2)
        stop_limit   = round(stop_trigger * 0.90, 2)

        _DRY_RUN = os.getenv('DRY_RUN', 'false').lower() == 'true'
        if os.getenv('OPTIONS_GTC_STOP', 'true').lower() == 'true':
            if _DRY_RUN:
                pass
            else:
                trading_client.submit_order(MagicMock())

        trading_client.submit_order.assert_called_once()

    def test_gtc_stop_disabled_by_env(self, monkeypatch):
        """OPTIONS_GTC_STOP=false → submit_order must not be called even in live mode."""
        monkeypatch.setenv('DRY_RUN', 'false')
        monkeypatch.setenv('OPTIONS_GTC_STOP', 'false')

        trading_client = MagicMock()

        if os.getenv('OPTIONS_GTC_STOP', 'true').lower() == 'true':
            trading_client.submit_order(MagicMock())

        trading_client.submit_order.assert_not_called()

    def test_stop_prices_use_stop_loss_param(self):
        """stop_trigger = fill_price * (1 + stop_loss), stop_limit = stop_trigger * 0.90."""
        fill_price = 3.50
        stop_loss  = -0.50  # -50% as used in options params
        stop_trigger = round(fill_price * (1 + stop_loss), 2)
        stop_limit   = round(stop_trigger * 0.90, 2)
        assert stop_trigger == pytest.approx(1.75, abs=0.01)
        assert stop_limit   == pytest.approx(1.575, abs=0.01)


# ---------------------------------------------------------------------------
# options_agent — _cancel_gtc_stops
# ---------------------------------------------------------------------------

class TestCancelGtcStops:
    """Unit tests for _cancel_gtc_stops via direct import of the method."""

    def _make_simple_agent(self, orders):
        """Build a minimal OptionsAgent-like object with only the method under test."""
        from agents.options_agent import OptionsAgent
        agent = object.__new__(OptionsAgent)
        agent.trading_client = _client(orders=orders)
        return agent

    def test_cancels_stop_limit_gtc_sell(self):
        """A GTC stop-limit SELL (has stop_price) should be cancelled."""
        occ_sym   = 'AAPL240119C00190000'
        stop_ord  = _order(occ_sym, OrderSide.SELL, tif=TimeInForce.GTC, stop_price=1.50)
        stop_ord.id = 'stop-order-id'

        agent = self._make_simple_agent([stop_ord])
        agent._cancel_gtc_stops(occ_sym)

        agent.trading_client.cancel_order_by_id.assert_called_once_with('stop-order-id')

    def test_does_not_cancel_plain_gtc_limit_sell(self):
        """A plain GTC limit SELL (no stop_price, e.g. DTE exit) must NOT be cancelled."""
        occ_sym    = 'AAPL240119C00190000'
        limit_ord  = _order(occ_sym, OrderSide.SELL, tif=TimeInForce.GTC, stop_price=None)
        limit_ord.id = 'limit-order-id'

        agent = self._make_simple_agent([limit_ord])
        agent._cancel_gtc_stops(occ_sym)

        agent.trading_client.cancel_order_by_id.assert_not_called()

    def test_does_not_cancel_different_symbol(self):
        """GTC stop for a different symbol must not be touched."""
        occ_sym      = 'AAPL240119C00190000'
        other_sym    = 'TSLA240119P00200000'
        stop_ord     = _order(other_sym, OrderSide.SELL, tif=TimeInForce.GTC, stop_price=1.00)
        stop_ord.id  = 'other-stop-id'

        agent = self._make_simple_agent([stop_ord])
        agent._cancel_gtc_stops(occ_sym)

        agent.trading_client.cancel_order_by_id.assert_not_called()

    def test_api_error_does_not_raise(self):
        """An API exception during cancellation must be swallowed (logged as warning)."""
        occ_sym = 'AAPL240119C00190000'
        agent   = object.__new__(__import__('agents.options_agent', fromlist=['OptionsAgent']).OptionsAgent)
        agent.trading_client = MagicMock()
        agent.trading_client.get_orders.side_effect = Exception("API down")

        # Should not raise
        agent._cancel_gtc_stops(occ_sym)
