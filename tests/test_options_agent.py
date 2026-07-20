"""
Tests for DTE-based exit feature and gap fixes in OptionsAgent.

Covers:
  - parse_dte_from_symbol()           (DTE feature + Gap 3: EST-aware date)
  - _has_open_exit_order()            (Gap 1: duplicate exit guard)
  - manage_existing_positions()       (DTE exit logic, Gap 1, Gap 4, Gap 5)
  - run_options_session() ordering    (Gap 2: circuit breaker no longer blocks exits)
  - research param override           (exit_dte_threshold from monday_params_options.json)

All tests are fully offline — no Alpaca API calls, no Ollama, no file I/O required.

Run with:
    python3 -m pytest tests/test_options_agent.py -v
"""
import sys
from pathlib import Path
from datetime import datetime, date, timedelta
from unittest.mock import MagicMock, patch, call

import pytest
from alpaca.trading.requests import LimitOrderRequest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'agents'))

import decision_parser


# ---------------------------------------------------------------------------
# Shared helper — builds an OptionsAgent with all external deps patched out
# ---------------------------------------------------------------------------

def _make_agent():
    """Return an OptionsAgent with all external calls patched (no API / Ollama)."""
    with patch('options_agent.TradingClient'), \
         patch('options_agent.OptionHistoricalDataClient'), \
         patch('options_agent.load_dotenv'):
        from options_agent import OptionsAgent
        agent = OptionsAgent.__new__(OptionsAgent)

    agent.params = {
        'take_profit':            0.50,
        'stop_loss':             -0.50,
        'exit_dte_threshold':     2,
        'max_daily_loss_pct':     0.05,
        'max_daily_trades':       5,
        'max_portfolio_allocation': 0.15,
        'min_portfolio_allocation': 0.10,
        'max_position_size':      0.03,
        'min_confidence':         0.75,
        'max_dte':                45,
        'min_dte':                7,
        'target_delta':           0.30,
        'max_loss_per_trade':     0.02,
    }
    agent.trading_client      = MagicMock()
    agent.option_data_client  = MagicMock()
    agent.daily_trades        = 0
    agent.last_reset_date     = datetime.now().date()
    agent.daily_start_equity  = None
    agent.trade_cooldown      = {}
    agent.watchlist           = ['SPY', 'QQQ']
    agent.session_id          = 'test_session'
    agent._decision_log       = Path('/tmp/test_options_decisions.jsonl')
    agent._params_file        = Path('/tmp/non_existent_params.json')
    agent._prev_equity        = None
    # R1 WI-4: placeholder only -- every test that exercises
    # manage_existing_positions() must mock options_agent._db.insert_pending_exit
    # itself (matching the existing patch('builtins.open', MagicMock())
    # pattern for the JSONL write); this attribute existing is just so
    # `self._db_path` doesn't raise AttributeError before that mock is hit.
    agent._db_path            = Path('/tmp/test_options_pending_exits.db')
    return agent


def _make_position(symbol, unrealized_plpc='-0.10', qty='2'):
    pos = MagicMock()
    pos.symbol          = symbol
    pos.unrealized_plpc = unrealized_plpc
    pos.qty             = qty
    pos.current_price   = '1.50'
    return pos


# ===========================================================================
# 1. parse_dte_from_symbol — DTE feature + Gap 3 (EST-aware date)
# ===========================================================================

class TestParseDteFromSymbol:

    def test_spy_put_2_days_before_expiry(self):
        """SPY250328P00560000 expires 2025-03-28; today 2025-03-26 → DTE = 2."""
        agent = _make_agent()
        dte = agent.parse_dte_from_symbol(
            'SPY250328P00560000', _today=date(2025, 3, 26)
        )
        assert dte == 2

    def test_aapl_call_4char_ticker(self):
        """AAPL (4-char ticker) — symbol[-15:-9] still lands on the date field."""
        agent = _make_agent()
        # AAPL260319C00207000 expires 2026-03-19; today 2026-03-10 → DTE = 9
        dte = agent.parse_dte_from_symbol(
            'AAPL260319C00207000', _today=date(2026, 3, 10)
        )
        assert dte == 9

    def test_5char_ticker(self):
        """A 5-char ticker — date still parsed correctly from symbol[-15:-9]."""
        agent = _make_agent()
        # NVDAA251121C01200000 expires 2025-11-21; today 2025-11-19 → DTE = 2
        dte = agent.parse_dte_from_symbol(
            'NVDAA251121C01200000', _today=date(2025, 11, 19)
        )
        assert dte == 2

    def test_expiration_day_returns_zero(self):
        """On the expiration day itself DTE should be 0."""
        agent = _make_agent()
        dte = agent.parse_dte_from_symbol(
            'SPY250328P00560000', _today=date(2025, 3, 28)
        )
        assert dte == 0

    def test_past_expiry_returns_negative(self):
        """After expiry DTE is negative (contract already expired)."""
        agent = _make_agent()
        dte = agent.parse_dte_from_symbol(
            'SPY250328P00560000', _today=date(2025, 3, 30)
        )
        assert dte == -2

    def test_invalid_symbol_returns_none(self):
        """Too-short symbol cannot be parsed — must return None, not raise."""
        agent = _make_agent()
        assert agent.parse_dte_from_symbol('BADDATA') is None

    def test_stock_ticker_returns_none(self):
        """A bare stock ticker (3 chars) returns None without crashing."""
        agent = _make_agent()
        assert agent.parse_dte_from_symbol('SPY') is None

    def test_uses_est_when_no_today_override(self):
        """Without _today override, the method should call datetime.now() — no crash."""
        agent = _make_agent()
        # Just verify it doesn't raise; the exact DTE value depends on wall-clock time
        result = agent.parse_dte_from_symbol('SPY260101P00560000')
        assert result is None or isinstance(result, int)


# ===========================================================================
# 2. _has_open_exit_order — Gap 1 (duplicate exit guard)
# ===========================================================================

class TestHasOpenExitOrder:

    def test_returns_true_when_open_sell_exists(self):
        """Should return True if a SELL order for the symbol is already open."""
        from alpaca.trading.enums import OrderSide
        agent = _make_agent()
        mock_order        = MagicMock()
        mock_order.symbol = 'SPY250328P00560000'
        mock_order.side   = OrderSide.SELL
        agent.trading_client.get_orders.return_value = [mock_order]

        assert agent._has_open_exit_order('SPY250328P00560000') is True

    def test_returns_false_when_only_buy_order_open(self):
        """A pending entry (BUY) order must not block a new exit submission."""
        from alpaca.trading.enums import OrderSide
        agent = _make_agent()
        mock_order        = MagicMock()
        mock_order.symbol = 'SPY250328P00560000'
        mock_order.side   = OrderSide.BUY
        agent.trading_client.get_orders.return_value = [mock_order]

        assert agent._has_open_exit_order('SPY250328P00560000') is False

    def test_returns_false_for_different_symbol(self):
        """Open SELL on a different symbol should not block this symbol's exit."""
        from alpaca.trading.enums import OrderSide
        agent = _make_agent()
        mock_order        = MagicMock()
        mock_order.symbol = 'QQQ250328P00450000'
        mock_order.side   = OrderSide.SELL
        agent.trading_client.get_orders.return_value = [mock_order]

        assert agent._has_open_exit_order('SPY250328P00560000') is False

    def test_returns_false_when_no_open_orders(self):
        """Empty order list returns False."""
        agent = _make_agent()
        agent.trading_client.get_orders.return_value = []
        assert agent._has_open_exit_order('SPY250328P00560000') is False

    def test_returns_false_on_api_exception(self):
        """API failure must default to False — never silently block an exit."""
        agent = _make_agent()
        agent.trading_client.get_orders.side_effect = Exception("API unavailable")
        assert agent._has_open_exit_order('SPY250328P00560000') is False


# ===========================================================================
# 3. manage_existing_positions — DTE exit logic
# ===========================================================================

class TestManageExistingPositionsDTE:

    def test_dte_exit_fires_at_threshold(self):
        """Position exactly at exit_dte_threshold (DTE=2) must trigger exit."""
        agent = _make_agent()
        pos = _make_position('SPY250328P00560000', unrealized_plpc='-0.10')
        agent.trading_client.get_all_positions.return_value = [pos]
        agent.trading_client.get_orders.return_value = []
        mock_order = MagicMock(); mock_order.id = 'order-dte'
        agent.trading_client.submit_order.return_value = mock_order

        with patch.object(agent, 'parse_dte_from_symbol', return_value=2), \
             patch('options_agent.alert_trade_executed'), \
             patch('builtins.open', MagicMock()):
            agent.manage_existing_positions()

        agent.trading_client.submit_order.assert_called_once()
        submitted = agent.trading_client.submit_order.call_args[0][0]
        assert isinstance(submitted, LimitOrderRequest)

    def test_dte_exit_fires_below_threshold(self):
        """DTE=0 (expiration day) must also trigger exit."""
        agent = _make_agent()
        pos = _make_position('SPY250328P00560000', unrealized_plpc='-0.10')
        agent.trading_client.get_all_positions.return_value = [pos]
        agent.trading_client.get_orders.return_value = []
        mock_order = MagicMock(); mock_order.id = 'order-expday'
        agent.trading_client.submit_order.return_value = mock_order

        with patch.object(agent, 'parse_dte_from_symbol', return_value=0), \
             patch('options_agent.alert_trade_executed'), \
             patch('builtins.open', MagicMock()):
            agent.manage_existing_positions()

        agent.trading_client.submit_order.assert_called_once()

    def test_dte_exit_not_fired_outside_threshold(self):
        """DTE=10 with no P&L trigger — position must not be closed."""
        agent = _make_agent()
        pos = _make_position('SPY250405P00560000', unrealized_plpc='-0.10')
        agent.trading_client.get_all_positions.return_value = [pos]
        agent.trading_client.get_orders.return_value = []

        with patch.object(agent, 'parse_dte_from_symbol', return_value=10):
            agent.manage_existing_positions()

        agent.trading_client.submit_order.assert_not_called()

    def test_dte_exit_uses_gtc(self):
        """DTE exit orders must use GTC time-in-force."""
        from alpaca.trading.enums import TimeInForce
        agent = _make_agent()
        pos = _make_position('SPY250328P00560000', unrealized_plpc='-0.10')
        agent.trading_client.get_all_positions.return_value = [pos]
        agent.trading_client.get_orders.return_value = []
        mock_order = MagicMock(); mock_order.id = 'gtc-1'
        agent.trading_client.submit_order.return_value = mock_order

        with patch.object(agent, 'parse_dte_from_symbol', return_value=2), \
             patch('options_agent.alert_trade_executed'), \
             patch('builtins.open', MagicMock()):
            agent.manage_existing_positions()

        submitted = agent.trading_client.submit_order.call_args[0][0]
        assert submitted.time_in_force == TimeInForce.GTC
        assert isinstance(submitted, LimitOrderRequest)

    def test_dte_exit_skips_timeout_retry(self):
        """_fill_timeout_retry must NOT be called for DTE (GTC) exits."""
        agent = _make_agent()
        pos = _make_position('SPY250328P00560000', unrealized_plpc='-0.10')
        agent.trading_client.get_all_positions.return_value = [pos]
        agent.trading_client.get_orders.return_value = []
        mock_order = MagicMock(); mock_order.id = 'gtc-no-retry'
        agent.trading_client.submit_order.return_value = mock_order

        with patch.object(agent, 'parse_dte_from_symbol', return_value=2), \
             patch.object(agent, '_fill_timeout_retry') as mock_retry, \
             patch('options_agent.alert_trade_executed'), \
             patch('builtins.open', MagicMock()):
            agent.manage_existing_positions()

        mock_retry.assert_not_called()

    def test_pl_stop_loss_still_uses_day(self):
        """Stop-loss exits (not DTE) must keep DAY time-in-force."""
        from alpaca.trading.enums import TimeInForce
        agent = _make_agent()
        # -55% triggers stop_loss; DTE=30 → no DTE trigger
        pos = _make_position('SPY250425P00560000', unrealized_plpc='-0.55')
        agent.trading_client.get_all_positions.return_value = [pos]
        agent.trading_client.get_orders.return_value = []
        mock_order = MagicMock(); mock_order.id = 'sl-1'
        agent.trading_client.submit_order.return_value = mock_order

        with patch.object(agent, 'parse_dte_from_symbol', return_value=30), \
             patch.object(agent, '_fill_timeout_retry', return_value=mock_order), \
             patch('options_agent.alert_trade_executed'), \
             patch('builtins.open', MagicMock()):
            agent.manage_existing_positions()

        submitted = agent.trading_client.submit_order.call_args[0][0]
        assert submitted.time_in_force == TimeInForce.DAY
        assert isinstance(submitted, LimitOrderRequest)

    def test_pl_take_profit_uses_day(self):
        """Take-profit exits must keep DAY time-in-force."""
        from alpaca.trading.enums import TimeInForce
        agent = _make_agent()
        pos = _make_position('SPY250425C00560000', unrealized_plpc='0.55')
        agent.trading_client.get_all_positions.return_value = [pos]
        agent.trading_client.get_orders.return_value = []
        mock_order = MagicMock(); mock_order.id = 'tp-1'
        agent.trading_client.submit_order.return_value = mock_order

        with patch.object(agent, 'parse_dte_from_symbol', return_value=30), \
             patch.object(agent, '_fill_timeout_retry', return_value=mock_order), \
             patch('options_agent.alert_trade_executed'), \
             patch('builtins.open', MagicMock()):
            agent.manage_existing_positions()

        submitted = agent.trading_client.submit_order.call_args[0][0]
        assert submitted.time_in_force == TimeInForce.DAY
        assert isinstance(submitted, LimitOrderRequest)

    def test_pl_exit_takes_priority_over_dte(self):
        """When both P&L and DTE thresholds are met, P&L reason wins (fires first)."""
        agent = _make_agent()
        # -55% (stop loss) AND DTE=1 (DTE exit)
        pos = _make_position('SPY250328P00560000', unrealized_plpc='-0.55')
        agent.trading_client.get_all_positions.return_value = [pos]
        agent.trading_client.get_orders.return_value = []
        mock_order = MagicMock(); mock_order.id = 'dual-1'
        agent.trading_client.submit_order.return_value = mock_order

        with patch.object(agent, 'parse_dte_from_symbol', return_value=1), \
             patch.object(agent, '_fill_timeout_retry', return_value=mock_order), \
             patch('options_agent._db.insert_pending_exit') as mock_pending:
            agent.manage_existing_positions()

        # Only one order submitted (not two)
        agent.trading_client.submit_order.assert_called_once()
        mock_pending.assert_called_once()
        pending_rec = mock_pending.call_args[0][0]
        assert 'stop_loss' in pending_rec['intended_reason']
        assert 'dte_exit' not in pending_rec['intended_reason']

    def test_stock_symbols_skipped(self):
        """Short symbols (stocks) must be ignored even if len > 10 guard fails."""
        agent = _make_agent()
        stock_pos = _make_position('SPY', unrealized_plpc='-0.80')  # len=3 ≤ 10
        agent.trading_client.get_all_positions.return_value = [stock_pos]

        agent.manage_existing_positions()
        agent.trading_client.submit_order.assert_not_called()


# ===========================================================================
# 4. Gap 1 — duplicate exit guard inside manage_existing_positions
# ===========================================================================

class TestDuplicateExitGuard:

    def test_exit_skipped_when_open_sell_order_exists(self):
        """No new order submitted if an open SELL already exists for the symbol."""
        from alpaca.trading.enums import OrderSide
        agent = _make_agent()
        # Stop-loss position
        pos = _make_position('SPY250328P00560000', unrealized_plpc='-0.55')
        agent.trading_client.get_all_positions.return_value = [pos]

        existing = MagicMock()
        existing.symbol = 'SPY250328P00560000'
        existing.side   = OrderSide.SELL
        agent.trading_client.get_orders.return_value = [existing]

        with patch.object(agent, 'parse_dte_from_symbol', return_value=30):
            agent.manage_existing_positions()

        agent.trading_client.submit_order.assert_not_called()

    def test_exit_proceeds_when_no_open_sell_order(self):
        """Exit proceeds normally when no duplicate order exists."""
        agent = _make_agent()
        pos = _make_position('SPY250328P00560000', unrealized_plpc='-0.55')
        agent.trading_client.get_all_positions.return_value = [pos]
        agent.trading_client.get_orders.return_value = []
        mock_order = MagicMock(); mock_order.id = 'new-sl'
        agent.trading_client.submit_order.return_value = mock_order

        with patch.object(agent, 'parse_dte_from_symbol', return_value=30), \
             patch.object(agent, '_fill_timeout_retry', return_value=mock_order), \
             patch('options_agent.alert_trade_executed'), \
             patch('builtins.open', MagicMock()):
            agent.manage_existing_positions()

        agent.trading_client.submit_order.assert_called_once()


# ===========================================================================
# 5. Gap 2 — manage_existing_positions runs before circuit breaker
# ===========================================================================

class TestCircuitBreakerOrdering:

    def test_manage_positions_called_even_when_circuit_breaker_fires(self):
        """
        manage_existing_positions must be called before run_options_session returns
        due to the circuit breaker — DTE exits must fire on worst-loss days.
        """
        agent = _make_agent()
        agent.daily_start_equity = 100_000.0

        # Equity down 6% → circuit breaker should fire
        mock_account = MagicMock()
        mock_account.equity = '94000'
        agent.trading_client.get_account.return_value = mock_account
        agent.trading_client.get_all_positions.return_value = []
        agent.trading_client.get_orders.return_value = []

        call_order = []

        def track_manage():
            call_order.append('manage')

        def track_circuit(*_):
            call_order.append('circuit')

        with patch.object(agent, 'manage_existing_positions', side_effect=track_manage), \
             patch.object(agent, 'load_research_params', return_value={}), \
             patch('options_agent.alert_circuit_breaker', side_effect=track_circuit):
            agent.run_options_session()

        assert 'manage' in call_order, "manage_existing_positions was never called"
        assert 'circuit' in call_order, "circuit breaker did not fire"
        assert call_order.index('manage') < call_order.index('circuit'), \
            "manage_existing_positions must run BEFORE the circuit breaker check"

    def test_no_new_trades_after_circuit_breaker(self):
        """Watchlist scanning must not happen when circuit breaker fires."""
        agent = _make_agent()
        agent.daily_start_equity = 100_000.0

        mock_account = MagicMock()
        mock_account.equity = '94000'  # -6%
        agent.trading_client.get_account.return_value = mock_account
        agent.trading_client.get_all_positions.return_value = []
        agent.trading_client.get_orders.return_value = []

        with patch.object(agent, 'manage_existing_positions'), \
             patch.object(agent, 'load_research_params', return_value={}), \
             patch.object(agent, 'get_available_capital') as mock_cap, \
             patch('options_agent.alert_circuit_breaker'):
            agent.run_options_session()

        # get_available_capital (called only for new trades) must NOT be reached
        mock_cap.assert_not_called()


# ===========================================================================
# 6. Gap 4 — alert_trade_executed fired on every exit including DTE
# ===========================================================================

class TestAlertOnExit:
    """R1 WI-4: manage_existing_positions() must NOT alert (or write a
    trade_log row) at submit time anymore -- that's exactly the defect that
    fabricated the NVDA/AAPL false "closed at a profit" records. It now
    records a pending_exits row instead; exit_reconciler.py fires the real
    alert later, only once the broker confirms an actual fill."""

    def test_dte_exit_records_pending_not_alert(self):
        """A DTE exit submits an order and records it pending -- no immediate alert."""
        agent = _make_agent()
        pos = _make_position('SPY250328P00560000', unrealized_plpc='-0.10')
        agent.trading_client.get_all_positions.return_value = [pos]
        agent.trading_client.get_orders.return_value = []
        mock_order = MagicMock(); mock_order.id = 'dte-alert-test'
        agent.trading_client.submit_order.return_value = mock_order

        with patch.object(agent, 'parse_dte_from_symbol', return_value=2), \
             patch('options_agent.alert_trade_executed') as mock_alert, \
             patch('options_agent._db.insert_pending_exit') as mock_pending:
            agent.manage_existing_positions()

        mock_alert.assert_not_called()
        mock_pending.assert_called_once()
        pending_rec = mock_pending.call_args[0][0]
        assert pending_rec['symbol'] == 'SPY250328P00560000'
        assert pending_rec['order_id'] == 'dte-alert-test'
        assert 'dte_exit' in pending_rec['intended_reason']

    def test_stop_loss_exit_records_pending_not_alert(self):
        """A stop-loss exit submits an order and records it pending -- no immediate alert."""
        agent = _make_agent()
        pos = _make_position('SPY250425P00560000', unrealized_plpc='-0.55')
        agent.trading_client.get_all_positions.return_value = [pos]
        agent.trading_client.get_orders.return_value = []
        mock_order = MagicMock(); mock_order.id = 'sl-alert-test'
        agent.trading_client.submit_order.return_value = mock_order

        with patch.object(agent, 'parse_dte_from_symbol', return_value=30), \
             patch.object(agent, '_fill_timeout_retry', return_value=mock_order), \
             patch('options_agent.alert_trade_executed') as mock_alert, \
             patch('options_agent._db.insert_pending_exit') as mock_pending:
            agent.manage_existing_positions()

        mock_alert.assert_not_called()
        mock_pending.assert_called_once()
        pending_rec = mock_pending.call_args[0][0]
        assert 'stop_loss' in pending_rec['intended_reason']

    def test_no_alert_when_no_exit(self):
        """alert_trade_executed must NOT be called when no exit triggers."""
        agent = _make_agent()
        pos = _make_position('SPY250425P00560000', unrealized_plpc='-0.10')
        agent.trading_client.get_all_positions.return_value = [pos]
        agent.trading_client.get_orders.return_value = []

        with patch.object(agent, 'parse_dte_from_symbol', return_value=30), \
             patch('options_agent.alert_trade_executed') as mock_alert:
            agent.manage_existing_positions()

        mock_alert.assert_not_called()


# ===========================================================================
# 7. Gap 5 — near-zero value warning for DTE exits
# ===========================================================================

class TestNearZeroValueWarning:

    def test_warning_logged_when_near_zero_contract_skipped(self):
        """
        A near-zero contract (total value < _MIN_EXIT_VALUE_USD) must log a warning
        and skip the exit order (expiry is the same outcome without the commission).

        price=$0.02, qty=1 → total=$0.02×100×1=$2.00 < $10 → skip with warning.
        """
        agent = _make_agent()
        pos = _make_position('SPY250328P00560000', unrealized_plpc='-0.10')
        agent.trading_client.get_all_positions.return_value = [pos]
        agent.trading_client.get_orders.return_value = []

        import options_agent as oa
        with patch.object(agent, 'parse_dte_from_symbol', return_value=2), \
             patch.object(agent, 'get_option_price', return_value=0.02), \
             patch.object(oa, '_MIN_EXIT_VALUE_USD', 10.0), \
             patch('options_agent.alert_trade_executed'), \
             patch('builtins.open', MagicMock()), \
             patch('options_agent.logging') as mock_log:
            agent.manage_existing_positions()

        # No order submitted — skipped because total value is below threshold
        agent.trading_client.submit_order.assert_not_called()

        # A warning-level log must have been emitted about the skip
        assert mock_log.warning.called
        warning_msgs = [str(c) for c in mock_log.warning.call_args_list]
        assert any('total value' in m or 'expire worthless' in m or 'min' in m
                   for m in warning_msgs)

    def test_exit_proceeds_when_contract_has_enough_value(self):
        """
        A DTE exit must still be submitted when the total contract value
        is at or above _MIN_EXIT_VALUE_USD.

        price=$0.15, qty=1 → total=$0.15×100×1=$15.00 >= $10 → submit.
        """
        agent = _make_agent()
        pos = _make_position('SPY250328P00560000', unrealized_plpc='-0.10')
        agent.trading_client.get_all_positions.return_value = [pos]
        agent.trading_client.get_orders.return_value = []
        mock_order = MagicMock(); mock_order.id = 'still-exit'
        agent.trading_client.submit_order.return_value = mock_order

        import options_agent as oa
        with patch.object(agent, 'parse_dte_from_symbol', return_value=2), \
             patch.object(agent, 'get_option_price', return_value=0.15), \
             patch.object(oa, '_MIN_EXIT_VALUE_USD', 10.0), \
             patch('options_agent.alert_trade_executed'), \
             patch('builtins.open', MagicMock()):
            agent.manage_existing_positions()

        agent.trading_client.submit_order.assert_called_once()

    def test_no_warning_when_option_has_value(self):
        """No warning when the option still has meaningful value (>= $0.05)."""
        agent = _make_agent()
        pos = _make_position('SPY250328P00560000', unrealized_plpc='-0.10')
        agent.trading_client.get_all_positions.return_value = [pos]
        agent.trading_client.get_orders.return_value = []
        mock_order = MagicMock(); mock_order.id = 'has-value'
        agent.trading_client.submit_order.return_value = mock_order

        with patch.object(agent, 'parse_dte_from_symbol', return_value=2), \
             patch.object(agent, 'get_option_price', return_value=0.50), \
             patch('options_agent.alert_trade_executed'), \
             patch('builtins.open', MagicMock()), \
             patch('options_agent.logging') as mock_log:
            agent.manage_existing_positions()

        # No warning should be emitted for this case
        warning_msgs = [str(c) for c in mock_log.warning.call_args_list]
        assert not any('near-zero' in m or '$0.50' in m for m in warning_msgs)

    def test_no_warning_for_pl_exits(self):
        """Near-zero price warning only applies to DTE exits, not P&L exits."""
        agent = _make_agent()
        # Stop-loss at -55%, DTE=30 (no DTE trigger)
        pos = _make_position('SPY250425P00560000', unrealized_plpc='-0.55')
        agent.trading_client.get_all_positions.return_value = [pos]
        agent.trading_client.get_orders.return_value = []
        mock_order = MagicMock(); mock_order.id = 'sl-no-warn'
        agent.trading_client.submit_order.return_value = mock_order

        with patch.object(agent, 'parse_dte_from_symbol', return_value=30), \
             patch.object(agent, 'get_option_price', return_value=0.01), \
             patch.object(agent, '_fill_timeout_retry', return_value=mock_order), \
             patch('options_agent.alert_trade_executed'), \
             patch('builtins.open', MagicMock()), \
             patch('options_agent.logging') as mock_log:
            agent.manage_existing_positions()

        warning_msgs = [str(c) for c in mock_log.warning.call_args_list]
        assert not any('near-zero' in m for m in warning_msgs)


# ===========================================================================
# 8. Research param override — exit_dte_threshold from monday_params_options.json
# ===========================================================================

class TestResearchParamOverride:

    def test_research_overrides_exit_dte_threshold(self):
        """
        When monday_params_options.json contains exit_dte_threshold, it must
        be applied to self.params before manage_existing_positions runs.
        """
        agent = _make_agent()
        agent.daily_start_equity = 100_000.0

        mock_account = MagicMock()
        mock_account.equity = '100000'
        agent.trading_client.get_account.return_value = mock_account
        agent.trading_client.get_all_positions.return_value = []
        agent.trading_client.get_orders.return_value = []

        applied_threshold = []

        def capture_manage():
            applied_threshold.append(agent.params['exit_dte_threshold'])

        with patch.object(agent, 'load_research_params',
                          return_value={'exit_dte_threshold': 5}), \
             patch.object(agent, 'manage_existing_positions',
                          side_effect=capture_manage), \
             patch.object(agent, 'get_available_capital', return_value=0):
            agent.run_options_session()

        assert applied_threshold == [5], (
            f"Expected exit_dte_threshold=5 inside manage_existing_positions, "
            f"got {applied_threshold}"
        )

    def test_default_threshold_used_when_not_in_research(self):
        """If research params omit exit_dte_threshold, the default (2) is kept."""
        agent = _make_agent()
        agent.daily_start_equity = 100_000.0

        mock_account = MagicMock()
        mock_account.equity = '100000'
        agent.trading_client.get_account.return_value = mock_account
        agent.trading_client.get_all_positions.return_value = []
        agent.trading_client.get_orders.return_value = []

        applied_threshold = []

        def capture_manage():
            applied_threshold.append(agent.params['exit_dte_threshold'])

        with patch.object(agent, 'load_research_params', return_value={}), \
             patch.object(agent, 'manage_existing_positions',
                          side_effect=capture_manage), \
             patch.object(agent, 'get_available_capital', return_value=0):
            agent.run_options_session()

        assert applied_threshold == [2]


# ===========================================================================
# 9. Fix 4 — _fill_timeout_retry retry behavior
# ===========================================================================

class TestFillTimeoutRetry:

    def test_pl_exit_retries_at_aggressive_price(self):
        """
        When a DAY limit exit is not filled after 5 minutes, _fill_timeout_retry
        must cancel the original order and resubmit at limit_price * 0.99.
        """
        from alpaca.trading.enums import TimeInForce
        agent = _make_agent()

        original_order = MagicMock()
        original_order.id = 'orig-order-1'

        retry_order = MagicMock()
        retry_order.id = 'retry-order-1'

        # Simulate unfilled order status
        unfilled_status = MagicMock()
        unfilled_status.status.value = 'accepted'
        agent.trading_client.get_order_by_id.return_value = unfilled_status
        agent.trading_client.submit_order.return_value = retry_order

        pos = _make_position('SPY250425P00560000')
        limit_price = 2.00

        with patch('options_agent.time') as mock_time:
            result = agent._fill_timeout_retry(original_order, pos, 2, limit_price)

        mock_time.sleep.assert_called_once_with(300)
        agent.trading_client.cancel_order_by_id.assert_called_once_with('orig-order-1')

        submitted = agent.trading_client.submit_order.call_args[0][0]
        assert isinstance(submitted, LimitOrderRequest)
        assert submitted.limit_price == round(limit_price * 0.99, 2)
        assert submitted.time_in_force == TimeInForce.DAY
        assert result is retry_order

    def test_no_retry_when_already_filled(self):
        """_fill_timeout_retry must not cancel/resubmit if order is already filled."""
        agent = _make_agent()

        original_order = MagicMock()
        original_order.id = 'filled-order-1'

        filled_status = MagicMock()
        filled_status.status.value = 'filled'
        agent.trading_client.get_order_by_id.return_value = filled_status

        pos = _make_position('SPY250425P00560000')

        with patch('options_agent.time') as mock_time:
            result = agent._fill_timeout_retry(original_order, pos, 2, 2.00)

        mock_time.sleep.assert_called_once_with(300)
        agent.trading_client.cancel_order_by_id.assert_not_called()
        agent.trading_client.submit_order.assert_not_called()
        assert result is original_order


# ===========================================================================
# 10. Sprint01 C1.2 — options call/put substring-mapping bug fix
# ===========================================================================

def _make_analysis(momentum=0.0):
    return {
        'symbol':        'AAPL',
        'current_price': 150.0,
        'rsi':           55.0,
        'volatility':    0.25,
        'momentum':      momentum,
    }


class TestOptionsCallPutMapping:
    """get_ai_options_decision must trust parse_decision's structured result,
    never re-scan the raw response text for 'call'/'put' substrings — a HOLD
    whose Reasoning happens to contain those words must stay HOLD.

    Other test modules in this suite (test_options_session.py, test_option_price.py)
    register their own `sys.modules['model_inference_lora']` stub at import time,
    which — because `options_agent.py` does `from model_inference_lora import
    parse_decision` — decides what `options_agent.parse_decision` resolves to for
    the rest of the pytest session, regardless of collection order. These tests
    pin `options_agent.parse_decision` to the real decision_parser implementation
    explicitly so they exercise the actual sprint01 C1 parsing/mapping logic
    rather than whatever stub happened to win the cross-file import race.
    """

    def test_hold_reasoning_mentioning_call_is_not_promoted(self):
        agent = _make_agent()
        response = "Decision: HOLD\nConfidence: 0.60\nReasoning: the technicals call for caution here"
        with patch('options_agent.get_trading_decision', return_value=response), \
             patch('options_agent.parse_decision', new=decision_parser.parse_decision):
            decision, _, _ = agent.get_ai_options_decision(_make_analysis(momentum=0.05))
        assert decision['decision'] == 'hold'

    def test_hold_reasoning_mentioning_put_is_not_promoted(self):
        agent = _make_agent()
        response = "Decision: HOLD\nConfidence: 0.60\nReasoning: put simply, wait for confirmation"
        with patch('options_agent.get_trading_decision', return_value=response), \
             patch('options_agent.parse_decision', new=decision_parser.parse_decision):
            decision, _, _ = agent.get_ai_options_decision(_make_analysis(momentum=-0.05))
        assert decision['decision'] == 'hold'

    def test_buy_call_structured_line_respected(self):
        agent = _make_agent()
        response = "Decision: BUY_CALL\nConfidence: 0.80\nReasoning: strong bullish breakout"
        with patch('options_agent.get_trading_decision', return_value=response), \
             patch('options_agent.parse_decision', new=decision_parser.parse_decision):
            decision, _, _ = agent.get_ai_options_decision(_make_analysis(momentum=0.0))
        assert decision['decision'] == 'buy_call'

    def test_buy_put_structured_line_respected(self):
        agent = _make_agent()
        response = "Decision: BUY_PUT\nConfidence: 0.80\nReasoning: strong bearish breakdown"
        with patch('options_agent.get_trading_decision', return_value=response), \
             patch('options_agent.parse_decision', new=decision_parser.parse_decision):
            decision, _, _ = agent.get_ai_options_decision(_make_analysis(momentum=0.0))
        assert decision['decision'] == 'buy_put'

    def test_bare_buy_maps_to_call_only_with_positive_momentum(self):
        agent = _make_agent()
        response = "Decision: BUY\nConfidence: 0.80\nReasoning: momentum building"
        with patch('options_agent.get_trading_decision', return_value=response), \
             patch('options_agent.parse_decision', new=decision_parser.parse_decision):
            decision_neg, _, _ = agent.get_ai_options_decision(_make_analysis(momentum=-0.02))
            decision_pos, _, _ = agent.get_ai_options_decision(_make_analysis(momentum=0.02))
        assert decision_neg['decision'] == 'hold'
        assert decision_pos['decision'] == 'buy_call'

    def test_bare_sell_maps_to_put_only_with_negative_momentum(self):
        agent = _make_agent()
        response = "Decision: SELL\nConfidence: 0.80\nReasoning: weakening momentum"
        with patch('options_agent.get_trading_decision', return_value=response), \
             patch('options_agent.parse_decision', new=decision_parser.parse_decision):
            decision_pos, _, _ = agent.get_ai_options_decision(_make_analysis(momentum=0.02))
            decision_neg, _, _ = agent.get_ai_options_decision(_make_analysis(momentum=-0.02))
        assert decision_pos['decision'] == 'hold'
        assert decision_neg['decision'] == 'buy_put'
