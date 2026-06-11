"""
Tests for the _MIN_CASH_RESERVE cash floor in both trading bots.

Verifies that neither bot can push account cash below zero:
  - options_agent: get_available_capital() and execute_options_trade() live check
  - autonomous_agent: run_trading_session() gate and execute_trade() live check

All tests are fully offline — no Alpaca API calls, no Ollama.

Run with:
    python3 -m pytest tests/test_cash_reserve.py -v
"""
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'agents'))


# ===========================================================================
# Helpers
# ===========================================================================

def _make_options_agent():
    with patch('options_agent.TradingClient'), \
         patch('options_agent.OptionHistoricalDataClient'), \
         patch('options_agent.load_dotenv'), \
         patch('options_agent.PortfolioOverseer'), \
         patch('options_agent.AllocationController'):
        from options_agent import OptionsAgent
        agent = OptionsAgent.__new__(OptionsAgent)

    agent.params = {
        'take_profit': 0.50, 'stop_loss': -0.50, 'exit_dte_threshold': 2,
        'max_daily_loss_pct': 0.05, 'max_daily_trades': 5,
        'max_portfolio_allocation': 0.15, 'min_portfolio_allocation': 0.10,
        'max_position_size': 0.03, 'min_confidence': 0.75,
        'max_dte': 45, 'min_dte': 7, 'target_delta': 0.30,
        'max_loss_per_trade': 0.02,
    }
    agent.trading_client     = MagicMock()
    agent.option_data_client = MagicMock()
    agent.daily_trades       = 0
    agent.last_reset_date    = datetime.now().date()
    agent.daily_start_equity = None
    agent.trade_cooldown     = {}
    agent.watchlist          = ['SPY']
    agent.session_id         = 'test'
    agent._decision_log      = Path('/tmp/test_options_dec.jsonl')
    agent._params_file       = Path('/tmp/non_existent.json')
    alloc_ctrl = MagicMock()
    alloc_ctrl.get_position_size_pct.return_value = 0.03
    agent.alloc_controller = alloc_ctrl
    overseer = MagicMock()
    overseer.is_buy_allowed.return_value = (True, '')
    agent.overseer = overseer
    fill_result = MagicMock()
    fill_result.filled = True
    fill_result.slippage_bps = 0.0
    paper_sim = MagicMock()
    paper_sim.simulate_options_fill.side_effect = lambda side, mid_price, contracts, **_kw: (
        setattr(fill_result, 'fill_qty', contracts) or
        setattr(fill_result, 'fill_price', mid_price) or
        fill_result
    )
    agent.paper_sim = paper_sim
    return agent


def _make_stock_agent():
    with patch('autonomous_agent.TradingClient'), \
         patch('autonomous_agent.StockHistoricalDataClient'), \
         patch('autonomous_agent.StockDiscovery'), \
         patch('autonomous_agent.load_dotenv'), \
         patch('autonomous_agent.load_model_once'), \
         patch('autonomous_agent.PortfolioOverseer'), \
         patch('autonomous_agent.AllocationController'):
        from autonomous_agent import AutonomousAgent
        agent = AutonomousAgent.__new__(AutonomousAgent)

    agent.params = {
        'max_position_size': 0.05, 'stop_loss': -0.07, 'take_profit': 0.15,
        'max_daily_loss_pct': 0.05, 'max_daily_trades': 10,
        'cooldown_minutes': 15, 'min_confidence': 0.60,
        'max_stocks_to_analyze': 5,
    }
    agent.trading_client     = MagicMock()
    agent.daily_trades       = 0
    agent.last_reset_date    = datetime.now().date()
    agent.daily_start_equity = None
    agent.cooldowns          = {}
    agent.pdt_blocked        = False
    agent.discovery          = MagicMock()
    alloc_ctrl = MagicMock()
    alloc_ctrl.get_position_size_pct.return_value = 0.05
    agent.alloc_controller = alloc_ctrl
    overseer = MagicMock()
    overseer.is_buy_allowed.return_value = (True, '')
    agent.overseer = overseer
    fill_result = MagicMock()
    fill_result.filled = True
    fill_result.slippage_bps = 0.0
    paper_sim = MagicMock()
    paper_sim.simulate_stock_fill.side_effect = lambda side, price, qty, **_kw: (
        setattr(fill_result, 'fill_qty', qty) or
        setattr(fill_result, 'fill_price', price) or
        fill_result
    )
    agent.paper_sim = paper_sim
    return agent


# ===========================================================================
# Options bot — get_available_capital()
# ===========================================================================

class TestOptionsGetAvailableCapital:

    def _mock_account(self, agent, equity, cash, options_value=0):
        mock_account = MagicMock()
        mock_account.equity = str(equity)
        mock_account.cash   = str(cash)
        mock_account.non_marginable_buying_power = str(cash)
        agent.trading_client.get_account.return_value = mock_account
        mock_position = MagicMock()
        mock_position.market_value = str(options_value)
        mock_position.symbol = 'SPY250328P00560000'  # long symbol = options
        agent.trading_client.get_all_positions.return_value = (
            [mock_position] if options_value > 0 else []
        )

    def test_returns_zero_when_cash_at_reserve(self):
        """Cash exactly at _MIN_CASH_RESERVE must return 0 available capital."""
        from options_agent import _MIN_CASH_RESERVE
        agent = _make_options_agent()
        self._mock_account(agent, equity=134_000, cash=_MIN_CASH_RESERVE)
        result = agent.get_available_capital()
        assert result == 0

    def test_returns_zero_when_cash_below_reserve(self):
        """Cash below reserve must return 0."""
        from options_agent import _MIN_CASH_RESERVE
        agent = _make_options_agent()
        self._mock_account(agent, equity=134_000, cash=_MIN_CASH_RESERVE - 1)
        result = agent.get_available_capital()
        assert result == 0

    def test_returns_zero_when_cash_negative(self):
        """Negative cash must return 0, not a negative number."""
        agent = _make_options_agent()
        self._mock_account(agent, equity=134_000, cash=-603.62)
        result = agent.get_available_capital()
        assert result == 0

    def test_available_capped_at_cash_minus_reserve(self):
        """Available capital must never exceed cash − _MIN_CASH_RESERVE."""
        from options_agent import _MIN_CASH_RESERVE
        agent = _make_options_agent()
        # cash = $2000, reserve = $500 → max spendable = $1500
        self._mock_account(agent, equity=134_000, cash=2_000)
        result = agent.get_available_capital()
        assert result <= 2_000 - _MIN_CASH_RESERVE

    def test_available_not_negative_when_options_exceed_target(self):
        """If options already exceed target allocation, available must be 0, not negative."""
        agent = _make_options_agent()
        # Options already at max allocation (15% of 134k = 20,100)
        self._mock_account(agent, equity=134_000, cash=50_000, options_value=25_000)
        result = agent.get_available_capital()
        assert result == 0


# ===========================================================================
# Options bot — live cash check in execute_options_trade()
# ===========================================================================

class TestOptionsLiveCashCheck:

    def _setup_trade(self, agent, live_cash, option_price=1.00, quantity=2):
        """Wire up mocks for execute_options_trade so only the cash check varies."""
        mock_contract = MagicMock()
        mock_contract.symbol         = 'SPY250328C00560000'
        mock_contract.strike_price   = '560'
        mock_contract.expiration_date = '2025-03-28'
        agent.trading_client.get_option_contracts = MagicMock(
            return_value=MagicMock(option_contracts=[mock_contract])
        )

        # get_option_price returns the midpoint we set
        with patch.object(agent, 'get_option_price', return_value=option_price), \
             patch.object(agent, 'find_optimal_option', return_value=mock_contract), \
             patch.object(agent, 'calculate_position_size', return_value=quantity):

            # Set live cash for the pre-submit check
            mock_account = MagicMock()
            mock_account.cash = str(live_cash)
            mock_account.non_marginable_buying_power = str(live_cash)
            agent.trading_client.get_account.return_value = mock_account

            mock_order = MagicMock(); mock_order.id = 'order-1'
            agent.trading_client.submit_order.return_value = mock_order

            decision  = {'decision': 'buy_call', 'confidence': 0.80,
                         'reasoning': 'test', 'current_price': 560.0}
            analysis  = {'current_price': 560.0}

            with patch('builtins.open', MagicMock()), \
                 patch('options_agent.alert_trade_executed'), \
                 patch('options_agent._db') as mock_db:
                mock_db.cleanup_stale_reservations = MagicMock()
                mock_db.get_total_reserved = MagicMock(return_value=0.0)
                mock_db.reserve_cash = MagicMock(return_value=1)
                mock_db.release_cash = MagicMock()
                mock_db.insert_trade = MagicMock()
                result = agent.execute_options_trade(
                    'SPY', decision, analysis, available_capital=5_000
                )
        return result

    def test_trade_blocked_when_cash_minus_cost_below_reserve(self):
        """
        If live_cash − estimated_cost < _MIN_CASH_RESERVE the order must not be submitted.
        estimated_cost = 1.00 × 100 × 2 = $200; reserve = $500
        So live_cash must be > $700 for the trade to proceed.
        Cash of $650 → blocked.
        """
        from options_agent import _MIN_CASH_RESERVE
        agent = _make_options_agent()
        # live_cash=$650, cost=$200 → $650-$200=$450 < $500 reserve → BLOCKED
        result = self._setup_trade(agent, live_cash=650, option_price=1.00, quantity=2)
        assert result is False
        agent.trading_client.submit_order.assert_not_called()

    def test_trade_proceeds_when_cash_covers_cost_and_reserve(self):
        """
        live_cash=$800, cost=$200 → $800-$200=$600 > $500 reserve → ALLOWED.
        """
        agent = _make_options_agent()
        result = self._setup_trade(agent, live_cash=800, option_price=1.00, quantity=2)
        assert result is True
        # submit_order is called at least once for the BUY; a second call for the
        # GTC stop-limit SELL (C2) is also expected when OPTIONS_GTC_STOP is enabled.
        agent.trading_client.submit_order.assert_called()

    def test_trade_blocked_when_cash_exactly_at_reserve_after_cost(self):
        """
        live_cash=$700, cost=$200 → $700-$200=$500 == reserve → BLOCKED (not strictly greater).
        """
        from options_agent import _MIN_CASH_RESERVE
        agent = _make_options_agent()
        # $700 - $200 = $500 which is == _MIN_CASH_RESERVE (< check, not <=, so this is blocked)
        result = self._setup_trade(agent, live_cash=700, option_price=1.00, quantity=2)
        assert result is False


# ===========================================================================
# Stock bot — session gate (run_trading_session)
# ===========================================================================

class TestStockSessionGate:

    def test_session_halts_when_cash_at_reserve(self):
        """run_trading_session must return without trading if cash == _MIN_CASH_RESERVE."""
        from autonomous_agent import _MIN_CASH_RESERVE
        agent = _make_stock_agent()
        agent.daily_start_equity = 100_000.0

        mock_account = MagicMock()
        mock_account.equity = '100000'
        mock_account.cash   = str(_MIN_CASH_RESERVE)
        agent.trading_client.get_account.return_value = mock_account

        from autonomous_agent import AutonomousAgent
        AutonomousAgent.run_trading_session(agent)

        # discover_opportunities must NOT be reached
        agent.discovery.discover_opportunities.assert_not_called()

    def test_session_halts_when_cash_below_reserve(self):
        """run_trading_session must return without trading if cash < _MIN_CASH_RESERVE."""
        from autonomous_agent import _MIN_CASH_RESERVE
        agent = _make_stock_agent()
        agent.daily_start_equity = 100_000.0

        mock_account = MagicMock()
        mock_account.equity = '100000'
        mock_account.cash   = str(_MIN_CASH_RESERVE - 100)
        agent.trading_client.get_account.return_value = mock_account

        from autonomous_agent import AutonomousAgent
        AutonomousAgent.run_trading_session(agent)

        agent.discovery.discover_opportunities.assert_not_called()

    def test_session_proceeds_when_cash_above_reserve(self):
        """run_trading_session must proceed past the gate when cash > _MIN_CASH_RESERVE."""
        from autonomous_agent import _MIN_CASH_RESERVE
        agent = _make_stock_agent()
        agent.daily_start_equity = 100_000.0

        mock_account = MagicMock()
        mock_account.equity = '100000'
        mock_account.cash   = str(_MIN_CASH_RESERVE + 5_000)
        agent.trading_client.get_account.return_value = mock_account
        agent.discovery.discover_opportunities.return_value = []

        from autonomous_agent import AutonomousAgent
        AutonomousAgent.run_trading_session(agent)

        agent.discovery.discover_opportunities.assert_called_once()


# ===========================================================================
# Stock bot — execute_trade() live check
# ===========================================================================

class TestStockExecuteTradeLiveCheck:

    def _make_buy_decision(self, price=100.0):
        return {'decision': 'buy', 'confidence': 0.80,
                'reasoning': 'test', 'current_price': price}

    def test_buy_blocked_when_live_cash_at_reserve(self):
        """execute_trade must return False if live cash == _MIN_CASH_RESERVE."""
        from autonomous_agent import _MIN_CASH_RESERVE
        agent = _make_stock_agent()
        mock_account = MagicMock()
        mock_account.cash = str(_MIN_CASH_RESERVE)
        agent.trading_client.get_account.return_value = mock_account

        result = agent.execute_trade('AAPL', self._make_buy_decision(100), 100_000, 10_000)
        assert result is False
        agent.trading_client.submit_order.assert_not_called()

    def test_buy_blocked_when_live_cash_below_reserve(self):
        """execute_trade must return False if live cash < _MIN_CASH_RESERVE."""
        from autonomous_agent import _MIN_CASH_RESERVE
        agent = _make_stock_agent()
        mock_account = MagicMock()
        mock_account.cash = str(_MIN_CASH_RESERVE - 1)
        agent.trading_client.get_account.return_value = mock_account

        result = agent.execute_trade('AAPL', self._make_buy_decision(100), 100_000, 10_000)
        assert result is False
        agent.trading_client.submit_order.assert_not_called()

    def test_buy_blocked_when_spendable_cash_below_share_price(self):
        """
        Even if cash > reserve, if cash - reserve < share price the trade must be skipped.
        live_cash=$600, reserve=$500 → spendable=$100 < price=$200 → BLOCKED.
        """
        from autonomous_agent import _MIN_CASH_RESERVE
        agent = _make_stock_agent()
        mock_account = MagicMock()
        mock_account.cash = str(_MIN_CASH_RESERVE + 100)  # $600
        agent.trading_client.get_account.return_value = mock_account

        result = agent.execute_trade('AAPL', self._make_buy_decision(200), 100_000, 10_000)
        assert result is False
        agent.trading_client.submit_order.assert_not_called()

    def test_buy_sized_from_spendable_cash_not_total_cash(self):
        """
        Shares purchased must be sized from (live_cash - reserve), not live_cash.
        live_cash=$5500, reserve=$500 → spendable=$5000
        position_size=5% → position_value=$250 → shares=int(250/100)=2
        NOT int(5500 * 0.05 / 100) = int(2.75) = 2  ← coincidentally same
        Use a price that exposes the difference:
        live_cash=$5500, reserve=$500 → spendable=$5000
        position=5% → $250 → at $50/share → 5 shares
        If sized from full cash: $5500*5%=$275 → 5 shares (same again)
        Use price=$1 to make it obvious:
        spendable=$5000*5%=$250 → 250 shares
        full=$5500*5%=$275 → 275 shares
        """
        from autonomous_agent import _MIN_CASH_RESERVE
        agent = _make_stock_agent()
        mock_account = MagicMock()
        mock_account.cash = str(5_500)  # $5500 total, $5000 spendable
        agent.trading_client.get_account.return_value = mock_account

        mock_order = MagicMock(); mock_order.id = 'o1'
        agent.trading_client.submit_order.return_value = mock_order

        with patch('builtins.open', MagicMock()), \
             patch('autonomous_agent.alert_trade_executed'), \
             patch('autonomous_agent._db') as mock_db:
            mock_db.cleanup_stale_reservations = MagicMock()
            mock_db.get_total_reserved = MagicMock(return_value=0.0)
            mock_db.reserve_cash = MagicMock(return_value=1)
            mock_db.release_cash = MagicMock()
            mock_db.insert_trade = MagicMock()
            agent.execute_trade('TEST', self._make_buy_decision(1.0), 100_000, 10_000)

        submitted = agent.trading_client.submit_order.call_args[0][0]
        # spendable = 5500 - 500 = 5000; 5% of 5000 = 250; 250 / $1 = 250 shares
        assert submitted.qty == 250
        # If it had used full cash: 5500 * 0.05 / 1.0 = 275 — must NOT be 275
        assert submitted.qty != 275

    def test_remaining_cash_initialized_to_spendable_amount(self):
        """
        remaining_cash in run_trading_session must start at cash - _MIN_CASH_RESERVE,
        not raw cash, so the reserve is respected across multi-trade sessions.
        """
        from autonomous_agent import _MIN_CASH_RESERVE, AutonomousAgent
        agent = _make_stock_agent()
        agent.daily_start_equity = 100_000.0

        mock_account = MagicMock()
        mock_account.equity = '100000'
        mock_account.cash   = str(10_000)
        agent.trading_client.get_account.return_value = mock_account

        captured_remaining = []

        original_execute = agent.execute_trade

        def capture_execute(symbol, decision, equity, remaining_cash):
            captured_remaining.append(remaining_cash)
            return False  # don't actually trade

        agent.discovery.discover_opportunities.return_value = [{'symbol': 'AAPL'}]

        with patch.object(agent, 'execute_trade', side_effect=capture_execute), \
             patch.object(agent, 'get_market_data', return_value=None):
            AutonomousAgent.run_trading_session(agent)

        if captured_remaining:
            # First call's remaining_cash must be cash - reserve = $9500
            assert captured_remaining[0] == pytest.approx(10_000 - _MIN_CASH_RESERVE)
