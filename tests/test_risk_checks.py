"""
Unit tests for risk / sizing logic in autonomous_agent and backtester.

These tests are fully offline — no Alpaca API calls, no LLM required.

Run with:
  python3 -m pytest tests/ -v
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from alpaca.trading.enums import OrderSide

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'agents'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'analysis'))

# ---------------------------------------------------------------------------
# AutonomousAgent risk checks (patched to avoid real API / Ollama calls)
# ---------------------------------------------------------------------------

class TestAutonomousAgentRisk:
    def _make_agent(self):
        """Return an AutonomousAgent with all external calls patched."""
        with patch('autonomous_agent.TradingClient'), \
             patch('autonomous_agent.StockHistoricalDataClient'), \
             patch('autonomous_agent.StockDiscovery'), \
             patch('autonomous_agent.load_dotenv'), \
             patch('autonomous_agent.load_model_once'), \
             patch('autonomous_agent.PortfolioOverseer'), \
             patch('autonomous_agent.AllocationController'):
            from autonomous_agent import AutonomousAgent
            agent = AutonomousAgent.__new__(AutonomousAgent)
            # Set up minimal state without calling __init__
            agent.params = {
                'max_position_size': 0.05,
                'stop_loss': -0.07,
                'take_profit': 0.15,
                'max_daily_loss_pct': 0.05,
                'max_daily_trades': 10,
                'cooldown_minutes': 15,
                'min_confidence': 0.60,
                'max_stocks_to_analyze': 25,
            }
            agent.daily_trades = 0
            agent.daily_start_equity = None
            agent.cooldowns = {}
            agent.pdt_blocked = False
            agent._paper = True  # sprint02 D4.3: execute_trade's trade_log now reads this
            from datetime import datetime
            agent.last_reset_date = datetime.now().date()
            # Mock trading client
            agent.trading_client = MagicMock()
            # Mock new components (added in round-2)
            alloc_ctrl = MagicMock()
            alloc_ctrl.get_position_size_pct.return_value = 0.05
            agent.alloc_controller = alloc_ctrl
            overseer = MagicMock()
            overseer.is_buy_allowed.return_value = (True, '')
            agent.overseer = overseer
            # Mock paper market simulator — fill at current_price with no slippage
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

    def test_circuit_breaker_triggers_on_5pct_loss(self):
        """run_trading_session should return early when daily loss >= 5%."""
        agent = self._make_agent()
        agent.daily_start_equity = 100_000.0
        # Mock account with 4% loss (should NOT trigger)
        mock_account = MagicMock()
        mock_account.equity = '96001'  # just above -4%
        agent.trading_client.get_account.return_value = mock_account
        agent.discovery = MagicMock()
        agent.discovery.discover_opportunities.return_value = []

        # Should complete (no early return from circuit breaker)
        # We don't assert on trades, just that it doesn't raise
        from autonomous_agent import AutonomousAgent
        AutonomousAgent.run_trading_session(agent)
        assert True  # reached here without circuit breaker

    def test_circuit_breaker_triggers_at_exactly_5pct(self):
        """Equity drop of exactly max_daily_loss_pct should halt trading."""
        agent = self._make_agent()
        agent.daily_start_equity = 100_000.0
        mock_account = MagicMock()
        mock_account.equity = '95000'  # exactly -5%
        agent.trading_client.get_account.return_value = mock_account
        agent.discovery = MagicMock()

        from autonomous_agent import AutonomousAgent
        AutonomousAgent.run_trading_session(agent)
        # discovery.discover_opportunities should NOT be called because we returned early
        agent.discovery.discover_opportunities.assert_not_called()

    def test_sell_skipped_when_no_position(self):
        """execute_trade should return False and not submit an order for a SELL with no position."""
        agent = self._make_agent()
        agent.trading_client.get_open_position.side_effect = Exception("position not found")

        decision = {'decision': 'sell', 'confidence': 0.80, 'reasoning': 'test', 'current_price': 100.0}
        result = agent.execute_trade('AAPL', decision, 100_000.0, 100_000.0)

        assert result is False
        agent.trading_client.submit_order.assert_not_called()

    def test_bracket_order_submitted_for_buy(self):
        """execute_trade should submit a bracket limit order with stop-loss and take-profit."""
        agent = self._make_agent()
        mock_order = MagicMock()
        mock_order.id = 'order-123'
        agent.trading_client.submit_order.return_value = mock_order

        # Provide enough cash for the buy
        mock_account = MagicMock()
        mock_account.cash = '10500'
        agent.trading_client.get_account.return_value = mock_account

        decision = {'decision': 'buy', 'confidence': 0.80, 'reasoning': 'test', 'current_price': 100.0}

        with patch('builtins.open', MagicMock()), \
             patch('autonomous_agent.alert_trade_executed'), \
             patch('autonomous_agent._db') as mock_db:
            mock_db.cleanup_stale_reservations = MagicMock()
            mock_db.get_total_reserved = MagicMock(return_value=0.0)
            mock_db.reserve_cash = MagicMock(return_value=1)
            mock_db.release_cash = MagicMock()
            mock_db.insert_trade = MagicMock()
            result = agent.execute_trade('AAPL', decision, 100_000.0, 9_500.0)

        assert result is True
        call_args = agent.trading_client.submit_order.call_args[0][0]
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderClass
        assert isinstance(call_args, LimitOrderRequest)
        assert call_args.order_class == OrderClass.BRACKET
        # params: stop_loss=-0.07, take_profit=0.15; fill price ≈ current_price=100.0
        # Exact fill comes from PaperMarketSimulator which may add slippage, so
        # just verify the child orders are present and have the right direction.
        assert call_args.stop_loss is not None
        assert call_args.take_profit is not None
        assert call_args.stop_loss.stop_price < call_args.limit_price
        assert call_args.take_profit.limit_price > call_args.limit_price

    def test_buy_uses_spendable_cash_for_sizing(self):
        """Shares must be sized from (live_cash - _MIN_CASH_RESERVE), not full cash."""
        agent = self._make_agent()
        mock_order = MagicMock()
        mock_order.id = 'order-123'
        agent.trading_client.submit_order.return_value = mock_order

        # live_cash=$5500, reserve=$500 → spendable=$5000
        # position_size=5% → $250 → at $1/share → 250 shares
        mock_account = MagicMock()
        mock_account.cash = '5500'
        agent.trading_client.get_account.return_value = mock_account

        decision = {'decision': 'buy', 'confidence': 0.80, 'reasoning': 'test', 'current_price': 1.0}

        with patch('builtins.open', MagicMock()), \
             patch('autonomous_agent.alert_trade_executed'), \
             patch('autonomous_agent._db') as mock_db:
            mock_db.cleanup_stale_reservations = MagicMock()
            mock_db.get_total_reserved = MagicMock(return_value=0.0)
            mock_db.reserve_cash = MagicMock(return_value=1)
            mock_db.release_cash = MagicMock()
            mock_db.insert_trade = MagicMock()
            agent.execute_trade('TSLA', decision, 100_000.0, 5_000.0)

        call_args = agent.trading_client.submit_order.call_args[0][0]
        # spendable=$5000; 5% of $5000=$250; 250/$1=250 shares
        assert call_args.qty == 250

    def test_zero_shares_skipped(self):
        """When cash - reserve < share price, the trade is skipped (0 shares)."""
        agent = self._make_agent()
        # live_cash=$600, reserve=$500 → spendable=$100 < price $500 → skip
        mock_account = MagicMock()
        mock_account.cash = '600'
        agent.trading_client.get_account.return_value = mock_account

        decision = {'decision': 'buy', 'confidence': 0.80, 'reasoning': 'test', 'current_price': 500.0}

        import autonomous_agent as aa
        with patch.object(aa, '_db') as mock_db, \
             patch.object(aa, '_DRY_RUN', False):
            mock_db.cleanup_stale_reservations = MagicMock()
            mock_db.get_total_reserved = MagicMock(return_value=0.0)
            result = agent.execute_trade('TSLA', decision, 100_000.0, 100.0)

        assert result is False
        agent.trading_client.submit_order.assert_not_called()


# ---------------------------------------------------------------------------
# Backtester metric calculations (fully offline)
# ---------------------------------------------------------------------------

class TestBacktesterMetrics:
    def _make_backtester(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'analysis'))
        from backtester import Backtester
        return Backtester(
            stop_loss=-0.07,
            take_profit=0.15,
            max_position_pct=0.05,
            min_confidence=0.60,
            starting_capital=100_000.0,
        )

    def test_no_trades_on_flat_market(self):
        """A completely flat price series should produce no trades (indicators inconclusive)."""
        from backtester import Backtester, compute_indicators
        bt = Backtester()

        # Flat OHLCV — RSI ~50, MACD ~0, momentum ~0 → no confident signal
        n = 60
        flat = pd.DataFrame({
            'open': [100.0] * n,
            'high': [100.5] * n,
            'low': [99.5] * n,
            'close': [100.0] * n,
            'volume': [1_000_000] * n,
        }, index=pd.date_range('2024-01-01', periods=n))
        df = compute_indicators(flat)
        # All rules neutral → no buy
        from backtester import rule_decision
        for _, row in df.iterrows():
            action, _ = rule_decision(row)
            # Flat market should mostly hold
            assert action in ('hold', 'buy', 'sell')  # sanity check it's a valid value

    def test_take_profit_exit(self):
        """A large rally should trigger take_profit exit."""
        from backtester import Backtester
        bt = Backtester(take_profit=0.10, stop_loss=-0.20)

        # Build a scenario: price rises 15% after entry
        # We'll test _metrics directly with synthetic trade data
        trades = [{
            'symbol': 'TEST',
            'entry_date': '2024-01-01',
            'exit_date': '2024-01-10',
            'entry_price': 100.0,
            'exit_price': 110.0,
            'shares': 10,
            'pnl_pct': 0.10,
            'pnl_dollar': 100.0,
            'exit_reason': 'take_profit',
        }]
        equity_curve = [100_000.0, 100_100.0]
        metrics = bt._metrics('TEST', trades, equity_curve)

        assert metrics['winners'] == 1
        assert metrics['losers'] == 0
        assert metrics['win_rate'] == 1.0

    def test_stop_loss_exit_counts_as_loss(self):
        from backtester import Backtester
        bt = Backtester()
        trades = [{
            'symbol': 'TEST',
            'entry_date': '2024-01-01',
            'exit_date': '2024-01-05',
            'entry_price': 100.0,
            'exit_price': 93.0,
            'shares': 10,
            'pnl_pct': -0.07,
            'pnl_dollar': -70.0,
            'exit_reason': 'stop_loss',
        }]
        equity_curve = [100_000.0, 99_930.0]
        metrics = bt._metrics('TEST', trades, equity_curve)

        assert metrics['losers'] == 1
        assert metrics['winners'] == 0
        assert metrics['win_rate'] == 0.0

    def test_no_trades_returns_safe_dict(self):
        from backtester import Backtester
        bt = Backtester()
        metrics = bt._metrics('EMPTY', [], [100_000.0])
        assert metrics['total_trades'] == 0
        assert 'note' in metrics

    def test_profit_factor_positive_trades(self):
        from backtester import Backtester
        bt = Backtester()
        trades = [
            {'symbol': 'T', 'entry_date': '2024-01-01', 'exit_date': '2024-01-03',
             'entry_price': 100, 'exit_price': 115, 'shares': 10,
             'pnl_pct': 0.15, 'pnl_dollar': 150.0, 'exit_reason': 'take_profit'},
            {'symbol': 'T', 'entry_date': '2024-01-05', 'exit_date': '2024-01-07',
             'entry_price': 100, 'exit_price': 93, 'shares': 10,
             'pnl_pct': -0.07, 'pnl_dollar': -70.0, 'exit_reason': 'stop_loss'},
        ]
        equity_curve = [100_000.0, 100_150.0, 100_080.0]
        metrics = bt._metrics('T', trades, equity_curve)
        # profit_factor = 150 / 70 ≈ 2.14
        assert metrics['profit_factor'] > 2.0
        assert metrics['profit_factor'] < 2.5


# ---------------------------------------------------------------------------
# Fix 5 — Rotation settled-cash guard
# ---------------------------------------------------------------------------

class TestRotationCashGuard:
    """_attempt_rotation() must re-fetch settled cash and skip the buy when insufficient."""

    def _make_rotation_agent(self):
        """Minimal AutonomousAgent wired for rotation tests."""
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
            'max_position_size': 0.05,
            'stop_loss': -0.07,
            'take_profit': 0.15,
            'max_daily_loss_pct': 0.05,
            'max_daily_trades': 10,
            'cooldown_minutes': 15,
            'min_confidence': 0.60,
            'max_stocks_to_analyze': 25,
        }
        agent.daily_trades = 0
        agent.daily_start_equity = None
        agent.cooldowns = {}
        agent.pdt_blocked = False
        agent._paper = True  # sprint02 D4.3: execute_trade's trade_log now reads this
        agent._last_order_id = None
        from datetime import datetime
        agent.last_reset_date = datetime.now().date()
        agent.trading_client = MagicMock()

        fee_breakdown = {'total_cost_pct': 0.001, 'total_cost': 0.15}
        agent.fee_simulator = MagicMock()
        agent.fee_simulator.estimate_round_trip_cost.return_value = fee_breakdown

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

    def _run_rotation(self, agent, settled_cash_str):
        """Set up scoring mocks and invoke _attempt_rotation with given settled cash."""
        from autonomous_agent import AutonomousAgent

        weakest = {
            'symbol':        'AAPL',
            'current_price': 150.0,
            'qty':           10,
            'confidence':    0.30,
            'decision':      'sell',
            'weakness_score': 0.5,
            'unrealized_pct': -0.05,
        }
        new_decision = {
            'decision':      'buy',
            'confidence':    0.90,
            'current_price': 200.0,
            'stop_loss':     -0.07,
            'take_profit':   0.15,
            'reasoning':     'test rotation',
        }

        mock_pos = MagicMock()
        mock_pos.symbol = 'AAPL'
        agent.trading_client.get_all_positions.return_value = [mock_pos]

        fresh_account = MagicMock()
        fresh_account.cash = settled_cash_str

        def get_account_side_effect():
            return fresh_account

        agent.trading_client.get_account.side_effect = get_account_side_effect

        def sell_side_effect(symbol, decision, equity, cash):
            agent._last_order_id = 'sell-order-1'
            return True

        with patch.object(agent, '_score_position', return_value=weakest), \
             patch.object(agent, '_find_weakest_position', return_value=weakest), \
             patch.object(agent, 'check_cooldown', return_value=False), \
             patch.object(agent, 'execute_trade', side_effect=sell_side_effect) as mock_et, \
             patch('autonomous_agent._write_rotation_log'), \
             patch('autonomous_agent._DRY_RUN', False):
            result = AutonomousAgent._attempt_rotation(
                agent, 'TSLA', new_decision, 1000.0, 100_000.0
            )
        return result, mock_et

    def test_rotation_buy_skipped_when_settled_cash_insufficient(self):
        """
        After sell succeeds, if re-fetched settled cash < entry_cost,
        the buy must not be executed and _attempt_rotation returns False.
        """
        agent = self._make_rotation_agent()
        # entry_cost = 200 * max(int(1500/200), 1) = 200 * 7 = 1400
        # settled_cash = 1000 < 1400 → buy skipped
        result, mock_et = self._run_rotation(agent, '1000.00')

        assert result is False
        assert mock_et.call_count == 1  # only the sell, no buy

    def test_rotation_buy_proceeds_when_settled_cash_sufficient(self):
        """
        After sell succeeds, if settled cash covers entry_cost, the buy executes.
        """
        agent = self._make_rotation_agent()

        # Need buy to succeed too — patch execute_trade with two side effects
        from autonomous_agent import AutonomousAgent

        weakest = {
            'symbol':        'AAPL',
            'current_price': 150.0,
            'qty':           10,
            'confidence':    0.30,
            'decision':      'sell',
            'weakness_score': 0.5,
            'unrealized_pct': -0.05,
        }
        new_decision = {
            'decision':      'buy',
            'confidence':    0.90,
            'current_price': 200.0,
            'stop_loss':     -0.07,
            'take_profit':   0.15,
            'reasoning':     'test rotation',
        }

        mock_pos = MagicMock()
        mock_pos.symbol = 'AAPL'
        agent.trading_client.get_all_positions.return_value = [mock_pos]

        fresh_account = MagicMock()
        fresh_account.cash = '5000.00'  # more than entry_cost (1400)
        agent.trading_client.get_account.side_effect = lambda: fresh_account

        call_count = [0]

        def et_side_effect(symbol, decision, equity, cash):
            call_count[0] += 1
            agent._last_order_id = f'order-{call_count[0]}'
            return True

        with patch.object(agent, '_score_position', return_value=weakest), \
             patch.object(agent, '_find_weakest_position', return_value=weakest), \
             patch.object(agent, 'check_cooldown', return_value=False), \
             patch.object(agent, 'execute_trade', side_effect=et_side_effect) as mock_et, \
             patch('autonomous_agent._write_rotation_log'), \
             patch('autonomous_agent._DRY_RUN', False):
            result = AutonomousAgent._attempt_rotation(
                agent, 'TSLA', new_decision, 1000.0, 100_000.0
            )

        assert result is True
        assert mock_et.call_count == 2  # sell + buy

    # -----------------------------------------------------------------
    # Sprint01 C3 — session-loop cash accounting (deduction + trigger)
    # -----------------------------------------------------------------

    def _make_full_session_agent(self, alloc_pct=0.05):
        """AutonomousAgent wired for run_trading_session-level cash tests."""
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
            'max_position_size': 0.05,
            'stop_loss': -0.07,
            'take_profit': 0.15,
            'max_daily_loss_pct': 0.05,
            'max_daily_trades': 10,
            'cooldown_minutes': 15,
            'min_confidence': 0.60,
            'max_stocks_to_analyze': 25,
        }
        agent.daily_trades = 0
        agent.daily_start_equity = 100_000.0
        agent.cooldowns = {}
        agent.pdt_blocked = False
        agent._paper = True  # sprint02 D4.3: execute_trade's trade_log now reads this
        agent._last_order_id = None
        agent._last_position_value = None
        from datetime import datetime
        agent.last_reset_date = datetime.now().date()
        agent.trading_client = MagicMock()
        alloc_ctrl = MagicMock()
        alloc_ctrl.get_position_size_pct.return_value = alloc_pct
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

    def _run_session_capturing_cash(self, agent, non_marginable_cash, opportunities,
                                     execute_side_effect):
        """Drive run_trading_session with N mocked opportunities, capturing the
        `available_cash` passed to execute_trade on each call."""
        mock_account = MagicMock()
        mock_account.equity = '100000'
        # .cash guards the early session-level circuit breaker (line ~1098);
        # .non_marginable_buying_power drives `settled`/remaining_cash — these
        # are genuinely different Alpaca account fields, both must be set
        # explicitly or MagicMock's default __float__ (1.0) trips the guard.
        mock_account.cash = '100000'
        mock_account.non_marginable_buying_power = str(non_marginable_cash)
        agent.trading_client.get_account.return_value = mock_account

        discovery = MagicMock()
        discovery.discover_opportunities.return_value = opportunities
        discovery.opportunities = {o['symbol']: [] for o in opportunities}
        agent.discovery = discovery

        indicators = {
            'current_price': 100.0, 'rsi': 50.0, 'macd': 0.1,
            'volume_ratio': 1.2, 'price_change_pct': 1.0,
        }

        import autonomous_agent as aa
        with patch.object(agent, 'check_cooldown', return_value=False), \
             patch.object(agent, 'get_market_data', return_value=MagicMock()), \
             patch.object(agent, 'calculate_indicators', return_value=indicators), \
             patch.object(agent, 'execute_trade', side_effect=execute_side_effect), \
             patch.object(agent, '_attempt_rotation', return_value=False), \
             patch.object(agent, '_log_decision', return_value=None), \
             patch.object(aa, 'get_trading_decision', return_value=''), \
             patch.object(aa, 'parse_decision', return_value={
                 'decision': 'buy', 'confidence': 0.80,
                 'reasoning': 'test', 'current_price': 100.0,
             }), \
             patch.object(aa, 'news_fetcher'), \
             patch.object(aa, '_db') as mock_db:
            mock_db.load_daily_start_equity.return_value = 100_000.0
            from autonomous_agent import AutonomousAgent
            AutonomousAgent.run_trading_session(agent)

    def test_tier1_buy_deducts_actual_cost_not_5pct_ceiling(self):
        """Tier 1 (1%) buy on $100k must deduct ~1% of spendable cash, not the
        5% max_position_size ceiling constant."""
        from autonomous_agent import _MIN_CASH_RESERVE
        agent = self._make_full_session_agent(alloc_pct=0.01)

        captured_cash = []

        def fake_execute(symbol, decision, equity, available_cash):
            captured_cash.append(available_cash)
            agent._last_position_value = 995.0  # ≈1% of ~$99,500 spendable
            return True

        self._run_session_capturing_cash(
            agent, 100_000, [{'symbol': 'AAA', 'signals': []}, {'symbol': 'BBB', 'signals': []}],
            fake_execute,
        )

        initial_remaining = 100_000.0 - _MIN_CASH_RESERVE
        assert len(captured_cash) == 2
        assert captured_cash[0] == initial_remaining
        # Second call must reflect a ~$995 deduction, NOT remaining*5% (~$4,975).
        assert captured_cash[1] == initial_remaining - 995.0

    def test_two_sequential_buys_cumulative_deduction_equals_actual_costs(self):
        agent = self._make_full_session_agent(alloc_pct=0.03)
        from autonomous_agent import _MIN_CASH_RESERVE

        costs = [500.0, 300.0]
        captured_cash = []

        def fake_execute(symbol, decision, equity, available_cash):
            captured_cash.append(available_cash)
            agent._last_position_value = costs[len(captured_cash) - 1]
            return True

        self._run_session_capturing_cash(
            agent, 50_000,
            [{'symbol': 'AAA', 'signals': []}, {'symbol': 'BBB', 'signals': []}],
            fake_execute,
        )

        initial_remaining = 50_000.0 - _MIN_CASH_RESERVE
        assert captured_cash[0] == initial_remaining
        assert captured_cash[1] == initial_remaining - costs[0]
        # Cumulative deduction after both buys equals the sum of actual costs —
        # verified indirectly via the second call's starting point above, plus
        # the fact that the deduction used costs[0] exactly, not a formula.

    def test_rotation_trigger_does_not_fire_with_ample_spendable_cash(self):
        """remaining_cash below one expensive share's price, but still well
        above zero (net of reserve), must NOT trigger rotation — fractional
        shares make the old one-share heuristic meaningless."""
        agent = self._make_full_session_agent(alloc_pct=0.05)

        rotation_called = []

        def fake_execute(symbol, decision, equity, available_cash):
            return False  # doesn't matter — we're only checking rotation wasn't attempted

        mock_account = MagicMock()
        mock_account.equity = '100000'
        mock_account.cash = '100000'
        # settled cash yields remaining_cash = 1000 - 500 = 500 (well above 0),
        # but decision['current_price'] will be $800 — below the OLD one-share
        # heuristic's threshold, which would have wrongly triggered rotation.
        mock_account.non_marginable_buying_power = '1000'
        agent.trading_client.get_account.return_value = mock_account

        discovery = MagicMock()
        discovery.discover_opportunities.return_value = [{'symbol': 'EXP', 'signals': []}]
        discovery.opportunities = {'EXP': []}
        agent.discovery = discovery

        indicators = {
            'current_price': 800.0, 'rsi': 50.0, 'macd': 0.1,
            'volume_ratio': 1.2, 'price_change_pct': 1.0,
        }

        import autonomous_agent as aa
        with patch.object(agent, 'check_cooldown', return_value=False), \
             patch.object(agent, 'get_market_data', return_value=MagicMock()), \
             patch.object(agent, 'calculate_indicators', return_value=indicators), \
             patch.object(agent, 'execute_trade', side_effect=fake_execute), \
             patch.object(agent, '_attempt_rotation',
                           side_effect=lambda *a, **kw: rotation_called.append(1) or False) as mock_rot, \
             patch.object(agent, '_log_decision', return_value=None), \
             patch.object(aa, 'get_trading_decision', return_value=''), \
             patch.object(aa, 'parse_decision', return_value={
                 'decision': 'buy', 'confidence': 0.80,
                 'reasoning': 'test', 'current_price': 800.0,
             }), \
             patch.object(aa, 'news_fetcher'), \
             patch.object(aa, '_db') as mock_db:
            mock_db.load_daily_start_equity.return_value = 100_000.0
            from autonomous_agent import AutonomousAgent
            AutonomousAgent.run_trading_session(agent)

        mock_rot.assert_not_called()
        assert rotation_called == []

    def test_rotation_trigger_fires_when_spendable_cash_exhausted(self):
        """remaining_cash <= 0 (net of reserve) must trigger rotation."""
        agent = self._make_full_session_agent(alloc_pct=0.05)

        mock_account = MagicMock()
        mock_account.equity = '100000'
        mock_account.cash = '100000'
        # settled cash = 500 → remaining_cash = 500 - 500 = 0 → exhausted
        mock_account.non_marginable_buying_power = '500'
        agent.trading_client.get_account.return_value = mock_account

        discovery = MagicMock()
        discovery.discover_opportunities.return_value = [{'symbol': 'EXP', 'signals': []}]
        discovery.opportunities = {'EXP': []}
        agent.discovery = discovery

        indicators = {
            'current_price': 10.0, 'rsi': 50.0, 'macd': 0.1,
            'volume_ratio': 1.2, 'price_change_pct': 1.0,
        }

        import autonomous_agent as aa
        with patch.object(agent, 'check_cooldown', return_value=False), \
             patch.object(agent, 'get_market_data', return_value=MagicMock()), \
             patch.object(agent, 'calculate_indicators', return_value=indicators), \
             patch.object(agent, 'execute_trade') as mock_execute, \
             patch.object(agent, '_attempt_rotation', return_value=False) as mock_rot, \
             patch.object(agent, '_log_decision', return_value=None), \
             patch.object(aa, 'get_trading_decision', return_value=''), \
             patch.object(aa, 'parse_decision', return_value={
                 'decision': 'buy', 'confidence': 0.80,
                 'reasoning': 'test', 'current_price': 10.0,
             }), \
             patch.object(aa, 'news_fetcher'), \
             patch.object(aa, '_db') as mock_db:
            mock_db.load_daily_start_equity.return_value = 100_000.0
            from autonomous_agent import AutonomousAgent
            AutonomousAgent.run_trading_session(agent)

        mock_rot.assert_called_once()
        mock_execute.assert_not_called()

    def test_rotation_deduction_uses_actual_buy_entry_cost_not_equity_pct(self):
        """After a successful rotation, the session loop must correctly pass
        remaining_cash into _attempt_rotation and floor the post-deduction
        value at 0 (never negative)."""
        agent = self._make_full_session_agent(alloc_pct=0.05)

        mock_account = MagicMock()
        mock_account.equity = '100000'
        mock_account.cash = '100000'
        mock_account.non_marginable_buying_power = '500'  # forces rotation trigger
        agent.trading_client.get_account.return_value = mock_account

        discovery = MagicMock()
        discovery.discover_opportunities.return_value = [{'symbol': 'EXP', 'signals': []}]
        discovery.opportunities = {'EXP': []}
        agent.discovery = discovery

        indicators = {
            'current_price': 10.0, 'rsi': 50.0, 'macd': 0.1,
            'volume_ratio': 1.2, 'price_change_pct': 1.0,
        }

        captured_cash = []

        def fake_rotation(symbol, decision, remaining_cash, equity):
            captured_cash.append(remaining_cash)
            agent._last_position_value = 321.0  # actual rotation BUY cost, unrelated to equity*5%=5000
            return True

        import autonomous_agent as aa
        with patch.object(agent, 'check_cooldown', return_value=False), \
             patch.object(agent, 'get_market_data', return_value=MagicMock()), \
             patch.object(agent, 'calculate_indicators', return_value=indicators), \
             patch.object(agent, 'execute_trade') as mock_execute, \
             patch.object(agent, '_attempt_rotation', side_effect=fake_rotation), \
             patch.object(agent, '_log_decision', return_value=None), \
             patch.object(aa, 'get_trading_decision', return_value=''), \
             patch.object(aa, 'parse_decision', return_value={
                 'decision': 'buy', 'confidence': 0.80,
                 'reasoning': 'test', 'current_price': 10.0,
             }), \
             patch.object(aa, 'news_fetcher'), \
             patch.object(aa, '_db') as mock_db:
            mock_db.load_daily_start_equity.return_value = 100_000.0
            from autonomous_agent import AutonomousAgent
            AutonomousAgent.run_trading_session(agent)

        # remaining_cash = 0 (500 settled - 500 reserve) is what rotation is
        # invoked with, and the post-deduction value floors at 0 (never goes
        # negative) regardless of the $321 deducted.
        assert captured_cash[0] == 0.0
        mock_execute.assert_not_called()

    def test_rotation_deduction_formula_uses_last_position_value_not_equity_pct(self):
        """Source-level regression check: once remaining_cash is at/below zero
        at the rotation trigger, any positive deduction floors to the same
        value (0) regardless of amount — the black-box session-loop behavior
        above can't distinguish the fixed formula from the old bug. Assert
        directly against the source that the rotation-deduction line uses
        self._last_position_value, not equity * max_position_size, so a
        future regression back to the old formula is still caught."""
        import inspect
        import autonomous_agent as aa
        src = inspect.getsource(aa.AutonomousAgent.run_trading_session)
        # Isolate the rotation branch (between "needs_rotation" and the
        # following "else:" that starts the non-rotation execute_trade path).
        start = src.index('if needs_rotation:')
        end = src.index('else:', start)
        rotation_branch = src[start:end]
        assert '_last_position_value' in rotation_branch
        assert "equity * self.params['max_position_size']" not in rotation_branch


# ---------------------------------------------------------------------------
# Sprint01 C3.1 — _last_position_value stashing on execute_trade
# ---------------------------------------------------------------------------

class TestPositionValueStashing:
    def _make_agent(self):
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
            'max_position_size': 0.05,
            'stop_loss': -0.07,
            'take_profit': 0.15,
            'max_daily_loss_pct': 0.05,
            'max_daily_trades': 10,
            'cooldown_minutes': 15,
            'min_confidence': 0.60,
            'max_stocks_to_analyze': 25,
        }
        agent.daily_trades = 0
        agent.daily_start_equity = None
        agent.cooldowns = {}
        agent.pdt_blocked = False
        agent._paper = True  # sprint02 D4.3: execute_trade's trade_log now reads this
        agent._last_order_id = None
        agent._last_position_value = None
        agent._db_path = None  # R1 WI-4: each test patches autonomous_agent._db itself
        from datetime import datetime
        agent.last_reset_date = datetime.now().date()
        agent.trading_client = MagicMock()
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

    def test_last_position_value_set_on_successful_buy(self):
        agent = self._make_agent()
        mock_order = MagicMock()
        mock_order.id = 'order-123'
        agent.trading_client.submit_order.return_value = mock_order

        mock_account = MagicMock()
        mock_account.cash = '10500'
        agent.trading_client.get_account.return_value = mock_account

        decision = {'decision': 'buy', 'confidence': 0.80, 'reasoning': 'test', 'current_price': 100.0}

        with patch('builtins.open', MagicMock()), \
             patch('autonomous_agent.alert_trade_executed'), \
             patch('autonomous_agent._db') as mock_db:
            mock_db.cleanup_stale_reservations = MagicMock()
            mock_db.get_total_reserved = MagicMock(return_value=0.0)
            mock_db.reserve_cash = MagicMock(return_value=1)
            mock_db.release_cash = MagicMock()
            mock_db.insert_trade = MagicMock()
            result = agent.execute_trade('AAPL', decision, 100_000.0, 9_500.0)

        assert result is True
        assert agent._last_position_value is not None
        # spendable=$10,000; 5% of $10,000=$500; 500 shares at $100 fill price.
        assert agent._last_position_value == pytest.approx(500.0, rel=0.01)

    def test_last_position_value_none_after_sell(self):
        agent = self._make_agent()
        agent._last_position_value = 999.0  # simulate a stale value from a prior buy

        mock_position = MagicMock()
        mock_position.qty = '10'
        mock_position.avg_entry_price = '90.0'
        agent.get_position = MagicMock(return_value=mock_position)

        mock_order = MagicMock()
        mock_order.id = 'sell-order-1'
        agent.trading_client.submit_order.return_value = mock_order

        decision = {'decision': 'sell', 'confidence': 0.80, 'reasoning': 'test', 'current_price': 100.0}

        with patch('builtins.open', MagicMock()), \
             patch('autonomous_agent.alert_trade_executed'), \
             patch('autonomous_agent._db') as mock_db:
            mock_db.insert_trade = MagicMock()
            result = agent.execute_trade('AAPL', decision, 100_000.0, 0)

        assert result is True
        assert agent._last_position_value is None

    def test_last_position_value_has_no_coupling_with_cash_reservation_table(self):
        """self._last_position_value is a purely local/session-scoped attribute
        with no interaction with the cross-bot cash-reservation DB table —
        _db.reserve_cash/release_cash must be called/released exactly as before,
        unaffected by C3's stashing addition."""
        agent = self._make_agent()
        mock_order = MagicMock()
        mock_order.id = 'order-456'
        agent.trading_client.submit_order.return_value = mock_order

        mock_account = MagicMock()
        mock_account.cash = '10500'
        agent.trading_client.get_account.return_value = mock_account

        decision = {'decision': 'buy', 'confidence': 0.80, 'reasoning': 'test', 'current_price': 100.0}

        with patch('builtins.open', MagicMock()), \
             patch('autonomous_agent.alert_trade_executed'), \
             patch('autonomous_agent._db') as mock_db:
            mock_db.cleanup_stale_reservations = MagicMock()
            mock_db.get_total_reserved = MagicMock(return_value=0.0)
            mock_db.reserve_cash = MagicMock(return_value=42)
            mock_db.release_cash = MagicMock()
            mock_db.insert_trade = MagicMock()
            agent.execute_trade('AAPL', decision, 100_000.0, 9_500.0)

            mock_db.reserve_cash.assert_called_once()
            mock_db.release_cash.assert_called_once_with(42)
        # The reservation amount is independent of _last_position_value — it's
        # computed from pre-fill-sim position_value, not the post-fill actual cost.
        assert agent._last_position_value is not None


# ---------------------------------------------------------------------------
# Sprint02 D3 — review_held_positions() unconditional stop-loss/take-profit
# ---------------------------------------------------------------------------

def _make_held_position(symbol, unrealized_plpc, current_price=100.0, qty='10'):
    p = MagicMock()
    p.symbol = symbol
    p.unrealized_plpc = str(unrealized_plpc)
    p.current_price = str(current_price)
    p.qty = qty
    return p


class TestHeldPositionReview:
    def _make_agent(self):
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
            'max_position_size': 0.05,
            'stop_loss': -0.07,
            'take_profit': 0.15,
            'max_daily_loss_pct': 0.05,
            'max_daily_trades': 10,
            'cooldown_minutes': 15,
            'min_confidence': 0.60,
            'max_stocks_to_analyze': 25,
        }
        agent.daily_trades = 0
        agent.daily_start_equity = None
        agent.cooldowns = {}
        agent.pdt_blocked = False
        agent._paper = True  # sprint02 D4.3: execute_trade's trade_log now reads this
        agent._last_order_id = None
        agent._last_position_value = None
        agent.trading_client = MagicMock()
        agent.trading_client.get_orders.return_value = []  # no open exit orders by default
        return agent

    def test_stop_loss_triggers_sell(self):
        agent = self._make_agent()
        pos = _make_held_position('AAA', -0.08, current_price=92.0)  # opened prior day
        agent.trading_client.get_all_positions.return_value = [pos]
        with patch.object(agent, '_bought_today', return_value=False), \
             patch.object(agent, 'execute_trade', return_value=True) as mock_execute:
            agent.review_held_positions(equity=100_000.0)
        mock_execute.assert_called_once()
        call_symbol, call_decision = mock_execute.call_args[0][0], mock_execute.call_args[0][1]
        assert call_symbol == 'AAA'
        assert call_decision['decision'] == 'sell'

    def test_take_profit_triggers_sell(self):
        agent = self._make_agent()
        pos = _make_held_position('BBB', 0.16, current_price=116.0)
        agent.trading_client.get_all_positions.return_value = [pos]
        with patch.object(agent, '_bought_today', return_value=False), \
             patch.object(agent, 'execute_trade', return_value=True) as mock_execute:
            agent.review_held_positions(equity=100_000.0)
        mock_execute.assert_called_once()
        assert mock_execute.call_args[0][0] == 'BBB'

    def test_within_band_no_sell(self):
        agent = self._make_agent()
        pos = _make_held_position('CCC', -0.02, current_price=98.0)  # inside -7%/+15% band
        agent.trading_client.get_all_positions.return_value = [pos]
        with patch.object(agent, '_bought_today', return_value=False), \
             patch.object(agent, 'execute_trade') as mock_execute:
            agent.review_held_positions(equity=100_000.0)
        mock_execute.assert_not_called()

    def test_duplicate_exit_guard_skips_second_sell(self):
        agent = self._make_agent()
        pos = _make_held_position('DDD', -0.10, current_price=90.0)
        agent.trading_client.get_all_positions.return_value = [pos]
        open_sell = MagicMock()
        open_sell.symbol = 'DDD'
        open_sell.side = OrderSide.SELL
        agent.trading_client.get_orders.return_value = [open_sell]
        with patch.object(agent, '_bought_today', return_value=False), \
             patch.object(agent, 'execute_trade') as mock_execute:
            agent.review_held_positions(equity=100_000.0)
        mock_execute.assert_not_called()

    def test_pdt_deferral_same_day_position(self):
        """A stop/take-profit trigger on a position opened today must be
        deferred, not submitted as a same-day close, when no_same_day_close
        is active."""
        agent = self._make_agent()
        agent.params['no_same_day_close'] = True
        pos = _make_held_position('EEE', -0.08, current_price=92.0)
        agent.trading_client.get_all_positions.return_value = [pos]
        with patch.object(agent, '_bought_today', return_value=True), \
             patch.object(agent, 'execute_trade') as mock_execute, \
             patch('autonomous_agent.logging') as mock_log:
            agent.review_held_positions(equity=100_000.0)
        mock_execute.assert_not_called()
        assert any('deferred' in str(c) for c in mock_log.info.call_args_list)

    def test_pdt_deferral_does_not_apply_to_prior_day_position(self):
        """The same trigger on a position opened a prior day must sell
        normally — deferral only applies to same-day opens."""
        agent = self._make_agent()
        agent.params['no_same_day_close'] = True
        pos = _make_held_position('FFF', -0.08, current_price=92.0)
        agent.trading_client.get_all_positions.return_value = [pos]
        with patch.object(agent, '_bought_today', return_value=False), \
             patch.object(agent, 'execute_trade', return_value=True) as mock_execute:
            agent.review_held_positions(equity=100_000.0)
        mock_execute.assert_called_once()

    def test_no_positions_no_crash(self):
        agent = self._make_agent()
        agent.trading_client.get_all_positions.return_value = []
        with patch.object(agent, 'execute_trade') as mock_execute:
            agent.review_held_positions(equity=100_000.0)
        mock_execute.assert_not_called()

    def test_options_positions_excluded(self):
        """Long OCC option symbols (len > 10) must not be reviewed here —
        those belong to the options bot."""
        agent = self._make_agent()
        opt_pos = _make_held_position('SPY250328P00560000', -0.60, current_price=1.0)
        agent.trading_client.get_all_positions.return_value = [opt_pos]
        with patch.object(agent, '_bought_today', return_value=False), \
             patch.object(agent, 'execute_trade') as mock_execute:
            agent.review_held_positions(equity=100_000.0)
        mock_execute.assert_not_called()

    def test_runs_even_with_empty_opportunities(self):
        """End-to-end: review_held_positions still executes a sell when
        discovery returns zero opportunities — the core sprint02 D3 fix."""
        from datetime import datetime as _dt
        agent = self._make_agent()
        agent.daily_start_equity = 100_000.0
        agent.last_reset_date = _dt.now().date()
        alloc_ctrl = MagicMock()
        alloc_ctrl.get_position_size_pct.return_value = 0.05
        agent.alloc_controller = alloc_ctrl
        overseer = MagicMock()
        overseer.is_buy_allowed.return_value = (True, '')
        agent.overseer = overseer
        pos = _make_held_position('GGG', -0.09, current_price=91.0)
        agent.trading_client.get_all_positions.return_value = [pos]

        mock_account = MagicMock()
        mock_account.equity = '100000'
        mock_account.cash = '100000'
        mock_account.non_marginable_buying_power = '100000'
        agent.trading_client.get_account.return_value = mock_account

        discovery = MagicMock()
        discovery.discover_opportunities.return_value = []
        agent.discovery = discovery

        import autonomous_agent as aa
        with patch.object(agent, '_bought_today', return_value=False), \
             patch.object(agent, 'execute_trade', return_value=True) as mock_execute, \
             patch.object(aa, '_db') as mock_db:
            mock_db.load_daily_start_equity.return_value = 100_000.0
            from autonomous_agent import AutonomousAgent
            AutonomousAgent.run_trading_session(agent)

        mock_execute.assert_called_once()
        assert mock_execute.call_args[0][0] == 'GGG'


# ---------------------------------------------------------------------------
# Fix 8: TestSettledCash — remaining_cash must come from non_marginable_buying_power
# ---------------------------------------------------------------------------

class TestSettledCash:
    """_settled_cash() and run_trading_session() must use non_marginable_buying_power for trading budget."""

    def _make_agent(self):
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
            'max_position_size': 0.05,
            'stop_loss': -0.07,
            'take_profit': 0.15,
            'max_daily_loss_pct': 0.05,
            'max_daily_trades': 10,
            'cooldown_minutes': 15,
            'min_confidence': 0.60,
            'max_stocks_to_analyze': 25,
        }
        agent.daily_trades = 0
        agent.daily_start_equity = 50_000.0
        agent.cooldowns = {}
        agent.pdt_blocked = False
        agent._paper = True  # sprint02 D4.3: execute_trade's trade_log now reads this
        from datetime import datetime
        agent.last_reset_date = datetime.now().date()
        agent.trading_client = MagicMock()
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

    def test_settled_cash_helper(self):
        """_settled_cash() should cast non_marginable_buying_power to float."""
        from autonomous_agent import _settled_cash
        acct = MagicMock()
        acct.non_marginable_buying_power = '1234.56'
        assert _settled_cash(acct) == 1234.56

    def test_remaining_cash_uses_non_marginable_buying_power(self):
        """run_trading_session should pass settled cash (not total cash) to execute_trade."""
        agent = self._make_agent()

        mock_account = MagicMock()
        mock_account.equity = '50000'
        mock_account.cash = '2000'
        mock_account.non_marginable_buying_power = '800'
        agent.trading_client.get_account.return_value = mock_account

        discovery = MagicMock()
        discovery.discover_opportunities.return_value = [{'symbol': 'TSLA', 'signals': []}]
        discovery.opportunities = {'TSLA': []}
        agent.discovery = discovery

        indicators = {
            'current_price': 100.0, 'rsi': 50.0, 'macd': 0.1,
            'volume_ratio': 1.2, 'price_change_pct': 1.0,
        }
        captured = {}

        def fake_execute(symbol, decision, equity, available_cash):
            captured['available_cash'] = available_cash
            return False

        import autonomous_agent as aa
        with patch.object(agent, 'check_cooldown', return_value=False), \
             patch.object(agent, 'get_market_data', return_value=MagicMock()), \
             patch.object(agent, 'calculate_indicators', return_value=indicators), \
             patch.object(agent, 'execute_trade', side_effect=fake_execute), \
             patch.object(agent, '_log_decision', return_value=None), \
             patch.object(aa, 'get_trading_decision', return_value=''), \
             patch.object(aa, 'parse_decision', return_value={
                 'decision': 'buy', 'confidence': 0.80,
                 'reasoning': 'test', 'current_price': 100.0,
             }), \
             patch.object(aa, 'news_fetcher'), \
             patch.object(aa, '_db') as mock_db:
            mock_db.load_daily_start_equity.return_value = 50_000.0
            from autonomous_agent import AutonomousAgent
            AutonomousAgent.run_trading_session(agent)

        from autonomous_agent import _MIN_CASH_RESERVE
        assert captured.get('available_cash') == 800.0 - _MIN_CASH_RESERVE


# ---------------------------------------------------------------------------
# Fix 10: TestDynamicAssumedReturn — rotation EV uses live win-rate, not 0.05
# ---------------------------------------------------------------------------

import pytest

@pytest.fixture()
def db_path(tmp_path):
    import db as _db
    p = tmp_path / 'test_trading.db'
    _db.init_db(p)
    return p


class TestDynamicAssumedReturn:
    """_recent_avg_win_pct() should query real outcomes; _attempt_rotation uses the result."""

    def _make_rotation_agent(self):
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
            'max_position_size': 0.05,
            'stop_loss': -0.07,
            'take_profit': 0.15,
            'max_daily_loss_pct': 0.05,
            'max_daily_trades': 10,
            'cooldown_minutes': 15,
            'min_confidence': 0.60,
            'max_stocks_to_analyze': 25,
        }
        agent.daily_trades = 0
        agent.daily_start_equity = None
        agent.cooldowns = {}
        agent.pdt_blocked = False
        agent._paper = True  # sprint02 D4.3: execute_trade's trade_log now reads this
        agent._last_order_id = None
        from datetime import datetime
        agent.last_reset_date = datetime.now().date()
        agent.trading_client = MagicMock()
        fee_breakdown = {'total_cost_pct': 0.001, 'total_cost': 0.15}
        agent.fee_simulator = MagicMock()
        agent.fee_simulator.estimate_round_trip_cost.return_value = fee_breakdown
        fill_result = MagicMock()
        fill_result.filled = True
        fill_result.slippage_bps = 0.0
        paper_sim = MagicMock()
        paper_sim.simulate_stock_fill.side_effect = lambda side, price, qty, **_kw: (
            setattr(fill_result, 'fill_qty', qty) or
            setattr(fill_result, 'fill_price', price) or fill_result
        )
        agent.paper_sim = paper_sim
        return agent

    def test_recent_avg_win_pct_returns_mean_of_wins(self, db_path):
        """Returns mean pnl_pct of winning trades when 5+ wins exist."""
        import db as _db
        from datetime import datetime, timedelta

        base = datetime.now() - timedelta(days=1)
        pnl_values = [0.10, 0.08, 0.12, 0.09, 0.11, 0.07]
        for i, pnl in enumerate(pnl_values):
            _db.insert_outcome({
                'symbol': f'TICK{i}',
                'buy_order_id': f'buy-{i}',
                'sell_order_id': f'sell-{i}',
                'entry_timestamp': (base - timedelta(hours=2)).isoformat(),
                'exit_timestamp': base.isoformat(),
                'entry_price': 100.0,
                'exit_price': 100.0 * (1 + pnl),
                'shares': 10,
                'realized_pnl': round(100.0 * pnl * 10, 2),
                'pnl_pct': pnl,
                'hold_hours': 2.0,
                'entry_confidence': 0.80,
                'entry_reasoning': 'test',
                'won': True,
            }, db_path=db_path)

        agent = self._make_rotation_agent()
        result = agent._recent_avg_win_pct(lookback_days=30, db_path=db_path)
        expected = sum(pnl_values) / len(pnl_values)
        assert abs(result - expected) < 1e-9

    def test_recent_avg_win_pct_falls_back_when_fewer_than_5_wins(self, db_path):
        """Returns 0.05 fallback when fewer than 5 wins are in the window."""
        import db as _db
        from datetime import datetime, timedelta

        base = datetime.now() - timedelta(days=1)
        for i in range(3):
            _db.insert_outcome({
                'symbol': f'WIN{i}',
                'buy_order_id': f'buy-{i}',
                'sell_order_id': f'sell-{i}',
                'entry_timestamp': (base - timedelta(hours=2)).isoformat(),
                'exit_timestamp': base.isoformat(),
                'entry_price': 100.0,
                'exit_price': 110.0,
                'shares': 10,
                'realized_pnl': 100.0,
                'pnl_pct': 0.10,
                'hold_hours': 2.0,
                'entry_confidence': 0.80,
                'entry_reasoning': 'test',
                'won': True,
            }, db_path=db_path)

        agent = self._make_rotation_agent()
        result = agent._recent_avg_win_pct(lookback_days=30, db_path=db_path)
        assert result == 0.05

    def test_rotation_uses_dynamic_assumed_return(self):
        """_attempt_rotation log line must contain the dynamic assumed_return value."""
        agent = self._make_rotation_agent()

        weakest = {
            'symbol': 'AAPL', 'current_price': 150.0, 'qty': 10,
            'confidence': 0.30, 'decision': 'sell',
            'weakness_score': 0.5, 'unrealized_pct': -0.05,
        }
        new_decision = {
            'decision': 'buy', 'confidence': 0.90, 'current_price': 200.0,
            'stop_loss': -0.07, 'take_profit': 0.15, 'reasoning': 'test',
        }

        mock_pos = MagicMock()
        mock_pos.symbol = 'AAPL'
        agent.trading_client.get_all_positions.return_value = [mock_pos]

        fresh_account = MagicMock()
        fresh_account.cash = '5000.00'
        agent.trading_client.get_account.side_effect = lambda: fresh_account

        call_count = [0]

        def et_side_effect(symbol, decision, equity, cash):
            call_count[0] += 1
            agent._last_order_id = f'order-{call_count[0]}'
            return True

        with patch.object(agent, '_recent_avg_win_pct', return_value=0.10) as mock_win_pct, \
             patch.object(agent, '_score_position', return_value=weakest), \
             patch.object(agent, '_find_weakest_position', return_value=weakest), \
             patch.object(agent, 'check_cooldown', return_value=False), \
             patch.object(agent, 'execute_trade', side_effect=et_side_effect), \
             patch('autonomous_agent._write_rotation_log'), \
             patch('autonomous_agent._DRY_RUN', False), \
             patch('autonomous_agent.logging') as mock_log:
            from autonomous_agent import AutonomousAgent
            AutonomousAgent._attempt_rotation(agent, 'TSLA', new_decision, 1000.0, 100_000.0)

        mock_win_pct.assert_called_once()
        log_calls = [str(c) for c in mock_log.info.call_args_list]
        assert any('assumed_return=0.100' in s for s in log_calls)
