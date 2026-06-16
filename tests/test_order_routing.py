"""
Regression tests for execute_trade() order routing.

Bugs fixed 2026-06-16 (silent since 2026-05-22):
  1. SELL orders were routed to the bracket path → Alpaca error 42210000
     ("take_profit.limit_price must be < stop_loss.stop_price").
     Fix: `side == OrderSide.SELL` added as first clause in `no_bracket`.
  2. `position_value` was only assigned in the BUY branch but read by the
     sanity-check at line 659 for both sides → UnboundLocalError on SELL (OXY).
     Fix: `position_value = shares * current_price` in the SELL branch.

These tests assert on the actual order object fields, not mock call counts.
"""
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'agents'))

_ENTRY_PRICE = 100.0
_POSITION_QTY = 25.0   # integer so BUY sizing also yields integer shares


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _make_agent(cash: float = 50_500.0, position_qty: float = _POSITION_QTY,
                avg_entry: float = 90.0, alloc_pct: float = 0.05) -> object:
    """Build a minimally-wired AutonomousAgent for order-routing tests.

    cash=50_500 chosen so:
      effective_cash = 50_500 - 0 reserved = 50_500
      spendable = 50_500 - 500 (_MIN_CASH_RESERVE) = 50_000
      position_value = 50_000 * 0.05 = 2_500
      shares = 2_500 / 100 = 25.0  ← integer → bracket-eligible for BUY
    """
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
        'stop_loss': -0.07,
        'take_profit': 0.15,
        'max_daily_trades': 10,
        'max_daily_loss_pct': 0.05,
        'cooldown_minutes': 0,
        'min_confidence': 0.60,
        # no_same_day_close intentionally absent — PDT path not under test
    }

    mock_account = MagicMock()
    mock_account.cash = str(cash)

    mock_pos = MagicMock()
    mock_pos.qty = str(position_qty)
    mock_pos.avg_entry_price = str(avg_entry)

    mock_submitted_order = MagicMock()
    mock_submitted_order.id = 'test-order-id'

    trading_client = MagicMock()
    trading_client.get_account.return_value = mock_account
    trading_client.get_open_position.return_value = mock_pos
    trading_client.submit_order.return_value = mock_submitted_order
    agent.trading_client = trading_client

    agent.daily_trades = 0
    agent.last_reset_date = datetime.now().date()
    agent.daily_start_equity = None
    agent.cooldowns = {}
    agent.pdt_blocked = False
    agent.discovery = MagicMock()
    agent._paper = True
    agent._last_order_id = None

    alloc_ctrl = MagicMock()
    alloc_ctrl.get_position_size_pct.return_value = alloc_pct
    agent.alloc_controller = alloc_ctrl

    overseer = MagicMock()
    overseer.is_buy_allowed.return_value = (True, '')
    agent.overseer = overseer

    fill_result = MagicMock()
    fill_result.filled = True
    fill_result.slippage_bps = 0.0

    def _sim_fill(side, price, qty, **kw):
        fill_result.fill_qty = qty
        fill_result.fill_price = price
        return fill_result

    paper_sim = MagicMock()
    paper_sim.simulate_stock_fill.side_effect = _sim_fill
    agent.paper_sim = paper_sim

    return agent


def _run(agent, decision: dict) -> bool:
    """Call execute_trade with all external dependencies mocked out."""
    with patch('autonomous_agent._db') as mock_db, \
         patch('autonomous_agent.alert_trade_executed'), \
         patch('autonomous_agent.alert_trade_failed'), \
         patch.object(agent, '_count_todays_roundtrips', return_value=0), \
         patch('builtins.open', mock_open()):
        mock_db.cleanup_stale_reservations.return_value = None
        mock_db.get_total_reserved.return_value = 0.0
        mock_db.reserve_cash.return_value = 'rsv-test'
        mock_db.release_cash.return_value = None
        mock_db.insert_trade.return_value = None
        return agent.execute_trade(
            'AAPL', decision, equity=100_000.0, available_cash=50_000.0
        ), agent.trading_client.submit_order.call_args


def _buy_decision(price: float = _ENTRY_PRICE) -> dict:
    return {
        'decision': 'buy', 'confidence': 0.80,
        'reasoning': 'Strong breakout.',
        'current_price': price,
        'indicators': {'avg_volume': 1_000_000, 'session_volume': 500_000},
    }


def _sell_decision(price: float = _ENTRY_PRICE) -> dict:
    return {
        'decision': 'sell', 'confidence': 0.80,
        'reasoning': 'Exit signal.',
        'current_price': price,
        'indicators': {'avg_volume': 1_000_000, 'session_volume': 500_000},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOrderRouting:

    def test_buy_produces_bracket_order(self):
        """BUY with integer shares → LimitOrderRequest with OrderClass.BRACKET."""
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderClass

        agent = _make_agent()
        ok, call_args = _run(agent, _buy_decision())

        assert ok is True
        order = call_args[0][0]
        assert isinstance(order, LimitOrderRequest), (
            f"BUY must submit LimitOrderRequest, got {type(order).__name__}"
        )
        assert order.order_class == OrderClass.BRACKET, (
            "BUY order must carry OrderClass.BRACKET"
        )
        assert order.stop_loss is not None, "BUY bracket must have stop_loss leg"
        assert order.take_profit is not None, "BUY bracket must have take_profit leg"

    def test_buy_bracket_price_ordering(self):
        """take_profit.limit_price > entry > stop_loss.stop_price for a BUY bracket."""
        from alpaca.trading.requests import LimitOrderRequest

        agent = _make_agent()
        ok, call_args = _run(agent, _buy_decision())

        assert ok is True
        order = call_args[0][0]
        entry = order.limit_price
        stop  = order.stop_loss.stop_price
        target = order.take_profit.limit_price
        assert target > entry > stop, (
            f"Expected target({target}) > entry({entry}) > stop({stop})"
        )

    def test_sell_produces_plain_market_order(self):
        """SELL → MarketOrderRequest, no OrderClass.BRACKET, no bracket legs."""
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderClass, OrderSide

        agent = _make_agent(position_qty=10.0)
        ok, call_args = _run(agent, _sell_decision())

        assert ok is True, "SELL must succeed"
        order = call_args[0][0]
        assert isinstance(order, MarketOrderRequest), (
            f"SELL must submit MarketOrderRequest, got {type(order).__name__}"
        )
        assert order.side == OrderSide.SELL
        order_class = getattr(order, 'order_class', None)
        assert order_class != OrderClass.BRACKET, (
            "SELL must NOT carry OrderClass.BRACKET (Alpaca error 42210000)"
        )
        assert getattr(order, 'stop_loss', None) is None, \
            "SELL must have no stop_loss leg"
        assert getattr(order, 'take_profit', None) is None, \
            "SELL must have no take_profit leg"

    def test_sell_qty_from_position_not_sizing_calc(self):
        """SELL qty = position.qty, not the allocation-controller formula."""
        agent = _make_agent(position_qty=7.5, avg_entry=85.0)
        ok, call_args = _run(agent, _sell_decision())

        assert ok is True
        order = call_args[0][0]
        # round(7.5, 9) → 7.5
        assert abs(float(order.qty) - 7.5) < 1e-9, (
            f"SELL qty must match position.qty=7.5, got {order.qty}"
        )

    def test_sell_position_value_unbound_regression(self):
        """Regression: position_value was unbound on SELL → UnboundLocalError (OXY 2026-06-16)."""
        # Before the fix, execute_trade caught UnboundLocalError and returned False.
        agent = _make_agent(position_qty=5.0, avg_entry=80.0)
        ok, _ = _run(agent, _sell_decision())
        assert ok is True, (
            "SELL must not raise UnboundLocalError for position_value"
        )

    def test_fractional_buy_still_gets_plain_order(self):
        """Fractional BUY (non-integer shares) routes to plain MarketOrderRequest."""
        from alpaca.trading.requests import MarketOrderRequest

        # $50_500 cash, 5% alloc, $50 price → 49 shares? Let me compute:
        # spendable = 50_500 - 500 = 50_000
        # position_value = 50_000 * 0.05 = 2_500
        # shares = 2_500 / 33 ≈ 75.75... fractional
        agent = _make_agent(cash=50_500.0)
        ok, call_args = _run(agent, _buy_decision(price=33.0))

        assert ok is True
        order = call_args[0][0]
        assert isinstance(order, MarketOrderRequest), (
            "Fractional BUY must submit MarketOrderRequest, not a bracket"
        )
