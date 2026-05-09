#!/usr/bin/env python3
"""
End-to-end mock tests for OptionsAgent.run_options_session().

No live API calls — every Alpaca/yfinance/LLM interaction is mocked.
Run from the project root with the venv active:
  source venv/bin/activate
  python3 test_options_session.py

Tests:
  1. Full happy path: AAPL analysis → buy_call → trade executed
  2. Take-profit exits existing position (≥50% gain)
  3. Stop-loss exits existing position (≤-50% loss)
  4. Low AI confidence → hold, no trade
  5. No option price returned → trade skipped
  6. Insufficient capital → no new trades
  7. Daily trade limit → no new trades after cap
"""
import os
import sys
import json
import types
import unittest
import logging
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, call
from pathlib import Path

import pandas as pd
import numpy as np

# ── Suppress log noise during tests ──────────────────────────────────────────
logging.disable(logging.CRITICAL)

# ── Stub heavy imports BEFORE importing OptionsAgent ─────────────────────────
dummy_lora = types.ModuleType('model_inference_lora')
dummy_lora.parse_decision = lambda r: {'decision': 'buy', 'confidence': 0.85, 'reasoning': r[:80]}
dummy_lora.get_trading_decision = lambda prompt, **kw: 'BUY_CALL. Confidence: 0.85.'
dummy_lora.load_model_once = lambda: None
sys.modules['model_inference_lora'] = dummy_lora

dummy_ollama = types.ModuleType('ollama')
dummy_ollama.generate = lambda **kw: {'response': 'BUY_CALL. Confidence: 0.85. Strong bullish momentum.'}
sys.modules['ollama'] = dummy_ollama

sys.path.insert(0, 'scripts')
sys.path.insert(0, 'scripts/agents')


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_price_df(base=180.0, days=65):
    """Build a realistic OHLCV DataFrame that satisfies all indicator calculations."""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
    n = len(dates)  # may differ from `days` if end falls on a weekend
    rng = np.random.default_rng(42)
    closes = base + np.cumsum(rng.normal(0, 1.5, n))
    df = pd.DataFrame({
        'open':   closes * 0.998,
        'high':   closes * 1.005,
        'low':    closes * 0.995,
        'close':  closes,
        'volume': rng.integers(20_000_000, 80_000_000, n),
    }, index=dates)
    return df


def make_account(equity='100000.00'):
    acct = MagicMock()
    acct.equity = equity
    acct.cash = equity
    acct.non_marginable_buying_power = equity
    return acct


def make_option_position(symbol, qty, unrealized_plpc):
    pos = MagicMock()
    pos.symbol = symbol
    pos.qty = str(qty)
    pos.unrealized_plpc = str(unrealized_plpc)
    pos.market_value = str(float(qty) * 360.0)  # arbitrary option market value
    pos.current_price = '3.60'
    return pos


def make_stock_position(symbol, qty, market_value='5000.00'):
    pos = MagicMock()
    pos.symbol = symbol
    pos.qty = str(qty)
    pos.unrealized_plpc = '0.05'
    pos.market_value = market_value
    return pos


def make_contract(symbol='AAPL270319C00207000', strike='207.00', exp='2026-03-19'):
    c = MagicMock()
    c.symbol = symbol
    c.strike_price = strike
    c.expiration_date = exp
    return c


def make_contracts_resp(contracts):
    resp = MagicMock()
    resp.option_contracts = contracts
    return resp


def make_quote(bid='3.50', ask='3.70'):
    q = MagicMock()
    q.bid_price = bid
    q.ask_price = ask
    return q


def make_order(order_id='test-order-abc123'):
    o = MagicMock()
    o.id = order_id
    return o


def build_agent(
    positions=None,
    contracts=None,
    quote_bid='3.50',
    quote_ask='3.70',
    equity='100000.00',
    lora_decision=None,
    ollama_response=None,
):
    """
    Construct an OptionsAgent with fully mocked Alpaca + yfinance + LLM.
    Returns (agent, mock_trading_client, mock_option_data_client).
    """
    from options_agent import OptionsAgent

    # --- Alpaca TradingClient ------------------------------------------------
    mock_tc = MagicMock()
    mock_tc.get_account.return_value = make_account(equity)
    mock_tc.get_all_positions.return_value = positions or []
    mock_tc.get_option_contracts.return_value = make_contracts_resp(
        contracts if contracts is not None else [make_contract()]
    )
    mock_tc.submit_order.return_value = make_order()

    # --- Alpaca OptionHistoricalDataClient -----------------------------------
    mock_odc = MagicMock()
    contract_sym = (contracts[0].symbol if contracts else 'AAPL270319C00207000')
    mock_odc.get_option_latest_quote.return_value = {
        contract_sym: make_quote(quote_bid, quote_ask)
    }

    # --- yfinance ------------------------------------------------------------
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = make_price_df()

    # --- LLM stubs -----------------------------------------------------------
    ollama_resp = ollama_response or 'BUY_CALL. Confidence: 0.85. Strong bullish momentum.'
    lora_dec = lora_decision or {'decision': 'buy', 'confidence': 0.85, 'reasoning': 'bullish'}

    with (
        patch('options_agent.TradingClient', return_value=mock_tc),
        patch('options_agent.OptionHistoricalDataClient', return_value=mock_odc),
        patch('options_agent.yf.Ticker', return_value=mock_ticker),
        patch('options_agent.get_trading_decision', return_value=ollama_resp),
        patch('options_agent.parse_decision', return_value=lora_dec),
    ):
        agent = OptionsAgent()

    # Rebind clients so live patches are still effective after __init__
    agent.trading_client = mock_tc
    agent.option_data_client = mock_odc

    return agent, mock_tc, mock_odc


# ── Test Cases ────────────────────────────────────────────────────────────────

class TestHappyPath(unittest.TestCase):
    """Full session: scan → analyze → buy_call → submit order."""

    def setUp(self):
        self.agent, self.tc, self.odc = build_agent(positions=[])

    def _patch_and_run(self):
        contract = make_contract()
        self.tc.get_option_contracts.return_value = make_contracts_resp([contract])
        self.odc.get_option_latest_quote.return_value = {
            contract.symbol: make_quote('3.50', '3.70')
        }
        with (
            patch('options_agent.yf.Ticker', return_value=MagicMock(
                history=MagicMock(return_value=make_price_df())
            )),
            patch('options_agent.get_trading_decision',
                  return_value='BUY_CALL. Confidence: 0.85.'),
            patch('options_agent.parse_decision',
                  return_value={'decision': 'buy', 'confidence': 0.85, 'reasoning': 'bullish'}),
            patch('builtins.open', unittest.mock.mock_open()),
        ):
            self.agent.run_options_session()

    def test_trade_executed(self):
        """At least one order should be submitted when there's capital and a buy_call signal."""
        self._patch_and_run()
        self.assertTrue(
            self.tc.submit_order.called,
            "Expected submit_order to be called but it was not"
        )

    def test_daily_trade_counter_increments(self):
        """daily_trades should be incremented for each successful execution."""
        self._patch_and_run()
        self.assertGreater(self.agent.daily_trades, 0)

    def test_trade_cooldown_set(self):
        """After a trade, the symbol should appear in trade_cooldown."""
        self._patch_and_run()
        self.assertTrue(
            len(self.agent.trade_cooldown) > 0,
            "Expected trade_cooldown to contain at least one symbol"
        )


class TestPositionManagement(unittest.TestCase):
    """manage_existing_positions() — take-profit and stop-loss exits."""

    def _run_manage(self, positions):
        agent, tc, _ = build_agent(positions=positions)
        with patch('builtins.open', unittest.mock.mock_open()), \
             patch.object(agent, '_fill_timeout_retry',
                          side_effect=lambda order, *a, **kw: order):
            agent.manage_existing_positions()
        return tc

    def test_take_profit_closes_position(self):
        """A position with +55% gain should trigger a sell order."""
        from alpaca.trading.requests import LimitOrderRequest
        pos = make_option_position('AAPL270319C00207000', qty=2, unrealized_plpc=0.55)
        tc = self._run_manage([pos])
        tc.submit_order.assert_called_once()
        order_arg = tc.submit_order.call_args[0][0]
        self.assertEqual(order_arg.side.value, 'sell')
        self.assertIsInstance(order_arg, LimitOrderRequest)

    def test_stop_loss_closes_position(self):
        """A position with -55% loss should trigger a sell order."""
        from alpaca.trading.requests import LimitOrderRequest
        pos = make_option_position('TSLA270319P00250000', qty=1, unrealized_plpc=-0.55)
        tc = self._run_manage([pos])
        tc.submit_order.assert_called_once()
        order_arg = tc.submit_order.call_args[0][0]
        self.assertEqual(order_arg.side.value, 'sell')
        self.assertIsInstance(order_arg, LimitOrderRequest)

    def test_normal_position_not_closed(self):
        """A position within normal range (-20%) should NOT trigger a sell."""
        pos = make_option_position('MSFT270319C00380000', qty=1, unrealized_plpc=-0.20)
        tc = self._run_manage([pos])
        tc.submit_order.assert_not_called()

    def test_stock_positions_ignored(self):
        """Regular stock positions (short symbol) must be ignored."""
        stock = make_stock_position('AAPL', qty=10)
        tc = self._run_manage([stock])
        tc.submit_order.assert_not_called()

    def test_both_exits_in_same_session(self):
        """Take-profit and stop-loss in same position list → two sell orders."""
        from alpaca.trading.requests import LimitOrderRequest
        pos_tp = make_option_position('AAPL270319C00207000', qty=2, unrealized_plpc=0.55)
        pos_sl = make_option_position('TSLA270319P00250000', qty=1, unrealized_plpc=-0.55)
        pos_ok = make_option_position('MSFT270319C00380000', qty=1, unrealized_plpc=0.10)
        tc = self._run_manage([pos_tp, pos_sl, pos_ok])
        self.assertEqual(tc.submit_order.call_count, 2)
        for call_args in tc.submit_order.call_args_list:
            self.assertIsInstance(call_args[0][0], LimitOrderRequest)


class TestLowConfidenceHold(unittest.TestCase):
    """When AI returns low confidence, no trade should be placed."""

    def test_hold_when_below_min_confidence(self):
        agent, tc, _ = build_agent(
            positions=[],
            lora_decision={'decision': 'buy', 'confidence': 0.60, 'reasoning': 'weak signal'},
            ollama_response='BUY_CALL. Confidence: 0.60.',
        )
        with (
            patch('options_agent.yf.Ticker', return_value=MagicMock(
                history=MagicMock(return_value=make_price_df())
            )),
            patch('options_agent.get_trading_decision',
                  return_value='BUY_CALL. Confidence: 0.60.'),
            patch('options_agent.parse_decision',
                  return_value={'decision': 'buy', 'confidence': 0.60, 'reasoning': 'weak'}),
        ):
            agent.run_options_session()

        tc.submit_order.assert_not_called()
        self.assertEqual(agent.daily_trades, 0)


class TestNoPriceSkipsTrade(unittest.TestCase):
    """If get_option_price returns None, the trade should be skipped."""

    def test_no_price_no_order(self):
        agent, tc, odc = build_agent(positions=[])

        # Return empty dict → contract symbol not found → price = None
        odc.get_option_latest_quote.return_value = {}

        with (
            patch('options_agent.yf.Ticker', return_value=MagicMock(
                history=MagicMock(return_value=make_price_df())
            )),
            patch('options_agent.get_trading_decision',
                  return_value='BUY_CALL. Confidence: 0.90.'),
            patch('options_agent.parse_decision',
                  return_value={'decision': 'buy', 'confidence': 0.90, 'reasoning': 'strong'}),
        ):
            agent.run_options_session()

        tc.submit_order.assert_not_called()


class TestInsufficientCapital(unittest.TestCase):
    """Session should skip new trades if available capital < $100."""

    def test_no_trade_when_options_fully_allocated(self):
        # Set equity to $1000. Options allocation target is 12.5% = $125.
        # If options_value already = $200 (> 15% of $1000 = $150), available = 0.
        agent, tc, _ = build_agent(
            equity='1000.00',
            positions=[
                make_option_position('AAPL270319C00207000', qty=1, unrealized_plpc=0.05),
            ],
        )
        # The option market_value helper sets market_value to qty * 360 = $360 > $150 limit
        with (
            patch('options_agent.yf.Ticker', return_value=MagicMock(
                history=MagicMock(return_value=make_price_df())
            )),
            patch('options_agent.get_trading_decision',
                  return_value='BUY_CALL. Confidence: 0.90.'),
            patch('options_agent.parse_decision',
                  return_value={'decision': 'buy', 'confidence': 0.90, 'reasoning': 'strong'}),
        ):
            agent.run_options_session()

        # submit_order may be called for exit management but NOT for new entries
        new_buy_calls = [
            c for c in tc.submit_order.call_args_list
            if hasattr(c[0][0], 'side') and c[0][0].side.value == 'buy'
        ]
        self.assertEqual(len(new_buy_calls), 0)


class TestDailyTradeLimit(unittest.TestCase):
    """Session should stop placing new trades once max_daily_trades is reached."""

    def test_respects_daily_limit(self):
        agent, tc, _ = build_agent(positions=[])
        # Pre-fill the daily trade counter to the limit
        agent.daily_trades = agent.params['max_daily_trades']

        with (
            patch('options_agent.yf.Ticker', return_value=MagicMock(
                history=MagicMock(return_value=make_price_df())
            )),
            patch('options_agent.get_trading_decision',
                  return_value='BUY_CALL. Confidence: 0.90.'),
            patch('options_agent.parse_decision',
                  return_value={'decision': 'buy', 'confidence': 0.90, 'reasoning': 'strong'}),
        ):
            agent.run_options_session()

        tc.submit_order.assert_not_called()


class TestGetOptionPrice(unittest.TestCase):
    """Unit tests for get_option_price() mid-price calculation."""

    def setUp(self):
        self.agent, _, self.odc = build_agent()

    def _set_quote(self, sym, bid, ask):
        q = make_quote(str(bid) if bid is not None else None,
                       str(ask) if ask is not None else None)
        if bid is None:
            q.bid_price = None
        if ask is None:
            q.ask_price = None
        self.odc.get_option_latest_quote.return_value = {sym: q}

    def test_midpoint_returned_when_both_bid_ask(self):
        self._set_quote('X', 3.50, 3.70)
        price = self.agent.get_option_price('X')
        self.assertAlmostEqual(price, 3.60)

    def test_ask_only_when_no_bid(self):
        self._set_quote('X', None, 3.70)
        price = self.agent.get_option_price('X')
        self.assertAlmostEqual(price, 3.70)

    def test_bid_only_when_no_ask(self):
        self._set_quote('X', 3.50, None)
        price = self.agent.get_option_price('X')
        self.assertAlmostEqual(price, 3.50)

    def test_none_when_both_zero(self):
        self._set_quote('X', 0.0, 0.0)
        price = self.agent.get_option_price('X')
        self.assertIsNone(price)

    def test_none_when_symbol_not_in_response(self):
        self.odc.get_option_latest_quote.return_value = {}
        price = self.agent.get_option_price('MISSING')
        self.assertIsNone(price)

    def test_none_on_api_exception(self):
        self.odc.get_option_latest_quote.side_effect = Exception("API error")
        price = self.agent.get_option_price('X')
        self.assertIsNone(price)


class TestFindOptimalOption(unittest.TestCase):
    """find_optimal_option() — ensure it uses resp.option_contracts, not the response itself."""

    def setUp(self):
        self.agent, self.tc, _ = build_agent()

    def test_returns_best_contract(self):
        """Should return the contract whose strike is closest to target and DTE ~30."""
        from datetime import date, timedelta
        # Use dates relative to today so the test never goes stale
        today = date.today()
        d_near  = (today + timedelta(days=10)).strftime('%Y-%m-%d')
        d_mid   = (today + timedelta(days=32)).strftime('%Y-%m-%d')   # ~30 DTE, closer to target
        d_far   = (today + timedelta(days=75)).strftime('%Y-%m-%d')
        contracts = [
            make_contract('AAPL_NEAR_C00200000', '200.00', d_near),
            make_contract('AAPL_MID_C00207000',  '207.00', d_mid),   # closest to 30 DTE + target
            make_contract('AAPL_FAR_C00215000',  '215.00', d_far),
        ]
        self.tc.get_option_contracts.return_value = make_contracts_resp(contracts)
        from alpaca.trading.enums import ContractType
        result = self.agent.find_optimal_option('AAPL', ContractType.CALL, 207.0)
        self.assertIsNotNone(result)
        self.assertEqual(result.strike_price, '207.00')

    def test_returns_none_when_no_contracts(self):
        """Empty option_contracts list → return None."""
        self.tc.get_option_contracts.return_value = make_contracts_resp([])
        from alpaca.trading.enums import ContractType
        result = self.agent.find_optimal_option('AAPL', ContractType.CALL, 200.0)
        self.assertIsNone(result)

    def test_returns_none_on_api_error(self):
        self.tc.get_option_contracts.side_effect = Exception("API down")
        from alpaca.trading.enums import ContractType
        result = self.agent.find_optimal_option('AAPL', ContractType.CALL, 200.0)
        self.assertIsNone(result)


class TestOptionDecisionLogic(unittest.TestCase):
    """get_ai_options_decision() — verify put/call/hold mapping."""

    def setUp(self):
        self.agent, _, _ = build_agent()

    def _decide(self, response_text, parse_result, momentum=0.05):
        analysis = {
            'symbol': 'AAPL',
            'current_price': 180.0,
            'rsi': 55.0,
            'volatility': 0.25,
            'momentum': momentum,
            'volume': 50_000_000,
        }
        with (
            patch('options_agent.get_trading_decision', return_value=response_text),
            patch('options_agent.parse_decision', return_value=parse_result),
        ):
            decision, _, _ = self.agent.get_ai_options_decision(analysis)
            return decision

    def test_call_keyword_maps_to_buy_call(self):
        result = self._decide(
            'BUY_CALL. Strong uptrend.',
            {'decision': 'buy', 'confidence': 0.85, 'reasoning': 'uptrend'},
        )
        self.assertEqual(result['decision'], 'buy_call')

    def test_put_keyword_maps_to_buy_put(self):
        result = self._decide(
            'BUY_PUT. Bearish reversal.',
            {'decision': 'sell', 'confidence': 0.80, 'reasoning': 'bearish'},
            momentum=-0.05,
        )
        self.assertEqual(result['decision'], 'buy_put')

    def test_hold_when_no_directional_signal(self):
        result = self._decide(
            'HOLD. Market uncertain.',
            {'decision': 'hold', 'confidence': 0.60, 'reasoning': 'uncertain'},
            momentum=0.0,
        )
        self.assertEqual(result['decision'], 'hold')

    def test_positive_momentum_with_buy_maps_to_call(self):
        """buy decision + positive momentum → buy_call even without 'call' keyword."""
        result = self._decide(
            'BUY. Positive momentum.',
            {'decision': 'buy', 'confidence': 0.80, 'reasoning': 'momentum'},
            momentum=0.03,
        )
        self.assertEqual(result['decision'], 'buy_call')

    def test_negative_momentum_with_sell_maps_to_put(self):
        """sell decision + negative momentum → buy_put even without 'put' keyword."""
        result = self._decide(
            'SELL. Downtrend forming.',
            {'decision': 'sell', 'confidence': 0.80, 'reasoning': 'downtrend'},
            momentum=-0.03,
        )
        self.assertEqual(result['decision'], 'buy_put')


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestGetOptionPrice,
        TestFindOptimalOption,
        TestOptionDecisionLogic,
        TestPositionManagement,
        TestLowConfidenceHold,
        TestNoPriceSkipsTrade,
        TestInsufficientCapital,
        TestDailyTradeLimit,
        TestHappyPath,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
