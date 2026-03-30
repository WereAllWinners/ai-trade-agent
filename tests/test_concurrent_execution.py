"""
Tests for concurrent stock + options execution safety.

Covers:
  1. Cash reservation prevents cross-bot race conditions.
  2. DRY_RUN mode logs orders without submitting them.
  3. Near-zero option exit is skipped when total value < _MIN_EXIT_VALUE_USD.
  4. Proactive PDT counter blocks buys after 3 day-trades.
  5. Discovery cache returns fresh results without a full scan.

All tests are fully offline — no Alpaca API calls, no Ollama, no network.

Run with:
    python3 -m pytest tests/test_concurrent_execution.py -v
"""
import sys
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Make scripts importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))

# Stub out heavy optional deps that may not be installed in the test environment.
# Must be done before any script module is imported so module-level imports succeed.
for _stub in ('ollama', 'yfinance', 'unsloth', 'bitsandbytes'):
    if _stub not in sys.modules:
        sys.modules[_stub] = MagicMock()


# ===========================================================================
# Helpers
# ===========================================================================

def _make_options_agent(tmp_db=None):
    """Build a minimal OptionsAgent with mocked clients."""
    with patch('options_agent.TradingClient'), \
         patch('options_agent.OptionHistoricalDataClient'), \
         patch('options_agent.load_dotenv'), \
         patch('options_agent.load_model_once'), \
         patch('options_agent.PortfolioOverseer'), \
         patch('options_agent.AllocationController'), \
         patch('options_agent._db') as mock_db:
        mock_db.init_db = MagicMock()
        mock_db.cleanup_stale_reservations = MagicMock()
        mock_db.get_total_reserved = MagicMock(return_value=0.0)
        mock_db.reserve_cash = MagicMock(return_value=1)
        mock_db.release_cash = MagicMock()
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
    agent._decision_log      = Path('/tmp/test_options_concurrent.jsonl')
    agent._params_file       = Path('/tmp/non_existent_options.json')
    # Mock new round-2 components
    alloc_ctrl = MagicMock()
    alloc_ctrl.get_position_size_pct.return_value = 0.03
    agent.alloc_controller = alloc_ctrl
    overseer = MagicMock()
    overseer.is_buy_allowed.return_value = (True, '')
    agent.overseer = overseer
    return agent


def _make_stock_agent():
    """Build a minimal AutonomousAgent with mocked clients."""
    with patch('autonomous_agent.TradingClient'), \
         patch('autonomous_agent.StockHistoricalDataClient'), \
         patch('autonomous_agent.StockDiscovery'), \
         patch('autonomous_agent.load_dotenv'), \
         patch('autonomous_agent.load_model_once'), \
         patch('autonomous_agent.PortfolioOverseer'), \
         patch('autonomous_agent.AllocationController'), \
         patch('autonomous_agent._db') as mock_db:
        mock_db.init_db = MagicMock()
        mock_db.cleanup_stale_reservations = MagicMock()
        mock_db.get_total_reserved = MagicMock(return_value=0.0)
        mock_db.reserve_cash = MagicMock(return_value=1)
        mock_db.release_cash = MagicMock()
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
    agent.fee_simulator      = MagicMock()
    # Mock new round-2 components
    alloc_ctrl = MagicMock()
    alloc_ctrl.get_position_size_pct.return_value = 0.05
    agent.alloc_controller = alloc_ctrl
    overseer = MagicMock()
    overseer.is_buy_allowed.return_value = (True, '')
    agent.overseer = overseer
    return agent


# ===========================================================================
# 1. Cash reservation — cross-bot race condition guard
# ===========================================================================

class TestCashReservation:
    """Verify that the cross-bot cash reservation functions work atomically."""

    def test_reserve_and_release_roundtrip(self, tmp_path):
        """reserve_cash creates a row; release_cash marks it released; get_total_reserved updates."""
        import db as _db
        db_file = tmp_path / 'test.db'
        _db.init_db(db_file)

        rid = _db.reserve_cash('stock', 500.0, db_file)
        assert rid > 0
        assert _db.get_total_reserved(db_file) == pytest.approx(500.0)

        _db.release_cash(rid, db_file)
        assert _db.get_total_reserved(db_file) == pytest.approx(0.0)

    def test_two_bots_reservations_sum_correctly(self, tmp_path):
        """Two concurrent reservations from different bots both count toward total."""
        import db as _db
        db_file = tmp_path / 'test.db'
        _db.init_db(db_file)

        rid1 = _db.reserve_cash('stock', 300.0, db_file)
        rid2 = _db.reserve_cash('options', 200.0, db_file)
        assert _db.get_total_reserved(db_file) == pytest.approx(500.0)

        _db.release_cash(rid1, db_file)
        assert _db.get_total_reserved(db_file) == pytest.approx(200.0)

        _db.release_cash(rid2, db_file)
        assert _db.get_total_reserved(db_file) == pytest.approx(0.0)

    def test_stale_reservation_cleanup(self, tmp_path):
        """cleanup_stale_reservations releases old rows that were never explicitly released."""
        import db as _db
        db_file = tmp_path / 'test.db'
        _db.init_db(db_file)

        # Insert a reservation with a timestamp well in the past
        stale_time = (datetime.now() - timedelta(minutes=5)).isoformat()
        with _db.get_conn(db_file) as conn:
            conn.execute(
                "INSERT INTO cash_reservations (bot, amount, reserved_at) VALUES (?, ?, ?)",
                ['stock', 999.0, stale_time]
            )

        assert _db.get_total_reserved(db_file) == pytest.approx(999.0)

        cleaned = _db.cleanup_stale_reservations(max_age_seconds=60, db_path=db_file)
        assert cleaned == 1
        assert _db.get_total_reserved(db_file) == pytest.approx(0.0)

    def test_stock_bot_blocks_buy_when_options_has_reservation(self):
        """
        Stock bot's execute_trade must block a BUY when the cross-bot reservation
        makes effective_cash fall below _MIN_CASH_RESERVE.

        Scenario:
          live_cash = $2000, options has reserved $1400 → effective = $600
          reserve = $500 → spendable = $100 < share price $200 → BLOCKED
        """
        agent = _make_stock_agent()

        mock_account = MagicMock()
        mock_account.cash = '2000'
        agent.trading_client.get_account.return_value = mock_account

        decision = {'decision': 'buy', 'confidence': 0.80,
                    'reasoning': 'test', 'current_price': 200.0}

        import autonomous_agent as aa
        with patch.object(aa, '_db') as mock_db, \
             patch.object(aa, '_DRY_RUN', False):
            mock_db.cleanup_stale_reservations = MagicMock()
            mock_db.get_total_reserved = MagicMock(return_value=1400.0)  # options reserved $1400
            mock_db.reserve_cash = MagicMock(return_value=1)
            mock_db.release_cash = MagicMock()

            result = agent.execute_trade('AAPL', decision, 100_000, 10_000)

        assert result is False
        agent.trading_client.submit_order.assert_not_called()

    def test_stock_bot_releases_reservation_on_failure(self):
        """
        Cash reservation must be released even when submit_order raises an exception.
        """
        agent = _make_stock_agent()

        mock_account = MagicMock()
        mock_account.cash = '5000'
        agent.trading_client.get_account.return_value = mock_account
        agent.trading_client.submit_order.side_effect = Exception("API error")

        decision = {'decision': 'buy', 'confidence': 0.80,
                    'reasoning': 'test', 'current_price': 50.0}

        import autonomous_agent as aa
        with patch.object(aa, '_db') as mock_db, \
             patch.object(aa, '_DRY_RUN', False), \
             patch('autonomous_agent.alert_trade_failed'):
            mock_db.cleanup_stale_reservations = MagicMock()
            mock_db.get_total_reserved = MagicMock(return_value=0.0)
            mock_db.reserve_cash = MagicMock(return_value=42)
            mock_db.release_cash = MagicMock()

            result = agent.execute_trade('AAPL', decision, 100_000, 4500)

        assert result is False
        mock_db.release_cash.assert_called_once_with(42)


# ===========================================================================
# 2. DRY_RUN mode
# ===========================================================================

class TestDryRunMode:

    def _run_dry_buy(self, agent, live_cash=5000, price=50.0):
        """Helper: run execute_trade in dry-run mode and return (result, submit_call_count)."""
        mock_account = MagicMock()
        mock_account.cash = str(live_cash)
        agent.trading_client.get_account.return_value = mock_account
        mock_order = MagicMock(); mock_order.id = 'real-order'
        agent.trading_client.submit_order.return_value = mock_order

        decision = {'decision': 'buy', 'confidence': 0.80,
                    'reasoning': 'test', 'current_price': price}

        import autonomous_agent as aa
        with patch.object(aa, '_db') as mock_db, \
             patch.object(aa, '_DRY_RUN', True), \
             patch('builtins.open', MagicMock()), \
             patch('autonomous_agent.alert_trade_executed'):
            mock_db.cleanup_stale_reservations = MagicMock()
            mock_db.get_total_reserved = MagicMock(return_value=0.0)
            mock_db.reserve_cash = MagicMock(return_value=1)
            mock_db.release_cash = MagicMock()
            mock_db.insert_trade = MagicMock()

            result = agent.execute_trade('AAPL', decision, 100_000, live_cash)
        return result, agent.trading_client.submit_order.call_count

    def test_dry_run_returns_true_without_submitting(self):
        """DRY_RUN mode must return True (order logged) but submit_order must not be called."""
        agent = _make_stock_agent()
        result, submit_count = self._run_dry_buy(agent)
        assert result is True
        assert submit_count == 0

    def test_options_dry_run_does_not_submit(self):
        """Options DRY_RUN mode must skip submit_order for entry orders."""
        agent = _make_options_agent()

        mock_contract = MagicMock()
        mock_contract.symbol = 'SPY250328C00560000'
        mock_contract.strike_price = '560'
        mock_contract.expiration_date = '2025-03-28'

        mock_account = MagicMock()
        mock_account.cash = '5000'
        agent.trading_client.get_account.return_value = mock_account
        mock_order = MagicMock(); mock_order.id = 'real-order'
        agent.trading_client.submit_order.return_value = mock_order

        decision = {'decision': 'buy_call', 'confidence': 0.80,
                    'reasoning': 'test', 'current_price': 560.0}
        analysis = {'current_price': 560.0}

        import options_agent as oa
        with patch.object(agent, 'find_optimal_option', return_value=mock_contract), \
             patch.object(agent, 'get_option_price', return_value=1.00), \
             patch.object(agent, 'calculate_position_size', return_value=2), \
             patch.object(oa, '_db') as mock_db, \
             patch.object(oa, '_DRY_RUN', True), \
             patch('builtins.open', MagicMock()), \
             patch('options_agent.alert_trade_executed'):
            mock_db.cleanup_stale_reservations = MagicMock()
            mock_db.get_total_reserved = MagicMock(return_value=0.0)
            mock_db.reserve_cash = MagicMock(return_value=1)
            mock_db.release_cash = MagicMock()
            mock_db.insert_trade = MagicMock()

            result = agent.execute_options_trade('SPY', decision, analysis, 5000)

        assert result is True
        agent.trading_client.submit_order.assert_not_called()


# ===========================================================================
# 3. Near-zero option exit skip
# ===========================================================================

class TestNearZeroOptionExit:

    def _make_position(self, symbol, qty, unrealized_plpc):
        pos = MagicMock()
        pos.symbol = symbol
        pos.qty = str(qty)
        pos.unrealized_plpc = str(unrealized_plpc)
        return pos

    def test_skips_exit_when_total_value_below_min(self):
        """
        If contract price × 100 × qty < _MIN_EXIT_VALUE_USD, no exit order is submitted.
        With price=$0.02 and qty=2: total = $0.02 × 100 × 2 = $4.00 < $10 → skip.
        """
        agent = _make_options_agent()
        position = self._make_position('SPY250128P00560000', qty=2, unrealized_plpc=-0.60)
        agent.trading_client.get_all_positions.return_value = [position]

        import options_agent as oa
        with patch.object(agent, 'parse_dte_from_symbol', return_value=1), \
             patch.object(agent, '_has_open_exit_order', return_value=False), \
             patch.object(agent, 'get_option_price', return_value=0.02), \
             patch.object(oa, '_DRY_RUN', False), \
             patch.object(oa, '_MIN_EXIT_VALUE_USD', 10.0):
            agent.manage_existing_positions()

        agent.trading_client.submit_order.assert_not_called()

    def test_exits_when_total_value_above_min(self):
        """
        If contract price × 100 × qty >= _MIN_EXIT_VALUE_USD, the exit order is submitted.
        With price=$0.10 and qty=2: total = $0.10 × 100 × 2 = $20.00 >= $10 → exit.
        """
        agent = _make_options_agent()
        position = self._make_position('SPY250128P00560000', qty=2, unrealized_plpc=-0.60)
        agent.trading_client.get_all_positions.return_value = [position]

        mock_order = MagicMock(); mock_order.id = 'exit-order'
        agent.trading_client.submit_order.return_value = mock_order

        import options_agent as oa
        with patch.object(agent, 'parse_dte_from_symbol', return_value=1), \
             patch.object(agent, '_has_open_exit_order', return_value=False), \
             patch.object(agent, 'get_option_price', return_value=0.10), \
             patch.object(oa, '_DRY_RUN', False), \
             patch.object(oa, '_MIN_EXIT_VALUE_USD', 10.0), \
             patch('builtins.open', MagicMock()), \
             patch('options_agent.alert_trade_executed'):
            agent.manage_existing_positions()

        agent.trading_client.submit_order.assert_called_once()

    def test_take_profit_exit_also_respects_min_value(self):
        """
        Near-zero check applies to all exit types, including take-profit exits.
        price=$0.01, qty=1 → total=$1.00 < $10 → skip.
        """
        agent = _make_options_agent()
        position = self._make_position('SPY250128C00600000', qty=1, unrealized_plpc=0.60)
        agent.trading_client.get_all_positions.return_value = [position]

        import options_agent as oa
        with patch.object(agent, 'parse_dte_from_symbol', return_value=10), \
             patch.object(agent, '_has_open_exit_order', return_value=False), \
             patch.object(agent, 'get_option_price', return_value=0.01), \
             patch.object(oa, '_DRY_RUN', False), \
             patch.object(oa, '_MIN_EXIT_VALUE_USD', 10.0):
            agent.manage_existing_positions()

        agent.trading_client.submit_order.assert_not_called()


# ===========================================================================
# 4. Proactive PDT day-trade counter
# ===========================================================================

class TestPDTCounter:

    def test_buy_blocked_after_pdt_limit_reached(self):
        """
        After 3 same-day round trips in the DB, a new BUY must be blocked
        and pdt_blocked set to True — before the order even reaches Alpaca.
        """
        agent = _make_stock_agent()

        mock_account = MagicMock()
        mock_account.cash = '10000'
        agent.trading_client.get_account.return_value = mock_account

        decision = {'decision': 'buy', 'confidence': 0.80,
                    'reasoning': 'test', 'current_price': 100.0}

        import autonomous_agent as aa
        with patch.object(aa, '_db') as mock_db, \
             patch.object(aa, '_DRY_RUN', False), \
             patch.object(aa, '_PDT_DAY_TRADE_LIMIT', 3):
            mock_db.cleanup_stale_reservations = MagicMock()
            mock_db.get_total_reserved = MagicMock(return_value=0.0)
            mock_db.reserve_cash = MagicMock(return_value=1)
            mock_db.release_cash = MagicMock()
            # _count_todays_roundtrips uses get_conn; patch at agent level instead
            with patch.object(agent, '_count_todays_roundtrips', return_value=3):
                result = agent.execute_trade('AAPL', decision, 100_000, 9500)

        assert result is False
        assert agent.pdt_blocked is True
        agent.trading_client.submit_order.assert_not_called()

    def test_buy_allowed_when_under_pdt_limit(self):
        """
        With only 2 round trips today, a new BUY should proceed past the PDT check.
        """
        agent = _make_stock_agent()

        mock_account = MagicMock()
        mock_account.cash = '10000'
        agent.trading_client.get_account.return_value = mock_account
        mock_order = MagicMock(); mock_order.id = 'o1'
        agent.trading_client.submit_order.return_value = mock_order

        decision = {'decision': 'buy', 'confidence': 0.80,
                    'reasoning': 'test', 'current_price': 50.0}

        import autonomous_agent as aa
        with patch.object(aa, '_db') as mock_db, \
             patch.object(aa, '_DRY_RUN', False), \
             patch.object(aa, '_PDT_DAY_TRADE_LIMIT', 3), \
             patch('builtins.open', MagicMock()), \
             patch('autonomous_agent.alert_trade_executed'):
            mock_db.cleanup_stale_reservations = MagicMock()
            mock_db.get_total_reserved = MagicMock(return_value=0.0)
            mock_db.reserve_cash = MagicMock(return_value=1)
            mock_db.release_cash = MagicMock()
            mock_db.insert_trade = MagicMock()
            with patch.object(agent, '_count_todays_roundtrips', return_value=2):
                result = agent.execute_trade('AAPL', decision, 100_000, 9500)

        assert result is True
        assert agent.pdt_blocked is False


# ===========================================================================
# 5. Discovery cache
# ===========================================================================

class TestDiscoveryCache:

    def test_fresh_cache_returned_without_scan(self, tmp_path, monkeypatch):
        """
        When a discovery cache younger than the TTL exists, discover_opportunities
        must return the cached stocks without running any yfinance scans.
        """
        import stock_discovery as sd

        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'total_discovered': 3,
            'universe_scanned': 100,
            'stocks': ['AAPL', 'MSFT', 'NVDA'],
            'opportunities': {
                'AAPL': ['Unusual volume: 2.5x'],
                'MSFT': ['52W breakout: +0.5%'],
                'NVDA': ['20D momentum: +8.2%'],
            },
        }
        cache_file = tmp_path / 'discovery_cache.json'
        cache_file.write_text(json.dumps(cache_data))

        monkeypatch.setattr(sd, '_DISCOVERY_CACHE_PATH', cache_file)
        monkeypatch.setattr(sd, '_DISCOVERY_CACHE_TTL_HOURS', 4)

        discovery = sd.StockDiscovery()
        with patch.object(discovery, 'get_index_constituents', side_effect=AssertionError("should not scan")), \
             patch.object(discovery, 'get_actively_traded_universe', side_effect=AssertionError("should not scan")):
            result = discovery.discover_opportunities(max_stocks=10, use_cache=True)

        assert result == ['AAPL', 'MSFT', 'NVDA']
        assert 'Unusual volume: 2.5x' in discovery.opportunities['AAPL']

    def test_stale_cache_triggers_fresh_scan(self, tmp_path, monkeypatch):
        """
        When the cache is older than the TTL, a fresh scan must run.
        """
        import stock_discovery as sd

        old_time = (datetime.now() - timedelta(hours=5)).isoformat()
        cache_data = {
            'timestamp': old_time,
            'total_discovered': 1,
            'stocks': ['OUTDATED'],
            'opportunities': {},
        }
        cache_file = tmp_path / 'discovery_cache.json'
        cache_file.write_text(json.dumps(cache_data))

        monkeypatch.setattr(sd, '_DISCOVERY_CACHE_PATH', cache_file)
        monkeypatch.setattr(sd, '_DISCOVERY_CACHE_TTL_HOURS', 4)

        discovery = sd.StockDiscovery()

        fresh_stocks = ['AAPL', 'GOOGL']
        with patch.object(discovery, 'get_index_constituents', return_value=fresh_stocks), \
             patch.object(discovery, 'get_actively_traded_universe', return_value=[]), \
             patch.object(discovery, 'scan_unusual_volume', return_value=['AAPL']), \
             patch.object(discovery, 'scan_breakouts', return_value=[]), \
             patch.object(discovery, 'scan_oversold', return_value=[]), \
             patch.object(discovery, 'scan_momentum', return_value=[]), \
             patch.object(discovery, 'scan_gap_moves', return_value=[]), \
             patch.object(discovery, 'scan_mean_reversion', return_value=[]), \
             patch.object(discovery, 'get_premarket_movers', return_value=[]), \
             patch.object(discovery, 'filter_by_liquidity', return_value=['AAPL']), \
             patch.object(discovery, 'save_opportunities'):
            result = discovery.discover_opportunities(max_stocks=10, use_cache=True)

        assert 'OUTDATED' not in result

    def test_use_cache_false_forces_fresh_scan(self, tmp_path, monkeypatch):
        """
        use_cache=False must bypass a valid cache and run a fresh scan.
        """
        import stock_discovery as sd

        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'total_discovered': 1,
            'stocks': ['CACHED_ONLY'],
            'opportunities': {},
        }
        cache_file = tmp_path / 'discovery_cache.json'
        cache_file.write_text(json.dumps(cache_data))

        monkeypatch.setattr(sd, '_DISCOVERY_CACHE_PATH', cache_file)
        monkeypatch.setattr(sd, '_DISCOVERY_CACHE_TTL_HOURS', 4)

        discovery = sd.StockDiscovery()
        scan_called = []

        with patch.object(discovery, 'get_index_constituents', side_effect=lambda: scan_called.append(True) or []), \
             patch.object(discovery, 'get_actively_traded_universe', return_value=[]), \
             patch.object(discovery, 'scan_unusual_volume', return_value=[]), \
             patch.object(discovery, 'scan_breakouts', return_value=[]), \
             patch.object(discovery, 'scan_oversold', return_value=[]), \
             patch.object(discovery, 'scan_momentum', return_value=[]), \
             patch.object(discovery, 'scan_gap_moves', return_value=[]), \
             patch.object(discovery, 'scan_mean_reversion', return_value=[]), \
             patch.object(discovery, 'get_premarket_movers', return_value=[]), \
             patch.object(discovery, 'filter_by_liquidity', return_value=[]), \
             patch.object(discovery, 'save_opportunities'):
            discovery.discover_opportunities(max_stocks=10, use_cache=False)

        assert scan_called, "Fresh scan should have been triggered when use_cache=False"
