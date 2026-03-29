"""
Unit tests for the four new components added in round-2:
  - portfolio_overseer.PortfolioOverseer
  - allocation_controller.AllocationController
  - preflight_check.run_preflight
  - inference_client (generate / warmup / stop_for_finetuning / backend_info)

All tests are fully offline — no Alpaca API, no LLM, no GPU required.

Run with:
  python3 -m pytest tests/test_new_components.py -v
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))


# ===========================================================================
# PortfolioOverseer
# ===========================================================================

class TestPortfolioOverseer:
    def _make_overseer(self, positions=None):
        from portfolio_overseer import PortfolioOverseer
        client = MagicMock()
        client.get_all_positions.return_value = positions or []
        return PortfolioOverseer(client)

    def _make_position(self, symbol, market_value):
        p = MagicMock()
        p.symbol = symbol
        p.market_value = str(market_value)
        return p

    def test_buy_allowed_empty_portfolio(self):
        """No positions → any buy is allowed."""
        overseer = self._make_overseer(positions=[])
        allowed, reason = overseer.is_buy_allowed('AAPL', spend_usd=500, equity=10_000)
        assert allowed is True
        assert reason == ''

    def test_diversified_etf_never_vetoed(self):
        """SPY is always allowed regardless of other holdings."""
        # Fill tech sector to 30% already
        positions = [self._make_position('AAPL', 3_000)]
        overseer = self._make_overseer(positions=positions)
        allowed, reason = overseer.is_buy_allowed('SPY', spend_usd=1_000, equity=10_000)
        assert allowed is True

    def test_sector_cap_blocks_over_30pct(self):
        """Buying into tech after already holding 30% should be blocked."""
        from portfolio_overseer import MAX_SECTOR_PCT
        # tech already at 30%
        positions = [self._make_position('NVDA', 3_000)]
        overseer = self._make_overseer(positions=positions)
        # Add another $1 to tech → 3001/10000 = 30.01% > 30%
        allowed, reason = overseer.is_buy_allowed('AAPL', spend_usd=1, equity=10_000)
        assert allowed is False
        assert 'sector cap' in reason.lower()

    def test_sector_cap_allows_under_30pct(self):
        """Buying when projected sector % stays at or below cap is allowed."""
        positions = [self._make_position('NVDA', 2_000)]  # 20% tech
        overseer = self._make_overseer(positions=positions)
        # Adding $500 → 25% < 30%
        allowed, reason = overseer.is_buy_allowed('AAPL', spend_usd=500, equity=10_000)
        assert allowed is True

    def test_correlation_cap_blocks_5th_tech_position(self):
        """Max 4 open positions in same sector — 5th should be blocked."""
        from portfolio_overseer import MAX_CORR_POSITIONS
        assert MAX_CORR_POSITIONS == 4

        # 4 distinct tech tickers held (combined value well under 30%)
        tech = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']
        positions = [self._make_position(s, 100) for s in tech]  # tiny positions
        overseer = self._make_overseer(positions=positions)
        # 5th tech buy
        allowed, reason = overseer.is_buy_allowed('AMD', spend_usd=10, equity=100_000)
        assert allowed is False
        assert 'correlation cap' in reason.lower()

    def test_adding_to_existing_position_not_counted_twice(self):
        """Adding to AAPL when AAPL already held doesn't increment position count."""
        positions = [
            self._make_position('AAPL',  100),
            self._make_position('MSFT',  100),
            self._make_position('GOOGL', 100),
            self._make_position('NVDA',  100),
        ]
        overseer = self._make_overseer(positions=positions)
        # Buying more AAPL — should NOT count as a 5th distinct position
        allowed, _ = overseer.is_buy_allowed('AAPL', spend_usd=10, equity=100_000)
        assert allowed is True

    def test_zero_equity_always_allowed(self):
        """If equity is 0 (account not loaded) we skip the guard."""
        overseer = self._make_overseer(positions=[])
        allowed, _ = overseer.is_buy_allowed('AAPL', spend_usd=500, equity=0)
        assert allowed is True

    def test_sector_headroom_returns_dict(self):
        positions = [self._make_position('XOM', 1_000)]  # energy at 10%
        overseer = self._make_overseer(positions=positions)
        headroom = overseer.sector_headroom(equity=10_000)
        # 30% cap = $3 000; used = $1 000 → headroom = $2 000
        assert 'energy' in headroom
        assert abs(headroom['energy'] - 2_000.0) < 1.0

    def test_cache_is_used_within_ttl(self):
        """get_all_positions should only be called once within the cache TTL."""
        overseer = self._make_overseer(positions=[])
        overseer.is_buy_allowed('AAPL', 100, 10_000)
        overseer.is_buy_allowed('MSFT', 100, 10_000)
        assert overseer._client.get_all_positions.call_count == 1

    def test_cache_invalidated_after_invalidate_call(self):
        """invalidate_cache() forces a fresh fetch."""
        overseer = self._make_overseer(positions=[])
        overseer.is_buy_allowed('AAPL', 100, 10_000)
        overseer.invalidate_cache()
        overseer.is_buy_allowed('AAPL', 100, 10_000)
        assert overseer._client.get_all_positions.call_count == 2

    def test_get_sector_unknown(self):
        """Unknown tickers fall back to 'unknown' and are never blocked by sector cap."""
        from portfolio_overseer import get_sector
        assert get_sector('XYZQQ') == 'unknown'

    def test_api_error_falls_back_to_cached(self):
        """If Alpaca API call fails, the overseer uses the previous cache."""
        from portfolio_overseer import PortfolioOverseer
        client = MagicMock()
        client.get_all_positions.side_effect = Exception("network error")
        overseer = PortfolioOverseer(client)
        # Should not raise — falls back to empty list
        allowed, _ = overseer.is_buy_allowed('AAPL', 100, 10_000)
        assert allowed is True  # empty fallback → no concentration


# ===========================================================================
# AllocationController
# ===========================================================================

class TestAllocationController:
    def _make_controller(self, trades=None):
        from allocation_controller import AllocationController
        db = MagicMock()
        db.get_recent_trades.return_value = trades or []
        return AllocationController(db)

    def _make_trade(self, pnl_pct):
        return {'pnl_pct': pnl_pct}

    def test_tier1_on_empty_history(self):
        """No trades → Tier 1 (cold-start, 1% position)."""
        ctrl = self._make_controller(trades=[])
        assert ctrl.current_tier(force_refresh=True) == 1

    def test_tier1_position_size_is_1pct(self):
        from allocation_controller import TIER1_PCT
        ctrl = self._make_controller(trades=[])
        assert ctrl.get_position_size_pct(force_refresh=True) == TIER1_PCT

    def test_tier2_reached_with_enough_good_trades(self):
        """20 winning trades with Sharpe > 1.0 and max-DD < 8% → Tier 2."""
        from allocation_controller import MIN_TRADES_TIER2, TIER2_PCT
        # 20 trades each +3% → Sharpe >> 1, max-DD = 0
        trades = [self._make_trade(0.03)] * MIN_TRADES_TIER2
        ctrl = self._make_controller(trades=trades)
        assert ctrl.current_tier(force_refresh=True) == 2
        assert ctrl.get_position_size_pct() == TIER2_PCT

    def test_tier3_reached_with_excellent_trades(self):
        """50 winning trades → Tier 3."""
        from allocation_controller import MIN_TRADES_TIER3, TIER3_PCT
        trades = [self._make_trade(0.05)] * MIN_TRADES_TIER3
        ctrl = self._make_controller(trades=trades)
        assert ctrl.current_tier(force_refresh=True) == 3
        assert ctrl.get_position_size_pct() == TIER3_PCT

    def test_tier2_not_reached_below_min_trades(self):
        """19 good trades (< 20 minimum) → stays at Tier 1."""
        from allocation_controller import MIN_TRADES_TIER2
        trades = [self._make_trade(0.03)] * (MIN_TRADES_TIER2 - 1)
        ctrl = self._make_controller(trades=trades)
        assert ctrl.current_tier(force_refresh=True) == 1

    def test_high_drawdown_prevents_tier3(self):
        """High draw-down prevents promotion even with enough trades."""
        from allocation_controller import MIN_TRADES_TIER3
        # Mix of big wins and big losses → high max-DD
        trades = [self._make_trade(0.20)] * (MIN_TRADES_TIER3 // 2)
        trades += [self._make_trade(-0.15)] * (MIN_TRADES_TIER3 // 2)
        ctrl = self._make_controller(trades=trades)
        tier = ctrl.current_tier(force_refresh=True)
        assert tier < 3  # not tier 3

    def test_metrics_keys_present(self):
        """_compute_metrics returns expected keys."""
        from allocation_controller import AllocationController
        db = MagicMock()
        db.get_recent_trades.return_value = []
        ctrl = AllocationController(db)
        ctrl.refresh()
        m = ctrl.metrics()
        assert m is not None
        for key in ('total_trades', 'win_rate', 'sharpe', 'max_dd', 'annualised_return'):
            assert key in m

    def test_db_error_falls_back_to_tier1(self):
        """DB failure during refresh → safe Tier 1 fallback."""
        from allocation_controller import AllocationController
        db = MagicMock()
        db.get_recent_trades.side_effect = Exception("db error")
        ctrl = AllocationController(db)
        assert ctrl.current_tier(force_refresh=True) == 1


# ===========================================================================
# Preflight Check
# ===========================================================================

class TestPreflightCheck:
    def _make_account(self, **kwargs):
        defaults = {
            'trading_blocked':        False,
            'account_blocked':        False,
            'transfers_blocked':      False,
            'pattern_day_trader':     False,
            'options_approved_level': 2,
            'buying_power':           '10000',
            'equity':                 '50000',
            'cash':                   '10000',
            'status':                 'ACTIVE',
        }
        defaults.update(kwargs)
        a = MagicMock()
        for k, v in defaults.items():
            setattr(a, k, v)
        return a

    def test_ok_account_passes(self):
        from preflight_check import run_preflight
        client = MagicMock()
        client.get_account.return_value = self._make_account()
        result = run_preflight(client, require_options=True)
        assert result.ok is True
        assert result.issues == []

    def test_account_blocked_fails(self):
        from preflight_check import run_preflight
        client = MagicMock()
        client.get_account.return_value = self._make_account(account_blocked=True)
        result = run_preflight(client, require_options=False)
        assert result.ok is False
        assert any('account_blocked' in i for i in result.issues)

    def test_trading_blocked_fails(self):
        from preflight_check import run_preflight
        client = MagicMock()
        client.get_account.return_value = self._make_account(trading_blocked=True)
        result = run_preflight(client, require_options=False)
        assert result.ok is False
        assert any('trading_blocked' in i for i in result.issues)

    def test_insufficient_buying_power_fails(self):
        from preflight_check import run_preflight
        client = MagicMock()
        client.get_account.return_value = self._make_account(buying_power='100')
        result = run_preflight(client, require_options=False)
        assert result.ok is False
        assert any('buying_power' in i for i in result.issues)

    def test_options_level_below_required_fails(self):
        from preflight_check import run_preflight
        client = MagicMock()
        client.get_account.return_value = self._make_account(options_approved_level=1)
        result = run_preflight(client, require_options=True)
        assert result.ok is False
        assert any('options_approved_level' in i for i in result.issues)

    def test_options_check_skipped_when_not_required(self):
        """options_approved_level=0 should NOT fail when require_options=False."""
        from preflight_check import run_preflight
        client = MagicMock()
        client.get_account.return_value = self._make_account(options_approved_level=0)
        result = run_preflight(client, require_options=False)
        assert result.ok is True

    def test_api_error_returns_not_ok(self):
        from preflight_check import run_preflight
        client = MagicMock()
        client.get_account.side_effect = Exception("connection refused")
        result = run_preflight(client)
        assert result.ok is False
        assert any('Could not fetch' in i for i in result.issues)

    def test_inactive_account_status_fails(self):
        from preflight_check import run_preflight
        client = MagicMock()
        client.get_account.return_value = self._make_account(status='ACCOUNT_CLOSED')
        result = run_preflight(client)
        assert result.ok is False
        assert any('status' in i for i in result.issues)

    def test_info_dict_populated(self):
        from preflight_check import run_preflight
        client = MagicMock()
        client.get_account.return_value = self._make_account()
        result = run_preflight(client)
        assert 'equity' in result.info
        assert 'buying_power' in result.info


# ===========================================================================
# inference_client
# ===========================================================================

class TestInferenceClient:
    def test_generate_ollama_returns_string(self):
        import inference_client
        with patch('inference_client.INFERENCE_BACKEND', 'ollama'), \
             patch('inference_client._generate_ollama', return_value='buy') as mock_gen:
            result = inference_client.generate('test prompt', max_tokens=10)
        mock_gen.assert_called_once_with('test prompt', 10, 0.7)
        assert result == 'buy'

    def test_generate_vllm_returns_string(self):
        import inference_client
        with patch('inference_client.INFERENCE_BACKEND', 'vllm'), \
             patch('inference_client._generate_vllm', return_value='sell') as mock_gen:
            result = inference_client.generate('test prompt', max_tokens=5)
        mock_gen.assert_called_once_with('test prompt', 5, 0.7)
        assert result == 'sell'

    def test_warmup_calls_ollama_backend(self):
        import inference_client
        with patch('inference_client.INFERENCE_BACKEND', 'ollama'), \
             patch('inference_client._warmup_ollama') as mock_wu:
            inference_client.warmup()
        mock_wu.assert_called_once()

    def test_warmup_calls_vllm_backend(self):
        import inference_client
        with patch('inference_client.INFERENCE_BACKEND', 'vllm'), \
             patch('inference_client._warmup_vllm') as mock_wu:
            inference_client.warmup()
        mock_wu.assert_called_once()

    def test_stop_for_finetuning_only_stops_ollama(self):
        """stop_for_finetuning is a no-op for vLLM backend."""
        import inference_client
        with patch('inference_client.INFERENCE_BACKEND', 'vllm'), \
             patch('inference_client._stop_ollama') as mock_stop:
            inference_client.stop_for_finetuning()
        mock_stop.assert_not_called()

    def test_stop_for_finetuning_stops_ollama_backend(self):
        import inference_client
        with patch('inference_client.INFERENCE_BACKEND', 'ollama'), \
             patch('inference_client._stop_ollama') as mock_stop:
            inference_client.stop_for_finetuning()
        mock_stop.assert_called_once()

    def test_backend_info_ollama(self):
        import inference_client
        with patch('inference_client.INFERENCE_BACKEND', 'ollama'), \
             patch('inference_client.OLLAMA_MODEL', 'qwen3:8b'):
            info = inference_client.backend_info()
        assert info['backend'] == 'ollama'
        assert info['model'] == 'qwen3:8b'

    def test_backend_info_vllm(self):
        import inference_client
        with patch('inference_client.INFERENCE_BACKEND', 'vllm'), \
             patch('inference_client.VLLM_MODEL', 'Qwen/test'), \
             patch('inference_client.VLLM_BASE_URL', 'http://localhost:8000/v1'):
            info = inference_client.backend_info()
        assert info['backend'] == 'vllm'
        assert info['model'] == 'Qwen/test'
        assert 'url' in info

    def test_vllm_falls_back_to_ollama_on_error(self):
        """When vLLM request fails, _generate_vllm falls back to Ollama."""
        import inference_client
        with patch('inference_client._generate_ollama', return_value='hold') as mock_ollama, \
             patch('inference_client.INFERENCE_BACKEND', 'vllm'), \
             patch('inference_client.VLLM_BASE_URL', 'http://localhost:8000/v1'):
            import requests
            with patch('requests.post', side_effect=Exception("connection refused")):
                result = inference_client._generate_vllm('prompt', 10, 0.7)
        mock_ollama.assert_called_once()
        assert result == 'hold'
