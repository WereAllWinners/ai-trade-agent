"""
tests/test_alpha_measurement.py — PR-6: Alpha measurement (D3, D4)

Covers:
  - _bootstrap_expectancy: identical returns, mixed returns, p5 < mean on mixed
  - _bootstrap_sharpe: returns median > p5 on volatile sample
  - AllocationController._evaluate_tier with bootstrap gate
  - WeeklyReporter.calculate_regime_breakdown: grouping and per-regime stats
  - WeeklyReporter.spy_benchmark_return: cache key, graceful failure
"""
import sys
import math
from pathlib import Path
from unittest.mock import MagicMock, patch
from collections import defaultdict

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / 'scripts'
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from allocation_controller import (  # noqa: E402
    _bootstrap_expectancy,
    _bootstrap_sharpe,
    AllocationController,
    MIN_TRADES_TIER2,
    MIN_TRADES_TIER3,
    MAX_DD_THRESHOLD_TIER2,
    MAX_DD_THRESHOLD_TIER3,
    SHARPE_THRESHOLD_TIER3,
)


# ---------------------------------------------------------------------------
# _bootstrap_expectancy
# ---------------------------------------------------------------------------

class TestBootstrapExpectancy:
    def test_identical_positive_returns_high_p5(self):
        """50 identical +3% returns → expectancy_p5 should be close to 0.03."""
        pnl = [0.03] * 50
        mean, p5 = _bootstrap_expectancy(pnl, n_boot=2000, seed=42)
        assert mean == pytest.approx(0.03, abs=0.001)
        assert p5   == pytest.approx(0.03, abs=0.002)

    def test_mixed_returns_p5_below_mean(self):
        """Mix of +5% and -3% → bootstrapped p5 must be < mean (uncertainty)."""
        pnl = [0.05] * 25 + [-0.03] * 25    # mean = +1%, some downside risk
        mean, p5 = _bootstrap_expectancy(pnl, n_boot=5000, seed=42)
        assert p5 < mean

    def test_all_negative_p5_negative(self):
        """All losing trades → both mean and p5 should be negative."""
        pnl = [-0.02] * 30
        mean, p5 = _bootstrap_expectancy(pnl, n_boot=2000, seed=42)
        assert mean < 0
        assert p5   < 0

    def test_small_sample_returns_sample_mean(self):
        """< 10 trades → bootstrap not reliable; both values equal the sample mean."""
        pnl = [0.05, -0.03, 0.01]
        mean, p5 = _bootstrap_expectancy(pnl)
        expected = sum(pnl) / len(pnl)
        assert mean == pytest.approx(expected, abs=1e-9)
        assert p5   == pytest.approx(expected, abs=1e-9)

    def test_empty_returns_zero(self):
        """Empty list → (0.0, 0.0)."""
        mean, p5 = _bootstrap_expectancy([])
        assert mean == 0.0
        assert p5   == 0.0


# ---------------------------------------------------------------------------
# _bootstrap_sharpe
# ---------------------------------------------------------------------------

class TestBootstrapSharpe:
    def test_positive_consistent_returns_sharpe_positive(self):
        """Consistent positive returns → Sharpe median > 0."""
        pnl = [0.02] * 50
        median, p5 = _bootstrap_sharpe(pnl, trades_per_year=50, rf_per_trade=0.0)
        assert median > 0
        assert p5     >= 0

    def test_volatile_sample_p5_below_median(self):
        """High variance → bootstrapped p5 < median (wider CI)."""
        import random as _r
        _r.seed(0)
        pnl = [_r.gauss(0.01, 0.08) for _ in range(50)]
        median, p5 = _bootstrap_sharpe(pnl, trades_per_year=50, rf_per_trade=0.0, n_boot=2000)
        assert p5 <= median

    def test_small_sample_returns_zeros(self):
        """< 10 trades → (0.0, 0.0)."""
        pnl = [0.03, 0.02]
        median, p5 = _bootstrap_sharpe(pnl, trades_per_year=50, rf_per_trade=0.0)
        assert median == 0.0
        assert p5     == 0.0


# ---------------------------------------------------------------------------
# AllocationController._evaluate_tier with bootstrap gate
# ---------------------------------------------------------------------------

class TestAllocationControllerBootstrap:
    def _make_controller(self, trades: list[dict]) -> AllocationController:
        db_mock = MagicMock()
        db_mock.get_recent_trades.return_value = trades
        ctrl = AllocationController(db_mock, lookback_trades=100)
        ctrl.refresh()
        return ctrl

    def _trade(self, pnl_pct: float, hold_hours: float = 6.5) -> dict:
        return {'pnl_pct': pnl_pct, 'hold_hours': hold_hours}

    def test_consistent_winners_reach_tier2(self):
        """≥ MIN_TRADES_TIER2 consistent winners with low DD → Tier 2."""
        trades = [self._trade(0.03)] * MIN_TRADES_TIER2
        ctrl   = self._make_controller(trades)
        assert ctrl.current_tier() == 2

    def test_consistent_losers_stay_tier1(self):
        """All losing trades → expectancy_p5 < 0 → stays at Tier 1."""
        trades = [self._trade(-0.05)] * MIN_TRADES_TIER2
        ctrl   = self._make_controller(trades)
        assert ctrl.current_tier() == 1

    def test_too_few_trades_stay_tier1(self):
        """Below MIN_TRADES_TIER2 threshold → Tier 1 regardless of returns."""
        trades = [self._trade(0.05)] * (MIN_TRADES_TIER2 - 1)
        ctrl   = self._make_controller(trades)
        assert ctrl.current_tier() == 1

    def test_tier3_requires_min_trades(self):
        """Fewer than MIN_TRADES_TIER3 trades → cannot be Tier 3."""
        trades = [self._trade(0.10)] * (MIN_TRADES_TIER3 - 1)
        ctrl   = self._make_controller(trades)
        assert ctrl.current_tier() < 3

    def test_metrics_include_bootstrap_fields(self):
        """_compute_metrics output must include expectancy_p5 and sharpe_median."""
        trades = [self._trade(0.02)] * 30
        ctrl   = self._make_controller(trades)
        m = ctrl.metrics()
        assert 'expectancy_p5'  in m
        assert 'expectancy_mean' in m
        assert 'sharpe_median'  in m
        assert 'sharpe_p5'      in m


# ---------------------------------------------------------------------------
# WeeklyReporter.calculate_regime_breakdown
# ---------------------------------------------------------------------------

class TestRegimeBreakdown:
    def _reporter(self):
        """Minimal WeeklyReporter stub — no Alpaca connection needed."""
        sys.path.insert(0, str(_SCRIPTS_DIR / 'analysis'))
        with patch('weekly_report.TradingClient'), \
             patch('weekly_report.load_dotenv'):
            from weekly_report import WeeklyReporter
            r = object.__new__(WeeklyReporter)
            return r

    def _outcome(self, won: bool, regime: str | None = None,
                 excess: float | None = None) -> dict:
        return {
            'won': won,
            'pnl_pct': 0.05 if won else -0.03,
            'realized_pnl': 100.0 if won else -60.0,
            'regime': regime,
            'excess_return_pct': excess,
            'symbol': 'TEST',
            'hold_hours': 6.5,
            'exit_timestamp': '2026-01-01T12:00:00',
            'entry_confidence': 0.75,
        }

    def test_groups_by_regime(self):
        reporter = self._reporter()
        outcomes = [
            self._outcome(True,  'bull_lowvol'),
            self._outcome(False, 'bull_lowvol'),
            self._outcome(True,  'bear_highvol'),
        ]
        result = reporter.calculate_regime_breakdown(outcomes)
        assert 'bull_lowvol'  in result
        assert 'bear_highvol' in result
        assert result['bull_lowvol']['trade_count']  == 2
        assert result['bear_highvol']['trade_count'] == 1

    def test_none_regime_bins_as_unknown(self):
        reporter = self._reporter()
        outcomes = [self._outcome(True, regime=None)]
        result   = reporter.calculate_regime_breakdown(outcomes)
        assert 'unknown' in result

    def test_win_rate_computed_per_regime(self):
        reporter = self._reporter()
        outcomes = [
            self._outcome(True,  'bull'),
            self._outcome(True,  'bull'),
            self._outcome(False, 'bull'),
        ]
        result   = reporter.calculate_regime_breakdown(outcomes)
        assert result['bull']['win_rate'] == pytest.approx(2/3, abs=0.001)

    def test_avg_excess_return_pct(self):
        reporter = self._reporter()
        outcomes = [
            self._outcome(True, 'bull', excess=0.04),
            self._outcome(True, 'bull', excess=0.02),
        ]
        result = reporter.calculate_regime_breakdown(outcomes)
        assert result['bull']['avg_excess_return_pct'] == pytest.approx(0.03, abs=0.001)

    def test_excess_none_when_no_data(self):
        """avg_excess_return_pct is None when outcomes have no excess_return_pct."""
        reporter = self._reporter()
        outcomes = [self._outcome(True, 'bull', excess=None)]
        result   = reporter.calculate_regime_breakdown(outcomes)
        assert result['bull']['avg_excess_return_pct'] is None

    def test_empty_outcomes_returns_empty_dict(self):
        reporter = self._reporter()
        assert reporter.calculate_regime_breakdown([]) == {}


# ---------------------------------------------------------------------------
# WeeklyReporter.spy_benchmark_return (SPY fetch cache)
# ---------------------------------------------------------------------------

class TestSpyBenchmarkReturn:
    def _reporter(self):
        with patch('weekly_report.TradingClient'), \
             patch('weekly_report.load_dotenv'):
            from weekly_report import WeeklyReporter
            r = object.__new__(WeeklyReporter)
            return r

    def test_returns_none_on_yfinance_error(self):
        import weekly_report
        reporter = self._reporter()
        with patch('weekly_report._SPY_CACHE', {}), \
             patch('yfinance.Ticker', side_effect=Exception("no network")):
            from datetime import datetime
            result = reporter.spy_benchmark_return(datetime(2025, 1, 1), datetime(2025, 1, 31))
        assert result is None

    def test_result_cached_on_second_call(self):
        """Second call with same dates must not invoke yfinance again."""
        import weekly_report
        from datetime import datetime
        reporter = self._reporter()

        mock_ticker = MagicMock()
        mock_hist   = MagicMock()
        mock_hist.empty = False
        mock_hist.__len__ = lambda self: 5
        mock_hist.columns = ['Close', 'Open']

        call_count = {'n': 0}

        def fake_ticker(sym):
            call_count['n'] += 1
            t = MagicMock()
            import pandas as pd
            hist_df = pd.DataFrame({'close': [100.0, 105.0]})
            t.history.return_value = hist_df
            return t

        with patch.dict('weekly_report._SPY_CACHE', {}):
            with patch('yfinance.Ticker', side_effect=fake_ticker):
                start, end = datetime(2025, 3, 1), datetime(2025, 3, 31)
                r1 = reporter.spy_benchmark_return(start, end)
                r2 = reporter.spy_benchmark_return(start, end)

        assert call_count['n'] == 1   # yfinance called only once
        assert r1 == r2
