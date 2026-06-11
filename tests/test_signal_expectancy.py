"""
Unit tests for signal_expectancy.compute_signal_ev and load_signal_ev (PR-8 E4).

All tests are fully offline — no DB, no network, no GPU.
"""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'analysis'))


class TestComputeSignalEV:
    """compute_signal_ev — core EV arithmetic via mocked DB layer."""

    def _make_decision(self, order_id: str, signals: list[str]) -> dict:
        return {
            'order_id': order_id,
            'indicators': json.dumps({'signals': signals}),
        }

    def _make_outcome(self, buy_order_id: str, pnl_pct: float) -> dict:
        return {'buy_order_id': buy_order_id, 'pnl_pct': pnl_pct}

    def _run(self, decisions, outcomes, min_samples=1):
        import signal_expectancy as se
        with patch.object(se._db, 'get_executed_decisions', return_value=decisions), \
             patch.object(se._db, 'get_all_outcomes', return_value=outcomes):
            return se.compute_signal_ev(min_samples=min_samples)

    def test_single_winning_signal(self):
        """One signal, all wins → positive EV equal to avg win."""
        decisions = [self._make_decision('o1', ['breakout'])]
        outcomes  = [self._make_outcome('o1', 0.05)]
        result = self._run(decisions, outcomes)
        assert 'breakout' in result
        assert result['breakout'] > 0

    def test_single_losing_signal(self):
        """One signal, all losses → negative EV."""
        decisions = [self._make_decision('o1', ['gap_fill'])]
        outcomes  = [self._make_outcome('o1', -0.04)]
        result = self._run(decisions, outcomes)
        assert 'gap_fill' in result
        assert result['gap_fill'] < 0

    def test_ev_formula_exact(self):
        """EV = win_rate * avg_win - (1-win_rate) * abs(avg_loss)."""
        # 2 wins at +10%, 1 loss at -6% → wr=2/3, avg_win=0.10, avg_loss=0.06
        # EV = (2/3)*0.10 - (1/3)*0.06 = 0.0667 - 0.0200 = 0.0467
        decisions = [
            self._make_decision('o1', ['momentum']),
            self._make_decision('o2', ['momentum']),
            self._make_decision('o3', ['momentum']),
        ]
        outcomes = [
            self._make_outcome('o1', 0.10),
            self._make_outcome('o2', 0.10),
            self._make_outcome('o3', -0.06),
        ]
        result = self._run(decisions, outcomes, min_samples=3)
        assert 'momentum' in result
        assert abs(result['momentum'] - (2/3 * 0.10 - 1/3 * 0.06)) < 1e-4

    def test_min_samples_excludes_low_count_signals(self):
        """Signals with fewer samples than min_samples are excluded."""
        decisions = [self._make_decision('o1', ['rare_signal'])]
        outcomes  = [self._make_outcome('o1', 0.05)]
        result = self._run(decisions, outcomes, min_samples=2)
        assert 'rare_signal' not in result

    def test_min_samples_includes_exact_threshold(self):
        """A signal with exactly min_samples should be included."""
        decisions = [
            self._make_decision('o1', ['sig']),
            self._make_decision('o2', ['sig']),
        ]
        outcomes = [
            self._make_outcome('o1', 0.03),
            self._make_outcome('o2', -0.01),
        ]
        result = self._run(decisions, outcomes, min_samples=2)
        assert 'sig' in result

    def test_multiple_signals_per_decision(self):
        """A decision tagged with multiple signals contributes to each signal's pool."""
        decisions = [self._make_decision('o1', ['breakout', 'momentum'])]
        outcomes  = [self._make_outcome('o1', 0.06)]
        result = self._run(decisions, outcomes, min_samples=1)
        assert 'breakout' in result
        assert 'momentum' in result

    def test_db_error_returns_empty_dict(self):
        """Any DB exception → {} (graceful fallback)."""
        import signal_expectancy as se
        with patch.object(se._db, 'get_executed_decisions', side_effect=RuntimeError('db down')):
            result = se.compute_signal_ev()
        assert result == {}

    def test_decision_without_outcome_skipped(self):
        """Decisions that have no matching outcome are silently skipped."""
        decisions = [self._make_decision('o1', ['oversold'])]
        outcomes  = []  # no matching outcome
        result = self._run(decisions, outcomes, min_samples=1)
        assert 'oversold' not in result

    def test_comma_separated_signals_string(self):
        """signals field can be a comma-separated string instead of a list."""
        dec = {
            'order_id': 'o1',
            'indicators': json.dumps({'signals': 'breakout, momentum'}),
        }
        outcomes = [self._make_outcome('o1', 0.05)]
        import signal_expectancy as se
        with patch.object(se._db, 'get_executed_decisions', return_value=[dec]), \
             patch.object(se._db, 'get_all_outcomes', return_value=outcomes):
            result = se.compute_signal_ev(min_samples=1)
        assert 'breakout' in result
        assert 'momentum' in result


class TestEVRanking:
    """rank_opportunities uses high-EV signals to place stocks above low-EV ones."""

    def _setup_discovery(self, ev_map: dict):
        """Return a StockDiscovery whose _load_signal_ev returns ev_map."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'data'))
        with patch('stock_discovery._load_delisted_cache', return_value=set()):
            import stock_discovery
            d = stock_discovery.StockDiscovery()
        return d, stock_discovery

    def test_high_ev_signal_ranks_above_low_ev_signal(self):
        """Symbol with positive-EV signals outranks symbol with negative-EV signals."""
        with patch('stock_discovery._load_delisted_cache', return_value=set()):
            import stock_discovery
            d = stock_discovery.StockDiscovery()

        d.opportunities['AAPL'] = ['breakout']  # high EV
        d.opportunities['GME']  = ['gap_fill']  # low EV

        ev_map = {'breakout': 0.08, 'gap_fill': -0.03}
        with patch('stock_discovery._load_signal_ev', return_value=ev_map):
            ranked = d.rank_opportunities(['AAPL', 'GME'])

        assert ranked.index('AAPL') < ranked.index('GME')

    def test_fallback_to_count_when_ev_map_empty(self):
        """When ev_map is {}, ranking falls back to signal count."""
        with patch('stock_discovery._load_delisted_cache', return_value=set()):
            import stock_discovery
            d = stock_discovery.StockDiscovery()

        d.opportunities['A'] = ['s1', 's2', 's3']  # 3 signals
        d.opportunities['B'] = ['s1']               # 1 signal

        with patch('stock_discovery._load_signal_ev', return_value={}):
            ranked = d.rank_opportunities(['A', 'B'])

        assert ranked.index('A') < ranked.index('B')

    def test_unknown_signal_gets_count_bonus_not_error(self):
        """A signal absent from ev_map gets len(signals)*0.001 — not a KeyError."""
        with patch('stock_discovery._load_delisted_cache', return_value=set()):
            import stock_discovery
            d = stock_discovery.StockDiscovery()

        d.opportunities['X'] = ['unknown_signal']
        ev_map = {'breakout': 0.05}  # 'unknown_signal' not present

        with patch('stock_discovery._load_signal_ev', return_value=ev_map):
            ranked = d.rank_opportunities(['X'])

        assert ranked == ['X']  # no exception, just returns the symbol


class TestLoadSignalEV:
    """load_signal_ev reads from disk and falls back gracefully."""

    def test_returns_ev_map_from_valid_file(self):
        import signal_expectancy as se
        data = {'signal_ev': {'breakout': 0.05, 'gap_fill': -0.02}, 'computed_at': '2026-01-01'}
        with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as f:
            json.dump(data, f)
            p = Path(f.name)
        try:
            result = se.load_signal_ev(p)
            assert result == {'breakout': 0.05, 'gap_fill': -0.02}
        finally:
            p.unlink(missing_ok=True)

    def test_returns_empty_dict_when_file_missing(self):
        import signal_expectancy as se
        result = se.load_signal_ev(Path('/nonexistent/path/signal_expectancy.json'))
        assert result == {}

    def test_returns_empty_dict_on_malformed_json(self):
        import signal_expectancy as se
        with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as f:
            f.write('not valid json{{')
            p = Path(f.name)
        try:
            result = se.load_signal_ev(p)
            assert result == {}
        finally:
            p.unlink(missing_ok=True)

    def test_returns_empty_dict_when_key_missing(self):
        import signal_expectancy as se
        data = {'computed_at': '2026-01-01'}  # no 'signal_ev' key
        with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as f:
            json.dump(data, f)
            p = Path(f.name)
        try:
            result = se.load_signal_ev(p)
            assert result == {}
        finally:
            p.unlink(missing_ok=True)
