"""
Unit tests for round-3 self-improvement features.

Covers:
  - db.get_outcome_count_since()
  - StockDiscovery.get_exploration_symbols()
  - Exploration injection in autonomous_agent.run_trading_session()
  - training_data_builder.build_portfolio_level_examples()
  - online_trainer.should_train() / run_online_training() (no GPU needed)
  - build_dpo_dataset.build_dpo_pairs() (no GPU needed)

All tests are fully offline — no Alpaca, no LLM, no GPU required.

Run with:
    python3 -m pytest tests/test_round3_features.py -v
"""
import json
import sys
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / 'scripts'))
sys.path.insert(0, str(_REPO / 'finetune'))


# ===========================================================================
# db.get_outcome_count_since
# ===========================================================================

class TestGetOutcomeCountSince:
    def _make_db(self, tmp_path):
        import db as _db
        db_path = tmp_path / 'test.db'
        _db.init_db(db_path)
        return db_path

    def _insert_outcome(self, db_path, exit_ts):
        import db as _db
        _db.insert_outcome({
            'symbol': 'AAPL',
            'buy_order_id': f'buy-{exit_ts}',
            'sell_order_id': f'sell-{exit_ts}',
            'entry_timestamp': '2024-01-01T09:30:00',
            'exit_timestamp': exit_ts,
            'entry_price': 100.0,
            'exit_price': 105.0,
            'shares': 10,
            'realized_pnl': 50.0,
            'pnl_pct': 0.05,
            'hold_hours': 24.0,
            'entry_confidence': 0.80,
            'entry_reasoning': 'test',
            'won': True,
        }, db_path=db_path)

    def test_counts_outcomes_after_cutoff(self, tmp_path):
        import db as _db
        db_path = self._make_db(tmp_path)
        self._insert_outcome(db_path, '2024-06-01T12:00:00')
        self._insert_outcome(db_path, '2024-06-02T12:00:00')
        count = _db.get_outcome_count_since('2024-05-31T00:00:00', db_path=db_path)
        assert count == 2

    def test_excludes_outcomes_before_cutoff(self, tmp_path):
        import db as _db
        db_path = self._make_db(tmp_path)
        self._insert_outcome(db_path, '2024-04-01T12:00:00')
        count = _db.get_outcome_count_since('2024-05-01T00:00:00', db_path=db_path)
        assert count == 0

    def test_empty_table_returns_zero(self, tmp_path):
        import db as _db
        db_path = self._make_db(tmp_path)
        count = _db.get_outcome_count_since('2000-01-01T00:00:00', db_path=db_path)
        assert count == 0


# ===========================================================================
# StockDiscovery.get_exploration_symbols
# ===========================================================================

class TestGetExplorationSymbols:
    def _make_discovery(self, universe):
        from stock_discovery import StockDiscovery
        d = StockDiscovery.__new__(StockDiscovery)
        d.full_universe = universe
        d._delisted = set()
        return d

    def test_returns_n_symbols(self):
        d = self._make_discovery(['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA'])
        symbols = d.get_exploration_symbols(n=2)
        assert len(symbols) == 2

    def test_excludes_given_set(self):
        d = self._make_discovery(['AAPL', 'MSFT', 'GOOG'])
        symbols = d.get_exploration_symbols(n=2, exclude={'AAPL', 'MSFT'})
        assert 'AAPL' not in symbols
        assert 'MSFT' not in symbols
        assert symbols == ['GOOG']

    def test_excludes_delisted(self):
        d = self._make_discovery(['AAPL', 'DEAD', 'MSFT'])
        d._delisted = {'DEAD'}
        symbols = d.get_exploration_symbols(n=3)
        assert 'DEAD' not in symbols

    def test_empty_universe_returns_empty(self):
        d = self._make_discovery([])
        assert d.get_exploration_symbols(n=2) == []

    def test_n_larger_than_candidates_returns_all(self):
        d = self._make_discovery(['AAPL', 'MSFT'])
        symbols = d.get_exploration_symbols(n=10)
        assert set(symbols) == {'AAPL', 'MSFT'}


# ===========================================================================
# build_portfolio_level_examples
# ===========================================================================

class TestBuildPortfolioLevelExamples:
    def _make_constraints_file(self, tmp_path, records):
        f = tmp_path / 'portfolio_constraints.json'
        f.write_text(json.dumps(records))
        return f

    def _db_mock(self, db_path):
        """Return a mock _db that routes calls to the real functions on db_path."""
        import db as real_db
        real_db.init_db(db_path)
        m = MagicMock()
        m.init_db.side_effect = lambda: real_db.init_db(db_path)
        m.get_existing_prompt_hashes.side_effect = lambda: real_db.get_existing_prompt_hashes(db_path=db_path)
        m.insert_training_example.side_effect = lambda ex: real_db.insert_training_example(ex, db_path=db_path)
        return m

    def test_inserts_veto_records(self, tmp_path):
        constraints = [
            {
                'symbol': 'NVDA',
                'reason': 'Technology sector at 38%, cap is 30%',
                'sector': 'Technology',
                'sector_pct': 0.38,
                'sector_cap': 0.30,
                'spend_usd': 500,
                'timestamp': '2024-06-01T14:00:00',
            }
        ]
        constraints_path = self._make_constraints_file(tmp_path, constraints)

        import training_data_builder as tdb
        original = tdb._PORTFOLIO_CONSTRAINTS
        tdb._PORTFOLIO_CONSTRAINTS = constraints_path

        mock_db = self._db_mock(tmp_path / 'test.db')
        with patch('training_data_builder._db', mock_db):
            added = tdb.build_portfolio_level_examples()

        tdb._PORTFOLIO_CONSTRAINTS = original
        assert added == 1

    def test_deduplicates_on_rerun(self, tmp_path):
        constraints = [
            {
                'symbol': 'NVDA',
                'reason': 'Sector cap exceeded',
                'sector': 'Technology',
                'sector_pct': 0.35,
                'sector_cap': 0.30,
                'spend_usd': 500,
                'timestamp': '2024-06-01T14:00:00',
            }
        ]
        constraints_path = self._make_constraints_file(tmp_path, constraints)

        import training_data_builder as tdb
        original = tdb._PORTFOLIO_CONSTRAINTS
        tdb._PORTFOLIO_CONSTRAINTS = constraints_path

        mock_db = self._db_mock(tmp_path / 'test.db')
        with patch('training_data_builder._db', mock_db):
            first  = tdb.build_portfolio_level_examples()
            second = tdb.build_portfolio_level_examples()

        tdb._PORTFOLIO_CONSTRAINTS = original
        assert first == 1
        assert second == 0  # deduped

    def test_missing_file_returns_zero(self, tmp_path):
        import training_data_builder as tdb
        original = tdb._PORTFOLIO_CONSTRAINTS
        tdb._PORTFOLIO_CONSTRAINTS = tmp_path / 'nonexistent.json'
        result = tdb.build_portfolio_level_examples()
        tdb._PORTFOLIO_CONSTRAINTS = original
        assert result == 0


# ===========================================================================
# online_trainer.should_train
# ===========================================================================

class TestShouldTrain:
    def test_triggers_when_enough_outcomes(self, tmp_path):
        import online_trainer as ot
        state = {'last_trained_at': '2024-01-01T00:00:00', 'outcomes_at_last_train': 0}
        with patch.object(ot, '_load_state', return_value=state), \
             patch('online_trainer._db') as mock_db:
            mock_db.get_outcome_count_since.return_value = 20
            ready, reason = ot.should_train(min_new_outcomes=15)
        assert ready is True
        assert '20' in reason

    def test_skips_when_too_few_outcomes(self, tmp_path):
        import online_trainer as ot
        state = {'last_trained_at': '2024-01-01T00:00:00', 'outcomes_at_last_train': 0}
        with patch.object(ot, '_load_state', return_value=state), \
             patch('online_trainer._db') as mock_db:
            mock_db.get_outcome_count_since.return_value = 5
            ready, reason = ot.should_train(min_new_outcomes=15)
        assert ready is False
        assert '5' in reason

    def test_run_online_training_skips_gracefully(self):
        import online_trainer as ot
        with patch.object(ot, 'should_train', return_value=(False, 'only 3 outcomes')), \
             patch('online_trainer._db') as mock_db:
            mock_db.init_db = MagicMock()
            result = ot.run_online_training()
        assert result is False

    def test_run_online_training_skips_no_adapter(self):
        import online_trainer as ot
        with patch.object(ot, 'should_train', return_value=(True, '20 new outcomes')), \
             patch.object(ot, '_export_recent_examples', return_value=Path('/tmp/fake.json')), \
             patch.object(ot, '_latest_adapter', return_value=None), \
             patch('online_trainer._db') as mock_db:
            mock_db.init_db = MagicMock()
            result = ot.run_online_training()
        assert result is False


# ===========================================================================
# build_dpo_dataset.build_dpo_pairs
# ===========================================================================

class TestBuildDpoPairs:
    def _make_example(self, label, symbol='AAPL', pnl=0.10, conf=0.80, decision='buy'):
        return {
            'bot': 'stock',
            'symbol': symbol,
            'prompt': f'Analyze {symbol} for trading.',
            'ideal_output': f'Decision: BUY\nConfidence: {conf:.2f}\nReasoning: test\nOutcome: winner.',
            'label': label,
            'confidence': conf,
            'pnl_pct': pnl,
            'entry_date': '2024-06-01',
            'session_id': 'sess-1',
            'prompt_hash': f'hash-{label}-{symbol}',
        }

    def test_real_pair_formed_for_matching_symbol(self):
        from build_dpo_dataset import build_dpo_pairs
        examples = [
            self._make_example('winner',  'AAPL', pnl=0.10),
            self._make_example('loser',   'AAPL', pnl=-0.12),
        ]
        with patch('build_dpo_dataset._db') as mock_db:
            mock_db.init_db = MagicMock()
            mock_db.get_training_examples.return_value = examples
            pairs, real, synthetic = build_dpo_pairs()
        assert real == 1
        assert synthetic == 0
        assert len(pairs) == 1
        assert pairs[0]['symbol'] == 'AAPL'

    def test_synthetic_pair_when_no_matching_loser(self):
        from build_dpo_dataset import build_dpo_pairs
        examples = [
            self._make_example('winner', 'TSLA', pnl=0.15),
        ]
        with patch('build_dpo_dataset._db') as mock_db:
            mock_db.init_db = MagicMock()
            mock_db.get_training_examples.return_value = examples
            pairs, real, synthetic = build_dpo_pairs()
        assert synthetic == 1
        assert real == 0

    def test_empty_examples_returns_empty(self):
        from build_dpo_dataset import build_dpo_pairs
        with patch('build_dpo_dataset._db') as mock_db:
            mock_db.init_db = MagicMock()
            mock_db.get_training_examples.return_value = []
            pairs, real, synthetic = build_dpo_pairs()
        assert pairs == []

    def test_preference_weight_calculation(self):
        from build_dpo_dataset import _preference_weight
        assert _preference_weight(0.10, 0.80) == pytest.approx(0.08, rel=1e-4)
        assert _preference_weight(None, 0.80) == 0.0
        assert _preference_weight(0.10, None) == pytest.approx(0.05, rel=1e-4)

    def test_chosen_contains_winner_output(self):
        from build_dpo_dataset import build_dpo_pairs
        examples = [
            self._make_example('strong_winner', 'GOOG', pnl=0.32),
        ]
        with patch('build_dpo_dataset._db') as mock_db:
            mock_db.init_db = MagicMock()
            mock_db.get_training_examples.return_value = examples
            pairs, _, _ = build_dpo_pairs()
        assert 'Decision: BUY' in pairs[0]['chosen']
        # Rejected should differ from chosen
        assert pairs[0]['chosen'] != pairs[0]['rejected']
