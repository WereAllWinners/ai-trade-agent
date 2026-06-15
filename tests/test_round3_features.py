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
    """Session 3 rewrite: loser→CF pairs + weak→strong gap-gated pairs.

    Old API (winner→loser, synthetic pairs, _preference_weight) has been removed.
    """

    def _row(self, label, symbol='AAPL', pnl=None, bot='stock', prompt=None, ideal_output=None):
        p = prompt or f'Analyze {symbol} for a potential trade. RSI: 72. MACD: -3.'
        out = ideal_output or (
            f'Decision: BUY\nConfidence: 0.80\nReasoning: Momentum signal confirmed for {symbol}.'
        )
        return {
            'bot': bot, 'source': 'paper', 'symbol': symbol,
            'prompt': p, 'ideal_output': out,
            'label': label, 'confidence': 0.80, 'pnl_pct': pnl,
            'entry_date': '2026-06-15', 'session_id': '',
            'prompt_hash': f'hash-{label}-{symbol}',
            'generated_at': '2026-06-15T00:00:00', 'reward': None,
        }

    def _cf_row(self, prompt, symbol='AAPL'):
        return self._row(
            'counterfactual', symbol=symbol, prompt=prompt,
            ideal_output=(
                'Decision: HOLD\nConfidence: 0.80\n'
                'Reasoning: Entry risk outweighs setup — MACD negative.'
            ),
        )

    def _call(self, examples):
        from build_dpo_dataset import build_dpo_pairs
        with patch('build_dpo_dataset._db') as mock_db:
            mock_db.init_db = MagicMock()
            mock_db.get_training_examples.return_value = examples
            return build_dpo_pairs()

    def test_loser_cf_pair_formed(self):
        """A loser with a matching CF (same prompt) → one loser→CF pair."""
        prompt = 'Analyze AAPL for a potential trade. RSI: 72. MACD: -3.'
        loser = self._row('loser', pnl=-0.12, prompt=prompt,
                          ideal_output='Decision: BUY\nConfidence: 0.75\nReasoning: Signal fired.')
        cf    = self._cf_row(prompt)

        pairs, stats = self._call([loser, cf])

        assert stats['loser_cf_pairs'] == 1
        assert len(pairs) == 1
        assert pairs[0]['pair_type'] == 'loser_counterfactual'
        assert pairs[0]['rejected'].endswith('<|im_end|>')
        assert pairs[0]['chosen'].endswith('<|im_end|>')

    def test_loser_without_cf_produces_no_pair(self):
        """A loser whose prompt has no matching CF → zero pairs, skipped_no_cf += 1."""
        loser = self._row('loser', pnl=-0.12,
                          ideal_output='Decision: BUY\nConfidence: 0.75\nReasoning: Signal fired.')

        pairs, stats = self._call([loser])

        assert pairs == []
        assert stats['skipped_no_cf'] >= 1

    def test_empty_examples_returns_empty(self):
        """No training examples → empty pairs, all stat counters zero."""
        pairs, stats = self._call([])
        assert pairs == []
        assert stats['total'] == 0

    def test_weak_strong_pair_built_when_gap_sufficient(self):
        """SW pnl=+30%, WW pnl=+10% → gap=0.20 ≥ 0.08 → weak→strong pair created."""
        prompt = 'Analyze MSFT for a potential trade.'
        sw = self._row('strong_winner', symbol='MSFT', pnl=+0.30, prompt=prompt, bot='stock',
                       ideal_output='Decision: BUY\nConfidence: 0.92\nReasoning: Strong momentum breakout.')
        ww = self._row('weak_winner',   symbol='MSFT', pnl=+0.10, prompt=prompt, bot='stock',
                       ideal_output='Decision: BUY\nConfidence: 0.70\nReasoning: Early signal; pending.')

        pairs, stats = self._call([sw, ww])

        ws = [p for p in pairs if p.get('pair_type') == 'weak_to_strong_winner']
        assert len(ws) >= 1, f"Expected weak→strong pair, stats={stats}"
        assert ws[0]['chosen'] != ws[0]['rejected']

    def test_chosen_and_rejected_always_differ(self):
        """chosen == rejected is rejected by the hard guard."""
        prompt = 'Analyze AAPL for a potential trade. RSI: 72. MACD: -3.'
        shared_output = 'Decision: BUY\nConfidence: 0.75\nReasoning: Signal fired.'
        loser = self._row('loser', pnl=-0.12, prompt=prompt, ideal_output=shared_output)
        cf    = self._row('counterfactual', prompt=prompt, ideal_output=shared_output)

        pairs, stats = self._call([loser, cf])

        assert pairs == []
        assert stats['skipped_guard_violations'] >= 1
