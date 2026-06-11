"""
Unit tests for PR-4: replay-based model evaluation and promotion gate.

All tests are offline — yfinance, ollama, and model inference are mocked.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'training'))


# ---------------------------------------------------------------------------
# replay_eval pure-function tests (no I/O)
# ---------------------------------------------------------------------------

class TestComputeExpectancy:
    def test_empty_list_returns_zero(self):
        from replay_eval import compute_expectancy
        assert compute_expectancy([]) == 0.0

    def test_all_positive_returns_positive(self):
        from replay_eval import compute_expectancy
        assert compute_expectancy([0.05, 0.10, 0.08]) == pytest.approx(0.0767, rel=0.01)

    def test_mixed_returns_correct_average(self):
        from replay_eval import compute_expectancy
        assert compute_expectancy([0.10, -0.07]) == pytest.approx(0.015)

    def test_all_negative_returns_negative(self):
        from replay_eval import compute_expectancy
        assert compute_expectancy([-0.05, -0.10]) == pytest.approx(-0.075)


class TestComputeMaxDD:
    def test_no_drawdown_returns_zero(self):
        from replay_eval import compute_max_dd
        assert compute_max_dd([1.0, 1.05, 1.10, 1.15]) == pytest.approx(0.0)

    def test_single_element_returns_zero(self):
        from replay_eval import compute_max_dd
        assert compute_max_dd([1.0]) == 0.0

    def test_drawdown_calculated_correctly(self):
        from replay_eval import compute_max_dd
        # Peak 1.20, trough 0.90 → dd = 0.30/1.20 = 0.25
        result = compute_max_dd([1.0, 1.20, 0.90])
        assert result == pytest.approx(0.25, rel=0.01)

    def test_max_dd_picks_worst_case(self):
        from replay_eval import compute_max_dd
        # Peak 1.25, trough 0.99 → dd = (1.25 - 0.99) / 1.25 = 0.208
        result = compute_max_dd([1.0, 1.25, 1.00, 1.10, 0.99])
        assert result == pytest.approx(0.208, abs=0.001)


class TestLoadHeldOutDecisions:
    def _write_log(self, path: Path, records: list[dict]) -> None:
        with open(path, 'w') as f:
            for r in records:
                f.write(json.dumps(r) + '\n')

    def _record(self, decision='buy', bot='stock', days_ago=5, symbol='AAPL',
                 prompt='Analyze AAPL today.') -> dict:
        from datetime import datetime, timedelta
        ts = (datetime.now() - timedelta(days=days_ago)).isoformat()
        return {
            'decision':   decision,
            'bot':        bot,
            'symbol':     symbol,
            'timestamp':  ts,
            'prompt':     prompt,
            'confidence': 0.75,
        }

    def test_filters_out_trained_hashes(self, tmp_path):
        from replay_eval import _load_held_out_decisions, _prompt_hash
        log = tmp_path / 'decision_log.jsonl'
        rec = self._record(prompt='Analyze AAPL today.')
        self._write_log(log, [rec])

        trained = {_prompt_hash('Analyze AAPL today.')}
        results = _load_held_out_decisions(log, lookback_days=30, trained_hashes=trained)
        assert len(results) == 0

    def test_includes_untrained_decisions(self, tmp_path):
        from replay_eval import _load_held_out_decisions
        log = tmp_path / 'decision_log.jsonl'
        rec = self._record(prompt='Analyze MSFT today.')
        self._write_log(log, [rec])

        results = _load_held_out_decisions(log, lookback_days=30, trained_hashes=set())
        assert len(results) == 1

    def test_filters_out_hold_decisions(self, tmp_path):
        from replay_eval import _load_held_out_decisions
        log = tmp_path / 'decision_log.jsonl'
        self._write_log(log, [self._record(decision='hold')])
        results = _load_held_out_decisions(log, lookback_days=30, trained_hashes=set())
        assert len(results) == 0

    def test_filters_out_old_decisions_beyond_lookback(self, tmp_path):
        from replay_eval import _load_held_out_decisions
        log = tmp_path / 'decision_log.jsonl'
        self._write_log(log, [self._record(days_ago=60)])  # 60 days ago, lookback=30
        results = _load_held_out_decisions(log, lookback_days=30, trained_hashes=set())
        assert len(results) == 0

    def test_includes_buy_call_decisions(self, tmp_path):
        from replay_eval import _load_held_out_decisions
        log = tmp_path / 'decision_log.jsonl'
        self._write_log(log, [self._record(decision='buy_call', bot='options')])
        results = _load_held_out_decisions(log, lookback_days=30, trained_hashes=set(),
                                           bot='options')
        assert len(results) == 1

    def test_skips_malformed_json_lines(self, tmp_path):
        from replay_eval import _load_held_out_decisions
        log = tmp_path / 'decision_log.jsonl'
        with open(log, 'w') as f:
            f.write('{"decision": "buy", "timestamp": "' +
                    str(__import__('datetime').datetime.now().isoformat()) +
                    '", "prompt": "Test", "bot": "stock"}\n')
            f.write('{bad json\n')
        results = _load_held_out_decisions(log, lookback_days=30, trained_hashes=set())
        assert len(results) == 1  # bad line skipped

    def test_returns_empty_when_file_missing(self, tmp_path):
        from replay_eval import _load_held_out_decisions
        missing = tmp_path / 'nonexistent.jsonl'
        results = _load_held_out_decisions(missing, lookback_days=30, trained_hashes=set())
        assert results == []


class TestPromptSetHash:
    def test_same_records_give_same_hash(self):
        from replay_eval import _prompt_set_hash, _prompt_hash
        records = [
            {'_prompt_hash': _prompt_hash('Analyze AAPL.')},
            {'_prompt_hash': _prompt_hash('Analyze MSFT.')},
        ]
        h1 = _prompt_set_hash(records)
        h2 = _prompt_set_hash(list(reversed(records)))  # order shouldn't matter
        assert h1 == h2

    def test_different_prompts_give_different_hash(self):
        from replay_eval import _prompt_set_hash, _prompt_hash
        recs_a = [{'_prompt_hash': _prompt_hash('Analyze AAPL.')}]
        recs_b = [{'_prompt_hash': _prompt_hash('Analyze TSLA.')}]
        assert _prompt_set_hash(recs_a) != _prompt_set_hash(recs_b)


# ---------------------------------------------------------------------------
# run_replay_eval integration (mocked I/O)
# ---------------------------------------------------------------------------

class TestRunReplayEval:
    def _write_log(self, path: Path, n: int = 10) -> None:
        from datetime import datetime, timedelta
        with open(path, 'w') as f:
            for i in range(n):
                ts = (datetime.now() - timedelta(days=i)).isoformat()
                f.write(json.dumps({
                    'decision':   'buy',
                    'bot':        'stock',
                    'symbol':     f'SYM{i}',
                    'timestamp':  ts,
                    'prompt':     f'Analyze SYM{i} at step {i}.',
                    'confidence': 0.80,
                }) + '\n')

    def test_candidate_better_expectancy_higher(self, tmp_path):
        """When model always buys and trades win, expectancy is positive."""
        from replay_eval import run_replay_eval
        log = tmp_path / 'decision_log.jsonl'
        self._write_log(log, n=10)

        def mock_rerun(prompt, adapter_path=None):
            return {'decision': 'buy', 'confidence': 0.80}

        def mock_simulate(symbol, ts):
            return {'pnl_pct': 0.05, 'exit_reason': 'take_profit'}

        with patch('replay_eval._db') as mock_db, \
             patch('replay_eval._rerun_decision', side_effect=mock_rerun), \
             patch('replay_eval._simulate_trade', side_effect=mock_simulate):
            mock_db.get_existing_prompt_hashes.return_value = set()
            result = run_replay_eval(log_path=log, bot='stock',
                                     lookback_days=30, seed=42)

        assert result['expectancy'] == pytest.approx(0.05)
        assert result['trade_count'] == 10
        assert result['max_dd'] == 0.0  # no drawdown on all-wins

    def test_model_always_holds_gives_zero_trades(self, tmp_path):
        """When model always HOLDs, trade_count should be 0."""
        from replay_eval import run_replay_eval
        log = tmp_path / 'decision_log.jsonl'
        self._write_log(log, n=5)

        with patch('replay_eval._db') as mock_db, \
             patch('replay_eval._rerun_decision', return_value={'decision': 'hold', 'confidence': 0.8}):
            mock_db.get_existing_prompt_hashes.return_value = set()
            result = run_replay_eval(log_path=log, bot='stock',
                                     lookback_days=30, seed=42)

        assert result['trade_count'] == 0
        assert result['expectancy'] == 0.0

    def test_fewer_than_floor_prompts_logs_warning(self, tmp_path):
        """With fewer than 30 held-out prompts, a warning is logged but function still completes."""
        from replay_eval import run_replay_eval
        log = tmp_path / 'decision_log.jsonl'
        self._write_log(log, n=5)  # < 30 floor

        with patch('replay_eval._db') as mock_db, \
             patch('replay_eval._rerun_decision', return_value={'decision': 'buy', 'confidence': 0.8}), \
             patch('replay_eval._simulate_trade', return_value={'pnl_pct': 0.03, 'exit_reason': 'take_profit'}), \
             patch('replay_eval.logging') as mock_log:
            mock_db.get_existing_prompt_hashes.return_value = set()
            result = run_replay_eval(log_path=log, bot='stock',
                                     lookback_days=30, seed=42)

        # Warning should have been logged
        assert mock_log.warning.called

    def test_result_json_written_to_eval_dir(self, tmp_path):
        from replay_eval import run_replay_eval
        log = tmp_path / 'decision_log.jsonl'
        self._write_log(log, n=5)
        eval_dir = tmp_path / 'eval'

        with patch('replay_eval._db') as mock_db, \
             patch('replay_eval._rerun_decision', return_value={'decision': 'buy', 'confidence': 0.8}), \
             patch('replay_eval._simulate_trade', return_value={'pnl_pct': 0.02, 'exit_reason': 'end_of_period'}), \
             patch('replay_eval._EVAL_DIR', eval_dir):
            mock_db.get_existing_prompt_hashes.return_value = set()
            eval_dir.mkdir(parents=True)
            run_replay_eval(log_path=log, lookback_days=30, seed=42)

        written = list(eval_dir.glob('replay_eval_*.json'))
        assert len(written) == 1
        data = json.loads(written[0].read_text())
        assert 'expectancy' in data
        assert 'max_dd' in data
        assert 'prompt_set_hash' in data


# ---------------------------------------------------------------------------
# model_promoter two-stage gate (PR-4)
# ---------------------------------------------------------------------------

class TestModelPromoterTwoStageGate:
    """Smoke test hard fail and replay gate behaviour."""

    def _make_eval_result(self, grade='B', pass_rate=0.75, format_score=0.95,
                          directional_acc=0.80) -> dict:
        return {
            'grade':          grade,
            'pass_rate':      pass_rate,
            'format_score':   format_score,
            'directional_acc': directional_acc,
        }

    def _make_replay_result(self, expectancy=0.02, max_dd=0.08,
                             trade_count=15) -> dict:
        return {
            'expectancy':      expectancy,
            'max_dd':          max_dd,
            'trade_count':     trade_count,
            'n_symbols':       5,
            'prompt_set_hash': 'abc123',
        }

    def test_grade_f_always_rejected(self, tmp_path):
        import model_promoter
        adapter1 = tmp_path / 'adapter1'
        adapter1.mkdir()
        adapter2 = tmp_path / 'adapter2'
        adapter2.mkdir()
        latest = tmp_path / 'lora_latest'
        latest.symlink_to(str(adapter1))

        with patch.object(model_promoter, '_current_production_adapter', return_value=str(adapter1)), \
             patch.object(model_promoter, '_run_eval', side_effect=[
                 self._make_eval_result(grade='F', pass_rate=0.0, format_score=0.3),
                 self._make_eval_result(grade='B', pass_rate=0.75),
             ]), \
             patch.object(model_promoter, '_write_promotion_record'), \
             patch.object(model_promoter, '_LATEST', latest):
            result = model_promoter.promote(str(adapter2))

        assert result is False

    def test_format_below_90_rejected(self, tmp_path):
        import model_promoter
        adapter1 = tmp_path / 'adapter1'
        adapter1.mkdir()
        adapter2 = tmp_path / 'adapter2'
        adapter2.mkdir()

        with patch.object(model_promoter, '_current_production_adapter', return_value=str(adapter1)), \
             patch.object(model_promoter, '_run_eval', side_effect=[
                 self._make_eval_result(grade='C', pass_rate=0.5, format_score=0.80),
                 self._make_eval_result(grade='B', pass_rate=0.75),
             ]), \
             patch.object(model_promoter, '_write_promotion_record'):
            result = model_promoter.promote(str(adapter2))

        assert result is False

    def test_replay_gate_promotes_when_candidate_better(self, tmp_path):
        import model_promoter
        adapter1 = tmp_path / 'adapter1'
        adapter1.mkdir()
        adapter2 = tmp_path / 'adapter2'
        adapter2.mkdir()
        latest = tmp_path / 'lora_latest'
        latest.symlink_to(str(adapter1))

        with patch.object(model_promoter, '_ENABLE_REPLAY_GATE', True), \
             patch.object(model_promoter, '_PROMOTER_SHADOW', False), \
             patch.object(model_promoter, '_current_production_adapter', return_value=str(adapter1)), \
             patch.object(model_promoter, '_run_eval', side_effect=[
                 self._make_eval_result(),
                 self._make_eval_result(),
             ]), \
             patch.object(model_promoter, '_run_replay_eval', side_effect=[
                 self._make_replay_result(expectancy=0.05, max_dd=0.06),  # candidate: better
                 self._make_replay_result(expectancy=0.02, max_dd=0.08),  # incumbent
             ]), \
             patch.object(model_promoter, '_update_latest_symlink'), \
             patch.object(model_promoter, '_update_env'), \
             patch.object(model_promoter, '_restart_services'), \
             patch.object(model_promoter, '_write_promotion_record'), \
             patch.object(model_promoter, '_LATEST', latest), \
             patch.object(model_promoter, '_ENV_FILE', tmp_path / '.env'):
            result = model_promoter.promote(str(adapter2))

        assert result is True

    def test_replay_gate_rejects_when_candidate_worse(self, tmp_path):
        import model_promoter
        adapter1 = tmp_path / 'adapter1'
        adapter1.mkdir()
        adapter2 = tmp_path / 'adapter2'
        adapter2.mkdir()

        with patch.object(model_promoter, '_ENABLE_REPLAY_GATE', True), \
             patch.object(model_promoter, '_PROMOTER_SHADOW', False), \
             patch.object(model_promoter, '_current_production_adapter', return_value=str(adapter1)), \
             patch.object(model_promoter, '_run_eval', side_effect=[
                 self._make_eval_result(),
                 self._make_eval_result(),
             ]), \
             patch.object(model_promoter, '_run_replay_eval', side_effect=[
                 self._make_replay_result(expectancy=-0.03, max_dd=0.15),  # candidate: worse
                 self._make_replay_result(expectancy=0.02, max_dd=0.08),   # incumbent
             ]), \
             patch.object(model_promoter, '_write_promotion_record'):
            result = model_promoter.promote(str(adapter2))

        assert result is False

    def test_shadow_mode_uses_smoke_test_decision(self, tmp_path):
        """In shadow mode, replay verdict is logged but smoke-test decision applies."""
        import model_promoter
        adapter1 = tmp_path / 'adapter1'
        adapter1.mkdir()
        adapter2 = tmp_path / 'adapter2'
        adapter2.mkdir()
        latest = tmp_path / 'lora_latest'
        latest.symlink_to(str(adapter1))

        with patch.object(model_promoter, '_ENABLE_REPLAY_GATE', True), \
             patch.object(model_promoter, '_PROMOTER_SHADOW', True), \
             patch.object(model_promoter, '_current_production_adapter', return_value=str(adapter1)), \
             patch.object(model_promoter, '_run_eval', side_effect=[
                 # Candidate beats incumbent by smoke test (pass_rate higher)
                 self._make_eval_result(pass_rate=0.90),
                 self._make_eval_result(pass_rate=0.75),
             ]), \
             patch.object(model_promoter, '_run_replay_eval', side_effect=[
                 self._make_replay_result(expectancy=-0.05),  # replay says reject
                 self._make_replay_result(expectancy=0.02),
             ]), \
             patch.object(model_promoter, '_update_latest_symlink'), \
             patch.object(model_promoter, '_update_env'), \
             patch.object(model_promoter, '_restart_services'), \
             patch.object(model_promoter, '_write_promotion_record'), \
             patch.object(model_promoter, '_LATEST', latest), \
             patch.object(model_promoter, '_ENV_FILE', tmp_path / '.env'):
            result = model_promoter.promote(str(adapter2))

        # Smoke-test says promote (pass_rate 90% > 75%), shadow mode applied → promoted
        assert result is True

    def test_insufficient_trades_rejects_promotion(self, tmp_path):
        """Candidate < 5 trades, incumbent ≥ 5 → replay rejects."""
        import model_promoter
        adapter1 = tmp_path / 'adapter1'
        adapter1.mkdir()
        adapter2 = tmp_path / 'adapter2'
        adapter2.mkdir()

        with patch.object(model_promoter, '_ENABLE_REPLAY_GATE', True), \
             patch.object(model_promoter, '_PROMOTER_SHADOW', False), \
             patch.object(model_promoter, '_current_production_adapter', return_value=str(adapter1)), \
             patch.object(model_promoter, '_run_eval', side_effect=[
                 self._make_eval_result(),
                 self._make_eval_result(),
             ]), \
             patch.object(model_promoter, '_run_replay_eval', side_effect=[
                 self._make_replay_result(trade_count=2),   # candidate: too few trades
                 self._make_replay_result(trade_count=12),  # incumbent: many trades
             ]), \
             patch.object(model_promoter, '_write_promotion_record'):
            result = model_promoter.promote(str(adapter2))

        assert result is False

    def test_promotion_record_includes_replay_metrics(self, tmp_path):
        """When replay gate runs, promotion record must contain replay metrics."""
        import model_promoter
        adapter1 = tmp_path / 'adapter1'
        adapter1.mkdir()
        adapter2 = tmp_path / 'adapter2'
        adapter2.mkdir()
        latest = tmp_path / 'lora_latest'
        latest.symlink_to(str(adapter1))
        eval_dir = tmp_path / 'eval'
        eval_dir.mkdir()

        with patch.object(model_promoter, '_ENABLE_REPLAY_GATE', True), \
             patch.object(model_promoter, '_PROMOTER_SHADOW', False), \
             patch.object(model_promoter, '_current_production_adapter', return_value=str(adapter1)), \
             patch.object(model_promoter, '_run_eval', side_effect=[
                 self._make_eval_result(),
                 self._make_eval_result(),
             ]), \
             patch.object(model_promoter, '_run_replay_eval', side_effect=[
                 self._make_replay_result(expectancy=0.04),
                 self._make_replay_result(expectancy=0.02),
             ]), \
             patch.object(model_promoter, '_update_latest_symlink'), \
             patch.object(model_promoter, '_update_env'), \
             patch.object(model_promoter, '_restart_services'), \
             patch.object(model_promoter, '_EVAL_DIR', eval_dir), \
             patch.object(model_promoter, '_LATEST', latest), \
             patch.object(model_promoter, '_ENV_FILE', tmp_path / '.env'):
            model_promoter.promote(str(adapter2))

        records = list(eval_dir.glob('promotion_*.json'))
        assert len(records) == 1
        data = json.loads(records[0].read_text())
        assert 'replay_candidate' in data
        assert 'replay_current' in data
        assert 'expectancy' in data['replay_candidate']
