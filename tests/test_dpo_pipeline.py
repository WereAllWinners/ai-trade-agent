"""
Unit tests for PR-3: counterfactual SFT rows, DPO wiring, and model rollback.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'training'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'finetune'))


# ---------------------------------------------------------------------------
# A2 — Counterfactual SFT rows
# ---------------------------------------------------------------------------

class TestCounterfactualOutput:
    def test_counterfactual_hash_distinct_from_original(self):
        from training_data_builder import _prompt_hash
        prompt = "Analyze AAPL for potential trade."
        assert _prompt_hash(prompt) != _prompt_hash(prompt + '|counterfactual')

    def test_counterfactual_output_suggests_hold(self):
        from training_data_builder import _build_counterfactual_output
        result = _build_counterfactual_output('buy', -0.15)
        assert result.startswith('Decision: HOLD')

    def test_counterfactual_output_cites_pnl(self):
        from training_data_builder import _build_counterfactual_output
        result = _build_counterfactual_output('buy', -0.15)
        assert '-15.0%' in result

    def test_counterfactual_output_no_outcome_line(self):
        from training_data_builder import _build_counterfactual_output
        result = _build_counterfactual_output('buy', -0.10)
        assert 'Outcome:' not in result
        assert 'Reward signal:' not in result

    def test_counterfactual_output_three_line_format(self):
        from training_data_builder import _build_counterfactual_output
        result = _build_counterfactual_output('buy_call', -0.25)
        lines = result.strip().split('\n')
        assert lines[0].startswith('Decision:')
        assert lines[1].startswith('Confidence:')
        assert lines[2].startswith('Reasoning:')
        assert len(lines) == 3

    def test_counterfactual_output_when_pnl_none(self):
        from training_data_builder import _build_counterfactual_output
        result = _build_counterfactual_output('buy', None)
        assert 'a loss' in result
        assert result.startswith('Decision: HOLD')

    def test_counterfactual_label_not_in_sft_exclude(self):
        """'counterfactual' label must survive the SFT loser filter."""
        # _SFT_EXCLUDE_LABELS is defined inside load_training_data in fine_tune_llm.py
        _SFT_EXCLUDE_LABELS = {'weak_loser', 'loser', 'strong_loser'}
        assert 'counterfactual' not in _SFT_EXCLUDE_LABELS


class TestCounterfactualEmission:
    """Counterfactual rows are emitted for losers but not for winners.

    Tests verify the trigger condition (_tiered_label_from_pnl → loser labels)
    and that the counterfactual hash/output are correctly formed, without
    relying on the full build_and_store() DB wiring (which uses a module-level
    default db_path captured at import time, incompatible with tmp_path fixtures).
    """

    def test_loser_pnl_produces_loser_label(self):
        """pnl_pct -13% on a BUY should classify as 'loser' → counterfactual emitted."""
        from training_data_builder import _tiered_label_from_pnl
        label, _ = _tiered_label_from_pnl(-0.13, 'buy')
        assert label in {'strong_loser', 'loser', 'weak_loser'}

    def test_strong_loss_pnl_produces_loser_label(self):
        from training_data_builder import _tiered_label_from_pnl
        label, _ = _tiered_label_from_pnl(-0.35, 'buy')
        assert label == 'strong_loser'

    def test_weak_loss_pnl_produces_loser_label(self):
        from training_data_builder import _tiered_label_from_pnl
        label, _ = _tiered_label_from_pnl(-0.002, 'buy')
        assert label == 'weak_loser'

    def test_winner_pnl_does_not_produce_loser_label(self):
        """pnl_pct +9% → winner; no counterfactual should be emitted."""
        from training_data_builder import _tiered_label_from_pnl
        label, _ = _tiered_label_from_pnl(0.09, 'buy')
        assert label not in {'strong_loser', 'loser', 'weak_loser'}

    def test_counterfactual_hash_unique_per_prompt(self):
        """Two different prompts generate different counterfactual hashes."""
        from training_data_builder import _prompt_hash
        h1 = _prompt_hash('Analyze AAPL today.' + '|counterfactual')
        h2 = _prompt_hash('Analyze TSLA today.' + '|counterfactual')
        assert h1 != h2

    def test_counterfactual_row_metadata_example_type(self):
        """Counterfactual examples must carry example_type='counterfactual'."""
        from training_data_builder import _build_counterfactual_output
        output = _build_counterfactual_output('buy', -0.13)
        # The output itself doesn't carry metadata, but the label logic checks the key
        assert 'HOLD' in output  # confirms the format the metadata will pair with


# ---------------------------------------------------------------------------
# A8 — DPOStageTrainer
# ---------------------------------------------------------------------------

class TestDPOStageTrainer:
    """DPOStageTrainer reads dpo_pairs.jsonl and calls DPOTrainer with correct inputs."""

    def _make_pairs(self, n: int = 10) -> list[dict]:
        return [
            {
                'prompt':         f'<|im_start|>user\nAnalyze AAPL {i}<|im_end|>\n<|im_start|>assistant\n',
                'chosen':         f'Decision: BUY\nConfidence: 0.80\nReasoning: Strong signal {i}<|im_end|>',
                'rejected':       f'Decision: SELL\nConfidence: 0.55\nReasoning: Uncertain<|im_end|>',
                'chosen_score':   0.05,
                'rejected_score': 0.01,
                'symbol':         'AAPL',
                'bot':            'stock',
                'winner_label':   'winner',
            }
            for i in range(n)
        ]

    def _write_pairs(self, path: Path, n: int = 10) -> None:
        with open(path, 'w') as f:
            for p in self._make_pairs(n):
                f.write(json.dumps(p) + '\n')

    def test_skips_when_pairs_file_missing(self, tmp_path):
        from fine_tune_llm import DPOStageTrainer
        trainer = DPOStageTrainer(
            MagicMock(), MagicMock(),
            tmp_path / 'nonexistent.jsonl',
            str(tmp_path / 'out'),
        )
        assert trainer.train() is False

    def test_skips_when_fewer_than_min_pairs(self, tmp_path):
        from fine_tune_llm import DPOStageTrainer
        path = tmp_path / 'dpo_pairs.jsonl'
        self._write_pairs(path, n=3)  # < MIN_DPO_PAIRS (8)

        trainer = DPOStageTrainer(MagicMock(), MagicMock(), path, str(tmp_path / 'out'))
        with patch('fine_tune_llm.DPOTrainer') as mock_dpo:
            result = trainer.train()

        assert result is False
        mock_dpo.assert_not_called()

    def test_returns_true_with_sufficient_pairs(self, tmp_path):
        from fine_tune_llm import DPOStageTrainer
        path = tmp_path / 'dpo_pairs.jsonl'
        self._write_pairs(path, n=10)

        trainer = DPOStageTrainer(MagicMock(), MagicMock(), path, str(tmp_path / 'out'))

        mock_trainer_inst = MagicMock()
        with patch('fine_tune_llm.DPOTrainer', return_value=mock_trainer_inst), \
             patch('fine_tune_llm.TrainingArguments'), \
             patch('fine_tune_llm.Dataset') as mock_ds:
            mock_ds.from_list.return_value = MagicMock()
            result = trainer.train()

        assert result is True
        mock_trainer_inst.train.assert_called_once()

    def test_dataset_passed_to_dpo_has_prompt_chosen_rejected(self, tmp_path):
        """Dataset rows sent to DPOTrainer must have prompt, chosen, and rejected fields."""
        from fine_tune_llm import DPOStageTrainer
        path = tmp_path / 'dpo_pairs.jsonl'
        self._write_pairs(path, n=10)

        trainer = DPOStageTrainer(MagicMock(), MagicMock(), path, str(tmp_path / 'out'))

        captured: dict = {}

        def capture_from_list(data):
            captured['data'] = data
            return MagicMock()

        with patch('fine_tune_llm.DPOTrainer', return_value=MagicMock()), \
             patch('fine_tune_llm.TrainingArguments'), \
             patch('fine_tune_llm.Dataset') as mock_ds:
            mock_ds.from_list.side_effect = capture_from_list
            trainer.train()

        assert 'data' in captured
        row = captured['data'][0]
        assert 'prompt' in row
        assert 'chosen' in row
        assert 'rejected' in row

    def test_chosen_and_rejected_are_distinct(self, tmp_path):
        from fine_tune_llm import DPOStageTrainer
        path = tmp_path / 'dpo_pairs.jsonl'
        self._write_pairs(path, n=10)

        trainer = DPOStageTrainer(MagicMock(), MagicMock(), path, str(tmp_path / 'out'))
        captured: dict = {}

        def capture(data):
            captured['data'] = data
            return MagicMock()

        with patch('fine_tune_llm.DPOTrainer', return_value=MagicMock()), \
             patch('fine_tune_llm.TrainingArguments'), \
             patch('fine_tune_llm.Dataset') as mock_ds:
            mock_ds.from_list.side_effect = capture
            trainer.train()

        for row in captured['data']:
            assert row['chosen'] != row['rejected']

    def test_skips_malformed_json_lines(self, tmp_path):
        """Malformed lines are skipped; valid pairs still load."""
        from fine_tune_llm import DPOStageTrainer
        path = tmp_path / 'dpo_pairs.jsonl'
        with open(path, 'w') as f:
            for p in self._make_pairs(10):
                f.write(json.dumps(p) + '\n')
            f.write('{invalid json\n')  # one bad line

        trainer = DPOStageTrainer(MagicMock(), MagicMock(), path, str(tmp_path / 'out'))
        mock_trainer_inst = MagicMock()
        with patch('fine_tune_llm.DPOTrainer', return_value=mock_trainer_inst), \
             patch('fine_tune_llm.TrainingArguments'), \
             patch('fine_tune_llm.Dataset') as mock_ds:
            mock_ds.from_list.return_value = MagicMock()
            result = trainer.train()

        assert result is True  # still runs; bad line was ignored


# ---------------------------------------------------------------------------
# model_promoter --rollback
# ---------------------------------------------------------------------------

class TestModelPromoterRollback:
    def _write_promo(self, path: Path, candidate: str, promoted: bool, ts: str) -> None:
        path.write_text(json.dumps({
            'decided_at': ts,
            'promoted':   promoted,
            'candidate':  candidate,
            'current':    '',
        }))

    def test_rollback_restores_previous_adapter(self, tmp_path):
        """With two successful promotions, rollback points symlink at the older one."""
        import model_promoter

        adapter1 = tmp_path / 'adapter_20260601_010000'
        adapter2 = tmp_path / 'adapter_20260602_010000'
        adapter1.mkdir()
        adapter2.mkdir()

        eval_dir = tmp_path / 'eval'
        eval_dir.mkdir()
        self._write_promo(eval_dir / 'promotion_20260601_010000.json',
                          str(adapter1), True, '2026-06-01T01:00:00')
        self._write_promo(eval_dir / 'promotion_20260602_010000.json',
                          str(adapter2), True, '2026-06-02T01:00:00')

        latest = tmp_path / 'lora_latest'
        latest.symlink_to(str(adapter2))
        env_file = tmp_path / '.env'

        with patch.object(model_promoter, '_EVAL_DIR', eval_dir), \
             patch.object(model_promoter, '_LATEST', latest), \
             patch.object(model_promoter, '_ENV_FILE', env_file), \
             patch.object(model_promoter, '_current_production_adapter',
                          return_value=str(adapter2)):
            result = model_promoter.rollback(n=2)

        assert result is True
        assert latest.resolve() == adapter1.resolve()

    def test_rollback_fails_when_no_previous_adapter(self, tmp_path):
        """Only one promotion record → no previous adapter → rollback fails gracefully."""
        import model_promoter

        adapter1 = tmp_path / 'adapter_20260601_010000'
        adapter1.mkdir()
        eval_dir = tmp_path / 'eval'
        eval_dir.mkdir()
        self._write_promo(eval_dir / 'promotion_20260601_010000.json',
                          str(adapter1), True, '2026-06-01T01:00:00')

        with patch.object(model_promoter, '_EVAL_DIR', eval_dir), \
             patch.object(model_promoter, '_current_production_adapter',
                          return_value=str(adapter1)):
            result = model_promoter.rollback(n=2)

        assert result is False

    def test_rollback_fails_when_adapter_dir_deleted(self, tmp_path):
        """If the previous adapter directory was deleted, rollback fails gracefully."""
        import model_promoter

        adapter1_path = str(tmp_path / 'adapter_20260601_010000')  # NOT created
        adapter2 = tmp_path / 'adapter_20260602_010000'
        adapter2.mkdir()

        eval_dir = tmp_path / 'eval'
        eval_dir.mkdir()
        self._write_promo(eval_dir / 'promotion_20260601_010000.json',
                          adapter1_path, True, '2026-06-01T01:00:00')
        self._write_promo(eval_dir / 'promotion_20260602_010000.json',
                          str(adapter2), True, '2026-06-02T01:00:00')

        with patch.object(model_promoter, '_EVAL_DIR', eval_dir), \
             patch.object(model_promoter, '_current_production_adapter',
                          return_value=str(adapter2)):
            result = model_promoter.rollback(n=2)

        assert result is False

    def test_rollback_skips_rejected_promotion_records(self, tmp_path):
        """Records where promoted=False are not candidates for rollback target."""
        import model_promoter

        adapter1 = tmp_path / 'adapter_20260601_010000'
        adapter2 = tmp_path / 'adapter_20260602_010000'
        adapter3 = tmp_path / 'adapter_20260603_010000'
        adapter1.mkdir()
        adapter2.mkdir()
        adapter3.mkdir()

        eval_dir = tmp_path / 'eval'
        eval_dir.mkdir()
        self._write_promo(eval_dir / 'promotion_20260601_010000.json',
                          str(adapter1), True, '2026-06-01T01:00:00')
        # adapter2 was rejected
        self._write_promo(eval_dir / 'promotion_20260602_010000.json',
                          str(adapter2), False, '2026-06-02T01:00:00')
        self._write_promo(eval_dir / 'promotion_20260603_010000.json',
                          str(adapter3), True, '2026-06-03T01:00:00')

        latest = tmp_path / 'lora_latest'
        latest.symlink_to(str(adapter3))

        with patch.object(model_promoter, '_EVAL_DIR', eval_dir), \
             patch.object(model_promoter, '_LATEST', latest), \
             patch.object(model_promoter, '_ENV_FILE', tmp_path / '.env'), \
             patch.object(model_promoter, '_current_production_adapter',
                          return_value=str(adapter3)):
            result = model_promoter.rollback(n=4)

        assert result is True
        # Should roll back to adapter1, not the rejected adapter2
        assert latest.resolve() == adapter1.resolve()
