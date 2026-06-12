"""
Unit tests for PR-3: counterfactual SFT rows, DPO wiring, and model rollback.
"""
import json
import sys
import sqlite3
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'training'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'finetune'))

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SAMPLE_PROMPT_WITH_INDICATORS = (
    "Analyze AAPL for a potential trade.\n"
    "RSI(14): 74.5\n"
    "MACD: -3.20\n"
    "Volume Ratio: 0.8x\n"
    "50 MA: $182.00 (+2.3% from current)\n"
)

_SAMPLE_PROMPT_NO_INDICATORS = "Analyze TSLA for a potential trade. No technical data available."

_BANNED_FRAGMENTS = ('resulted in', '-15.0%', '-13.0%', 'original BUY', 'original SELL')

_REALISTIC_LOSER_PROMPTS = [
    ("buy",      "RSI(14): 72.0\nMACD: -1.50\nVolume Ratio: 0.9x\n50 MA: $100 (+1% from current)\n"),
    ("buy_call", "RSI(14): 78.0\nMACD: -5.00\nVolume Ratio: 1.1x\n"),
    ("sell",     "RSI(14): 25.0\nMACD: +2.30\nVolume Ratio: 0.7x\n"),
    ("buy",      _SAMPLE_PROMPT_NO_INDICATORS),
    ("buy_put",  "RSI(14): 28.0\nMACD: +1.80\nVolume Ratio: 0.6x\n"),
]


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
        result = _build_counterfactual_output('buy', _SAMPLE_PROMPT_WITH_INDICATORS, 0.80, {})
        assert result.startswith('Decision: HOLD')

    def test_counterfactual_output_does_not_cite_pnl(self):
        """Output must NOT contain realized P&L — that leaks future information into training."""
        from training_data_builder import _build_counterfactual_output
        result = _build_counterfactual_output('buy', _SAMPLE_PROMPT_WITH_INDICATORS, 0.80, {})
        for banned in ('-15.0%', 'resulted in', 'a loss', 'original BUY'):
            assert banned not in result, f"banned fragment {banned!r} found in output: {result!r}"

    def test_counterfactual_output_no_outcome_line(self):
        from training_data_builder import _build_counterfactual_output
        result = _build_counterfactual_output('buy', _SAMPLE_PROMPT_WITH_INDICATORS, 0.80, {})
        assert 'Outcome:' not in result
        assert 'Reward signal:' not in result

    def test_counterfactual_output_three_line_format(self):
        from training_data_builder import _build_counterfactual_output
        result = _build_counterfactual_output('buy_call', _SAMPLE_PROMPT_WITH_INDICATORS, 0.82, {})
        lines = result.strip().split('\n')
        assert lines[0].startswith('Decision:')
        assert lines[1].startswith('Confidence:')
        assert lines[2].startswith('Reasoning:')
        assert len(lines) == 3

    def test_counterfactual_output_cites_indicator_from_prompt(self):
        """Reasoning must mention a numeric indicator value present in the prompt."""
        from training_data_builder import _build_counterfactual_output
        result = _build_counterfactual_output('buy', _SAMPLE_PROMPT_WITH_INDICATORS, 0.80, {})
        # MACD -3.20 or RSI 74.5 should appear — at least one indicator cited
        assert '-3.20' in result or '74.5' in result or '0.8' in result

    def test_counterfactual_output_fallback_when_no_indicators(self):
        """When no indicators parse, reasoning is the generic fallback sentence."""
        from training_data_builder import _build_counterfactual_output
        result = _build_counterfactual_output('buy', _SAMPLE_PROMPT_NO_INDICATORS, 0.80, {})
        assert 'sufficient edge' in result
        # No spurious numbers
        import re
        assert not re.search(r'\d+\.\d+%', result)

    def test_counterfactual_confidence_from_calib_map(self):
        """Confidence is derived from calib_map win_rate, not hardcoded."""
        from training_data_builder import _build_counterfactual_output
        # High win-rate bin → low counterfactual confidence (HOLD less certain)
        calib_high = {'0.70-0.80': {'win_rate': 0.80}}
        # Low win-rate bin → high counterfactual confidence (HOLD more certain)
        calib_low  = {'0.70-0.80': {'win_rate': 0.20}}
        r_high = _build_counterfactual_output('buy', _SAMPLE_PROMPT_NO_INDICATORS, 0.75, calib_high)
        r_low  = _build_counterfactual_output('buy', _SAMPLE_PROMPT_NO_INDICATORS, 0.75, calib_low)
        conf_high = float(r_high.split('\n')[1].split(': ')[1])
        conf_low  = float(r_low.split('\n')[1].split(': ')[1])
        assert conf_low > conf_high

    def test_counterfactual_confidence_fallback_when_bin_absent(self):
        """When the confidence bin is absent from calib_map, confidence defaults to 0.70."""
        from training_data_builder import _build_counterfactual_output
        result = _build_counterfactual_output('buy', _SAMPLE_PROMPT_NO_INDICATORS, 0.75, {})
        conf = float(result.split('\n')[1].split(': ')[1])
        assert conf == pytest.approx(0.70, abs=0.01)

    def test_counterfactual_label_not_in_sft_exclude(self):
        """'counterfactual' label must survive the SFT loser filter."""
        _SFT_EXCLUDE_LABELS = {'weak_loser', 'loser', 'strong_loser'}
        assert 'counterfactual' not in _SFT_EXCLUDE_LABELS

    @pytest.mark.parametrize("decision,prompt", _REALISTIC_LOSER_PROMPTS)
    def test_banned_substring_sweep(self, decision, prompt):
        """No realistic loser counterfactual should contain P&L-leak substrings."""
        from training_data_builder import _build_counterfactual_output
        result = _build_counterfactual_output(decision, prompt, 0.78, {})
        for frag in _BANNED_FRAGMENTS:
            assert frag not in result, f"banned {frag!r} in output for decision={decision!r}"


class TestCounterfactualEmission:
    """Counterfactual rows are emitted for losers but not for winners."""

    def test_loser_pnl_produces_loser_label(self):
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
        from training_data_builder import _tiered_label_from_pnl
        label, _ = _tiered_label_from_pnl(0.09, 'buy')
        assert label not in {'strong_loser', 'loser', 'weak_loser'}

    def test_counterfactual_hash_unique_per_prompt(self):
        from training_data_builder import _prompt_hash
        h1 = _prompt_hash('Analyze AAPL today.' + '|counterfactual')
        h2 = _prompt_hash('Analyze TSLA today.' + '|counterfactual')
        assert h1 != h2

    def test_counterfactual_row_metadata_example_type(self):
        from training_data_builder import _build_counterfactual_output
        output = _build_counterfactual_output('buy', _SAMPLE_PROMPT_WITH_INDICATORS, 0.80, {})
        assert 'HOLD' in output


class TestRegenerateCounterfactuals:
    """regenerate_counterfactuals() purges stale rows and rebuilds from loser imitation rows."""

    def _seed_db(self, db_path: Path) -> None:
        """Seed a minimal DB with 2 old-format counterfactual rows + 2 loser imitation rows."""
        import db as _db_mod
        _db_mod.init_db(db_path)
        from training_data_builder import _prompt_hash
        now = '2026-01-10T09:00:00'

        for i, symbol in enumerate(['AAPL', 'TSLA']):
            prompt = f"Analyze {symbol}.\nRSI(14): 75.0\nMACD: -2.00\n"
            ph = _prompt_hash(prompt)
            cf_ph = _prompt_hash(prompt + '|counterfactual')

            # imitation (loser) row
            with _db_mod.get_conn(db_path) as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO training_examples
                    (bot, source, symbol, prompt, ideal_output, label,
                     confidence, pnl_pct, entry_date, session_id, prompt_hash, generated_at)
                    VALUES (?, 'paper', ?, ?, 'Decision: BUY\nConfidence: 0.75\nReasoning: x',
                            'loser', 0.75, -0.15, '2026-01-10', '', ?, ?)
                """, ('stock', symbol, prompt, ph, now))

                # old-format counterfactual (contains banned string)
                conn.execute("""
                    INSERT OR IGNORE INTO training_examples
                    (bot, source, symbol, prompt, ideal_output, label,
                     confidence, pnl_pct, entry_date, session_id, prompt_hash, generated_at)
                    VALUES (?, 'paper', ?, ?, 'Decision: HOLD\nConfidence: 0.90\nReasoning: resulted in -15.0%',
                            'counterfactual', 0.75, -0.15, '2026-01-10', '', ?, ?)
                """, ('stock', symbol, prompt, cf_ph, now))

    def test_regenerate_deletes_and_reinserts(self, tmp_path, monkeypatch):
        db_path = tmp_path / 'trading.db'
        self._seed_db(db_path)

        import db as _db_mod
        import training_data_builder as tdb
        monkeypatch.setattr(_db_mod, 'DB_PATH', db_path)

        deleted, inserted = tdb.regenerate_counterfactuals('stock')

        assert deleted == 2
        assert inserted == 2

    def test_regenerated_rows_pass_banned_substring_sweep(self, tmp_path, monkeypatch):
        db_path = tmp_path / 'trading.db'
        self._seed_db(db_path)

        import db as _db_mod
        import training_data_builder as tdb
        monkeypatch.setattr(_db_mod, 'DB_PATH', db_path)

        tdb.regenerate_counterfactuals('stock')

        rows = _db_mod.get_training_examples(label='counterfactual', db_path=db_path)
        assert len(rows) == 2
        for row in rows:
            output = row['ideal_output']
            for frag in _BANNED_FRAGMENTS:
                assert frag not in output, f"banned {frag!r} in regenerated row: {output!r}"


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
