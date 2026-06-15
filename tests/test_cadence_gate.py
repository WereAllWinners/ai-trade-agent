"""
Session 5 — cadence gate tests.

The gate lives in finetune_model._cadence_gate(path, bot).
It reads the SFT-only file (training_data_sft.json / options_training_data_sft.json)
and refuses to launch the fine-tune subprocess when data is insufficient.

A fired gate (return False) is the CORRECT outcome when:
  take_trade < _MIN_TAKE_TRADE (40)
  hold_share > _SFT_MAX_HOLD_SHARE (0.65)
  total_sft  < _MIN_TOTAL_SFT (200)

Current live state fires on the first two conditions:
  Stock:   take_trade=10, hold_share=68.8%
  Options: take_trade=6,  hold_share=83.3%
"""
import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'training'))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WINNER_OUT   = 'Decision: BUY\nConfidence: 0.90\nReasoning: Strong momentum breakout.'
_SW_OUT       = 'Decision: BUY\nConfidence: 0.95\nReasoning: Very strong momentum breakout.'
_HOLD_OUT     = 'Decision: HOLD\nConfidence: 0.80\nReasoning: No clear edge; hold best.'
_CF_OUT       = 'Decision: HOLD\nConfidence: 0.80\nReasoning: Entry risk outweighs setup — MACD negative.'
_CB_OUT       = 'Decision: HOLD\nConfidence: 1.00\nReasoning: Trade blocked: correlation cap breached.'


def _sft_row(label: str, i: int = 0) -> dict:
    outputs = {
        'winner':          _WINNER_OUT,
        'strong_winner':   _SW_OUT,
        'correct_hold':    _HOLD_OUT,
        'counterfactual':  _CF_OUT,
        'constraint_block': _CB_OUT,
    }
    return {
        'input':  f'Analyze SYM{i} for a potential trade.',
        'output': outputs.get(label, _WINNER_OUT),
        'label':  label,
        'weight': 1.0,
        'metadata': {'entry_date': '2026-06-15'},
    }


def _write_sft_file(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))


# ---------------------------------------------------------------------------
# Live-state gate-fires tests
# ---------------------------------------------------------------------------

class TestCadenceGateLiveState:
    """Both bots fire at current data volumes (stock=10, options=6 take-trade)."""

    def _gate(self, path, bot='stock'):
        from finetune_model import _cadence_gate
        return _cadence_gate(path, bot)

    def test_stock_fires_at_10_take_trade(self, tmp_path, caplog):
        """Stock bot: 10 winner + 22 hold = 32 SFT, take_trade=10 < 40 → FIRES."""
        rows = (
            [_sft_row('winner', i) for i in range(9)]
            + [_sft_row('strong_winner', 100)]
            + [_sft_row('correct_hold', i + 200) for i in range(7)]
            + [_sft_row('constraint_block', i + 300) for i in range(8)]
            + [_sft_row('counterfactual', i + 400) for i in range(7)]
        )  # 10 take-trade, 22 hold, 32 total
        path = tmp_path / 'training_data_sft.json'
        _write_sft_file(path, rows)

        with caplog.at_level(logging.WARNING):
            result = self._gate(path, 'stock')

        assert result is False, "Gate must FIRE at take_trade=10 < 40"
        assert any('take_trade=10' in r.getMessage() for r in caplog.records), \
            "Log must cite take_trade=10 as reason"
        assert any('SKIPPED' in r.getMessage() for r in caplog.records), \
            "Log must say fine-tune SKIPPED"

    def test_options_fires_at_6_take_trade(self, tmp_path, caplog):
        """Options bot: 6 winner + 30 hold = 36 SFT, take_trade=6 < 40 → FIRES."""
        rows = (
            [_sft_row('winner', i) for i in range(4)]
            + [_sft_row('strong_winner', 100), _sft_row('strong_winner', 101)]
            + [_sft_row('correct_hold', i + 200) for i in range(27)]
            + [_sft_row('counterfactual', i + 300) for i in range(3)]
        )  # 6 take-trade, 30 hold, 36 total
        path = tmp_path / 'options_training_data_sft.json'
        _write_sft_file(path, rows)

        with caplog.at_level(logging.WARNING):
            result = self._gate(path, 'options')

        assert result is False, "Gate must FIRE at take_trade=6 < 40"
        assert any('take_trade=6' in r.getMessage() for r in caplog.records)

    def test_gate_fire_does_not_launch_subprocess(self, tmp_path):
        """When gate fires, finetune_model() must NOT call subprocess.run."""
        rows = [_sft_row('winner', i) for i in range(5)]  # 5 << 40, gate fires
        path = tmp_path / 'training_data_sft.json'
        _write_sft_file(path, rows)

        with patch('finetune_model.subprocess') as mock_sub, \
             patch('finetune_model._gpu_temp', return_value=None):
            from finetune_model import finetune_model as fm
            fm(path)

        mock_sub.run.assert_not_called()


# ---------------------------------------------------------------------------
# Gate-passes test
# ---------------------------------------------------------------------------

class TestCadenceGatePasses:
    """Gate passes when all three conditions are met."""

    def _gate(self, path, bot='stock'):
        from finetune_model import _cadence_gate
        return _cadence_gate(path, bot)

    def test_passes_with_sufficient_data(self, tmp_path, caplog):
        """45 take-trade + 25 hold / 50% hold / 220 total → gate PASSES."""
        rows = (
            [_sft_row('winner', i) for i in range(40)]
            + [_sft_row('strong_winner', i + 1000) for i in range(5)]
            + [_sft_row('correct_hold', i + 2000) for i in range(12)]
            + [_sft_row('constraint_block', i + 3000) for i in range(8)]
            + [_sft_row('counterfactual', i + 4000) for i in range(5)]
            + [_sft_row('winner', i + 5000) for i in range(150)]
        )  # 195 take-trade, 25 hold, 220 total → hold=11.4%
        path = tmp_path / 'training_data_sft.json'
        _write_sft_file(path, rows)

        with caplog.at_level(logging.INFO):
            result = self._gate(path, 'stock')

        assert result is True, "Gate must PASS with 195 take-trade, 11.4% hold, 220 total"
        assert not any('SKIPPED' in r.getMessage() for r in caplog.records)
        assert any('PASSED' in r.getMessage() for r in caplog.records)

    def test_passes_at_minimum_viable_data(self, tmp_path):
        """Exactly at all three thresholds: 40 take-trade, 65% hold, 200 total → PASSES."""
        # 40 take-trade + 75 hold (65% of 115) + must hit 200 total
        # hold_count / total_sft <= 0.65 and take_trade >= 40 and total >= 200
        # Let: hold = 130, take = 70, total = 200 → hold_share = 65% exactly → PASS (not strictly >)
        rows = (
            [_sft_row('winner', i) for i in range(40)]
            + [_sft_row('strong_winner', i + 1000) for i in range(30)]
            + [_sft_row('correct_hold', i + 2000) for i in range(130)]
        )  # 70 take-trade, 130 hold, 200 total → hold_share = 65.0% exactly → PASS
        path = tmp_path / 'training_data_sft.json'
        _write_sft_file(path, rows)

        from finetune_model import _cadence_gate
        result = _cadence_gate(path, 'stock')
        assert result is True, "65.0% hold_share == threshold (not >) must PASS"


# ---------------------------------------------------------------------------
# Single-condition flip tests
# ---------------------------------------------------------------------------

class TestCadenceGateConditionFlip:
    """Removing any single met condition flips PASS → FIRE."""

    def _make_passing_rows(self, take=45, hold=25, total=220):
        """Return a list of SFT rows that satisfy all three gate conditions."""
        filler = total - take - hold
        rows = (
            [_sft_row('winner', i) for i in range(take)]
            + [_sft_row('correct_hold', i + 1000) for i in range(hold)]
            + [_sft_row('winner', i + 5000) for i in range(filler)]
        )
        return rows

    def _gate(self, path, bot='stock'):
        from finetune_model import _cadence_gate
        return _cadence_gate(path, bot)

    def test_base_case_passes(self, tmp_path):
        """Sanity: base configuration (45 take, 25 hold, 220 total) passes."""
        path = tmp_path / 'base.json'
        _write_sft_file(path, self._make_passing_rows())
        assert self._gate(path) is True

    def test_drop_take_trade_below_floor_fires(self, tmp_path):
        """Take trade: 39 (below 40 floor) with hold rows for filler → FIRES.

        Filler must be correct_hold (not winner) so take_trade stays at 39.
        Note: all SFT labels are either take-trade or hold-teaching, so having
        enough rows for total_sft ≥ 200 while keeping take_trade=39 forces
        hold_share > 65% — multiple conditions fire simultaneously, all correct.
        """
        rows = (
            [_sft_row('winner', i) for i in range(39)]   # 39 < 40 → gate fires
            + [_sft_row('correct_hold', i + 1000) for i in range(25)]
        )  # total=64, take_trade=39 < 40, total_sft=64 < 200 — gate fires (both reasons)
        path = tmp_path / 'low_take.json'
        _write_sft_file(path, rows)
        assert self._gate(path) is False

    def test_raise_hold_share_above_ceiling_fires(self, tmp_path):
        """Hold share: ~11% → 66% (above 65% ceiling) → FIRES."""
        # 45 take-trade, 88 hold → 133 total → 88/133 = 66.2% > 65%
        rows = (
            [_sft_row('winner', i) for i in range(45)]
            + [_sft_row('correct_hold', i + 1000) for i in range(88)]
        )
        path = tmp_path / 'high_hold.json'
        _write_sft_file(path, rows)
        assert self._gate(path) is False

    def test_drop_total_below_floor_fires(self, tmp_path):
        """Total SFT: 220→199 (below 200 floor) → FIRES."""
        rows = (
            [_sft_row('winner', i) for i in range(45)]
            + [_sft_row('correct_hold', i + 1000) for i in range(25)]
            + [_sft_row('winner', i + 5000) for i in range(199 - 45 - 25)]
        )  # exactly 199 total
        path = tmp_path / 'low_total.json'
        _write_sft_file(path, rows)
        assert self._gate(path) is False


# ---------------------------------------------------------------------------
# Gate reads SFT file, not archive
# ---------------------------------------------------------------------------

class TestCadenceGateReadsCorrectFile:
    """Gate must read the SFT-only file, not the combined archive.

    Archive has 288 rows (32 SFT + 256 loser).  Gate on 288 would incorrectly
    count take-trade from DPO rows or use wrong hold-share denominator.
    """

    def test_gate_counts_from_sft_file_not_archive(self, tmp_path):
        """Gate reads SFT file (10 take-trade → FIRE). Archive has 250 winners but is ignored."""
        sft_rows = (
            [_sft_row('winner', i) for i in range(10)]     # 10 take-trade → fires
            + [_sft_row('correct_hold', i + 1000) for i in range(22)]
        )
        archive_rows = sft_rows + [_sft_row('winner', i + 9000) for i in range(240)]
        # Archive has 252 winners — if gate counted from archive, it would PASS

        sft_path = tmp_path / 'training_data_sft.json'
        archive_path = tmp_path / 'training_data_archive.json'
        _write_sft_file(sft_path, sft_rows)
        _write_sft_file(archive_path, archive_rows)

        from finetune_model import _cadence_gate
        result = _cadence_gate(sft_path, 'stock')
        assert result is False, \
            "Gate must read the SFT file (10 take-trade → FIRE), not the archive (250 winners)"
