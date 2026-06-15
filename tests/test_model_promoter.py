"""
test_model_promoter.py — Unit tests for model_promoter._write_promotion_record
and the three call sites that feed it.

Acceptance criteria (Item 1):
  - declined run  → merged_model = candidate's merged path (resolved), merged_model_promoted: false
  - promoted run  → merged_model = symlink-resolved path,              merged_model_promoted: true
  - no-merge run  → merged_model: null, merged_model_promoted: false,  no crash
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'training'))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def eval_dir(tmp_path):
    d = tmp_path / 'logs' / 'eval'
    d.mkdir(parents=True)
    return d


@pytest.fixture()
def merged_dir(tmp_path):
    d = tmp_path / 'finetune' / 'finance_qwen_32b_lora_merged_bf16_20260615_010237'
    d.mkdir(parents=True)
    return d


def _fake_result(pass_rate=0.80, directional_acc=0.75, format_score=0.95, grade='B'):
    return {
        'pass_rate': pass_rate,
        'directional_acc': directional_acc,
        'format_score': format_score,
        'grade': grade,
    }


def _latest_promotion(eval_dir: Path) -> dict:
    files = sorted(eval_dir.glob('promotion_*.json'), key=lambda f: f.stat().st_mtime)
    assert files, "no promotion record written"
    return json.loads(files[-1].read_text())


# ---------------------------------------------------------------------------
# Direct _write_promotion_record tests
# ---------------------------------------------------------------------------

def test_declined_run_writes_merged_model_and_promoted_false(tmp_path, eval_dir, merged_dir):
    """Declined: merged_model is the candidate's merged path; merged_model_promoted is false."""
    import importlib
    import model_promoter as mp
    importlib.reload(mp)

    # Patch _EVAL_DIR to our temp location
    with patch.object(mp, '_EVAL_DIR', eval_dir):
        mp._write_promotion_record(
            promoted=False,
            candidate_path='/fake/adapter',
            current_path='/fake/current',
            candidate_result=_fake_result(pass_rate=0.70),
            current_result=_fake_result(pass_rate=0.80),
            merged_model=str(merged_dir),
            merged_model_promoted=False,
        )

    record = _latest_promotion(eval_dir)
    assert record['promoted'] is False
    assert record['merged_model'] == str(merged_dir.resolve())
    assert record['merged_model_promoted'] is False


def test_promoted_run_writes_merged_model_and_promoted_true(tmp_path, eval_dir, merged_dir):
    """Promoted: merged_model is the merged path; merged_model_promoted is true."""
    import importlib
    import model_promoter as mp
    importlib.reload(mp)

    with patch.object(mp, '_EVAL_DIR', eval_dir):
        mp._write_promotion_record(
            promoted=True,
            candidate_path='/fake/adapter',
            current_path='/fake/current',
            candidate_result=_fake_result(pass_rate=0.85),
            current_result=_fake_result(pass_rate=0.80),
            merged_model=str(merged_dir),
            merged_model_promoted=True,
        )

    record = _latest_promotion(eval_dir)
    assert record['promoted'] is True
    assert record['merged_model'] == str(merged_dir.resolve())
    assert record['merged_model_promoted'] is True


def test_no_merge_run_writes_null_and_does_not_crash(tmp_path, eval_dir):
    """No merge: merged_model is null; merged_model_promoted is false; no crash."""
    import importlib
    import model_promoter as mp
    importlib.reload(mp)

    with patch.object(mp, '_EVAL_DIR', eval_dir):
        mp._write_promotion_record(
            promoted=False,
            candidate_path='/fake/adapter',
            current_path='/fake/current',
            candidate_result=_fake_result(),
            current_result=_fake_result(),
            merged_model=None,
            merged_model_promoted=False,
        )

    record = _latest_promotion(eval_dir)
    assert record['merged_model'] is None
    assert record['merged_model_promoted'] is False


# ---------------------------------------------------------------------------
# promote() integration tests — verify merged_model threads through all paths
# ---------------------------------------------------------------------------

def _make_eval_result(pass_rate=0.80, grade='B'):
    return _fake_result(pass_rate=pass_rate, grade=grade)


def _write_fake_eval(eval_dir: Path, pass_rate: float = 0.80) -> None:
    """Write a fake eval_*.json that _run_eval would find."""
    import time
    time.sleep(0.01)  # ensure mtime is newer than before_ts
    result = _make_eval_result(pass_rate=pass_rate)
    p = eval_dir / f"eval_fake_{pass_rate:.0%}.json"
    p.write_text(json.dumps(result))


def _patch_promote(tmp_path, eval_dir, candidate_pass=0.85, current_pass=0.80, merged_dir=None):
    """Helper that calls promote() with all heavy side-effects mocked."""
    import importlib
    import model_promoter as mp
    importlib.reload(mp)

    candidate_dir = tmp_path / 'finetune' / 'adapter_candidate'
    candidate_dir.mkdir(parents=True)
    current_dir = tmp_path / 'finetune' / 'adapter_current'
    current_dir.mkdir(parents=True)

    candidate_result = _make_eval_result(pass_rate=candidate_pass)
    current_result   = _make_eval_result(pass_rate=current_pass)

    with (
        patch.object(mp, '_EVAL_DIR', eval_dir),
        patch.object(mp, '_current_production_adapter', return_value=str(current_dir)),
        patch.object(mp, '_run_eval', side_effect=[
            (candidate_result, False),   # candidate eval (with_golden=True), no veto
            (current_result, None),      # current eval
        ]),
        patch.object(mp, '_update_latest_symlink'),
        patch.object(mp, '_update_env'),
        patch.object(mp, '_update_merged_symlink'),
        patch.object(mp, '_restart_services'),
    ):
        result = mp.promote(str(candidate_dir), merged_path=str(merged_dir) if merged_dir else None)
        record = _latest_promotion(eval_dir)
        return result, record


def test_promote_declined_records_merged_path(tmp_path, eval_dir, merged_dir):
    """promote() with candidate losing: merged_model recorded, merged_model_promoted=false."""
    promoted, record = _patch_promote(
        tmp_path, eval_dir,
        candidate_pass=0.70, current_pass=0.80,
        merged_dir=merged_dir,
    )
    assert promoted is False
    assert record['merged_model'] == str(merged_dir.resolve())
    assert record['merged_model_promoted'] is False


def test_promote_accepted_records_merged_path_and_promoted_true(tmp_path, eval_dir, merged_dir):
    """promote() with candidate winning: merged_model recorded, merged_model_promoted=true."""
    promoted, record = _patch_promote(
        tmp_path, eval_dir,
        candidate_pass=0.85, current_pass=0.80,
        merged_dir=merged_dir,
    )
    assert promoted is True
    assert record['merged_model'] == str(merged_dir.resolve())
    assert record['merged_model_promoted'] is True


def test_promote_no_merge_null_and_false(tmp_path, eval_dir):
    """promote() without merged_path: merged_model=null, merged_model_promoted=false."""
    promoted, record = _patch_promote(
        tmp_path, eval_dir,
        candidate_pass=0.85, current_pass=0.80,
        merged_dir=None,
    )
    assert record['merged_model'] is None
    assert record['merged_model_promoted'] is False
