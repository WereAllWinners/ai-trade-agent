"""
test_morning_model_check.py — Tests for check_merged_symlink_alignment()

Six scenarios (Item 2):
  1. promoted + symlink matches recorded path                  → PASS
  2. promoted + symlink mismatch                              → HARD FAIL + CRITICAL alert
  3. declined + symlink == declined model                     → HARD FAIL + CRITICAL alert
  4. declined + symlink != declined model (previous promoted) → PASS
  5. merged_model=null in record                              → WARN-ONLY
  6. vLLM started before symlink repointed                    → WARNING only (alignment still PASS)
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'tools'))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PASS_STATUS = "✅ PASS"
FAIL_STATUS = "❌ FAIL"
WARN_STATUS  = "⚠️  WARN"


@pytest.fixture()
def project_root(tmp_path):
    (tmp_path / 'finetune').mkdir()
    (tmp_path / 'logs' / 'eval').mkdir(parents=True)
    return tmp_path


def _make_merged_dir(project_root: Path, name: str) -> Path:
    d = project_root / 'finetune' / name
    d.mkdir(exist_ok=True)
    return d


def _write_promo(project_root: Path, merged_model: str | None,
                 merged_model_promoted: bool | None) -> None:
    record: dict = {
        'decided_at': datetime.now().isoformat(),
        'promoted': merged_model_promoted,
        'candidate': '/fake/candidate',
        'current': '/fake/current',
        'merged_model': merged_model,
    }
    if merged_model_promoted is not None:
        record['merged_model_promoted'] = merged_model_promoted
    # else: omit the key entirely (old-format record simulation)
    p = project_root / 'logs' / 'eval' / 'promotion_test.json'
    p.write_text(json.dumps(record))


def _symlink_to(project_root: Path, target: Path) -> Path:
    link = project_root / 'finetune' / 'finance_qwen_32b_merged_latest'
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target.resolve())
    return link


def _run_check(project_root: Path) -> 'CheckResult':
    import importlib
    import morning_model_check as mc
    importlib.reload(mc)

    results = mc.CheckResult()
    with (
        patch.object(mc, '_PROJECT_ROOT', project_root),
        patch('alerts.send_alert'),  # prevent real email/SMTP
    ):
        mc.check_merged_symlink_alignment(results)
    return results


def _statuses(results) -> dict[str, str]:
    return {name: status for name, status, _ in results.items}


# ---------------------------------------------------------------------------
# Scenario 1: promoted + symlink matches → PASS
# ---------------------------------------------------------------------------

def test_promoted_symlink_matches_is_pass(project_root):
    merged = _make_merged_dir(project_root, 'finance_qwen_32b_lora_merged_bf16_20260615_010237')
    _symlink_to(project_root, merged)
    _write_promo(project_root, str(merged.resolve()), merged_model_promoted=True)

    results = _run_check(project_root)
    s = _statuses(results)

    assert 'merged_symlink_alignment' in s
    assert s['merged_symlink_alignment'] == PASS_STATUS
    assert results.all_passed


# ---------------------------------------------------------------------------
# Scenario 2: promoted + symlink mismatch → HARD FAIL + CRITICAL alert
# ---------------------------------------------------------------------------

def test_promoted_symlink_mismatch_is_hard_fail(project_root):
    promoted_merged = _make_merged_dir(project_root, 'merged_promoted')
    other_merged    = _make_merged_dir(project_root, 'merged_other')
    # Symlink points at other_merged, but record says promoted_merged was promoted
    _symlink_to(project_root, other_merged)
    _write_promo(project_root, str(promoted_merged.resolve()), merged_model_promoted=True)

    import importlib
    import morning_model_check as mc
    importlib.reload(mc)

    results = mc.CheckResult()
    alert_calls = []

    def capture_alert(level, event, msg, data=None):
        alert_calls.append((level, event))

    with (
        patch.object(mc, '_PROJECT_ROOT', project_root),
        patch('alerts.send_alert', side_effect=capture_alert),
    ):
        mc.check_merged_symlink_alignment(results)

    s = _statuses(results)
    assert s.get('merged_symlink_alignment') == FAIL_STATUS, \
        f"expected HARD FAIL, got {s}"
    assert not results.all_passed

    # CRITICAL alert must have been sent
    from alerts import AlertLevel
    assert any(lvl == AlertLevel.CRITICAL and 'promoted' in evt
               for lvl, evt in alert_calls), \
        f"Expected CRITICAL alert with 'promoted' in event name, got: {alert_calls}"


# ---------------------------------------------------------------------------
# Scenario 3: declined + symlink == declined model → HARD FAIL + CRITICAL alert
# ---------------------------------------------------------------------------

def test_declined_symlink_is_declined_model_is_hard_fail(project_root):
    declined_merged = _make_merged_dir(project_root, 'merged_declined')
    # Symlink points at the DECLINED model — catastrophic
    _symlink_to(project_root, declined_merged)
    _write_promo(project_root, str(declined_merged.resolve()), merged_model_promoted=False)

    import importlib
    import morning_model_check as mc
    importlib.reload(mc)

    results = mc.CheckResult()
    alert_calls = []

    def capture_alert(level, event, msg, data=None):
        alert_calls.append((level, event))

    with (
        patch.object(mc, '_PROJECT_ROOT', project_root),
        patch('alerts.send_alert', side_effect=capture_alert),
    ):
        mc.check_merged_symlink_alignment(results)

    s = _statuses(results)
    assert s.get('merged_symlink_alignment') == FAIL_STATUS, \
        f"expected HARD FAIL, got {s}"
    assert not results.all_passed

    from alerts import AlertLevel
    assert any(lvl == AlertLevel.CRITICAL and 'declined' in evt
               for lvl, evt in alert_calls), \
        f"Expected CRITICAL alert with 'declined' in event name, got: {alert_calls}"


# ---------------------------------------------------------------------------
# Scenario 4: declined + symlink != declined model → PASS
# ---------------------------------------------------------------------------

def test_declined_symlink_not_declined_model_is_pass(project_root):
    declined_merged = _make_merged_dir(project_root, 'merged_declined')
    previous_merged = _make_merged_dir(project_root, 'merged_previous_promoted')
    # Symlink correctly keeps pointing at the previous promoted model
    _symlink_to(project_root, previous_merged)
    _write_promo(project_root, str(declined_merged.resolve()), merged_model_promoted=False)

    results = _run_check(project_root)
    s = _statuses(results)

    assert s.get('merged_symlink_alignment') == PASS_STATUS, \
        f"expected PASS, got {s}"
    assert results.all_passed


# ---------------------------------------------------------------------------
# Scenario 5: merged_model=null → WARN-ONLY
# ---------------------------------------------------------------------------

def test_null_merged_model_is_warn_only(project_root):
    some_merged = _make_merged_dir(project_root, 'merged_some')
    _symlink_to(project_root, some_merged)
    _write_promo(project_root, merged_model=None, merged_model_promoted=None)

    results = _run_check(project_root)
    s = _statuses(results)

    # Should not be a FAIL — either PASS or WARN
    assert s.get('merged_symlink_alignment') != FAIL_STATUS, \
        f"null merged_model must not hard-fail, got {s}"
    # all_passed means no FAIL entries (WARNs are OK)
    assert results.all_passed


# ---------------------------------------------------------------------------
# Scenario 6: vLLM started before symlink repointed → WARNING only; alignment PASS
# ---------------------------------------------------------------------------

def test_vllm_stale_is_warning_only_not_fail(project_root):
    merged = _make_merged_dir(project_root, 'merged_current')
    link = _symlink_to(project_root, merged)
    _write_promo(project_root, str(merged.resolve()), merged_model_promoted=True)

    # Simulate: vLLM started 1 hour BEFORE symlink mtime
    import time
    symlink_mtime = os.lstat(str(link)).st_mtime
    service_start_ts = symlink_mtime - 3600  # 1 hour earlier
    service_start_dt = datetime.fromtimestamp(service_start_ts, tz=timezone.utc)
    ts_str = service_start_dt.strftime('Mon %Y-%m-%d %H:%M:%S UTC')

    import importlib
    import morning_model_check as mc
    importlib.reload(mc)

    fake_proc = MagicMock()
    fake_proc.stdout = f'ActiveEnterTimestamp={ts_str}\n'
    fake_proc.returncode = 0

    results = mc.CheckResult()
    with (
        patch.object(mc, '_PROJECT_ROOT', project_root),
        patch('alerts.send_alert'),
        patch('subprocess.run', return_value=fake_proc),
        patch.dict(os.environ, {'INFERENCE_BACKEND': 'vllm'}),
    ):
        mc.check_merged_symlink_alignment(results)

    s = _statuses(results)

    # Alignment itself is PASS (symlink matches promoted model)
    assert s.get('merged_symlink_alignment') == PASS_STATUS, \
        f"alignment should be PASS, got {s}"

    # Staleness check should be WARN, not FAIL
    assert 'vllm_model_freshness' in s, \
        f"staleness check not run; statuses: {s}"
    assert s['vllm_model_freshness'] == WARN_STATUS, \
        f"staleness must be WARN not FAIL, got {s['vllm_model_freshness']}"

    # Overall: no hard fails
    assert results.all_passed
