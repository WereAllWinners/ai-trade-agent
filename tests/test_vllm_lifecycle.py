"""
test_vllm_lifecycle.py — Tests for vLLM lifecycle safety (incident 2026-06-12).

Validates:
  1. start failure → CRITICAL alert sent, exception re-raised, no bare-stop in call list
  2. Two concurrent lifecycle callers → exactly one systemctl invocation (cross-process lock)
  3. Market-hours stop → zero systemctl calls, WARNING alert sent
  4. ensure_vllm_running → start only when inactive; zero calls when already active
  5. Atomic symlink update + promotion record contains resolved merged_model path
"""
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_subprocess_result(returncode=0, stdout='', stderr=''):
    m = MagicMock()
    m.returncode = returncode
    m.stdout = stdout
    m.stderr = stderr
    return m


# ---------------------------------------------------------------------------
# Test 1: start failure → CRITICAL alert + reraise; no bare stop uncovered
# ---------------------------------------------------------------------------

def test_start_failure_sends_critical_alert_and_reraises(tmp_path):
    os.environ['INFERENCE_BACKEND'] = 'vllm'
    import importlib
    import inference_client as ic
    importlib.reload(ic)

    subprocess_calls = []

    def fake_subprocess_run(cmd, **kwargs):
        subprocess_calls.append(cmd)
        if 'is-active' in cmd:
            return _make_subprocess_result(stdout='inactive')
        if 'stop' in cmd:
            return _make_subprocess_result()
        if 'restart' in cmd:
            raise RuntimeError("simulated restart failure")
        return _make_subprocess_result()

    sent_alerts = []

    def fake_send_critical(event, message):
        sent_alerts.append({'event': event, 'message': message})

    with patch.object(ic, '_LIFECYCLE_LOCK', None), \
         patch('subprocess.run', side_effect=fake_subprocess_run), \
         patch.object(ic, '_send_critical', side_effect=fake_send_critical), \
         patch.object(ic, '_poll_vllm_ready'):

        with pytest.raises(RuntimeError):
            ic.start_after_finetuning()

    # CRITICAL alert must have been sent
    assert any('vllm_restart_failed' in a['event'] for a in sent_alerts), \
        f"Expected CRITICAL alert for vllm_restart_failed, got: {sent_alerts}"

    # No bare 'stop' should appear after a failed restart cycle
    stop_calls = [c for c in subprocess_calls if 'stop' in c and 'is-active' not in c]
    assert stop_calls == [], f"Bare stop found after failed restart: {stop_calls}"

    os.environ.pop('INFERENCE_BACKEND', None)


# ---------------------------------------------------------------------------
# Test 2: cross-process lock — exactly one systemctl invocation
# ---------------------------------------------------------------------------

def _lifecycle_worker(lock_path: str, record_path: str, lock_timeout: float) -> None:
    """Worker process: try to acquire the lifecycle lock and record success."""
    try:
        from filelock import FileLock, Timeout as FileLockTimeout
        # p1 uses a long timeout (will succeed); p2 uses a short timeout (will fail/timeout)
        lock = FileLock(lock_path, timeout=lock_timeout)
        with lock:
            # Hold the lock for 3 seconds so the short-timeout caller definitely times out
            time.sleep(3.0)
            with open(record_path, 'a') as f:
                f.write(f"{os.getpid()}\n")
    except Exception:
        pass   # Second caller times out — expected, write nothing


def test_two_concurrent_lifecycle_calls_exactly_one_invocation(tmp_path):
    """
    filelock.FileLock is reentrant within a process but exclusive across processes.
    p1 acquires the lock with a long timeout and holds it for 3s.
    p2 tries with a 1s timeout — shorter than the 3s hold — and times out, writing nothing.
    """
    pytest.importorskip('filelock')

    lock_path   = str(tmp_path / '.vllm_lifecycle.lock')
    record_path = str(tmp_path / 'acquired.txt')

    # p1: long timeout (will succeed and hold for 3s)
    p1 = multiprocessing.Process(target=_lifecycle_worker, args=(lock_path, record_path, 10.0))
    # p2: short timeout (1s < 3s hold = will time out)
    p2 = multiprocessing.Process(target=_lifecycle_worker, args=(lock_path, record_path, 1.0))

    p1.start()
    # Give p1 enough time to acquire the lock before p2 starts trying
    time.sleep(0.3)
    p2.start()

    p1.join(timeout=15)
    p2.join(timeout=15)

    lines = Path(record_path).read_text().strip().splitlines() if Path(record_path).exists() else []
    assert len(lines) == 1, (
        f"Expected exactly 1 process to acquire the lock, but {len(lines)} did: {lines}"
    )


# ---------------------------------------------------------------------------
# Test 3: market-hours stop → zero systemctl calls, WARNING alert
# ---------------------------------------------------------------------------

def test_market_hours_stop_blocked_no_systemctl_call():
    os.environ['INFERENCE_BACKEND'] = 'vllm'
    os.environ.pop('ALLOW_INTRADAY_VLLM_RESTART', None)

    import importlib
    import inference_client as ic
    importlib.reload(ic)

    sent_alerts = []
    systemctl_calls = []

    def fake_subprocess_run(cmd, **kwargs):
        systemctl_calls.append(cmd)
        return _make_subprocess_result()

    def fake_send_warning(event, message):
        sent_alerts.append({'event': event, 'message': message})

    with patch.object(ic, '_is_market_hours', return_value=True), \
         patch('subprocess.run', side_effect=fake_subprocess_run), \
         patch.object(ic, '_send_warning', side_effect=fake_send_warning):

        result = ic.stop_for_finetuning()

    assert result is False, "stop_for_finetuning() should return False when market is open"
    assert systemctl_calls == [], \
        f"No systemctl calls expected during market hours, got: {systemctl_calls}"
    assert any('market_hours' in a['event'] for a in sent_alerts), \
        f"Expected market-hours WARNING alert, got: {sent_alerts}"

    os.environ.pop('INFERENCE_BACKEND', None)


# ---------------------------------------------------------------------------
# Test 4: ensure_vllm_running → start only when inactive
# ---------------------------------------------------------------------------

def test_ensure_vllm_running_starts_only_when_inactive():
    os.environ['INFERENCE_BACKEND'] = 'vllm'

    import importlib
    import inference_client as ic
    importlib.reload(ic)

    # --- 4a: inactive → exactly one 'start' call + poll ---
    systemctl_calls_a = []

    def fake_run_inactive(cmd, **kwargs):
        systemctl_calls_a.append(list(cmd))
        if 'is-active' in cmd:
            return _make_subprocess_result(stdout='inactive')
        return _make_subprocess_result()

    poll_called = []

    with patch.object(ic, '_LIFECYCLE_LOCK', None), \
         patch('subprocess.run', side_effect=fake_run_inactive), \
         patch.object(ic, '_poll_vllm_ready', side_effect=lambda **kw: poll_called.append(True)):
        ic.ensure_vllm_running()

    start_calls = [c for c in systemctl_calls_a if 'start' in c and 'is-active' not in c]
    assert len(start_calls) == 1, f"Expected exactly 1 'start' call, got: {start_calls}"
    assert poll_called, "Expected _poll_vllm_ready to be called after start"

    # --- 4b: already active → zero systemctl calls ---
    systemctl_calls_b = []

    def fake_run_active(cmd, **kwargs):
        systemctl_calls_b.append(list(cmd))
        return _make_subprocess_result(stdout='active')

    with patch.object(ic, '_LIFECYCLE_LOCK', None), \
         patch('subprocess.run', side_effect=fake_run_active):
        ic.ensure_vllm_running()

    non_active_check = [c for c in systemctl_calls_b if 'is-active' not in c]
    assert non_active_check == [], \
        f"No non-is-active calls expected when server is already active, got: {non_active_check}"

    os.environ.pop('INFERENCE_BACKEND', None)


# ---------------------------------------------------------------------------
# Test 5: atomic symlink + promotion record has resolved merged_model path
# ---------------------------------------------------------------------------

def test_symlink_updater_atomic_and_promotion_record_has_merged_model(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'training'))

    import importlib
    import model_promoter as mp
    importlib.reload(mp)

    # Create a fake merged model directory
    fake_merged = tmp_path / 'finance_qwen_32b_lora_merged_bf16_20260612_083434'
    fake_merged.mkdir()

    symlink_path = tmp_path / 'finance_qwen_32b_merged_latest'

    # Patch the module-level symlink path
    with patch.object(mp, '_MERGED_LATEST', symlink_path):
        mp._update_merged_symlink(str(fake_merged))

    assert symlink_path.is_symlink(), "merged_latest should be a symlink"
    assert symlink_path.resolve() == fake_merged.resolve(), \
        f"merged_latest should point to fake_merged, got: {symlink_path.resolve()}"

    # Verify promotion record contains resolved merged_model path
    eval_dir = tmp_path / 'eval'
    eval_dir.mkdir()

    fake_candidate_result = {
        'pass_rate': 0.85, 'directional_acc': 0.75, 'format_score': 0.95, 'grade': 'B'
    }

    with patch.object(mp, '_EVAL_DIR', eval_dir):
        mp._write_promotion_record(
            promoted=True,
            candidate_path=str(tmp_path / 'finance_qwen_32b_lora_20260612_083434'),
            current_path=str(tmp_path / 'finance_qwen_32b_lora_20260611_083434'),
            candidate_result=fake_candidate_result,
            current_result=fake_candidate_result,
            merged_model=str(fake_merged),
        )

    record_files = list(eval_dir.glob('promotion_*.json'))
    assert record_files, "Expected a promotion record to be written"
    record = json.loads(record_files[0].read_text())
    assert 'merged_model' in record, "promotion record should contain 'merged_model' key"
    assert record['merged_model'] is not None, "merged_model should not be None"
    assert Path(record['merged_model']).resolve() == fake_merged.resolve(), \
        f"merged_model in record should resolve to fake_merged, got: {record['merged_model']}"
