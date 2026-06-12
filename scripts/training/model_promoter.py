#!/usr/bin/env python3
"""
Model Promoter — evaluate a candidate LoRA adapter against the current
production adapter and promote it if it scores at least as well.

Flow:
  1. Eval candidate adapter  (new, just fine-tuned)
  2. Eval current production adapter  (finance_qwen_32b_lora_latest symlink)
  3. Compare pass_rate (primary) and directional_acc (secondary)
  4. Promote if candidate.pass_rate >= current.pass_rate − TOLERANCE
     (ties go to the candidate — always prefer fresher training data)
  5. On promote: update finance_qwen_32b_lora_latest symlink + .env LORA_ADAPTER_PATH
               (next trading session subprocess picks up the new adapter automatically)
  6. On reject:  keep current adapter, log reason

Usage:
    python3 scripts/training/model_promoter.py --candidate finetune/finance_qwen_32b_lora_20260417_010236
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _pathfix  # noqa: F401

_SCRIPTS_DIR  = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
_EVAL_DIR     = _PROJECT_ROOT / 'logs' / 'eval'
_LATEST        = _PROJECT_ROOT / 'finetune' / 'finance_qwen_32b_lora_latest'
_MERGED_LATEST = _PROJECT_ROOT / 'finetune' / 'finance_qwen_32b_merged_latest'
_ENV_FILE      = _PROJECT_ROOT / '.env'

# Candidate must score at least (current − TOLERANCE) to be promoted.
# 0.0 = must be equal or better; 0.05 = allow up to 5% regression.
_PROMOTION_TOLERANCE = 0.0

# Replay gate — off by default; enable with ENABLE_REPLAY_GATE=true
_ENABLE_REPLAY_GATE        = os.getenv('ENABLE_REPLAY_GATE', 'false').lower() == 'true'
_PROMOTER_SHADOW           = os.getenv('PROMOTER_SHADOW', 'false').lower() == 'true'
_PROMOTION_EXPECTANCY_TOL  = float(os.getenv('PROMOTION_EXPECTANCY_TOL', '0.0'))
_REPLAY_MAX_DD_SCALE       = float(os.getenv('REPLAY_MAX_DD_SCALE', '1.25'))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_eval(
    adapter_path: str,
    with_golden: bool = False,
) -> tuple[dict | None, bool | None]:
    """
    Run eval_model.py for *adapter_path* in a subprocess.

    Returns (smoke_result, golden_veto):
      smoke_result  — parsed eval summary dict, or None on failure
      golden_veto   — True (veto), False (pass), None (golden gate skipped/absent)
    """
    eval_script = _SCRIPTS_DIR / 'training' / 'eval_model.py'
    before_ts   = datetime.now(timezone.utc).timestamp()

    cmd = [sys.executable, str(eval_script), '--adapter', adapter_path]
    if with_golden:
        cmd.append('--golden')

    timeout = 1200 if with_golden else 900  # golden adds ~5 min of inference

    logging.info(f"🔬 Evaluating: {Path(adapter_path).name}")
    result = subprocess.run(cmd, timeout=timeout, stderr=subprocess.PIPE)

    golden_veto: bool | None = None

    if with_golden:
        if result.returncode == 2:
            # Golden gate veto — smoke test result was still saved
            golden_veto = True
            logging.error(
                "❌ Golden gate VETO for %s — stderr: %s",
                Path(adapter_path).name,
                result.stderr.decode(errors='replace')[-500:],
            )
        elif result.returncode in (0, 1):
            golden_veto = False
        else:
            logging.error(
                "Eval subprocess exited with unexpected code %d — stderr: %s",
                result.returncode,
                result.stderr.decode(errors='replace')[-500:],
            )
            return None, None
    elif result.returncode not in (0, 1):
        logging.error(
            "Eval subprocess exited with code %d — stderr: %s",
            result.returncode,
            result.stderr.decode(errors='replace')[-500:],
        )
        return None, None

    # Find the eval JSON written after we started
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)
    written = sorted(
        [f for f in _EVAL_DIR.glob('eval_*.json') if f.stat().st_mtime >= before_ts],
        key=lambda f: f.stat().st_mtime,
    )
    if not written:
        logging.error("No eval result file found after run")
        return None, golden_veto

    return json.loads(written[-1].read_text()), golden_veto


def _run_replay_eval(adapter_path: str) -> dict | None:
    """
    Run replay_eval.py for adapter_path in a subprocess.
    Returns the parsed result dict from the saved JSON, or None on failure.
    """
    replay_script = _SCRIPTS_DIR / 'training' / 'replay_eval.py'
    if not replay_script.exists():
        logging.warning("replay_eval.py not found — skipping replay gate")
        return None

    log_path = _PROJECT_ROOT / 'logs' / 'decision_log.jsonl'
    if not log_path.exists():
        logging.warning("Decision log not found at %s — skipping replay gate", log_path)
        return None

    before_ts = datetime.now(timezone.utc).timestamp()
    # Force INFERENCE_BACKEND='direct' so the subprocess cold-loads the LoRA adapter
    # from LORA_ADAPTER_PATH.  Ollama/vLLM ignore LORA_ADAPTER_PATH and would test
    # both candidate and incumbent against the same already-loaded model.
    env = dict(os.environ, LORA_ADAPTER_PATH=adapter_path, INFERENCE_BACKEND='direct')

    logging.info("🔁 Replay eval: %s (direct backend)", Path(adapter_path).name)
    result = subprocess.run(
        [sys.executable, str(replay_script), '--log', str(log_path)],
        timeout=2400,  # 40 min: cold 32B load (~8 min) + 200 prompt inference (~25 min)
        env=env,
    )
    if result.returncode not in (0, 1):
        logging.error("Replay eval exited with code %d", result.returncode)
        return None

    _EVAL_DIR.mkdir(parents=True, exist_ok=True)
    written = sorted(
        [f for f in _EVAL_DIR.glob('replay_eval_*.json') if f.stat().st_mtime >= before_ts],
        key=lambda f: f.stat().st_mtime,
    )
    if not written:
        logging.error("No replay eval result file found")
        return None

    return json.loads(written[-1].read_text())


def _current_production_adapter() -> str | None:
    """Return the resolved path of the current production adapter, or None."""
    if _LATEST.is_symlink():
        return str(_LATEST.resolve())
    if _LATEST.is_dir():
        return str(_LATEST)
    return None


def _update_latest_symlink(adapter_path: str) -> None:
    if _LATEST.is_symlink() or _LATEST.exists():
        _LATEST.unlink()
    _LATEST.symlink_to(adapter_path)
    logging.info(f"🔗 Symlink updated: {_LATEST.name} → {Path(adapter_path).name}")


def _update_env(adapter_path: str) -> None:
    """Write LORA_ADAPTER_PATH in .env (relative to project root)."""
    try:
        rel = str(Path(adapter_path).relative_to(_PROJECT_ROOT))
    except ValueError:
        rel = adapter_path  # already relative or outside project root

    if _ENV_FILE.exists():
        text = _ENV_FILE.read_text()
        updated = re.sub(
            r'^(LORA_ADAPTER_PATH\s*=\s*).*$',
            rf'\g<1>{rel}',
            text,
            flags=re.MULTILINE,
        )
        if updated == text:
            # Key not present — append it
            updated = text.rstrip('\n') + f'\nLORA_ADAPTER_PATH={rel}\n'
    else:
        updated = f'LORA_ADAPTER_PATH={rel}\n'

    _ENV_FILE.write_text(updated)
    logging.info(f"📝 .env updated: LORA_ADAPTER_PATH={rel}")


def _update_merged_symlink(merged_path: str) -> None:
    """Update finance_qwen_32b_merged_latest to point at the newly merged BF16 model."""
    if not merged_path or not Path(merged_path).exists():
        logging.warning("Merged model not found at %s — skipping symlink update", merged_path)
        return
    target = Path(merged_path).resolve()
    tmp = _MERGED_LATEST.parent / f'.tmp_merged_latest_{os.getpid()}'
    try:
        tmp.unlink(missing_ok=True)
        os.symlink(target, tmp)
        os.replace(tmp, _MERGED_LATEST)   # atomic rename — no window where symlink is absent
        logging.info("🔗 Symlink updated: %s → %s", _MERGED_LATEST.name, target.name)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _restart_services() -> None:
    # Symlinks and .env are updated above. The inference server restart is handled
    # by the calling daemon (trading_daemon / options_daemon) via
    # inference_client.start_after_finetuning() AFTER this subprocess exits — that
    # guarantees all eval GPU memory is fully released before vLLM begins loading.
    logging.info("ℹ️  Symlinks and .env updated — daemon will restart inference server")


# ---------------------------------------------------------------------------
# Promotion record helper
# ---------------------------------------------------------------------------

def _write_promotion_record(
    *,
    promoted: bool,
    candidate_path: str,
    current_path: str,
    candidate_result: dict,
    current_result: dict,
    replay_candidate: dict | None = None,
    replay_current: dict | None = None,
    replay_verdict: bool | None = None,
    merged_model: str | None = None,
) -> None:
    record: dict = {
        'decided_at':    datetime.now().isoformat(),
        'promoted':      promoted,
        'candidate':     candidate_path,
        'current':       current_path,
        'merged_model':  str(Path(merged_model).resolve()) if merged_model else None,
        'candidate_scores': {k: candidate_result.get(k)
                             for k in ('pass_rate', 'directional_acc', 'format_score', 'grade')},
        'current_scores':   {k: current_result.get(k)
                             for k in ('pass_rate', 'directional_acc', 'format_score', 'grade')},
    }
    if replay_candidate is not None:
        record['replay_candidate'] = {k: replay_candidate.get(k)
                                      for k in ('expectancy', 'max_dd', 'trade_count', 'prompt_set_hash')}
        record['replay_current']   = {k: replay_current.get(k)
                                      for k in ('expectancy', 'max_dd', 'trade_count', 'prompt_set_hash')}
        record['replay_verdict']   = replay_verdict

    record_path = _EVAL_DIR / f"promotion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2))
    logging.info(f"📋 Promotion record → {record_path.name}")


# ---------------------------------------------------------------------------
# Core promotion logic
# ---------------------------------------------------------------------------

def promote(candidate_path: str, merged_path: str = None) -> bool:
    """
    Compare candidate adapter to the current production adapter.
    Returns True if the candidate was promoted, False if rejected.

    merged_path: optional path to the BF16-merged model directory produced by
                 fine_tune_llm.py. When provided and the candidate is promoted,
                 the finance_qwen_32b_merged_latest symlink is updated and the
                 vLLM inference server is restarted to load the new weights.
    """
    candidate_path = str(Path(candidate_path).resolve())
    current_path   = _current_production_adapter()

    # No existing production adapter — auto-promote
    if current_path is None:
        logging.info("No existing production adapter found — auto-promoting candidate")
        _update_latest_symlink(candidate_path)
        _update_env(candidate_path)
        if merged_path:
            _update_merged_symlink(merged_path)
        _restart_services()
        return True

    if candidate_path == current_path:
        logging.info("Candidate is already the production adapter — nothing to do")
        return False

    logging.info(f"\n{'='*60}")
    logging.info(f"🏆 Model Promotion Evaluation")
    logging.info(f"   Candidate : {Path(candidate_path).name}")
    logging.info(f"   Current   : {Path(current_path).name}")
    logging.info(f"{'='*60}\n")

    # ── Stage 1: Smoke test (+ golden gate for candidate) ───────────────────
    # Golden eval runs inside the candidate subprocess to share the model load.
    candidate_result, golden_veto = _run_eval(candidate_path, with_golden=True)
    current_result,   _           = _run_eval(current_path)

    if candidate_result is None or current_result is None:
        logging.error("❌ Eval failed — keeping current adapter as a safety measure")
        return False

    if golden_veto is True:
        logging.error("❌ REJECTED (golden gate veto) — pass_rate dropped >15 pts vs baseline")
        _write_promotion_record(
            promoted=False, candidate_path=candidate_path, current_path=current_path,
            candidate_result=candidate_result, current_result=current_result,
        )
        return False

    c_pass  = candidate_result['pass_rate']
    p_pass  = current_result['pass_rate']
    c_acc   = candidate_result['directional_acc']
    p_acc   = current_result['directional_acc']
    c_grade = candidate_result['grade']
    p_grade = current_result['grade']
    c_fmt   = candidate_result.get('format_score', 1.0)

    logging.info(f"\n{'Metric':<22} {'Candidate':>10} {'Current':>10} {'Delta':>8}")
    logging.info(f"{'-'*52}")
    logging.info(f"{'pass_rate':<22} {c_pass:>10.1%} {p_pass:>10.1%} {c_pass-p_pass:>+8.1%}")
    logging.info(f"{'directional_acc':<22} {c_acc:>10.1%} {p_acc:>10.1%} {c_acc-p_acc:>+8.1%}")
    logging.info(f"{'format_score':<22} {c_fmt:>10.1%}")
    logging.info(f"{'grade':<22} {c_grade:>10} {p_grade:>10}")

    # Hard fail: grade F means total output breakdown; format < 0.9 means the model
    # stopped producing Decision/Confidence/Reasoning reliably.
    smoke_passed = c_grade != 'F' and c_fmt >= 0.90
    if not smoke_passed:
        logging.info(
            f"\n⏪ REJECTED (smoke test) — grade {c_grade}, format {c_fmt:.0%} < 90%"
        )
        _write_promotion_record(
            promoted=False, candidate_path=candidate_path, current_path=current_path,
            candidate_result=candidate_result, current_result=current_result,
        )
        return False

    # Legacy pass/rate gate (still applies unless overridden by replay gate below)
    smoke_promoted = c_pass >= p_pass - _PROMOTION_TOLERANCE

    # ── Stage 2: Replay gate (optional, gated behind ENABLE_REPLAY_GATE=true) ──
    replay_candidate: dict | None = None
    replay_current:   dict | None = None
    replay_verdict:   bool | None = None

    # Use a local flag so we can skip replay without touching the module-level constant
    # (Python would treat _ENABLE_REPLAY_GATE as local if we assigned it, causing
    # UnboundLocalError on the outer `if _ENABLE_REPLAY_GATE:` reference).
    _run_replay = _ENABLE_REPLAY_GATE

    if _run_replay:
        # The replay eval uses INFERENCE_BACKEND='direct' (cold 32B GPU load).
        # Running that alongside a live vLLM server would OOM on 128 GB unified memory.
        # The daemon's stop_for_finetuning() should have stopped vLLM before launching
        # finetune_model.py; if vLLM is somehow still active, skip replay and fall back
        # to the smoke verdict — the promoter must never be granted stop authority.
        _backend = os.getenv('INFERENCE_BACKEND', 'direct').lower()
        if _backend == 'ollama':
            try:
                from inference_client import _stop_ollama
                _stop_ollama()
                logging.info("🛑 Ollama unloaded — freeing VRAM for replay eval")
            except Exception as _stop_err:
                logging.warning("Could not unload Ollama before replay eval: %s", _stop_err)
        elif _backend == 'vllm':
            _vllm_active = subprocess.run(
                ['sudo', 'systemctl', 'is-active', 'ai-inference-server.service'],
                capture_output=True, text=True,
            ).stdout.strip() == 'active'
            if _vllm_active:
                logging.error(
                    "❌ REPLAY GATE SKIPPED: vLLM is still active (daemon did not stop it). "
                    "Running direct-GPU replay alongside live vLLM would OOM. "
                    "Falling back to smoke-test verdict."
                )
                _run_replay = False

    if _run_replay:
        logging.info("\n🔁 Running replay gate...")
        replay_candidate = _run_replay_eval(candidate_path)
        replay_current   = _run_replay_eval(current_path)

        # Verify both runs evaluated the same prompt set.  A mismatch means one eval
        # ran on a different selection of held-out prompts (e.g. due to a race that
        # modified the decision log between the two runs).  The comparison would be
        # unfair, so we discard the replay verdict and fall back to the smoke test.
        if (replay_candidate is not None and replay_current is not None
                and replay_candidate.get('prompt_set_hash') != replay_current.get('prompt_set_hash')):
            logging.error(
                "⚠️  Replay eval prompt_set_hash MISMATCH — candidate=%s incumbent=%s. "
                "The two evals ran on different prompt sets; discarding replay verdict "
                "and falling back to smoke-test gate.",
                replay_candidate.get('prompt_set_hash'),
                replay_current.get('prompt_set_hash'),
            )
            replay_candidate = replay_current = None

        if replay_candidate is not None and replay_current is not None:
            c_exp  = replay_candidate['expectancy']
            p_exp  = replay_current['expectancy']
            c_mdd  = replay_candidate['max_dd']
            p_mdd  = replay_current['max_dd']
            c_cnt  = replay_candidate['trade_count']
            p_cnt  = replay_current['trade_count']

            logging.info(
                f"\n{'Replay Metric':<22} {'Candidate':>10} {'Current':>10} {'Delta':>8}"
            )
            logging.info(f"{'-'*52}")
            logging.info(f"{'expectancy':<22} {c_exp:>+10.2%} {p_exp:>+10.2%} {c_exp-p_exp:>+8.2%}")
            logging.info(f"{'max_dd':<22} {c_mdd:>10.1%} {p_mdd:>10.1%}")
            logging.info(f"{'trade_count':<22} {c_cnt:>10} {p_cnt:>10}")

            if (c_cnt < 5 and p_cnt >= 5) or (p_cnt < 5 and c_cnt >= 5):
                replay_verdict = False
                logging.info("   Replay: insufficient trades for fair comparison")
            else:
                expectancy_ok = c_exp >= p_exp - _PROMOTION_EXPECTANCY_TOL
                max_dd_ok     = c_mdd <= p_mdd * _REPLAY_MAX_DD_SCALE
                replay_verdict = expectancy_ok and max_dd_ok

            if _PROMOTER_SHADOW:
                logging.info(
                    "   [SHADOW] Replay verdict: %s (smoke-test decision applies)",
                    'PROMOTE' if replay_verdict else 'REJECT',
                )
            else:
                smoke_promoted = replay_verdict
                logging.info(
                    "   Replay verdict: %s", 'PROMOTE' if smoke_promoted else 'REJECT'
                )
        else:
            logging.warning("Replay eval unavailable — falling back to smoke-test decision")

    promoted = smoke_promoted

    if promoted:
        logging.info(
            f"\n✅ PROMOTED — pass_rate {c_pass:.1%} ≥ {p_pass:.1%} "
            f"(tolerance {_PROMOTION_TOLERANCE:.1%})"
        )
        _update_latest_symlink(candidate_path)
        _update_env(candidate_path)
        if merged_path:
            _update_merged_symlink(merged_path)
        _restart_services()
    else:
        logging.info(
            f"\n⏪ REJECTED — pass_rate {c_pass:.1%} < {p_pass:.1%} "
            f"(tolerance {_PROMOTION_TOLERANCE:.1%})"
        )
        logging.info(f"   Keeping: {Path(current_path).name}")

    _write_promotion_record(
        promoted=promoted, candidate_path=candidate_path, current_path=current_path,
        candidate_result=candidate_result, current_result=current_result,
        replay_candidate=replay_candidate, replay_current=replay_current,
        replay_verdict=replay_verdict,
        merged_model=merged_path if promoted else None,
    )
    return promoted


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------

def rollback(n: int = 2) -> bool:
    """
    Restore the production symlink to the previously promoted adapter.

    Reads promotion records in reverse-chronological order, finds the most
    recently promoted adapter that is NOT the current production one, and
    re-points the finance_qwen_32b_lora_latest symlink + .env.

    n controls how many promotion records to scan (default 2 means look back
    through up to 4 records to find 2 distinct promoted paths).
    """
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)
    record_files = sorted(
        _EVAL_DIR.glob('promotion_*.json'),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    # Collect promoted adapter paths in reverse-chronological order
    promoted_adapters: list[str] = []
    for f in record_files:
        try:
            data = json.loads(f.read_text())
            if data.get('promoted') and data.get('candidate'):
                promoted_adapters.append(data['candidate'])
        except Exception:
            continue
        if len(promoted_adapters) >= n * 2:
            break

    current = _current_production_adapter()
    logging.info(f"Current production adapter: {Path(current).name if current else 'none'}")

    # Find the most recent promoted adapter that differs from current
    target: str | None = None
    for path in promoted_adapters:
        if path != current:
            target = path
            break

    if target is None:
        logging.error("❌ No previous promoted adapter found (need at least 2 successful promotions)")
        return False

    if not Path(target).exists():
        logging.error(f"❌ Previous adapter directory no longer exists: {target}")
        return False

    logging.info(f"⏪ Rolling back to: {Path(target).name}")
    _update_latest_symlink(target)
    _update_env(target)
    _restart_services()

    record = {
        'decided_at': datetime.now().isoformat(),
        'action':     'rollback',
        'promoted':   True,
        'candidate':  target,
        'current':    current,
    }
    record_path = _EVAL_DIR / f"rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    record_path.write_text(json.dumps(record, indent=2))
    logging.info(f"📋 Rollback record → {record_path.name}")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Evaluate and promote a fine-tuned LoRA adapter')
    parser.add_argument('--candidate', type=str, default=None,
                        help='Path to the candidate (newly fine-tuned) adapter directory')
    parser.add_argument('--merged-model', type=str, default=None,
                        help='Path to the BF16-merged model directory for vLLM serving (optional)')
    parser.add_argument('--rollback', type=int, nargs='?', const=2, default=None,
                        metavar='N',
                        help='Roll back to the previous promoted adapter; '
                             'scans the last N*2 promotion records (default N=2)')
    args = parser.parse_args()

    if args.rollback is not None:
        rolled_back = rollback(args.rollback)
        sys.exit(0 if rolled_back else 2)

    if args.candidate is None:
        parser.error('--candidate is required unless --rollback is used')

    promoted = promote(args.candidate, merged_path=args.merged_model)
    # Exit 0 = promoted, 2 = rejected (not an error — caller can check)
    sys.exit(0 if promoted else 2)


if __name__ == '__main__':
    main()
