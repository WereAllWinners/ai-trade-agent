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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_eval(adapter_path: str) -> dict | None:
    """
    Run eval_model.py for adapter_path in a subprocess.
    Returns the parsed summary dict from the saved eval JSON, or None on failure.
    """
    eval_script = _SCRIPTS_DIR / 'training' / 'eval_model.py'
    before_ts   = datetime.now(timezone.utc).timestamp()

    logging.info(f"🔬 Evaluating: {Path(adapter_path).name}")
    result = subprocess.run(
        [sys.executable, str(eval_script), '--adapter', adapter_path],
        timeout=900,  # 15 min per eval
    )

    # returncode 1 = grade F — still a valid run, result was saved
    if result.returncode not in (0, 1):
        logging.error(f"Eval subprocess exited with code {result.returncode}")
        return None

    # Find the eval JSON written after we started
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)
    written = sorted(
        [f for f in _EVAL_DIR.glob('eval_*.json') if f.stat().st_mtime >= before_ts],
        key=lambda f: f.stat().st_mtime,
    )
    if not written:
        logging.error("No eval result file found after run")
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
        logging.warning(f"Merged model not found at {merged_path} — skipping symlink update")
        return
    if _MERGED_LATEST.is_symlink() or _MERGED_LATEST.exists():
        _MERGED_LATEST.unlink()
    _MERGED_LATEST.symlink_to(merged_path)
    logging.info(f"🔗 Symlink updated: {_MERGED_LATEST.name} → {Path(merged_path).name}")


def _restart_services() -> None:
    # Symlinks and .env are updated above. The inference server restart is handled
    # by the calling daemon (trading_daemon / options_daemon) via
    # inference_client.start_after_finetuning() AFTER this subprocess exits — that
    # guarantees all eval GPU memory is fully released before vLLM begins loading.
    logging.info("ℹ️  Symlinks and .env updated — daemon will restart inference server")


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

    candidate_result = _run_eval(candidate_path)
    current_result   = _run_eval(current_path)

    if candidate_result is None or current_result is None:
        logging.error("❌ Eval failed — keeping current adapter as a safety measure")
        return False

    c_pass  = candidate_result['pass_rate']
    p_pass  = current_result['pass_rate']
    c_acc   = candidate_result['directional_acc']
    p_acc   = current_result['directional_acc']
    c_grade = candidate_result['grade']
    p_grade = current_result['grade']

    logging.info(f"\n{'Metric':<22} {'Candidate':>10} {'Current':>10} {'Delta':>8}")
    logging.info(f"{'-'*52}")
    logging.info(f"{'pass_rate':<22} {c_pass:>10.1%} {p_pass:>10.1%} {c_pass-p_pass:>+8.1%}")
    logging.info(f"{'directional_acc':<22} {c_acc:>10.1%} {p_acc:>10.1%} {c_acc-p_acc:>+8.1%}")
    logging.info(f"{'grade':<22} {c_grade:>10} {p_grade:>10}")

    promoted = c_pass >= p_pass - _PROMOTION_TOLERANCE

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

    # Write promotion record to logs
    record = {
        'decided_at':    datetime.now().isoformat(),
        'promoted':      promoted,
        'candidate':     candidate_path,
        'current':       current_path,
        'candidate_scores': {k: candidate_result.get(k) for k in ('pass_rate', 'directional_acc', 'grade')},
        'current_scores':   {k: current_result.get(k)   for k in ('pass_rate', 'directional_acc', 'grade')},
    }
    record_path = _EVAL_DIR / f"promotion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2))
    logging.info(f"📋 Promotion record → {record_path.name}")

    return promoted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Evaluate and promote a fine-tuned LoRA adapter')
    parser.add_argument('--candidate', type=str, required=True,
                        help='Path to the candidate (newly fine-tuned) adapter directory')
    parser.add_argument('--merged-model', type=str, default=None,
                        help='Path to the BF16-merged model directory for vLLM serving (optional)')
    args = parser.parse_args()

    promoted = promote(args.candidate, merged_path=args.merged_model)
    # Exit 0 = promoted, 2 = rejected (not an error — caller can check)
    sys.exit(0 if promoted else 2)


if __name__ == '__main__':
    main()
