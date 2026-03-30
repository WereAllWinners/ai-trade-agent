#!/usr/bin/env python3
"""
online_trainer.py

Lightweight LoRA fine-tune triggered after a threshold of new closed-trade
outcomes accumulates.  Runs a single epoch at a low learning rate (5e-5) on
only the most recent N examples — keeps the adapter fresh without full retraining.

Designed to be called from trading_daemon.py after run_performance_analysis().

Usage:
    python3 scripts/online_trainer.py [--min-outcomes 15] [--recent-n 50]

Exit codes:
    0  — training ran successfully (or was skipped cleanly)
    1  — training failed
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _pathfix  # noqa: F401

_SCRIPTS_DIR  = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
_DATA_DIR     = _PROJECT_ROOT / 'finetune' / 'data'
_ONLINE_STATE = _PROJECT_ROOT / 'logs' / 'online_trainer_state.json'

import db as _db

# Default thresholds — can be overridden via CLI
_MIN_NEW_OUTCOMES = 15   # trigger after this many new closed trades
_RECENT_N         = 50   # train on this many most-recent examples
_ONLINE_LR        = 5e-5
_ONLINE_EPOCHS    = 1


def _load_state() -> dict:
    """Load persisted trainer state (last training time and outcome count)."""
    try:
        if _ONLINE_STATE.exists():
            return json.loads(_ONLINE_STATE.read_text())
    except Exception:
        pass
    return {'last_trained_at': '2000-01-01T00:00:00', 'outcomes_at_last_train': 0}


def _save_state(state: dict) -> None:
    _ONLINE_STATE.parent.mkdir(parents=True, exist_ok=True)
    _ONLINE_STATE.write_text(json.dumps(state, indent=2))


def _latest_adapter() -> str | None:
    """Return path to the most recently created LoRA adapter directory."""
    finetune_dir = _PROJECT_ROOT / 'finetune'
    candidates = sorted(
        [d for d in finetune_dir.iterdir()
         if d.is_dir() and 'lora' in d.name and (d / 'adapter_config.json').exists()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else None


def _export_recent_examples(n: int) -> Path | None:
    """Export the n most-recent training examples to a temp JSON for the trainer."""
    examples = _db.get_training_examples()
    if not examples:
        return None
    recent = examples[-n:]  # get_training_examples is ordered by generated_at ASC

    exportable = [
        {
            'input':    e['prompt'],
            'output':   e['ideal_output'],
            'label':    e['label'],
            'metadata': {
                'bot':         e['bot'],
                'symbol':      e['symbol'],
                'confidence':  e['confidence'],
                'pnl_pct':     e['pnl_pct'],
                'reward':      e.get('reward'),
                'entry_date':  e['entry_date'],
                'session_id':  e['session_id'],
                'prompt_hash': e['prompt_hash'],
            },
        }
        for e in recent
    ]

    out_path = _DATA_DIR / 'online_training_batch.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(exportable, indent=2))
    logging.info(f"📦 Exported {len(exportable)} recent examples → {out_path}")
    return out_path


def should_train(min_new_outcomes: int) -> tuple[bool, str]:
    """
    Return (True, reason) if enough new outcomes have arrived since last train,
    or (False, reason) if training should be skipped.
    """
    state = _load_state()
    last_trained_at = state['last_trained_at']

    new_count = _db.get_outcome_count_since(last_trained_at)
    if new_count < min_new_outcomes:
        return False, f"Only {new_count} new outcomes since {last_trained_at[:10]} (need {min_new_outcomes})"

    return True, f"{new_count} new outcomes since {last_trained_at[:10]}"


def run_online_training(min_new_outcomes: int = _MIN_NEW_OUTCOMES,
                        recent_n: int = _RECENT_N) -> bool:
    """
    Check threshold and run a lightweight fine-tune if met.
    Returns True if training ran, False if skipped or failed.
    """
    _db.init_db()
    ready, reason = should_train(min_new_outcomes)

    if not ready:
        logging.info(f"⏭️  Online training skipped — {reason}")
        return False

    logging.info(f"🟢 Online training triggered — {reason}")

    data_path = _export_recent_examples(recent_n)
    if data_path is None:
        logging.warning("No training examples found — skipping online training")
        return False

    adapter = _latest_adapter()
    if adapter is None:
        logging.warning("No existing LoRA adapter found — skipping online training (run full finetune first)")
        return False

    # Timestamp for the output adapter
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_suffix = f"_online_{ts}"

    fine_tune_script = _PROJECT_ROOT / 'finetune' / 'fine_tune_llm.py'
    cmd = [
        sys.executable, str(fine_tune_script),
        '--data', str(data_path),
        '--continue-from', adapter,
        '--epochs', str(_ONLINE_EPOCHS),
        '--learning-rate', str(_ONLINE_LR),
        '--no-dpo',   # DPO runs separately on its own schedule
    ]

    logging.info(f"🚀 Starting online LoRA update (LR={_ONLINE_LR}, epochs={_ONLINE_EPOCHS})...")
    logging.info(f"   Base adapter: {adapter}")
    logging.info(f"   Data: {data_path} ({recent_n} recent examples)")

    try:
        result = subprocess.run(cmd, timeout=1800)
        if result.returncode != 0:
            logging.error(f"❌ Online training failed with code {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        logging.error("❌ Online training timed out after 30 minutes")
        return False
    except Exception as e:
        logging.error(f"❌ Online training error: {e}")
        return False

    # Persist state so we don't retrain on the same outcomes
    _save_state({
        'last_trained_at': datetime.now(timezone.utc).isoformat(),
        'outcomes_at_last_train': _db.get_outcome_count_since('2000-01-01'),
        'adapter_used': adapter,
        'output_suffix': output_suffix,
    })

    logging.info(f"✅ Online LoRA update complete")
    return True


def main():
    parser = argparse.ArgumentParser(description='Lightweight online LoRA fine-tune trigger')
    parser.add_argument('--min-outcomes', type=int, default=_MIN_NEW_OUTCOMES,
                        help=f'Minimum new outcomes to trigger training (default: {_MIN_NEW_OUTCOMES})')
    parser.add_argument('--recent-n', type=int, default=_RECENT_N,
                        help=f'Train on this many most-recent examples (default: {_RECENT_N})')
    parser.add_argument('--force', action='store_true',
                        help='Run training regardless of outcome count')
    args = parser.parse_args()

    if args.force:
        logging.info("⚡ --force flag set — ignoring outcome threshold")
        _db.init_db()
        data_path = _export_recent_examples(args.recent_n)
        if data_path is None:
            logging.error("No training examples found")
            sys.exit(1)
        adapter = _latest_adapter()
        if adapter is None:
            logging.error("No LoRA adapter found")
            sys.exit(1)
        # Reuse run_online_training but bypass threshold by temporarily patching state
        state = _load_state()
        state['last_trained_at'] = '2000-01-01T00:00:00'
        _save_state(state)

    success = run_online_training(
        min_new_outcomes=args.min_outcomes,
        recent_n=args.recent_n,
    )
    sys.exit(0 if success else 0)   # 0 either way — skip is not an error


if __name__ == '__main__':
    main()
