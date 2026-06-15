#!/usr/bin/env python3
"""
Model Fine-tuning Script - Runs the nightly full fine-tune via fine_tune_llm.py.
Called by trading_daemon.py (stock) and options_daemon.py (options) after market
close and training data is built.
"""
import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # scripts/training/ for tdb imports
from training import FINETUNE_PYTHON as _FINETUNE_PYTHON, FINETUNE_PTXAS as _FINETUNE_PTXAS
from training_data_builder import (  # noqa: E402
    _HOLD_TEACHING_LABELS,
    _SFT_TRAIN_LABELS,
    _SFT_MAX_HOLD_SHARE,
)

_SCRIPTS_DIR  = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

_GPU_TEMP_LIMIT   = 75   # °C — skip training above this to prevent thermal shutdown
_MIN_TAKE_TRADE   = int(os.getenv('FINETUNE_MIN_TAKE_TRADE', '40'))
_MIN_TOTAL_SFT    = int(os.getenv('FINETUNE_MIN_TOTAL_SFT',  '200'))
_TAKE_TRADE_LABELS: frozenset[str] = frozenset({'winner', 'strong_winner'})


def _cadence_gate(training_data_path: Path, bot: str = 'stock') -> bool:
    """Return True iff fine-tuning should proceed.

    Reads the SFT-only file (training_data_sft.json / options_training_data_sft.json).
    Counts only from that file so the denominator matches what SFTTrainer will see.

    Fires (returns False) when any condition is unmet:
      take_trade < _MIN_TAKE_TRADE  — binding at current volume
      hold_share > _SFT_MAX_HOLD_SHARE
      total_sft  < _MIN_TOTAL_SFT   — informational floor

    A fired gate is the correct outcome when data is insufficient.  The log line
    surfaces the reason; no error is raised, no subprocess is launched.
    """
    if not training_data_path.exists():
        logging.info("⏳ %s: SFT file not found — waiting for training data", bot)
        return False

    with open(training_data_path) as f:
        sft_data = json.load(f)

    if not sft_data:
        logging.info("⏳ %s: SFT file is empty — waiting for training data", bot)
        return False

    total_sft   = len(sft_data)
    take_trade  = sum(1 for ex in sft_data if ex.get('label') in _TAKE_TRADE_LABELS)
    hold_n      = sum(1 for ex in sft_data if ex.get('label') in _HOLD_TEACHING_LABELS)
    hold_share  = hold_n / total_sft if total_sft else 0.0

    reasons = []
    if take_trade < _MIN_TAKE_TRADE:
        reasons.append(f"take_trade={take_trade} < {_MIN_TAKE_TRADE}")
    if hold_share > _SFT_MAX_HOLD_SHARE:
        reasons.append(f"hold_share={hold_share:.1%} > {_SFT_MAX_HOLD_SHARE:.0%}")
    if total_sft < _MIN_TOTAL_SFT:
        reasons.append(f"total_sft={total_sft} < {_MIN_TOTAL_SFT}")

    if reasons:
        logging.warning(
            "fine-tune SKIPPED (%s): %s.  Accumulating; current model retained.",
            bot, ", ".join(reasons),
        )
        return False

    logging.info(
        "✅ Cadence gate PASSED (%s): take_trade=%d, hold_share=%.1f%%, total_sft=%d",
        bot, take_trade, 100 * hold_share, total_sft,
    )
    return True


def _gpu_temp() -> int | None:
    """Return current GPU temperature in °C, or None if unavailable."""
    try:
        import subprocess as _sp
        out = _sp.check_output(
            ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'],
            timeout=5,
        )
        return int(out.decode().strip())
    except Exception:
        return None


def _run_promoter(finetune_dir: Path, adapters_before: set) -> None:
    """Find the adapter created by this fine-tune run and pass it to model_promoter."""
    adapters_after = {
        d for d in finetune_dir.iterdir()
        if d.is_dir() and 'lora' in d.name and (d / 'adapter_config.json').exists()
    }
    new_adapters = sorted(adapters_after - adapters_before, key=lambda d: d.stat().st_mtime)
    if not new_adapters:
        logging.warning("⚠️  Could not find new adapter directory — skipping promotion")
        return

    new_adapter = new_adapters[-1]
    logging.info(f"🔍 New adapter detected: {new_adapter.name}")

    # Look for a companion merged BF16 model written by fine_tune_llm.py.
    # Primary: read the absolute path from training_metadata.json (reliable across CWDs).
    # Fallback: reconstruct via name substitution for old adapters without metadata.
    merged_path = None
    metadata_file = new_adapter / 'training_metadata.json'
    if metadata_file.exists():
        try:
            meta = json.loads(metadata_file.read_text())
            candidate = Path(meta.get('merged_dir', ''))
            # fine_tune_llm.py writes a relative path; resolve against project root.
            # The original is_absolute() check always failed for relative paths.
            if not candidate.is_absolute():
                candidate = _PROJECT_ROOT / candidate
            if candidate.is_dir():
                merged_path = str(candidate)
        except Exception as _meta_err:
            logging.warning("Could not read training_metadata.json: %s", _meta_err)

    if merged_path is None:
        merged_candidate = finetune_dir / new_adapter.name.replace('_lora_', '_lora_merged_bf16_')
        if merged_candidate.is_dir():
            merged_path = str(merged_candidate)

    if merged_path:
        logging.info(f"🔍 Merged model detected: {Path(merged_path).name}")
    else:
        logging.warning("⚠️  No merged model found — vLLM will keep serving the previous merged model")

    promoter_script = Path(__file__).resolve().parent / 'model_promoter.py'
    cmd = [sys.executable, str(promoter_script), '--candidate', str(new_adapter)]
    if merged_path:
        cmd += ['--merged-model', merged_path]

    try:
        # Budget: smoke test (~5 min) + 2× replay eval at 40 min each = ~85 min.
        # 7200s (120 min) gives comfortable headroom for cold 32B loads.
        subprocess.run(cmd, timeout=7200)
    except subprocess.TimeoutExpired:
        logging.error("❌ Model promotion timed out after 120 minutes")
    except Exception as e:
        logging.error(f"❌ Model promotion failed: {e}")


def finetune_model(training_data_path: Path) -> None:
    """Run the full nightly fine-tune via fine_tune_llm.py."""
    temp = _gpu_temp()
    if temp is not None and temp >= _GPU_TEMP_LIMIT:
        logging.warning(f"🌡️  GPU at {temp}°C — skipping fine-tune to prevent thermal shutdown (limit: {_GPU_TEMP_LIMIT}°C)")
        return

    # Infer bot name from filename for gate logging.
    bot = 'options' if 'options' in training_data_path.name else 'stock'

    if not _cadence_gate(training_data_path, bot):
        return   # gate fires — reason already logged; do not launch subprocess

    with open(training_data_path) as f:
        training_data = json.load(f)

    logging.info(f"📚 {len(training_data)} SFT examples — starting full fine-tune")

    # Build DPO preference pairs from labelled outcomes before SFT training.
    dpo_script = _PROJECT_ROOT / 'finetune' / 'build_dpo_dataset.py'
    if dpo_script.exists():
        try:
            subprocess.run([sys.executable, str(dpo_script)], timeout=120, check=True)
            logging.info("✅ DPO pairs built")
        except Exception as e:
            logging.warning(f"⚠️  DPO dataset build failed (non-fatal): {e}")

    fine_tune_script = _PROJECT_ROOT / 'finetune' / 'fine_tune_llm.py'
    finetune_env = {**os.environ, "TRITON_PTXAS_BLACKWELL_PATH": _FINETUNE_PTXAS}

    # Always continue from the current production adapter so training is
    # incremental (1 epoch) rather than FRESH (3 epochs from base model).
    # FRESH mode loads the 33B base model and runs 3× more gradient steps,
    # which has caused kernel-level NVRM OOM crashes on the GB10.
    lora_latest = _PROJECT_ROOT / 'finetune' / 'finance_qwen_32b_lora_latest'
    continue_from = str(lora_latest.resolve()) if lora_latest.exists() else None

    cmd = [
        _FINETUNE_PYTHON, str(fine_tune_script),
        '--data', str(training_data_path),
        '--epochs', '1',       # nightly incremental: 1 epoch is sufficient
        '--batch-size', '1',   # batch 1 is safe on 128 GB unified memory;
                               # batch 2 + inference model = kernel OOM
        '--no-dpo',            # skip DPO for nightly runs (saves time + memory)
    ]
    if continue_from:
        cmd += ['--continue-from', continue_from]

    # Snapshot adapter directories before training so we can detect the new one
    finetune_dir = _PROJECT_ROOT / 'finetune'
    adapters_before = {
        d for d in finetune_dir.iterdir()
        if d.is_dir() and 'lora' in d.name and (d / 'adapter_config.json').exists()
    }

    try:
        result = subprocess.run(cmd, timeout=7200, env=finetune_env)
        if result.returncode == 0:
            logging.info("✅ Model fine-tuning complete")
            _run_promoter(finetune_dir, adapters_before)
        else:
            logging.error(f"❌ Fine-tuning failed with code {result.returncode}")
    except subprocess.TimeoutExpired:
        logging.error("❌ Fine-tuning timed out after 120 minutes")
    except Exception as e:
        logging.error(f"❌ Fine-tuning failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        type=Path,
        default=_PROJECT_ROOT / 'finetune' / 'data' / 'training_data_sft.json',
        help='Path to training data JSON (default: stock training data)',
    )
    args = parser.parse_args()
    finetune_model(args.data)
