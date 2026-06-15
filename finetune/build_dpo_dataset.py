#!/usr/bin/env python3
"""
build_dpo_dataset.py

Constructs (prompt, chosen, rejected) preference triplets from the labeled
training examples stored in SQLite.

Pair strategies (Session 3 revision):
  1. Loser → counterfactual pairs (primary, ~599 pairs):
     For each loser-family row with a matching counterfactual in the DB,
     rejected = the loser's ideal_output (the losing decision),
     chosen   = the counterfactual HOLD output (what should have been done).
     Gap is always unambiguous — a realized loss vs a HOLD that avoided it.
  2. weak_winner → strong_winner pairs (gap-gated, modest set):
     Only when strong_winner.pnl_pct - weak_winner.pnl_pct >= _DPO_MIN_GAP
     AND both rows share the same bot.  With 7 strong_winners this yields a
     small, high-quality set; pairs below the gap are never forced.

Output: finetune/data/dpo_pairs.jsonl  (one JSON object per line)
        finetune/data/dpo_pairs_stats.json  (summary)

Run after training_data_builder.py (and regenerate_historical for Session 4+).
"""

import json
import logging
import os
import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR  = _PROJECT_ROOT / 'scripts'
_TRAINING_DIR = _SCRIPTS_DIR / 'training'
_DATA_DIR     = _PROJECT_ROOT / 'finetune' / 'data'

import sys
sys.path.insert(0, str(_SCRIPTS_DIR))
sys.path.insert(0, str(_TRAINING_DIR))
import db as _db
from training_data_builder import _is_clean_ideal_output

_LOSER_LABELS = frozenset({'strong_loser', 'loser', 'weak_loser'})

_DPO_MIN_GAP:   float = float(os.getenv('DPO_MIN_GAP', '0.08'))
_DPO_MIN_PAIRS: int   = 8   # must match DPOStageTrainer.MIN_DPO_PAIRS

_SYSTEM_PROMPT = (
    "You are an expert financial trading advisor with knowledge from the world's best "
    "investors including Warren Buffett, Nancy Pelosi, Cathie Wood, and Michael Burry. "
    "You analyze stocks using technical indicators, fundamental analysis, insider trading "
    "patterns, congressional trades, and proven strategies."
)


def _format_pair(prompt: str, chosen: str, rejected: str) -> dict:
    """Wrap a (prompt, chosen, rejected) triple in the ChatML format DPOTrainer expects."""
    formatted_prompt = (
        f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return {
        "prompt":   formatted_prompt,
        "chosen":   chosen   + "<|im_end|>",
        "rejected": rejected + "<|im_end|>",
    }


def build_dpo_pairs(bot: str = None, db_path=None) -> tuple[list, dict]:
    """Build DPO preference pairs from labeled training examples.

    Returns (pairs, stats_dict) where stats_dict has keys:
      loser_cf_pairs, weak_strong_pairs, skipped_no_cf, skipped_guard_violations,
      total, skipped_gap_below_threshold.
    """
    if db_path:
        _db.init_db(Path(db_path))
        examples = _db.get_training_examples(bot=bot, db_path=Path(db_path))
    else:
        _db.init_db()
        examples = _db.get_training_examples(bot=bot)

    if not examples:
        logging.warning("No training examples found — DPO dataset will be empty")
        return [], {'loser_cf_pairs': 0, 'weak_strong_pairs': 0,
                    'skipped_no_cf': 0, 'skipped_guard_violations': 0,
                    'skipped_gap_below_threshold': 0, 'total': 0}

    # ── Index by label ────────────────────────────────────────────────────────
    loser_rows    = [e for e in examples if e['label'] in _LOSER_LABELS]
    weak_winners  = [e for e in examples
                     if e['label'] == 'weak_winner' and e.get('pnl_pct') is not None
                     and _is_clean_ideal_output(e.get('ideal_output', ''))]
    strong_winners = [e for e in examples
                      if e['label'] == 'strong_winner' and e.get('pnl_pct') is not None
                      and _is_clean_ideal_output(e.get('ideal_output', ''))]

    # Index counterfactuals by the raw prompt (same prompt as the loser row).
    # Each CF row's prompt_hash = _prompt_hash(prompt + '|counterfactual');
    # the prompt field is identical to the matching loser's prompt field.
    cf_output_by_prompt: dict[str, str] = {
        e['prompt']: e['ideal_output']
        for e in examples
        if e['label'] == 'counterfactual'
        and _is_clean_ideal_output(e.get('ideal_output', ''))
        and e.get('prompt')
    }

    logging.info(
        "DPO input: %d loser rows, %d counterfactuals, "
        "%d weak_winners, %d strong_winners (from %d total examples)",
        len(loser_rows), len(cf_output_by_prompt),
        len(weak_winners), len(strong_winners), len(examples),
    )

    pairs: list[dict] = []
    skipped_no_cf     = 0
    guard_violations  = 0
    loser_cf_count    = 0

    # ── Strategy 1: loser → counterfactual ───────────────────────────────────
    for loser in loser_rows:
        prompt   = loser.get('prompt', '')
        rejected = loser.get('ideal_output', '')

        if not prompt:
            skipped_no_cf += 1
            continue
        if not _is_clean_ideal_output(rejected):
            skipped_no_cf += 1
            continue

        chosen = cf_output_by_prompt.get(prompt)
        if not chosen:
            skipped_no_cf += 1
            continue

        # Hard guard: chosen must differ from rejected
        if chosen == rejected:
            guard_violations += 1
            continue

        pair = _format_pair(prompt, chosen, rejected)
        pair.update({
            'pair_type':    'loser_counterfactual',
            'loser_label':  loser['label'],
            'pnl_pct':      loser.get('pnl_pct'),
            'bot':          loser.get('bot', ''),
            'symbol':       loser.get('symbol', ''),
        })
        pairs.append(pair)
        loser_cf_count += 1

    # ── Strategy 2: weak_winner → strong_winner (gap-gated) ──────────────────
    weak_strong_count   = 0
    skipped_below_gap   = 0
    seen_ws_pairs: set  = set()   # dedup identical (chosen, rejected) combinations

    for sw in strong_winners:
        sw_pnl  = float(sw['pnl_pct'])
        sw_bot  = sw.get('bot', '')
        chosen  = sw.get('ideal_output', '')
        prompt  = sw.get('prompt', '')
        if not prompt or not chosen:
            continue

        for ww in weak_winners:
            if ww.get('bot', '') != sw_bot:
                continue
            ww_pnl   = float(ww['pnl_pct'])
            gap      = sw_pnl - ww_pnl
            if gap < _DPO_MIN_GAP:
                skipped_below_gap += 1
                continue

            rejected = ww.get('ideal_output', '')
            if not rejected or chosen == rejected:
                guard_violations += 1
                continue

            dedup_key = (chosen, rejected)
            if dedup_key in seen_ws_pairs:
                continue
            seen_ws_pairs.add(dedup_key)

            pair = _format_pair(prompt, chosen, rejected)
            pair.update({
                'pair_type':         'weak_to_strong_winner',
                'strong_winner_pnl': sw_pnl,
                'weak_winner_pnl':   ww_pnl,
                'pnl_gap':           round(gap, 6),
                'bot':               sw_bot,
                'symbol':            sw.get('symbol', ''),
            })
            pairs.append(pair)
            weak_strong_count += 1

    total = len(pairs)
    stats = {
        'loser_cf_pairs':             loser_cf_count,
        'weak_strong_pairs':          weak_strong_count,
        'skipped_no_cf':              skipped_no_cf,
        'skipped_guard_violations':   guard_violations,
        'skipped_gap_below_threshold': skipped_below_gap,
        'total':                      total,
    }

    if total < _DPO_MIN_PAIRS:
        logging.warning(
            "⚠️  Only %d valid DPO pairs (need ≥ %d) — DPO stage will be skipped. "
            "Stats: %s", total, _DPO_MIN_PAIRS, stats,
        )
    else:
        logging.info(
            "✅ %d DPO pairs: %d loser→CF + %d weak→strong (%d skipped: %d no-CF, "
            "%d guard, %d below-gap)",
            total, loser_cf_count, weak_strong_count,
            skipped_no_cf + guard_violations + skipped_below_gap,
            skipped_no_cf, guard_violations, skipped_below_gap,
        )

    return pairs, stats


def save_pairs(pairs: list, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for p in pairs:
            f.write(json.dumps(p) + '\n')
    logging.info("💾 Saved %d DPO pairs → %s", len(pairs), output_path)


def main():
    parser = argparse.ArgumentParser(description='Build DPO preference dataset')
    parser.add_argument('--bot', choices=['stock', 'options', 'both'], default='both')
    parser.add_argument('--output', type=str, default=None, help='Output JSONL path')
    parser.add_argument('--db', type=str, default=None, help='SQLite DB path override')
    args = parser.parse_args()

    bot_filter = None if args.bot == 'both' else args.bot
    pairs, stats = build_dpo_pairs(bot=bot_filter, db_path=args.db)

    if not pairs:
        logging.warning("No DPO pairs generated — need more labeled training data")
        return

    output_path = Path(args.output) if args.output else _DATA_DIR / 'dpo_pairs.jsonl'
    save_pairs(pairs, output_path)

    stats['bot']          = args.bot
    stats['generated_at'] = datetime.now().isoformat()
    stats['output_path']  = str(output_path)
    stats_path = output_path.with_name('dpo_pairs_stats.json')
    stats_path.write_text(json.dumps(stats, indent=2))

    logging.info("📊 Stats → %s", stats_path)


if __name__ == '__main__':
    main()
