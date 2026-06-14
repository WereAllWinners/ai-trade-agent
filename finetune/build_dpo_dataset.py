#!/usr/bin/env python3
"""
build_dpo_dataset.py

Constructs (prompt, chosen, rejected) preference triplets from the labeled
training examples stored in SQLite.  Preference weight = abs(pnl_pct) * confidence.

Winner outputs become "chosen"; loser outputs become "rejected".
Pairing strategy:
  1. Same-symbol pairs — winner trade vs loser trade on the same symbol.
  2. Cross-symbol synthetic pairs — when no real opposite exists, fabricate a
     rejected response by inverting the chosen decision and noting the loss.

Output: finetune/data/dpo_pairs.jsonl  (one JSON object per line)
        finetune/data/dpo_pairs_stats.json  (summary)

Run after training_data_builder.py, before fine_tune_llm.py.
"""

import json
import logging
import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR  = _PROJECT_ROOT / 'scripts'
_DATA_DIR     = _PROJECT_ROOT / 'finetune' / 'data'

import sys
sys.path.insert(0, str(_SCRIPTS_DIR))
import db as _db

_WINNER_LABELS = {'strong_winner', 'winner', 'weak_winner', 'correct_hold'}
_LOSER_LABELS  = {'strong_loser', 'loser', 'weak_loser', 'missed_opportunity'}

_SYSTEM_PROMPT = (
    "You are an expert financial trading advisor with knowledge from the world's best "
    "investors including Warren Buffett, Nancy Pelosi, Cathie Wood, and Michael Burry. "
    "You analyze stocks using technical indicators, fundamental analysis, insider trading "
    "patterns, congressional trades, and proven strategies."
)


def _preference_weight(pnl_pct: float | None, confidence: float | None) -> float:
    """Higher magnitude trades with high confidence produce stronger preference signal."""
    p = abs(pnl_pct) if pnl_pct is not None else 0.0
    c = confidence if confidence is not None else 0.5
    return round(p * c, 6)


def _format_chat(prompt: str, response: str) -> dict:
    return {
        "prompt": (
            f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        ),
        "chosen": response + "<|im_end|>",
    }


def _synthetic_rejected(chosen_output: str, symbol: str, decision: str) -> str:
    """
    Fabricate a low-quality rejected response by inverting the decision and
    adding generic hedging language that reflects poor analytical practice.
    """
    inv = {'BUY': 'SELL', 'SELL': 'BUY', 'BUY_CALL': 'BUY_PUT', 'BUY_PUT': 'BUY_CALL', 'HOLD': 'BUY'}
    first_line = chosen_output.split('\n')[0]  # "Decision: BUY"
    orig_action = first_line.replace('Decision: ', '').strip()
    wrong_action = inv.get(orig_action, 'HOLD')
    return (
        f"Decision: {wrong_action}\n"
        f"Confidence: 0.55\n"
        f"Reasoning: The market is uncertain. {symbol} could go either way. "
        f"Indicators are mixed and there is no clear signal to act on."
    )


def build_dpo_pairs(bot: str = None) -> tuple[int, int]:
    """
    Build DPO preference pairs from labeled training examples.
    Returns (real_pairs, synthetic_pairs).
    """
    _db.init_db()
    examples = _db.get_training_examples(bot=bot)

    if not examples:
        logging.warning("No training examples found — DPO dataset will be empty")
        return [], 0, 0

    winners = [e for e in examples if e['label'] in _WINNER_LABELS]
    losers  = [e for e in examples if e['label'] in _LOSER_LABELS]

    logging.info(f"Found {len(winners)} winners, {len(losers)} losers from {len(examples)} total examples")

    # Index losers by symbol for fast lookup
    loser_by_symbol: dict[str, list] = defaultdict(list)
    for ex in losers:
        loser_by_symbol[ex['symbol']].append(ex)

    pairs = []
    real_count = 0
    synthetic_count = 0

    for winner in winners:
        symbol   = winner['symbol']
        prompt   = winner['prompt']
        chosen   = winner['ideal_output']
        w_score  = _preference_weight(winner.get('pnl_pct'), winner.get('confidence'))

        # Try to find a real loser on the same symbol
        real_losers = loser_by_symbol.get(symbol, [])
        if real_losers:
            # Pick the worst loser (strongest negative signal)
            worst = max(real_losers, key=lambda e: _preference_weight(e.get('pnl_pct'), e.get('confidence')))
            rejected = worst['ideal_output']
            r_score  = _preference_weight(worst.get('pnl_pct'), worst.get('confidence'))
            real_count += 1
        else:
            # Synthetic rejected
            rejected = _synthetic_rejected(chosen, symbol, winner.get('decision', 'buy'))
            r_score  = 0.01
            synthetic_count += 1

        chat = _format_chat(prompt, chosen)
        pairs.append({
            "prompt":          chat["prompt"],
            "chosen":          chat["chosen"],
            "rejected":        rejected + "<|im_end|>",
            "chosen_score":    w_score,
            "rejected_score":  r_score,
            "symbol":          symbol,
            "bot":             winner.get('bot', ''),
            "winner_label":    winner['label'],
        })

    return pairs, real_count, synthetic_count


def save_pairs(pairs: list, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for p in pairs:
            f.write(json.dumps(p) + '\n')
    logging.info(f"💾 Saved {len(pairs)} DPO pairs → {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Build DPO preference dataset')
    parser.add_argument('--bot', choices=['stock', 'options', 'both'], default='both')
    parser.add_argument('--output', type=str, default=None, help='Output JSONL path')
    args = parser.parse_args()

    bot_filter = None if args.bot == 'both' else args.bot
    pairs, real_count, synthetic_count = build_dpo_pairs(bot=bot_filter)

    if not pairs:
        logging.warning("No DPO pairs generated — need more labeled training data")
        return

    output_path = Path(args.output) if args.output else _DATA_DIR / 'dpo_pairs.jsonl'
    save_pairs(pairs, output_path)

    stats = {
        'total_pairs':     len(pairs),
        'real_pairs':      real_count,
        'synthetic_pairs': synthetic_count,
        'bot':             args.bot,
        'generated_at':    datetime.now().isoformat(),
        'output_path':     str(output_path),
    }
    stats_path = output_path.with_name('dpo_pairs_stats.json')
    stats_path.write_text(json.dumps(stats, indent=2))

    logging.info(
        f"✅ DPO dataset ready: {len(pairs)} pairs "
        f"({real_count} real + {synthetic_count} synthetic) → {output_path}"
    )
    logging.info(f"📊 Stats → {stats_path}")


if __name__ == '__main__':
    main()
