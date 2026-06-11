#!/usr/bin/env python3
"""
build_golden_set.py — Build the B3 golden evaluation set from real trade history.

Sources:
  WINNERS: decisions + outcomes joined on outcomes.buy_order_id = decisions.order_id
           criterion: decision IN (buy, buy_call, buy_put), excess_return_pct >= 5%
  HOLDS:   training_examples WHERE label = 'correct_hold'
           (holds never execute so they never appear in outcomes)

Guards:
  - Refuses to write if < 5 correct_hold examples found
  - Write-once: refuses to overwrite existing golden_set.jsonl without --force
"""

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _pathfix  # noqa: F401

_SCRIPTS_DIR  = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
_DEFAULT_DB   = _PROJECT_ROOT / 'logs' / 'trading.db'
_DEFAULT_OUT  = _PROJECT_ROOT / 'logs' / 'eval' / 'golden_set.jsonl'

_MAX_WINNERS = 10
_MAX_HOLDS   = 10
_MIN_HOLDS   = 5   # refuse to write if fewer correct_hold examples found


def _load_winners(db_path: Path) -> list[dict]:
    """BUY decisions where excess_return_pct >= 5%, sourced from decisions+outcomes join."""
    if not db_path.exists():
        logging.warning("DB not found: %s — no winners loaded", db_path)
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("""
            SELECT d.id     AS decision_id,
                   d.bot,
                   d.symbol,
                   d.decision,
                   d.prompt,
                   d.confidence,
                   o.excess_return_pct,
                   o.regime
            FROM decisions d
            JOIN outcomes o ON o.buy_order_id = d.order_id
            WHERE d.executed = 1
              AND d.prompt IS NOT NULL AND d.prompt != ''
              AND d.decision IN ('buy', 'buy_call', 'buy_put')
              AND d.order_id IS NOT NULL
              AND o.excess_return_pct >= 0.05
            ORDER BY o.excess_return_pct DESC
        """).fetchall()
    finally:
        conn.close()

    return [
        {
            'prompt':             r['prompt'],
            'expected':           r['decision'],        # buy | buy_call | buy_put
            'bot':                r['bot'],
            'source':             'outcomes',
            'source_decision_id': r['decision_id'],
            'excess_return_pct':  r['excess_return_pct'],
            'regime':             r['regime'],
        }
        for r in rows[:_MAX_WINNERS]
    ]


def _load_holds(db_path: Path) -> list[dict]:
    """correct_hold examples from training_examples table (SPY-validated holds)."""
    if not db_path.exists():
        logging.warning("DB not found: %s — no holds loaded", db_path)
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("""
            SELECT bot, symbol, prompt, pnl_pct, generated_at
            FROM training_examples
            WHERE label = 'correct_hold'
              AND prompt IS NOT NULL AND prompt != ''
            ORDER BY generated_at DESC
        """).fetchall()
    finally:
        conn.close()

    return [
        {
            'prompt':   r['prompt'],
            'expected': 'hold',
            'bot':      r['bot'],
            'source':   'training_examples',
            'pnl_pct':  r['pnl_pct'],
        }
        for r in rows[:_MAX_HOLDS]
    ]


def build_golden_set(db_path: Path, out_path: Path, force: bool = False) -> int:
    """Build and write golden_set.jsonl. Returns 0 on success, 1 on error."""
    if out_path.exists() and not force:
        logging.error(
            "golden_set.jsonl already exists: %s — use --force to overwrite.", out_path
        )
        return 1

    winners = _load_winners(db_path)
    holds   = _load_holds(db_path)

    logging.info("Winners loaded: %d (max %d)", len(winners), _MAX_WINNERS)
    logging.info("Holds loaded:   %d (max %d)", len(holds),   _MAX_HOLDS)

    if len(holds) < _MIN_HOLDS:
        logging.error(
            "Only %d correct_hold example(s) found (minimum required: %d). "
            "Run more trading sessions to build up hold history before creating the golden set.",
            len(holds), _MIN_HOLDS,
        )
        return 1

    items = winners + holds
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(json.dumps(item) for item in items) + '\n')
    logging.info("Golden set written: %d items (%d winners + %d holds) → %s",
                 len(items), len(winners), len(holds), out_path)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description='Build golden evaluation set from real trade history')
    parser.add_argument('--db',    type=Path, default=_DEFAULT_DB,
                        help='SQLite DB path (default: logs/trading.db)')
    parser.add_argument('--out',   type=Path, default=_DEFAULT_OUT,
                        help='Output .jsonl path (default: logs/eval/golden_set.jsonl)')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing golden_set.jsonl')
    args = parser.parse_args()
    sys.exit(build_golden_set(args.db, args.out, args.force))


if __name__ == '__main__':
    main()
