#!/usr/bin/env python3
"""
training_data_builder.py

Converts live trading decisions + realized outcomes into labelled fine-tuning
examples and writes them to the SQLite database (training_examples table) and
the JSON files consumed by finetune_model.py.

Run nightly (before finetune_model.py) — wired into both daemons at 2 AM.

Decision → outcome matching strategy
--------------------------------------
1. PRIMARY  — query outcomes table in SQLite (realized P&L from outcome_tracker).
2. FALLBACK — for executed decisions where no outcome record exists yet, fetch
   forward price via yfinance at HOLD_DAYS_FALLBACK trading days after entry.
3. HOLD decisions are included when the model was correct to hold
   (price stayed flat / moved against the skipped direction).

Quality gates
-------------
- Only decisions with confidence >= MIN_CONFIDENCE are included.
- Deduplication is enforced by the UNIQUE prompt_hash constraint in SQLite.
"""

import json
import hashlib
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf
import db as _db

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR     = _PROJECT_ROOT / 'finetune' / 'data'

MIN_CONFIDENCE      = 0.70
HOLD_DAYS_FALLBACK  = 5
WIN_THRESHOLD_PCT   = 0.005
LOSS_THRESHOLD_PCT  = -0.005


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prompt_hash(prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()


def _forward_price_change(symbol: str, entry_dt: datetime, hold_days: int) -> float | None:
    try:
        end    = entry_dt + timedelta(days=hold_days * 2)
        ticker = yf.Ticker(symbol)
        hist   = ticker.history(start=entry_dt.date(), end=end.date())
        if len(hist) < 2:
            return None
        entry_price = hist['Close'].iloc[0]
        exit_price  = hist['Close'].iloc[min(hold_days, len(hist) - 1)]
        return float((exit_price - entry_price) / entry_price)
    except Exception as e:
        logging.warning(f"yfinance lookup failed for {symbol}: {e}")
        return None


def _label_from_pnl(pnl_pct: float, decision: str) -> str | None:
    if decision in ('buy', 'buy_call'):
        if pnl_pct >= WIN_THRESHOLD_PCT:
            return 'winner'
        if pnl_pct <= LOSS_THRESHOLD_PCT:
            return 'loser'
    elif decision in ('sell', 'buy_put'):
        if pnl_pct <= LOSS_THRESHOLD_PCT:
            return 'winner'
        if pnl_pct >= WIN_THRESHOLD_PCT:
            return 'loser'
    return None


def _build_ideal_output(decision: str, confidence: float, reasoning: str, label: str) -> str:
    action_map = {
        'buy': 'BUY', 'sell': 'SELL', 'hold': 'HOLD',
        'buy_call': 'BUY_CALL', 'buy_put': 'BUY_PUT',
    }
    action_str   = action_map.get(decision, decision.upper())
    outcome_note = (
        'This was a profitable trade.'          if label == 'winner' else
        'This trade resulted in a loss — review the signals more carefully.'
    )
    return (
        f"Decision: {action_str}\n"
        f"Confidence: {confidence:.2f}\n"
        f"Reasoning: {reasoning}\n"
        f"Outcome: {outcome_note}"
    )


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_and_store(bot: str) -> tuple[int, int]:
    """
    Build new training examples for `bot` ('stock' or 'options'), write them
    to SQLite, and export the full set to the JSON file for finetune_model.py.

    Returns (new_examples_added, total_in_db).
    """
    _db.init_db()
    existing_hashes = _db.get_existing_prompt_hashes()
    decisions       = _db.get_executed_decisions(bot=bot, min_confidence=MIN_CONFIDENCE)

    # Also include HOLDs with sufficient confidence (not in executed_decisions)
    with _db.get_conn() as conn:
        hold_rows = conn.execute(
            "SELECT * FROM decisions WHERE bot=? AND decision='hold' AND confidence>=? ORDER BY timestamp",
            [bot, MIN_CONFIDENCE]
        ).fetchall()
    hold_decisions = [dict(r) for r in hold_rows]

    all_candidates = decisions + hold_decisions
    added = 0

    for rec in all_candidates:
        decision   = rec.get('decision', 'hold')
        symbol     = rec.get('symbol', '')
        prompt     = rec.get('prompt', '')
        reasoning  = rec.get('reasoning', '')
        confidence = float(rec.get('confidence', 0.5))

        if not prompt or not symbol:
            continue

        ph = _prompt_hash(prompt)
        if ph in existing_hashes:
            continue   # already have this example

        try:
            entry_dt = datetime.fromisoformat(rec['timestamp'])
        except (KeyError, ValueError):
            continue

        # --- Determine label ---
        label   = None
        pnl_pct = None

        # 1. Check SQLite outcomes table
        outcome = _db.get_outcome_by_entry(symbol, entry_dt.date().isoformat())
        if outcome:
            pnl_pct = outcome.get('pnl_pct', 0)
            label   = 'winner' if outcome.get('won') else 'loser'

        # 2. yfinance forward price fallback (executed non-hold trades only)
        elif rec.get('executed') and decision != 'hold':
            days_elapsed = (datetime.now() - entry_dt).days
            if days_elapsed >= HOLD_DAYS_FALLBACK:
                pnl_pct = _forward_price_change(symbol, entry_dt, HOLD_DAYS_FALLBACK)
                if pnl_pct is not None:
                    label = _label_from_pnl(pnl_pct, decision)

        # 3. HOLD validation
        elif decision == 'hold':
            days_elapsed = (datetime.now() - entry_dt).days
            if days_elapsed >= HOLD_DAYS_FALLBACK:
                pnl_pct = _forward_price_change(symbol, entry_dt, HOLD_DAYS_FALLBACK)
                if pnl_pct is not None:
                    if abs(pnl_pct) < WIN_THRESHOLD_PCT:
                        label = 'winner'
                    elif pnl_pct > WIN_THRESHOLD_PCT:
                        label = 'loser'

        if label is None:
            continue

        example = {
            'input':  prompt,
            'output': _build_ideal_output(decision, confidence, reasoning, label),
            'label':  label,
            'metadata': {
                'bot':          bot,
                'symbol':       symbol,
                'decision':     decision,
                'confidence':   round(confidence, 4),
                'pnl_pct':      round(pnl_pct, 6) if pnl_pct is not None else None,
                'entry_date':   entry_dt.date().isoformat(),
                'session_id':   rec.get('session_id', ''),
                'prompt_hash':  ph,
                'generated_at': datetime.now().isoformat(),
            },
        }

        if _db.insert_training_example(example):
            existing_hashes.add(ph)
            added += 1

    # Export full set to JSON for finetune_model.py
    all_examples  = _db.get_training_examples(bot=bot)
    filename      = 'training_data.json' if bot == 'stock' else 'options_training_data.json'
    output_path   = _DATA_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to the format finetune_model.py expects
    exportable = [
        {'input': e['prompt'], 'output': e['ideal_output'], 'label': e['label'],
         'metadata': {
             'bot': e['bot'], 'symbol': e['symbol'], 'confidence': e['confidence'],
             'pnl_pct': e['pnl_pct'], 'entry_date': e['entry_date'],
             'session_id': e['session_id'], 'prompt_hash': e['prompt_hash'],
         }}
        for e in all_examples
    ]
    with open(output_path, 'w') as f:
        json.dump(exportable, f, indent=2)

    return added, len(all_examples)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(bot: str) -> None:
    if bot in ('stock', 'both'):
        added, total = build_and_store('stock')
        logging.info(f"✅ Stock: +{added} new examples → {total} total")

    if bot in ('options', 'both'):
        added, total = build_and_store('options')
        logging.info(f"✅ Options: +{added} new examples → {total} total")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build fine-tuning examples from live trade decisions')
    parser.add_argument('--bot', choices=['stock', 'options', 'both'], default='both')
    args = parser.parse_args()
    run(args.bot)
