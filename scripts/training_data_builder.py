#!/usr/bin/env python3
"""
training_data_builder.py

Converts live trading decisions + realized outcomes into labelled fine-tuning
examples and appends them to the training data files consumed by
finetune_model.py.

Run nightly (before finetune_model.py) — wired into both daemons at 2 AM.

Decision → outcome matching strategy
--------------------------------------
1. PRIMARY  — match executed decisions to trade_outcomes.jsonl (realized P&L
   written by outcome_tracker.py after a position closes).
2. FALLBACK — for executed decisions where no outcome record exists yet, fetch
   forward price via yfinance at hold_days after entry to estimate P&L.
3. HOLD decisions are included only when the model was correct to hold
   (i.e. price stayed flat / moved against the skipped direction).

Quality gates
-------------
- Only decisions with confidence >= MIN_CONFIDENCE are included.
- Examples already present in the training file (matched by prompt hash) are
  skipped to avoid duplicates across nightly runs.
"""

import json
import hashlib
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_SCRIPTS_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
_LOGS_DIR     = _PROJECT_ROOT / 'logs'
_DATA_DIR     = _PROJECT_ROOT / 'finetune' / 'data'

MIN_CONFIDENCE      = 0.70   # ignore low-conviction decisions
HOLD_DAYS_FALLBACK  = 5      # forward-price window when no outcome record exists
WIN_THRESHOLD_PCT   = 0.005  # +0.5 % = winner
LOSS_THRESHOLD_PCT  = -0.005 # -0.5 % = loser  (between = neutral, skip)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prompt_hash(prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except FileNotFoundError:
        pass
    return records


def _load_training_file(path: Path) -> list[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_training_file(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def _forward_price_change(symbol: str, entry_dt: datetime, hold_days: int) -> float | None:
    """Return % price change from entry_dt to entry_dt + hold_days trading days."""
    try:
        end = entry_dt + timedelta(days=hold_days * 2)   # buffer for weekends
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=entry_dt.date(), end=end.date())
        if len(hist) < 2:
            return None
        entry_price = hist['Close'].iloc[0]
        exit_price  = hist['Close'].iloc[min(hold_days, len(hist) - 1)]
        return float((exit_price - entry_price) / entry_price)
    except Exception as e:
        logging.warning(f"yfinance lookup failed for {symbol}: {e}")
        return None


def _label_from_pnl(pnl_pct: float, decision: str) -> str | None:
    """
    Map a P&L percentage to winner/loser given the direction of the trade.
    Returns None for results too close to zero to be meaningful.
    """
    if decision in ('buy', 'buy_call'):
        if pnl_pct >= WIN_THRESHOLD_PCT:
            return 'winner'
        if pnl_pct <= LOSS_THRESHOLD_PCT:
            return 'loser'
    elif decision in ('sell', 'buy_put'):
        # Short / put: profit when price falls
        if pnl_pct <= LOSS_THRESHOLD_PCT:  # price fell — correct call
            return 'winner'
        if pnl_pct >= WIN_THRESHOLD_PCT:   # price rose  — wrong call
            return 'loser'
    return None   # too close to call


def _build_ideal_output(decision: str, confidence: float, reasoning: str, label: str) -> str:
    """
    Construct the 'ideal' structured response the model should have produced.
    For losers we keep the decision/reasoning honest but note the outcome so
    the model learns from the mistake rather than learning a wrong answer.
    """
    action_map = {
        'buy':      'BUY',
        'sell':     'SELL',
        'hold':     'HOLD',
        'buy_call': 'BUY_CALL',
        'buy_put':  'BUY_PUT',
    }
    action_str = action_map.get(decision, decision.upper())

    if label == 'winner':
        outcome_note = 'This was a profitable trade.'
    elif label == 'loser':
        outcome_note = 'This trade resulted in a loss — review the signals more carefully.'
    else:
        outcome_note = ''

    parts = [
        f"Decision: {action_str}",
        f"Confidence: {confidence:.2f}",
        f"Reasoning: {reasoning}",
    ]
    if outcome_note:
        parts.append(f"Outcome: {outcome_note}")
    return '\n'.join(parts)


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_examples(
    decision_log: Path,
    outcomes_log: Path,
    bot: str,
) -> list[dict]:
    """
    Return a list of new training examples derived from decision_log.

    Each example follows the format expected by finetune_model.py:
        { "input": <prompt>, "output": <ideal response>, "label": <winner|loser>,
          "metadata": { ... } }
    """
    decisions = _load_jsonl(decision_log)
    outcomes  = _load_jsonl(outcomes_log)

    # Build a lookup: (symbol, date_str) → outcome record
    outcome_map: dict[tuple, dict] = {}
    for o in outcomes:
        try:
            dt  = datetime.fromisoformat(o['entry_timestamp'])
            key = (o['symbol'], dt.date().isoformat())
            outcome_map[key] = o
        except (KeyError, ValueError):
            pass

    examples = []
    for rec in decisions:
        # --- quality gate ---
        if rec.get('confidence', 0) < MIN_CONFIDENCE:
            continue
        if not rec.get('executed') and rec.get('decision') not in ('hold',):
            continue   # non-executed non-holds are noise

        decision  = rec.get('decision', 'hold')
        symbol    = rec.get('symbol', '')
        prompt    = rec.get('prompt', '')
        reasoning = rec.get('reasoning', '')
        confidence = float(rec.get('confidence', 0.5))

        if not prompt or not symbol:
            continue

        try:
            entry_dt = datetime.fromisoformat(rec['timestamp'])
        except (KeyError, ValueError):
            continue

        # --- determine label ---
        label    = None
        pnl_pct  = None

        # 1. Check outcome tracker record
        key = (symbol, entry_dt.date().isoformat())
        if key in outcome_map:
            pnl_pct = outcome_map[key].get('pnl_pct', 0)
            label   = 'winner' if outcome_map[key].get('won') else 'loser'

        # 2. Fallback: forward price via yfinance (only for executed trades)
        elif rec.get('executed') and decision != 'hold':
            days_elapsed = (datetime.now() - entry_dt).days
            if days_elapsed >= HOLD_DAYS_FALLBACK:
                pnl_pct = _forward_price_change(symbol, entry_dt, HOLD_DAYS_FALLBACK)
                if pnl_pct is not None:
                    label = _label_from_pnl(pnl_pct, decision)

        # 3. HOLDs: label as winner if price stayed within ±0.5% over hold window
        elif decision == 'hold':
            days_elapsed = (datetime.now() - entry_dt).days
            if days_elapsed >= HOLD_DAYS_FALLBACK:
                pnl_pct = _forward_price_change(symbol, entry_dt, HOLD_DAYS_FALLBACK)
                if pnl_pct is not None:
                    if abs(pnl_pct) < WIN_THRESHOLD_PCT:
                        label = 'winner'   # correctly stayed out
                    elif pnl_pct > WIN_THRESHOLD_PCT:
                        label = 'loser'    # missed a good buy

        if label is None:
            continue   # not enough data yet

        ideal_output = _build_ideal_output(decision, confidence, reasoning, label)

        examples.append({
            'input':    prompt,
            'output':   ideal_output,
            'label':    label,
            'metadata': {
                'bot':        bot,
                'symbol':     symbol,
                'decision':   decision,
                'confidence': round(confidence, 4),
                'pnl_pct':    round(pnl_pct, 6) if pnl_pct is not None else None,
                'entry_date': entry_dt.date().isoformat(),
                'session_id': rec.get('session_id', ''),
                'prompt_hash': _prompt_hash(prompt),
                'generated_at': datetime.now().isoformat(),
            },
        })

    return examples


def append_new_examples(training_path: Path, new_examples: list[dict]) -> tuple[int, int]:
    """
    Load existing training file, deduplicate by prompt hash, append new unique
    examples, save, and return (added_count, total_count).
    """
    existing    = _load_training_file(training_path)
    seen_hashes = {
        e.get('metadata', {}).get('prompt_hash')
        for e in existing
        if e.get('metadata', {}).get('prompt_hash')
    }

    added = 0
    for ex in new_examples:
        h = ex.get('metadata', {}).get('prompt_hash')
        if h and h in seen_hashes:
            continue
        existing.append(ex)
        if h:
            seen_hashes.add(h)
        added += 1

    _save_training_file(training_path, existing)
    return added, len(existing)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(bot: str) -> None:
    if bot in ('stock', 'both'):
        dec_log      = _LOGS_DIR / 'decision_log.jsonl'
        outcomes_log = _LOGS_DIR / 'trade_outcomes.jsonl'
        train_path   = _DATA_DIR / 'training_data.json'

        logging.info("📚 Building stock training examples...")
        examples = build_examples(dec_log, outcomes_log, bot='stock')
        added, total = append_new_examples(train_path, examples)
        logging.info(f"✅ Stock: +{added} new examples → {total} total  ({train_path})")

        winners = sum(1 for e in examples if e['label'] == 'winner')
        losers  = sum(1 for e in examples if e['label'] == 'loser')
        logging.info(f"   This run: {winners} winners, {losers} losers out of {len(examples)} candidates")

    if bot in ('options', 'both'):
        dec_log      = _LOGS_DIR / 'options_decision_log.jsonl'
        outcomes_log = _LOGS_DIR / 'trade_outcomes.jsonl'
        train_path   = _DATA_DIR / 'options_training_data.json'

        logging.info("📚 Building options training examples...")
        examples = build_examples(dec_log, outcomes_log, bot='options')
        added, total = append_new_examples(train_path, examples)
        logging.info(f"✅ Options: +{added} new examples → {total} total  ({train_path})")

        winners = sum(1 for e in examples if e['label'] == 'winner')
        losers  = sum(1 for e in examples if e['label'] == 'loser')
        logging.info(f"   This run: {winners} winners, {losers} losers out of {len(examples)} candidates")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build fine-tuning examples from live trade decisions')
    parser.add_argument('--bot', choices=['stock', 'options', 'both'], default='both',
                        help='Which bot to process (default: both)')
    args = parser.parse_args()
    run(args.bot)
