#!/usr/bin/env python3
"""
training_data_builder.py - ENRICHED VERSION

Converts live trading decisions + realized outcomes into high-quality labelled
fine-tuning examples. Now includes reward-aware tiered labels, risk-adjusted
scoring, refined loser reasoning, and portfolio context hints.

Run nightly before finetune_model.py.
"""

import json
import hashlib
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _pathfix  # noqa: F401

import yfinance as yf
import db as _db

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_PROJECT_ROOT           = Path(__file__).resolve().parent.parent.parent
_DATA_DIR               = _PROJECT_ROOT / 'finetune' / 'data'
_PORTFOLIO_CONSTRAINTS  = _PROJECT_ROOT / 'logs' / 'portfolio_constraints.json'

MIN_CONFIDENCE      = 0.70
HOLD_DAYS_FALLBACK  = 5

# New tiered thresholds
STRONG_WIN_PCT      = 0.30
WIN_PCT             = 0.08
WEAK_WIN_PCT        = 0.005
WEAK_LOSS_PCT       = -0.005
LOSS_PCT            = -0.12
STRONG_LOSS_PCT     = -0.30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prompt_hash(prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()


def _forward_price_change(symbol: str, entry_dt: datetime, hold_days: int) -> float | None:
    try:
        end = entry_dt + timedelta(days=hold_days * 2)
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=entry_dt.date(), end=end.date())
        if len(hist) < 2:
            return None
        entry_price = hist['Close'].iloc[0]
        exit_price = hist['Close'].iloc[min(hold_days, len(hist) - 1)]
        return float((exit_price - entry_price) / entry_price)
    except Exception as e:
        logging.warning(f"yfinance lookup failed for {symbol}: {e}")
        return None


def _tiered_label_from_pnl(pnl_pct: float, decision: str, position_size: float = 0.0) -> tuple[str, float]:
    """Return (label_tier, reward_score)"""
    # Base label
    if decision in ('buy', 'buy_call'):
        if pnl_pct >= STRONG_WIN_PCT:
            label = 'strong_winner'
        elif pnl_pct >= WIN_PCT:
            label = 'winner'
        elif pnl_pct >= WEAK_WIN_PCT:
            label = 'weak_winner'
        elif pnl_pct <= STRONG_LOSS_PCT:
            label = 'strong_loser'
        elif pnl_pct <= LOSS_PCT:
            label = 'loser'
        else:
            label = 'weak_loser'
    else:  # sell / buy_put
        if pnl_pct <= STRONG_LOSS_PCT:   # big loss on short = strong winner
            label = 'strong_winner'
        elif pnl_pct <= LOSS_PCT:
            label = 'winner'
        elif pnl_pct <= WEAK_LOSS_PCT:
            label = 'weak_winner'
        elif pnl_pct >= STRONG_WIN_PCT:  # big gain on short = strong loser
            label = 'strong_loser'
        elif pnl_pct >= WIN_PCT:
            label = 'loser'
        else:
            label = 'weak_loser'

    # Reward score: magnitude scaled by position size (encourages bigger wins, penalizes big losses more)
    reward = pnl_pct * (position_size or 0.03)   # default 3% position
    return label, round(reward, 6)


def _build_ideal_output(decision: str, confidence: float, reasoning: str,
                        label: str, pnl_pct: float | None, reward: float | None) -> str:
    action_map = {
        'buy': 'BUY', 'sell': 'SELL', 'hold': 'HOLD',
        'buy_call': 'BUY_CALL', 'buy_put': 'BUY_PUT',
    }
    action_str = action_map.get(decision, decision.upper())

    if label.startswith('strong_winner'):
        outcome_note = f"Excellent trade — strong winner (+{pnl_pct:+.1%})."
    elif label.startswith('winner'):
        outcome_note = f"Profitable trade ({pnl_pct:+.1%})."
    elif label == 'weak_winner':
        outcome_note = f"Small win ({pnl_pct:+.1%}) — room for improvement."
    elif label.startswith('strong_loser'):
        outcome_note = f"Significant loss ({pnl_pct:+.1%}) — review signals carefully."
    elif label.startswith('loser'):
        outcome_note = f"Loss ({pnl_pct:+.1%}) — avoid similar setups."
    else:
        _pnl_str = f"{pnl_pct:+.1%}" if pnl_pct is not None else "N/A"
        outcome_note = f"Minor loss or breakeven ({_pnl_str})."

    reward_str = f" Reward signal: {reward:+.4f}" if reward is not None else ""

    return (
        f"Decision: {action_str}\n"
        f"Confidence: {confidence:.2f}\n"
        f"Reasoning: {reasoning}\n"
        f"Outcome: {outcome_note}{reward_str}"
    )


# ---------------------------------------------------------------------------
# Core builder (enriched)
# ---------------------------------------------------------------------------

def build_and_store(bot: str) -> tuple[int, int]:
    _db.init_db()
    existing_hashes = _db.get_existing_prompt_hashes()
    decisions = _db.get_executed_decisions(bot=bot, min_confidence=MIN_CONFIDENCE)

    # Include high-confidence HOLDs
    with _db.get_conn() as conn:
        hold_rows = conn.execute(
            "SELECT * FROM decisions WHERE bot=? AND decision='hold' AND confidence>=? ORDER BY timestamp",
            [bot, MIN_CONFIDENCE]
        ).fetchall()
    hold_decisions = [dict(r) for r in hold_rows]

    all_candidates = decisions + hold_decisions
    added = 0

    for rec in all_candidates:
        decision = rec.get('decision', 'hold')
        symbol = rec.get('symbol', '')
        prompt = rec.get('prompt', '')
        reasoning = rec.get('reasoning', '')
        confidence = float(rec.get('confidence', 0.5))

        if not prompt or not symbol:
            continue

        ph = _prompt_hash(prompt)
        if ph in existing_hashes:
            continue

        try:
            entry_dt = datetime.fromisoformat(rec['timestamp'])
        except (KeyError, ValueError):
            continue

        # --- Determine tiered label + reward ---
        label = None
        pnl_pct = None
        reward = None
        position_size = 0.03  # default fallback

        # 1. Primary: SQLite outcomes
        outcome = _db.get_outcome_by_entry(symbol, entry_dt.date().isoformat())
        if outcome:
            pnl_pct = outcome.get('pnl_pct', 0)
            label, reward = _tiered_label_from_pnl(pnl_pct, decision, position_size)

        # 2. Fallback forward price (non-hold executed trades)
        elif rec.get('executed') and decision != 'hold':
            days_elapsed = (datetime.now() - entry_dt).days
            if days_elapsed >= HOLD_DAYS_FALLBACK:
                pnl_pct = _forward_price_change(symbol, entry_dt, HOLD_DAYS_FALLBACK)
                if pnl_pct is not None:
                    label, reward = _tiered_label_from_pnl(pnl_pct, decision, position_size)

        # 3. HOLD validation
        elif decision == 'hold':
            days_elapsed = (datetime.now() - entry_dt).days
            if days_elapsed >= HOLD_DAYS_FALLBACK:
                pnl_pct = _forward_price_change(symbol, entry_dt, HOLD_DAYS_FALLBACK)
                if pnl_pct is not None:
                    if abs(pnl_pct) < 0.008:
                        label = 'correct_hold'
                        reward = 0.0
                    elif pnl_pct > 0.01 and decision in ('buy', 'buy_call'):
                        label = 'missed_opportunity'
                        reward = -0.02  # penalize missed upside
                    else:
                        label = 'correct_hold'
                        reward = 0.01

        if label is None:
            continue

        example = {
            'input': prompt,
            'output': _build_ideal_output(decision, confidence, reasoning, label, pnl_pct, reward),
            'label': label,
            'metadata': {
                'bot': bot,
                'symbol': symbol,
                'decision': decision,
                'confidence': round(confidence, 4),
                'pnl_pct': round(pnl_pct, 6) if pnl_pct is not None else None,
                'reward': round(reward, 6) if reward is not None else None,
                'entry_date': entry_dt.date().isoformat(),
                'session_id': rec.get('session_id', ''),
                'prompt_hash': ph,
                'generated_at': datetime.now().isoformat(),
            },
        }

        if _db.insert_training_example(example):
            existing_hashes.add(ph)
            added += 1

    # Export
    all_examples = _db.get_training_examples(bot=bot)
    filename = 'training_data.json' if bot == 'stock' else 'options_training_data.json'
    output_path = _DATA_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exportable = [
        {
            'input': e['prompt'],
            'output': e['ideal_output'],
            'label': e['label'],
            'metadata': {
                'bot': e['bot'],
                'symbol': e['symbol'],
                'confidence': e['confidence'],
                'pnl_pct': e['pnl_pct'],
                'reward': e.get('reward'),
                'entry_date': e['entry_date'],
                'session_id': e['session_id'],
                'prompt_hash': e['prompt_hash'],
            }
        }
        for e in all_examples
    ]

    with open(output_path, 'w') as f:
        json.dump(exportable, f, indent=2)

    logging.info(f"✅ {bot.capitalize()}: +{added} new enriched examples → {len(all_examples)} total")
    return added, len(all_examples)


def build_portfolio_level_examples() -> int:
    """
    Generate training examples from portfolio-constraint veto records.
    Reads logs/portfolio_constraints.json and creates lesson pairs like:
      "You tried to buy X but were blocked — sector was 42% tech (cap 30%)"
    These teach the model to self-police concentration risk.
    Returns number of new examples inserted.
    """
    if not _PORTFOLIO_CONSTRAINTS.exists():
        logging.info("No portfolio_constraints.json — skipping portfolio lessons")
        return 0

    try:
        records = json.loads(_PORTFOLIO_CONSTRAINTS.read_text())
    except Exception as e:
        logging.warning(f"Could not read portfolio_constraints.json: {e}")
        return 0

    if not isinstance(records, list):
        records = [records]

    _db.init_db()
    existing_hashes = _db.get_existing_prompt_hashes()
    added = 0

    for rec in records:
        symbol  = rec.get('symbol', 'UNKNOWN')
        reason  = rec.get('reason', '')
        ts      = rec.get('timestamp', datetime.now().isoformat())
        sector  = rec.get('sector', '')
        sec_pct = rec.get('sector_pct')
        sec_cap = rec.get('sector_cap', 0.30)
        spend   = rec.get('spend_usd', 0)

        if not reason:
            continue

        # Build a natural-language prompt
        pct_str  = f"{sec_pct:.0%}" if sec_pct is not None else "unknown"
        cap_str  = f"{sec_cap:.0%}"
        pos_str  = f"${spend:,.0f}" if spend else "an amount"

        prompt = (
            f"Portfolio review — {ts[:10]}:\n"
            f"The agent attempted to buy {pos_str} of {symbol}.\n"
            f"Current portfolio concentration in {sector or 'this sector'}: {pct_str}.\n"
            f"Maximum allowed sector concentration: {cap_str}.\n"
            f"Veto reason: {reason}\n\n"
            f"Was this the right call to block? What should the agent learn from this?"
        )

        ph = _prompt_hash(prompt)
        if ph in existing_hashes:
            continue

        if sec_pct is not None and sec_pct > sec_cap:
            output = (
                f"Decision: HOLD\n"
                f"Confidence: 1.00\n"
                f"Reasoning: Correctly blocked — {sector} exposure was {pct_str} which exceeds "
                f"the {cap_str} sector cap. Adding {symbol} would have worsened concentration risk. "
                f"Next time, route capital to under-represented sectors or wait for rebalancing.\n"
                f"Outcome: Correct portfolio veto — no reward penalty."
            )
            label = 'correct_hold'
        else:
            output = (
                f"Decision: HOLD\n"
                f"Confidence: 1.00\n"
                f"Reasoning: Trade was blocked due to: {reason}. "
                f"Review position sizing and sector limits before retrying.\n"
                f"Outcome: Portfolio constraint enforced."
            )
            label = 'correct_hold'

        example = {
            'input': prompt,
            'output': output,
            'label': label,
            'metadata': {
                'bot': 'stock',
                'symbol': symbol,
                'decision': 'hold',
                'confidence': 1.0,
                'pnl_pct': None,
                'reward': 0.0,
                'entry_date': ts[:10],
                'session_id': '',
                'prompt_hash': ph,
                'generated_at': datetime.now().isoformat(),
            },
        }

        if _db.insert_training_example(example):
            existing_hashes.add(ph)
            added += 1

    logging.info(f"✅ Portfolio lessons: +{added} new examples from {len(records)} veto records")
    return added


def run(bot: str = 'both') -> None:
    if bot in ('stock', 'both'):
        build_and_store('stock')
    if bot in ('options', 'both'):
        build_and_store('options')
    build_portfolio_level_examples()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build enriched fine-tuning examples')
    parser.add_argument('--bot', choices=['stock', 'options', 'both'], default='both')
    args = parser.parse_args()
    run(args.bot)