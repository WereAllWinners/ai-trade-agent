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

import os
import random
_MISSED_OPP_EXCESS   = float(os.getenv('TDB_MISSED_OPP_EXCESS', '0.03'))
_HALF_LIFE_DAYS      = float(os.getenv('TDB_HALF_LIFE_DAYS', '45'))
_MAX_EXAMPLES        = int(os.getenv('TDB_MAX_EXAMPLES', '5000'))

# Module-level SPY cache so repeated HOLD lookups don't re-fetch the same window
_SPY_RETURN_CACHE: dict = {}


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


def _forward_excess_return(symbol: str, entry_dt: datetime,
                           hold_days: int) -> tuple[float | None, float | None]:
    """Return (stock_return, excess_vs_spy) for the hold window, or (None, None) on failure.

    SPY returns are cached per (entry_date, hold_days) to avoid repeated network calls
    when many HOLD decisions share the same window.
    """
    stock_pct = _forward_price_change(symbol, entry_dt, hold_days)
    cache_key = (entry_dt.date().isoformat(), hold_days)
    if cache_key not in _SPY_RETURN_CACHE:
        _SPY_RETURN_CACHE[cache_key] = _forward_price_change('SPY', entry_dt, hold_days)
    spy_pct = _SPY_RETURN_CACHE[cache_key]
    if stock_pct is None or spy_pct is None:
        return None, None
    return stock_pct, float(stock_pct - spy_pct)


def _pick_bin(confidence: float) -> str | None:
    """Map a confidence float to the calibration bin label used in weekly_report.py."""
    if confidence < 0.60:
        return None
    elif confidence < 0.70:
        return '0.60-0.70'
    elif confidence < 0.80:
        return '0.70-0.80'
    elif confidence < 0.90:
        return '0.80-0.90'
    else:
        return '0.90-1.00'


def _example_weight(entry_dt: datetime) -> float:
    """Recency weight: exponential decay with half-life _HALF_LIFE_DAYS. Floor 0.05."""
    age_days = (datetime.now() - entry_dt).days
    return max(0.05, 0.5 ** (age_days / _HALF_LIFE_DAYS))


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
                        label: str, pnl_pct: float | None = None,
                        reward: float | None = None) -> str:
    """Build the SFT target string.

    Format is strictly Decision / Confidence / Reasoning — no Outcome or Reward
    lines. Those belonged to the prompt context the model cannot see at inference
    time and were teaching hallucination rather than forward-looking analysis.
    """
    action_map = {
        'buy': 'BUY', 'sell': 'SELL', 'hold': 'HOLD',
        'buy_call': 'BUY_CALL', 'buy_put': 'BUY_PUT',
    }
    action_str = action_map.get(decision, decision.upper())
    return (
        f"Decision: {action_str}\n"
        f"Confidence: {confidence:.2f}\n"
        f"Reasoning: {reasoning}"
    )


def _build_counterfactual_output(decision: str, pnl_pct: float | None) -> str:
    """Counterfactual SFT target for loser-labeled decisions.

    Teaches the model that HOLD was the better call for this prompt context.
    The output format is identical to imitation targets so the SFT loss treats
    them uniformly; only the decision and reasoning differ.
    """
    pnl_str = f"{pnl_pct:+.1%}" if pnl_pct is not None else "a loss"
    return (
        f"Decision: HOLD\n"
        f"Confidence: 0.90\n"
        f"Reasoning: The original {decision.upper()} resulted in {pnl_str}; "
        f"indicators did not justify the entry risk at this time."
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

    # A5: calibration map built once per run; empty dict → fall back to raw confidence
    calib_map = _db.get_calibration_map(bot)

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

        # 1. Primary: look up by order_id when available (A6), fall back to (symbol, date)
        order_id = rec.get('order_id')
        outcome = None
        if order_id:
            outcome = _db.get_outcome_by_order_id(order_id)
        if outcome is None:
            outcome = _db.get_outcome_by_entry(symbol, entry_dt.date().isoformat())
            if outcome is None and order_id:
                logging.debug("outcome lookup fallback used for symbol=%s date=%s", symbol, entry_dt.date())

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

        # 3. HOLD validation — A1: use SPY-excess-return threshold, not absolute price change
        elif decision == 'hold':
            days_elapsed = (datetime.now() - entry_dt).days
            if days_elapsed >= HOLD_DAYS_FALLBACK:
                pnl_pct, excess = _forward_excess_return(symbol, entry_dt, HOLD_DAYS_FALLBACK)
                if excess is not None:
                    if excess >= _MISSED_OPP_EXCESS:
                        label = 'missed_opportunity'
                        reward = -0.02
                    elif abs(excess) < 0.008:
                        label = 'correct_hold'
                        reward = 0.0
                    else:
                        label = 'correct_hold'
                        reward = 0.01

        if label is None:
            continue

        # A5: replace cloned confidence with calibrated win-rate for that bin
        conf_bin = _pick_bin(confidence)
        if conf_bin and conf_bin in calib_map:
            calibrated_conf = float(calib_map[conf_bin]['win_rate'])
            calibrated_conf = max(0.05, min(0.95, calibrated_conf))
        else:
            calibrated_conf = confidence

        example = {
            'input': prompt,
            'output': _build_ideal_output(decision, calibrated_conf, reasoning, label),
            'label': label,
            'metadata': {
                'bot': bot,
                'source': rec.get('source', 'paper'),
                'symbol': symbol,
                'decision': decision,
                'confidence': round(calibrated_conf, 4),
                'pnl_pct': round(pnl_pct, 6) if pnl_pct is not None else None,
                'reward': round(reward, 6) if reward is not None else None,
                'entry_date': entry_dt.date().isoformat(),
                'session_id': rec.get('session_id', ''),
                'prompt_hash': ph,
                'generated_at': datetime.now().isoformat(),
                'example_type': 'imitation',
            },
        }

        if _db.insert_training_example(example):
            existing_hashes.add(ph)
            added += 1

        # A2: for losers, emit a second counterfactual row teaching HOLD
        if label in {'strong_loser', 'loser', 'weak_loser'}:
            cf_ph = _prompt_hash(prompt + '|counterfactual')
            if cf_ph not in existing_hashes:
                cf_example = {
                    'input': prompt,
                    'output': _build_counterfactual_output(decision, pnl_pct),
                    'label': 'counterfactual',
                    'metadata': {
                        'bot': bot,
                        'source': rec.get('source', 'paper'),
                        'symbol': symbol,
                        'decision': decision,
                        'confidence': round(calibrated_conf, 4),
                        'pnl_pct': round(pnl_pct, 6) if pnl_pct is not None else None,
                        'reward': round(reward, 6) if reward is not None else None,
                        'entry_date': entry_dt.date().isoformat(),
                        'session_id': rec.get('session_id', ''),
                        'prompt_hash': cf_ph,
                        'generated_at': datetime.now().isoformat(),
                        'example_type': 'counterfactual',
                    },
                }
                if _db.insert_training_example(cf_example):
                    existing_hashes.add(cf_ph)
                    added += 1

    # Export with probabilistic recency filtering (A7)
    # Each example is included with probability = its recency weight, so fresh examples
    # (~weight≈1.0) are almost always included and old examples (~weight≈0.05) are
    # included only 5% of the time.  After stochastic filtering, subsample without
    # replacement to at most TDB_MAX_EXAMPLES.
    all_examples = _db.get_training_examples(bot=bot)

    def _weight_for(ex: dict) -> float:
        try:
            entry_dt = datetime.fromisoformat(ex.get('entry_date') or '2020-01-01')
        except (ValueError, TypeError):
            return 0.05
        return _example_weight(entry_dt)

    _EXPORT_SEED = int(os.getenv('TDB_EXPORT_SEED', '42'))
    _export_rng  = random.Random(_EXPORT_SEED)

    candidates = []
    for e in all_examples:
        w = _weight_for(e)
        candidates.append({
            'input':  e['prompt'],
            'output': e['ideal_output'],
            'label':  e['label'],
            'weight': round(w, 6),
            'metadata': {
                'bot':          e['bot'],
                'symbol':       e['symbol'],
                'confidence':   e['confidence'],
                'pnl_pct':      e['pnl_pct'],
                'reward':       e.get('reward'),
                'entry_date':   e['entry_date'],
                'session_id':   e['session_id'],
                'prompt_hash':  e['prompt_hash'],
                'example_type': e.get('example_type', 'imitation'),
            }
        })

    # Probabilistic inclusion: each example retained with probability = weight
    exportable = [ex for ex in candidates if _export_rng.random() < ex['weight']]

    # Subsample without replacement if still over the cap
    if len(exportable) > _MAX_EXAMPLES:
        exportable = _export_rng.sample(exportable, _MAX_EXAMPLES)

    # Log retention by age bucket for observability
    _now = datetime.now()
    _age_buckets: dict[str, list[int]] = {
        '<7d': [0, 0], '7-30d': [0, 0], '31-90d': [0, 0], '>90d': [0, 0],
    }
    for ex in candidates:
        try:
            age_days = (_now - datetime.fromisoformat(ex['metadata']['entry_date'])).days
        except (ValueError, TypeError, KeyError):
            age_days = 9999
        if   age_days <  7:  bkt = '<7d'
        elif age_days < 30:  bkt = '7-30d'
        elif age_days < 90:  bkt = '31-90d'
        else:                bkt = '>90d'
        _age_buckets[bkt][1] += 1  # total
    for ex in exportable:
        try:
            age_days = (_now - datetime.fromisoformat(ex['metadata']['entry_date'])).days
        except (ValueError, TypeError, KeyError):
            age_days = 9999
        if   age_days <  7:  bkt = '<7d'
        elif age_days < 30:  bkt = '7-30d'
        elif age_days < 90:  bkt = '31-90d'
        else:                bkt = '>90d'
        _age_buckets[bkt][0] += 1  # retained
    for bkt, (kept, total) in _age_buckets.items():
        if total > 0:
            logging.info("  [export] %s: %d / %d retained (%.0f%%)", bkt, kept, total,
                         100 * kept / total)

    filename = 'training_data.json' if bot == 'stock' else 'options_training_data.json'
    output_path = _DATA_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(exportable, f, indent=2)

    logging.info(
        "✅ %s: +%d new examples → %d exported (probabilistic, of %d total)",
        bot.capitalize(), added, len(exportable), len(all_examples),
    )
    return added, len(exportable)


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
                f"Next time, route capital to under-represented sectors or wait for rebalancing."
            )
            label = 'correct_hold'
        else:
            output = (
                f"Decision: HOLD\n"
                f"Confidence: 1.00\n"
                f"Reasoning: Trade was blocked due to: {reason}. "
                f"Review position sizing and sector limits before retrying."
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