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
import re
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
STRONG_WIN_PCT      = 0.20  # lowered from 0.30; at 30% zero rows qualify; at 20% ~7 rows
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

# Labels whose ideal_output is a worthy imitation target for SFT.
# Excluded: weak_winner (borderline noise), missed_opportunity (teaches wrong hold),
# constraint_block (risk-plumbing, not a trading decision — capped separately),
# loser-family (used for DPO, not SFT imitation).
_SFT_IMITATION_LABELS: frozenset[str] = frozenset({
    'strong_winner',   # large positive outcome — rare but high-value exemplar
    'winner',          # solid positive outcome
    'correct_hold',    # SPY-excess-validated genuine hold (NOT constraint_block)
    'counterfactual',  # HOLD-teaching row derived from a loser prompt
})

# Loser-family rows are kept in the JSON for DPO pairing but excluded from SFT.
_DPO_LABELS: frozenset[str] = frozenset({'weak_loser', 'loser', 'strong_loser'})

# constraint_block is excluded from _SFT_IMITATION_LABELS to prevent 2,000+ near-
# duplicate portfolio-rejection rows from dominating the gradient (teaches HOLD
# boilerplate, not trading). A small semantically-deduplicated sample is exported
# so the model learns the constraint-rejection format without losing trading signal.
# Per-type cap: at most _CONSTRAINT_BLOCK_PER_TYPE_CAP examples per semantic type
# (e.g. "correlation_cap_breached", "sector_cap_breached") so all 2,293 correlation-
# cap variants don't crowd out other constraint lessons.
# Global ceiling: _CONSTRAINT_BLOCK_SFT_CAP guards against future constraint types.
_CONSTRAINT_BLOCK_PER_TYPE_CAP: int = int(os.getenv('TDB_CONSTRAINT_BLOCK_PER_TYPE_CAP', '4'))
_CONSTRAINT_BLOCK_SFT_CAP:      int = int(os.getenv('TDB_CONSTRAINT_BLOCK_SFT_CAP', '12'))

# Counterfactual cap for SFT: all 599 CFs stay in the DB and reach the DPO builder
# unchanged.  Only _COUNTERFACTUAL_SFT_CAP semantically-distinct examples enter the
# SFT JSON, selected by signal pattern so the 30 cover the variety of
# "looked tradeable but wasn't" rather than 30 identical MACD-negative HOLD lessons.
# Without this cap the Session-4 SFT set was 86% HOLD-teaching (599/695).
_COUNTERFACTUAL_SFT_CAP: int = int(os.getenv('TDB_COUNTERFACTUAL_SFT_CAP', '30'))

# SFT HOLD-teaching share tripwire.
# (counterfactual + correct_hold + constraint_block) / total_sft > threshold → WARNING.
# The threshold is intentionally below 70% so the Session-4 state (94%) would have
# failed immediately and Session 3 would have run before any fine-tune.
_SFT_MAX_HOLD_SHARE: float = float(os.getenv('TDB_SFT_MAX_HOLD_SHARE', '0.65'))

# All labels that teach "decline / hold" rather than "take the trade".
# Used by the tripwire and by Session 5's cadence gate.
_HOLD_TEACHING_LABELS: frozenset[str] = frozenset({
    'counterfactual', 'correct_hold', 'constraint_block',
})

# Single source of truth for the SFT training label set.
# Imported by fine_tune_llm.py (filter at load time) and used by the tripwire
# denominator so neither can silently drift from the other.  Two copies of the same
# label set is exactly how the denominator bug was introduced.
_SFT_TRAIN_LABELS: frozenset[str] = _SFT_IMITATION_LABELS | frozenset({'constraint_block'})

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


def _is_clean_ideal_output(s: str) -> bool:
    """Return True iff s is a valid 3-line Decision/Confidence/Reasoning block.

    Rejects:
    - Markdown code fences (```)
    - Duplicate Decision: lines (old ideal_output smuggled in as Reasoning)
    - Any line count other than exactly 3
    - Lines out of expected order
    """
    if not s or '```' in s:
        return False
    lines = s.splitlines()
    if len(lines) != 3:
        return False
    if s.lower().count('decision:') != 1:
        return False
    return (
        lines[0].lower().startswith('decision:')
        and lines[1].lower().startswith('confidence:')
        and lines[2].lower().startswith('reasoning:')
    )


_CONSTRAINT_BLOCK_MARKERS = (
    'blocked due to',
    'correctly blocked',
    'sector cap',
    'portfolio constraint',
    'review position sizing',
)


def _is_constraint_block(s: str) -> bool:
    """True iff s is a portfolio-constraint rejection, not a genuine trading decision.

    Used by the export guard (to cap constraint_block in SFT) and by Session 4
    regeneration to correctly re-label the 2,211 historical correct_hold rows that
    were actually portfolio-constraint blocks.
    """
    if not s:
        return False
    lower = s.lower()
    return any(marker in lower for marker in _CONSTRAINT_BLOCK_MARKERS)


def _constraint_semantic_key(ideal_output: str) -> str:
    """Return the semantic constraint TYPE for per-type deduplication.

    Maps all correlation-cap variants ("already has 3 positions", "has 5 positions")
    to the same key regardless of the varying count N.  New constraint types that
    don't match the known patterns fall back to a digit-normalized reasoning snippet
    so they still group correctly within their own type.
    """
    lower = (ideal_output or '').lower()
    if 'correlation cap' in lower:
        return 'correlation_cap_breached'
    if 'sector cap' in lower:
        return 'sector_cap_breached'
    if 'pdt' in lower:
        return 'pdt_limit'
    # Generic fallback: extract and normalize the Reasoning line
    for line in lower.splitlines():
        if line.startswith('reasoning:'):
            reason = line.split(':', 1)[1].strip()
            key = re.sub(r'\d+\.?\d*', 'N', reason)
            return re.sub(r'\s+', ' ', key).strip()[:80]
    key = re.sub(r'\d+\.?\d*', 'N', lower[:120])
    return re.sub(r'\s+', ' ', key).strip()


def _counterfactual_semantic_key(ideal_output: str) -> str:
    """Semantic key for SFT-dedup of counterfactual HOLD rows.

    Groups by the PRIMARY contrary signal cited in the reasoning, so the
    _COUNTERFACTUAL_SFT_CAP sample covers distinct "why not to enter" patterns
    rather than 30 identical "MACD negative" lessons.

    Pattern priority mirrors _build_counterfactual_output's signal ordering
    (MACD → RSI → MA → volume → fallback).
    """
    for line in (ideal_output or '').splitlines():
        if not line.lower().startswith('reasoning:'):
            continue
        r = line.split(':', 1)[1].strip().lower()
        if 'sufficient edge' in r or not r.startswith('entry risk'):
            return 'no_indicators'
        # Strip "entry risk not justified — " prefix
        body = r.split('—', 1)[-1].strip() if '—' in r else r
        # Take just the first signal (before the first ";")
        first = body.split(';')[0].strip()
        if 'macd negative' in first:
            return 'macd_neg'
        if 'macd positive' in first:
            return 'macd_pos'
        if 'rsi' in first and 'overbought' in first:
            return 'rsi_overbought'
        if 'rsi' in first and 'oversold' in first:
            return 'rsi_oversold'
        if 'below the moving average' in first:
            return 'price_below_ma'
        if 'above the moving average' in first:
            return 'price_above_ma'
        if 'volume ratio' in first:
            return 'volume_low'
        # Fallback: digit-normalised first 30 chars of the primary signal
        key = re.sub(r'[+-]?\d+\.?\d*', 'N', first[:30])
        return re.sub(r'\s+', '_', key.strip())[:30]
    return 'no_reasoning'


# ---------------------------------------------------------------------------
# Extraction helpers for regeneration (Session 4)
# ---------------------------------------------------------------------------

def _extract_decision(ideal_output: str) -> str:
    """Parse decision action from the first Decision: line of an ideal_output."""
    _amap = {'BUY': 'buy', 'SELL': 'sell', 'HOLD': 'hold',
             'BUY_CALL': 'buy_call', 'BUY_PUT': 'buy_put'}
    for line in (ideal_output or '').splitlines():
        if line.lower().startswith('decision:'):
            tok = line.split(':', 1)[1].strip().upper()
            return _amap.get(tok, 'hold')
    return 'hold'


def _extract_reasoning_clean(ideal_output: str) -> str:
    """Extract clean reasoning text from a contaminated ideal_output.

    Handles three contamination patterns found in historical rows:
    - 4-line format: 'Decision/Confidence/Reasoning/Outcome' → strip Outcome suffix
    - Dup-decision embedding: Reasoning contains a nested 'Decision: X Reasoning: <real>'
    - Triple/quadruple embedding: multiple nested levels → take rightmost Reasoning: segment

    In all cases the Outcome/Reward-signal trailer is stripped from the tail.
    """
    reasoning_line = ''
    for line in (ideal_output or '').splitlines():
        if line.lower().startswith('reasoning:'):
            reasoning_line = line.split(':', 1)[1].strip()
            break

    if not reasoning_line:
        return 'Analysis based on available market indicators.'

    # Unwrap all levels of "Decision: ... Reasoning: <actual>" embedding
    parts = re.split(r'(?i)reasoning:\s*', reasoning_line)
    # Take the last non-empty segment (rightmost = real reasoning after all embedding)
    actual = next((p.strip() for p in reversed(parts) if p.strip()), reasoning_line)

    # Strip trailing Outcome / Reward-signal line appended to the Reasoning field
    m = re.search(r'\s*\n?(?:outcome|reward\s+signal|result)\s*:', actual, re.IGNORECASE)
    if m:
        actual = actual[:m.start()].strip()

    return actual if actual else 'Analysis based on available market indicators.'


def _clean_constraint_ideal_output(ideal_output: str) -> str:
    """Return a 3-line version of a constraint-block ideal_output.

    Historical constraint rows have a 4th 'Outcome: ...' line appended.  Keep only
    Decision, Confidence, and Reasoning — strip Outcome and anything after it.
    """
    kept: list[str] = []
    for line in (ideal_output or '').splitlines():
        ll = line.lower().lstrip()
        if ll.startswith('outcome:') or ll.startswith('reward'):
            break
        kept.append(line)
    result = '\n'.join(kept[:3])
    return result if _is_clean_ideal_output(result) else ideal_output


# ---------------------------------------------------------------------------
# Compiled patterns for extracting indicator values from prompt text.
# Two MA formats are supported with opposite sign conventions; both are
# normalised internally to "price relative to MA" (positive = above).
_MACD_RE    = re.compile(r'MACD\s*:\s*([+-]?\d+\.?\d*)', re.IGNORECASE)
_RSI_RE     = re.compile(r'RSI\s*\(\d+\)\s*:\s*(\d+\.?\d*)', re.IGNORECASE)
_VOL_RE     = re.compile(r'Volume Ratio\s*:\s*(\d+\.?\d*)x', re.IGNORECASE)
# Format A: "50 MA: $110 (+5.2% from current)" — the pct is MA-relative-to-price;
# negate to get price-relative-to-MA.
_MA_FMT_A   = re.compile(
    r'\d+[- ](?:day )?MA\s*:.*?\(([+-]?\d+\.?\d*)%\s*from\s+(?:current|price)\)',
    re.IGNORECASE,
)
# Format B: "Price vs 50-day MA: +5.2%" — directly price-relative-to-MA.
_MA_FMT_B   = re.compile(r'Price vs \d+-?day MA\s*:\s*([+-]?\d+\.?\d*)%', re.IGNORECASE)


def _extract_indicators(prompt: str) -> dict:
    """Parse indicator values from a prompt string.

    Returns a dict with any subset of: macd, rsi, volume, ma_pct.
    ma_pct is normalised to price-relative-to-MA (positive = price above MA).
    """
    result: dict = {}
    m = _MACD_RE.search(prompt)
    if m:
        result['macd'] = float(m.group(1))
    m = _RSI_RE.search(prompt)
    if m:
        result['rsi'] = float(m.group(1))
    m = _VOL_RE.search(prompt)
    if m:
        result['volume'] = float(m.group(1))
    m = _MA_FMT_A.search(prompt)
    if m:
        result['ma_pct'] = -float(m.group(1))   # negate: MA above price → price below MA
    else:
        m = _MA_FMT_B.search(prompt)
        if m:
            result['ma_pct'] = float(m.group(1))
    return result


def _build_counterfactual_output(
    decision: str,
    prompt: str,
    confidence: float,
    calib_map: dict,
) -> str:
    """Forward-looking counterfactual HOLD target for loser-labelled decisions.

    Teaches the model that HOLD was the better call by citing ONLY indicator
    values visible in the prompt — no realized P&L, no reference to the trade taken.

    Confidence is derived from the calibration map (inverted win-rate of the
    original decision's confidence bin), not hardcoded.
    """
    # Derive HOLD confidence: if that bin lost often, we're highly confident HOLD was right
    conf_bin = _pick_bin(confidence)
    if conf_bin and conf_bin in calib_map:
        cf_conf = max(0.60, min(0.85, 1.0 - float(calib_map[conf_bin]['win_rate'])))
    else:
        cf_conf = 0.70

    indicators = _extract_indicators(prompt)
    is_long    = decision in ('buy', 'buy_call')

    # Select contrary signals in priority order
    contrary = []
    if 'macd' in indicators:
        v = indicators['macd']
        if is_long and v < 0:
            contrary.append(f'MACD negative at {v:+.2f}')
        elif not is_long and v > 0:
            contrary.append(f'MACD positive at {v:+.2f}')
    if 'rsi' in indicators:
        v = indicators['rsi']
        if is_long and v > 70:
            contrary.append(f'RSI at {v:.1f} (overbought)')
        elif not is_long and v < 30:
            contrary.append(f'RSI at {v:.1f} (oversold — expected bounce)')
    if 'ma_pct' in indicators:
        v = indicators['ma_pct']
        if is_long and v < 0:
            contrary.append(f'price {v:+.1f}% below the moving average')
        elif not is_long and v > 0:
            contrary.append(f'price {v:+.1f}% above the moving average')
    if 'volume' in indicators:
        v = indicators['volume']
        if v < 1.0:
            contrary.append(f'volume ratio {v:.1f}x (below average)')

    if contrary:
        signals   = '; '.join(contrary[:2])
        reasoning = f'Entry risk not justified — {signals}; waiting for a cleaner setup.'
    else:
        reasoning = ('Signals do not present a sufficient edge to justify '
                     'entry risk at this time.')

    return (
        f"Decision: HOLD\n"
        f"Confidence: {cf_conf:.2f}\n"
        f"Reasoning: {reasoning}"
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
                    'output': _build_counterfactual_output(decision, prompt, confidence, calib_map),
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

    # constraint_block and counterfactual are handled by separate capped pools below;
    # exclude them from the primary EXPORT_LABELS check so they can't slip through.
    _EXPORT_LABELS = (_SFT_IMITATION_LABELS | _DPO_LABELS) - {'counterfactual'}

    candidates = []
    _constraint_pool: list[dict] = []
    _cf_pool:         list[dict] = []   # all CFs available to DPO; SFT gets a capped sample
    _skipped_blank_label = 0
    _skipped_contaminated = 0
    _skipped_taxonomy = 0
    for e in all_examples:
        if not e.get('label'):
            _skipped_blank_label += 1
            continue
        if not _is_clean_ideal_output(e.get('ideal_output', '')):
            _skipped_contaminated += 1
            continue
        label = e['label']
        if label == 'constraint_block':
            _constraint_pool.append(e)
            continue
        if label == 'counterfactual':
            _cf_pool.append(e)
            continue
        if label not in _EXPORT_LABELS:
            _skipped_taxonomy += 1
            continue
        w = _weight_for(e)
        candidates.append({
            'input':  e['prompt'],
            'output': e['ideal_output'],
            'label':  label,
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

    if _skipped_blank_label:
        logging.warning("⚠️  SFT export: skipped %d rows with blank/null label", _skipped_blank_label)
    if _skipped_contaminated:
        logging.warning("⚠️  SFT export: skipped %d rows with contaminated ideal_output", _skipped_contaminated)
    if _skipped_taxonomy:
        logging.info("SFT export: skipped %d rows with taxonomy-excluded labels "
                     "(weak_winner, missed_opportunity, etc.)", _skipped_taxonomy)

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

    # Constraint-block semantic cap: per-semantic-type cap + global ceiling.
    # Step 0 found 2,293 blocks collapse to 2 types (correlation cap, sector cap).
    # String-only dedup would export 30 near-identical correlation-cap variants.
    # Semantic grouping caps each TYPE at _CONSTRAINT_BLOCK_PER_TYPE_CAP, giving
    # surface variation within a type without burning the SFT budget on one rule.
    _constraint_pool.sort(key=_weight_for, reverse=True)
    _cb_type_counts:  dict[str, int] = {}
    _cb_seen_strings: set[str]       = set()
    _cb_selected:     list[dict]     = []
    for e in _constraint_pool:
        if len(_cb_selected) >= _CONSTRAINT_BLOCK_SFT_CAP:
            break
        out = e.get('ideal_output', '')
        if out in _cb_seen_strings:
            continue
        sem_key = _constraint_semantic_key(out)
        if _cb_type_counts.get(sem_key, 0) >= _CONSTRAINT_BLOCK_PER_TYPE_CAP:
            continue
        _cb_seen_strings.add(out)
        _cb_type_counts[sem_key] = _cb_type_counts.get(sem_key, 0) + 1
        _cb_selected.append({
            'input':  e['prompt'],
            'output': out,
            'label':  e['label'],
            'weight': 1.0,
            'metadata': {
                'bot':          e['bot'],
                'symbol':       e['symbol'],
                'confidence':   e['confidence'],
                'pnl_pct':      e['pnl_pct'],
                'reward':       e.get('reward'),
                'entry_date':   e['entry_date'],
                'session_id':   e['session_id'],
                'prompt_hash':  e['prompt_hash'],
                'example_type': 'constraint_block',
            }
        })
    exportable.extend(_cb_selected)
    if _constraint_pool:
        logging.info(
            "SFT export: constraint_block %d/%d (types: %s, per_type_cap=%d, global_cap=%d)",
            len(_cb_selected), len(_constraint_pool),
            dict(_cb_type_counts), _CONSTRAINT_BLOCK_PER_TYPE_CAP, _CONSTRAINT_BLOCK_SFT_CAP,
        )

    # Counterfactual SFT cap: semantically-distinct sample only.
    # ALL _cf_pool rows remain available to the DPO builder (nothing is deleted).
    # We sort by recency weight so the 30 selected are recent AND diverse.
    _cf_pool.sort(key=_weight_for, reverse=True)
    _cf_seen_keys: set[str]   = set()
    _cf_selected:  list[dict] = []
    for e in _cf_pool:
        if len(_cf_selected) >= _COUNTERFACTUAL_SFT_CAP:
            break
        out     = e.get('ideal_output', '')
        sem_key = _counterfactual_semantic_key(out)
        if sem_key in _cf_seen_keys:
            continue
        _cf_seen_keys.add(sem_key)
        _cf_selected.append({
            'input':  e['prompt'],
            'output': out,
            'label':  e['label'],
            'weight': round(_weight_for(e), 6),
            'metadata': {
                'bot':          e['bot'],
                'symbol':       e['symbol'],
                'confidence':   e['confidence'],
                'pnl_pct':      e['pnl_pct'],
                'reward':       e.get('reward'),
                'entry_date':   e['entry_date'],
                'session_id':   e['session_id'],
                'prompt_hash':  e['prompt_hash'],
                'example_type': 'counterfactual',
            }
        })
    exportable.extend(_cf_selected)
    logging.info(
        "SFT export: counterfactual %d/%d (semantic keys: %s, cap=%d)",
        len(_cf_selected), len(_cf_pool), sorted(_cf_seen_keys), _COUNTERFACTUAL_SFT_CAP,
    )

    # ── HOLD-share tripwire ────────────────────────────────────────────────────
    # An SFT set where most examples teach HOLD/decline cannot build a useful
    # trading policy — it creates a do-nothing model. Log at WARNING and the
    # cadence gate (Session 5) will refuse to fine-tune on a failing SFT set.
    _hold_count = sum(1 for ex in exportable if ex.get('label') in _HOLD_TEACHING_LABELS)
    _total_sft  = sum(1 for ex in exportable if ex.get('label') in _SFT_TRAIN_LABELS)
    _hold_share = _hold_count / _total_sft if _total_sft else 0.0
    if _hold_share > _SFT_MAX_HOLD_SHARE:
        logging.warning(
            "⚠️  SFT TRIPWIRE FAIL: HOLD-teaching share %.1f%% (%d/%d) exceeds %.0f%% ceiling. "
            "SFT teaches do-nothing. Accumulate more take-trade examples before fine-tuning.",
            100 * _hold_share, _hold_count, _total_sft, 100 * _SFT_MAX_HOLD_SHARE,
        )
    else:
        logging.info(
            "  [tripwire] HOLD-share %.1f%% (%d/%d) — within %.0f%% ceiling ✓",
            100 * _hold_share, _hold_count, _total_sft, 100 * _SFT_MAX_HOLD_SHARE,
        )

    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    # SFT-only file: physically contains only _SFT_TRAIN_LABELS rows.
    # fine_tune_llm.py reads this file; its line-166 label filter is now belt-and-suspenders
    # (drops 0 rows in the normal case; a guard if the file is ever corrupt or misrouted).
    sft_only = [ex for ex in exportable if ex.get('label') in _SFT_TRAIN_LABELS]
    sft_filename = 'training_data_sft.json' if bot == 'stock' else 'options_training_data_sft.json'
    with open(_DATA_DIR / sft_filename, 'w') as f:
        json.dump(sft_only, f, indent=2)

    # Combined archive: all labels (loser-family included) — for debugging and DPO inspection.
    # DO NOT feed this file to fine_tune_llm.py or finetune_model.py.
    archive_filename = ('training_data_archive.json' if bot == 'stock'
                        else 'options_training_data_archive.json')
    with open(_DATA_DIR / archive_filename, 'w') as f:
        json.dump(exportable, f, indent=2)

    logging.info(
        "✅ %s: +%d new examples → %d SFT rows / %d archive rows (of %d total in DB)",
        bot.capitalize(), added, len(sft_only), len(exportable), len(all_examples),
    )
    return added, len(sft_only)


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
            label = 'constraint_block'
        else:
            output = (
                f"Decision: HOLD\n"
                f"Confidence: 1.00\n"
                f"Reasoning: Trade was blocked due to: {reason}. "
                f"Review position sizing and sector limits before retrying."
            )
            label = 'constraint_block'

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


def regenerate_counterfactuals(bot: str = 'both',
                               db_path: 'Path | str | None' = None) -> tuple[int, int]:
    """Purge stale counterfactual rows and regenerate them with the current format.

    Cannot simply DELETE + call build_and_store: the imitation hash for each loser
    prompt is already present, so build_and_store would skip the counterfactual block
    entirely. Instead this function queries loser imitation rows directly and builds
    counterfactual rows in a separate pass.

    db_path defaults to the live DB.  Pass a copy path for Session-4 dry-runs.

    Returns (deleted_count, inserted_count).
    """
    db_path = Path(db_path) if db_path else _db.DB_PATH
    _db.init_db(db_path)

    # Step 1: delete all counterfactual rows
    with _db.get_conn(db_path) as conn:
        deleted = conn.execute(
            "DELETE FROM training_examples WHERE label = 'counterfactual'"
        ).rowcount
    logging.info("🗑  Deleted %d stale counterfactual rows", deleted)

    # Step 2: query loser imitation rows to regenerate from
    _LOSER_LABELS = ('strong_loser', 'loser', 'weak_loser')
    loser_rows: list[dict] = []
    for lbl in _LOSER_LABELS:
        loser_rows.extend(_db.get_training_examples(label=lbl, db_path=db_path))

    if not loser_rows:
        logging.info("No loser rows found — nothing to regenerate")
        return deleted, 0

    inserted = 0
    for row in loser_rows:
        row_bot   = row.get('bot', '')
        if bot not in ('both', row_bot):
            continue

        prompt     = row.get('prompt', '')
        confidence = float(row.get('confidence') or 0.5)
        if not prompt:
            continue

        # decision is not a column in training_examples — parse from ideal_output
        _action_map = {'BUY_CALL': 'buy_call', 'BUY_PUT': 'buy_put',
                       'BUY': 'buy', 'SELL': 'sell', 'HOLD': 'hold'}
        decision = 'buy'
        for _line in (row.get('ideal_output', '') or '').splitlines():
            if _line.upper().startswith('DECISION:'):
                _tok = _line.split(':', 1)[1].strip().upper()
                decision = _action_map.get(_tok, 'buy')
                break

        calib_map = _db.get_calibration_map(row_bot, db_path=db_path)
        cf_ph     = _prompt_hash(prompt + '|counterfactual')

        cf_example = {
            'input':  prompt,
            'output': _build_counterfactual_output(decision, prompt, confidence, calib_map),
            'label':  'counterfactual',
            'metadata': {
                'bot':          row_bot,
                'source':       row.get('source', 'paper'),
                'symbol':       row.get('symbol', ''),
                'decision':     decision,
                'confidence':   round(confidence, 4),
                'pnl_pct':      row.get('pnl_pct'),
                'reward':       row.get('reward'),
                'entry_date':   row.get('entry_date', ''),
                'session_id':   row.get('session_id', ''),
                'prompt_hash':  cf_ph,
                'generated_at': datetime.now().isoformat(),
                'example_type': 'counterfactual',
            },
        }
        if _db.insert_training_example(cf_example, db_path=db_path):
            inserted += 1

    logging.info("✅ Regenerated %d counterfactual rows from %d loser rows",
                 inserted, len(loser_rows))
    return deleted, inserted


def regenerate_historical(db_path: 'Path | str', dry_run: bool = False) -> dict:
    """Regenerate contaminated historical rows using the current Session-2 taxonomy.

    Operates exclusively against db_path.  All writes run inside a single
    transaction; any exception triggers a full rollback so a partial pass cannot
    leave the DB half-rewritten.

    What this function does:
    - Selects every row where ideal_output fails _is_clean_ideal_output.
    - For rows whose ideal_output is constraint-block content (including the 2,262
      historical correct_hold rows): relabels to 'constraint_block' and strips the
      'Outcome:' trailer from the ideal_output.
    - For clean-format rows mislabeled as correct_hold but containing constraint
      content: relabels to 'constraint_block' (no ideal_output change needed).
    - For traded rows (BUY/SELL): recomputes label via _tiered_label_from_pnl with
      the current thresholds (STRONG_WIN_PCT=0.20, sell-side sign convention).
    - For genuine hold rows (HOLD, non-constraint): keeps label as correct_hold,
      rebuilds clean ideal_output by extracting reasoning from the contaminated string.
    - Leaves unrecoverable rows untouched (blank label; no pnl_pct and not a
      constraint block; rebuilt ideal_output still fails validation).
    - After writes, calls regenerate_counterfactuals to refresh counterfactual rows
      for all loser-family rows.

    dry_run=True computes and logs the projected distribution without any writes.

    Returns dict with keys: updated, skipped, label_changes, new_labels,
                            cf_deleted, cf_inserted (last two are 0 in dry-run).
    """
    from collections import Counter
    db_path = Path(db_path)
    _db.init_db(db_path)

    all_rows = _db.get_training_examples(db_path=db_path)
    logging.info("regenerate_historical: %d total rows in %s", len(all_rows), db_path)

    updates: list[tuple[int, str, str]] = []   # (id, new_label, new_ideal_output)
    skipped_ids:  list[int] = []
    skip_reasons: Counter   = Counter()
    label_changes: Counter  = Counter()         # (old_label, new_label) → count

    for row in all_rows:
        ideal    = row.get('ideal_output', '') or ''
        old_label = (row.get('label', '') or '').strip()
        row_id   = row['id']

        # ── Already clean: only need a label fix for clean constraint-block rows ─
        if _is_clean_ideal_output(ideal):
            if old_label == 'correct_hold' and _is_constraint_block(ideal):
                updates.append((row_id, 'constraint_block', ideal))
                label_changes[(old_label, 'constraint_block')] += 1
            # else: clean AND correctly labeled → leave untouched
            continue

        # ── No label → unrecoverable ─────────────────────────────────────────────
        if not old_label:
            skipped_ids.append(row_id)
            skip_reasons['blank_label'] += 1
            continue

        decision   = _extract_decision(ideal)
        confidence = float(row.get('confidence') or 0.75)

        # ── Constraint-block content ──────────────────────────────────────────────
        if _is_constraint_block(ideal):
            new_label = 'constraint_block'
            new_ideal = _clean_constraint_ideal_output(ideal)
            if not _is_clean_ideal_output(new_ideal):
                skipped_ids.append(row_id)
                skip_reasons['constraint_strip_failed'] += 1
                continue
            updates.append((row_id, new_label, new_ideal))
            label_changes[(old_label, new_label)] += 1
            continue

        # ── Genuine HOLD ─────────────────────────────────────────────────────────
        if decision == 'hold':
            reasoning = _extract_reasoning_clean(ideal)
            new_ideal = _build_ideal_output(decision, confidence, reasoning, old_label)
            if not _is_clean_ideal_output(new_ideal):
                skipped_ids.append(row_id)
                skip_reasons['hold_rebuild_failed'] += 1
                continue
            updates.append((row_id, old_label, new_ideal))
            label_changes[(old_label, old_label)] += 1
            continue

        # ── Traded rows (BUY / SELL / options) ───────────────────────────────────
        pnl_pct = row.get('pnl_pct')
        if pnl_pct is None:
            skipped_ids.append(row_id)
            skip_reasons['no_pnl_pct'] += 1
            continue

        new_label, _ = _tiered_label_from_pnl(float(pnl_pct), decision)
        reasoning     = _extract_reasoning_clean(ideal)
        new_ideal     = _build_ideal_output(decision, confidence, reasoning, new_label)
        if not _is_clean_ideal_output(new_ideal):
            skipped_ids.append(row_id)
            skip_reasons['traded_rebuild_failed'] += 1
            continue

        updates.append((row_id, new_label, new_ideal))
        label_changes[(old_label, new_label)] += 1

    new_label_dist = Counter(new_label for (_, new_label, _) in updates)
    # Include unchanged clean rows in distribution for completeness
    clean_unchanged = Counter(
        (row.get('label', '') or '').strip()
        for row in all_rows
        if _is_clean_ideal_output(row.get('ideal_output', '') or '')
        and not (
            (row.get('label', '') or '').strip() == 'correct_hold'
            and _is_constraint_block(row.get('ideal_output', '') or '')
        )
    )

    logging.info("regenerate_historical: %d rows to update, %d unrecoverable (skipped)",
                 len(updates), len(skipped_ids))
    logging.info("  Skip reasons: %s", dict(skip_reasons))
    logging.info("  Label changes (old→new): %s",
                 {f"{o}→{n}": c for (o, n), c in sorted(label_changes.items())})
    logging.info("  New label distribution (updated rows): %s", dict(new_label_dist))
    logging.info("  Unchanged clean rows: %s", dict(clean_unchanged))

    if dry_run:
        logging.info("DRY RUN — no writes performed against %s", db_path)
        return {
            'updated': len(updates),
            'skipped': len(skipped_ids),
            'skip_reasons': dict(skip_reasons),
            'label_changes': {f"{o}→{n}": c for (o, n), c in label_changes.items()},
            'new_labels': dict(new_label_dist),
            'unchanged_clean': dict(clean_unchanged),
            'cf_deleted': 0,
            'cf_inserted': 0,
        }

    # ── Write — single transaction, rollback on any exception ─────────────────
    try:
        with _db.get_conn(db_path) as conn:
            for row_id, new_label, new_ideal in updates:
                conn.execute(
                    "UPDATE training_examples SET label = ?, ideal_output = ? WHERE id = ?",
                    (new_label, new_ideal, row_id),
                )
        logging.info("✅ regenerate_historical: wrote %d updates to %s", len(updates), db_path)
    except Exception:
        logging.error("❌ regenerate_historical: transaction failed — DB left unchanged", exc_info=True)
        raise

    # ── Refresh counterfactuals for all loser-family rows ─────────────────────
    cf_deleted, cf_inserted = regenerate_counterfactuals('both', db_path=db_path)

    return {
        'updated': len(updates),
        'skipped': len(skipped_ids),
        'skip_reasons': dict(skip_reasons),
        'label_changes': {f"{o}→{n}": c for (o, n), c in label_changes.items()},
        'new_labels': dict(new_label_dist),
        'unchanged_clean': dict(clean_unchanged),
        'cf_deleted': cf_deleted,
        'cf_inserted': cf_inserted,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build enriched fine-tuning examples')
    parser.add_argument('--bot', choices=['stock', 'options', 'both'], default='both')
    parser.add_argument(
        '--regenerate-counterfactuals', action='store_true',
        help='Purge stale counterfactual rows and regenerate them (separate pass, no imitation hash collision)',
    )
    parser.add_argument(
        '--regenerate-historical', action='store_true',
        help='Regenerate contaminated historical rows with current Session-2 taxonomy (use with --db)',
    )
    parser.add_argument(
        '--db', default=None,
        help='DB path for --regenerate-historical (required; never defaults to live DB)',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='With --regenerate-historical: compute projected distribution without writing',
    )
    args = parser.parse_args()
    if args.regenerate_historical:
        if not args.db:
            parser.error('--regenerate-historical requires --db <path>')
        stats = regenerate_historical(args.db, dry_run=args.dry_run)
        logging.info("regenerate_historical complete: %s", stats)
    elif args.regenerate_counterfactuals:
        deleted, inserted = regenerate_counterfactuals(args.bot)
        logging.info("Regeneration complete: deleted=%d inserted=%d", deleted, inserted)
        run(args.bot)
    else:
        run(args.bot)