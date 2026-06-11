#!/usr/bin/env python3
"""
replay_eval.py — Held-out replay evaluation for the model promotion gate.

Evaluates the currently-loaded inference adapter by:
  1. Loading the decision log (JSONL)
  2. Filtering to held-out decisions (not in training_examples, within lookback window)
  3. Re-running each prompt through the active adapter
  4. Simulating trades with real price data
  5. Computing expectancy, max_dd, trade_count, and n_symbols

model_promoter.py calls this twice (candidate + incumbent) to compare.
The prompt_set_hash field in each result confirms both runs used the same prompts.

Usage:
    python3 scripts/training/replay_eval.py --log logs/decision_log.jsonl
    python3 scripts/training/replay_eval.py --adapter finetune/adapter_X --log ...
"""
import argparse
import hashlib
import json
import logging
import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _pathfix  # noqa: F401
import db as _db

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_SCRIPTS_DIR  = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
_EVAL_DIR     = _PROJECT_ROOT / 'logs' / 'eval'

_MIN_PROMPTS_NARROW = int(os.getenv('REPLAY_MIN_PROMPTS', '60'))
_MIN_PROMPTS_WIDE   = int(os.getenv('REPLAY_MIN_PROMPTS_FLOOR', '30'))
_MAX_REPLAY_PROMPTS = int(os.getenv('REPLAY_MAX_PROMPTS', '200'))
_STOP_LOSS          = float(os.getenv('REPLAY_STOP_LOSS', '-0.07'))
_TAKE_PROFIT        = float(os.getenv('REPLAY_TAKE_PROFIT', '0.15'))
_MIN_CONFIDENCE     = float(os.getenv('REPLAY_MIN_CONFIDENCE', '0.60'))


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _prompt_hash(prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()


def _load_held_out_decisions(
    log_path: Path,
    lookback_days: int,
    trained_hashes: set[str],
    bot: str = 'stock',
) -> list[dict]:
    """Load buy/buy_call decisions that are not in trained_hashes and within the lookback window."""
    cutoff  = datetime.now() - timedelta(days=lookback_days)
    records = []
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if bot and r.get('bot', 'stock') != bot:
                    continue
                if r.get('decision') not in ('buy', 'buy_call'):
                    continue

                try:
                    ts = datetime.fromisoformat(r['timestamp'])
                except (KeyError, ValueError):
                    continue
                if ts < cutoff:
                    continue

                prompt = r.get('prompt', '')
                if not prompt:
                    continue
                ph = _prompt_hash(prompt)
                if ph in trained_hashes:
                    continue

                records.append({**r, '_prompt_hash': ph})
    except FileNotFoundError:
        logging.warning("Decision log not found: %s", log_path)

    return records


def _prompt_set_hash(records: list[dict]) -> str:
    """Stable hash of the evaluated prompt set — used to verify candidate and incumbent
    were tested on the same prompts when the promoter compares their results."""
    hashes  = sorted(r['_prompt_hash'] for r in records)
    combined = '|'.join(hashes)
    return hashlib.md5(combined.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Inference + simulation helpers
# ---------------------------------------------------------------------------

def _rerun_decision(prompt: str, adapter_path: str | None = None) -> dict:
    """Re-run a saved prompt through the currently-loaded inference adapter.

    Returns {'decision': str, 'confidence': float}.  Falls back to HOLD on error
    so one bad inference never halts the whole eval run.
    """
    try:
        from model_inference_lora import get_trading_decision, parse_decision
        if adapter_path:
            os.environ['LORA_ADAPTER_PATH'] = adapter_path
        response = get_trading_decision(prompt)
        return parse_decision(response)
    except Exception as e:
        logging.debug("_rerun_decision failed: %s", e)
        return {'decision': 'hold', 'confidence': 0.5}


def _simulate_trade(symbol: str, entry_ts: str) -> dict | None:
    """Simulate a long trade from entry_ts forward using yfinance.

    Returns {'pnl_pct': float, 'exit_reason': str} or None on data failure.
    """
    try:
        import yfinance as yf
        entry_dt = datetime.fromisoformat(entry_ts)
        end_dt   = entry_dt + timedelta(days=90)
        hist = yf.Ticker(symbol).history(
            start=entry_dt.strftime('%Y-%m-%d'),
            end=end_dt.strftime('%Y-%m-%d'),
        )
        if hist.empty or len(hist) < 2:
            return None
        hist.columns = [c.lower() for c in hist.columns]

        entry_price = float(hist['close'].iloc[0])
        exit_price  = float(hist['close'].iloc[-1])
        exit_reason = 'end_of_period'

        for _, row in hist.iloc[1:].iterrows():
            price   = float(row['close'])
            pnl_pct = (price - entry_price) / entry_price
            if pnl_pct <= _STOP_LOSS:
                exit_price  = price
                exit_reason = 'stop_loss'
                break
            if pnl_pct >= _TAKE_PROFIT:
                exit_price  = price
                exit_reason = 'take_profit'
                break

        return {
            'pnl_pct':    (exit_price - entry_price) / entry_price,
            'exit_reason': exit_reason,
        }
    except Exception as e:
        logging.debug("_simulate_trade failed for %s: %s", symbol, e)
        return None


# ---------------------------------------------------------------------------
# Metric helpers (pure, no I/O — testable without mocking)
# ---------------------------------------------------------------------------

def compute_expectancy(pnl_pcts: list[float]) -> float:
    """Average per-trade return; 0.0 on empty list."""
    if not pnl_pcts:
        return 0.0
    return sum(pnl_pcts) / len(pnl_pcts)


def compute_max_dd(equity_curve: list[float]) -> float:
    """Max peak-to-trough drawdown as a positive fraction (0.12 = 12% DD)."""
    if len(equity_curve) < 2:
        return 0.0
    peak   = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def run_replay_eval(
    log_path: Path,
    bot: str = 'stock',
    lookback_days: int = 28,
    adapter_path: str | None = None,
    seed: int = 42,
) -> dict:
    """
    Evaluate the adapter by re-running held-out prompts and simulating trades.

    Returns a result dict with: expectancy, max_dd, trade_count, n_symbols,
    prompt_set_hash, seed, adapter_path, evaluated_at.
    Saves result to logs/eval/replay_eval_YYYYMMDD_HHMMSS.json.
    """
    trained_hashes = _db.get_existing_prompt_hashes()

    # Narrow window first; widen to 2× if < _MIN_PROMPTS_NARROW held-out records found
    records = _load_held_out_decisions(log_path, lookback_days, trained_hashes, bot)
    if len(records) < _MIN_PROMPTS_NARROW:
        wide = _load_held_out_decisions(log_path, lookback_days * 2, trained_hashes, bot)
        if len(wide) > len(records):
            records = wide
            logging.info("Widened lookback to %dd (found %d held-out prompts)",
                         lookback_days * 2, len(records))

    if len(records) < _MIN_PROMPTS_WIDE:
        logging.warning(
            "⚠️  Only %d held-out prompts found (need ≥ %d) — "
            "replay eval unreliable; smoke test will be the deciding gate",
            len(records), _MIN_PROMPTS_WIDE,
        )

    # Deterministic subsample so candidate and incumbent use the same subset
    rng = random.Random(seed)
    if len(records) > _MAX_REPLAY_PROMPTS:
        records = rng.sample(records, _MAX_REPLAY_PROMPTS)

    pset_hash = _prompt_set_hash(records)
    logging.info("🔁 Replay eval: %d held-out prompts (set hash %s, seed %d)",
                 len(records), pset_hash, seed)

    # Re-run prompts → decisions → trade simulations → metrics
    pnl_pcts:     list[float] = []
    equity        = 1.0
    equity_curve  = [equity]
    symbols:      set[str]   = set()

    for rec in records:
        prompt  = rec.get('prompt', '')
        symbol  = rec.get('symbol', '').upper()
        ts      = rec.get('timestamp', '')

        decision = _rerun_decision(prompt, adapter_path)
        if decision.get('decision') not in ('buy', 'buy_call'):
            continue
        if float(decision.get('confidence') or 0.0) < _MIN_CONFIDENCE:
            continue

        trade = _simulate_trade(symbol, ts)
        if trade is None:
            continue

        pnl = trade['pnl_pct']
        pnl_pcts.append(pnl)
        equity *= (1 + pnl)
        equity_curve.append(equity)
        symbols.add(symbol)

    expectancy = compute_expectancy(pnl_pcts)
    max_dd     = compute_max_dd(equity_curve)

    result = {
        'evaluated_at':   datetime.now().isoformat(),
        'adapter_path':   adapter_path,
        'bot':            bot,
        'lookback_days':  lookback_days,
        'seed':           seed,
        'prompt_set_hash': pset_hash,
        'n_candidates':   len(records),
        'trade_count':    len(pnl_pcts),
        'n_symbols':      len(symbols),
        'expectancy':     round(expectancy, 6),
        'max_dd':         round(max_dd, 6),
    }

    _EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _EVAL_DIR / f"replay_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(result, indent=2))
    logging.info("📋 Replay eval → %s  expectancy=%+.2f%%  max_dd=%.1f%%  trades=%d",
                 out_path.name, expectancy * 100, max_dd * 100, len(pnl_pcts))

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Replay-based model evaluation')
    parser.add_argument('--log', type=Path,
                        default=_PROJECT_ROOT / 'logs' / 'decision_log.jsonl',
                        help='Path to decision_log.jsonl')
    parser.add_argument('--bot', choices=['stock', 'options'], default='stock')
    parser.add_argument('--lookback', type=int, default=28,
                        help='Initial lookback window in days (widened to 2× if < 60 prompts)')
    parser.add_argument('--adapter', type=str, default=None,
                        help='Override LORA_ADAPTER_PATH for this eval run')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    result = run_replay_eval(
        log_path=args.log,
        bot=args.bot,
        lookback_days=args.lookback,
        adapter_path=args.adapter,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
