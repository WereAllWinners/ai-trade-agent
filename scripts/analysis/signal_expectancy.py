#!/usr/bin/env python3
"""
signal_expectancy.py — Per-signal EV (expected value) measurement.

Joins decisions.indicators JSON with outcomes.pnl_pct to compute, per
discovery signal type, the expected dollar return per dollar risked.

  EV = win_rate × avg_win_pct − (1 − win_rate) × |avg_loss_pct|

Results are written to logs/signal_expectancy.json and loaded by
stock_discovery.py to rank opportunities by signal EV rather than raw
signal count.

Usage:
    python3 scripts/analysis/signal_expectancy.py  # writes logs/signal_expectancy.json
    python3 scripts/analysis/signal_expectancy.py --min-samples 10

Scheduled: hooked into Saturday strategist run (weekend_strategist.py).
"""
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _pathfix  # noqa: F401
import db as _db

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_OUTPUT_PATH  = _PROJECT_ROOT / 'logs' / 'signal_expectancy.json'

_DEFAULT_MIN_SAMPLES = 15


def compute_signal_ev(
    bot: str = 'stock',
    min_samples: int = _DEFAULT_MIN_SAMPLES,
    db_path: Path | None = None,
) -> dict[str, float]:
    """Compute per-signal expected value from closed trades.

    Returns a dict mapping signal_name → ev (float).
    Signals with < min_samples closed-trade outcomes are excluded.
    Returns {} on any DB error so callers can fall back gracefully.

    EV formula:
        EV = win_rate × avg_win_pct − (1 − win_rate) × |avg_loss_pct|
    where pct values are fractions (e.g. 0.05 = 5%).
    """
    try:
        kw = {} if db_path is None else {'db_path': db_path}
        decisions = _db.get_executed_decisions(bot=bot, min_confidence=0.0, **kw)
        all_outcomes = _db.get_all_outcomes(bot=bot, **kw)
        outcomes_by_oid: dict[str, dict] = {
            o['buy_order_id']: dict(o)
            for o in all_outcomes
            if o.get('buy_order_id')
        }

        # Per-signal aggregation
        signal_trades: dict[str, list[float]] = defaultdict(list)

        for dec in decisions:
            dec = dict(dec)
            oid = dec.get('order_id')
            outcome = outcomes_by_oid.get(oid) if oid else None
            if outcome is None:
                continue

            pnl_pct = outcome.get('pnl_pct')
            if pnl_pct is None:
                continue
            pnl_pct = float(pnl_pct)

            indicators_raw = dec.get('indicators')
            if not indicators_raw:
                continue
            try:
                indicators = json.loads(indicators_raw) if isinstance(indicators_raw, str) else indicators_raw
                signals = indicators.get('signals') or []
                if isinstance(signals, str):
                    signals = [s.strip() for s in signals.split(',') if s.strip()]
            except (json.JSONDecodeError, AttributeError):
                signals = []

            for sig in signals:
                signal_trades[sig].append(pnl_pct)

        result: dict[str, float] = {}
        for signal, pnl_list in signal_trades.items():
            if len(pnl_list) < min_samples:
                continue
            wins  = [p for p in pnl_list if p > 0]
            losses = [p for p in pnl_list if p <= 0]
            wr    = len(wins) / len(pnl_list)
            avg_win  = sum(wins)  / len(wins)  if wins   else 0.0
            avg_loss = sum(losses) / len(losses) if losses else 0.0
            ev = wr * avg_win - (1 - wr) * abs(avg_loss)
            result[signal] = round(ev, 6)

        return result

    except Exception as e:
        logging.warning("signal_expectancy: DB query failed: %s", e)
        return {}


def write_signal_expectancy(
    bot: str = 'stock',
    min_samples: int = _DEFAULT_MIN_SAMPLES,
    output_path: Path = _OUTPUT_PATH,
) -> dict[str, float]:
    """Compute signal EV and write to output_path as JSON."""
    ev_map = compute_signal_ev(bot=bot, min_samples=min_samples)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        'computed_at': datetime.now().isoformat(),
        'bot': bot,
        'min_samples': min_samples,
        'signal_ev': ev_map,
    }, indent=2))
    logging.info("signal_expectancy: wrote %d signal EVs → %s", len(ev_map), output_path)
    return ev_map


def load_signal_ev(path: Path = _OUTPUT_PATH) -> dict[str, float]:
    """Load the most-recently-computed signal EV map from disk.

    Returns {} when the file does not exist or cannot be parsed, so callers
    can fall back to count-based ranking without raising.
    """
    try:
        data = json.loads(path.read_text())
        return data.get('signal_ev', {})
    except Exception:
        return {}


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compute per-signal expected value from trade history')
    parser.add_argument('--bot', choices=['stock', 'options'], default='stock')
    parser.add_argument('--min-samples', type=int, default=_DEFAULT_MIN_SAMPLES)
    args = parser.parse_args()

    ev_map = write_signal_expectancy(bot=args.bot, min_samples=args.min_samples)
    print(json.dumps({'signal_ev': ev_map}, indent=2))


if __name__ == '__main__':
    main()
