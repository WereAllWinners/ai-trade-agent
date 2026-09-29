#!/usr/bin/env python3
"""
data_quality_check.py — sprint03 E4: standing data-quality gate.

A pure read-only health check over trade_log.jsonl, the SQLite trading DB, and
the exported SFT training-corpus files — no broker calls, no LLM calls, so it
stays fast (<30s) enough to run as a hard precondition of every fine-tune,
not just a one-off audit. Follows risk_reconciler.reprotect_positions()'s
shape: a plain function that never raises on an expected-failure path, always
returns a flat structured result, with a thin CLI tail calling the same
function used programmatically.

Reuses (does not re-derive) the shared thresholds in training_constants.py
and _is_clean_ideal_output's contamination check from training_data_builder.py
— two copies of the same threshold/logic is exactly how the tripwire
denominator bug (Session 3) was introduced; don't repeat that.
"""
import argparse
import json
import logging
import os
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'training'))
import _pathfix  # noqa: F401
import db as _db
from training_constants import _SFT_MAX_HOLD_SHARE, _HOLD_TEACHING_LABELS, _MIN_TAKE_TRADE, _TAKE_TRADE_LABELS
from training_data_builder import _is_clean_ideal_output, _SFT_TRAIN_LABELS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_TRADE_LOG_PATH = _PROJECT_ROOT / 'logs' / 'trade_log.jsonl'
_DATA_DIR = _PROJECT_ROOT / 'finetune' / 'data'
_MACRO_CALENDAR_PATH = _PROJECT_ROOT / 'data' / 'macro_event_calendar.json'

_ORPHAN_WARN_PCT             = float(os.getenv('DQ_ORPHAN_WARN_PCT', '0.02'))
_ORPHAN_FAIL_PCT             = float(os.getenv('DQ_ORPHAN_FAIL_PCT', '0.05'))
_UNRECONCILED_TREND_FAIL_PCT = float(os.getenv('DQ_UNRECONCILED_TREND_FAIL_PCT', '0.20'))
_FALLBACK_WARN_PCT           = float(os.getenv('DQ_FALLBACK_WARN_PCT', '0.30'))
_FALLBACK_FAIL_PCT           = float(os.getenv('DQ_FALLBACK_FAIL_PCT', '0.60'))
_FALLBACK_MIN_TAGGED_N       = int(os.getenv('DQ_FALLBACK_MIN_TAGGED_N', '20'))
_PNL_SANE_MIN                = float(os.getenv('DQ_PNL_SANE_MIN', '-1.00'))
_PNL_SANE_MAX                = float(os.getenv('DQ_PNL_SANE_MAX', '5.00'))
_STOP_CLUSTER_WARN_PTS       = float(os.getenv('DQ_STOP_CLUSTER_WARN_PTS', '5.0'))
_UNRECONCILED_TREND_WINDOW_DAYS = int(os.getenv('DQ_UNRECONCILED_TREND_WINDOW_DAYS', '7'))
_CONTAMINATION_DIRTY_WARN_PCT = float(os.getenv('DQ_CONTAMINATION_DIRTY_WARN_PCT', '0.15'))
_CONTAMINATION_DIRTY_FAIL_PCT = float(os.getenv('DQ_CONTAMINATION_DIRTY_FAIL_PCT', '0.50'))


def _source_tagging_since() -> datetime:
    """The point after which every trade_log writer is expected to stamp
    `source` on every row. Derived from the latest mtime of the known
    writer files rather than a hand-maintained date constant that would go
    stale silently — if a writer fix ships again later, editing that file
    naturally pushes this cutoff forward too. Env override for tests/manual
    control (ISO datetime string).
    """
    override = os.getenv('DQ_SOURCE_TAGGING_SINCE')
    if override:
        return datetime.fromisoformat(override)
    writer_files = [
        _PROJECT_ROOT / 'scripts' / 'agents' / 'autonomous_agent.py',
        _PROJECT_ROOT / 'scripts' / 'agents' / 'options_agent.py',
        _PROJECT_ROOT / 'scripts' / 'sync_broker_exits.py',
    ]
    mtimes = [f.stat().st_mtime for f in writer_files if f.exists()]
    if not mtimes:
        return datetime.now() - timedelta(days=7)
    return datetime.fromtimestamp(max(mtimes))


def _check(name: str, status: str, detail: str, metric=None) -> dict:
    return {'name': name, 'status': status, 'detail': detail, 'metric': metric}


def _load_trade_log_rows(trade_log_path: Path, source: str | None) -> list:
    if not trade_log_path.exists():
        return []
    rows = []
    with open(trade_log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if source is not None and rec.get('source') != source:
                continue
            rows.append(rec)
    return rows


# ---------------------------------------------------------------------------
# Trade log layer
# ---------------------------------------------------------------------------

def _check_source_coverage(trade_log_path: Path, source: str | None) -> dict:
    """Rows written after every known trade_log writer was fixed to stamp
    `source` must all carry it — a post-cutoff sourceless row is a writer
    regression. Rows from before the cutoff are expected to be untagged
    (that's the accepted historical-data caveat, not a live bug) and are
    excluded entirely rather than being compared against a rolling window
    that would keep flagging the same already-fixed historical rows for a
    week after the fix landed."""
    if not trade_log_path.exists():
        return _check('source_coverage', 'pass', 'trade log does not exist yet', 1.0)

    all_rows = _load_trade_log_rows(trade_log_path, source=None)
    if source is not None:
        all_rows = [r for r in all_rows if r.get('source') == source or 'source' not in r]

    cutoff = _source_tagging_since()
    recent = []
    for r in all_rows:
        try:
            ts = datetime.fromisoformat(r.get('timestamp', ''))
        except (ValueError, TypeError):
            continue
        if ts >= cutoff:
            recent.append(r)

    if not recent:
        return _check('source_coverage', 'pass', 'no rows since the writer fix cutoff to check', 1.0)

    tagged = sum(1 for r in recent if 'source' in r)
    coverage = tagged / len(recent)
    if coverage < 1.0:
        missing = len(recent) - tagged
        return _check('source_coverage', 'fail',
                       f"{missing}/{len(recent)} rows since {cutoff.isoformat()} (writer fix cutoff) "
                       f"lack a source field — writer regression", coverage)
    return _check('source_coverage', 'pass', f"{len(recent)}/{len(recent)} recent rows tagged", coverage)


def _check_orphan_rate(trade_log_path: Path, source: str | None) -> dict:
    """Rows whose source is neither 'paper' nor 'live' (e.g. 'orphan' from a
    future E3-style marking) — warn/fail as this share grows."""
    rows = _load_trade_log_rows(trade_log_path, source=None)
    if source is not None:
        rows = [r for r in rows if r.get('source') == source or r.get('source') not in ('paper', 'live')]
    tagged = [r for r in rows if 'source' in r]
    if not tagged:
        return _check('orphan_rate', 'pass', 'no source-tagged rows yet', 0.0)
    orphaned = sum(1 for r in tagged if r.get('source') not in ('paper', 'live'))
    rate = orphaned / len(tagged)
    if rate > _ORPHAN_FAIL_PCT:
        return _check('orphan_rate', 'fail', f"{orphaned}/{len(tagged)} rows orphaned ({rate:.1%})", rate)
    if rate > _ORPHAN_WARN_PCT:
        return _check('orphan_rate', 'warn', f"{orphaned}/{len(tagged)} rows orphaned ({rate:.1%})", rate)
    return _check('orphan_rate', 'pass', f"{orphaned}/{len(tagged)} rows orphaned ({rate:.1%})", rate)


def _check_duplicate_order_ids(trade_log_path: Path, source: str | None) -> dict:
    """Same (order_id, action, source) appearing more than once — the JSONL
    append path isn't deduped, so this catches a double-logging regression."""
    rows = _load_trade_log_rows(trade_log_path, source=source)
    seen = {}
    dupes = []
    for r in rows:
        key = (r.get('order_id'), r.get('action'), r.get('source'))
        if key[0] is None:
            continue
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 2:
            dupes.append(key)
    if dupes:
        return _check('duplicate_order_ids', 'fail',
                       f"{len(dupes)} (order_id, action, source) combo(s) duplicated: "
                       f"{dupes[:5]}", len(dupes))
    return _check('duplicate_order_ids', 'pass', 'no duplicates found', 0)


# ---------------------------------------------------------------------------
# Outcome layer
# ---------------------------------------------------------------------------

def _check_unreconciled_trend(db_path: Path, source: str | None) -> dict:
    """Active (non-abandoned) unreconciled pairs *recently retried within the
    window* vs closed trades in the same window — both sides windowed the
    same way so this measures a current trend, not an all-time backlog
    against a short recent slice. A pre-E2.2 row that's never been retried
    since (last_attempt_at NULL or stale) is legacy debt, not evidence of an
    active, currently-growing problem — it's excluded here and will only
    start counting once a real pass retries it and refreshes last_attempt_at."""
    with _db.get_conn(db_path) as conn:
        src_clause = "AND source = :source" if source else ""
        cutoff = (datetime.now() - timedelta(days=_UNRECONCILED_TREND_WINDOW_DAYS)).isoformat()

        active = conn.execute(
            f"SELECT COUNT(*) AS n FROM unreconciled_orders "
            f"WHERE status != 'abandoned' AND last_attempt_at >= :cutoff {src_clause}",
            {'cutoff': cutoff, 'source': source} if source else {'cutoff': cutoff},
        ).fetchone()['n']

        closed = conn.execute(
            f"SELECT COUNT(*) AS n FROM outcomes WHERE exit_timestamp >= :cutoff {src_clause}",
            {'cutoff': cutoff, 'source': source} if source else {'cutoff': cutoff},
        ).fetchone()['n']

    if closed == 0:
        if active == 0:
            return _check('unreconciled_trend', 'pass', 'no closed trades or unreconciled pairs', 0.0)
        return _check('unreconciled_trend', 'warn',
                       f"{active} active unreconciled pair(s), no closed trades to compare against",
                       active)

    rate = active / closed
    if rate > _UNRECONCILED_TREND_FAIL_PCT:
        return _check('unreconciled_trend', 'fail',
                       f"{active} active unreconciled pairs vs {closed} closed trades ({rate:.1%})", rate)
    if active > 0:
        return _check('unreconciled_trend', 'warn',
                       f"{active} active unreconciled pairs vs {closed} closed trades ({rate:.1%})", rate)
    return _check('unreconciled_trend', 'pass', f"0 active unreconciled pairs vs {closed} closed trades", 0.0)


def _check_fallback_label_rate(db_path: Path, source: str | None) -> dict:
    """% of tagged training examples labeled via forward-price fallback
    rather than a realized outcome. label_source is NULL for legacy rows
    (untagged, excluded from both numerator and denominator — we don't know)."""
    with _db.get_conn(db_path) as conn:
        src_clause = "AND source = :source" if source else ""
        row = conn.execute(
            f"SELECT "
            f"  SUM(CASE WHEN label_source = 'fallback_forward_price' THEN 1 ELSE 0 END) AS fallback_n, "
            f"  COUNT(*) AS tagged_n "
            f"FROM training_examples WHERE label_source IS NOT NULL {src_clause}",
            {'source': source} if source else {},
        ).fetchone()

    tagged_n = row['tagged_n'] or 0
    fallback_n = row['fallback_n'] or 0
    if tagged_n < _FALLBACK_MIN_TAGGED_N:
        return _check('fallback_label_rate', 'pass',
                       f"insufficient tagged rows (n={tagged_n} < {_FALLBACK_MIN_TAGGED_N}) — "
                       f"check not yet meaningful", None)
    rate = fallback_n / tagged_n
    if rate > _FALLBACK_FAIL_PCT:
        return _check('fallback_label_rate', 'fail',
                       f"{fallback_n}/{tagged_n} tagged examples are fallback-labeled ({rate:.1%})", rate)
    if rate > _FALLBACK_WARN_PCT:
        return _check('fallback_label_rate', 'warn',
                       f"{fallback_n}/{tagged_n} tagged examples are fallback-labeled ({rate:.1%})", rate)
    return _check('fallback_label_rate', 'pass',
                   f"{fallback_n}/{tagged_n} tagged examples are fallback-labeled ({rate:.1%})", rate)


def _check_outcome_sanity(db_path: Path, source: str | None) -> dict:
    """Realized pnl_pct outside a sane range is an arithmetic/matching
    corruption canary. Stop-loss exits clustering far from the currently
    configured stop is a drift-between-config-and-reality warning."""
    with _db.get_conn(db_path) as conn:
        src_clause = "AND o.source = :source" if source else ""
        insane = conn.execute(
            f"SELECT COUNT(*) AS n FROM outcomes o "
            f"WHERE (o.pnl_pct < :lo OR o.pnl_pct > :hi) AND o.bot = 'stock' {src_clause}",
            {'lo': _PNL_SANE_MIN, 'hi': _PNL_SANE_MAX, 'source': source} if source
            else {'lo': _PNL_SANE_MIN, 'hi': _PNL_SANE_MAX},
        ).fetchone()['n']

    if insane:
        return _check('outcome_sanity', 'fail',
                       f"{insane} stock outcome(s) with pnl_pct outside "
                       f"[{_PNL_SANE_MIN:.0%}, {_PNL_SANE_MAX:.0%}]", insane)

    stop_loss_pct = float(os.getenv('STOP_LOSS_PCT', '-0.07'))
    tl_src_clause = "AND source = :source" if source else ""
    with _db.get_conn(db_path) as conn:
        rows = conn.execute(
            f"SELECT exit_pl_pct FROM trades WHERE exit_reason LIKE 'stop_loss%' "
            f"AND exit_pl_pct IS NOT NULL {tl_src_clause}",
            {'source': source} if source else {},
        ).fetchall()

    if not rows:
        return _check('outcome_sanity', 'pass', 'no stop-tagged exits to check clustering on', 0)

    diffs = [abs(r['exit_pl_pct'] - stop_loss_pct) for r in rows]
    median_diff = statistics.median(diffs)
    if median_diff > (_STOP_CLUSTER_WARN_PTS / 100):
        return _check('outcome_sanity', 'warn',
                       f"stop-tagged exits' median distance from configured stop "
                       f"({stop_loss_pct:.0%}) is {median_diff:.1%} — config/reality drift", median_diff)
    return _check('outcome_sanity', 'pass', 'pnl sane, stop clustering within bounds', median_diff)


# ---------------------------------------------------------------------------
# Training corpus layer
# ---------------------------------------------------------------------------

def _load_sft_export(bot: str) -> list:
    filename = 'training_data_sft.json' if bot == 'stock' else 'options_training_data_sft.json'
    path = _DATA_DIR / filename
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def _check_hold_share(bot: str = 'stock') -> dict:
    """Reuses training_data_builder's own tripwire constants/denominator
    logic exactly, against the actual exported SFT file — not a
    re-derivation, to avoid the exact denominator-drift bug class Session 3
    fixed. Not source-scoped — matches the real tripwire, which doesn't
    discriminate paper/live either."""
    exportable = _load_sft_export(bot)
    if not exportable:
        return _check('hold_share', 'pass', 'no exported SFT rows yet', None)
    hold_count = sum(1 for ex in exportable if ex.get('label') in _HOLD_TEACHING_LABELS)
    total_sft = sum(1 for ex in exportable if ex.get('label') in _SFT_TRAIN_LABELS)
    hold_share = hold_count / total_sft if total_sft else 0.0
    if hold_share > _SFT_MAX_HOLD_SHARE:
        return _check('hold_share', 'fail',
                       f"HOLD-teaching share {hold_share:.1%} ({hold_count}/{total_sft}) "
                       f"exceeds {_SFT_MAX_HOLD_SHARE:.0%} ceiling", hold_share)
    return _check('hold_share', 'pass',
                   f"HOLD-teaching share {hold_share:.1%} ({hold_count}/{total_sft})", hold_share)


def _check_label_distribution(bot: str = 'stock') -> dict:
    """Reuses finetune_model's own cadence-gate take-trade minimum. Also
    flags a degenerate corpus: a label tier at exactly 0 while its opposite
    tier is populated."""
    exportable = _load_sft_export(bot)
    if not exportable:
        return _check('label_distribution', 'pass', 'no exported SFT rows yet', None)
    take_trade_n = sum(1 for ex in exportable if ex.get('label') in _TAKE_TRADE_LABELS)
    if take_trade_n < _MIN_TAKE_TRADE:
        return _check('label_distribution', 'fail',
                       f"take-trade exemplars {take_trade_n} below minimum {_MIN_TAKE_TRADE}", take_trade_n)

    hold_n = sum(1 for ex in exportable if ex.get('label') in _HOLD_TEACHING_LABELS)
    if (take_trade_n == 0) != (hold_n == 0) and (take_trade_n + hold_n) > 0:
        return _check('label_distribution', 'warn',
                       f"degenerate corpus: take_trade={take_trade_n}, hold={hold_n} "
                       f"(one tier empty while the other is populated)", None)
    return _check('label_distribution', 'pass',
                   f"take_trade={take_trade_n} (>= {_MIN_TAKE_TRADE}), hold={hold_n}", take_trade_n)


def _check_contamination_canaries(db_path: Path) -> dict:
    """Structural ideal_output contamination (reuses
    training_data_builder._is_clean_ideal_output) + a hindsight-leak canary
    (an outcome's sell registering before its own buy is impossible and
    signals corruption).

    The structural check uses a tolerance band, not "any dirty row = fail":
    older rows from before the current 3-line Decision/Confidence/Reasoning
    convention was adopted legitimately don't match it — that's expected
    legacy format drift, not corruption. A rate spike above the tolerance
    (new generation producing malformed output) is what this actually
    guards against."""
    with _db.get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT ideal_output FROM training_examples WHERE ideal_output IS NOT NULL"
        ).fetchall()
        leaked = conn.execute(
            "SELECT COUNT(*) AS n FROM outcomes WHERE entry_timestamp > exit_timestamp"
        ).fetchone()['n']

    if leaked:
        return _check('contamination_canaries', 'fail',
                       f"{leaked} outcome(s) have entry_timestamp after exit_timestamp "
                       f"(hindsight-leak canary)", leaked)

    if rows:
        dirty = sum(1 for r in rows if not _is_clean_ideal_output(r['ideal_output']))
        dirty_rate = dirty / len(rows)
        if dirty_rate > _CONTAMINATION_DIRTY_FAIL_PCT:
            return _check('contamination_canaries', 'fail',
                           f"{dirty}/{len(rows)} ({dirty_rate:.1%}) training examples fail "
                           f"_is_clean_ideal_output — exceeds {_CONTAMINATION_DIRTY_FAIL_PCT:.0%} tolerance",
                           dirty_rate)
        if dirty_rate > _CONTAMINATION_DIRTY_WARN_PCT:
            return _check('contamination_canaries', 'warn',
                           f"{dirty}/{len(rows)} ({dirty_rate:.1%}) training examples fail "
                           f"_is_clean_ideal_output — within tolerance but check for a new source",
                           dirty_rate)
    return _check('contamination_canaries', 'pass', 'no contamination detected', 0)


def _check_macro_calendar_freshness() -> dict:
    """data/macro_event_calendar.json (sprint04 F1) must have future-dated
    coverage — this file is refreshed manually (annual cadence, see
    CLAUDE.md), not automatically, so a stale/missing/unparseable file is a
    silent-degradation risk the same way the dead Finnhub key was. Unlike
    check_calendar_staleness() (a same-day session warning), this is a
    hard training gate: missing/unparseable/empty, or <14 days of coverage
    left, fails and blocks the fine-tune."""
    calendar_path = _MACRO_CALENDAR_PATH
    try:
        with open(calendar_path) as f:
            data = json.load(f)
        events = data.get('events', [])
        dates = []
        for e in events:
            try:
                dates.append(datetime.strptime(e['date'], '%Y-%m-%d').date())
            except Exception:
                continue
        if not dates:
            return _check('macro_calendar_freshness', 'fail',
                           f"{calendar_path} has no parseable events", 0)
        days_left = (max(dates) - datetime.now().date()).days
    except Exception as e:
        return _check('macro_calendar_freshness', 'fail',
                       f"could not read {calendar_path}: {e}", None)

    if days_left < 14:
        return _check('macro_calendar_freshness', 'fail',
                       f"macro calendar coverage ends in {days_left} day(s) — "
                       f"refresh via scripts/tools/build_macro_calendar.py", days_left)
    if days_left < 60:
        return _check('macro_calendar_freshness', 'warn',
                       f"macro calendar coverage ends in {days_left} day(s) — "
                       f"refresh due soon", days_left)
    return _check('macro_calendar_freshness', 'pass',
                   f"macro calendar coverage ends in {days_left} day(s)", days_left)


def _check_synthetic_rows(db_path: Path) -> dict:
    """StrategyEvolver-origin rows (source='synthetic_backtest') still
    flowing despite STRATEGY_EVOLVER_ENABLED=false means old synthetic rows
    are leaking into the corpus after the kill switch was set."""
    with _db.get_conn(db_path) as conn:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM training_examples WHERE source = 'synthetic_backtest'"
        ).fetchone()['n']

    evolver_enabled = os.getenv('STRATEGY_EVOLVER_ENABLED', 'true').lower() != 'false'
    if not evolver_enabled and n > 0:
        return _check('synthetic_rows', 'warn',
                       f"{n} synthetic_backtest row(s) present while "
                       f"STRATEGY_EVOLVER_ENABLED=false", n)
    return _check('synthetic_rows', 'pass', f"{n} synthetic_backtest row(s)", n)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_data_quality_checks(db_path: Path | None = None,
                             trade_log_path: Path | None = None,
                             source: str | None = None,
                             bot: str = 'stock') -> dict:
    """Read-only. No broker calls, no LLM calls. Must complete <30s even on a
    10x-scaled fixture.

    Returns {'passed': bool, 'checks': [{'name', 'status': 'pass'|'warn'|'fail',
             'detail': str, 'metric': float|int|None}], 'summary': str}
    """
    db_path = db_path or _db.DB_PATH
    trade_log_path = trade_log_path or _DEFAULT_TRADE_LOG_PATH

    checks = [
        _check_source_coverage(trade_log_path, source),
        _check_orphan_rate(trade_log_path, source),
        _check_duplicate_order_ids(trade_log_path, source),
        _check_unreconciled_trend(db_path, source),
        _check_fallback_label_rate(db_path, source),
        _check_outcome_sanity(db_path, source),
        _check_hold_share(bot),
        _check_label_distribution(bot),
        _check_contamination_canaries(db_path),
        _check_synthetic_rows(db_path),
        _check_macro_calendar_freshness(),
    ]

    failing = [c['name'] for c in checks if c['status'] == 'fail']
    warning = [c['name'] for c in checks if c['status'] == 'warn']
    passed = not failing

    if failing:
        summary = f"FAILED: {', '.join(failing)}" + (f" (warnings: {', '.join(warning)})" if warning else '')
    elif warning:
        summary = f"PASSED with warnings: {', '.join(warning)}"
    else:
        summary = "PASSED: all checks clean"

    return {'passed': passed, 'checks': checks, 'summary': summary}


def main():
    parser = argparse.ArgumentParser(description='Standing data-quality gate for the training pipeline')
    parser.add_argument('--source', choices=['paper', 'live'], default=None)
    parser.add_argument('--bot', choices=['stock', 'options'], default='stock')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    result = run_data_quality_checks(source=args.source, bot=args.bot)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{result['summary']}\n")
        for c in result['checks']:
            metric_str = f" (metric={c['metric']})" if c['metric'] is not None else ''
            print(f"  [{c['status'].upper():5s}] {c['name']}: {c['detail']}{metric_str}")

    sys.exit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
