#!/usr/bin/env python3
"""
backfill_decision_reasoning.py — re-derive contaminated reasoning columns from
the preserved raw_response.

Problem this closes: `decisions.reasoning` is meant to hold reasoning text, but
in 29,934 of 37,975 rows (78.8%) it holds a whole nested block, usually with the
trade's outcome appended:

    Reasoning: Decision: BUY Confidence: 0.85 Reasoning: Strong momentum…
               Outcome: Small win (+4.2%). Reward signal: +0.0003

decision_parser.unwrap_reasoning now prevents this at write time, but every row
already stored stays contaminated, and training_data_builder wraps that text
straight into `ideal_output` — where it fails `_is_clean_ideal_output` and the
example is discarded. That is 1,953 of 8,667 training examples (22.5%), and it
includes the take-trade exemplars the cadence gate is short of.

`decisions.raw_response` is untouched by this script and is present on 100% of
rows, so the re-derivation is lossless, repeatable, and reversible from source.

TWO columns are rewritten, not one:

  decisions.reasoning        29,934 / 37,975 contaminated — feeds _build_ideal_output
  outcomes.entry_reasoning      522 /    661 contaminated — a SECOND copy, written
                             at outcome-close time from decisions.reasoning
                             (outcome_tracker.py) and read by
                             regenerate_synced_outcomes. Backfilling only the
                             first lets contamination flow back in later.

outcomes rows are repaired by joining back to the decision that produced them
(via buy_order_id → decisions.order_id) so the text comes from raw_response
rather than from another contaminated copy.

Rows whose re-derived reasoning is EMPTY are skipped, not blanked. Their
raw_response contains only outcome text with no decision at all (~795 rows).
They are already excluded downstream by _is_clean_ideal_output, so skipping
costs nothing, whereas writing an empty string introduces a new state whose
blast radius has not been mapped.

Matches sync_broker_exits.py / backfill_trade_log_source.py's dry-run/--execute
preview-gate pattern — --execute must NEVER run from automation; preview first,
human sign-off, then execute.

Usage (standalone):
    python3 scripts/backfill_decision_reasoning.py                  # dry-run preview
    python3 scripts/backfill_decision_reasoning.py --limit 100      # sample preview
    python3 scripts/backfill_decision_reasoning.py --execute        # apply — NEVER
                                                                     # run this from
                                                                     # automation
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pathfix  # noqa: F401

import db as _db
from decision_parser import unwrap_reasoning

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# A row needs repair if its stored reasoning still carries a nested decision
# block or an outcome trailer. Checked on the stored value, not the raw
# response, so an already-clean row is left alone and the run stays idempotent.
_CONTAMINATION_MARKERS = ('decision:', 'outcome:', 'reward signal')


def _is_contaminated(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in _CONTAMINATION_MARKERS)


def _clean_from_raw(raw_response: str) -> str:
    """Re-derive reasoning from a raw model response, mirroring the live parser.

    Deliberately routed through the same decision_parser helpers the agents use
    at write time, so a row repaired here is byte-identical to what the fixed
    parser would have stored in the first place.
    """
    from decision_parser import _REASONING_LINE

    if not raw_response:
        return ''
    match = _REASONING_LINE.search(raw_response)
    raw = match.group(1) if match else raw_response
    return unwrap_reasoning(raw).replace('\n', ' ').strip()[:200]


def backfill_reasoning(db_path: Path = None, limit: int = None,
                       dry_run: bool = True) -> dict:
    """Re-derive decisions.reasoning and outcomes.entry_reasoning from raw_response.

    Returns a summary dict; `report` is populated in both modes so preview and
    execute print the same table.
    """
    db_path = db_path or _db.DB_PATH
    summary = {
        'candidates': 0, 'rewritten': 0, 'skipped_empty': 0,
        'unchanged': 0, 'outcomes_rewritten': 0, 'errors': 0, 'report': [],
    }

    try:
        with _db.get_conn(db_path) as conn:
            sql = ("SELECT id, symbol, reasoning, raw_response FROM decisions "
                   "WHERE raw_response IS NOT NULL AND TRIM(raw_response) <> '' "
                   "ORDER BY id")
            rows = conn.execute(sql).fetchall()
    except Exception as e:
        logging.warning("backfill_decision_reasoning: could not read decisions: %s", e)
        summary['errors'] += 1
        return summary

    updates = []
    for row in rows:
        try:
            if not _is_contaminated(row['reasoning']):
                summary['unchanged'] += 1
                continue
            summary['candidates'] += 1
            cleaned = _clean_from_raw(row['raw_response'])
            if not cleaned:
                # raw_response held only outcome text — nothing to recover.
                summary['skipped_empty'] += 1
                continue
            if cleaned == (row['reasoning'] or ''):
                summary['unchanged'] += 1
                continue
            updates.append((row['id'], cleaned))
            if len(summary['report']) < 20:
                summary['report'].append({
                    'id': row['id'],
                    'symbol': row['symbol'] or '',
                    'before': (row['reasoning'] or '')[:70],
                    'after': cleaned[:70],
                })
            if limit and len(updates) >= limit:
                break
        except Exception as e:
            logging.warning("backfill_decision_reasoning: row id=%s failed: %s",
                            row['id'] if 'id' in row.keys() else '?', e)
            summary['errors'] += 1

    if not dry_run and updates:
        try:
            with _db.get_conn(db_path) as conn:
                conn.executemany("UPDATE decisions SET reasoning = ? WHERE id = ?",
                                 [(text, rid) for rid, text in updates])
            summary['rewritten'] = len(updates)
        except Exception as e:
            logging.warning("backfill_decision_reasoning: decisions write failed: %s", e)
            summary['errors'] += 1
            return summary
    elif dry_run:
        summary['rewritten'] = len(updates)

    summary['outcomes_rewritten'] = _backfill_outcomes(db_path, dry_run, summary)
    return summary


def _backfill_outcomes(db_path: Path, dry_run: bool, summary: dict) -> int:
    """Repair outcomes.entry_reasoning from the decision that produced each outcome.

    Joins outcomes.buy_order_id → decisions.order_id so the text is re-derived
    from raw_response rather than copied from another contaminated column.
    """
    try:
        with _db.get_conn(db_path) as conn:
            # LEFT JOIN, not JOIN: only 279 of 527 contaminated outcomes have a
            # buy_order_id matching a decisions.order_id (older rows predate
            # order_id being recorded on the decision). An inner join would
            # silently leave the other 248 contaminated.
            rows = conn.execute(
                "SELECT o.id AS oid, o.entry_reasoning AS entry_reasoning, "
                "       d.raw_response AS raw_response "
                "FROM outcomes o "
                "LEFT JOIN decisions d ON d.order_id = o.buy_order_id "
                "WHERE o.entry_reasoning IS NOT NULL"
            ).fetchall()
    except Exception as e:
        logging.warning("backfill_decision_reasoning: could not read outcomes: %s", e)
        summary['errors'] += 1
        return 0

    updates = []
    for row in rows:
        try:
            if not _is_contaminated(row['entry_reasoning']):
                continue
            # Prefer re-deriving from raw_response; it is the original source and
            # recovers text that a nested copy may have truncated. When no
            # decision row matches, unwrap the stored value in place — the same
            # algorithm, just applied to a second-hand copy.
            cleaned = _clean_from_raw(row['raw_response']) if row['raw_response'] else ''
            if not cleaned:
                cleaned = unwrap_reasoning(row['entry_reasoning']).replace('\n', ' ').strip()[:200]
            if cleaned and cleaned != (row['entry_reasoning'] or ''):
                updates.append((cleaned, row['oid']))
        except Exception as e:
            logging.warning("backfill_decision_reasoning: outcome id=%s failed: %s",
                            row['oid'], e)
            summary['errors'] += 1

    if not dry_run and updates:
        try:
            with _db.get_conn(db_path) as conn:
                conn.executemany(
                    "UPDATE outcomes SET entry_reasoning = ? WHERE id = ?", updates)
        except Exception as e:
            logging.warning("backfill_decision_reasoning: outcomes write failed: %s", e)
            summary['errors'] += 1
            return 0
    return len(updates)


def main():
    parser = argparse.ArgumentParser(
        description='Re-derive contaminated reasoning columns from raw_response')
    parser.add_argument('--db', type=Path, default=None,
                        help='SQLite path (default: the configured live DB)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Cap the number of decisions rows rewritten (staged runs)')
    parser.add_argument('--execute', action='store_true',
                        help='Apply the backfill. NEVER invoke this from automation — '
                             'preview with the default dry-run first and get explicit sign-off.')
    args = parser.parse_args()

    dry_run = not args.execute
    logging.info("🔄 backfill_decision_reasoning running in %s mode (limit=%s)",
                 'execute' if args.execute else 'dry-run', args.limit or 'none')

    summary = backfill_reasoning(db_path=args.db, limit=args.limit, dry_run=dry_run)

    print(f"\n{'Preview' if dry_run else 'Backfill'} report "
          f"({summary['candidates']} contaminated candidate(s), showing up to 20):\n")
    for entry in summary['report']:
        print(f"  id={entry['id']:<7} {entry['symbol']:<8}")
        print(f"     before: {entry['before']!r}")
        print(f"     after : {entry['after']!r}")

    print(
        f"\nSummary: candidates={summary['candidates']}  "
        f"rewritten={summary['rewritten']}  skipped_empty={summary['skipped_empty']}  "
        f"unchanged={summary['unchanged']}  "
        f"outcomes_rewritten={summary['outcomes_rewritten']}  errors={summary['errors']}"
    )
    if summary['skipped_empty']:
        print(
            f"\n⚠️  {summary['skipped_empty']} row(s) had no recoverable reasoning "
            f"(raw_response held only outcome text) — left untouched rather than blanked."
        )
    if dry_run:
        print("\nDry run — no writes made. Re-run with --execute to apply.")


if __name__ == '__main__':
    main()
