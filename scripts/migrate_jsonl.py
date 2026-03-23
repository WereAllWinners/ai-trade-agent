#!/usr/bin/env python3
"""
migrate_jsonl.py — One-time backfill of existing JSONL logs into SQLite.

Safe to re-run: all inserts use INSERT OR IGNORE so nothing is duplicated.

Usage:
    python3 scripts/migrate_jsonl.py
"""
import json
import logging
from pathlib import Path

from db import init_db, insert_decision, insert_trade, insert_outcome, insert_training_example, DB_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOGS_DIR     = _PROJECT_ROOT / 'logs'
_DATA_DIR     = _PROJECT_ROOT / 'finetune' / 'data'


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        logging.warning("Not found, skipping: %s", path)
        return records
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logging.warning("Skipping malformed line %d in %s", i, path.name)
    return records


def migrate_decisions() -> None:
    for filename, bot in [('decision_log.jsonl', 'stock'), ('options_decision_log.jsonl', 'options')]:
        records = _load_jsonl(_LOGS_DIR / filename)
        inserted = 0
        for rec in records:
            rec.setdefault('bot', bot)
            try:
                insert_decision(rec)
                inserted += 1
            except Exception as e:
                logging.warning("decisions insert failed: %s", e)
        logging.info("decisions (%s): %d/%d migrated", bot, inserted, len(records))


def migrate_trades() -> None:
    for filename, bot in [('trade_log.jsonl', 'stock'), ('options_trade_log.jsonl', 'options')]:
        records = _load_jsonl(_LOGS_DIR / filename)
        inserted = 0
        for rec in records:
            try:
                insert_trade(rec, bot=bot)
                inserted += 1
            except Exception as e:
                logging.warning("trades insert failed: %s", e)
        logging.info("trades (%s): %d/%d migrated", bot, inserted, len(records))


def migrate_outcomes() -> None:
    records = _load_jsonl(_LOGS_DIR / 'trade_outcomes.jsonl')
    inserted = 0
    for rec in records:
        try:
            insert_outcome(rec)
            inserted += 1
        except Exception as e:
            logging.warning("outcomes insert failed: %s", e)
    logging.info("outcomes: %d/%d migrated", inserted, len(records))


def migrate_training_examples() -> None:
    for filename, bot in [
        ('training_data.json', 'stock'),
        ('options_training_data.json', 'options'),
        ('finance_tuning/training_data.json', 'stock'),
    ]:
        path = _DATA_DIR / filename
        if not path.exists():
            continue
        try:
            with open(path) as f:
                records = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logging.warning("Could not load %s: %s", filename, e)
            continue

        inserted = 0
        for rec in records:
            rec.setdefault('bot', bot)
            meta = rec.setdefault('metadata', {})
            meta.setdefault('bot', bot)
            # Compute prompt hash if missing
            if not meta.get('prompt_hash'):
                import hashlib
                prompt = rec.get('input', rec.get('prompt', ''))
                meta['prompt_hash'] = hashlib.md5(prompt.encode()).hexdigest()
            try:
                if insert_training_example(rec):
                    inserted += 1
            except Exception as e:
                logging.warning("training_examples insert failed: %s", e)
        logging.info("training_examples (%s): %d/%d migrated", filename, inserted, len(records))


def main() -> None:
    logging.info("Initialising database at %s", DB_PATH)
    init_db()

    logging.info("--- Migrating decisions ---")
    migrate_decisions()

    logging.info("--- Migrating trades ---")
    migrate_trades()

    logging.info("--- Migrating outcomes ---")
    migrate_outcomes()

    logging.info("--- Migrating training examples ---")
    migrate_training_examples()

    # Summary
    from db import get_conn
    with get_conn() as conn:
        for table in ('decisions', 'trades', 'outcomes', 'training_examples'):
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            logging.info("  %-22s %d rows", table, count)

    logging.info("Migration complete.")


if __name__ == '__main__':
    main()
