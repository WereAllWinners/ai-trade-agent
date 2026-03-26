#!/usr/bin/env python3
"""
db.py — SQLite database layer for the AI trading agent.

Single source of truth for schema creation, inserts, and queries.
All other scripts import from here rather than touching the DB directly.

Database location: <project_root>/logs/trading.db
"""
import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

try:
    import numpy as np
    def _json_default(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
except ImportError:
    def _json_default(obj):
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = _PROJECT_ROOT / 'logs' / 'trading.db'


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

@contextmanager
def get_conn(db_path: Path = DB_PATH):
    """Context manager that yields a connection with WAL mode and row factory."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # safe for concurrent daemon writes
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
-- Every LLM call made by either agent
CREATE TABLE IF NOT EXISTS decisions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp      TEXT    NOT NULL,
    session_id     TEXT,
    bot            TEXT    NOT NULL,   -- 'stock' | 'options'
    model          TEXT,
    symbol         TEXT    NOT NULL,
    prompt         TEXT,
    raw_response   TEXT,
    decision       TEXT,               -- buy | sell | hold | buy_call | buy_put
    confidence     REAL,
    reasoning      TEXT,
    executed       INTEGER NOT NULL DEFAULT 0,  -- 1 = trade was placed
    indicators     TEXT,               -- JSON blob
    market_context TEXT,               -- JSON blob
    debate         TEXT                -- JSON blob or NULL
);

CREATE INDEX IF NOT EXISTS idx_decisions_symbol    ON decisions(symbol);
CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON decisions(timestamp);
CREATE INDEX IF NOT EXISTS idx_decisions_executed  ON decisions(executed);

-- Every order sent to Alpaca
CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    bot         TEXT NOT NULL,   -- 'stock' | 'options'
    symbol      TEXT NOT NULL,
    action      TEXT NOT NULL,   -- buy | sell | buy_call | buy_put
    shares      REAL,
    confidence  REAL,
    reasoning   TEXT,
    order_id    TEXT UNIQUE,
    contract    TEXT,            -- options contract symbol (options bot only)
    exit_pl_pct REAL,            -- options exit P&L % (options bot only)
    exit_reason TEXT             -- e.g. 'stop_loss_50%'
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol    ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_order_id  ON trades(order_id);

-- Realized P&L after a position closes (written by outcome_tracker.py)
CREATE TABLE IF NOT EXISTS outcomes (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol            TEXT NOT NULL,
    buy_order_id      TEXT UNIQUE,
    sell_order_id     TEXT,
    entry_timestamp   TEXT,
    exit_timestamp    TEXT,
    entry_price       REAL,
    exit_price        REAL,
    shares            REAL,
    realized_pnl      REAL,
    pnl_pct           REAL,
    hold_hours        REAL,
    entry_confidence  REAL,
    entry_reasoning   TEXT,
    won               INTEGER   -- 1 = profitable
);

CREATE INDEX IF NOT EXISTS idx_outcomes_symbol         ON outcomes(symbol);
CREATE INDEX IF NOT EXISTS idx_outcomes_entry_timestamp ON outcomes(entry_timestamp);

-- Labelled fine-tuning examples (written by training_data_builder.py)
CREATE TABLE IF NOT EXISTS training_examples (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    bot          TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    prompt       TEXT NOT NULL,
    ideal_output TEXT NOT NULL,
    label        TEXT NOT NULL,   -- winner | loser
    confidence   REAL,
    pnl_pct      REAL,
    entry_date   TEXT,
    session_id   TEXT,
    prompt_hash  TEXT UNIQUE,     -- MD5 of prompt — prevents duplicates
    generated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_training_label      ON training_examples(label);
CREATE INDEX IF NOT EXISTS idx_training_symbol     ON training_examples(symbol);
CREATE INDEX IF NOT EXISTS idx_training_entry_date ON training_examples(entry_date);
"""


def init_db(db_path: Path = DB_PATH) -> None:
    """Create tables and indexes if they don't exist yet."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with get_conn(db_path) as conn:
        conn.executescript(SCHEMA)
    logging.debug("Database initialised at %s", db_path)


# ---------------------------------------------------------------------------
# Insert helpers
# ---------------------------------------------------------------------------

def insert_decision(rec: dict, db_path: Path = DB_PATH) -> None:
    """Insert one decision record (from decision_log.jsonl format)."""
    sql = """
        INSERT OR IGNORE INTO decisions
            (timestamp, session_id, bot, model, symbol, prompt, raw_response,
             decision, confidence, reasoning, executed, indicators, market_context, debate)
        VALUES
            (:timestamp, :session_id, :bot, :model, :symbol, :prompt, :raw_response,
             :decision, :confidence, :reasoning, :executed, :indicators, :market_context, :debate)
    """
    row = {
        'timestamp':      rec.get('timestamp'),
        'session_id':     rec.get('session_id'),
        'bot':            rec.get('bot', 'stock'),
        'model':          rec.get('model'),
        'symbol':         rec.get('symbol'),
        'prompt':         rec.get('prompt'),
        'raw_response':   rec.get('raw_response'),
        'decision':       rec.get('decision'),
        'confidence':     rec.get('confidence'),
        'reasoning':      rec.get('reasoning'),
        'executed':       1 if rec.get('executed') else 0,
        'indicators':     json.dumps(rec['indicators'], default=_json_default) if rec.get('indicators') else None,
        'market_context': json.dumps(rec['market_context'], default=_json_default) if rec.get('market_context') else None,
        'debate':         json.dumps(rec['debate'], default=_json_default) if rec.get('debate') else None,
    }
    with get_conn(db_path) as conn:
        conn.execute(sql, row)


def insert_trade(rec: dict, bot: str = 'stock', db_path: Path = DB_PATH) -> None:
    """Insert one trade record (from trade_log.jsonl format)."""
    sql = """
        INSERT OR IGNORE INTO trades
            (timestamp, bot, symbol, action, shares, confidence, reasoning,
             order_id, contract, exit_pl_pct, exit_reason)
        VALUES
            (:timestamp, :bot, :symbol, :action, :shares, :confidence, :reasoning,
             :order_id, :contract, :exit_pl_pct, :exit_reason)
    """
    row = {
        'timestamp':   rec.get('timestamp'),
        'bot':         bot,
        'symbol':      rec.get('symbol'),
        'action':      rec.get('action'),
        'shares':      rec.get('shares') or rec.get('quantity'),
        'confidence':  rec.get('confidence'),
        'reasoning':   rec.get('reasoning'),
        'order_id':    rec.get('order_id'),
        'contract':    rec.get('contract'),
        'exit_pl_pct': rec.get('exit_pl_pct'),
        'exit_reason': rec.get('reason'),
    }
    with get_conn(db_path) as conn:
        conn.execute(sql, row)


def insert_outcome(rec: dict, db_path: Path = DB_PATH) -> None:
    """Insert one outcome record (from trade_outcomes.jsonl format)."""
    sql = """
        INSERT OR IGNORE INTO outcomes
            (symbol, buy_order_id, sell_order_id, entry_timestamp, exit_timestamp,
             entry_price, exit_price, shares, realized_pnl, pnl_pct, hold_hours,
             entry_confidence, entry_reasoning, won)
        VALUES
            (:symbol, :buy_order_id, :sell_order_id, :entry_timestamp, :exit_timestamp,
             :entry_price, :exit_price, :shares, :realized_pnl, :pnl_pct, :hold_hours,
             :entry_confidence, :entry_reasoning, :won)
    """
    row = {**rec, 'won': 1 if rec.get('won') else 0}
    with get_conn(db_path) as conn:
        conn.execute(sql, row)


def insert_training_example(rec: dict, db_path: Path = DB_PATH) -> bool:
    """
    Insert one training example. Returns True if inserted, False if duplicate.
    Deduplication is enforced by the UNIQUE constraint on prompt_hash.
    """
    sql = """
        INSERT OR IGNORE INTO training_examples
            (bot, symbol, prompt, ideal_output, label, confidence, pnl_pct,
             entry_date, session_id, prompt_hash, generated_at)
        VALUES
            (:bot, :symbol, :prompt, :ideal_output, :label, :confidence, :pnl_pct,
             :entry_date, :session_id, :prompt_hash, :generated_at)
    """
    meta = rec.get('metadata', {})
    row = {
        'bot':          meta.get('bot', rec.get('bot', '')),
        'symbol':       meta.get('symbol', ''),
        'prompt':       rec.get('input', rec.get('prompt', '')),
        'ideal_output': rec.get('output', rec.get('ideal_output', '')),
        'label':        rec.get('label', ''),
        'confidence':   meta.get('confidence'),
        'pnl_pct':      meta.get('pnl_pct'),
        'entry_date':   meta.get('entry_date'),
        'session_id':   meta.get('session_id'),
        'prompt_hash':  meta.get('prompt_hash'),
        'generated_at': meta.get('generated_at', datetime.now().isoformat()),
    }
    with get_conn(db_path) as conn:
        cursor = conn.execute(sql, row)
        return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# Query helpers (used by training_data_builder and performance_analyzer)
# ---------------------------------------------------------------------------

def get_executed_decisions(bot: str = None, min_confidence: float = 0.0,
                           db_path: Path = DB_PATH) -> list[dict]:
    """Return executed decisions optionally filtered by bot and confidence."""
    sql = "SELECT * FROM decisions WHERE executed = 1 AND confidence >= ?"
    params: list = [min_confidence]
    if bot:
        sql += " AND bot = ?"
        params.append(bot)
    sql += " ORDER BY timestamp"
    with get_conn(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def get_outcome_by_entry(symbol: str, entry_date: str,
                         db_path: Path = DB_PATH) -> dict | None:
    """Look up a closed-trade outcome by symbol and entry date (YYYY-MM-DD)."""
    sql = """
        SELECT * FROM outcomes
        WHERE symbol = ? AND entry_timestamp LIKE ?
        ORDER BY entry_timestamp
        LIMIT 1
    """
    with get_conn(db_path) as conn:
        row = conn.execute(sql, [symbol, f"{entry_date}%"]).fetchone()
    return dict(row) if row else None


def get_training_examples(bot: str = None, label: str = None,
                           db_path: Path = DB_PATH) -> list[dict]:
    """Return training examples optionally filtered by bot and/or label."""
    sql = "SELECT * FROM training_examples WHERE 1=1"
    params: list = []
    if bot:
        sql += " AND bot = ?"
        params.append(bot)
    if label:
        sql += " AND label = ?"
        params.append(label)
    sql += " ORDER BY generated_at"
    with get_conn(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def get_existing_prompt_hashes(db_path: Path = DB_PATH) -> set[str]:
    """Return all prompt hashes already in the training_examples table."""
    with get_conn(db_path) as conn:
        rows = conn.execute("SELECT prompt_hash FROM training_examples WHERE prompt_hash IS NOT NULL").fetchall()
    return {r['prompt_hash'] for r in rows}


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)
    init_db()
    logging.info("Database ready at %s", DB_PATH)

    # Quick schema dump
    with get_conn() as conn:
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        for t in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {t['name']}").fetchone()[0]
            print(f"  {t['name']}: {count} rows")
