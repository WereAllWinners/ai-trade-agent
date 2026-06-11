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
-- Cross-bot cash reservation lock.
-- Both bots write a row here atomically before sizing a buy,
-- ensuring neither can double-count the same cash.
CREATE TABLE IF NOT EXISTS cash_reservations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    bot         TEXT    NOT NULL,   -- 'stock' | 'options'
    amount      REAL    NOT NULL,
    reserved_at TEXT    NOT NULL,
    released    INTEGER NOT NULL DEFAULT 0,
    released_at TEXT
);

-- Every LLM call made by either agent
CREATE TABLE IF NOT EXISTS decisions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp      TEXT    NOT NULL,
    session_id     TEXT,
    bot            TEXT    NOT NULL,   -- 'stock' | 'options'
    source         TEXT    NOT NULL DEFAULT 'paper',  -- 'paper' | 'live'
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
    source      TEXT NOT NULL DEFAULT 'paper',  -- 'paper' | 'live'
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
    bot               TEXT NOT NULL DEFAULT 'stock',  -- 'stock' | 'options'
    source            TEXT NOT NULL DEFAULT 'paper',  -- 'paper' | 'live'
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
    source       TEXT NOT NULL DEFAULT 'paper',  -- 'paper' | 'live'
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

-- Daily circuit-breaker baseline: survives daemon restarts within the same trading day.
CREATE TABLE IF NOT EXISTS daily_state (
    trade_date          TEXT NOT NULL,
    bot                 TEXT NOT NULL,   -- 'stock' | 'options'
    source              TEXT NOT NULL DEFAULT 'paper',  -- 'paper' | 'live'
    daily_start_equity  REAL NOT NULL,
    PRIMARY KEY (trade_date, bot, source)
);

CREATE TABLE IF NOT EXISTS unreconciled_orders (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at TEXT NOT NULL,
    bot         TEXT NOT NULL,
    order_id    TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    status      TEXT NOT NULL,
    reason      TEXT,
    UNIQUE(order_id, reason)
);

-- E3: Historical IV snapshots for IV-rank computation.
-- One row per (symbol, timestamp); UNIQUE prevents duplicate intraday upserts.
CREATE TABLE IF NOT EXISTS iv_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    iv          REAL,
    hv_20       REAL,
    iv_rank     REAL,
    UNIQUE(symbol, timestamp)
);
"""


def _migrate(conn) -> None:
    """Add columns introduced after initial schema creation."""
    for table in ('decisions', 'trades', 'outcomes', 'training_examples'):
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if 'source' not in existing:
            conn.execute(
                f"ALTER TABLE {table} ADD COLUMN source TEXT NOT NULL DEFAULT 'paper'"
            )
            logging.info("db: migrated %s — added source column", table)

    # Add source column to daily_state so paper and live circuit breakers don't share baseline
    ds_cols = {row[1] for row in conn.execute("PRAGMA table_info(daily_state)")}
    if 'source' not in ds_cols:
        conn.execute("ALTER TABLE daily_state ADD COLUMN source TEXT NOT NULL DEFAULT 'paper'")
        logging.info("db: migrated daily_state — added source column")

    # Add bot column to outcomes (options outcomes were merged without a bot label)
    outcomes_cols = {row[1] for row in conn.execute("PRAGMA table_info(outcomes)")}
    if 'bot' not in outcomes_cols:
        conn.execute("ALTER TABLE outcomes ADD COLUMN bot TEXT NOT NULL DEFAULT 'stock'")
        # Backfill: options contract symbols contain 'C0' or 'P0' followed by digits
        conn.execute("""
            UPDATE outcomes SET bot = 'options'
            WHERE symbol GLOB '*C0*' OR symbol GLOB '*P0*'
        """)
        updated = conn.execute("SELECT COUNT(*) FROM outcomes WHERE bot='options'").fetchone()[0]
        logging.info("db: migrated outcomes — added bot column, tagged %d options rows", updated)

    # A6: order_id on decisions for accurate outcome lookup (avoids same-symbol/same-day ambiguity)
    dec_cols = {row[1] for row in conn.execute("PRAGMA table_info(decisions)")}
    if 'order_id' not in dec_cols:
        conn.execute("ALTER TABLE decisions ADD COLUMN order_id TEXT")
        logging.info("db: migrated decisions — added order_id column")
    if 'regime' not in dec_cols:
        conn.execute("ALTER TABLE decisions ADD COLUMN regime TEXT")
        logging.info("db: migrated decisions — added regime column")

    # D1: SPY benchmark columns on outcomes for excess-return labeling
    out_cols = {row[1] for row in conn.execute("PRAGMA table_info(outcomes)")}
    if 'spy_return_pct' not in out_cols:
        conn.execute("ALTER TABLE outcomes ADD COLUMN spy_return_pct REAL")
        logging.info("db: migrated outcomes — added spy_return_pct column")
    if 'excess_return_pct' not in out_cols:
        conn.execute("ALTER TABLE outcomes ADD COLUMN excess_return_pct REAL")
        logging.info("db: migrated outcomes — added excess_return_pct column")

    # D2: regime tag on outcomes for per-regime performance breakdowns
    if 'regime' not in out_cols:
        conn.execute("ALTER TABLE outcomes ADD COLUMN regime TEXT")
        logging.info("db: migrated outcomes — added regime column")


def init_db(db_path: Path = DB_PATH) -> None:
    """Create tables and indexes if they don't exist yet, then run migrations."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with get_conn(db_path) as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)
    logging.debug("Database initialised at %s", db_path)


# ---------------------------------------------------------------------------
# Insert helpers
# ---------------------------------------------------------------------------

def insert_decision(rec: dict, source: str = 'paper', db_path: Path = DB_PATH) -> None:
    """Insert one decision record (from decision_log.jsonl format)."""
    sql = """
        INSERT OR IGNORE INTO decisions
            (timestamp, session_id, bot, source, model, symbol, prompt, raw_response,
             decision, confidence, reasoning, executed, indicators, market_context, debate,
             order_id, regime)
        VALUES
            (:timestamp, :session_id, :bot, :source, :model, :symbol, :prompt, :raw_response,
             :decision, :confidence, :reasoning, :executed, :indicators, :market_context, :debate,
             :order_id, :regime)
    """
    row = {
        'timestamp':      rec.get('timestamp'),
        'session_id':     rec.get('session_id'),
        'bot':            rec.get('bot', 'stock'),
        'source':         rec.get('source', source),
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
        'order_id':       rec.get('order_id'),
        'regime':         rec.get('regime'),
    }
    with get_conn(db_path) as conn:
        conn.execute(sql, row)


def insert_trade(rec: dict, bot: str = 'stock', source: str = 'paper',
                 db_path: Path = DB_PATH) -> None:
    """Insert one trade record (from trade_log.jsonl format)."""
    sql = """
        INSERT OR IGNORE INTO trades
            (timestamp, bot, source, symbol, action, shares, confidence, reasoning,
             order_id, contract, exit_pl_pct, exit_reason)
        VALUES
            (:timestamp, :bot, :source, :symbol, :action, :shares, :confidence, :reasoning,
             :order_id, :contract, :exit_pl_pct, :exit_reason)
    """
    row = {
        'timestamp':   rec.get('timestamp'),
        'bot':         bot,
        'source':      source,
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


def insert_outcome(rec: dict, bot: str = 'stock', source: str = 'paper',
                   db_path: Path = DB_PATH) -> None:
    """Insert one outcome record (from trade_outcomes.jsonl format)."""
    sql = """
        INSERT OR IGNORE INTO outcomes
            (symbol, bot, source, buy_order_id, sell_order_id, entry_timestamp, exit_timestamp,
             entry_price, exit_price, shares, realized_pnl, pnl_pct, hold_hours,
             entry_confidence, entry_reasoning, won,
             spy_return_pct, excess_return_pct, regime)
        VALUES
            (:symbol, :bot, :source, :buy_order_id, :sell_order_id, :entry_timestamp, :exit_timestamp,
             :entry_price, :exit_price, :shares, :realized_pnl, :pnl_pct, :hold_hours,
             :entry_confidence, :entry_reasoning, :won,
             :spy_return_pct, :excess_return_pct, :regime)
    """
    row = {
        'symbol':            rec.get('symbol'),
        'bot':               rec.get('bot', bot),
        'source':            rec.get('source', source),
        'buy_order_id':      rec.get('buy_order_id'),
        'sell_order_id':     rec.get('sell_order_id'),
        'entry_timestamp':   rec.get('entry_timestamp'),
        'exit_timestamp':    rec.get('exit_timestamp'),
        'entry_price':       rec.get('entry_price'),
        'exit_price':        rec.get('exit_price'),
        'shares':            rec.get('shares'),
        'realized_pnl':      rec.get('realized_pnl'),
        'pnl_pct':           rec.get('pnl_pct'),
        'hold_hours':        rec.get('hold_hours'),
        'entry_confidence':  rec.get('entry_confidence'),
        'entry_reasoning':   rec.get('entry_reasoning'),
        'won':               1 if rec.get('won') else 0,
        'spy_return_pct':    rec.get('spy_return_pct'),
        'excess_return_pct': rec.get('excess_return_pct'),
        'regime':            rec.get('regime'),
    }
    with get_conn(db_path) as conn:
        conn.execute(sql, row)


def insert_training_example(rec: dict, source: str = 'paper', db_path: Path = DB_PATH) -> bool:
    """
    Insert one training example. Returns True if inserted, False if duplicate.
    Deduplication is enforced by the UNIQUE constraint on prompt_hash.
    """
    sql = """
        INSERT OR IGNORE INTO training_examples
            (bot, source, symbol, prompt, ideal_output, label, confidence, pnl_pct,
             entry_date, session_id, prompt_hash, generated_at)
        VALUES
            (:bot, :source, :symbol, :prompt, :ideal_output, :label, :confidence, :pnl_pct,
             :entry_date, :session_id, :prompt_hash, :generated_at)
    """
    meta = rec.get('metadata', {})
    row = {
        'bot':          meta.get('bot', rec.get('bot', '')),
        'source':       meta.get('source', rec.get('source', source)),
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
                           source: str = None, db_path: Path = DB_PATH) -> list[dict]:
    """Return executed decisions optionally filtered by bot, confidence, and source."""
    sql = "SELECT * FROM decisions WHERE executed = 1 AND confidence >= ?"
    params: list = [min_confidence]
    if bot:
        sql += " AND bot = ?"
        params.append(bot)
    if source:
        sql += " AND source = ?"
        params.append(source)
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


def get_outcome_by_order_id(order_id: str, db_path: Path = DB_PATH) -> dict | None:
    """Look up a closed-trade outcome by exact Alpaca buy_order_id.

    Preferred over get_outcome_by_entry when the order ID is available — avoids
    mislabeling when the same symbol trades more than once on the same day.
    """
    sql = "SELECT * FROM outcomes WHERE buy_order_id = ?"
    with get_conn(db_path) as conn:
        row = conn.execute(sql, [order_id]).fetchone()
    return dict(row) if row else None


def get_calibration_map(bot: str, lookback_days: int = 90,
                        db_path: Path = DB_PATH) -> dict:
    """Return realized win-rate per confidence bin for *bot* over the last *lookback_days*.

    Bins match the four ranges used in weekly_report.py::calculate_confidence_calibration.
    Returns {} gracefully when no data exists (early in training).
    Requires a minimum of 10 outcomes per bin; bins below the minimum are omitted.
    """
    from datetime import timedelta
    cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
    sql = """
        SELECT entry_confidence, won
        FROM outcomes
        WHERE bot = ? AND exit_timestamp >= ? AND entry_confidence IS NOT NULL
    """
    with get_conn(db_path) as conn:
        rows = conn.execute(sql, [bot, cutoff]).fetchall()

    if not rows:
        return {}

    bins: dict[str, list] = {
        '0.60-0.70': [],
        '0.70-0.80': [],
        '0.80-0.90': [],
        '0.90-1.00': [],
    }
    for row in rows:
        conf = row['entry_confidence']
        won  = row['won']
        if conf < 0.60:
            continue
        elif conf < 0.70:
            bins['0.60-0.70'].append(won)
        elif conf < 0.80:
            bins['0.70-0.80'].append(won)
        elif conf < 0.90:
            bins['0.80-0.90'].append(won)
        else:
            bins['0.90-1.00'].append(won)

    result = {}
    for bin_label, outcomes in bins.items():
        if len(outcomes) >= 10:
            result[bin_label] = {'count': len(outcomes), 'win_rate': sum(outcomes) / len(outcomes)}
    return result


def get_training_examples(bot: str = None, label: str = None,
                           source: str = None, db_path: Path = DB_PATH) -> list[dict]:
    """Return training examples optionally filtered by bot, label, and/or source."""
    sql = "SELECT * FROM training_examples WHERE 1=1"
    params: list = []
    if bot:
        sql += " AND bot = ?"
        params.append(bot)
    if label:
        sql += " AND label = ?"
        params.append(label)
    if source:
        sql += " AND source = ?"
        params.append(source)
    sql += " ORDER BY generated_at"
    with get_conn(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Cash reservation helpers (cross-bot race-condition guard)
# ---------------------------------------------------------------------------

def reserve_cash(bot: str, amount: float, db_path: Path = DB_PATH) -> int:
    """
    Atomically insert a cash reservation for *bot* and return the reservation id.

    Callers must call release_cash(reservation_id) after the order is submitted
    (or abandoned).  This creates a short advisory lock window so two bots
    cannot both size their trades against the same un-committed cash.

    The reservation is written inside an EXCLUSIVE transaction so concurrent
    SQLite writers see a consistent view even under WAL mode.
    """
    sql = """
        INSERT INTO cash_reservations (bot, amount, reserved_at)
        VALUES (?, ?, ?)
    """
    now = datetime.now().isoformat()
    conn = sqlite3.connect(str(db_path), timeout=10)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("BEGIN EXCLUSIVE")
        cursor = conn.execute(sql, [bot, amount, now])
        reservation_id = cursor.lastrowid
        conn.commit()
        return reservation_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def release_cash(reservation_id: int, db_path: Path = DB_PATH) -> None:
    """Mark a cash reservation as released so other bots can use that cash."""
    sql = """
        UPDATE cash_reservations
        SET released = 1, released_at = ?
        WHERE id = ?
    """
    with get_conn(db_path) as conn:
        conn.execute(sql, [datetime.now().isoformat(), reservation_id])


def get_total_reserved(db_path: Path = DB_PATH) -> float:
    """Return the sum of all currently active (unreleased) cash reservations."""
    sql = "SELECT COALESCE(SUM(amount), 0) FROM cash_reservations WHERE released = 0"
    with get_conn(db_path) as conn:
        row = conn.execute(sql).fetchone()
    return float(row[0]) if row else 0.0


def cleanup_stale_reservations(max_age_seconds: int = 120, db_path: Path = DB_PATH) -> int:
    """
    Release reservations that were never explicitly released — e.g. after a crash.
    Returns the number of rows cleaned up.
    """
    cutoff = (datetime.now() - __import__('datetime').timedelta(seconds=max_age_seconds)).isoformat()
    sql = """
        UPDATE cash_reservations
        SET released = 1, released_at = ?
        WHERE released = 0 AND reserved_at < ?
    """
    with get_conn(db_path) as conn:
        cursor = conn.execute(sql, [datetime.now().isoformat(), cutoff])
        return cursor.rowcount


def get_recent_trades(limit: int = 100, bot: str = None,
                      db_path: Path = DB_PATH) -> list[dict]:
    """
    Return the *limit* most-recent closed-trade outcomes (from the outcomes
    table) suitable for Sharpe / draw-down calculations.

    Falls back to the trades table if the outcomes table is empty, returning
    rows with only the fields AllocationController needs (pnl_pct may be None).
    """
    sql_outcomes = """
        SELECT symbol, pnl_pct, realized_pnl, hold_hours,
               entry_timestamp AS timestamp
        FROM outcomes
        ORDER BY entry_timestamp DESC
        LIMIT ?
    """
    with get_conn(db_path) as conn:
        rows = conn.execute(sql_outcomes, [limit]).fetchall()

    if rows:
        return [dict(r) for r in rows]

    # Fallback: trades table (no pnl data, but lets AllocationController
    # at least count trades without crashing on an empty outcomes table)
    sql_trades = """
        SELECT symbol, exit_pl_pct AS pnl_pct, timestamp
        FROM trades
        ORDER BY timestamp DESC
        LIMIT ?
    """
    params: list = [limit]
    if bot:
        sql_trades = sql_trades.replace("ORDER BY", "WHERE bot = ? ORDER BY")
        params.insert(0, bot)
    with get_conn(db_path) as conn:
        rows = conn.execute(sql_trades, params).fetchall()
    return [dict(r) for r in rows]


def get_outcome_count_since(cutoff_timestamp: str, db_path: Path = DB_PATH) -> int:
    """Return number of outcomes recorded after cutoff_timestamp (ISO string)."""
    sql = "SELECT COUNT(*) FROM outcomes WHERE exit_timestamp > ?"
    with get_conn(db_path) as conn:
        row = conn.execute(sql, [cutoff_timestamp]).fetchone()
    return int(row[0]) if row else 0


def get_existing_prompt_hashes(db_path: Path = DB_PATH) -> set[str]:
    """Return all prompt hashes already in the training_examples table."""
    with get_conn(db_path) as conn:
        rows = conn.execute("SELECT prompt_hash FROM training_examples WHERE prompt_hash IS NOT NULL").fetchall()
    return {r['prompt_hash'] for r in rows}


def get_all_outcomes(bot: str = 'stock', db_path: Path = DB_PATH) -> list[dict]:
    """Return all outcome rows for the given bot, ordered by entry_timestamp."""
    sql = "SELECT * FROM outcomes WHERE bot = ? ORDER BY entry_timestamp"
    with get_conn(db_path) as conn:
        rows = conn.execute(sql, [bot]).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Daily circuit-breaker state
# ---------------------------------------------------------------------------

def save_daily_start_equity(bot: str, equity: float, source: str = 'paper',
                            db_path: Path = DB_PATH) -> None:
    """Persist today's opening equity so the circuit breaker survives daemon restarts."""
    today = datetime.now().date().isoformat()
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO daily_state (trade_date, bot, source, daily_start_equity) VALUES (?, ?, ?, ?)",
            (today, bot, source, equity),
        )


def load_daily_start_equity(bot: str, source: str = 'paper',
                            db_path: Path = DB_PATH) -> float | None:
    """Return today's saved opening equity for *bot* and *source*, or None if not yet recorded."""
    today = datetime.now().date().isoformat()
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT daily_start_equity FROM daily_state WHERE trade_date = ? AND bot = ? AND source = ?",
            (today, bot, source),
        ).fetchone()
    return float(row['daily_start_equity']) if row else None


# ---------------------------------------------------------------------------
# Unreconciled order audit
# ---------------------------------------------------------------------------

def insert_unreconciled_order(rec: dict, bot: str, db_path: Path = DB_PATH) -> None:
    """Record an order that could not be matched to a filled P&L pair."""
    with get_conn(db_path) as conn:
        conn.execute(
            """INSERT OR IGNORE INTO unreconciled_orders
               (recorded_at, bot, order_id, symbol, status, reason)
               VALUES (:recorded_at, :bot, :order_id, :symbol, :status, :reason)""",
            {**rec, 'bot': bot},
        )


# ---------------------------------------------------------------------------
# IV history helpers (E3 — Options IV enrichment)
# ---------------------------------------------------------------------------

_IV_RANK_MIN_ROWS = 30  # minimum historical rows to trust iv_rank


def upsert_iv_snapshot(
    symbol: str,
    iv: float | None,
    hv_20: float | None = None,
    timestamp: str | None = None,
    db_path: Path = DB_PATH,
) -> None:
    """Insert (or ignore duplicate) an IV snapshot for *symbol*."""
    ts = timestamp or datetime.now().isoformat()
    with get_conn(db_path) as conn:
        conn.execute(
            """INSERT OR IGNORE INTO iv_history (symbol, timestamp, iv, hv_20)
               VALUES (:symbol, :timestamp, :iv, :hv_20)""",
            {'symbol': symbol, 'timestamp': ts, 'iv': iv, 'hv_20': hv_20},
        )


def get_iv_rank(
    symbol: str,
    current_iv: float,
    db_path: Path = DB_PATH,
) -> float | None:
    """Return the percentile rank of *current_iv* vs stored iv_history for *symbol*.

    Returns None when fewer than _IV_RANK_MIN_ROWS snapshots exist (not enough
    history to produce a reliable rank).

    Example: 0.75 means current_iv exceeds 75% of historical readings.
    """
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT iv FROM iv_history WHERE symbol = ? AND iv IS NOT NULL",
            (symbol,),
        ).fetchall()

    if len(rows) < _IV_RANK_MIN_ROWS:
        return None

    ivs = sorted(float(r['iv']) for r in rows)
    n   = len(ivs)
    below = sum(1 for v in ivs if v < current_iv)
    return round(below / n, 4)


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
