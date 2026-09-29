#!/usr/bin/env python3
"""
order_status_cache.py — persistent Alpaca order-status cache for the outcome
trackers.

Why this exists
---------------
outcome_tracker.py and options_outcome_tracker.py reprocess the ENTIRE trade
log on every run and call get_order_by_id once per buy row and once per sell
row, with no caching. The stock trade log reached 3,390 rows, so each nightly
pass issued 3,390 sequential Alpaca lookups at ~53 ms each — about 4 minutes
of pure network wait against the daemon's 120-second subprocess budget.

The result was a silent, self-reinforcing stall: the tracker timed out before
writing anything, so no outcomes were recorded (the outcomes table froze on
2026-07-02), the unreconciled backlog grew, and every subsequent run had more
work than the last. Downstream, online_trainer.py never saw its 15-new-outcome
threshold, so the whole learning loop starved.

Caching is exact here, not an approximation: a *terminal* order never changes
again. Once Alpaca reports an order filled/canceled/expired/rejected/replaced,
its status and fill price are immutable, so a cached answer is identical to a
fresh lookup — forever. Re-fetching them is pure waste.

What is deliberately NOT cached
-------------------------------
- Non-terminal statuses (new, accepted, partially_filled, held, ...) — these
  can still change, so they are always re-fetched.
- The {'status': 'unknown'} failure sentinel. This is the important one: it is
  what get_order_status returns when the API call *raises*, and it is
  indistinguishable from a genuine miss. Caching it would make a transient
  network blip permanent — exactly the misclassification bug diagnosed in
  docs/sprint02-d7-findings.md, where orders that had demonstrably filled were
  recorded as unreconciled because one lookup failed. A failed lookup must stay
  retryable.

Progress is flushed periodically rather than only at the end, so a run killed
by the subprocess timeout still persists what it resolved. That makes the cache
converge across runs even if a single pass cannot finish, which matters while a
cold cache is being warmed.
"""
import json
import logging
import os
from pathlib import Path

# Alpaca order states that can never change again. Conservative on purpose:
# anything not listed here is re-fetched, which costs one lookup and is always
# safe. Adding a non-terminal state to this set would cache a moving value.
TERMINAL_STATUSES: frozenset[str] = frozenset({
    'filled', 'canceled', 'cancelled', 'expired', 'rejected', 'replaced',
})

# The sentinel get_order_status returns when the underlying API call raised.
# Never cached — see module docstring.
_FAILURE_STATUS = 'unknown'

_DEFAULT_FLUSH_EVERY = int(os.getenv('ORDER_CACHE_FLUSH_EVERY', '250'))


class OrderStatusCache:
    """Order-id → {status, filled_qty, avg_price}, persisted as JSON.

    Usage:
        cache = OrderStatusCache(Path('logs/order_status_cache.json'))
        status = cache.get_or_fetch(order_id, lambda: real_lookup(order_id))
        cache.flush()
    """

    def __init__(self, path: Path, flush_every: int = _DEFAULT_FLUSH_EVERY):
        self.path = Path(path)
        self.flush_every = max(1, int(flush_every))
        self._entries: dict[str, dict] = {}
        self._pending_writes = 0
        self.hits = 0
        self.misses = 0
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        """Load the cache from disk. A missing or corrupt file is not fatal."""
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
        except Exception as e:
            # A truncated file (e.g. killed mid-write) must not break the run —
            # the worst case is a cold cache, which is the old behaviour.
            logging.warning("order-status cache unreadable, starting cold: %s", e)
            return
        if not isinstance(raw, dict):
            logging.warning("order-status cache is not an object, starting cold")
            return
        for oid, entry in raw.items():
            if isinstance(entry, dict) and entry.get('status') in TERMINAL_STATUSES:
                self._entries[oid] = entry
        logging.info("order-status cache: loaded %d terminal entries from %s",
                     len(self._entries), self.path.name)

    # ------------------------------------------------------------------
    def get_or_fetch(self, order_id, fetch) -> dict:
        """Return the status dict for *order_id*, calling *fetch* only on a miss.

        *fetch* takes no arguments and returns the same dict shape
        get_order_status produces: {'status', 'filled_qty', 'avg_price'}.
        """
        if not order_id:
            return {'status': _FAILURE_STATUS, 'filled_qty': 0, 'avg_price': None}

        key = str(order_id)
        cached = self._entries.get(key)
        if cached is not None:
            self.hits += 1
            # Return a copy so a caller mutating the result cannot corrupt the
            # cache (and thus what gets written to disk).
            return dict(cached)

        self.misses += 1
        result = fetch()
        if isinstance(result, dict) and result.get('status') in TERMINAL_STATUSES:
            self._entries[key] = dict(result)
            self._pending_writes += 1
            if self._pending_writes >= self.flush_every:
                self.flush()
        return result

    # ------------------------------------------------------------------
    def flush(self) -> None:
        """Persist the cache. Writes atomically so a kill cannot truncate it."""
        if self._pending_writes == 0 and self.path.exists():
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + '.tmp')
            tmp.write_text(json.dumps(self._entries))
            os.replace(tmp, self.path)   # atomic on POSIX
            self._pending_writes = 0
        except Exception as e:
            # Losing the cache costs time on the next run, never correctness.
            logging.warning("could not persist order-status cache: %s", e)

    # ------------------------------------------------------------------
    def summary(self) -> str:
        total = self.hits + self.misses
        pct = (100 * self.hits / total) if total else 0.0
        return (f"order-status cache: {self.hits} hits / {self.misses} lookups "
                f"({pct:.0f}% avoided, {len(self._entries)} cached)")
