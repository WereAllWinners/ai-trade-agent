# Sprint 02 — D7: `missing_entry_fill` / `missing_exit_fill` Root Cause

Read-only investigation, no code changes. Builds on D4.3's empirical
cross-contamination finding with a targeted broker-side sample of
`unreconciled_orders` and an exact code trace of the classification path.

## 1. 10-row broker-side sample

Sampled the 5 oldest and 5 newest `unreconciled_orders` rows (distinct
symbols), looked each `order_id` up read-only against both the paper and
live `TradingClient`s (same method already proven in D4.3's spot-check).

| id | recorded_at | symbol | reason | DB `source` | broker-resolved | broker status | filled_qty |
|----|---|---|---|---|---|---|---|
| 1 | 2026-05-08 | NFLX | missing_exit_fill | paper | paper | expired | 0 |
| 2 | 2026-05-08 | XOM | missing_exit_fill | paper | paper | expired | 0 |
| 3 | 2026-05-08 | AEP | missing_entry_fill | paper | paper | expired | 0 |
| 4 | 2026-05-08 | ABT | missing_exit_fill | paper | paper | expired | 0 |
| 5 | 2026-05-08 | LMT | missing_exit_fill | paper | paper | expired | 0 |
| 981 | 2026-07-02 | AEP | missing_entry_fill | paper | paper | **filled** | 6 |
| 974 | 2026-07-02 | PFE | missing_entry_fill | paper | paper | **filled** | 35 |
| 960 | 2026-07-02 | AMTM | missing_entry_fill | paper | paper | **filled** | 205 |
| 950 | 2026-07-02 | ABM | missing_entry_fill | paper | paper | **filled** | 176 |
| 949 | 2026-07-02 | COP | missing_entry_fill | paper | paper | **filled** | 52 |

**Finding 1 — no additional cross-contamination found in this sample.**
10/10 rows' recorded `source` matched the independent broker lookup exactly.
Small sample (10 of 108 rows), so this doesn't rule out other contaminated
rows in the table, but it's a clean result and consistent with D4.3/D4.4
already having addressed the mechanism that caused it.

**Finding 2 — two distinct, previously-conflated root causes, not one:**

- **The 5 oldest rows (2026-05-08) are legitimately unreconciled.** The
  underlying orders `status=expired, filled_qty=0` at the broker — they
  never filled at all. `missing_entry_fill`/`missing_exit_fill` is *correct*
  classification here (there genuinely is no fill price to report), even
  though the label reads like "we lost track of a fill" rather than "there
  was no fill." Not a bug; a naming/observability nit at most.

- **The 5 newest rows (2026-07-02) are a real, separate bug.** Direct lookup
  moments ago shows these orders **did fill**, with valid `filled_avg_price`
  (e.g. AEP order `24162bf4…` — `status=filled, filled_avg_price=135.84,
  filled_qty=6`, filled back on 2026-04-28). Yet the DB row itself records
  `status=unknown` — meaning `outcome_tracker`'s *own* lookup, at the moment
  it ran, failed to retrieve data for an order that is demonstrably
  retrievable via the exact same API call. This points to a transient
  failure (rate-limit/network blip, plausibly from concurrent paper+live+
  options daemons hitting Alpaca simultaneously — the same class of
  collapse the sprint's opening doc hypothesized for yfinance, here showing
  up against the Alpaca API instead) silently misclassifying a genuinely
  successful trade as unreconciled.

## 2. Exact code trace — `missing_entry_fill` path

`scripts/analysis/outcome_tracker.py`, `match_and_calculate_pnl()`:

```python
# lines 183-185 — buy processed, fill price looked up and cached on the queue entry
if action == 'buy':
    fill_price = self.get_fill_price(trade.get('order_id'))
    open_buys.append({**trade, 'fill_price': fill_price})

# lines 187-212 — sell encountered; FIFO-pop the oldest open buy and pair it
elif action == 'sell' and open_buys:
    entry = open_buys.pop(0)
    exit_fill = self.get_fill_price(trade.get('order_id'))
    entry_price = entry.get('fill_price')
    exit_price = exit_fill

    if not entry_price or not exit_price:
        missing_side = 'missing_entry_fill' if not entry_price else 'missing_exit_fill'
        missing_oid = entry.get('order_id') if not entry_price else trade.get('order_id')
        oid_status = self.get_order_status(missing_oid)
        logging.warning(...)
        try:
            _db.insert_unreconciled_order({...}, bot=self._bot_name,
                                           source='paper' if self.paper else 'live')
        except Exception as db_err:
            logging.debug(...)
        continue   # <-- pair permanently dropped here, no retry
```

`get_order_status()` (lines 121-134) wraps the actual Alpaca call in a bare
`except Exception` that returns `{'status': 'unknown', ...}` on **any**
failure — network error, rate limit, timeout, or a genuine missing order are
all indistinguishable to the caller:

```python
def get_order_status(self, order_id) -> dict:
    if not order_id:
        return {'status': 'unknown', 'filled_qty': 0, 'avg_price': None}
    try:
        order = self.trading_client.get_order_by_id(order_id)
        return {...}
    except Exception as e:
        logging.debug(...)
        return {'status': 'unknown', 'filled_qty': 0, 'avg_price': None}
```

**Concrete gap found:** `scripts/utils/alpaca_retry.py`'s
`retry_on_rate_limit` decorator (exponential backoff on Alpaca 429/5xx)
already exists and is already applied to `autonomous_agent.py`'s
`trading_client` methods (`submit_order`, `get_account`,
`get_all_positions`, `get_orders`, `get_order_by_id` — see
`autonomous_agent.py` `__init__`). `outcome_tracker.py`'s own
`self.trading_client` (constructed at line 63) never gets this wrapper —
its `get_order_by_id` calls are fully unprotected against exactly the kind
of transient failure Finding 2 above demonstrates actually happens in
production.

The sole consumer of `unreconciled_orders` today is the passive count in
`performance_analyzer.py` (~lines 401-407) — there is no retry or
reconciliation sweep anywhere that revisits a dropped pair later.

## 3. Recommendation (not implemented here — read-only stage)

D4.3's `(symbol, source)` bucketing fix and D4.4's targeted backfill address
the **cross-account commingling** mechanism directly and are sufficient for
that specific failure mode — Finding 1 above (0/10 sampled rows showing any
remaining contamination) is consistent with that.

They do **not** address Finding 2 — transient lookup failures during
`match_and_calculate_pnl` misclassifying genuinely-filled trades. This is a
distinct bug with a low-risk, mechanical fix already proven elsewhere in the
codebase:

1. Apply `retry_on_rate_limit` to `outcome_tracker.py`'s (and
   `options_outcome_tracker.py`'s) `trading_client` methods, same pattern as
   `autonomous_agent.py`'s `__init__`.
2. Optionally, distinguish "order genuinely never filled" (Finding 1 shape —
   `status=expired/canceled`, `filled_qty=0`) from "lookup failed" (Finding 2
   shape — exception raised) in `get_order_status`'s return, so
   `unreconciled_orders.status` stops recording `unknown` for both cases —
   today they're indistinguishable in the DB even though they mean very
   different things operationally (one is expected, one is a bug).

Recommend as a small, separate follow-up — not bundled into this sprint,
since it's an independent fix with its own test surface and D4.3/D4.4 are
already a large, verified change set.

## Addendum (sprint03 E3) — accounting for the D4.4 backfill gap (915 → 895)

D4.4's preview run found 915 candidates; the subsequent `--execute` run only
stamped 895 — a 20-row gap the original report didn't explain. Fully resolved
using the two runs' own captured output (still on disk from the same session):

```
Preview  (18:59): candidates=915  stamped=0    unresolved=0  errors=0
Execute  (19:07): candidates=895  stamped=895  unresolved=0  errors=0
```

**`unresolved=0` in both runs is the key fact** — it rules out "order_id
found in neither Alpaca account" entirely; that bucket is empty. There was
nothing to mark as an orphan, so no `trade_log.jsonl` write and no
STOP-AND-ASK gate were ever needed here — this stage turned out to be a pure
accounting question, not a live investigation.

Diffing the two runs' order_id lists directly (`comm -23` on the sorted sets
extracted from each report) isolates the exact 20-row gap:

- **100% of the 20 gap rows have `reason=orphaned_recent`; zero are
  `reason=open_position`.**
- Symbols: AMAT (3), MU (4), CRM (2), COP (1), CAT (4), MSFT (2), UNH (1),
  PLTR (2) — none of these appear in either account's currently-open-position
  list (cross-checked against the same D4.4 preview's captured `open_syms`).

**Explanation:** `backfill_trade_log_source.py`'s `orphaned_recent` bucket
sorts its full eligible pool (~437 rows within the 30-day lookback) by
timestamp and takes only the earliest `--max-candidates` (default 50). Since
the eligible pool is far larger than the cap, *any* small change to what
counts as "unmatched" between two independent live scans ~8 minutes apart —
a new trade_log row appended, or one buy pairing off against a sell that
arrived in between — shifts which rows fall inside that top-50 prefix
window. A handful of real underlying changes can therefore surface as a
larger set of surface-level candidate-identity differences between the two
runs, without requiring 20 separate real-world events. This is expected
behavior of a capped, sorted, re-scanned-from-scratch selection — not a bug
in the backfill script, and not evidence of any additional cross-contamination
beyond what D4.3/D4.4 already fixed.
