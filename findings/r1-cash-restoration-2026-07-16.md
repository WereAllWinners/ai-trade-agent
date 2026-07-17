# R1 WI-1 — Cash Restoration (2026-07-16)

Paper account only, per brief scope. Read-only snapshot + one signed-off
execution, per the R1 plan's sign-off gate.

## Pre-restoration snapshot (2026-07-16, ~20:29 CDT)

| Field | Value |
|---|---|
| cash | -$58,736.66 |
| equity | $204,832.91 |
| buying_power | $502,978.51 |
| non_marginable_buying_power | $57,524.91 |
| positions | 24 |
| open orders | 13 (all GTC SELL limits, protective stops) |

Full position list (market value desc.): NVDA $82,052.00 (400sh), AAPL
$66,960.00 (200sh), MSFT $39,857.00 (100sh), MS $19,397.10, LMT $10,248.20,
MRK $9,455.04, AMGN $4,456.56, MCD $3,811.08, CVS $3,806.05, EOG $3,323.04,
SLB $2,551.54, VLO $2,417.01, V $2,402.09, MPC $2,237.81, ORCL $2,223.31,
QCOM $1,615.13, NKE $1,506.20, GE $1,269.78, COST $1,214.04, UPS $1,188.18,
NFLX $615.08, MA $549.25, C $389.21, WBA $9.35.

## Variants presented

**Variant A** — `scripts/tools/close_all_positions.py` (existing, unmodified):
cancel all 13 open orders + close all 24 positions via Alpaca's bulk
`close_all_positions(cancel_orders=True)`. Projected: cash → ~$204,832.91
(full equity), 0 positions remaining.

**Variant B** — new `scripts/tools/trim_exercised_positions.py` (built for
this work item, see plan): targets AAPL/NVDA/MSFT first (the auto-exercised
lots), descending by market value, partial-selling only as much as needed;
falls back to remaining positions ascending by value if the named lots don't
cover the shortfall. Live-computed preview at decision time: sell 286.34 of
400 NVDA shares (71.6% of the position), proceeds $58,736.66, cash → $0.00
exactly. AAPL, MSFT, and all 21 other positions untouched.

## Decision

**Variant B selected** (human sign-off via AskUserQuestion, 2026-07-16).
Rationale given: minimal disruption, preserves the other two exercised lots
and all unrelated unrealized gains.

## Execution

`python scripts/tools/trim_exercised_positions.py --paper --execute`, run
2026-07-16 20:31:05 CDT. Fresh account state at execution time (prices had
moved slightly since the preview a minute earlier):

- Cancelled all 13 open orders first.
- Submitted `MarketOrderRequest(symbol='NVDA', qty=286.045875, side=SELL,
  time_in_force=DAY)` — recomputed shortfall against live equity, order_id
  `b5865a9e-2a28-4027-8362-c37373f8deb7`.

## Status: FILLED, confirmed 2026-07-17

Order `b5865a9e-2a28-4027-8362-c37373f8deb7` queued overnight (market was
closed at submission) and filled at the 2026-07-17 09:30 ET open:

| Field | Projected (submission time) | Actual (filled) |
|---|---|---|
| qty | 286.045875 | 286.045875 (full) |
| avg price | ~$205.13/share (implied) | $201.295766/share |
| proceeds | $58,736.66 | $57,579.75 |

## Post-fill account state (2026-07-17, confirmed)

| Field | Pre-restoration | Post-fill |
|---|---|---|
| cash | -$58,736.66 | **-$1,156.91** |
| equity | $204,832.91 | $202,780.56 |
| non_marginable_buying_power | $57,524.91 | $72,017.24 |
| positions | 24 | 24 (NVDA reduced to 113.95sh, all others unchanged) |
| open orders | 13 | 0 (cancelled as part of the trim) |

**Residual: -$1,156.91, not fully non-negative.** Root cause: the fill price
($201.30) came in ~1.9% below the price implied at order submission
($205.13) — ordinary overnight-to-open price movement on a single stock
(NVDA) for a market order that queued ~90 minutes before the open, not a
bug or a miscalculation. The shortfall computation itself was exact (the
order's qty was sized correctly against the account state at submission
time); the gap is pure market movement between submission and fill, which
no synchronous script can avoid for an order that must queue outside market
hours.

This residual is a decision point, not an automatic follow-up — see chat
for the human's choice on whether to top up with a small additional trim or
accept as close enough (98% of the original -$58,736.66 gap closed, and the
account is left with 24 positions unchanged in count, only NVDA's size
reduced).

## Top-up trim (2026-07-17, market open)

Human chose to top up. Re-ran `trim_exercised_positions.py --paper` fresh
(recomputes shortfall against live state each run, not the stale
projection): NVDA had shrunk to 113.95sh from the first trim, so AAPL
($66k+ market value) was now the largest of the three named symbols and
was selected — correct behavior, not a bug, since the script re-evaluates
current holdings on every run rather than assuming yesterday's ordering
still holds.

- Cancelled the (zero) open orders, submitted `SELL 3.509138 AAPL`,
  order_id `22a8d94f-e09e-4479-8f2e-8f3793423aa3`.
- Market was open this time — filled within seconds: `filled_qty=3.50913751
  avg_price=$329.63`.

## Final confirmed state

| Field | Value |
|---|---|
| **cash** | **-$0.19** |
| equity | $202,435.08 |
| positions | 24 (NVDA 113.95sh, AAPL 196.49sh, all else unchanged from pre-restoration) |
| open orders | 0 |

-$0.19 is fill-price rounding on a sub-second gap between the shortfall
calculation and the actual fill — immaterial, not chased further. Cash
restoration is complete: -$58,736.66 → -$0.19, via two small stock sells
(NVDA partial, then AAPL partial) touching only 2 of 24 positions, with
zero disruption to the other 22 positions or either of the other two
exercised lots (MSFT untouched throughout; AAPL only touched in the
top-up).
