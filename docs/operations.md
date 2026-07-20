# Operations Runbook

Operational procedures for running the trading services safely. Written as
part of R1 (paper negative-cash incident remediation).

## Custody rule (R1 WI-9)

**No trading service may be manually stopped while it has custody of open
positions or open orders unless (a) positions are flattened or protection
is verified first, or (b) the position watchdog (`ai-position-watchdog.timer`)
is confirmed active.**

### Why

On 2026-07-02, `ai-options-bot.service` was manually stopped for Sprint 03
development work while it held four long call option positions. It was not
restarted for 10.5 days. Nothing was watching those positions in the
meantime — the stop itself was reasonable, but nothing enforced or even
checked the custody rule above before it happened. All four contracts rode
past their expiration dates unmonitored and were auto-exercised by Alpaca,
forcing $177,500 in stock purchases the paper account didn't have cash for.
Full account: `findings/paper-negcash-rca-report-2026-07-16.md`.

The position watchdog (R1 WI-2, `ai-position-watchdog.timer`, every 15 min)
now makes "service down with open custody" visible even when a service
*is* stopped — but the custody rule above is about not creating that
exposure unnecessarily in the first place when it's avoidable.

### How to check before stopping a service

```bash
python scripts/tools/pre_stop_check.py <service-unit>
```

Example:
```bash
$ python scripts/tools/pre_stop_check.py ai-options-bot.service
Custody check: ai-options-bot.service (paper/options)
  Watchdog (ai-position-watchdog.timer) active: True
  Positions: 0  Orders: 0
  SAFE TO STOP
  - No open positions or orders for this bot.
```

Exit code 0 = safe to stop, 1 = not safe (read the reasons printed), 2 =
couldn't check (unknown unit or missing credentials).

**This tool is advisory only** — it does not wrap, call, or block
`systemctl stop`. It's a check to run first, not an enforcement mechanism.
If it reports NOT SAFE, either flatten/protect the open positions/orders
first, or confirm the watchdog timer is running
(`systemctl is-active ai-position-watchdog.timer`) before proceeding.

## HALT_BUYS — manual clear procedure (R1 WI-7)

`scripts/halt_buys.py` is a fail-closed flag file checked at the top of
both agents' BUY paths (`autonomous_agent.py::execute_trade()`,
`options_agent.py::execute_options_trade()`). It is written by
`position_watchdog.py`'s `check_cash_invariant()` whenever an account's
`cash < 0` or `non_marginable_buying_power < 0` — the exact failure mode
of the 2026-07 incident — and checked every 15 minutes (the watchdog
timer's own cadence) for as long as the breach persists.

**By design, there is no code path that clears the flag.** Presence alone
is authoritative — the file's content (reason, timestamp, cash snapshot)
is informational only, never parsed to decide whether to unblock. A flag
from days ago blocks exactly as hard as one from a minute ago. This is
deliberate: an automated auto-clear would recreate the same "nothing was
watching" failure mode this brief exists to close.

Flag paths (paper and live are independent — a live breach never blocks
paper BUYs or vice versa):
```
logs/halt_buys.flag        # paper
logs/halt_buys_live.flag   # live
```

**To clear, once you've confirmed the underlying condition is actually
resolved** (re-check real account cash — don't trust the flag's own
stale snapshot):
```bash
python scripts/tools/position_watchdog.py --dry-run   # confirm no breach reported
rm logs/halt_buys.flag        # or halt_buys_live.flag for the live account
```
If the underlying cash breach isn't actually fixed yet, the next watchdog
cycle (≤15 min) simply rewrites the flag — clearing it early doesn't
skip that, it only creates a brief unprotected window. Fix cash first,
then clear.

**Real incident, 2026-07-20**: this flow was exercised for real during R1
implementation — paper cash had drifted to -$0.25 (residual float/fee
drift after WI-1's initial restoration). The watchdog correctly detected
the breach, alerted, and wrote the flag every cycle from 07:55 to 09:03.
Resolved via a small signed-off `trim_exercised_positions.py` run (sold
$5.25 of AAPL, cash → $5.00 buffer), then the stale flag was cleared per
the procedure above.

## Alert channels

`scripts/alerts.py::send_alert()` fans out to email (SMTP, configured
live in `.env`) and `logs/alerts.jsonl`. **Telegram is wired in code but
deliberately unused** — the bot token / chat ID path exists and will
silently no-op without configured credentials. Don't assume alerts are
reaching Telegram; email + `alerts.jsonl` are the only channels actually
in use. `health_server.py`'s `/health` endpoint only surfaces
`WARNING`/`CRITICAL`-level alerts from `alerts.jsonl` — `INFO`-level
entries (e.g. WI-8's correction rows) never appear there.

## Expiry sentinel tiers (R1 WI-3)

`scripts/tools/expiry_sentinel.py`, run daily at 8am via
`ai-expiry-sentinel.timer`, scans held option contracts and classifies
each by days-to-expiry:

| Tier | Threshold | Note |
|---|---|---|
| WARN | ≤ 5 calendar days | |
| CRITICAL | ≤ 2 trading days | Takes priority over WARN if both match. Weekday-only approximation — no NYSE holiday calendar, an accepted, documented limitation. |

Exposure is always shown at worst-case value (`qty × 100 × strike`)
regardless of moneyness — a deliberately conservative default for a
monitor, never under-warns. Dedup is once-per-calendar-day per
`{tier}_{account}_{symbol}` key (`alert_once_per_day`), so an unresolved
position re-alerts daily until it's gone, not just once.

## Wash-trade rejections (R1 WI-6)

Diagnosed, not fixed, in R1 — see
`findings/r1-wash-trade-diagnosis-2026-07-20.md` for the full pattern
breakdown (Pattern A: protective stop rejected by its own just-filled
entry, 65 occurrences; Pattern B: new entry blocked by a resting stop
from an earlier lot, 25 events) and the proposed fix shapes. No code
changed by this finding.

## Correction entries reference (R1 WI-5 / WI-8)

Three retroactive, sign-off-gated, idempotent entries were appended to
`logs/options_trade_log.jsonl` / `logs/alerts.jsonl` via
`scripts/tools/apply_r1_log_corrections.py` — never modifying or
deleting an original line, only appending truth on top of it:

| Item | Contract | What it corrects |
|---|---|---|
| `msft_backfill` (WI-5) | MSFT260710C00382500 | Real GTC-stop exit (-38.3%) that had zero app record at all |
| `nvda_correction` (WI-8) | NVDA260710C00200000 | False "closed +71.9%" record — order actually expired unfilled, position was exercised |
| `aapl_correction` (WI-8) | AAPL260713C00300000 | False "closed +179.7%" record — order actually expired unfilled, position was exercised |

All three carry a real `order_id` for `msft_backfill` (broker-confirmed
fill, `action: 'sell'`) or `type: 'correction'` with **no `action` key**
for the two expiry corrections (nothing filled, nothing true to claim —
this is what makes `options_outcome_tracker.py`'s FIFO matcher, which
only matches `action in ('buy','sell')`, ignore them by construction).
Re-running `apply_r1_log_corrections.py --execute --item <name>` on an
already-applied item is a safe no-op — each checks for its `order_id` (or
the errata's marker string) before writing.

## Accepted gaps — found during R1, owned by R2

Two structural findings surfaced while planning R1 that are real,
documented, and deliberately **not fixed here** — R1's scope was the
incident's exit-side monitoring gap specifically, not a full audit of
every fill-confirmation path in the codebase:

1. **Stock agent's cash hard-gate can deadlock its own defensive sells.**
   `autonomous_agent.py::run_trading_session()`'s cash gate `return`s from
   the *entire* session function when cash is too low, not just the BUY
   path — so `review_held_positions()` (the function that would sell
   positions on stop-loss/take-profit) never runs at all once cash goes
   negative. The stock bot ran 2026-07-07 → 2026-07-13 unable to execute
   even defensive sells because of this; the same would happen on the live
   account under the same condition. R1's WI-4 reconcile call was
   deliberately placed *before* this gate so reconciliation at least isn't
   blocked by it, and WI-7's HALT_BUYS check was deliberately placed
   strictly inside the BUY branch (not at this outer gate) so as not to
   compound the same deadlock on a second axis — but the underlying
   deadlock in `review_held_positions()` itself remains unfixed.
2. **The BUY-side entry path has the same fill-blindness class WI-4 fixes
   on the exit side.** `execute_options_trade()` logs the entry the
   instant it submits a DAY-TIF limit order that can rest unfilled at the
   broker — the same fact WI-6's wash-trade diagnosis leans on (the entry
   can still be open when the protective stop is submitted ~40 lines
   later). An entry that expires unfilled leaves a phantom BUY in the
   JSONL the same way the incident's phantom exits did, corrupting FIFO
   buy/sell matching the same way. Fixing this well wants the outcome
   pipeline to be healthy first (per the Addendum A finding that it's
   currently frozen), so it's grouped with that repair in R2 rather than
   bolted on here.
