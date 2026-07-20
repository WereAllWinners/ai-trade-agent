# R1 WI-6 — Wash-Trade Rejection Diagnosis (2026-07-20)

Read-only diagnosis per the brief. No code changes. Data pulled from
`journalctl -u ai-options-bot.service` (full retained history,
2026-06-12 → present) and direct read-only Alpaca order lookups.

## Headline finding: two distinct patterns, not one

The original hypothesis (from the RCA) was a single pattern: a protective
GTC stop rejected because the entry BUY it protects is still open at the
broker. That pattern is real and confirmed — but a second, previously
unnoticed pattern is at least as frequent: **a brand-new entry BUY itself
gets rejected because a resting protective stop from an earlier, unrelated
lot in the same underlying is still open.** Both are the same root defect
class as WI-4 (proceeding to the next action without confirming the
previous order's terminal state) — here it surfaces as a broker-side
rejection instead of a false internal completion record.

## Volume: chronic and ongoing, not incident-specific

| | Count | Window |
|---|---|---|
| Pattern A (protective stop rejected) | **65** | 2026-06-12 → 2026-07-02 |
| Pattern B (new entry rejected) | **25** confirmed distinct events (50 raw log lines — each event is logged twice: once as an ERROR, once as the `trade_failed` alert) | 2026-06-12 → 2026-07-02 |
| **Total distinct collision events** | **90** | across the full retained journal |
| Successful protective-stop placements | 38 | same window |

This directly answers the RCA's open question #3: **how many contracts,
past the four exercised ones, were sitting unprotected at any given time**
— a lot, continuously. Of every ~103 attempts to attach a protective stop
(38 succeeded + 65 Pattern-A failures), **roughly 63% failed on first
attempt**, with no retry logic anywhere in the code for this specific
failure (confirmed: `_fill_timeout_retry()` only applies to DAY-TIF P&L/DTE
exits, never to the entry-time protective stop). Zero occurrences since the
2026-07-13 restart, but zero new options entries have happened since the
restart either (confirmed via journalctl) — this is an absence of
opportunity, not evidence the issue is resolved.

## Pattern A — protective stop rejected by its own just-filled entry

**Mechanism, confirmed:** `execute_options_trade()` submits the entry BUY
as a marketable `LimitOrderRequest` (`options_agent.py:918-936`) which, in
every sampled case, fills within milliseconds of submission (`submitted_at`
== `filled_at` to the microsecond in Alpaca's own order records). The code
then submits the protective GTC `StopLimitOrderRequest` (`options_agent.py:980-989`)
immediately after, with **zero wait or fill-confirmation in between** — no
`get_order_by_id` call, no delay, nothing. Alpaca's rejection payload names
an `existing_order_id` and states `"reject_reason":"buy order exists, sell
limit price should be greater than existing buy limit price"`.

**Cross-checked 5 of 65 occurrences directly against real order history**
(read-only `get_order_by_id`), spanning the full date range and three
different underlyings (SPY, TSLA, NVDA):

| Rejection timestamp | Symbol | `existing_order_id` | Confirmed as |
|---|---|---|---|
| 2026-06-12T12:05:06 | SPY260622C00756000 | `21f50130-...` | The just-filled entry BUY, same instant |
| 2026-06-16T08:30:54 | SPY260623C00755000 | `cae9535a-...` | The just-filled entry BUY, same instant |
| 2026-06-18T12:47:16 | TSLA260626P00387500 | `17275cd1-...` | The just-filled entry BUY, same instant |
| 2026-06-25T08:46:11 | SPY260702P00715000 | `820ea53e-...` | The just-filled entry BUY, same instant |
| 2026-06-29T11:25:42 | NVDA260708P00190000 | `725ce8d9-...` | The just-filled entry BUY, same instant |

**5/5 confirmed.** Also confirmed on the three NVDA260710C00197500 lots
specifically implicated in the incident: each of the three separate
rejections correctly references its own corresponding buy lot (`139d9530`,
`7f23d4cb`, `8b2dd39b`), not a stale earlier one — the pattern holds even
across repeated same-symbol buys within one day.

**A structural note, not just a timing race:** a protective stop-loss is
*designed* to sit below the entry price (`stop_trigger = fill_price * (1 +
stop_loss_pct)` with `stop_loss_pct = -0.50`, then `stop_limit = stop_trigger
* 0.90` — roughly 45% of the entry fill price). Alpaca's own stated
constraint ("sell limit price should be greater than existing buy limit
price") is therefore in direct tension with what a legitimate protective
stop necessarily looks like. Some stops do succeed (38 of 103), so this
isn't an absolute rule — but it means the current "fire the stop
immediately, no wait" approach is fighting the wash-trade check's design
intent on every single attempt, not occasionally colliding with it.

## Pattern B — new entry blocked by a resting stop from an earlier lot

**Newly found during this diagnosis, not previously documented.** 25
distinct events where an entry BUY itself fails with `"❌ Options trade
execution failed"` / a `trade_failed` alert, `reject_reason:"sell order
exists, buy limit price should be less than existing sell limit price"`.

**Confirmed via direct order lookup** on the clearest example
(`existing_order_id` `7abcf423-...`, referenced by two separate rejections
76 minutes apart, 2026-06-15T11:13:06 and 2026-06-15T12:29:13): this order
is a `SELL STOP_LIMIT` on
`SPY260622C00758000`, **submitted 2026-06-15T15:04:41Z and not filled
until 2026-06-16T15:46:45Z — resting open for over 24 hours.** Every new
options entry the agent attempted on a colliding SPY contract during that
entire window was blocked by this single stale order.

**Mechanism:** GTC protective stops (when they do attach successfully) rest
indefinitely until filled or canceled — by design, so they survive a
daemon restart or outage (the exact protection the incident needed more
of, elsewhere). But nothing in the entry path checks for or works around a
resting opposite-side order in the same underlying before submitting a new
BUY. The consequence is different from Pattern A's: not a naked position,
but a **silently missed trading decision** — the LLM decided to buy, the
order was rejected, and (confirmed: no retry exists for this path either)
the attempt is simply logged as failed and abandoned for that cycle.

## Does agent sequencing cause it?

**Yes, unambiguously, for both patterns.** Neither is a rare timing
coincidence:
- Pattern A is 100% attributable to submitting the protective stop with no
  fill-confirmation or delay after the entry — confirmed by the
  `submitted_at`==`filled_at` timing on every sampled entry order, and by
  the structural tension described above.
- Pattern B is 100% attributable to the entry path having no awareness of
  resting opposite-side orders in the same symbol before submitting a new
  one.

## Proposed fix shape (for sign-off — not implemented here)

Two candidates, not mutually exclusive:

1. **Primary candidate: submit the entry as a bracket/complex order**
   (stop-loss attached as a child leg of the same parent order), rather
   than two sequential independent orders. This is literally what Alpaca's
   own rejection message suggests ("use complex orders") and would
   sidestep the self-trade heuristic entirely, since the stop is
   recognized as an intentional child of the same trade rather than a
   second, independently-adversarial order. `risk_reconciler.py` notes
   Alpaca doesn't support GTC brackets on **fractional-share stock**
   quantities — but option contracts are always integer quantities, so
   this specific limitation likely doesn't apply here. **Flagged as
   needing empirical confirmation before assuming it works, not assumed.**
2. **Fallback/mitigation for Pattern A specifically:** a short wait plus a
   bounded retry loop before giving up on stop attachment, mirroring the
   existing `_fill_timeout_retry()` pattern already used for DAY-TIF exits.
   Untested whether Alpaca's dedup window is short enough for a brief wait
   to help — would need empirical testing, not assumed to work.
3. **For Pattern B:** either (a) check for and wait out / cancel-and-replace
   a resting opposite-side order in the same underlying before submitting
   a new entry, or (b) accept it as bounded, low-severity friction (a
   missed entry, not a safety gap) and let the LLM's normal candidate
   scoring move on — a smaller fix than Pattern A's, since Pattern B
   doesn't leave anything unprotected, it just occasionally blocks a new
   trade.

## Relationship to the 2026-07 incident

Pattern A directly explains why 3 of the 4 later-exercised contracts
(NVDA200C, all three NVDA197.5C lots, one of two MSFT382.5C lots) had no
broker-side protective stop at all — their entries collided with exactly
this pattern (confirmed in the original RCA with the same evidence
re-confirmed here). Had the outage not also occurred, these positions
would still have been exposed to the same accumulated damage the fourth
un-exercised, protected MSFT lot avoided (it filled its stop independently
at the broker on 2026-07-08, while the daemon was down — proof that a
successfully-attached stop is a real, daemon-independent safety net; the
problem is that ~63% of attempts to attach one fail).

## Open questions this diagnosis could not resolve read-only

- The exact condition Alpaca uses to allow 38 of 103 stop attachments to
  succeed despite the same near-zero timing gap — this would need Alpaca's
  own internal wash-trade-check implementation or a support inquiry to
  fully explain; not observable from client-side order records alone.
- Whether bracket/complex orders genuinely sidestep this for options
  contracts specifically (candidate fix #1) — needs a small, deliberate
  test in a future remediation brief, not assumed here.
