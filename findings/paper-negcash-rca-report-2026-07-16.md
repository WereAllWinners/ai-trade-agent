# Paper Account Negative Cash — Root Cause Analysis (2026-07-16)

**Status:** Investigation complete per brief v2 (supersedes the unresolved 7/13
brief — no prior findings file for that version exists on disk to compare
against; this is the first completed pass). Read-only throughout. No code
changes, no orders, no account/DB/service mutations.

Raw evidence, full cash ledger, and the attribution table live in
`paper-negcash-2026-07-16.md`. This document is the explanation only —
remediation is explicitly out of scope and will be scoped in a separate
follow-up brief after human review.

---

## Executive summary

On 2026-07-02 at 7:53pm CDT, Jerome manually stopped all three trading services
(`ai-trading-bot`, `ai-trading-bot-live`, `ai-options-bot`) from the command line
to work on Sprint 03 directly. That stop was intentional and correctly scoped —
but the options bot was never restarted afterward. It came back up 10 days and
~12 hours later, at 7:44am CDT on 2026-07-13, not because anyone deliberately
resumed trading, but because the machine rebooted and the service is enabled to
auto-start.

During that 10.5-day gap, the options bot held four long call-option positions —
1 NVDA $200 call, 3 NVDA $197.50 calls, the second of two MSFT $382.50 calls, and
2 AAPL $300 calls — that had been legitimately opened by the agent in the days
just before the shutdown. Two of those contracts (the NVDA and MSFT calls)
expired on 2026-07-10, squarely inside the outage. The AAPL calls expired
2026-07-13, the very morning the service happened to come back — leaving almost
no time to react before market close.

Nothing was watching these positions. The code that is supposed to close options
before expiry — the "2-DTE exit" — only runs when the options daemon is running a
session; it cannot fire on a stopped process. So all four contracts rode into
expiration still in the money. Alpaca, like essentially all brokers, automatically
exercises long in-the-money options at expiration by default — this is standard
behavior, not a bug in Alpaca's system, and the account's own paper-trading
"guardrails" have no say in it because exercise isn't an order our code submits;
it's the broker converting the option into a stock purchase at the strike price
on its own initiative. Each exercise forced a full-price stock buy: 100 shares of
NVDA at $200 (twice, plus 300 more at $197.50), 100 shares of MSFT at $382.50, 200
shares of AAPL at $300. Combined, that's $177,500 of stock the account was forced
to buy with cash it didn't have set aside for it — because the premium paid for an
option (a few hundred dollars) is a tiny fraction of the cash needed to actually
buy the underlying shares outright.

Cash went from a healthy +$87,741.65 to -$29,758.35 in a single overnight
settlement batch (the NVDA/MSFT exercises posting together), then to -$89,758.35
three days later when the AAPL exercise posted. Two unrelated stock sells on
2026-07-16 partially clawed it back to the current -$58,736.66.

None of the app's pre-submit cash guards had any chance to catch this, because
none of the money-losing events were orders our code submitted. They were
broker-side exercises. The guards that exist — checking live cash immediately
before every `submit_order` call — are real, working, and were not the point of
failure. The point of failure was that the code responsible for *preventing this
category of event from ever arising* (closing ITM options before they expire)
simply never got a chance to run for over a week and a half, because the process
that hosts it was off.

Two smaller, independently-real defects compounded the exposure but did not
cause it: most of these four contracts never had a broker-side protective stop
attached in the first place (Alpaca kept rejecting the stop submissions as
"potential wash trades"), and one of the exit attempts that *was* made, before the
outage even began, was logged internally as a completed, profitable sale despite
the real broker order having expired unfilled — meaning even a human glancing at
the trade log that week would have seen a false "closed" record for that
position.

---

## The mechanism, step by step

**1. Entries (legitimate, agent-driven, not at fault).**
Between 2026-07-01 and 2026-07-02, the options bot bought, in the ordinary course
of its trading sessions: 1 contract NVDA260710C00200000 (`b4fd5ccf`, 2026-07-01
14:05:48Z), 3 contracts NVDA260710C00197500 (`139d9530`/`7f23d4cb`/`8b2dd39b`,
2026-07-02 15:10–19:05Z), 2 contracts MSFT260710C00382500 (`59323df4`/`6165fb5a`,
2026-07-01 15:43–18:04Z), and 2 contracts AAPL260713C00300000 (`e4facc5e`/`f29b4a43`,
2026-07-02 13:37–15:06Z). All eight fills are logged with matching order_ids in
`logs/options_trade_log.jsonl` — normal, intended trades.

**2. Most of these positions never got broker-side downside protection.**
Immediately after most of these buys, `options_agent.py` attempts to submit a GTC
stop-limit SELL as a protective stop (`options_agent.py:988`-ish, "GTC stop-limit
SELL placed" on success). For NVDA200C (2026-07-01T14:05:48Z) and all three
NVDA197.5C lots (2026-07-02), this submission was **rejected by Alpaca**:
```
⚠️  GTC stop submission failed for NVDA260710C00197500: {"buy_limit_price":"3.27",
"code":40310000,"message":"potential wash trade detected. use complex orders",
"reject_reason":"buy order exists, sell limit price should be greater than
existing buy limit price","sell_limit_price":"1.47"}
```
Sessions throughout this period logged `"⚠️  RECONCILE: N unprotected options
position(s) — broker-side stop missing"` — a warning that was firing routinely and
was not specific to this incident. The one MSFT382.5C lot that *did* get a stop
attached later filled it correctly and independently at the broker on 2026-07-08,
proving broker-resident GTC orders keep working even while the daemon is down —
but that safety net simply wasn't present for the other three contracts.

**3. One exit attempt, before the outage, was logged as successful when it
wasn't.** On 2026-07-01, NVDA200C's unrealized P&L crossed the +50% take-profit
threshold and `manage_existing_positions()` submitted a DAY-TIF limit SELL. It
didn't fill within 5 minutes, so `_fill_timeout_retry()` canceled it and
resubmitted more aggressively — standard behavior for DAY exits. That resubmitted
order also never filled; broker records show it sitting open until it **expired,
unfilled, at market close** (2026-07-01T20:00:01Z). But
`manage_existing_positions()` (`options_agent.py:1223-1258`) writes the
"closed"/profit record and fires the `trade_executed` alert immediately after
calling `submit_order`/`_fill_timeout_retry`, without ever checking the resulting
order's actual terminal status:
```
19:11:51Z  [ALERT:AlertLevel.INFO] trade_executed — OptionsAgent: SELL 1
           NVDA260710C00200000 @ $4.40 | ✅ PROFIT +$184.00 (+71.9%)
20:00:01Z  (the real order actually EXPIRES, unfilled, ~49 minutes later)
```
The internal trade log said this position was closed at a profit. The broker
still held it, fully exposed, for the next nine days. This is a real,
independently-confirmed defect — but it is dated 2026-07-01, before the shutdown,
so it is a contributing factor, not the trigger.

**4. The shutdown.** At 2026-07-02T19:52:36 through 19:55:09 CDT, user `zgx`
issued four sequential `sudo systemctl stop` commands from an interactive shell
(`~/personal-projects/ai-trade-agent`), stopping `ai-trading-bot.service`,
`ai-trading-bot-live.service`, `ai-options-bot.service`, and
`ai-health-server.service` in turn. This lines up exactly with the still-uncommitted
Sprint 03 working tree (34 modified files, last commit 2026-06-16) and prior
session notes that Jerome stopped all services to work through that sprint
directly. **This action was reasonable on its own terms.** The failure is not
that the services were stopped — it's that nothing then watched, or was designed
to watch, the four already-open, mostly-unprotected option positions during the
stop.

**5. The outage.** `ai-options-bot.service` logged **zero lines of any kind**
between 2026-07-03T00:00 and 2026-07-10T00:00 (verified directly against
`journalctl`). There is no start/stop transition anywhere in that window — the
process was simply not running. `manage_existing_positions()`, the only code that
evaluates take-profit/stop-loss/DTE thresholds and submits exit orders for open
option positions, cannot run when the hosting process doesn't exist. The 2-DTE
exit rule (`exit_dte_threshold = 2`, `options_agent.py:297`, checked at
`1128-1137`) is correct as written — it simply never got invoked for these
positions during the only window that mattered.

**6. Expiration and auto-exercise.** NVDA260710C00200000 and NVDA260710C00197500
(3 contracts) and MSFT260710C00382500 (the unprotected lot) all expired
2026-07-10, entirely inside the outage — no session ran that day, or any day from
7/3 through 7/12, to even notice these positions were three days, then two days,
then zero days from expiry. Alpaca's standard practice is to automatically
exercise long options that are in the money at expiration (this is the industry-
and broker-standard default, not an app-specific setting anywhere in this
codebase — grepping the entire repo for `OPEXC`/`OPASN`/`OPEXP`/exercise-handling
returns zero hits; there is no code path that even knows this feature exists,
let alone reacts to it). Each exercise converts the contract into a real stock
purchase at the strike:

| Contract | Qty | Strike | Forced stock buy |
|---|---|---|---|
| NVDA260710C00197500 | 3 | $197.50 | 300 sh × $197.50 = **$59,250.00** |
| MSFT260710C00382500 | 1 | $382.50 | 100 sh × $382.50 = **$38,250.00** |
| NVDA260710C00200000 | 1 | $200.00 | 100 sh × $200.00 = **$20,000.00** |

These three settled together in a single overnight batch, posted
2026-07-11T08:24:03Z: cash went from **+$87,741.65 → -$29,758.35** in one shot —
a single event, three orders of magnitude past the brief's "hundreds of dollars"
band for auto-exercise, let alone the "tens of dollars" band for slippage.

**7. The one contract with a real chance, and why it still failed.** The
`ai-options-bot.service` came back up at **2026-07-13T07:44:03 CDT** — not from a
deliberate `systemctl start`, but because the machine rebooted (confirmed:
`NetworkManager.service` reload logged at 07:42:04, all three trading services
logged "Initialized" in the same second, 07:44:03 — the signature of a boot
sequence, not a selective restart; all three units are `systemctl enable`d and
therefore auto-start). AAPL260713C00300000 expired that same day. The daemon had
from 8:30am CDT (first session post-restart) to market close to close 2 contracts
that were, by then, up +179.7% — deeply in the money. It tried: canceled the
stale GTC stop, submitted a DAY-TIF take-profit SELL at 09:35:13 CDT for a real
profit alert (`+$2,525.00`), then — per the order history — a **second, larger**
exit attempt for the full 2-contract position at 10:30am CDT, retried once via
`_fill_timeout_retry` (canceled after 5 minutes, resubmitted 1% more
aggressively), and that resubmission sat unfilled until it **expired at market
close** (2026-07-13T20:00:05Z). With no further retry logic for a second failed
DAY-TIF exit and no time left in the trading day, the 2 remaining contracts rode
into expiration and were exercised: **-$60,000.00**, posted 2026-07-14T01:00:17Z,
taking cash to **-$89,758.35**.

**8. Partial recovery.** Two unrelated stock sells (ABT, 305 shares combined) on
2026-07-16T13:52:56Z brought cash to the current **-$58,736.66**. This is
ordinary trading activity, not a fix — cash remains deeply negative.

---

## Attribution table (annotated)

See `paper-negcash-2026-07-16.md` §1b.4 for the full table with evidence. Summary:

| Debit | Amount | Origin | Why |
|---|---|---|---|
| NVDA260710C00197500 ×3 exercise | -$59,250 | **BROKER** | No order_id exists; daemon confirmed down |
| MSFT260710C00382500 ×1 exercise | -$38,250 | **BROKER** | No order_id exists; daemon confirmed down |
| NVDA260710C00200000 ×1 exercise | -$20,000 | **BROKER** | No order_id exists; sole exit attempt (pre-outage) expired unfilled |
| AAPL260713C00300000 ×2 exercise | -$60,000 | **BROKER** | No order_id exists; exit attempted same-day post-restart, expired unfilled |

All four crossing-point debits are BROKER-origin. All eight entry fills that
created these positions are AGENT-origin (matched cleanly to
`logs/options_trade_log.jsonl` by order_id). **No ORPHAN (unlogged
manual/script) activity was found anywhere in or near the crossing window.**

---

## Hypotheses eliminated

- **B — reservation released at submit, not fill.** Confirmed as real, pre-existing
  behavior (release-at-submission, in a `finally` block, both agents) — but
  irrelevant to this incident. None of the four crossing-point debits ever went
  through `reserve_cash`/`submit_order` at all; they are broker-generated
  non-trade activities. Eliminated as the cause of *this* incident; remains a
  latent gap for ordinary cross-bot BUY races, with no evidence it manifested in
  the observed window.
- **C — slippage, no buffer.** Confirmed as a real, pre-existing gap (sizing uses
  pre-slippage price; the post-slippage actual cost is never re-checked against
  cash before order submission) — but the crossing debits here have no relation
  to slippage; they are fixed at the strike price by contract terms, not by any
  fill-price simulation.
- **D — unguarded order path.** Every BUY-side `submit_order` call site in the
  repo was enumerated; both agents' entry paths re-check live `account.cash`
  immediately pre-submit. One untracked, never-executed diagnostic script
  (`scripts/tools/bracket_probe.py`) is unlogged and ungated — flagged as a latent
  risk, absent from `.bash_history`, and structurally incapable of producing a
  no-order_id `OPTRD`/`OPEXC` activity regardless. Eliminated as the cause.
- **E — manual/script activity (ORPHAN fills).** `.bash_history` was checked for
  every `scripts/tools/*` invocation; none fall in the 2026-07-02–2026-07-13
  window. The crossing debits carry no order_id at all, so ORPHAN classification
  doesn't even apply to them by definition. Eliminated.
- **Test-suite hermeticity.** Re-confirmed: no test touches a non-tmp_path DB, no
  test instantiates a real `TradingClient`. Ruled out, as it was going in.

**Confirmed: Hypothesis A (options auto-exercise), with the specific mechanism
being an unmonitored 10.5-day service outage rather than a logic defect in the
2-DTE threshold check itself.**

---

## Contributing factors vs. root cause

**Root cause:** `ai-options-bot.service` was stopped 2026-07-02T19:53 CDT for
Sprint 03 work and not restarted until an incidental machine reboot on
2026-07-13T07:44 CDT — a 10-day-11-hour gap with no session ever evaluating the
open option positions' P&L, DTE, or exit conditions, during which two of the four
contracts' expiration dates passed unattended.

**Amplifiers (would not by themselves have caused this, but widened the exposure
or removed backstops that could have limited the damage):**
1. **GTC protective-stop submissions frequently rejected by Alpaca as
   "potential wash trade"** — left 3 of the 4 later-exercised contracts with zero
   broker-side downside protection, meaning even a daemon-independent safety net
   wasn't present for most of the exposure. (Confirmed recurring, not incident-
   specific — logged in nearly every session's RECONCILE warnings.)
2. **Premature "trade_executed" logging in `manage_existing_positions()`** —
   the NVDA200C take-profit exit was logged as a completed, profitable sale on
   2026-07-01 while the real broker order expired unfilled 49 minutes later,
   uncorrected. Predates and is independent of the outage, but means the trade
   log itself would have misled anyone reviewing it that this position was
   already safely closed.
3. **GTC DTE-exit orders have no retry, re-pricing, or staleness check**
   (`_fill_timeout_retry` explicitly excludes GTC TIF), and a resting GTC exit
   causes the position to be skipped on every subsequent session indefinitely.
   Not proven to have fired for these specific four contracts (the outage alone
   fully explains the lack of monitoring), but confirmed as a real structural gap
   that would independently reproduce a similar failure mode even with the
   daemon running continuously.
4. **No code anywhere reacts to Alpaca's own exercise/assignment/expiration
   activity feed** (OPEXC/OPASN/OPEXP) — zero hits repo-wide. There is no
   automated detection or alerting for this class of event, with or without the
   daemon running, so even a shorter outage carrying the same expiry dates would
   have gone unnoticed until someone checked cash directly.
5. **Startup preflight check never inspects `account.cash`.** The daemon came
   back online on 2026-07-13T07:44:04Z already $29.8K cash-negative and logged
   `"preflight: OK"`.
6. **Margin/RegT buying-power figures stay large and positive throughout**
   (current: buying_power +$504,977, non_marginable_buying_power +$57,882,
   against cash -$58,737) — this is standard 2x-RegT margin-account behavior, not
   a bug, but it means equity/buying-power alone (what the preflight log prints)
   look completely healthy and would not prompt a human to check cash
   specifically.

---

## Confidence assessment

**High confidence in the root cause.** The service-outage window (from `sudo`
audit log and `journalctl`, both authoritative, both precise to the second) maps
exactly onto both expiration dates. The cash-ledger replay independently
reconstructs the broker's own reported current cash to within $0.14 across 4,558
events, which validates that no activity type or amount was missed. The
crossing-point debits are broker-generated with no order_id, which is dispositive
for BROKER origin — this isn't an inference, it's what the data literally is.

**Lower confidence, flagged as open questions below:** the precise reason the
AAPL exit attempt's second (larger) DAY-TIF resubmission failed to find a
counterparty on 2026-07-13 — the code-level mechanism (single retry, no further
escalation) is confirmed, but *why* the market didn't fill a limit order on a
position that was +130-180% ITM (thin quotes near expiry vs. a mispriced limit)
cannot be determined from broker activity/order records alone; it would need
historical NBBO/quote data this investigation didn't pull.

**No evidence contradicts the outage-driven auto-exercise explanation.** All
eliminated hypotheses were eliminated by direct evidence (order_id absence,
committed test hermeticity, bash-history gaps), not by process of elimination
alone.

---

## Quarantine boundary

**Anomaly window for the training pipeline: 2026-07-02T19:53:10-05:00 (options
service stop) through 2026-07-16T13:52:56Z (latest activity pulled at time of
this report), cash negative continuously from 2026-07-11T08:24:03Z onward.**
Any options-side training examples, outcome records, or fine-tuning data drawing
on this window should be treated as anomalous — the account was operating outside
its designed cash-safety invariant for the second half of it, and the four
exercised positions' P&L/outcome records (if any exist in `outcomes`/DB) do not
reflect a normal, agent-controlled trade lifecycle. Per Preconditions,
`FINETUNE_ENABLED=false` currently, so no live contamination risk exists right
now — this boundary is for when/if the pipeline is re-enabled and needs to decide
what to exclude or flag.

---

## Open questions for the human

1. **Was the 2026-07-02 shutdown known to leave open option positions exposed
   through their expiration dates?** If Sprint 03 work was expected to take
   multiple days, was there ever a plan to either close open option positions
   before stopping the daemon, or to check on them manually during the outage?
   This investigation has no way to know intent — only that no such action
   appears in the account's activity or order history during the outage.
2. **Was the 2026-07-13 reboot itself intentional**, or did the machine restart
   for an unrelated reason (power event, kernel update, etc.)? The services
   auto-starting is a `systemctl enable` consequence either way, but it matters
   for understanding whether "resume trading" was ever a deliberate decision on
   that date.
3. **How many more contracts, past these four, were sitting unprotected** (GTC
   stop rejected, per the recurring RECONCILE warning) at any given time during
   normal operation, unrelated to this specific outage? This report did not
   quantify the wash-trade rejection rate outside the incident window.
4. **Does the `outcomes`/training-data pipeline contain rows for these four
   contracts already**, and if so, do they reflect the false "closed at
   +71.9%" record from step 3 rather than the real -100%-of-premium-plus-forced-
   stock-purchase outcome? Not checked in this pass — would require a DB query
   this brief didn't authorize (read-only against files/broker only, not a full
   DB audit).
5. **Is there a reason the current 13 open GTC SELL orders should be trusted**,
   given the same wash-trade rejection pattern was observed elsewhere? Not
   verified whether any of the 24 current stock positions are similarly
   unprotected.

---

*End of RCA. Per brief scope: no remediation options, code fixes, or liquidation
plans are proposed here. This document and `paper-negcash-2026-07-16.md` are the
full deliverable; a follow-up brief will scope remediation after human review.*
