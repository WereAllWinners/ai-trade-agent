# Paper Account Negative Cash — Raw Evidence (2026-07-16)

Investigation-only, read-only. See `paper-negcash-rca-report-2026-07-16.md` for the
explanatory report. This document is the working evidence file per Phase 1/1b of the brief.

All timestamps UTC unless marked local (CDT = UTC-5).

---

## Preconditions

**Paper fine-tune pipeline: DISABLED.** `.env`: `FINETUNE_ENABLED=false`,
`STRATEGY_EVOLVER_ENABLED=false`. No escalation needed; training data is not being
actively contaminated by this anomaly window. (It may already contain contaminated
rows from before these flags were set false — not verified here, out of scope for
this read-only pass.)

**7/13-brief remediation items landed: NONE, as git commits.** `git log --since
2026-07-13 --oneline` returns empty. Last commit on `main` is `49f9288`
(2026-06-16). However the working tree has 34 modified-but-uncommitted files plus
~25 untracked new files (`git status` — not reproduced here in full; see repo).
This is Sprint 03 work-in-progress (per prior-session memory: "Jerome had stopped
all services to work through this sprint directly," Sprint 03 completed
2026-07-02). systemd runs whatever is on disk, not `git HEAD`, so this uncommitted
code **is** what the live services execute.

Checked specifically whether `non_marginable_buying_power` gating, fill-terminal
reservation release, or a slippage buffer landed (committed or uncommitted):
- `_settled_cash()` (`account.non_marginable_buying_power`) is **pre-existing**,
  committed in `a9a9b0f` (2026-05-28) — not new, not part of the uncommitted diff.
- Reservation release is, and always has been, release-at-submission (in a
  `finally` block), not fill-terminal. Not changed by the uncommitted diff
  (confirmed via `git diff HEAD -- scripts/db.py scripts/agents/*.py`).
- No slippage buffer exists on the reservation/gate path in either committed or
  uncommitted code.
- **None of the three 7/13 remediation items have landed, committed or
  uncommitted.** Per the brief's own instruction this would normally elevate
  Hypothesis D/A — moot here because Hypothesis A is directly confirmed by broker
  records regardless (see below).

---

## Phase 1 — Current account snapshot (as of 2026-07-16T14:xx UTC pull)

| Field | Value |
|---|---|
| cash | **-$58,736.66** |
| equity | $205,531.25 |
| buying_power | $504,977.33 |
| non_marginable_buying_power | $57,881.84 |
| regt_buying_power | $146,785.24 |
| initial_margin | $132,138.63 |
| maintenance_margin | $79,286.92 |
| last_equity | $202,698.24 |
| open positions | 24 (all common stock — no options currently held) |
| open orders | 13 (all GTC SELL limits, protective stops on stock positions) |

Note the sharp divergence between `cash` (-$58,736.66) and `non_marginable_buying_power`
(+$57,881.84) / `buying_power` (+$504,977.33): this is a 2x RegT margin account, so
margin-derived buying-power figures stay large and positive even while raw cash is
deeply negative. Anyone (human or code) reading equity/buying_power alone would see
a healthy-looking account.

Full position list: AAPL 200sh, AMGN 12sh, C 2.96sh, COST 1.29sh, CVS 35.64sh, EOG
24sh, GE 3.65sh, LMT 20sh, MA 1sh, MCD 14sh, MPC 7.32sh, MRK 73.65sh, MS 89.49sh,
**MSFT 100sh**, NFLX 9.08sh, NKE 34sh, **NVDA 400sh**, ORCL 18.04sh, QCOM 9.51sh,
SLB 54sh, UPS 10.18sh, V 6.6sh, VLO 8.04sh, WBA 0.78sh.

**AAPL (200sh), NVDA (400sh), and MSFT (100sh) are the exact share counts produced
by option-exercise conversions** (see below) — 2, 4, and 1 contracts respectively ×
100 shares/contract.

## Phase 1 — Cash timeline (activity replay)

Pulled full paginated account activity history via Alpaca REST
(`TradingClient.get`, read-only GET, `/account/activities`, `direction=asc`,
paged by `page_token`): **4,558 activities, 2025-07-21 → 2026-07-16.**
Type breakdown: FILL 3,628, FEE 916, OPEXP 5, OPTRD 4, OPEXC 4, JNLC 1 (the
$100,000 account-inception deposit, 2025-07-21).

Replayed a running cash ledger chronologically (FILL: ±qty×price×[100 if option
symbol else 1] by side; non-trade: `net_amount` as-is). **Final replayed cash:
-$58,736.80, vs. actual broker-reported cash -$58,736.66 (14¢ rounding
divergence across 4,558 events)** — this cross-check confirms the replay is
complete and methodologically sound (no missing activity types, no systematic
math error).

23 lifetime positive→negative crossings total. All but the last are small (single
dollars to low thousands) and self-correct within hours to weeks — ordinary
intraday cash-timing noise, not the subject of this brief. **The crossing that
matters** is the last one:

### The crossing

```
2026-07-11T08:23:59Z   FEE     -0.01   running:  87,741.65   (last known-positive)
2026-07-11T08:24:03Z   OPTRD  -59,250.00  running:  28,491.65   NVDA (group da3cd76f)
2026-07-11T08:24:03Z   OPEXC       0.00  running:  28,491.65   MSFT260710C00382500 exercised
2026-07-11T08:24:03Z   OPEXC       0.00  running:  28,491.65   NVDA260710C00200000 exercised
2026-07-11T08:24:03Z   OPTRD  -38,250.00  running:  -9,758.35   MSFT (group e0c04347)
2026-07-11T08:24:03Z   OPEXC       0.00  running:  -9,758.35   NVDA260710C00197500 exercised
2026-07-11T08:24:03Z   OPTRD  -20,000.00  running: -29,758.35   NVDA (group de5e67e3)
2026-07-11T08:24:10Z   OPEXP ×4    0.00  running: -29,758.35   (unrelated worthless expiries, no cash impact)
...
2026-07-14T01:00:16Z   OPTRD  -60,000.00  running: -89,758.35   AAPL (group 915601c2)
2026-07-14T01:00:16Z   OPEXC       0.00  running: -89,758.35   AAPL260713C00300000 exercised ×2
...
2026-07-16T13:52:56Z   FILL  +15,561.63  running: -74,196.72   ABT sell (unrelated stock, partial recovery)
2026-07-16T13:52:56Z   FILL  +15,459.92  running: -58,736.80   ABT sell (unrelated stock, partial recovery)
```

**Magnitude: crossing #1 alone is -$117,500 in a single batched broker settlement
event** (three contracts' worth of exercise), immediately followed 3 days later by
a second -$60,000 event. This is two to three orders of magnitude larger than the
brief's "tens of dollars → slippage" or "hundreds → auto-exercise or unguarded
path" bands — call it what it is: **auto-exercise, at a scale the account cannot
absorb.**

All four debit events (`OPTRD`) and their paired `OPEXC` records carry **no
`order_id`** — they are broker-generated non-trade activities, not fills of any
order our code submitted. This is definitional BROKER origin (see attribution
table).

### The four exercised contracts

| Symbol | Contracts | Strike buy cost | Bought (order_id) | Expiry |
|---|---|---|---|---|
| NVDA260710C00200000 | 1 | $20,000 | `b4fd5ccf` 2026-07-01 14:05:48Z @ 2.56 | 2026-07-10 |
| NVDA260710C00197500 | 3 | $59,250 | `139d9530`/`7f23d4cb`/`8b2dd39b`, 2026-07-02 15:10–19:05Z | 2026-07-10 |
| MSFT260710C00382500 | 1 (of 2 bought; other lot sold via GTC stop 2026-07-08) | $38,250 | `59323df4`/`6165fb5a`, 2026-07-01 15:43–18:04Z | 2026-07-10 |
| AAPL260713C00300000 | 2 | $60,000 | `e4facc5e`/`f29b4a43`, 2026-07-02 13:37–15:06Z | 2026-07-13 |

All 8 BUY order_ids confirmed present in `logs/options_trade_log.jsonl` +
`logs/alerts.jsonl` (AGENT origin, `OptionsAgent`, options bot session). No orphan
entries among the entries — every position that was later exercised was
legitimately opened by the agent.

---

## Phase 1b — Order attribution

### 1b.1 — Decision-log matching

The crossing-point cash debits themselves (`OPTRD` ×4, totaling -$177,500) have
**no order_id** and cannot be matched to any log by definition — Alpaca generates
them internally at expiration/exercise, with no order ever submitted by any
client. Classification: **BROKER**, all four.

The 8 BUY fills that *created* the exercised positions all match cleanly to
`logs/options_trade_log.jsonl` entries by order_id (see table above).
Classification: **AGENT**, all eight — the entries themselves were legitimate,
intentional agent trades. The problem is exclusively in exit management, not entry.

No ORPHAN (unlogged manual/script) fills were found in or near the crossing
window. Checked `~/.bash_history` for any `scripts/tools/*` invocation in the
2026-07-02 – 2026-07-13 window (liquidate_to_cash.py, cover_cash_deficit.py,
close_all_positions.py, bracket_probe.py, sync_broker_exits.py) — none found
in this window (all such invocations in history predate Sprint 03 by a wide
margin, or are unrelated to this window).

### Unlogged submit_order paths (code audit)

Every BUY-side `submit_order` call site in the repo was enumerated
(`autonomous_agent.py:750`, `options_agent.py:936`) — both write to their
respective trade_log + DB + decision_log on execution, and both re-check live
`account.cash` immediately pre-submit. One BUY-capable path,
`scripts/tools/bracket_probe.py` (untracked, diagnostic-only, hardcoded
`paper=True`), submits without any log write and without a cash gate — flagged as
a **latent risk**, not implicated here (no evidence of it ever having been run —
absent from `.bash_history` entirely).

Two SELL-side gaps were also found (protective-stop submission in
`options_agent.py:988` and `risk_reconciler.py`'s `reprotect_positions`): neither
writes a trade_log entry at submission. SELL-side gaps cannot cause negative cash
directly, but they do mean broker-side protective fills for these specific orders
would never show up as reconciled AGENT rows if matched by log alone — moot here
since the crossing debits are BROKER-type, not FILL-type, so log-matching doesn't
apply to them regardless.

### 1b.2 — Host-side corroboration

**Service stop (manual, confirmed via sudo log):**
```
2026-07-02T19:52:36-05:00  sudo ... systemctl stop ai-trading-bot.service       (user zgx)
2026-07-02T19:52:51-05:00  sudo ... systemctl stop ai-trading-bot-live.service  (user zgx)
2026-07-02T19:53:10-05:00  sudo ... systemctl stop ai-options-bot.service       (user zgx)
2026-07-02T19:55:09-05:00  sudo ... systemctl stop ai-health-server.service     (user zgx)
```
All four trading-related services stopped within a 3-minute window by user `zgx`
from an interactive TTY (pts/1), cwd `~/personal-projects/ai-trade-agent`. This
lines up exactly with the Sprint 03 uncommitted working-tree state (see
Preconditions) — consistent with prior-session memory that Jerome stopped all
services to work through Sprint 03 directly, completed 2026-07-02.

**Service restart (automatic, on boot — NOT a deliberate resume):**
```
2026-07-13T07:42:04-05:00  systemd: Reloading requested ... (unit NetworkManager.service)
2026-07-13T07:44:03-05:00  systemd: Started ai-options-bot.service
2026-07-13T07:44:03-05:00  systemd: [ai-trading-bot, ai-trading-bot-live, ai-options-bot all log "Initialized" in the same second]
```
No `sudo systemctl start ai-options-bot` appears in the sudo log or
`.bash_history` around this timestamp — all three trading services came up
together, in the same second, consistent with a **machine reboot** rather than a
selective manual restart. All three units are `systemctl enable`d, so they
auto-started. `ai-options-bot.service` first `manage_existing_positions()` /
P&L-check log line after restart: `2026-07-13T08:30:05-05:00`.

**Outage window: `ai-options-bot.service` was stopped continuously from
2026-07-02T19:53:10-05:00 to 2026-07-13T07:44:03-05:00 — 10 days, 11 hours, 51
minutes.** Confirmed by:
- Zero `journalctl -u ai-options-bot.service` log lines of any kind between
  2026-07-03T00:00 and 2026-07-10T00:00 (checked; zero total lines in that
  6.5-day sub-window).
- `journalctl -u ai-options-bot.service` shows no `Started`/`Stopped` transition
  between the 7/2 stop and the 7/13 start.
- No errors/exceptions/tracebacks logged in the outage window (there's nothing to
  log — the process was not running).

**Both expiration dates for the exercised contracts (2026-07-10 and
2026-07-13) fall inside or at the boundary of this outage.** The AAPL contract
(expiry 7/13) got exactly one session's worth of (failed) exit attempts because
the reboot happened to land on expiry morning itself, ~1h45m before market open.

**Git timeline:** no commits in the outage window (consistent with "last commit
2026-06-16"; the outage is orthogonal to git activity — the services were stopped
at the shell, not via a deploy).

### 1b.3 — Test-suite hermeticity spot-check

- All `reserve_cash(`, `release_cash(`, `cleanup_stale_reservations(` calls in
  `tests/` pass an explicit `tmp_path`-derived `db_file`/`db_path`
  (`tests/test_concurrent_execution.py:160,164,173-174,177,180,199`). No hits
  without an explicit path.
- Single `insert_trade(` call in tests (`tests/test_sync_broker_exits.py:292-296`)
  passes `db_path=db_path` explicitly.
- `grep -rn "TradingClient(" tests/` → zero matches. No test instantiates a real
  broker client.

**No hermeticity violations. The committed/uncommitted test suite is confirmed
hermetic and is ruled out as a contributor**, consistent with the brief's working
assumption.

### 1b.4 — Attribution verdict table

| Timestamp (UTC) | Amount | Type | Order ID | Origin | Evidence |
|---|---|---|---|---|---|
| 2026-07-11T08:24:03Z | -$59,250.00 | OPTRD (paired OPEXC, NVDA260710C00197500 ×3) | — (none) | **BROKER** | Alpaca non-trade activity, no order_id; `ai-options-bot.service` confirmed down since 2026-07-02T19:53 CDT |
| 2026-07-11T08:24:03Z | -$38,250.00 | OPTRD (paired OPEXC, MSFT260710C00382500 ×1) | — (none) | **BROKER** | same; other MSFT lot's GTC stop filled independently at broker 2026-07-08 (broker-resident order, daemon-independent) |
| 2026-07-11T08:24:03Z | -$20,000.00 | OPTRD (paired OPEXC, NVDA260710C00200000 ×1) | — (none) | **BROKER** | same; sole take-profit exit attempt (2026-07-01) expired unfilled at market close before the outage even began |
| 2026-07-14T01:00:17Z | -$60,000.00 | OPTRD (paired OPEXC, AAPL260713C00300000 ×2) | — (none) | **BROKER** | same; daemon resumed 2026-07-13T07:44 CDT, same-day exit attempt expired unfilled at market close |
| 2026-07-01T14:05:48Z (entry, not crossing) | -$256.00 | FILL buy 1 NVDA260710C00200000 | `b4fd5ccf` | AGENT | matched `logs/options_trade_log.jsonl` |
| 2026-07-01/02 (entry, not crossing) | -$2,151.00 total | FILL buy 5 NVDA260710C00197500 + MSFT260710C00382500 lots | `139d9530`/`7f23d4cb`/`8b2dd39b`/`59323df4`/`6165fb5a` | AGENT | matched `logs/options_trade_log.jsonl` |
| 2026-07-02T13:37/15:06Z (entry, not crossing) | -$1,405.00 total | FILL buy 2 AAPL260713C00300000 | `e4facc5e`/`f29b4a43` | AGENT | matched `logs/options_trade_log.jsonl` |

No ORPHAN rows exist in the crossing set — the entries were AGENT, the crossing
debits themselves are 100% BROKER.

---

## Working notes / things checked and eliminated

- **GTC protective-stop rejections**: recurring `"potential wash trade detected"`
  rejections from Alpaca when submitting a GTC stop immediately after a BUY fill
  while another same-symbol order is still technically open — confirmed via exact
  reject payloads for NVDA200C (2026-07-01T14:05:48Z) and all 3 NVDA197.5C lots
  (2026-07-02). `RECONCILE: N unprotected options position(s) — broker-side stop
  missing` warnings appear in nearly every options-bot session log in the weeks
  surrounding the incident (spot-checked 2026-06-25/26). This left 3 of the 4
  later-exercised contracts with **no broker-side protection at all**, meaning
  even a hypothetical daemon-independent safety net (a resting GTC stop) wasn't
  present for most of the exposure.
- **Premature "trade_executed" logging**: `manage_existing_positions()`
  (`options_agent.py:1223-1258`) writes `options_trade_log.jsonl` + fires
  `alert_trade_executed(...)` with realized P&L immediately after
  `submit_order`/`_fill_timeout_retry` return, without checking the resulting
  order's terminal status. Directly observed for NVDA200C: broker order history
  shows the take-profit exit order was canceled (5-min timeout) then resubmitted
  and left to **expire unfilled** at market close (2026-07-01T20:00:01Z) — yet the
  app logged `"✅ Closed option position: take_profit_50% (+71.9%)"` and a
  `trade_executed...PROFIT +$184.00` alert at `19:11:51Z`, ~9 minutes before the
  resubmitted order's fate was even known, and never corrected the record. This
  predates the service outage and is an independent contributing defect.
- **GTC DTE-exit has no retry/re-pricing** (`_fill_timeout_retry` explicitly
  excludes `TimeInForce.GTC`), and `_has_open_exit_order()` causes a position with
  a resting (stale) GTC exit to be skipped on every subsequent session
  indefinitely. Not proven to be the proximate cause here (the outage alone fully
  explains the lack of monitoring), but confirmed as a real, independent
  structural gap.
- **Reservation release-at-submit** (not fill-terminal) and **no slippage
  buffer on reservation sizing** are both confirmed real, pre-existing gaps
  (background code-path research, see repo history) — neither is implicated in
  this crossing, since none of the crossing debits ever went through
  `reserve_cash`/`submit_order` at all.
- **No code anywhere in the repo watches Alpaca non-trade activities**
  (OPEXC/OPASN/OPEXP) — `grep -rn "OPEXC\|OPASN\|OPEXP" scripts/` returns zero
  hits outside this investigation's own scratch scripts. There is no automated
  detection or alerting for auto-exercise, with or without the daemon running.
- **Startup preflight check** (`options_daemon.py`, logged `"preflight: OK —
  equity=... buying_power=... options_level=... pdt=..."` at every daemon start
  including 2026-07-13T07:44:04Z) never inspects `account.cash`. The daemon came
  back online already deeply cash-negative and logged "OK."
