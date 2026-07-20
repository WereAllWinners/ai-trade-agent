# Negative Cash RCA — Addendum A: Training Data Contamination Check (2026-07-16)

Follow-up to `paper-negcash-rca-report-2026-07-16.md`. Read-only DB/log queries
only — no writes, no pipeline changes, no service/env changes. All SQL below is
SELECT-only against `logs/trading.db`.

**Anomaly window used:** 2026-07-02T19:53:10-05:00 (options service stop) through
now (2026-07-16), cash negative continuously from 2026-07-11T08:24:03Z onward —
per the quarantine boundary in the RCA report.

---

## Verdict, Check 1: false outcome rows

**No `outcomes`, `trades`, or `training_examples` row exists for any of the four
exercised contracts, or for any of their eight BUY order_ids, in `logs/trading.db`.**
This holds independent of the JSONL-log question below — verified three ways:

1. Direct query by exact OCC symbol against all three tables: zero rows.
2. Direct query by all eight known BUY `order_id`s against all three tables
   (`outcomes.buy_order_id`, `trades.order_id`): zero rows.
3. **`outcomes` (bot='options') has had no new row of any kind since
   `exit_timestamp` 2026-06-03T14:27:12** — three and a half weeks before the
   shutdown. `outcomes` (bot='stock') froze similarly at `exit_timestamp`
   2026-06-25T13:14:42. Neither the four exercised contracts nor any other trade
   from inside the anomaly window could have reached this table even in
   principle — the outcome-matching pipeline had already stopped advancing for
   both bots well before 2026-07-02.
4. **`trades` (bot='options') has zero rows, period — not just for these four
   contracts.** `SELECT bot, COUNT(*) FROM trades GROUP BY bot` returns only
   `stock | 1491`; there is no `options` row at all in this table's history.
   `_db.insert_trade()` for the options bot's BUY path appears to be either not
   actually executing or silently no-op-ing — no error is logged at the moment
   of any of the eight BUY fills (spot-checked NVDA260710C00200000's buy,
   2026-07-01T09:05:48 CDT: full session log shows the buy, the GTC-stop
   rejection, the trade_executed alert, and the next candidate scan — no
   exception, no DB error). This is a pre-existing options-pipeline gap, not
   specific to the incident, and is flagged as an open question below rather
   than diagnosed further (out of this brief's scope).
5. **`decisions` (bot='options') does contain all eight BUY entries**, correctly
   tagged `executed=1` with matching `order_id` (ids 18989, 19044, 19124, 19165,
   19186, 19188, 19237, 19362) — consistent with the original RCA's AGENT
   attribution. `decisions` is a separate table from `trades`/`outcomes` and is
   not what feeds `training_examples`.

### Per-contract cross-check against broker truth

| Contract | DB row? | JSONL exit record? | Broker truth | Classification |
|---|---|---|---|---|
| NVDA260710C00200000 | None (any table) | **Yes — FALSE.** `options_trade_log.jsonl` + `alerts.jsonl`, 2026-07-01T19:11:51Z (UTC): `"take_profit_50%"`, exit_pl_pct +0.719, alert `"PROFIT +$184.00 (+71.9%)"`, order_id `4d903428-5cca-4fdd-8f63-c3c275ebd06f` | Order `4d903428` status=**EXPIRED**, `expired_at=2026-07-01T20:00:01Z`, `filled_qty=0`. Position rode to expiry, exercised 2026-07-10. | **FALSE record confirmed** (documented in original RCA §3; DB unaffected) |
| NVDA260710C00197500 (×3) | None (any table) | **None found.** No exit entry of any kind in `options_trade_log.jsonl`/`alerts.jsonl` for this symbol at any timestamp. | No SELL order of any kind was ever submitted for this symbol (confirmed against full Alpaca order history — 3 BUY fills only). Exercised 2026-07-10. | **TRUE** (silence, not fabrication — consistent with broker truth that nothing was ever attempted) |
| MSFT260710C00382500 | None (any table) | **None found** for either lot. | One lot's GTC stop (`e83926ab-52fc-4ab3-b5cc-3956b34847e4`) **actually filled at the broker** 2026-07-08T13:30:33Z @ $6.05 (a real, profitable, broker-independent exit) — the app has **zero record of this real win**, in JSONL or DB. The other lot was exercised 2026-07-10. | **AMBIGUOUS / MISSING-TRUE** — opposite failure mode from the NVDA case: a real gain went unrecorded rather than a fake one being recorded. Root cause: `options_agent.py`'s protective-stop submission logs only `logging.info` at submission time and never writes a trade_log/alert entry when the stop later fills independently at the broker (confirmed structurally in the original RCA's background code research). |
| AAPL260713C00300000 (×2) | None (any table) | **Yes — FALSE, newly found in this addendum.** `options_trade_log.jsonl` + `alerts.jsonl`, 2026-07-13T09:35:13 CDT (14:35:13Z): `"take_profit_50%"`, exit_pl_pct +1.797, alert `"PROFIT +$2,525.00 (+179.7%)"`, order_id `d0a4c159-11a6-4490-96ca-b4ff26739dc0` | Order `d0a4c159` (SELL 2 @ limit 19.99, submitted 2026-07-13T14:35:13.235Z — same instant as the log write) status=**EXPIRED**, `expired_at=2026-07-13T20:00:05Z`, `filled_qty=0`. Position exercised, posted 2026-07-14T01:00:17Z. | **FALSE record confirmed.** The brief asked whether the phantom-close defect fired more than once — **it did.** Same mechanism as NVDA200C: `manage_existing_positions()` logs the exit as complete immediately after `submit_order`/`_fill_timeout_retry` return, without checking the resulting order's terminal status. |

**Summary: 2 of 4 contracts have a confirmed FALSE "closed at a profit" record
in the JSONL trade/alert logs (NVDA200C, AAPL300C×2); 1 has a TRUE absence of
any record; 1 has a missing TRUE record (a real win, never logged). None of the
four have any row — true, false, or otherwise — in `outcomes`, `trades`, or
`training_examples`.**

### training_examples — explicit check

`SELECT ... FROM training_examples WHERE symbol IN (the 4 OCC symbols)` → zero
rows. Broader check: `MAX(generated_at)` in `training_examples` is
**2026-07-02T19:06:00.974805 (stock)** and **2026-07-02T01:00:17.935899
(options)** — i.e., **no `training_examples` row of any kind, for either bot,
has been generated since before the shutdown.** The table has not grown at all
during the anomaly window. There is no lineage by which a false outcome could
have reached `training_examples`, because `training_examples` generation itself
did not run during the window.

---

## Verdict, Check 2: pipeline ingestion status during the window

**Mechanism:** `FINETUNE_ENABLED` / `STRATEGY_EVOLVER_ENABLED` in `.env`
(untracked, gitignored — no git history available). No systemd timer or cron
job exists for fine-tuning (`systemctl list-timers`, `crontab -l`: no hits);
scheduling is entirely in-process, inside `trading_daemon.py`/`options_daemon.py`'s
own event loop (8/9PM and 2AM EST daily triggers, checked at
`trading_daemon.py:393` / `options_daemon.py:301`).

**Flag state history (reconstructed from journalctl, `.env` has no git log):**
- **Before 2026-07-07: `FINETUNE_ENABLED=true`.** Confirmed — the scheduled
  options fine-tune actually ran (not skip-logged) at 2026-07-02T01:00:21 CDT
  (`"🎓 RUNNING OPTIONS MODEL FINE-TUNING"`, skipped only by the internal SFT
  data-quality tripwire, not the flag).
- **`.env` mtime: 2026-07-07T15:33:52-05:00.** This is the only physical
  evidence of when the flag changed; content-diff isn't possible (untracked
  file, no prior copy).
- **From 2026-07-07T19:00:00 CDT onward: `FINETUNE_ENABLED=false`**, confirmed
  by explicit skip-log lines (`"⏭️  FINETUNE_ENABLED=false — skipping fine-tune
  cycle"`) recurring through 2026-07-15 (latest checked). Still `false` as of
  this addendum.
- The `.env` edit (15:33:52) landed **~31 minutes before** Jerome restarted
  `ai-trading-bot`/`ai-trading-bot-live` at 16:05:16 that same day — consistent
  with a deliberate precaution taken alongside that restart. `ai-options-bot`
  was not restarted at that point (it stayed down until 2026-07-13).

**A second, ungated fine-tune path exists and DID run inside the window.**
`scripts/training/online_trainer.py`'s `run_online_training()` is called
unconditionally from both daemons (`trading_daemon.py:568`,
`options_daemon.py:480`) — **neither call site checks `FINETUNE_ENABLED`**
(that flag only gates the separate scheduled 8/9PM cycle). This function
independently gates on `_db.get_outcome_count_since(last_trained_at) >= 15`
(shared state file `logs/online_trainer_state.json`, not bot-scoped).

**Runs found inside the window (2026-07-02 19:53 → now):**

| Date | Bot | Result | Adapter output | Merged? |
|---|---|---|---|---|
| 2026-07-02T16:34:36 CDT | options | **Crashed**, exit code 1 (Unsloth import traceback in `fine_tune_llm.py:21`), before shutdown | none | no |
| 2026-07-02T16:02:12 CDT | stock | Succeeded | `finance_qwen_32b_lora_20260702_160449` | no |
| 2026-07-08T16:02:05 CDT | stock | Succeeded | `finance_qwen_32b_lora_20260708_160446` | no |
| 2026-07-09T16:02:05 CDT | stock | Succeeded | `finance_qwen_32b_lora_20260709_160458` | no |
| 2026-07-10T16:02:xx CDT | stock | Succeeded | `finance_qwen_32b_lora_20260710_160435` | **yes** → `finance_qwen_32b_lora_merged_bf16_20260710_160435` |
| 2026-07-13T16:02:05 CDT | stock | Succeeded | `finance_qwen_32b_lora_20260713_160420` | **yes** → `finance_qwen_32b_lora_merged_bf16_20260713_160420` |

Options bot's online-training checked its threshold again post-restart
(2026-07-13T16:32:05, 07-14, 07-15) and correctly reported `"Only 0 new
outcomes since 2026-07-13 (need 15)"` each time — consistent with the frozen
`outcomes` table (Check 1). **Zero successful options-bot fine-tune runs
occurred anywhere inside the window.**

**Ingestion overlap: none.** All five stock runs called `_export_recent_examples(50)`,
which pulls the last 50 rows of `training_examples` ordered by `generated_at`
ASC. Since that table has not grown since 2026-07-02T19:06:00 (Check 1), **all
five runs consumed the exact same 50 rows — `training_examples` ids 5400–5449,
`generated_at` 2026-07-02T01:00:17–19:06:00, `entry_date` range 2026-06-26 to
2026-07-03.** Directly confirmed by inspecting the current
`finetune/data/online_training_batch.json` (overwritten each run, last written
2026-07-13T16:02:05): 50 examples, `entry_date` 2026-06-26 to 2026-07-03,
symbols are all plain stock tickers (`AAPL, ABBV, ABNB, ... NVDA` as underlying,
`SPY`, `TSLA`, etc.) — **none of the four OCC option symbols, and no entry_date
inside the anomaly window, appear anywhere in the batch.** These five runs
retrained repeatedly on an identical, frozen, pre-incident snapshot — wasteful,
but not contaminated.

**Promotion status: none of the five in-window runs were promoted.**
- `logs/eval/promotion_*.json` — the most recent record on disk is
  `promotion_20260615_012503.json` (`promoted: false`, candidate 2026-06-15
  graded "C" vs. current 2026-06-11 graded "B"). **No promotion record exists
  for any July date at all** — none of the five in-window runs were ever
  evaluated by `model_promoter.py`, successfully or otherwise.
- `.env`'s `LORA_ADAPTER_PATH=finetune/finance_qwen_32b_lora_20260611_010305`
  and the `finetune/finance_qwen_32b_lora_latest` symlink both still point to
  the **2026-06-11** checkpoint, unchanged (symlink mtime: Jun 11 01:19).
- **Model actually serving right now** (confirmed via read-only
  `curl localhost:8000/v1/models`, no mutation): root path resolves to
  `finetune/finance_qwen_32b_merged_latest` → `finance_qwen_32b_merged_bf16`, a
  static (non-timestamped) directory with **mtime 2026-05-13** — older even
  than the June 11 adapter `LORA_ADAPTER_PATH` names. This merge-vs-adapter
  pointer lag predates and is unrelated to this incident; noted for accuracy,
  not investigated further (out of scope).

**Conclusion: the pipeline's primary gate (`FINETUNE_ENABLED`) was correctly
enabled until 2026-07-07T15:33:52 CDT and correctly disabled after — but a
second, ungated online-training path ran five times inside the window
regardless. Every one of those runs trained on an identical, frozen,
pre-incident data snapshot with zero overlap with the anomaly window, and none
of their output reached the model actually serving trading decisions today.**

---

## Contamination classification

# **CLEAN**

No false or anomalous row exists in `outcomes`, `trades`, or `training_examples`
for any of the four exercised contracts or any trade inside the anomaly window
(none exist there to begin with — the relevant DB pipelines were already frozen
before the window began). The five online-training runs that did execute inside
the window are confirmed, by direct content inspection, to have trained
exclusively on a fixed pre-window data snapshot, and none were promoted to the
model currently serving. The two FALSE records found (NVDA200C, AAPL300C×2)
exist only in `logs/options_trade_log.jsonl`/`logs/alerts.jsonl` — human-facing
audit trail files with no code path into `training_examples` — so they are a
data-accuracy problem for anyone reading those logs, not a training-contamination
risk.

This is CLEAN by a fortunate combination of circumstances, not by design: the
`training_examples`/`outcomes` pipeline for both bots had already stalled for
unrelated reasons weeks before the incident, and the options bot's `trades`
table write path appears to never populate at all. Had either of those
pipelines been healthy and current, the two false records identified above
would very likely have been picked up. See Open Questions.

---

## Row inventory for later quarantine (nothing to re-derive)

**False JSONL records (not DB rows — for a future brief to decide whether/how
to annotate or correct the human-facing logs):**
- `logs/options_trade_log.jsonl` line matching
  `"order_id": "4d903428-5cca-4fdd-8f63-c3c275ebd06f"` (NVDA200C,
  timestamp `2026-07-01T19:11:51`, claims `take_profit_50%` / +71.9%)
- `logs/alerts.jsonl` line, same order_id, same timestamp, `trade_executed`
  event, `realized_pnl: 184.0`
- `logs/options_trade_log.jsonl` line matching
  `"order_id": "d0a4c159-11a6-4490-96ca-b4ff26739dc0"` (AAPL300C×2,
  timestamp `2026-07-13T09:35:13`, claims `take_profit_50%` / +179.7%)
- `logs/alerts.jsonl` line, same order_id, same timestamp, `trade_executed`
  event, `realized_pnl: 2525.0`

**Missing-true record (a real win never logged anywhere):**
- MSFT260710C00382500, order `e83926ab-52fc-4ab3-b5cc-3956b34847e4`, GTC stop
  filled at broker 2026-07-08T13:30:33Z @ $6.05/contract, entry lot filled
  2026-07-01T15:43:38Z @ $9.80 — real realized gain, zero app record.

**Training data actually consumed by in-window fine-tune runs (confirmed
clean, but exact for audit purposes):**
- `training_examples` rows **id 5400–5449** (50 rows), `generated_at`
  2026-07-02T01:00:17.632520 – 2026-07-02T19:06:00.974805, `entry_date` range
  2026-06-26 – 2026-07-03. Consumed identically by all 5 in-window online-training
  runs.

**In-window adapter artifacts (none promoted, none currently serving):**
- `finetune/finance_qwen_32b_lora_20260702_160449`
- `finetune/finance_qwen_32b_lora_20260708_160446`
- `finetune/finance_qwen_32b_lora_20260709_160458`
- `finetune/finance_qwen_32b_lora_20260710_160435` (+ merged
  `finance_qwen_32b_lora_merged_bf16_20260710_160435`)
- `finetune/finance_qwen_32b_lora_20260713_160420` (+ merged
  `finance_qwen_32b_lora_merged_bf16_20260713_160420`)

**Currently active pointers (unaffected by the window):**
- `.env` `LORA_ADAPTER_PATH` = `finetune/finance_qwen_32b_lora_20260611_010305`
- `finetune/finance_qwen_32b_lora_latest` → same, 2026-06-11
- `finetune/finance_qwen_32b_merged_latest` → `finance_qwen_32b_merged_bf16`
  (static dir, mtime 2026-05-13) — this is what vLLM is actually serving
- Last promotion record: `logs/eval/promotion_20260615_012503.json`
  (`promoted: false`)

---

## Open questions

1. **Why does `trades` have zero rows for the options bot, ever — not just in
   this window?** This looks like a pre-existing, possibly long-standing gap in
   `_db.insert_trade()` for the options BUY path, independent of the negative-cash
   incident. It happens to be part of why this contamination check came back
   clean, which is a fragile reason to be clean. Worth its own investigation.
2. **Why did `outcomes` (options) stop advancing on 2026-06-03 and `outcomes`
   (stock) on 2026-06-25 — three-plus weeks before the shutdown?** Both predate
   and are unrelated to the service outage. If/when this pipeline is repaired,
   whatever fix is applied should not simply resume matching from where it left
   off without checking whether the four exercised contracts (and any other
   trades from the anomaly window) get correctly matched as losses/exercises
   rather than silently skipped or mismatched.
3. **The `online_trainer.py` → `run_online_training()` call sites in both
   daemons are not gated by `FINETUNE_ENABLED`.** This had no contamination
   consequence this time only because the data it drew on happened to already be
   frozen pre-window. A future incident with an actively-updating
   `training_examples` table and `FINETUNE_ENABLED=false` would not get the
   protection the flag name implies. Flagged for the human to decide whether
   this warrants scoping into a remediation brief.
4. **Should the two false JSONL trade_log/alert entries (NVDA200C, AAPL300C×2)
   be corrected or annotated?** They're outside the training pipeline but are
   still the human-facing record of what happened, and currently misrepresent
   two of the four exercised positions as profitable closes.
5. **The currently-serving merged model (`finance_qwen_32b_merged_bf16`, mtime
   2026-05-13) is older than even the `LORA_ADAPTER_PATH` pointer (June 11).**
   Unrelated to this incident but noted for accuracy — not investigated further
   here.

---

*End of Addendum A. Per brief scope: no remediation applied, no rows modified,
no pipeline changes made. This is the sole deliverable; further action is a
separate future brief.*


## Errata (R1 WI-8, 2026-07-20)

**Correction to the MSFT260710C00382500 characterization above.** This
document originally described the MSFT GTC protective-stop fill (order
`e83926ab-52fc-4ab3-b5cc-3956b34847e4`, entry 2026-07-01T15:43:38Z @ $9.80,
exit 2026-07-08T13:30:33Z @ $6.05) as "a real, profitable, broker-independent
exit" / "a real win." That is incorrect: (6.05 - 9.80) / 9.80 = **-38.3%, a
loss**, and structurally this is a stop-*loss* fill (`stop_loss_pct =
-0.50` by construction), not a take-profit. The finding that the app has
zero record of this real, broker-confirmed exit still stands — only the
win/loss characterization was wrong. Found and corrected during R1
implementation planning; see `findings/r1-cash-restoration-2026-07-16.md`
and the R1 plan for detail. The retroactive backfill applied in
`logs/options_trade_log.jsonl`/`logs/alerts.jsonl` (R1 WI-5) uses the
correct -38.3% figure.
