# Operations Runbook

Operational procedures for running the trading services safely. Started as
part of R1 (paper negative-cash incident remediation) — this file will grow
as later R1 work items land (see the TODO markers below).

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

<!-- TODO (R1 WI-9b, after WI-3/6/7/8 land):
  - HALT_BUYS manual-clear procedure
  - Alert channel reference (Telegram is configured in code but
    deliberately unused — email + logs/alerts.jsonl only)
  - Expiry-sentinel tier reference (WARN/CRITICAL)
  - Wash-trade diagnosis findings pointer
  - Correction-entries reference (the two false profit records + MSFT backfill)
  - Two accepted gaps found during R1 planning, owned by R2:
    1. Stock agent's cash hard-gate can deadlock its own defensive sells
       (autonomous_agent.py::run_trading_session()'s cash gate returns from
       the whole session, so review_held_positions() never runs once cash
       goes negative)
    2. BUY-side entry path has the same fill-confirmation blindness WI-4
       fixes on the exit side (execute_options_trade() logs the entry
       immediately, before confirming the order actually filled)
-->
