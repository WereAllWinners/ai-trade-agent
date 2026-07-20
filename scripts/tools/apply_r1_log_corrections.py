#!/usr/bin/env python3
"""
apply_r1_log_corrections.py — R1 WI-5/WI-8: shared, idempotent append tool
for the incident's one-time retroactive log corrections. Every correction
this script knows about is structurally the same operation: preview exact
content, get explicit human sign-off, append (never modify or delete an
original line). Safe to re-run — each item checks whether its order_id (or,
for the addendum errata, a marker string) is already present before writing.

Correction items:
  msft_backfill   (WI-5) — the real MSFT260710C00382500 GTC-stop exit that
                   filled at the broker 2026-07-08 with zero app record.
  nvda_correction (WI-8) — corrects the false "closed at +71.9%" record for
                   NVDA260710C00200000 (order actually expired unfilled).
  aapl_correction (WI-8) — corrects the false "closed at +179.7%" record for
                   AAPL260713C00300000 (order actually expired unfilled).
  addendum_errata (WI-8) — appends a correction note to the addendum
                   findings file for its "profitable"/"real win"
                   mischaracterization of the MSFT fill (actually a loss).

Usage:
    python scripts/tools/apply_r1_log_corrections.py --preview [--item ITEM]
    python scripts/tools/apply_r1_log_corrections.py --execute --item ITEM
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_OPTIONS_TRADE_LOG = _PROJECT_ROOT / 'logs' / 'options_trade_log.jsonl'
_ALERTS_LOG = _PROJECT_ROOT / 'logs' / 'alerts.jsonl'
_ADDENDUM_FILE = _PROJECT_ROOT / 'findings' / 'paper-negcash-addendum-a-2026-07-16.md'

_CITATION = (
    "See findings/paper-negcash-rca-report-2026-07-16.md and "
    "findings/paper-negcash-addendum-a-2026-07-16.md for full context."
)


def _order_id_in_jsonl(path: Path, order_id: str) -> bool:
    if not path.exists():
        return False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get('order_id') == order_id:
                return True
    return False


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(record) + '\n')


# ---------------------------------------------------------------------------
# msft_backfill (WI-5)
# ---------------------------------------------------------------------------

_MSFT_ORDER_ID = 'e83926ab-52fc-4ab3-b5cc-3956b34847e4'


def _msft_backfill_records() -> tuple:
    trade_log_rec = {
        "timestamp": "2026-07-08T13:30:33",
        "contract": "MSFT260710C00382500",
        "action": "sell",
        "quantity": 1,
        "reason": "gtc_protective_stop",
        "order_id": _MSFT_ORDER_ID,
        "source": "paper",
        "source_tag": "broker_reconcile",
        "exit_pl_pct": -0.3827,
    }
    alert_rec = {
        "timestamp": "2026-07-08T13:30:33",
        "level": "AlertLevel.CRITICAL",
        "event": "broker_fill_unaccounted",
        "message": (
            "RETROACTIVE (R1 WI-5): Broker fill with NO app record found: "
            "SELL 1 MSFT260710C00382500 @ $6.05 (order e83926ab-52fc-4ab3-b5cc-3956b34847e4). "
            "GTC protective stop filled at the broker 2026-07-08T13:30:33Z; entry was "
            "2026-07-01T15:43:38Z @ $9.80 -- a -38.3% loss, not the 'profitable'/'real win' "
            "the addendum originally characterized it as (see addendum_errata). "
            f"{_CITATION}"
        ),
        "host": "livermore",
        "source": "paper",
        "data": {
            "symbol": "MSFT260710C00382500", "order_id": _MSFT_ORDER_ID,
            "filled_qty": 1, "avg_price": 6.05, "entry_price": 9.80,
            "realized_pnl_pct": -0.3827,
        },
    }
    return trade_log_rec, alert_rec


def _msft_backfill_already_applied() -> bool:
    return _order_id_in_jsonl(_OPTIONS_TRADE_LOG, _MSFT_ORDER_ID)


def _msft_backfill_preview() -> None:
    trade_log_rec, alert_rec = _msft_backfill_records()
    print(f"Already applied: {_msft_backfill_already_applied()}")
    print(f"\n--- would append to {_OPTIONS_TRADE_LOG} ---")
    print(json.dumps(trade_log_rec, indent=2))
    print(f"\n--- would append to {_ALERTS_LOG} ---")
    print(json.dumps(alert_rec, indent=2))


def _msft_backfill_apply() -> None:
    if _msft_backfill_already_applied():
        print(f"Already applied (order_id {_MSFT_ORDER_ID} found) — skipping, no duplicate written.")
        return
    trade_log_rec, alert_rec = _msft_backfill_records()
    _append_jsonl(_OPTIONS_TRADE_LOG, trade_log_rec)
    _append_jsonl(_ALERTS_LOG, alert_rec)
    print(f"Appended MSFT backfill to {_OPTIONS_TRADE_LOG} and {_ALERTS_LOG}.")


# ---------------------------------------------------------------------------
# nvda_correction / aapl_correction (WI-8)
# ---------------------------------------------------------------------------

_NVDA_ORDER_ID = '4d903428-5cca-4fdd-8f63-c3c275ebd06f'
_AAPL_ORDER_ID = 'd0a4c159-11a6-4490-96ca-b4ff26739dc0'


def _correction_records(order_id: str, contract: str, claimed: str,
                        original_ts: str, expired_ts: str, exercised_note: str) -> tuple:
    trade_log_rec = {
        "timestamp": datetime.now().isoformat(),
        "type": "correction",
        "contract": contract,
        "order_id": order_id,
        "corrects_claim": claimed,
        "corrects_entry_timestamp": original_ts,
        "broker_truth": (
            f"Order {order_id} status=EXPIRED filled_qty=0 at {expired_ts}. "
            f"{exercised_note}"
        ),
        "reason": "phantom_close_defect_r1_wi4",
        "note": f"No 'action' key by design -- nothing filled, nothing true to claim. {_CITATION}",
    }
    alert_rec = {
        "timestamp": datetime.now().isoformat(),
        "level": "AlertLevel.INFO",
        "event": "correction",
        "message": (
            f"CORRECTION (R1 WI-8): the {original_ts} entry claiming '{claimed}' for "
            f"{contract} (order {order_id}) was erroneous. The order actually expired "
            f"unfilled at {expired_ts}. {exercised_note} {_CITATION}"
        ),
        "host": "livermore",
        "source": "paper",
        "data": {"contract": contract, "order_id": order_id, "corrects_claim": claimed},
    }
    return trade_log_rec, alert_rec


def _nvda_correction_records() -> tuple:
    return _correction_records(
        _NVDA_ORDER_ID, 'NVDA260710C00200000', 'take_profit_50% +71.9%',
        '2026-07-01T19:11:51Z', '2026-07-01T20:00:01Z',
        'Position was exercised 2026-07-10.',
    )


def _aapl_correction_records() -> tuple:
    return _correction_records(
        _AAPL_ORDER_ID, 'AAPL260713C00300000', 'take_profit_50% +179.7%',
        '2026-07-13T09:35:13Z', '2026-07-13T20:00:05Z',
        'Position was exercised, posted 2026-07-14.',
    )


def _correction_already_applied(order_id: str) -> bool:
    if not _OPTIONS_TRADE_LOG.exists():
        return False
    with open(_OPTIONS_TRADE_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get('type') == 'correction' and rec.get('order_id') == order_id:
                return True
    return False


def _make_correction_item(name: str, order_id: str, records_fn):
    def preview():
        trade_log_rec, alert_rec = records_fn()
        print(f"Already applied: {_correction_already_applied(order_id)}")
        print(f"\n--- would append to {_OPTIONS_TRADE_LOG} ---")
        print(json.dumps(trade_log_rec, indent=2))
        print(f"\n--- would append to {_ALERTS_LOG} ---")
        print(json.dumps(alert_rec, indent=2))

    def apply():
        if _correction_already_applied(order_id):
            print(f"Already applied (correction for {order_id} found) — skipping.")
            return
        trade_log_rec, alert_rec = records_fn()
        _append_jsonl(_OPTIONS_TRADE_LOG, trade_log_rec)
        _append_jsonl(_ALERTS_LOG, alert_rec)
        print(f"Appended {name} correction to {_OPTIONS_TRADE_LOG} and {_ALERTS_LOG}.")

    return preview, apply


# ---------------------------------------------------------------------------
# addendum_errata (WI-8)
# ---------------------------------------------------------------------------

_ERRATA_MARKER = "## Errata (R1 WI-8"
_ERRATA_TEXT = f"""

{_ERRATA_MARKER}, {datetime.now().date().isoformat()})

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
"""


def _errata_already_applied() -> bool:
    if not _ADDENDUM_FILE.exists():
        return False
    return _ERRATA_MARKER in _ADDENDUM_FILE.read_text()


def _errata_preview() -> None:
    print(f"Already applied: {_errata_already_applied()}")
    print(f"\n--- would append to {_ADDENDUM_FILE} ---")
    print(_ERRATA_TEXT)


def _errata_apply() -> None:
    if _errata_already_applied():
        print("Already applied (errata marker found) — skipping.")
        return
    if not _ADDENDUM_FILE.exists():
        print(f"ERROR: {_ADDENDUM_FILE} does not exist — cannot append.")
        sys.exit(1)
    with open(_ADDENDUM_FILE, 'a') as f:
        f.write(_ERRATA_TEXT)
    print(f"Appended errata note to {_ADDENDUM_FILE}.")


# ---------------------------------------------------------------------------

_nvda_preview, _nvda_apply = _make_correction_item('NVDA', _NVDA_ORDER_ID, _nvda_correction_records)
_aapl_preview, _aapl_apply = _make_correction_item('AAPL', _AAPL_ORDER_ID, _aapl_correction_records)

_ITEMS = {
    'msft_backfill':   (_msft_backfill_preview, _msft_backfill_apply),
    'nvda_correction': (_nvda_preview, _nvda_apply),
    'aapl_correction': (_aapl_preview, _aapl_apply),
    'addendum_errata': (_errata_preview, _errata_apply),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--item', choices=list(_ITEMS.keys()), default=None,
                         help='Apply/preview only this item (default: all)')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preview', action='store_true')
    mode.add_argument('--execute', action='store_true')
    args = parser.parse_args()

    items = [args.item] if args.item else list(_ITEMS.keys())
    for name in items:
        preview_fn, apply_fn = _ITEMS[name]
        print(f"\n{'='*70}\n{name}\n{'='*70}")
        if args.preview:
            preview_fn()
        else:
            apply_fn()


if __name__ == '__main__':
    main()
