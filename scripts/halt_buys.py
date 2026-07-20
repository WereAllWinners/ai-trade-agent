#!/usr/bin/env python3
"""
halt_buys.py — R1 WI-7: a flag-file circuit breaker for the account cash
invariant (cash >= 0 and non_marginable_buying_power >= 0). Written by
position_watchdog.py on breach; checked by both agents at the top of their
BUY path. Clearing is manual only — no code path removes the flag.
"""
import json
import logging
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_HALT_FLAG_PATH = _PROJECT_ROOT / 'logs' / 'halt_buys.flag'


def flag_path(paper: bool, base: Path = _DEFAULT_HALT_FLAG_PATH) -> Path:
    """logs/halt_buys.flag (paper) / logs/halt_buys_live.flag (live).

    Deliberately takes an explicit bool, NOT service_suffix()/suffixed_path()
    — those read PAPER_TRADING from the CALLING PROCESS's own ambient env,
    correct for a single-account-per-process agent but wrong for the
    dual-account watchdog (which checks both paper and live in one process).
    Both agents already carry self._paper as a real attribute, so they call
    flag_path(self._paper); the watchdog calls it twice, explicitly, once
    per account.
    """
    if paper:
        return base
    return base.with_name(base.stem + '_live' + base.suffix)


def is_halt_active(path: Path) -> bool:
    """FAIL CLOSED: any exception while determining state -> True (halt).

    The inverse of this codebase's usual fail-open convention (e.g.
    alerts.alert_once_per_day's bare except that lets the alert through
    regardless of dedup-state failure) — here, "we couldn't tell" must
    block buys, not allow them. Only checks path.exists(), never parses
    content, never checks age/TTL -> a flag from days ago still blocks
    (deliberate: staleness is not a reason to auto-clear) -> clearing is
    only ever a manual file deletion (see docs/operations.md).
    """
    try:
        return path.exists()
    except Exception:
        return True


def write_halt_flag(path: Path, reason: str, data: dict | None = None) -> None:
    """Called only by the watchdog, only on breach. JSON payload is purely
    informational — is_halt_active() never reads it back. Does not swallow
    its own exceptions: the watchdog wraps this call so a failed flag WRITE
    doesn't suppress the CRITICAL alert — the two are independent safety
    signals, not one gated on the other."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'reason': reason, 'written_at': __import__('datetime').datetime.now().isoformat()}
    if data:
        payload['data'] = data
    path.write_text(json.dumps(payload, indent=2))
    logging.critical(f"🛑 HALT_BUYS flag written: {path} — {reason}")
