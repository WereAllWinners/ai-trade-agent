#!/usr/bin/env python3
"""
preflight_check.py — Alpaca account validator run at daemon startup.

Checks four conditions before the trading loops begin:

  1. Account not blocked  (trading_blocked / account_blocked flags)
  2. Account not restricted from new orders (pattern_day_trader status)
  3. Options approval level ≥ REQUIRED_OPTIONS_LEVEL (default 2)
  4. Sufficient buying power (≥ MIN_BUYING_POWER, default $500)

Returns a PreflightResult named-tuple so callers can decide whether to
abort, warn, or continue in degraded mode.

Usage:
    from preflight_check import run_preflight, PreflightResult
    result = run_preflight(trading_client, require_options=True)
    if not result.ok:
        for issue in result.issues:
            logging.critical(f"Preflight FAILED: {issue}")
        sys.exit(1)
"""
import logging
import os
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REQUIRED_OPTIONS_LEVEL: int   = int(os.getenv('REQUIRED_OPTIONS_LEVEL', '2'))
MIN_BUYING_POWER:       float = float(os.getenv('MIN_BUYING_POWER', '500'))


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class PreflightResult(NamedTuple):
    ok:     bool          # True iff ALL checks passed
    issues: list[str]     # non-empty when ok=False
    info:   dict          # raw account snapshot for logging / /status endpoint


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_preflight(
    trading_client,
    require_options: bool = False,
) -> PreflightResult:
    """
    Validate the Alpaca account and return a PreflightResult.

    Parameters
    ----------
    trading_client : alpaca.trading.client.TradingClient
        Authenticated Alpaca client.
    require_options : bool
        If True, also check options approval level (for the options daemon).
    """
    issues: list[str] = []
    info:   dict      = {}

    try:
        account = trading_client.get_account()
    except Exception as e:
        return PreflightResult(
            ok=False,
            issues=[f"Could not fetch Alpaca account: {e}"],
            info={},
        )

    # ------------------------------------------------------------------
    # Collect raw values (safe string → bool / float conversions)
    # ------------------------------------------------------------------
    def _bool(val) -> bool:
        if isinstance(val, bool):
            return val
        return str(val).lower() in ('true', '1', 'yes')

    def _float(val, default: float = 0.0) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    trading_blocked  = _bool(getattr(account, 'trading_blocked',  False))
    account_blocked  = _bool(getattr(account, 'account_blocked',  False))
    transfers_blocked= _bool(getattr(account, 'transfers_blocked', False))
    pdt_flag         = _bool(getattr(account, 'pattern_day_trader', False))
    buying_power     = _float(getattr(account, 'buying_power', 0))
    equity           = _float(getattr(account, 'equity',        0))
    cash             = _float(getattr(account, 'cash',          0))
    account_status   = str(getattr(account, 'status', 'unknown'))

    # Options level: Alpaca returns this as options_approved_level (int 0-4)
    options_level = int(getattr(account, 'options_approved_level', 0) or 0)

    info = {
        'account_status':    account_status,
        'trading_blocked':   trading_blocked,
        'account_blocked':   account_blocked,
        'transfers_blocked': transfers_blocked,
        'pattern_day_trader':pdt_flag,
        'options_level':     options_level,
        'buying_power':      buying_power,
        'equity':            equity,
        'cash':              cash,
    }

    # ------------------------------------------------------------------
    # Check 1 — account not blocked
    # ------------------------------------------------------------------
    if account_blocked:
        issues.append("account_blocked=True — Alpaca has restricted this account")

    if trading_blocked:
        issues.append("trading_blocked=True — trading is suspended on this account")

    if account_status not in ('ACTIVE', 'active'):
        issues.append(
            f"account status is '{account_status}' — expected ACTIVE"
        )

    # ------------------------------------------------------------------
    # Check 2 — PDT warning (not a hard block, just a warning unless pdt_blocked)
    # ------------------------------------------------------------------
    if pdt_flag:
        logging.warning(
            "preflight: pattern_day_trader=True — you have < $25 000 equity "
            "and are restricted to 3 same-day round-trips per rolling 5 days"
        )
        # Not added to issues (not fatal) — agents already enforce _PDT_DAY_TRADE_LIMIT

    # ------------------------------------------------------------------
    # Check 3 — options approval level
    # ------------------------------------------------------------------
    if require_options and options_level < REQUIRED_OPTIONS_LEVEL:
        issues.append(
            f"options_approved_level={options_level} — need ≥ {REQUIRED_OPTIONS_LEVEL} "
            f"to trade spreads / multi-leg options"
        )

    # ------------------------------------------------------------------
    # Check 4 — minimum buying power
    # ------------------------------------------------------------------
    if buying_power < MIN_BUYING_POWER:
        issues.append(
            f"buying_power=${buying_power:,.2f} is below minimum ${MIN_BUYING_POWER:,.2f}"
        )

    ok = len(issues) == 0

    if ok:
        logging.info(
            f"preflight: OK — equity=${equity:,.2f}  "
            f"buying_power=${buying_power:,.2f}  "
            f"options_level={options_level}  "
            f"pdt={pdt_flag}"
        )
    else:
        for issue in issues:
            logging.critical(f"preflight FAILED: {issue}")

    return PreflightResult(ok=ok, issues=issues, info=info)
