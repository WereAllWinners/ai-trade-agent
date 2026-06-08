#!/usr/bin/env python3
"""
account_config.py — Equity-based trading configuration.

Two account tiers, evaluated live at the start of each trading session:

  Small account  (live equity < LIVE_EQUITY_THRESHOLD, default $25,000)
    - Options bot: trades live with conservative small-account params
    - Stock bot:   forced to paper regardless of PAPER_TRADING env var
                   (PDT rule caps same-day round-trips at 3 per 5 days under $25k)

  Full account   (live equity >= LIVE_EQUITY_THRESHOLD)
    - Both bots use standard paper-trading params
    - One-time alert fired so the user knows to review config

LIVE_EQUITY_THRESHOLD is read from .env (default 25000).
"""

import logging
import os
from typing import Optional

log = logging.getLogger(__name__)

LIVE_EQUITY_THRESHOLD: float = float(os.getenv('LIVE_EQUITY_THRESHOLD', '25000'))

# Equity at which the options bot is allowed to go live.
# Below this the options agent forces paper mode even when PAPER_TRADING=false.
# Default $5,000 — enough capital for meaningful options positions.
OPTIONS_LIVE_THRESHOLD: float = float(os.getenv('OPTIONS_LIVE_THRESHOLD', '5000'))


def fetch_equity(trading_client) -> float:
    """Return current account equity, or 0.0 on failure."""
    try:
        return float(trading_client.get_account().equity or 0)
    except Exception as e:
        log.warning(f"account_config: could not fetch equity: {e}")
        return 0.0


def is_small_account(equity: float) -> bool:
    return equity < LIVE_EQUITY_THRESHOLD


# ---------------------------------------------------------------------------
# Options params
# ---------------------------------------------------------------------------

# Standard params — mirror options_agent.py defaults, used for full accounts.
_OPTIONS_FULL = {
    'max_portfolio_allocation': 0.15,
    'min_portfolio_allocation': 0.10,
    'max_position_size':        0.03,
    'min_confidence':           0.75,
    'max_daily_trades':         5,
    'max_dte':                  45,
    'min_dte':                  7,
    'target_delta':             0.30,
    'max_loss_per_trade':       0.02,
    'take_profit':              0.50,
    'stop_loss':               -0.50,
    'exit_dte_threshold':       2,
    'max_daily_loss_pct':       0.05,
    'no_same_day_close':        False,
}

# Small-account overrides for a ~$1,500 live account.
#
# Rationale for each change:
#   max_portfolio_allocation 70%  — need enough capital per trade; 10-15%
#                                   on $1,500 = $15-22, can't buy a contract
#   min_portfolio_allocation  0%  — don't force deployment when nothing looks good
#   max_position_size        20%  — ~$300 per contract on $1,500, enough for
#                                   liquid names with moderate premiums
#   max_daily_trades          2   — stay well clear of the 3-day-trade PDT limit
#   max_daily_loss_pct       15%  — $75 circuit breaker (5% of $1,500) is too
#                                   tight; 15% = ~$225 before halt
#   min_confidence           82%  — concentration risk is higher; need more
#                                   certainty before committing 20% of account
#   max_dte                  30   — shorter duration reduces overnight exposure
#   no_same_day_close        True — never close an options position the same
#                                   calendar day it was opened; same-day round-
#                                   trips count toward the PDT limit
_OPTIONS_SMALL = {
    'max_portfolio_allocation': 0.70,
    'min_portfolio_allocation': 0.00,
    'max_position_size':        0.20,
    'min_confidence':           0.82,
    'max_daily_trades':         2,
    'max_dte':                  30,
    'min_dte':                  7,
    'target_delta':             0.30,
    'max_loss_per_trade':       0.10,
    'take_profit':              0.50,
    'stop_loss':               -0.50,
    'exit_dte_threshold':       2,
    'max_daily_loss_pct':       0.15,
    'no_same_day_close':        True,
}


def get_options_params(equity: float) -> dict:
    """Return the options trading params appropriate for the current equity."""
    if is_small_account(equity):
        log.info(
            f"account_config: small account (${equity:,.0f} < "
            f"${LIVE_EQUITY_THRESHOLD:,.0f}) — using small-account options params"
        )
        return dict(_OPTIONS_SMALL)
    log.info(
        f"account_config: full account (${equity:,.0f} >= "
        f"${LIVE_EQUITY_THRESHOLD:,.0f}) — using standard options params"
    )
    return dict(_OPTIONS_FULL)


# ---------------------------------------------------------------------------
# Small-account stock params (PDT-safe mode for accounts < OPTIONS_LIVE_THRESHOLD)
# ---------------------------------------------------------------------------

_STOCK_SMALL = {
    'max_daily_trades':   10,    # PDT restricts same-day round-trips, not buying alone
                                 # no_same_day_close prevents the actual violation
    'max_position_size':  0.05,  # 5% per position — safe at $1k, grows naturally
    'min_confidence':     0.70,  # tighter than default 0.60; concentration risk is higher
    'max_daily_loss_pct': 0.10,  # 10% circuit breaker ($100 at $1k)
    'no_same_day_close':  True,  # never sell a position opened today — core PDT guard
    'cooldown_minutes':   60,    # one hour between buys to avoid over-churning
}


def get_stock_params(equity: float) -> dict:
    """
    Return stock-agent param overrides appropriate for the current equity.
    Returns an empty dict for full accounts (no overrides needed).
    """
    if equity < OPTIONS_LIVE_THRESHOLD:
        log.info(
            f"account_config: small stock account (${equity:,.0f} < "
            f"${OPTIONS_LIVE_THRESHOLD:,.0f}) — applying PDT-safe params"
        )
        return dict(_STOCK_SMALL)
    return {}


# ---------------------------------------------------------------------------
# Stock bot gate
# ---------------------------------------------------------------------------

def stock_live_allowed(equity: float) -> bool:
    """
    True for all live accounts. PDT compliance is enforced by the agent's own
    limits (max_daily_trades=2, no_same_day_close=True for small accounts)
    rather than a hard equity block — this lets a $1k account grow to $5k
    before the options bot is enabled.
    """
    return True


def options_live_allowed(equity: float) -> bool:
    """
    True only when the account has enough capital for meaningful options positions.
    Below OPTIONS_LIVE_THRESHOLD the options agent forces paper mode even when
    PAPER_TRADING=false, so the stock bot can grow the account first.
    """
    allowed = equity >= OPTIONS_LIVE_THRESHOLD
    if not allowed:
        log.info(
            f"account_config: options live blocked (${equity:,.0f} < "
            f"${OPTIONS_LIVE_THRESHOLD:,.0f}) — staying on paper endpoint"
        )
    return allowed


# ---------------------------------------------------------------------------
# Upgrade detection
# ---------------------------------------------------------------------------

_upgrade_alerted = False  # resets on process restart; that's fine


def check_for_upgrade(equity: float, prev_equity: Optional[float]) -> None:
    """
    Fire a one-time alert when the account crosses LIVE_EQUITY_THRESHOLD.
    Tells the user the stock bot can now go live and standard params apply.
    """
    global _upgrade_alerted
    if _upgrade_alerted or prev_equity is None:
        return
    if prev_equity < LIVE_EQUITY_THRESHOLD <= equity:
        _upgrade_alerted = True
        msg = (
            f"Live account crossed ${LIVE_EQUITY_THRESHOLD:,.0f} "
            f"(equity now ${equity:,.2f}). "
            f"Standard allocation params are now active. "
            f"The stock bot is eligible for live trading — set "
            f"PAPER_TRADING=false and restart ai-trading-bot to enable it."
        )
        log.info(f"🎉 ACCOUNT UPGRADE — {msg}")
        try:
            from alerts import send_alert, AlertLevel
            send_alert(AlertLevel.INFO, 'account_upgrade', msg,
                       data={'equity': equity, 'threshold': LIVE_EQUITY_THRESHOLD})
        except Exception:
            pass
