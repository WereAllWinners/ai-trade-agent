#!/usr/bin/env python3
"""
trading_service_registry.py — shared unit -> (bot, paper/live) mapping and
instrument-class filter, used by position_watchdog.py and pre_stop_check.py
so neither duplicates this table or the options-symbol-length idiom already
used identically in options_agent.py, risk_reconciler.py, and
liquidate_to_cash.py.
"""
import subprocess

TRADING_UNITS = [
    {'unit': 'ai-trading-bot.service',      'bot': 'stock',   'paper': True},
    {'unit': 'ai-trading-bot-live.service',  'bot': 'stock',   'paper': False},
    {'unit': 'ai-options-bot.service',       'bot': 'options', 'paper': True},
    {'unit': 'ai-options-bot-live.service',  'bot': 'options', 'paper': False},
]


def bot_positions(positions: list, bot: str) -> list:
    """Filter positions to one bot's instrument class.

    Options symbols (OCC format, e.g. SPY250328P00560000) are always longer
    than 10 characters; stock tickers never are. Same idiom used throughout
    the codebase (options_agent.py, risk_reconciler.py, etc.).
    """
    if bot == 'options':
        return [p for p in positions if len(p.symbol) > 10]
    return [p for p in positions if len(p.symbol) <= 10]


def service_state(unit: str) -> str:
    """Return 'active' | 'inactive' | 'not_found' for a systemd unit.

    Deliberately three states, not a bool: a genuinely uninstalled unit
    (e.g. ai-options-bot-live.service before live options is ever deployed)
    must not be treated the same as an installed-but-stopped one. Both
    report stdout 'inactive', but systemctl's exit code distinguishes them:
    3 = unit exists, not active; 4 = unit file not found at all. Confirmed
    empirically on this host: an installed-but-disabled unit returns exit
    code 3; a nonexistent unit name returns exit code 4.

    No sudo — `is-active` is a read-only D-Bus query. This function is
    called from unattended timer-triggered scripts with no TTY; a query
    that accidentally required sudo would hang forever on a password
    prompt with no visible error, which is exactly the kind of silent
    monitoring failure this whole brief exists to prevent.
    """
    try:
        result = subprocess.run(
            ['systemctl', 'is-active', unit],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return 'not_found'

    stdout = result.stdout.strip()
    if stdout == 'active':
        return 'active'
    if result.returncode == 4:
        return 'not_found'
    return 'inactive'
