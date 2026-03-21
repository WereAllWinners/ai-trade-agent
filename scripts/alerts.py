#!/usr/bin/env python3
"""
Alerts module - sends notifications for critical trading events.

Supports two channels:
  1. File log (always active) — appends to logs/alerts.jsonl
  2. Email (optional) — configure via environment variables:
       ALERT_EMAIL_FROM    sender address
       ALERT_EMAIL_TO      recipient address (comma-separated for multiple)
       ALERT_SMTP_HOST     SMTP server host      (default: smtp.gmail.com)
       ALERT_SMTP_PORT     SMTP server port      (default: 587)
       ALERT_SMTP_USER     SMTP login user       (defaults to ALERT_EMAIL_FROM)
       ALERT_SMTP_PASSWORD SMTP password / app password

Usage:
  from alerts import send_alert, AlertLevel
  send_alert(AlertLevel.CRITICAL, "circuit_breaker", "Daily loss -5.1%, trading halted.")
  send_alert(AlertLevel.INFO, "trade_executed", "BUY 50 AAPL @ $185.20")
"""
import json
import logging
import os
import smtplib
import socket
from datetime import datetime
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path

_ALERTS_LOG = Path(__file__).resolve().parent.parent / 'logs' / 'alerts.jsonl'
_ALERTS_LOG.parent.mkdir(parents=True, exist_ok=True)

log = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    INFO = 'INFO'
    WARNING = 'WARNING'
    CRITICAL = 'CRITICAL'


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def send_alert(level: AlertLevel, event: str, message: str, data: dict = None) -> None:
    """
    Send an alert via all configured channels.

    Args:
        level:   AlertLevel.INFO / WARNING / CRITICAL
        event:   Short machine-readable event name, e.g. 'circuit_breaker'
        message: Human-readable description
        data:    Optional extra context dict (included in file log and email)
    """
    record = {
        'timestamp': datetime.now().isoformat(),
        'level': str(level),
        'event': event,
        'message': message,
        'host': socket.gethostname(),
    }
    if data:
        record['data'] = data

    _write_to_log(record)
    _log_to_stderr(level, event, message)
    _send_email(level, event, message, record)


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------

def _write_to_log(record: dict) -> None:
    try:
        with open(_ALERTS_LOG, 'a') as f:
            f.write(json.dumps(record) + '\n')
    except Exception as e:
        log.error(f"alerts: failed to write to log file: {e}")


def _log_to_stderr(level: AlertLevel, event: str, message: str) -> None:
    msg = f"[ALERT:{level}] {event} — {message}"
    if level == AlertLevel.CRITICAL:
        log.critical(msg)
    elif level == AlertLevel.WARNING:
        log.warning(msg)
    else:
        log.info(msg)


def _send_email(level: AlertLevel, event: str, message: str, record: dict) -> None:
    """Send email if ALERT_EMAIL_FROM and ALERT_EMAIL_TO are configured."""
    email_from = os.getenv('ALERT_EMAIL_FROM', '').strip()
    email_to_raw = os.getenv('ALERT_EMAIL_TO', '').strip()

    if not email_from or not email_to_raw:
        return  # email not configured

    smtp_host = os.getenv('ALERT_SMTP_HOST', 'smtp.gmail.com')
    smtp_port = int(os.getenv('ALERT_SMTP_PORT', '587'))
    smtp_user = os.getenv('ALERT_SMTP_USER', email_from)
    smtp_password = os.getenv('ALERT_SMTP_PASSWORD', '')
    recipients = [r.strip() for r in email_to_raw.split(',') if r.strip()]

    subject = f"[AI Trade Agent] {level} — {event}"
    body = (
        f"Level:   {level}\n"
        f"Event:   {event}\n"
        f"Time:    {record['timestamp']}\n"
        f"Host:    {record['host']}\n\n"
        f"Message: {message}\n"
    )
    if record.get('data'):
        body += f"\nDetails:\n{json.dumps(record['data'], indent=2)}\n"

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = email_from
    msg['To'] = ', '.join(recipients)

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(email_from, recipients, msg.as_string())
        log.info(f"alerts: email sent to {recipients}")
    except Exception as e:
        log.error(f"alerts: email failed ({e}). Check ALERT_SMTP_* env vars.")


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def alert_circuit_breaker(agent_name: str, daily_pnl_pct: float, equity: float) -> None:
    send_alert(
        AlertLevel.CRITICAL,
        'circuit_breaker',
        f"{agent_name}: daily P&L {daily_pnl_pct:.1%} breached limit — trading halted.",
        data={'agent': agent_name, 'daily_pnl_pct': daily_pnl_pct, 'equity': equity},
    )


def alert_trade_executed(agent_name: str, symbol: str, action: str,
                         qty: int, price: float, order_id: str) -> None:
    send_alert(
        AlertLevel.INFO,
        'trade_executed',
        f"{agent_name}: {action.upper()} {qty} {symbol} @ ${price:.2f}",
        data={'agent': agent_name, 'symbol': symbol, 'action': action,
              'qty': qty, 'price': price, 'order_id': order_id},
    )


def alert_trade_failed(agent_name: str, symbol: str, error: str) -> None:
    send_alert(
        AlertLevel.WARNING,
        'trade_failed',
        f"{agent_name}: order for {symbol} failed — {error}",
        data={'agent': agent_name, 'symbol': symbol, 'error': error},
    )


if __name__ == '__main__':
    # Quick smoke test: logs to file, prints to console, skips email if unconfigured
    print("Testing alerts system...")
    send_alert(AlertLevel.INFO, 'test', 'Alerts module loaded successfully.')
    send_alert(AlertLevel.WARNING, 'test', 'This is a warning alert test.')
    send_alert(AlertLevel.CRITICAL, 'test', 'This is a critical alert test.')
    print(f"Check {_ALERTS_LOG} for entries.")
