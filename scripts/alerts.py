#!/usr/bin/env python3
"""
Alerts module - sends notifications for critical trading events.

Supports three channels:
  1. File log (always active) — appends to logs/alerts.jsonl
  2. Email (optional) — configure via environment variables:
       ALERT_EMAIL_FROM    sender address
       ALERT_EMAIL_TO      recipient address (comma-separated for multiple)
       ALERT_SMTP_HOST     SMTP server host      (default: smtp.gmail.com)
       ALERT_SMTP_PORT     SMTP server port      (default: 587)
       ALERT_SMTP_USER     SMTP login user       (defaults to ALERT_EMAIL_FROM)
       ALERT_SMTP_PASSWORD SMTP password / app password
  3. Telegram (optional) — configure via environment variables:
       TELEGRAM_BOT_TOKEN       Bot token from @BotFather
       TELEGRAM_CHAT_ID         Chat/channel ID to send messages to
       TELEGRAM_ALERTS_ENABLED  Set to 'false' to disable (default: true)

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

# Agent-level source tag written into every alert record.
# Set once at agent init via set_alert_source() so no individual call site
# needs to remember. Separate daemon processes are isolated, so this is safe.
_ALERT_SOURCE: str = 'unknown'


def set_alert_source(source: str) -> None:
    """Set the source tag for all alerts from this process ('paper' or 'live')."""
    global _ALERT_SOURCE
    _ALERT_SOURCE = source


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
        'source': _ALERT_SOURCE,
    }
    if data:
        record['data'] = data

    full_message = f"[{level}] {event}\n{message}"

    _write_to_log(record)
    _log_to_stderr(level, event, message)
    _send_email(level, event, message, record)
    _send_telegram(full_message)


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

    if event == 'trade_executed' and record.get('data'):
        d = record['data']
        action = d.get('action', '').upper()
        symbol = d.get('symbol', '')
        qty    = d.get('qty', '')
        price  = d.get('fill_price', 0)

        if action == 'BUY':
            return  # BUYs are batched into the daily close summary email

        if action == 'SELL' and 'realized_pnl' in d:
            pnl     = d['realized_pnl']
            pnl_pct = d['realized_pnl_pct']
            verdict = '✅ PROFITABLE' if pnl >= 0 else '🔴 LOSS'
            sign    = '+' if pnl >= 0 else ''
            subject = f"[AI Trade Agent] SELL {symbol} — {verdict} {sign}${pnl:,.2f}"
            body = (
                f"{'='*48}\n"
                f"  {verdict}  |  {action} {symbol}\n"
                f"{'='*48}\n\n"
                f"  Agent:        {d.get('agent', '')}\n"
                f"  Symbol:       {symbol}\n"
                f"  Qty:          {qty} shares\n"
                f"  Avg Entry:    ${d.get('avg_entry_price', 0):,.4f}\n"
                f"  Exit Price:   ${price:,.4f}\n"
                f"  Realized P&L: {sign}${pnl:,.2f}  ({pnl_pct})\n\n"
                f"  Time:         {record['timestamp']}\n"
                f"  Order ID:     {d.get('order_id', '')}\n"
            )
        else:
            stop   = d.get('stop_loss')
            target = d.get('take_profit')
            subject = f"[AI Trade Agent] BUY {symbol} — {qty} shares @ ${price:,.2f}"
            body = (
                f"{'='*48}\n"
                f"  📈 BUY EXECUTED  |  {symbol}\n"
                f"{'='*48}\n\n"
                f"  Agent:        {d.get('agent', '')}\n"
                f"  Symbol:       {symbol}\n"
                f"  Qty:          {qty} shares\n"
                f"  Fill Price:   ${price:,.4f}\n"
            )
            if stop is not None:
                body += f"  Stop Loss:    ${stop:,.2f}\n"
            if target is not None:
                body += f"  Take Profit:  ${target:,.2f}\n"
            body += (
                f"\n  Time:         {record['timestamp']}\n"
                f"  Order ID:     {d.get('order_id', '')}\n"
            )
    else:
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


def _send_telegram(message: str) -> None:
    """Send message to Telegram if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are configured."""
    if os.getenv('TELEGRAM_ALERTS_ENABLED', 'true').lower() == 'false':
        return

    token = os.getenv('TELEGRAM_BOT_TOKEN', '').strip()
    chat_id = os.getenv('TELEGRAM_CHAT_ID', '').strip()

    if not token or not chat_id:
        return

    import requests as _requests
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        _requests.post(url, json={'chat_id': chat_id, 'text': message}, timeout=10)
    except Exception as e:
        log.debug(f"alerts: telegram failed: {e}")


# ---------------------------------------------------------------------------
# Daily buy summary
# ---------------------------------------------------------------------------

def send_daily_buy_summary(agent_filter: str = None) -> None:
    """
    Send one end-of-day email summarising all BUY orders placed today.
    Called by each daemon's run_performance_analysis() after market close.
    agent_filter: if set (e.g. 'StockAgent', 'OptionsAgent') only include that agent's buys.
    """
    email_from = os.getenv('ALERT_EMAIL_FROM', '').strip()
    email_to_raw = os.getenv('ALERT_EMAIL_TO', '').strip()
    if not email_from or not email_to_raw:
        return

    today = datetime.now().date().isoformat()
    buys = []
    try:
        with open(_ALERTS_LOG) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get('event') != 'trade_executed':
                    continue
                if not rec.get('timestamp', '').startswith(today):
                    continue
                d = rec.get('data', {})
                if d.get('action', '').upper() != 'BUY':
                    continue
                if agent_filter and d.get('agent') != agent_filter:
                    continue
                buys.append(d)
    except FileNotFoundError:
        return

    label = agent_filter or 'All Agents'

    if not buys:
        subject = f"[AI Trade Agent] {today} — No buys today ({label})"
        body = f"No BUY orders were placed today by {label}.\n"
    else:
        total_deployed = sum(b.get('fill_price', 0) * b.get('qty', 0) for b in buys)
        rows = []
        for b in buys:
            sym    = b.get('symbol', '')
            qty    = b.get('qty', 0)
            price  = b.get('fill_price', 0)
            cost   = price * qty
            stop   = b.get('stop_loss')
            target = b.get('take_profit')
            ts     = b.get('timestamp', rec.get('timestamp', ''))[:16]

            risk_str = ''
            if stop is not None:
                max_loss = (stop - price) * qty
                risk_str += f"  Stop: ${stop:,.2f}  (max loss ${max_loss:,.2f})"
            if target is not None:
                max_gain = (target - price) * qty
                risk_str += f"  Target: ${target:,.2f}  (max gain ${max_gain:,.2f})"

            rows.append(
                f"  {sym:<8}  {qty:>5} sh @ ${price:>9,.4f}  "
                f"cost ${cost:>10,.2f}  {ts}\n"
                + (f"           {risk_str}\n" if risk_str else '')
            )

        subject = f"[AI Trade Agent] {today} — {len(buys)} buy{'s' if len(buys)!=1 else ''} | ${total_deployed:,.2f} deployed ({label})"
        body = (
            f"{'='*60}\n"
            f"  DAILY BUY SUMMARY  |  {today}  |  {label}\n"
            f"{'='*60}\n\n"
            + ''.join(rows)
            + f"\n  Total deployed:  ${total_deployed:,.2f}\n"
            f"  Orders:          {len(buys)}\n"
        )

    smtp_host     = os.getenv('ALERT_SMTP_HOST', 'smtp.gmail.com')
    smtp_port     = int(os.getenv('ALERT_SMTP_PORT', '587'))
    smtp_user     = os.getenv('ALERT_SMTP_USER', email_from)
    smtp_password = os.getenv('ALERT_SMTP_PASSWORD', '')
    recipients    = [r.strip() for r in email_to_raw.split(',') if r.strip()]

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From']    = email_from
    msg['To']      = ', '.join(recipients)
    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(email_from, recipients, msg.as_string())
        log.info(f"alerts: daily buy summary sent ({len(buys)} buys) to {recipients}")
    except Exception as e:
        log.error(f"alerts: daily buy summary email failed ({e})")


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
                         qty: int, price: float, order_id: str,
                         pnl: float = None, pnl_pct: float = None,
                         avg_entry_price: float = None,
                         stop_price: float = None, target_price: float = None) -> None:
    action_up = action.upper()
    if action_up == 'SELL' and pnl is not None:
        sign = '+' if pnl >= 0 else ''
        verdict = '✅ PROFIT' if pnl >= 0 else '🔴 LOSS'
        message = (f"{agent_name}: SELL {qty} {symbol} @ ${price:.2f} | "
                   f"{verdict} {sign}${pnl:,.2f} ({sign}{pnl_pct:.1%})")
    else:
        message = f"{agent_name}: {action_up} {qty} {symbol} @ ${price:.2f}"

    data = {'agent': agent_name, 'symbol': symbol, 'action': action_up,
            'qty': qty, 'fill_price': price, 'order_id': order_id}
    if avg_entry_price is not None:
        data['avg_entry_price'] = round(avg_entry_price, 4)
    if pnl is not None:
        data['realized_pnl'] = round(pnl, 2)
        data['realized_pnl_pct'] = f"{pnl_pct:+.2%}"
    if stop_price is not None:
        data['stop_loss'] = stop_price
    if target_price is not None:
        data['take_profit'] = target_price

    send_alert(AlertLevel.INFO, 'trade_executed', message, data=data)


def alert_trade_failed(agent_name: str, symbol: str, error: str) -> None:
    send_alert(
        AlertLevel.WARNING,
        'trade_failed',
        f"{agent_name}: order for {symbol} failed — {error}",
        data={'agent': agent_name, 'symbol': symbol, 'error': error},
    )


def alert_live_trading_ready(agent_name: str, metrics: dict, equity: float,
                              paper_days: int, unmet: list[str]) -> None:
    """
    Log a live-trading readiness scorecard to journalctl and send a daily email.
    Called once per daily performance analysis cycle regardless of pass/fail.
    """
    today     = datetime.now().date().isoformat()
    is_ready  = not unmet
    verdict   = '✅ READY FOR LIVE TRADING' if is_ready else f'❌ NOT READY ({len(unmet)} gate{"s" if len(unmet) != 1 else ""} failing)'

    def _gate(label: str, keyword: str, actual: str, required: str) -> str:
        passed = not any(keyword.lower() in u.lower() for u in unmet)
        status = '✅ PASS' if passed else '❌ FAIL'
        return f"  {label:<20} {status}   {actual:<12} {required}\n"

    scorecard = (
        f"{'='*58}\n"
        f"  LIVE TRADING READINESS — {agent_name}  {today}\n"
        f"  {verdict}\n"
        f"{'='*58}\n"
        + _gate('Equity',       'equity',       f"${equity:,.0f}",                          '≥ $2,000')
        + _gate('Paper days',   'paper trading', f"{paper_days}d",                           '≥ 30d')
        + _gate('Total trades', 'trades',        f"{metrics.get('total_trades', 0)}",        '≥ 50')
        + _gate('Sharpe ratio', 'sharpe',        f"{metrics.get('sharpe', 0):.2f}",          '≥ 1.0')
        + _gate('Max drawdown', 'drawdown',      f"{metrics.get('max_dd', 1):.1%}",          '≤ 8.0%')
        + _gate('Win rate',     'win rate',      f"{metrics.get('win_rate', 0):.1%}",        '≥ 45.0%')
        + f"{'='*58}\n"
    )
    if is_ready:
        scorecard += "  ACTION: Set PAPER_TRADING=false in .env and restart.\n"
    else:
        scorecard += f"  Failing: {'; '.join(unmet)}\n"

    # Always log to journalctl so it appears in `journalctl -u ai-*-bot -f`
    for line in scorecard.splitlines():
        log.info(line)

    # Always send email scorecard
    _send_readiness_email(agent_name, scorecard, verdict, is_ready, today)

    # Also fire the JSONL alert (and Telegram if configured) when newly ready
    if is_ready:
        send_alert(
            AlertLevel.INFO,
            'live_trading_ready',
            f"{agent_name}: all live-trading criteria met — you may go live",
            data={'agent': agent_name, 'metrics': metrics,
                  'equity': equity, 'paper_days': paper_days},
        )


def _send_readiness_email(agent_name: str, scorecard: str, verdict: str,
                           is_ready: bool, today: str) -> None:
    email_from = os.getenv('ALERT_EMAIL_FROM', '').strip()
    email_to_raw = os.getenv('ALERT_EMAIL_TO', '').strip()
    if not email_from or not email_to_raw:
        return

    icon    = '✅' if is_ready else '❌'
    subject = f"[AI Trade Agent] {icon} {agent_name} Readiness — {verdict.split('(')[0].strip()} | {today}"
    recipients = [r.strip() for r in email_to_raw.split(',') if r.strip()]

    msg = MIMEText(scorecard)
    msg['Subject'] = subject
    msg['From']    = email_from
    msg['To']      = ', '.join(recipients)

    smtp_host     = os.getenv('ALERT_SMTP_HOST', 'smtp.gmail.com')
    smtp_port     = int(os.getenv('ALERT_SMTP_PORT', '587'))
    smtp_user     = os.getenv('ALERT_SMTP_USER', email_from)
    smtp_password = os.getenv('ALERT_SMTP_PASSWORD', '')
    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(email_from, recipients, msg.as_string())
        log.info(f"alerts: readiness email sent to {recipients}")
    except Exception as e:
        log.error(f"alerts: readiness email failed ({e})")


if __name__ == '__main__':
    # Quick smoke test: logs to file, prints to console, skips email if unconfigured
    print("Testing alerts system...")
    send_alert(AlertLevel.INFO, 'test', 'Alerts module loaded successfully.')
    send_alert(AlertLevel.WARNING, 'test', 'This is a warning alert test.')
    send_alert(AlertLevel.CRITICAL, 'test', 'This is a critical alert test.')
    print(f"Check {_ALERTS_LOG} for entries.")
