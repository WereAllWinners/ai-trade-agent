"""
Unit tests for alerts.py — Telegram channel, flag behavior, and existing channels.

All tests are fully offline — no SMTP, no Telegram API, no file I/O side effects.
"""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, call

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))

import alerts
from alerts import AlertLevel, send_alert, _send_telegram


# ── _send_telegram ─────────────────────────────────────────────────────────────

class TestSendTelegram:
    def test_posts_to_correct_url(self):
        """_send_telegram POSTs to the Telegram sendMessage endpoint."""
        mock_post = MagicMock()
        with patch.dict('os.environ', {
            'TELEGRAM_BOT_TOKEN': 'mytoken123',
            'TELEGRAM_CHAT_ID': '987654321',
            'TELEGRAM_ALERTS_ENABLED': 'true',
        }), patch('requests.post', mock_post):
            _send_telegram('Test message')

        mock_post.assert_called_once()
        url = mock_post.call_args[0][0]
        assert 'mytoken123' in url
        assert 'sendMessage' in url

    def test_sends_correct_payload(self):
        """Payload contains chat_id and text."""
        mock_post = MagicMock()
        with patch.dict('os.environ', {
            'TELEGRAM_BOT_TOKEN': 'tok',
            'TELEGRAM_CHAT_ID': '111',
            'TELEGRAM_ALERTS_ENABLED': 'true',
        }), patch('requests.post', mock_post):
            _send_telegram('Hello trading alert')

        kwargs = mock_post.call_args[1]
        assert kwargs['json']['chat_id'] == '111'
        assert kwargs['json']['text'] == 'Hello trading alert'

    def test_uses_10s_timeout(self):
        """Request is sent with a 10-second timeout."""
        mock_post = MagicMock()
        with patch.dict('os.environ', {
            'TELEGRAM_BOT_TOKEN': 'tok',
            'TELEGRAM_CHAT_ID': '111',
        }), patch('requests.post', mock_post):
            _send_telegram('msg')

        assert mock_post.call_args[1]['timeout'] == 10

    def test_no_token_skips_request(self):
        """Missing TELEGRAM_BOT_TOKEN means no HTTP call is made."""
        mock_post = MagicMock()
        with patch.dict('os.environ', {'TELEGRAM_CHAT_ID': '111'}, clear=False), \
             patch.dict('os.environ', {'TELEGRAM_BOT_TOKEN': ''}), \
             patch('requests.post', mock_post):
            _send_telegram('msg')
        mock_post.assert_not_called()

    def test_no_chat_id_skips_request(self):
        """Missing TELEGRAM_CHAT_ID means no HTTP call is made."""
        mock_post = MagicMock()
        with patch.dict('os.environ', {'TELEGRAM_BOT_TOKEN': 'tok'}), \
             patch.dict('os.environ', {'TELEGRAM_CHAT_ID': ''}), \
             patch('requests.post', mock_post):
            _send_telegram('msg')
        mock_post.assert_not_called()

    def test_alerts_disabled_skips_request(self):
        """TELEGRAM_ALERTS_ENABLED=false prevents any HTTP call."""
        mock_post = MagicMock()
        with patch.dict('os.environ', {
            'TELEGRAM_BOT_TOKEN': 'tok',
            'TELEGRAM_CHAT_ID': '111',
            'TELEGRAM_ALERTS_ENABLED': 'false',
        }), patch('requests.post', mock_post):
            _send_telegram('msg')
        mock_post.assert_not_called()

    def test_network_exception_is_silenced(self):
        """A network error does not propagate — _send_telegram never raises."""
        with patch.dict('os.environ', {
            'TELEGRAM_BOT_TOKEN': 'tok',
            'TELEGRAM_CHAT_ID': '111',
        }), patch('requests.post', side_effect=ConnectionError('timeout')):
            _send_telegram('msg')  # must not raise

    def test_enabled_true_uppercase_accepted(self):
        """TELEGRAM_ALERTS_ENABLED=True (any case) still sends."""
        mock_post = MagicMock()
        with patch.dict('os.environ', {
            'TELEGRAM_BOT_TOKEN': 'tok',
            'TELEGRAM_CHAT_ID': '111',
            'TELEGRAM_ALERTS_ENABLED': 'True',
        }), patch('requests.post', mock_post):
            _send_telegram('msg')
        mock_post.assert_called_once()


# ── send_alert integration ─────────────────────────────────────────────────────

class TestSendAlertCallsTelegram:
    def test_send_alert_calls_telegram(self):
        """send_alert() invokes _send_telegram with a non-empty message."""
        with patch.dict('os.environ', {
            'TELEGRAM_BOT_TOKEN': 'tok',
            'TELEGRAM_CHAT_ID': '111',
            'TELEGRAM_ALERTS_ENABLED': 'true',
        }), \
             patch('alerts._write_to_log'), \
             patch('alerts._send_email'), \
             patch('requests.post') as mock_post:
            send_alert(AlertLevel.CRITICAL, 'circuit_breaker', 'Daily loss -5%')

        mock_post.assert_called_once()
        text = mock_post.call_args[1]['json']['text']
        assert 'circuit_breaker' in text
        assert 'Daily loss' in text

    def test_send_alert_message_includes_level(self):
        """The Telegram message includes the alert level."""
        with patch.dict('os.environ', {
            'TELEGRAM_BOT_TOKEN': 'tok',
            'TELEGRAM_CHAT_ID': '111',
        }), \
             patch('alerts._write_to_log'), \
             patch('alerts._send_email'), \
             patch('requests.post') as mock_post:
            send_alert(AlertLevel.WARNING, 'test_event', 'Something happened')

        text = mock_post.call_args[1]['json']['text']
        assert 'WARNING' in text

    def test_telegram_failure_does_not_break_other_channels(self):
        """If Telegram raises, the alert still writes to log and email."""
        written = []
        with patch.dict('os.environ', {
            'TELEGRAM_BOT_TOKEN': 'tok',
            'TELEGRAM_CHAT_ID': '111',
        }), \
             patch('alerts._write_to_log', side_effect=lambda r: written.append(r)), \
             patch('alerts._send_email'), \
             patch('requests.post', side_effect=RuntimeError('boom')):
            send_alert(AlertLevel.INFO, 'test', 'msg')  # must not raise

        assert len(written) == 1
