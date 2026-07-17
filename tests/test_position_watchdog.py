"""
tests/test_position_watchdog.py — R1 WI-2 (WI-7's cash-invariant extension
is added in tests/test_position_watchdog.py by a later work item; not here).

Covers:
  - all services active -> no alerts, no exposure check attempted
  - service down, holds positions/orders in its instrument class -> alert
  - service down, holds nothing in its instrument class -> no alert
  - service down, account fetch raises -> WARNING alert (degraded info,
    never silently treated as clean)
  - unit not installed ('not_found') -> skipped entirely, not an alert
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'tools'))

import position_watchdog  # noqa: E402


def _pos(symbol):
    p = MagicMock()
    p.symbol = symbol
    return p


def _order(symbol):
    o = MagicMock()
    o.symbol = symbol
    return o


def _run_with(states: dict, paper_positions=None, paper_orders=None,
              paper_fetch_error=None, dry_run=False):
    """states: {unit: 'active'|'inactive'|'not_found'}. Only a paper client
    is configured (live omitted) unless a test needs otherwise."""
    paper_client = MagicMock()
    if paper_fetch_error is not None:
        paper_client.get_all_positions.side_effect = paper_fetch_error
        paper_client.get_orders.side_effect = paper_fetch_error
    else:
        paper_client.get_all_positions.return_value = paper_positions or []
        paper_client.get_orders.return_value = paper_orders or []

    with patch('position_watchdog._build_clients', return_value={'paper': paper_client}), \
         patch('position_watchdog.service_state', side_effect=lambda u: states.get(u, 'not_found')), \
         patch('position_watchdog._alerts.alert_with_backoff') as mock_backoff, \
         patch('position_watchdog._alerts.alert_service_down_with_positions') as mock_alert_exposed, \
         patch('position_watchdog._alerts.send_alert') as mock_send_alert:
        exit_code = position_watchdog.run(dry_run=dry_run)
    return exit_code, mock_backoff, mock_alert_exposed, mock_send_alert


ALL_ACTIVE = {
    'ai-trading-bot.service': 'active',
    'ai-trading-bot-live.service': 'not_found',
    'ai-options-bot.service': 'active',
    'ai-options-bot-live.service': 'not_found',
}


class TestAllUp:
    def test_no_alert_when_all_active(self):
        exit_code, mock_backoff, _, _ = _run_with(ALL_ACTIVE)
        assert exit_code == 0
        assert not mock_backoff.called


class TestDownWithPositions:
    def test_alert_fires_with_position_detail(self):
        states = dict(ALL_ACTIVE, **{'ai-options-bot.service': 'inactive'})
        exit_code, mock_backoff, _, _ = _run_with(
            states,
            paper_positions=[_pos('SPY250328P00560000')],
            paper_orders=[],
        )
        assert exit_code == 1
        assert mock_backoff.called
        call_key = mock_backoff.call_args[0][0]
        assert 'watchdog_exposed_paper_options' == call_key

    def test_alert_ignores_stock_positions_for_options_unit(self):
        # A down OPTIONS unit with only STOCK positions held is still clean
        # for the options instrument class -- bot_positions() must filter.
        states = dict(ALL_ACTIVE, **{'ai-options-bot.service': 'inactive'})
        exit_code, mock_backoff, _, _ = _run_with(
            states, paper_positions=[_pos('AAPL'), _pos('NVDA')], paper_orders=[],
        )
        assert exit_code == 0
        assert not mock_backoff.called


class TestDownWithoutPositions:
    def test_no_alert_when_down_but_clean(self):
        states = dict(ALL_ACTIVE, **{'ai-options-bot.service': 'inactive'})
        exit_code, mock_backoff, _, _ = _run_with(states, paper_positions=[], paper_orders=[])
        assert exit_code == 0
        assert not mock_backoff.called


class TestApiError:
    def test_fetch_failure_alerts_degraded_not_silent(self):
        states = dict(ALL_ACTIVE, **{'ai-options-bot.service': 'inactive'})
        exit_code, mock_backoff, _, _ = _run_with(
            states, paper_fetch_error=Exception('Alpaca API unavailable'),
        )
        assert exit_code == 1
        assert mock_backoff.called
        call_key = mock_backoff.call_args[0][0]
        assert call_key == 'watchdog_unknown_paper_options'


class TestNotFoundUnit:
    def test_not_found_unit_skipped_entirely_no_alert(self):
        # ai-options-bot-live.service may not be installed yet (live options
        # is equity-gated) -- must never alert forever as a false positive.
        states = {
            'ai-trading-bot.service': 'active',
            'ai-trading-bot-live.service': 'not_found',
            'ai-options-bot.service': 'active',
            'ai-options-bot-live.service': 'not_found',
        }
        exit_code, mock_backoff, _, mock_send_alert = _run_with(states)
        assert exit_code == 0
        assert not mock_backoff.called
        assert not mock_send_alert.called


class TestDryRun:
    def test_dry_run_never_calls_alert_with_backoff(self):
        states = dict(ALL_ACTIVE, **{'ai-options-bot.service': 'inactive'})
        exit_code, mock_backoff, _, _ = _run_with(
            states, paper_positions=[_pos('SPY250328P00560000')], paper_orders=[], dry_run=True,
        )
        assert exit_code == 1  # still reports what it would have done
        assert not mock_backoff.called
