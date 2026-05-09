"""
Unit tests for economic_calendar — halt logic, formatting, API failure safety.

All tests are fully offline — no Finnhub API, no real network calls.
"""
import json
import sys
import tempfile
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'data'))

import economic_calendar


def _fresh_cache_path() -> Path:
    """Return a unique non-existent temp path (no file created)."""
    return Path(tempfile.gettempdir()) / f'econ_test_{uuid.uuid4().hex}.json'


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_event(name: str, offset_minutes: float = 0, impact: str = 'high',
                country: str = 'US') -> dict:
    """Build a synthetic event dict. offset_minutes: + = future, - = past."""
    event_utc = datetime.now(timezone.utc) + timedelta(minutes=offset_minutes)
    return {
        'time':    event_utc.strftime('%Y-%m-%dT%H:%M:%S+00:00'),
        'event':   name,
        'impact':  impact,
        'country': country,
    }


# ── should_halt_trading ───────────────────────────────────────────────────────

class TestShouldHaltTrading:
    def test_empty_list_returns_false(self):
        halt, reason = economic_calendar.should_halt_trading([])
        assert halt is False
        assert reason == ''

    def test_fomc_in_20_min_triggers_halt(self):
        """FOMC 20 minutes in the future is within the 30-minute pre-halt window."""
        events = [_make_event('FOMC Rate Decision', offset_minutes=20)]
        halt, reason = economic_calendar.should_halt_trading(events)
        assert halt is True
        assert 'FOMC' in reason

    def test_fomc_30_min_exactly_triggers_halt(self):
        """Exactly 30 minutes out is still in the halt window (≤30)."""
        events = [_make_event('FOMC Rate Decision', offset_minutes=30)]
        halt, reason = economic_calendar.should_halt_trading(events)
        assert halt is True

    def test_fomc_45_min_away_no_halt(self):
        """45 minutes before is outside the window — no halt."""
        events = [_make_event('FOMC Rate Decision', offset_minutes=45)]
        halt, _ = economic_calendar.should_halt_trading(events)
        assert halt is False

    def test_cpi_30_min_ago_triggers_halt(self):
        """CPI that occurred 30 minutes ago is within the 60-minute post-halt window."""
        events = [_make_event('CPI (Consumer Price Index)', offset_minutes=-30)]
        halt, reason = economic_calendar.should_halt_trading(events)
        assert halt is True
        assert 'CPI' in reason or 'Consumer Price Index' in reason

    def test_cpi_65_min_ago_no_halt(self):
        """CPI 65 minutes ago is outside the 60-minute post-halt window."""
        events = [_make_event('CPI Consumer Price Index', offset_minutes=-65)]
        halt, _ = economic_calendar.should_halt_trading(events)
        assert halt is False

    def test_nfp_in_halt_window_triggers(self):
        """Nonfarm Payroll release in 10 minutes triggers halt."""
        events = [_make_event('Nonfarm Payroll', offset_minutes=10)]
        halt, _ = economic_calendar.should_halt_trading(events)
        assert halt is True

    def test_non_halt_event_does_not_halt(self):
        """A high-impact event that is NOT FOMC/NFP/CPI does not trigger halt."""
        events = [_make_event('ISM Manufacturing PMI', offset_minutes=5)]
        halt, _ = economic_calendar.should_halt_trading(events)
        assert halt is False

    def test_guard_disabled_never_halts(self):
        """When MACRO_GUARD_ENABLED=False, halt is always False regardless of events."""
        events = [_make_event('FOMC Rate Decision', offset_minutes=5)]
        original = economic_calendar.MACRO_GUARD_ENABLED
        try:
            economic_calendar.MACRO_GUARD_ENABLED = False
            halt, _ = economic_calendar.should_halt_trading(events)
            assert halt is False
        finally:
            economic_calendar.MACRO_GUARD_ENABLED = original

    def test_unparseable_time_is_skipped_gracefully(self):
        """An event with an unparseable time string is skipped without raising."""
        events = [{'time': 'not-a-date', 'event': 'FOMC Rate Decision',
                   'impact': 'high', 'country': 'US'}]
        halt, _ = economic_calendar.should_halt_trading(events)
        assert halt is False


# ── get_todays_high_impact_events ─────────────────────────────────────────────

class TestGetTodaysHighImpactEvents:
    def test_api_failure_returns_empty_list(self):
        """Network error from Finnhub returns [] without raising."""
        with patch('requests.get', side_effect=ConnectionError("timeout")), \
             patch('economic_calendar._FINNHUB_KEY', 'fake_key'), \
             patch('economic_calendar._cache_path') as mock_path:
            # Point cache to a non-existent path so we don't hit the file cache
            mock_path.return_value = _fresh_cache_path()
            result = economic_calendar.get_todays_high_impact_events()
        assert result == []

    def test_no_api_key_returns_empty_list(self):
        """Missing FINNHUB_API_KEY returns [] without making any network call."""
        with patch('economic_calendar._FINNHUB_KEY', ''), \
             patch('economic_calendar._cache_path') as mock_path, \
             patch('requests.get') as mock_get:
            mock_path.return_value = _fresh_cache_path()
            result = economic_calendar.get_todays_high_impact_events()
        assert result == []
        mock_get.assert_not_called()

    def test_filters_non_us_events(self):
        """Events from non-US countries are excluded."""
        fake_resp = MagicMock()
        fake_resp.json.return_value = {
            'economicCalendar': [
                {'country': 'EU', 'event': 'ECB Rate Decision',
                 'impact': '3', 'time': '2024-01-15T12:00:00'},
                {'country': 'US', 'event': 'FOMC Minutes',
                 'impact': '3', 'time': '2024-01-15T14:00:00'},
            ]
        }
        fake_resp.raise_for_status = MagicMock()

        with patch('requests.get', return_value=fake_resp), \
             patch('economic_calendar._FINNHUB_KEY', 'fake_key'), \
             patch('economic_calendar._cache_path') as mock_path:
            mock_path.return_value = _fresh_cache_path()
            result = economic_calendar.get_todays_high_impact_events()

        assert len(result) == 1
        assert result[0]['country'] == 'US'

    def test_filters_non_high_impact_events(self):
        """Low and medium impact events are excluded."""
        fake_resp = MagicMock()
        fake_resp.json.return_value = {
            'economicCalendar': [
                {'country': 'US', 'event': 'Building Permits', 'impact': '1', 'time': '2024-01-15T13:30:00'},
                {'country': 'US', 'event': 'FOMC Rate Decision', 'impact': '3', 'time': '2024-01-15T19:00:00'},
            ]
        }
        fake_resp.raise_for_status = MagicMock()

        with patch('requests.get', return_value=fake_resp), \
             patch('economic_calendar._FINNHUB_KEY', 'fake_key'), \
             patch('economic_calendar._cache_path') as mock_path:
            mock_path.return_value = _fresh_cache_path()
            result = economic_calendar.get_todays_high_impact_events()

        assert len(result) == 1
        assert result[0]['event'] == 'FOMC Rate Decision'

    def test_file_cache_is_used_on_second_call(self):
        """Second call within the same day reads from file cache, not API."""
        events_data = [{'time': '2024-01-15T14:00:00', 'event': 'FOMC',
                        'impact': 'high', 'country': 'US'}]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(events_data, f)
            cache_file = Path(f.name)

        try:
            with patch('economic_calendar._cache_path', return_value=cache_file), \
                 patch('requests.get') as mock_get:
                result = economic_calendar.get_todays_high_impact_events()

            mock_get.assert_not_called()
            assert result == events_data
        finally:
            cache_file.unlink(missing_ok=True)

    def test_guard_disabled_returns_empty_without_api_call(self):
        """When MACRO_GUARD_ENABLED=False, returns [] immediately."""
        original = economic_calendar.MACRO_GUARD_ENABLED
        try:
            economic_calendar.MACRO_GUARD_ENABLED = False
            with patch('requests.get') as mock_get:
                result = economic_calendar.get_todays_high_impact_events()
            assert result == []
            mock_get.assert_not_called()
        finally:
            economic_calendar.MACRO_GUARD_ENABLED = original


# ── format_macro_guard_block ──────────────────────────────────────────────────

class TestFormatMacroGuardBlock:
    def test_empty_inputs_return_empty_string(self):
        assert economic_calendar.format_macro_guard_block([], []) == ''

    def test_events_only(self):
        events = [{'time': '2024-01-15T19:00:00+00:00', 'event': 'FOMC Rate Decision',
                   'impact': 'high', 'country': 'US'}]
        output = economic_calendar.format_macro_guard_block(events, [])
        assert 'FOMC Rate Decision' in output
        assert 'HIGH IMPACT' in output
        assert '⚠️' in output

    def test_earnings_only(self):
        output = economic_calendar.format_macro_guard_block([], ['NVDA', 'AAPL'])
        assert 'NVDA' in output
        assert 'AAPL' in output
        assert 'IV crush' in output

    def test_both_events_and_earnings(self):
        events = [{'time': '2024-01-15T19:00:00+00:00', 'event': 'CPI',
                   'impact': 'high', 'country': 'US'}]
        output = economic_calendar.format_macro_guard_block(events, ['TSLA'])
        assert 'CPI' in output
        assert 'TSLA' in output
        # Should be two lines
        assert output.count('⚠️') == 2
