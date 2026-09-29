"""
Unit tests for economic_calendar — halt logic, formatting, static-calendar
loading, staleness checking.

All tests are fully offline — no network calls (sprint04 F1 replaced the
Finnhub integration with a static committed JSON file).
"""
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'data'))

import economic_calendar


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


def _write_calendar(tmp_path, events) -> Path:
    path = tmp_path / 'macro_event_calendar.json'
    with open(path, 'w') as f:
        json.dump({'generated_at': 'test', 'source_urls': [], 'events': events}, f)
    return path


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


# ── get_todays_high_impact_events (sprint04 F1: static file, no network) ──────

class TestGetTodaysHighImpactEvents:
    def test_missing_file_returns_empty_list(self, tmp_path):
        missing = tmp_path / 'does_not_exist.json'
        with patch('economic_calendar._CALENDAR_PATH', missing):
            result = economic_calendar.get_todays_high_impact_events()
        assert result == []

    def test_malformed_json_returns_empty_list(self, tmp_path):
        path = tmp_path / 'bad.json'
        path.write_text('{not valid json')
        with patch('economic_calendar._CALENDAR_PATH', path):
            result = economic_calendar.get_todays_high_impact_events()
        assert result == []

    def test_no_events_key_returns_empty_list(self, tmp_path):
        path = tmp_path / 'no_events.json'
        path.write_text(json.dumps({'generated_at': 'test'}))
        with patch('economic_calendar._CALENDAR_PATH', path):
            result = economic_calendar.get_todays_high_impact_events()
        assert result == []

    def test_guard_disabled_returns_empty_without_reading_file(self, tmp_path):
        cal_path = _write_calendar(tmp_path, [
            {'date': '2020-01-01', 'time_et': '14:00', 'event': 'FOMC Statement'},
        ])
        original = economic_calendar.MACRO_GUARD_ENABLED
        try:
            economic_calendar.MACRO_GUARD_ENABLED = False
            with patch('economic_calendar._CALENDAR_PATH', cal_path):
                result = economic_calendar.get_todays_high_impact_events()
            assert result == []
        finally:
            economic_calendar.MACRO_GUARD_ENABLED = original

    def test_non_today_events_excluded(self, tmp_path):
        """Events dated anything other than today (ET) are filtered out."""
        cal_path = _write_calendar(tmp_path, [
            {'date': '2020-01-01', 'time_et': '14:00', 'event': 'Old FOMC Statement'},
            {'date': '2099-01-01', 'time_et': '14:00', 'event': 'Far Future CPI'},
        ])
        with patch('economic_calendar._CALENDAR_PATH', cal_path):
            result = economic_calendar.get_todays_high_impact_events()
        assert result == []

    def test_returns_impact_high_and_country_us(self, tmp_path):
        today_et = datetime.now(economic_calendar._ET).date().isoformat()
        cal_path = _write_calendar(tmp_path, [
            {'date': today_et, 'time_et': '14:00', 'event': 'FOMC Statement'},
        ])
        with patch('economic_calendar._CALENDAR_PATH', cal_path):
            result = economic_calendar.get_todays_high_impact_events()
        assert len(result) == 1
        assert result[0]['impact'] == 'high'
        assert result[0]['country'] == 'US'
        assert result[0]['event'] == 'FOMC Statement'

    def test_malformed_event_entry_skipped_not_raised(self, tmp_path):
        today_et = datetime.now(economic_calendar._ET).date().isoformat()
        cal_path = _write_calendar(tmp_path, [
            {'date': 'not-a-date', 'time_et': '14:00', 'event': 'Bad Date'},
            {'date': today_et, 'time_et': 'not-a-time', 'event': 'Bad Time'},
            {'date': today_et, 'time_et': '14:00', 'event': 'Good Event'},
        ])
        with patch('economic_calendar._CALENDAR_PATH', cal_path):
            result = economic_calendar.get_todays_high_impact_events()
        assert len(result) == 1
        assert result[0]['event'] == 'Good Event'

    def test_et_to_utc_conversion_summer_edt(self, tmp_path):
        """Summer: 14:00 ET (EDT, UTC-4) -> 18:00 UTC. 'Today' is frozen to a
        known summer date so the DST offset is deterministic across runs."""
        class _FixedDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                fixed = datetime(2026, 7, 8, 9, 0, tzinfo=economic_calendar._ET)
                return fixed.astimezone(tz) if tz else fixed

        cal_path = _write_calendar(tmp_path, [
            {'date': '2026-07-08', 'time_et': '14:00', 'event': 'FOMC Statement'},
        ])
        with patch('economic_calendar._CALENDAR_PATH', cal_path), \
             patch('economic_calendar.datetime', _FixedDatetime):
            result = economic_calendar.get_todays_high_impact_events()
        assert len(result) == 1
        assert result[0]['time'] == '2026-07-08 18:00:00'

    def test_et_to_utc_conversion_winter_est(self, tmp_path):
        """Winter: 14:00 ET (EST, UTC-5) -> 19:00 UTC. Same fixed-'today'
        technique as the summer test, pinned to a known winter date."""
        class _FixedDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                fixed = datetime(2026, 1, 8, 9, 0, tzinfo=economic_calendar._ET)
                return fixed.astimezone(tz) if tz else fixed

        cal_path = _write_calendar(tmp_path, [
            {'date': '2026-01-08', 'time_et': '14:00', 'event': 'FOMC Statement'},
        ])
        with patch('economic_calendar._CALENDAR_PATH', cal_path), \
             patch('economic_calendar.datetime', _FixedDatetime):
            result = economic_calendar.get_todays_high_impact_events()
        assert len(result) == 1
        assert result[0]['time'] == '2026-01-08 19:00:00'


# ── check_calendar_staleness ───────────────────────────────────────────────────

class TestCheckCalendarStaleness:
    def test_no_future_events_fires_alert(self, tmp_path):
        cal_path = _write_calendar(tmp_path, [
            {'date': '2020-01-01', 'time_et': '14:00', 'event': 'FOMC Statement'},
        ])
        fake_dedup_path = tmp_path / 'alert_dedup.json'
        with patch('economic_calendar._CALENDAR_PATH', cal_path), \
             patch('alerts._ALERT_DEDUP_PATH', fake_dedup_path), \
             patch('alerts.send_alert') as mock_send_alert:
            economic_calendar.check_calendar_staleness()
        mock_send_alert.assert_called_once()

    def test_future_events_present_no_alert(self, tmp_path):
        future_date = (datetime.now(timezone.utc) + timedelta(days=30)).strftime('%Y-%m-%d')
        cal_path = _write_calendar(tmp_path, [
            {'date': future_date, 'time_et': '14:00', 'event': 'FOMC Statement'},
        ])
        fake_dedup_path = tmp_path / 'alert_dedup.json'
        with patch('economic_calendar._CALENDAR_PATH', cal_path), \
             patch('alerts._ALERT_DEDUP_PATH', fake_dedup_path), \
             patch('alerts.send_alert') as mock_send_alert:
            economic_calendar.check_calendar_staleness()
        mock_send_alert.assert_not_called()

    def test_missing_file_treated_as_stale_fires_alert(self, tmp_path):
        missing = tmp_path / 'does_not_exist.json'
        fake_dedup_path = tmp_path / 'alert_dedup.json'
        with patch('economic_calendar._CALENDAR_PATH', missing), \
             patch('alerts._ALERT_DEDUP_PATH', fake_dedup_path), \
             patch('alerts.send_alert') as mock_send_alert:
            economic_calendar.check_calendar_staleness()
        mock_send_alert.assert_called_once()

    def test_two_calls_same_day_alert_exactly_once(self, tmp_path):
        """Mirrors sprint02 D5b's dedup precedent: this process restarts every
        ~30 min, so a naive per-session guard would over-alert."""
        cal_path = _write_calendar(tmp_path, [
            {'date': '2020-01-01', 'time_et': '14:00', 'event': 'FOMC Statement'},
        ])
        fake_dedup_path = tmp_path / 'alert_dedup.json'
        with patch('economic_calendar._CALENDAR_PATH', cal_path), \
             patch('alerts._ALERT_DEDUP_PATH', fake_dedup_path), \
             patch('alerts.send_alert') as mock_send_alert:
            economic_calendar.check_calendar_staleness()
            economic_calendar.check_calendar_staleness()
        mock_send_alert.assert_called_once()

    def test_never_raises_on_unexpected_error(self, tmp_path):
        """Fails open: an unexpected error while checking staleness must not
        propagate and block trading."""
        with patch('economic_calendar._load_static_calendar', side_effect=RuntimeError("boom")):
            economic_calendar.check_calendar_staleness()  # must not raise


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
