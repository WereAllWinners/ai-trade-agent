"""
tests/test_build_macro_calendar.py — sprint04 F1.1 build-time extraction
tool. Fixtures use real strings pulled from the live Fed/BLS pages during
this sprint's planning (e.g. "17-18*" -> 2026-03-18, "Dec. 18, 2025"), plus a
synthetic single-day fixture since no real single-day (non-range) FOMC
meeting was observed live this session.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'tools'))

import build_macro_calendar as bmc


def _wrap(lines: list[str]) -> str:
    """Wrap plain text lines in enough HTML tags that _strip_tags round-trips
    them back to one-per-line, mirroring the real pages' structure."""
    return '\n'.join(f'<div>{line}</div>' for line in lines)


# ── extract_fomc_dates ──────────────────────────────────────────────────────────

class TestExtractFomcDates:
    def test_range_entry_uses_second_day(self):
        html = _wrap(['2026 FOMC Meetings', 'January', '27-28'])
        events = bmc.extract_fomc_dates(html)
        assert events == [{'date': '2026-01-28', 'time_et': '14:00', 'event': 'FOMC Statement'}]

    def test_real_asterisked_range_from_live_page(self):
        """Real example fetched live this sprint: 'March' / '17-18*' under
        the 2026 FOMC Meetings header -> 2026-03-18. The asterisk (SEP-
        associated meeting) is irrelevant to date extraction."""
        html = _wrap(['2026 FOMC Meetings', 'March', '17-18*'])
        events = bmc.extract_fomc_dates(html)
        assert events == [{'date': '2026-03-18', 'time_et': '14:00', 'event': 'FOMC Statement'}]

    def test_single_day_entry_synthetic(self):
        """No real single-day (non-range) FOMC meeting was observed live
        this session, but the parser must not assume ranges are guaranteed —
        synthetic fixture proves the single-day branch works."""
        html = _wrap(['2026 FOMC Meetings', 'August', '19'])
        events = bmc.extract_fomc_dates(html)
        assert events == [{'date': '2026-08-19', 'time_et': '14:00', 'event': 'FOMC Statement'}]

    def test_multiple_meetings_across_months(self):
        html = _wrap(['2026 FOMC Meetings', 'January', '27-28', 'March', '17-18*'])
        events = bmc.extract_fomc_dates(html)
        assert events == [
            {'date': '2026-01-28', 'time_et': '14:00', 'event': 'FOMC Statement'},
            {'date': '2026-03-18', 'time_et': '14:00', 'event': 'FOMC Statement'},
        ]

    def test_multiple_years(self):
        html = _wrap(['2026 FOMC Meetings', 'January', '27-28',
                       '2027 FOMC Meetings', 'January', '26-27'])
        events = bmc.extract_fomc_dates(html)
        assert events == [
            {'date': '2026-01-28', 'time_et': '14:00', 'event': 'FOMC Statement'},
            {'date': '2027-01-27', 'time_et': '14:00', 'event': 'FOMC Statement'},
        ]

    def test_unmatched_date_line_skipped_not_raised(self):
        """A date line matching neither the range nor single-day shape is
        skipped, not guessed at — degrade to a parse-skip."""
        html = _wrap(['2026 FOMC Meetings', 'January', 'TBD', 'March', '17-18*'])
        events = bmc.extract_fomc_dates(html)
        assert events == [{'date': '2026-03-18', 'time_et': '14:00', 'event': 'FOMC Statement'}]

    def test_date_line_before_any_year_header_skipped(self):
        html = _wrap(['January', '27-28'])
        events = bmc.extract_fomc_dates(html)
        assert events == []

    def test_malformed_html_returns_empty_list(self):
        events = bmc.extract_fomc_dates('<<<not html>>>')
        assert events == []

    def test_empty_html_returns_empty_list(self):
        events = bmc.extract_fomc_dates('')
        assert events == []


# ── extract_bls_dates ────────────────────────────────────────────────────────────

class TestExtractBlsDates:
    def test_real_cpi_release_line_from_live_page(self):
        """Real example fetched live this sprint from bls.gov/schedule/
        news_release/cpi.htm: 'Dec. 18, 2025' -> 2025-12-18."""
        html = _wrap(['November 2025', 'Scheduled for release at 8:30 A.M. (ET)',
                       'Dec. 18, 2025'])
        events = bmc.extract_bls_dates(html, 'CPI')
        assert events == [{'date': '2025-12-18', 'time_et': '08:30', 'event': 'CPI'}]

    def test_multiple_release_dates(self):
        html = _wrap(['Dec. 18, 2025', 'Jan. 13, 2026', 'Feb. 13, 2026'])
        events = bmc.extract_bls_dates(html, 'Employment Situation (NFP)')
        assert [e['date'] for e in events] == ['2025-12-18', '2026-01-13', '2026-02-13']
        assert all(e['event'] == 'Employment Situation (NFP)' for e in events)
        assert all(e['time_et'] == '08:30' for e in events)

    def test_no_dates_returns_empty_list(self):
        html = _wrap(['Nothing to see here'])
        events = bmc.extract_bls_dates(html, 'CPI')
        assert events == []

    def test_malformed_html_returns_empty_list(self):
        events = bmc.extract_bls_dates('<<<not html>>>', 'CPI')
        assert events == []

    def test_empty_html_returns_empty_list(self):
        events = bmc.extract_bls_dates('', 'CPI')
        assert events == []

    def test_period_after_month_abbreviation_is_optional(self):
        """BLS release lines sometimes omit the period after 3-letter month
        abbreviations (e.g. 'May' has no trailing dot to begin with)."""
        html = _wrap(['May 8, 2026'])
        events = bmc.extract_bls_dates(html, 'CPI')
        assert events == [{'date': '2026-05-08', 'time_et': '08:30', 'event': 'CPI'}]
