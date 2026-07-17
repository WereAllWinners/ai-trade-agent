"""
tests/test_expiry_sentinel.py — R1 WI-3

Covers:
  - parse_occ_symbol: valid, malformed, wrong-length, stock ticker
  - tier boundaries: exactly 5/6 calendar days, exactly 2/3 trading days,
    a weekend-spanning boundary proving calendar-days != trading-days
  - compute_exposure arithmetic
  - no-options-held -> silent
  - paper/live dedup-key disambiguation
"""
import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'tools'))

from expiry_sentinel import (  # noqa: E402
    calendar_days_to_expiry, classify_tier, compute_exposure,
    parse_occ_symbol, run, scan_account, trading_days_to_expiry,
)


def _pos(symbol, qty):
    p = MagicMock()
    p.symbol = symbol
    p.qty = str(qty)
    return p


class TestParseOccSymbol:
    def test_valid_call(self):
        result = parse_occ_symbol('SPY250328C00560000')
        assert result == {
            'underlying': 'SPY', 'expiry': date(2025, 3, 28),
            'option_type': 'C', 'strike': 560.0,
        }

    def test_valid_put_multi_char_underlying(self):
        result = parse_occ_symbol('NVDA260710P00197500')
        assert result['underlying'] == 'NVDA'
        assert result['expiry'] == date(2026, 7, 10)
        assert result['option_type'] == 'P'
        assert result['strike'] == 197.5

    def test_stock_ticker_returns_none(self):
        assert parse_occ_symbol('AAPL') is None
        assert parse_occ_symbol('NVDA') is None

    def test_too_short_returns_none(self):
        assert parse_occ_symbol('SPY2503') is None

    def test_malformed_date_returns_none(self):
        assert parse_occ_symbol('SPY259999C00560000') is None

    def test_malformed_option_type_returns_none(self):
        assert parse_occ_symbol('SPY250328X00560000') is None

    def test_empty_string(self):
        assert parse_occ_symbol('') is None


class TestDayCounts:
    def test_calendar_days_simple(self):
        assert calendar_days_to_expiry(date(2026, 7, 20), today=date(2026, 7, 17)) == 3

    def test_calendar_days_same_day(self):
        assert calendar_days_to_expiry(date(2026, 7, 17), today=date(2026, 7, 17)) == 0

    def test_trading_days_no_weekend(self):
        # Monday 2026-07-13 -> Wednesday 2026-07-15: Tue, Wed both weekdays
        assert trading_days_to_expiry(date(2026, 7, 15), today=date(2026, 7, 13)) == 2

    def test_trading_days_spans_weekend(self):
        # 2026-07-17 is a Friday; 2026-07-20 is the following Monday.
        assert date(2026, 7, 17).weekday() == 4  # sanity: Friday
        cal = calendar_days_to_expiry(date(2026, 7, 20), today=date(2026, 7, 17))
        trd = trading_days_to_expiry(date(2026, 7, 20), today=date(2026, 7, 17))
        assert cal == 3       # Sat, Sun, Mon
        assert trd == 1       # only Monday counts as a trading day
        assert trd < cal      # the divergence this whole day-count split exists to capture

    def test_trading_days_already_expired(self):
        assert trading_days_to_expiry(date(2026, 7, 10), today=date(2026, 7, 17)) == 0


class TestClassifyTier:
    def test_exactly_5_calendar_days_is_warn(self):
        assert classify_tier(calendar_days=5, trading_days=5) == 'warn'

    def test_exactly_6_calendar_days_is_none(self):
        assert classify_tier(calendar_days=6, trading_days=6) is None

    def test_exactly_2_trading_days_is_critical(self):
        assert classify_tier(calendar_days=2, trading_days=2) == 'critical'

    def test_exactly_3_trading_days_is_not_critical(self):
        # 3 trading days, but still within the 5-calendar-day WARN window
        assert classify_tier(calendar_days=3, trading_days=3) == 'warn'

    def test_critical_takes_priority_over_warn(self):
        # weekend-spanning case: 3 calendar days but only 1 trading day
        assert classify_tier(calendar_days=3, trading_days=1) == 'critical'

    def test_beyond_both_windows_is_none(self):
        assert classify_tier(calendar_days=10, trading_days=8) is None


class TestComputeExposure:
    def test_matches_incident_arithmetic(self):
        # NVDA260710C00197500 x3, strike 197.50 -> $59,250 (RCA's own figure)
        assert compute_exposure(qty=3, strike=197.50) == 59250.0

    def test_single_contract(self):
        assert compute_exposure(qty=1, strike=200.0) == 20000.0


class TestScanAccount:
    def test_no_options_held_returns_empty(self):
        client = MagicMock()
        client.get_all_positions.return_value = [_pos('AAPL', 200), _pos('NVDA', 400)]
        assert scan_account(client, 'paper') == []

    def test_options_outside_window_not_flagged(self):
        far_expiry = (date.today().replace(year=date.today().year + 1)).strftime('%y%m%d')
        client = MagicMock()
        client.get_all_positions.return_value = [_pos(f'SPY{far_expiry}C00560000', 1)]
        assert scan_account(client, 'paper') == []

    def test_options_within_window_flagged_with_exposure(self):
        near = (date.today() + __import__('datetime').timedelta(days=1)).strftime('%y%m%d')
        client = MagicMock()
        client.get_all_positions.return_value = [_pos(f'SPY{near}C00560000', 2)]
        result = scan_account(client, 'paper')
        assert len(result) == 1
        assert result[0]['tier'] in ('warn', 'critical')
        assert result[0]['exposure'] == 2 * 100 * 560.0


class TestRunDedupKeys:
    def _run_with_flagged(self, symbol='SPY250328C00560000', account_only='paper'):
        client = MagicMock()
        client.get_all_positions.return_value = [_pos(symbol, 1)]
        client.get_account.return_value = MagicMock(cash='1000.0')
        clients = {account_only: client}
        with patch('expiry_sentinel._build_clients', return_value=clients), \
             patch('expiry_sentinel.classify_tier', return_value='critical'), \
             patch('expiry_sentinel._alerts.alert_once_per_day') as mock_dedup:
            exit_code = run(dry_run=False)
        return exit_code, mock_dedup

    def test_condition_key_includes_account_label(self):
        exit_code, mock_dedup = self._run_with_flagged(account_only='paper')
        assert exit_code == 1
        key = mock_dedup.call_args[0][0]
        assert 'paper' in key

    def test_paper_and_live_same_symbol_get_distinct_keys(self):
        symbol = 'SPY250328C00560000'
        client = MagicMock()
        client.get_all_positions.return_value = [_pos(symbol, 1)]
        client.get_account.return_value = MagicMock(cash='1000.0')
        clients = {'paper': client, 'live': client}
        with patch('expiry_sentinel._build_clients', return_value=clients), \
             patch('expiry_sentinel.classify_tier', return_value='critical'), \
             patch('expiry_sentinel._alerts.alert_once_per_day') as mock_dedup:
            run(dry_run=False)
        keys = [c[0][0] for c in mock_dedup.call_args_list]
        assert len(keys) == 2
        assert keys[0] != keys[1]
        assert 'paper' in keys[0] and 'live' in keys[1]


class TestDryRun:
    def test_dry_run_never_calls_dedup(self):
        client = MagicMock()
        client.get_all_positions.return_value = [_pos('SPY250328C00560000', 1)]
        client.get_account.return_value = MagicMock(cash='1000.0')
        with patch('expiry_sentinel._build_clients', return_value={'paper': client}), \
             patch('expiry_sentinel.classify_tier', return_value='warn'), \
             patch('expiry_sentinel._alerts.alert_once_per_day') as mock_dedup:
            exit_code = run(dry_run=True)
        assert exit_code == 1
        assert not mock_dedup.called
