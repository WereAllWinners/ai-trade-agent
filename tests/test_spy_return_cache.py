"""Tests for outcome_tracker's SPY-return caching.

_fetch_spy_return was issuing one yf.download per closed trade — up to ~534
network calls in a single pass, which together with the uncached Alpaca lookups
is why the nightly tracker never finished inside its subprocess budget.

The subtle requirement is range coverage: the cached series must be keyed to the
range it actually spans and expand when a later trade falls outside it. A naive
"fetch once on the first call" cache silently returns None for every trade
outside the first pair's window.
"""
import datetime as dt
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'analysis'))

pd = pytest.importorskip('pandas')

import outcome_tracker as ot  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_caches():
    ot._SPY_RETURN_CACHE.clear()
    ot._SPY_SERIES_CACHE.clear()
    yield
    ot._SPY_RETURN_CACHE.clear()
    ot._SPY_SERIES_CACHE.clear()


def _series(start='2026-01-01', days=400, start_price=100.0, step=0.5):
    """A daily SPY close series (no weekend/holiday gaps — fine for these tests)."""
    idx = pd.date_range(start=start, periods=days, freq='D')
    return pd.Series([start_price + i * step for i in range(days)], index=idx)


class _Download:
    """yf.download stub returning a slice of a full synthetic series."""

    def __init__(self, full=None):
        self.full = full if full is not None else _series()
        self.calls = []

    def __call__(self, ticker, start=None, end=None, **kwargs):
        self.calls.append((start, end))
        sliced = self.full.loc[str(start):str(end)]
        return pd.DataFrame({'Close': sliced})


class TestOneFetchForManyTrades:
    def test_many_windows_trigger_a_single_download(self):
        dl = _Download()
        with patch.dict('sys.modules', {'yfinance': type('m', (), {'download': dl})}):
            for day in range(1, 20):
                ot._fetch_spy_return(f'2026-03-{day:02d}T10:00:00',
                                     f'2026-03-{day + 1:02d}T15:00:00')
        assert len(dl.calls) == 1, f'expected 1 download for overlapping windows, got {len(dl.calls)}'

    def test_identical_window_is_memoized(self):
        dl = _Download()
        with patch.dict('sys.modules', {'yfinance': type('m', (), {'download': dl})}):
            a = ot._fetch_spy_return('2026-03-02T10:00:00', '2026-03-06T15:00:00')
            b = ot._fetch_spy_return('2026-03-02T09:00:00', '2026-03-06T16:00:00')
        assert a == b
        assert len(dl.calls) == 1


class TestRangeCoverage:
    """The regression this file exists for."""

    def test_window_far_outside_the_first_range_still_resolves(self):
        dl = _Download()
        with patch.dict('sys.modules', {'yfinance': type('m', (), {'download': dl})}):
            early = ot._fetch_spy_return('2026-02-02T10:00:00', '2026-02-06T15:00:00')
            late = ot._fetch_spy_return('2026-11-02T10:00:00', '2026-11-06T15:00:00')

        assert early is not None, 'first window should resolve'
        assert late is not None, (
            'a trade outside the first fetched range must trigger an expanded '
            'refetch, not silently return None'
        )
        assert len(dl.calls) == 2, 'expected exactly one expansion refetch'

    def test_expanded_range_does_not_lose_earlier_coverage(self):
        dl = _Download()
        with patch.dict('sys.modules', {'yfinance': type('m', (), {'download': dl})}):
            first = ot._fetch_spy_return('2026-02-02T10:00:00', '2026-02-06T15:00:00')
            ot._fetch_spy_return('2026-11-02T10:00:00', '2026-11-06T15:00:00')
            ot._SPY_RETURN_CACHE.clear()      # force a re-slice, not a memo hit
            again = ot._fetch_spy_return('2026-02-02T10:00:00', '2026-02-06T15:00:00')
        assert again == first, 'expansion must union the range, never shrink it'


class TestCorrectness:
    def test_return_matches_the_window_endpoints(self):
        full = _series(start='2026-01-01', days=200, start_price=100.0, step=1.0)
        dl = _Download(full)
        with patch.dict('sys.modules', {'yfinance': type('m', (), {'download': dl})}):
            got = ot._fetch_spy_return('2026-02-01T10:00:00', '2026-02-11T15:00:00')
        window = full.loc['2026-02-01':'2026-02-11']
        expected = float((window.iloc[-1] - window.iloc[0]) / window.iloc[0])
        assert got == pytest.approx(expected)

    def test_entry_on_or_after_exit_returns_none_without_fetching(self):
        dl = _Download()
        with patch.dict('sys.modules', {'yfinance': type('m', (), {'download': dl})}):
            assert ot._fetch_spy_return('2026-03-05T10:00:00', '2026-03-05T15:00:00') is None
            assert ot._fetch_spy_return('2026-03-06T10:00:00', '2026-03-05T15:00:00') is None
        assert dl.calls == []


class TestFailureHandling:
    def test_download_failure_returns_none_and_is_not_retried(self):
        calls = {'n': 0}

        def boom(*a, **k):
            calls['n'] += 1
            raise RuntimeError('network down')

        with patch.dict('sys.modules', {'yfinance': type('m', (), {'download': boom})}):
            assert ot._fetch_spy_return('2026-03-02T10:00:00', '2026-03-06T15:00:00') is None
            assert ot._fetch_spy_return('2026-04-02T10:00:00', '2026-04-06T15:00:00') is None
        assert calls['n'] == 1, 'a hard failure should be remembered, not retried per trade'

    def test_empty_series_returns_none(self):
        empty = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
        with patch.dict('sys.modules',
                        {'yfinance': type('m', (), {'download': _Download(empty)})}):
            assert ot._fetch_spy_return('2026-03-02T10:00:00', '2026-03-06T15:00:00') is None

    def test_malformed_timestamp_returns_none(self):
        assert ot._fetch_spy_return('not-a-date', '2026-03-06T15:00:00') is None
