"""Tests for the persistent order-status cache.

Context: the outcome trackers issued one Alpaca get_order_by_id call per
trade-log row on every run — 3,390 sequential lookups (~4 min) against a 120 s
subprocess budget, so the nightly pass timed out before writing any outcome and
the outcomes table froze on 2026-07-02. See analysis/order_status_cache.py.

The correctness-critical property is NOT the speedup — it is which answers are
allowed to be cached. A terminal order is immutable and safe to cache forever.
A failed lookup ({'status': 'unknown'}) must never be cached, or a transient
network blip becomes a permanent misclassification — the exact bug diagnosed in
docs/sprint02-d7-findings.md.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))

from analysis.order_status_cache import OrderStatusCache, TERMINAL_STATUSES  # noqa: E402


def _filled(price=101.5, qty=10):
    return {'status': 'filled', 'filled_qty': qty, 'avg_price': price}


def _unknown():
    return {'status': 'unknown', 'filled_qty': 0, 'avg_price': None}


class _Counter:
    """Fetch stub that records how many times it was called."""

    def __init__(self, result):
        self.result = result
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.result() if callable(self.result) else self.result


# ---------------------------------------------------------------------------
# Caching behaviour
# ---------------------------------------------------------------------------

class TestTerminalCaching:
    def test_filled_order_is_fetched_once_then_served_from_cache(self, tmp_path):
        cache = OrderStatusCache(tmp_path / 'c.json')
        fetch = _Counter(_filled())

        first = cache.get_or_fetch('order-1', fetch)
        second = cache.get_or_fetch('order-1', fetch)
        third = cache.get_or_fetch('order-1', fetch)

        assert fetch.calls == 1, 'a terminal order must only ever be fetched once'
        assert first == second == third == _filled()

    @pytest.mark.parametrize('status', sorted(TERMINAL_STATUSES))
    def test_every_terminal_status_is_cached(self, tmp_path, status):
        cache = OrderStatusCache(tmp_path / 'c.json')
        fetch = _Counter({'status': status, 'filled_qty': 1, 'avg_price': 5.0})
        cache.get_or_fetch('o', fetch)
        cache.get_or_fetch('o', fetch)
        assert fetch.calls == 1, f'{status} is terminal and must be cached'

    @pytest.mark.parametrize('status', [
        'new', 'accepted', 'pending_new', 'partially_filled', 'held',
        'pending_cancel', 'calculated', 'done_for_day',
    ])
    def test_non_terminal_status_is_never_cached(self, tmp_path, status):
        """A status that can still change must be re-fetched every time."""
        cache = OrderStatusCache(tmp_path / 'c.json')
        fetch = _Counter({'status': status, 'filled_qty': 0, 'avg_price': None})
        cache.get_or_fetch('o', fetch)
        cache.get_or_fetch('o', fetch)
        assert fetch.calls == 2, f'{status} can still change — must not be cached'


class TestFailuresAreNeverCached:
    """The regression that matters most: a transient failure must stay retryable."""

    def test_unknown_status_is_not_cached(self, tmp_path):
        cache = OrderStatusCache(tmp_path / 'c.json')
        fetch = _Counter(_unknown())
        cache.get_or_fetch('o', fetch)
        cache.get_or_fetch('o', fetch)
        assert fetch.calls == 2, \
            "a failed lookup must never be cached — that is how a network blip " \
            "became a permanent 'unreconciled' misclassification (sprint02 D7)"

    def test_failed_then_successful_lookup_resolves(self, tmp_path):
        """The real-world D7 shape: first lookup fails, a later one succeeds."""
        results = [_unknown(), _unknown(), _filled(price=135.84, qty=6)]
        calls = {'n': 0}

        def fetch():
            r = results[min(calls['n'], len(results) - 1)]
            calls['n'] += 1
            return r

        cache = OrderStatusCache(tmp_path / 'c.json')
        assert cache.get_or_fetch('o', fetch)['avg_price'] is None
        assert cache.get_or_fetch('o', fetch)['avg_price'] is None
        assert cache.get_or_fetch('o', fetch)['avg_price'] == 135.84
        # And now that it is terminal, it sticks.
        before = calls['n']
        assert cache.get_or_fetch('o', fetch)['avg_price'] == 135.84
        assert calls['n'] == before, 'resolved order should now be cached'

    def test_unknown_entries_are_not_persisted(self, tmp_path):
        path = tmp_path / 'c.json'
        cache = OrderStatusCache(path)
        cache.get_or_fetch('o', _Counter(_unknown()))
        cache.flush()
        if path.exists():
            assert json.loads(path.read_text()) == {}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_cache_survives_a_new_instance(self, tmp_path):
        path = tmp_path / 'c.json'
        c1 = OrderStatusCache(path)
        c1.get_or_fetch('order-1', _Counter(_filled()))
        c1.flush()

        fetch = _Counter(_filled())
        c2 = OrderStatusCache(path)
        assert c2.get_or_fetch('order-1', fetch) == _filled()
        assert fetch.calls == 0, 'second run must not re-fetch a cached order'

    def test_periodic_flush_persists_progress_before_completion(self, tmp_path):
        """A run killed by the subprocess timeout must not lose everything."""
        path = tmp_path / 'c.json'
        cache = OrderStatusCache(path, flush_every=5)
        for i in range(12):
            cache.get_or_fetch(f'o{i}', _Counter(_filled(price=i)))
        # No explicit flush() — only the periodic ones have fired.
        assert path.exists(), 'progress should already be on disk'
        saved = json.loads(path.read_text())
        assert len(saved) >= 10, f'expected periodic flushes to persist, got {len(saved)}'

    def test_corrupt_cache_file_starts_cold_without_raising(self, tmp_path):
        path = tmp_path / 'c.json'
        path.write_text('{ this is not valid json')
        cache = OrderStatusCache(path)          # must not raise
        fetch = _Counter(_filled())
        assert cache.get_or_fetch('o', fetch) == _filled()
        assert fetch.calls == 1

    def test_non_dict_cache_file_starts_cold(self, tmp_path):
        path = tmp_path / 'c.json'
        path.write_text('["not", "a", "dict"]')
        cache = OrderStatusCache(path)
        assert cache.get_or_fetch('o', _Counter(_filled())) == _filled()

    def test_stale_non_terminal_entries_in_file_are_discarded_on_load(self, tmp_path):
        """Defence in depth: a file written by an older/buggier version that
        contains a non-terminal entry must not be trusted."""
        path = tmp_path / 'c.json'
        path.write_text(json.dumps({
            'good': _filled(),
            'bad': {'status': 'new', 'filled_qty': 0, 'avg_price': None},
        }))
        cache = OrderStatusCache(path)
        good_fetch = _Counter(_filled())
        bad_fetch = _Counter({'status': 'new', 'filled_qty': 0, 'avg_price': None})
        cache.get_or_fetch('good', good_fetch)
        cache.get_or_fetch('bad', bad_fetch)
        assert good_fetch.calls == 0, 'terminal entry should load from file'
        assert bad_fetch.calls == 1, 'non-terminal entry must be re-fetched'


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_order_id_returns_unknown_without_fetching(self, tmp_path):
        cache = OrderStatusCache(tmp_path / 'c.json')
        fetch = _Counter(_filled())
        for empty in (None, '', 0):
            assert cache.get_or_fetch(empty, fetch)['status'] == 'unknown'
        assert fetch.calls == 0

    def test_mutating_a_returned_dict_does_not_corrupt_the_cache(self, tmp_path):
        cache = OrderStatusCache(tmp_path / 'c.json')
        result = cache.get_or_fetch('o', _Counter(_filled()))
        result['avg_price'] = 999_999
        assert cache.get_or_fetch('o', _Counter(_filled()))['avg_price'] == 101.5

    def test_summary_reports_hits_and_misses(self, tmp_path):
        cache = OrderStatusCache(tmp_path / 'c.json')
        cache.get_or_fetch('a', _Counter(_filled()))   # miss
        cache.get_or_fetch('a', _Counter(_filled()))   # hit
        assert cache.hits == 1 and cache.misses == 1
        assert 'order-status cache' in cache.summary()
