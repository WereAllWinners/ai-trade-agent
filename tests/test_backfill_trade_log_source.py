"""
tests/test_backfill_trade_log_source.py — sprint02 D4.4 targeted source-backfill.

Covers:
  - find_candidates: open-position scope, orphaned-recent scope, already-tagged
    rows excluded, out-of-scope (closed + stale) rows excluded, max_candidates cap
  - determine_source: read-only get_order_by_id lookup against both accounts
  - backfill_source: dry-run leaves the file untouched; --execute stamps only
    the targeted lines, preserving every other line byte-for-byte
  - End-to-end: a legacy BUY stamped via the backfill then pairs with a new
    source-tagged SELL in OutcomeTracker.match_and_calculate_pnl, producing a
    realized outcome instead of staying unmatched
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / 'scripts'
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_SCRIPTS_DIR / 'analysis') not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR / 'analysis'))

from backfill_trade_log_source import find_candidates, determine_source, backfill_source
from outcome_tracker import OutcomeTracker


def _position(symbol):
    p = MagicMock()
    p.symbol = symbol
    return p


def _client(positions=None, order_ids_known=None):
    """positions: list of symbols currently held. order_ids_known: set of
    order_ids get_order_by_id will resolve for this account (else raises)."""
    c = MagicMock()
    c.get_all_positions.return_value = [_position(s) for s in (positions or [])]
    known = set(order_ids_known or [])

    def _get_order_by_id(order_id):
        if order_id in known:
            return MagicMock(id=order_id)
        raise Exception("order not found")

    c.get_order_by_id.side_effect = _get_order_by_id
    return c


def _write_rows(path, rows):
    with open(path, 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')


def _recent_ts(days_ago=1):
    return (datetime.now() - timedelta(days=days_ago)).isoformat()


def _stale_ts():
    return (datetime.now() - timedelta(days=200)).isoformat()


# ---------------------------------------------------------------------------
# find_candidates — scoping
# ---------------------------------------------------------------------------

class TestFindCandidates:
    def test_open_position_buy_included_regardless_of_age(self, tmp_path):
        trade_log_path = tmp_path / 'trade_log.jsonl'
        _write_rows(trade_log_path, [
            {'timestamp': _stale_ts(), 'symbol': 'ALKS', 'action': 'buy',
             'order_id': 'live-buy-1'},
        ])
        paper_client = _client(positions=[])
        live_client = _client(positions=['ALKS'])

        candidates = find_candidates(trade_log_path, paper_client, live_client)

        assert len(candidates) == 1
        assert candidates[0]['symbol'] == 'ALKS'
        assert candidates[0]['reason'] == 'open_position'

    def test_orphaned_recent_buy_with_no_sell_included(self, tmp_path):
        trade_log_path = tmp_path / 'trade_log.jsonl'
        _write_rows(trade_log_path, [
            {'timestamp': _recent_ts(2), 'symbol': 'MRK', 'action': 'buy',
             'order_id': 'buy-orphan-1'},
        ])
        paper_client = _client(positions=[])
        live_client = _client(positions=[])

        candidates = find_candidates(trade_log_path, paper_client, live_client, lookback_days=30)

        assert len(candidates) == 1
        assert candidates[0]['reason'] == 'orphaned_recent'

    def test_closed_and_stale_buy_excluded(self, tmp_path):
        """Not currently open, not recent, and already has a matching sell —
        squarely out of scope, must not be guessed at."""
        trade_log_path = tmp_path / 'trade_log.jsonl'
        _write_rows(trade_log_path, [
            {'timestamp': _stale_ts(), 'symbol': 'XOM', 'action': 'buy',
             'order_id': 'buy-old-1'},
            {'timestamp': _stale_ts(), 'symbol': 'XOM', 'action': 'sell',
             'order_id': 'sell-old-1'},
        ])
        paper_client = _client(positions=[])
        live_client = _client(positions=[])

        candidates = find_candidates(trade_log_path, paper_client, live_client, lookback_days=30)

        assert candidates == []

    def test_stale_orphan_without_sell_excluded_by_lookback(self, tmp_path):
        """Old + unmatched + not currently open — outside the lookback window,
        must not be swept in just because it's unmatched."""
        trade_log_path = tmp_path / 'trade_log.jsonl'
        _write_rows(trade_log_path, [
            {'timestamp': _stale_ts(), 'symbol': 'CVX', 'action': 'buy',
             'order_id': 'buy-stale-orphan'},
        ])
        paper_client = _client(positions=[])
        live_client = _client(positions=[])

        candidates = find_candidates(trade_log_path, paper_client, live_client, lookback_days=30)

        assert candidates == []

    def test_already_tagged_rows_excluded(self, tmp_path):
        trade_log_path = tmp_path / 'trade_log.jsonl'
        _write_rows(trade_log_path, [
            {'timestamp': _recent_ts(1), 'symbol': 'ALKS', 'action': 'buy',
             'order_id': 'tagged-buy-1', 'source': 'live'},
        ])
        paper_client = _client(positions=[])
        live_client = _client(positions=['ALKS'])

        candidates = find_candidates(trade_log_path, paper_client, live_client)

        assert candidates == []

    def test_sell_rows_never_returned(self, tmp_path):
        trade_log_path = tmp_path / 'trade_log.jsonl'
        _write_rows(trade_log_path, [
            {'timestamp': _recent_ts(1), 'symbol': 'ALKS', 'action': 'sell',
             'order_id': 'sell-1'},
        ])
        paper_client = _client(positions=[])
        live_client = _client(positions=['ALKS'])

        candidates = find_candidates(trade_log_path, paper_client, live_client)

        assert candidates == []

    def test_max_candidates_cap_only_applies_to_orphaned_recent(self, tmp_path):
        """open_position candidates are bounded by real broker state, not an
        arbitrary limit — verified against a real account, this can
        legitimately run into the hundreds (sprint02 D4.4 finding). The cap
        only needs to guard the genuinely open-ended orphaned_recent scope."""
        trade_log_path = tmp_path / 'trade_log.jsonl'
        rows = [
            {'timestamp': _recent_ts(1), 'symbol': f'SYM{i}', 'action': 'buy',
             'order_id': f'buy-{i}'}
            for i in range(10)
        ]
        _write_rows(trade_log_path, rows)
        paper_client = _client(positions=[f'SYM{i}' for i in range(10)])
        live_client = _client(positions=[])

        candidates = find_candidates(trade_log_path, paper_client, live_client, max_candidates=3)

        assert len(candidates) == 10
        assert all(c['reason'] == 'open_position' for c in candidates)

    def test_max_candidates_cap_enforced_on_orphaned_recent(self, tmp_path):
        trade_log_path = tmp_path / 'trade_log.jsonl'
        rows = [
            {'timestamp': _recent_ts(1), 'symbol': f'SYM{i}', 'action': 'buy',
             'order_id': f'buy-{i}'}
            for i in range(10)
        ]
        _write_rows(trade_log_path, rows)
        paper_client = _client(positions=[])
        live_client = _client(positions=[])

        candidates = find_candidates(trade_log_path, paper_client, live_client, max_candidates=3)

        assert len(candidates) == 3
        assert all(c['reason'] == 'orphaned_recent' for c in candidates)


# ---------------------------------------------------------------------------
# determine_source — dual-client read-only lookup
# ---------------------------------------------------------------------------

class TestDetermineSource:
    def test_resolves_paper_when_paper_client_has_order(self):
        paper_client = _client(order_ids_known=['oid-1'])
        live_client = _client(order_ids_known=[])

        assert determine_source('oid-1', paper_client, live_client) == 'paper'

    def test_resolves_live_when_live_client_has_order(self):
        paper_client = _client(order_ids_known=[])
        live_client = _client(order_ids_known=['oid-2'])

        assert determine_source('oid-2', paper_client, live_client) == 'live'

    def test_returns_none_when_neither_client_has_order(self):
        paper_client = _client(order_ids_known=[])
        live_client = _client(order_ids_known=[])

        assert determine_source('oid-3', paper_client, live_client) is None


# ---------------------------------------------------------------------------
# backfill_source — dry-run vs execute, in-place targeted rewrite
# ---------------------------------------------------------------------------

class TestBackfillSource:
    def test_dry_run_does_not_modify_file(self, tmp_path):
        trade_log_path = tmp_path / 'trade_log.jsonl'
        _write_rows(trade_log_path, [
            {'timestamp': _recent_ts(1), 'symbol': 'ALKS', 'action': 'buy',
             'order_id': 'oid-1'},
        ])
        before = trade_log_path.read_text()
        paper_client = _client(positions=[], order_ids_known=[])
        live_client = _client(positions=['ALKS'], order_ids_known=['oid-1'])

        summary = backfill_source(trade_log_path, paper_client=paper_client,
                                   live_client=live_client, dry_run=True)

        assert trade_log_path.read_text() == before
        assert summary['candidates'] == 1
        assert summary['stamped'] == 0
        assert summary['report'][0]['resolved_source'] == 'live'

    def test_execute_stamps_only_targeted_lines(self, tmp_path):
        trade_log_path = tmp_path / 'trade_log.jsonl'
        _write_rows(trade_log_path, [
            {'timestamp': _recent_ts(1), 'symbol': 'ALKS', 'action': 'buy',
             'order_id': 'oid-1'},
            {'timestamp': _recent_ts(1), 'symbol': 'UNRELATED', 'action': 'buy',
             'order_id': 'oid-unrelated', 'source': 'paper'},
        ])
        paper_client = _client(positions=[], order_ids_known=[])
        live_client = _client(positions=['ALKS'], order_ids_known=['oid-1'])

        summary = backfill_source(trade_log_path, paper_client=paper_client,
                                   live_client=live_client, dry_run=False)

        assert summary['stamped'] == 1
        lines = [json.loads(l) for l in trade_log_path.read_text().splitlines()]
        assert lines[0]['source'] == 'live'
        assert lines[1]['source'] == 'paper'
        assert lines[1]['order_id'] == 'oid-unrelated'

    def test_unresolved_candidate_left_untouched(self, tmp_path):
        trade_log_path = tmp_path / 'trade_log.jsonl'
        _write_rows(trade_log_path, [
            {'timestamp': _recent_ts(1), 'symbol': 'ALKS', 'action': 'buy',
             'order_id': 'oid-purged'},
        ])
        paper_client = _client(positions=[], order_ids_known=[])
        live_client = _client(positions=['ALKS'], order_ids_known=[])

        summary = backfill_source(trade_log_path, paper_client=paper_client,
                                   live_client=live_client, dry_run=False)

        assert summary['stamped'] == 0
        assert summary['unresolved'] == 1
        line = json.loads(trade_log_path.read_text().strip())
        assert 'source' not in line

    def test_no_candidates_is_a_clean_noop(self, tmp_path):
        trade_log_path = tmp_path / 'trade_log.jsonl'
        _write_rows(trade_log_path, [
            {'timestamp': _recent_ts(1), 'symbol': 'AAPL', 'action': 'buy',
             'order_id': 'oid-tagged', 'source': 'paper'},
        ])
        paper_client = _client(positions=[])
        live_client = _client(positions=[])

        summary = backfill_source(trade_log_path, paper_client=paper_client,
                                   live_client=live_client, dry_run=False)

        assert summary == {'candidates': 0, 'stamped': 0, 'unresolved': 0, 'errors': 0, 'report': []}


# ---------------------------------------------------------------------------
# End-to-end: backfilled BUY pairs with a new tagged SELL via OutcomeTracker
# ---------------------------------------------------------------------------

class TestBackfillClosesOutcomeTrackerGap:
    def test_legacy_buy_stamped_then_pairs_with_tagged_sell(self, tmp_path):
        trade_log_path = tmp_path / 'trade_log.jsonl'
        _write_rows(trade_log_path, [
            {'timestamp': '2026-03-01T09:00:00', 'symbol': 'AEP', 'action': 'buy',
             'shares': 8.0, 'order_id': 'legacy-buy-aep'},  # no 'source' — pre-D4.3
            {'timestamp': '2026-07-01T09:00:00', 'symbol': 'AEP', 'action': 'sell',
             'shares': 8.0, 'order_id': 'tagged-sell-aep', 'source': 'live'},
        ])
        paper_client = _client(positions=[], order_ids_known=[])
        live_client = _client(positions=['AEP'], order_ids_known=['legacy-buy-aep'])

        summary = backfill_source(trade_log_path, paper_client=paper_client,
                                   live_client=live_client, dry_run=False)
        assert summary['stamped'] == 1

        tracker = OutcomeTracker.__new__(OutcomeTracker)
        tracker.trade_log_path = trade_log_path
        tracker.paper = False  # live tracker instance

        fill_prices = {'legacy-buy-aep': 130.0, 'tagged-sell-aep': 128.0}
        with patch.object(tracker, 'get_fill_price', side_effect=lambda oid: fill_prices.get(oid)), \
             patch('outcome_tracker._fetch_spy_return', return_value=None):
            trades = tracker.load_raw_trades()
            outcomes = tracker.match_and_calculate_pnl(trades)

        assert len(outcomes) == 1
        assert outcomes[0]['buy_order_id'] == 'legacy-buy-aep'
        assert outcomes[0]['sell_order_id'] == 'tagged-sell-aep'

    def test_out_of_scope_buy_stays_unmatched_not_guessed(self, tmp_path):
        """A buy that's neither a currently-open position nor recent/orphaned
        must be left alone by the backfill — and therefore still fails to
        pair against a tagged sell, same as before D4.4 ran."""
        trade_log_path = tmp_path / 'trade_log.jsonl'
        _write_rows(trade_log_path, [
            {'timestamp': _stale_ts(), 'symbol': 'XOM', 'action': 'buy',
             'shares': 3.0, 'order_id': 'legacy-buy-xom'},
            {'timestamp': _stale_ts(), 'symbol': 'XOM', 'action': 'sell',
             'shares': 3.0, 'order_id': 'legacy-sell-xom'},
            {'timestamp': _recent_ts(1), 'symbol': 'XOM', 'action': 'sell',
             'shares': 3.0, 'order_id': 'tagged-sell-xom', 'source': 'live'},
        ])
        paper_client = _client(positions=[], order_ids_known=[])
        live_client = _client(positions=[], order_ids_known=[])  # XOM not held open

        summary = backfill_source(trade_log_path, paper_client=paper_client,
                                   live_client=live_client, dry_run=False)

        # Both XOM buy/sell already have a matching pair (legacy-buy/legacy-sell),
        # so neither buy row is a candidate — nothing stamped.
        assert summary['stamped'] == 0

        tracker = OutcomeTracker.__new__(OutcomeTracker)
        tracker.trade_log_path = trade_log_path
        tracker.paper = False

        fill_prices = {'legacy-buy-xom': 100.0, 'legacy-sell-xom': 101.0,
                        'tagged-sell-xom': 103.0}
        with patch.object(tracker, 'get_fill_price', side_effect=lambda oid: fill_prices.get(oid)), \
             patch.object(tracker, 'get_order_status', return_value={'status': 'filled'}), \
             patch('outcome_tracker._fetch_spy_return', return_value=None), \
             patch('outcome_tracker._db'):
            trades = tracker.load_raw_trades()
            outcomes = tracker.match_and_calculate_pnl(trades)

        # The legacy pair still matches each other (both source=None); the new
        # tagged sell has no buy left to pair with — correctly unmatched, not
        # guessed at by grabbing the already-consumed legacy buy.
        assert len(outcomes) == 1
        assert outcomes[0]['buy_order_id'] == 'legacy-buy-xom'
        assert outcomes[0]['sell_order_id'] == 'legacy-sell-xom'
