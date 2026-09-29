"""Tests for the reasoning backfill.

Repairs decisions.reasoning and outcomes.entry_reasoning by re-deriving them
from the preserved raw_response. See scripts/backfill_decision_reasoning.py.

Every test passes `db_path=` explicitly rather than patching _db.DB_PATH — the
repo convention, and a necessary one: db.py helpers bind `db_path: Path = DB_PATH`
as a default at import, so patching the module attribute does NOT redirect them
and writes land on the live database.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))

import db as _db  # noqa: E402
from backfill_decision_reasoning import (  # noqa: E402
    backfill_reasoning, _is_contaminated, _clean_from_raw,
)

CONTAMINATED_RAW = (
    ' Decision: SELL\nConfidence: 0.90\n'
    'Reasoning: Decision: SELL Confidence: 0.90 Reasoning: Oversold with negative momentum.\n'
    'Outcome: Small win (-0.6%). Reward signal: -0.0004'
)
CONTAMINATED_STORED = (
    'Decision: SELL Confidence: 0.90 Reasoning: Oversold with negative momentum.'
)
CLEAN_RAW = 'Decision: BUY\nConfidence: 0.85\nReasoning: Strong breakout on volume.'


@pytest.fixture()
def db_path(tmp_path):
    p = tmp_path / 'test_trading.db'
    _db.init_db(p)
    return p


def _insert_decision(db_path, *, symbol='AAPL', reasoning='', raw_response='',
                     order_id=None, decision='buy'):
    _db.insert_decision({
        'timestamp': '2026-09-01T10:00:00', 'session_id': 's1', 'bot': 'stock',
        'model': 'test', 'symbol': symbol, 'prompt': f'analyze {symbol}',
        'raw_response': raw_response, 'decision': decision, 'confidence': 0.8,
        'reasoning': reasoning, 'executed': True, 'indicators': {},
        'order_id': order_id,
    }, db_path=db_path)


def _reasonings(db_path):
    with _db.get_conn(db_path) as conn:
        return [r['reasoning'] for r in conn.execute(
            'SELECT reasoning FROM decisions ORDER BY id')]


# ---------------------------------------------------------------------------

class TestContaminationDetection:
    @pytest.mark.parametrize('text', [
        'Decision: BUY Confidence: 0.8 Reasoning: x',
        'Good setup. Outcome: Small win (+4.2%)',
        'Momentum. Reward signal: +0.0003',
    ])
    def test_detects_contamination(self, text):
        assert _is_contaminated(text) is True

    @pytest.mark.parametrize('text', [
        'Strong breakout on volume.', '', None,
        'RSI is oversold and volume is elevated.',
    ])
    def test_clean_text_is_not_flagged(self, text):
        assert _is_contaminated(text) is False


class TestCleanFromRaw:
    def test_unwraps_nested_block_and_strips_trailer(self):
        assert _clean_from_raw(CONTAMINATED_RAW) == 'Oversold with negative momentum.'

    def test_clean_raw_is_preserved(self):
        assert _clean_from_raw(CLEAN_RAW) == 'Strong breakout on volume.'

    def test_outcome_only_raw_yields_empty(self):
        assert _clean_from_raw('Outcome: Small win (+3.2%). Reward signal: +0.0006') == ''

    def test_empty_input(self):
        assert _clean_from_raw('') == ''
        assert _clean_from_raw(None) == ''

    def test_result_never_contains_newline(self):
        """_is_clean_ideal_output requires exactly 3 lines downstream."""
        assert '\n' not in _clean_from_raw(CONTAMINATED_RAW)


class TestDryRunMakesNoWrites:
    def test_dry_run_leaves_rows_untouched(self, db_path):
        _insert_decision(db_path, reasoning=CONTAMINATED_STORED,
                         raw_response=CONTAMINATED_RAW)
        before = _reasonings(db_path)

        summary = backfill_reasoning(db_path=db_path, dry_run=True)

        assert _reasonings(db_path) == before, 'dry run must not write'
        assert summary['candidates'] == 1
        assert summary['rewritten'] == 1, 'dry run still reports what it would do'
        assert summary['report'], 'report is populated in dry-run mode too'

    def test_report_shows_before_and_after(self, db_path):
        _insert_decision(db_path, reasoning=CONTAMINATED_STORED,
                         raw_response=CONTAMINATED_RAW)
        entry = backfill_reasoning(db_path=db_path, dry_run=True)['report'][0]
        assert 'Decision:' in entry['before']
        assert 'Decision:' not in entry['after']


class TestExecuteRepairsRows:
    def test_contaminated_row_is_repaired(self, db_path):
        _insert_decision(db_path, reasoning=CONTAMINATED_STORED,
                         raw_response=CONTAMINATED_RAW)
        summary = backfill_reasoning(db_path=db_path, dry_run=False)
        assert summary['rewritten'] == 1
        assert _reasonings(db_path) == ['Oversold with negative momentum.']

    def test_clean_row_is_left_alone(self, db_path):
        _insert_decision(db_path, reasoning='Strong breakout on volume.',
                         raw_response=CLEAN_RAW)
        summary = backfill_reasoning(db_path=db_path, dry_run=False)
        assert summary['candidates'] == 0
        assert summary['unchanged'] == 1
        assert _reasonings(db_path) == ['Strong breakout on volume.']

    def test_unrecoverable_row_is_skipped_not_blanked(self, db_path):
        """Blanking would introduce a new state; these rows are already filtered."""
        stored = 'Outcome: Small win (+3.2%). Reward signal: +0.0006'
        _insert_decision(db_path, reasoning=stored, raw_response=stored)
        summary = backfill_reasoning(db_path=db_path, dry_run=False)
        assert summary['skipped_empty'] == 1
        assert summary['rewritten'] == 0
        assert _reasonings(db_path) == [stored], 'must be left untouched, not emptied'

    def test_raw_response_is_never_modified(self, db_path):
        """The backfill must stay reversible from source."""
        _insert_decision(db_path, reasoning=CONTAMINATED_STORED,
                         raw_response=CONTAMINATED_RAW)
        backfill_reasoning(db_path=db_path, dry_run=False)
        with _db.get_conn(db_path) as conn:
            assert conn.execute(
                'SELECT raw_response FROM decisions').fetchone()[0] == CONTAMINATED_RAW

    def test_is_idempotent(self, db_path):
        _insert_decision(db_path, reasoning=CONTAMINATED_STORED,
                         raw_response=CONTAMINATED_RAW)
        backfill_reasoning(db_path=db_path, dry_run=False)
        after_first = _reasonings(db_path)
        second = backfill_reasoning(db_path=db_path, dry_run=False)
        assert second['candidates'] == 0, 'a second run should find nothing to do'
        assert _reasonings(db_path) == after_first

    def test_limit_caps_the_sweep(self, db_path):
        for i in range(5):
            _insert_decision(db_path, symbol=f'SYM{i}', reasoning=CONTAMINATED_STORED,
                             raw_response=CONTAMINATED_RAW)
        summary = backfill_reasoning(db_path=db_path, limit=2, dry_run=False)
        assert summary['rewritten'] == 2
        assert sum(1 for r in _reasonings(db_path) if 'Decision:' in r) == 3


class TestOutcomesEntryReasoning:
    """The second contaminated copy — missing it lets contamination flow back."""

    def _seed_outcome(self, db_path, entry_reasoning):
        _insert_decision(db_path, reasoning=CONTAMINATED_STORED,
                         raw_response=CONTAMINATED_RAW, order_id='buy-1')
        _db.insert_outcome({
            'symbol': 'AAPL', 'buy_order_id': 'buy-1', 'sell_order_id': 'sell-1',
            'entry_timestamp': '2026-09-01T10:00:00',
            'exit_timestamp': '2026-09-02T10:00:00',
            'entry_price': 100.0, 'exit_price': 110.0, 'shares': 10,
            'realized_pnl': 100.0, 'pnl_pct': 0.1, 'hold_hours': 24.0,
            'entry_confidence': 0.8, 'entry_reasoning': entry_reasoning, 'won': True,
        }, db_path=db_path)

    def _entry_reasoning(self, db_path):
        with _db.get_conn(db_path) as conn:
            return conn.execute('SELECT entry_reasoning FROM outcomes').fetchone()[0]

    def test_contaminated_entry_reasoning_is_repaired(self, db_path):
        self._seed_outcome(db_path, CONTAMINATED_STORED)
        summary = backfill_reasoning(db_path=db_path, dry_run=False)
        assert summary['outcomes_rewritten'] == 1
        assert self._entry_reasoning(db_path) == 'Oversold with negative momentum.'

    def test_dry_run_does_not_touch_outcomes(self, db_path):
        self._seed_outcome(db_path, CONTAMINATED_STORED)
        backfill_reasoning(db_path=db_path, dry_run=True)
        assert self._entry_reasoning(db_path) == CONTAMINATED_STORED

    def test_clean_entry_reasoning_is_left_alone(self, db_path):
        self._seed_outcome(db_path, 'Already clean reasoning.')
        summary = backfill_reasoning(db_path=db_path, dry_run=False)
        assert summary['outcomes_rewritten'] == 0
        assert self._entry_reasoning(db_path) == 'Already clean reasoning.'


class TestRepairedRowsSurviveTheCleanlinessFilter:
    """End-to-end: repaired reasoning must produce an ideal_output the builder keeps."""

    def test_repaired_reasoning_builds_a_clean_ideal_output(self, db_path):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'training'))
        from training_data_builder import _is_clean_ideal_output, _build_ideal_output

        _insert_decision(db_path, reasoning=CONTAMINATED_STORED,
                         raw_response=CONTAMINATED_RAW)

        before = _build_ideal_output('sell', 0.9, CONTAMINATED_STORED, 'winner')
        assert not _is_clean_ideal_output(before), 'precondition: was being discarded'

        backfill_reasoning(db_path=db_path, dry_run=False)
        repaired = _reasonings(db_path)[0]
        after = _build_ideal_output('sell', 0.9, repaired, 'winner')
        assert _is_clean_ideal_output(after), \
            'repaired reasoning must yield an ideal_output the builder keeps'

    def test_outcome_without_a_matching_decision_is_still_repaired(self, db_path):
        """248 of 527 contaminated outcomes have no decisions.order_id match.

        An inner join would leave them contaminated; the in-place unwrap of the
        stored value recovers them using the same algorithm.
        """
        _db.insert_outcome({
            'symbol': 'TSLA', 'buy_order_id': 'orphan-order', 'sell_order_id': 'sell-9',
            'entry_timestamp': '2026-09-01T10:00:00',
            'exit_timestamp': '2026-09-02T10:00:00',
            'entry_price': 100.0, 'exit_price': 90.0, 'shares': 5,
            'realized_pnl': -50.0, 'pnl_pct': -0.1, 'hold_hours': 24.0,
            'entry_confidence': 0.7, 'entry_reasoning': CONTAMINATED_STORED, 'won': False,
        }, db_path=db_path)

        summary = backfill_reasoning(db_path=db_path, dry_run=False)

        assert summary['outcomes_rewritten'] == 1
        with _db.get_conn(db_path) as conn:
            got = conn.execute('SELECT entry_reasoning FROM outcomes').fetchone()[0]
        assert got == 'Oversold with negative momentum.'
        assert 'Decision:' not in got
