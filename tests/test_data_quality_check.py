"""
tests/test_data_quality_check.py — sprint03 E4 standing data-quality gate.
"""
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'analysis'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'training'))

import db as _db
import data_quality_check as dq


@pytest.fixture()
def db_path(tmp_path):
    p = tmp_path / 'test.db'
    _db.init_db(p)
    return p


@pytest.fixture()
def trade_log_path(tmp_path):
    return tmp_path / 'trade_log.jsonl'


def _write_rows(path, rows):
    with open(path, 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')


def _recent_ts(days_ago=1):
    return (datetime.now() - timedelta(days=days_ago)).isoformat()


def _old_ts(days_ago=30):
    return (datetime.now() - timedelta(days=days_ago)).isoformat()


_WINNER_OUT = 'Decision: BUY\nConfidence: 0.90\nReasoning: Strong momentum breakout.'
_HOLD_OUT   = 'Decision: HOLD\nConfidence: 0.80\nReasoning: No clear edge; hold best.'


def _sft_row(label, i=0):
    return {
        'input': f'Analyze SYM{i} for a potential trade.',
        'output': _WINNER_OUT if label in dq._TAKE_TRADE_LABELS else _HOLD_OUT,
        'label': label,
        'weight': 1.0,
        'metadata': {'entry_date': '2026-06-15'},
    }


class TestSourceCoverageCheck:
    @pytest.fixture(autouse=True)
    def _pin_source_tagging_cutoff(self, monkeypatch):
        """_source_tagging_since() derives its cutoff from the mtime of the
        writer files (autonomous_agent.py, options_agent.py, ...) — any edit
        to those files (e.g. sprint04 F1.3's check_calendar_staleness()
        wiring) pushes the cutoff to 'now' and silently exempts recent test
        rows from the coverage check. Pin it so this test doesn't depend on
        real file mtimes."""
        monkeypatch.setenv('DQ_SOURCE_TAGGING_SINCE', (datetime.now() - timedelta(days=7)).isoformat())

    def test_all_recent_rows_tagged_passes(self, trade_log_path):
        _write_rows(trade_log_path, [
            {'timestamp': _recent_ts(1), 'symbol': 'AAPL', 'action': 'buy', 'order_id': 'o1', 'source': 'paper'},
        ])
        result = dq._check_source_coverage(trade_log_path, None)
        assert result['status'] == 'pass'

    def test_recent_row_missing_source_fails(self, trade_log_path):
        _write_rows(trade_log_path, [
            {'timestamp': _recent_ts(1), 'symbol': 'AAPL', 'action': 'buy', 'order_id': 'o1'},
        ])
        result = dq._check_source_coverage(trade_log_path, None)
        assert result['status'] == 'fail'

    def test_old_row_missing_source_does_not_fail(self, trade_log_path):
        """Legacy rows outside the recency window are expected to lack source."""
        _write_rows(trade_log_path, [
            {'timestamp': _old_ts(30), 'symbol': 'AAPL', 'action': 'buy', 'order_id': 'o1'},
        ])
        result = dq._check_source_coverage(trade_log_path, None)
        assert result['status'] == 'pass'

    def test_missing_file_passes(self, tmp_path):
        result = dq._check_source_coverage(tmp_path / 'nonexistent.jsonl', None)
        assert result['status'] == 'pass'


class TestOrphanRateCheck:
    def test_no_orphans_passes(self, trade_log_path):
        _write_rows(trade_log_path, [
            {'timestamp': _recent_ts(1), 'symbol': 'A', 'action': 'buy', 'order_id': 'o1', 'source': 'paper'},
            {'timestamp': _recent_ts(1), 'symbol': 'B', 'action': 'buy', 'order_id': 'o2', 'source': 'live'},
        ])
        result = dq._check_orphan_rate(trade_log_path, None)
        assert result['status'] == 'pass'
        assert result['metric'] == 0.0

    def test_high_orphan_rate_fails(self, trade_log_path):
        rows = [
            {'timestamp': _recent_ts(1), 'symbol': f'S{i}', 'action': 'buy',
             'order_id': f'o{i}', 'source': 'orphan'}
            for i in range(10)
        ]
        rows += [{'timestamp': _recent_ts(1), 'symbol': 'X', 'action': 'buy',
                   'order_id': 'ox', 'source': 'paper'}]
        _write_rows(trade_log_path, rows)
        result = dq._check_orphan_rate(trade_log_path, None)
        assert result['status'] == 'fail'

    def test_low_orphan_rate_warns(self, trade_log_path):
        rows = [
            {'timestamp': _recent_ts(1), 'symbol': f'S{i}', 'action': 'buy',
             'order_id': f'o{i}', 'source': 'paper'}
            for i in range(29)
        ]
        rows += [{'timestamp': _recent_ts(1), 'symbol': 'X', 'action': 'buy',
                   'order_id': 'ox', 'source': 'orphan'}]
        _write_rows(trade_log_path, rows)
        result = dq._check_orphan_rate(trade_log_path, None)
        assert result['status'] == 'warn'


class TestDuplicateOrderIdsCheck:
    def test_no_duplicates_passes(self, trade_log_path):
        _write_rows(trade_log_path, [
            {'timestamp': _recent_ts(1), 'symbol': 'A', 'action': 'buy', 'order_id': 'o1', 'source': 'paper'},
            {'timestamp': _recent_ts(1), 'symbol': 'A', 'action': 'sell', 'order_id': 'o2', 'source': 'paper'},
        ])
        result = dq._check_duplicate_order_ids(trade_log_path, None)
        assert result['status'] == 'pass'

    def test_same_order_id_action_source_duplicated_fails(self, trade_log_path):
        _write_rows(trade_log_path, [
            {'timestamp': _recent_ts(1), 'symbol': 'A', 'action': 'buy', 'order_id': 'o1', 'source': 'paper'},
            {'timestamp': _recent_ts(1), 'symbol': 'A', 'action': 'buy', 'order_id': 'o1', 'source': 'paper'},
        ])
        result = dq._check_duplicate_order_ids(trade_log_path, None)
        assert result['status'] == 'fail'

    def test_cross_source_same_order_id_not_a_duplicate(self, trade_log_path):
        """Different source counts as a different key — not a same-service double-log."""
        _write_rows(trade_log_path, [
            {'timestamp': _recent_ts(1), 'symbol': 'A', 'action': 'buy', 'order_id': 'o1', 'source': 'paper'},
            {'timestamp': _recent_ts(1), 'symbol': 'A', 'action': 'buy', 'order_id': 'o1', 'source': 'live'},
        ])
        result = dq._check_duplicate_order_ids(trade_log_path, None)
        assert result['status'] == 'pass'


class TestUnreconciledTrendCheck:
    def test_no_unreconciled_no_closed_passes(self, db_path):
        result = dq._check_unreconciled_trend(db_path, 'paper')
        assert result['status'] == 'pass'

    def test_high_ratio_fails(self, db_path):
        for i in range(5):
            _db.upsert_unreconciled_order({
                'recorded_at': _recent_ts(1), 'order_id': f'o{i}', 'symbol': 'A',
                'status': 'unknown', 'reason': 'missing_entry_fill',
            }, bot='stock', source='paper', db_path=db_path)
        _db.insert_outcome({
            'symbol': 'A', 'buy_order_id': 'b1', 'sell_order_id': 's1',
            'entry_timestamp': _recent_ts(2), 'exit_timestamp': _recent_ts(1),
            'entry_price': 10.0, 'exit_price': 11.0, 'shares': 1.0,
            'realized_pnl': 1.0, 'pnl_pct': 0.1, 'hold_hours': 24.0,
            'entry_confidence': 0.8,
        }, source='paper', db_path=db_path)
        result = dq._check_unreconciled_trend(db_path, 'paper')
        assert result['status'] == 'fail'

    def test_abandoned_rows_excluded_from_active_count(self, db_path):
        _db.upsert_unreconciled_order({
            'recorded_at': _recent_ts(1), 'order_id': 'o1', 'symbol': 'A',
            'status': 'unknown', 'reason': 'missing_entry_fill',
        }, bot='stock', source='paper', db_path=db_path)
        _db.mark_unreconciled_order_abandoned('o1', 'missing_entry_fill', bot='stock',
                                               source='paper', db_path=db_path)
        result = dq._check_unreconciled_trend(db_path, 'paper')
        assert result['metric'] == 0.0


class TestFallbackLabelRateCheck:
    def test_insufficient_tagged_rows_passes(self, db_path):
        result = dq._check_fallback_label_rate(db_path, 'paper')
        assert result['status'] == 'pass'
        assert 'insufficient' in result['detail']

    def _insert_examples(self, db_path, n_realized, n_fallback, source='paper'):
        for i in range(n_realized):
            _db.insert_training_example({
                'label': 'winner', 'input': f'p{i}', 'output': _WINNER_OUT,
                'metadata': {'bot': 'stock', 'source': source, 'symbol': f'S{i}',
                             'prompt_hash': f'r{i}', 'label_source': 'realized'},
            }, source=source, db_path=db_path)
        for i in range(n_fallback):
            _db.insert_training_example({
                'label': 'winner', 'input': f'q{i}', 'output': _WINNER_OUT,
                'metadata': {'bot': 'stock', 'source': source, 'symbol': f'T{i}',
                             'prompt_hash': f'f{i}', 'label_source': 'fallback_forward_price'},
            }, source=source, db_path=db_path)

    def test_low_fallback_rate_passes(self, db_path):
        self._insert_examples(db_path, n_realized=25, n_fallback=1)
        result = dq._check_fallback_label_rate(db_path, 'paper')
        assert result['status'] == 'pass'

    def test_high_fallback_rate_fails(self, db_path):
        self._insert_examples(db_path, n_realized=5, n_fallback=20)
        result = dq._check_fallback_label_rate(db_path, 'paper')
        assert result['status'] == 'fail'

    def test_moderate_fallback_rate_warns(self, db_path):
        self._insert_examples(db_path, n_realized=15, n_fallback=10)
        result = dq._check_fallback_label_rate(db_path, 'paper')
        assert result['status'] == 'warn'

    def test_null_label_source_rows_excluded(self, db_path):
        """Legacy untagged rows must not count toward the ratio at all."""
        for i in range(50):
            _db.insert_training_example({
                'label': 'winner', 'input': f'z{i}', 'output': _WINNER_OUT,
                'metadata': {'bot': 'stock', 'source': 'paper', 'symbol': f'Z{i}',
                             'prompt_hash': f'z{i}'},  # no label_source key
            }, source='paper', db_path=db_path)
        result = dq._check_fallback_label_rate(db_path, 'paper')
        assert 'insufficient' in result['detail']


class TestOutcomeSanityCheck:
    def test_sane_pnl_passes(self, db_path):
        _db.insert_outcome({
            'symbol': 'A', 'buy_order_id': 'b1', 'sell_order_id': 's1',
            'entry_timestamp': _recent_ts(2), 'exit_timestamp': _recent_ts(1),
            'entry_price': 10.0, 'exit_price': 11.0, 'shares': 1.0,
            'realized_pnl': 1.0, 'pnl_pct': 0.1, 'hold_hours': 24.0,
            'entry_confidence': 0.8,
        }, source='paper', db_path=db_path)
        result = dq._check_outcome_sanity(db_path, None)
        assert result['status'] in ('pass', 'warn')

    def test_insane_pnl_fails(self, db_path):
        _db.insert_outcome({
            'symbol': 'A', 'buy_order_id': 'b1', 'sell_order_id': 's1',
            'entry_timestamp': _recent_ts(2), 'exit_timestamp': _recent_ts(1),
            'entry_price': 10.0, 'exit_price': 11.0, 'shares': 1.0,
            'realized_pnl': 1.0, 'pnl_pct': 12.0, 'hold_hours': 24.0,
            'entry_confidence': 0.8,
        }, source='paper', db_path=db_path)
        result = dq._check_outcome_sanity(db_path, None)
        assert result['status'] == 'fail'


class TestHoldShareCheck:
    def test_no_exported_file_passes(self, tmp_path):
        with patch.object(dq, '_DATA_DIR', tmp_path):
            result = dq._check_hold_share('stock')
        assert result['status'] == 'pass'

    def test_high_hold_share_fails(self, tmp_path):
        rows = [_sft_row('correct_hold', i) for i in range(50)] + [_sft_row('winner', i) for i in range(3)]
        (tmp_path / 'training_data_sft.json').write_text(json.dumps(rows))
        with patch.object(dq, '_DATA_DIR', tmp_path):
            result = dq._check_hold_share('stock')
        assert result['status'] == 'fail'

    def test_low_hold_share_passes(self, tmp_path):
        rows = [_sft_row('correct_hold', i) for i in range(10)] + [_sft_row('winner', i) for i in range(30)]
        (tmp_path / 'training_data_sft.json').write_text(json.dumps(rows))
        with patch.object(dq, '_DATA_DIR', tmp_path):
            result = dq._check_hold_share('stock')
        assert result['status'] == 'pass'


class TestLabelDistributionCheck:
    def test_below_min_take_trade_fails(self, tmp_path):
        rows = [_sft_row('winner', i) for i in range(5)] + [_sft_row('correct_hold', i) for i in range(5)]
        (tmp_path / 'training_data_sft.json').write_text(json.dumps(rows))
        with patch.object(dq, '_DATA_DIR', tmp_path):
            result = dq._check_label_distribution('stock')
        assert result['status'] == 'fail'

    def test_sufficient_take_trade_passes(self, tmp_path):
        rows = [_sft_row('winner', i) for i in range(45)] + [_sft_row('correct_hold', i) for i in range(10)]
        (tmp_path / 'training_data_sft.json').write_text(json.dumps(rows))
        with patch.object(dq, '_DATA_DIR', tmp_path):
            result = dq._check_label_distribution('stock')
        assert result['status'] == 'pass'

    def test_degenerate_corpus_warns(self, tmp_path):
        """take_trade populated, hold tier at exactly 0 — degenerate."""
        rows = [_sft_row('winner', i) for i in range(45)]
        (tmp_path / 'training_data_sft.json').write_text(json.dumps(rows))
        with patch.object(dq, '_DATA_DIR', tmp_path):
            result = dq._check_label_distribution('stock')
        assert result['status'] == 'warn'


class TestContaminationCanaries:
    def test_clean_corpus_passes(self, db_path):
        _db.insert_training_example({
            'label': 'winner', 'input': 'p', 'output': _WINNER_OUT,
            'metadata': {'bot': 'stock', 'source': 'paper', 'symbol': 'A', 'prompt_hash': 'h1'},
        }, source='paper', db_path=db_path)
        result = dq._check_contamination_canaries(db_path)
        assert result['status'] == 'pass'

    def test_malformed_ideal_output_fails(self, db_path):
        _db.insert_training_example({
            'label': 'winner', 'input': 'p', 'output': 'not a clean 3-line block at all',
            'metadata': {'bot': 'stock', 'source': 'paper', 'symbol': 'A', 'prompt_hash': 'h1'},
        }, source='paper', db_path=db_path)
        result = dq._check_contamination_canaries(db_path)
        assert result['status'] == 'fail'

    def test_hindsight_leak_fails(self, db_path):
        _db.insert_outcome({
            'symbol': 'A', 'buy_order_id': 'b1', 'sell_order_id': 's1',
            'entry_timestamp': _recent_ts(1), 'exit_timestamp': _recent_ts(2),  # exit BEFORE entry
            'entry_price': 10.0, 'exit_price': 11.0, 'shares': 1.0,
            'realized_pnl': 1.0, 'pnl_pct': 0.1, 'hold_hours': 24.0,
            'entry_confidence': 0.8,
        }, source='paper', db_path=db_path)
        result = dq._check_contamination_canaries(db_path)
        assert result['status'] == 'fail'


class TestSyntheticRowsCheck:
    def test_no_synthetic_rows_passes(self, db_path, monkeypatch):
        monkeypatch.setenv('STRATEGY_EVOLVER_ENABLED', 'false')
        result = dq._check_synthetic_rows(db_path)
        assert result['status'] == 'pass'

    def test_synthetic_rows_present_while_disabled_warns(self, db_path, monkeypatch):
        monkeypatch.setenv('STRATEGY_EVOLVER_ENABLED', 'false')
        _db.insert_training_example({
            'label': 'winner', 'input': 'p', 'output': _WINNER_OUT,
            'metadata': {'bot': 'stock', 'source': 'synthetic_backtest', 'symbol': 'A', 'prompt_hash': 'h1'},
        }, source='synthetic_backtest', db_path=db_path)
        result = dq._check_synthetic_rows(db_path)
        assert result['status'] == 'warn'

    def test_synthetic_rows_present_while_enabled_passes(self, db_path, monkeypatch):
        monkeypatch.setenv('STRATEGY_EVOLVER_ENABLED', 'true')
        _db.insert_training_example({
            'label': 'winner', 'input': 'p', 'output': _WINNER_OUT,
            'metadata': {'bot': 'stock', 'source': 'synthetic_backtest', 'symbol': 'A', 'prompt_hash': 'h1'},
        }, source='synthetic_backtest', db_path=db_path)
        result = dq._check_synthetic_rows(db_path)
        assert result['status'] == 'pass'


class TestMacroCalendarFreshness:
    def _write_calendar(self, tmp_path, dates):
        path = tmp_path / 'macro_event_calendar.json'
        events = [{'date': d, 'time_et': '14:00', 'event': 'FOMC Statement'} for d in dates]
        path.write_text(json.dumps({'generated_at': 'test', 'source_urls': [], 'events': events}))
        return path

    def test_missing_file_fails(self, tmp_path):
        missing = tmp_path / 'does_not_exist.json'
        with patch.object(dq, '_MACRO_CALENDAR_PATH', missing):
            result = dq._check_macro_calendar_freshness()
        assert result['status'] == 'fail'

    def test_malformed_json_fails(self, tmp_path):
        path = tmp_path / 'bad.json'
        path.write_text('{not valid json')
        with patch.object(dq, '_MACRO_CALENDAR_PATH', path):
            result = dq._check_macro_calendar_freshness()
        assert result['status'] == 'fail'

    def test_empty_events_fails(self, tmp_path):
        path = self._write_calendar(tmp_path, [])
        with patch.object(dq, '_MACRO_CALENDAR_PATH', path):
            result = dq._check_macro_calendar_freshness()
        assert result['status'] == 'fail'

    def test_coverage_under_14_days_fails(self, tmp_path):
        soon = (datetime.now() + timedelta(days=5)).strftime('%Y-%m-%d')
        path = self._write_calendar(tmp_path, [soon])
        with patch.object(dq, '_MACRO_CALENDAR_PATH', path):
            result = dq._check_macro_calendar_freshness()
        assert result['status'] == 'fail'

    def test_coverage_under_60_days_warns(self, tmp_path):
        soon = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
        path = self._write_calendar(tmp_path, [soon])
        with patch.object(dq, '_MACRO_CALENDAR_PATH', path):
            result = dq._check_macro_calendar_freshness()
        assert result['status'] == 'warn'

    def test_coverage_over_60_days_passes(self, tmp_path):
        far = (datetime.now() + timedelta(days=200)).strftime('%Y-%m-%d')
        path = self._write_calendar(tmp_path, [far])
        with patch.object(dq, '_MACRO_CALENDAR_PATH', path):
            result = dq._check_macro_calendar_freshness()
        assert result['status'] == 'pass'


class TestRunDataQualityChecksEndToEnd:
    def test_clean_fixtures_pass_overall(self, db_path, trade_log_path, tmp_path):
        _write_rows(trade_log_path, [
            {'timestamp': _recent_ts(1), 'symbol': 'A', 'action': 'buy', 'order_id': 'o1', 'source': 'paper'},
        ])
        rows = [_sft_row('winner', i) for i in range(45)] + [_sft_row('correct_hold', i) for i in range(10)]
        (tmp_path / 'training_data_sft.json').write_text(json.dumps(rows))
        far = (datetime.now() + timedelta(days=200)).strftime('%Y-%m-%d')
        cal_path = tmp_path / 'macro_event_calendar.json'
        cal_path.write_text(json.dumps({'events': [{'date': far, 'time_et': '14:00', 'event': 'FOMC Statement'}]}))
        with patch.object(dq, '_DATA_DIR', tmp_path), \
             patch.object(dq, '_MACRO_CALENDAR_PATH', cal_path):
            result = dq.run_data_quality_checks(db_path=db_path, trade_log_path=trade_log_path, source='paper')
        assert result['passed'] is True
        assert 'PASSED' in result['summary']

    def test_failing_fixtures_do_not_pass_overall(self, db_path, trade_log_path, tmp_path, monkeypatch):
        monkeypatch.setenv('DQ_SOURCE_TAGGING_SINCE', (datetime.now() - timedelta(days=7)).isoformat())
        _write_rows(trade_log_path, [
            {'timestamp': _recent_ts(1), 'symbol': 'A', 'action': 'buy', 'order_id': 'o1'},  # no source
        ])
        with patch.object(dq, '_DATA_DIR', tmp_path):
            result = dq.run_data_quality_checks(db_path=db_path, trade_log_path=trade_log_path, source='paper')
        assert result['passed'] is False
        assert 'source_coverage' in result['summary']


class TestGateSpeed:
    def test_runs_under_30s_on_scaled_fixture(self, db_path, trade_log_path, tmp_path):
        rows = [
            {'timestamp': _recent_ts(1), 'symbol': f'S{i}', 'action': 'buy',
             'order_id': f'o{i}', 'source': 'paper'}
            for i in range(5000)
        ]
        _write_rows(trade_log_path, rows)
        for i in range(500):
            _db.upsert_unreconciled_order({
                'recorded_at': _recent_ts(1), 'order_id': f'u{i}', 'symbol': 'A',
                'status': 'unknown', 'reason': 'missing_entry_fill',
            }, bot='stock', source='paper', db_path=db_path)
        sft_rows = [_sft_row('winner', i) for i in range(450)] + [_sft_row('correct_hold', i) for i in range(100)]
        (tmp_path / 'training_data_sft.json').write_text(json.dumps(sft_rows))

        with patch.object(dq, '_DATA_DIR', tmp_path):
            start = time.perf_counter()
            dq.run_data_quality_checks(db_path=db_path, trade_log_path=trade_log_path, source='paper')
            elapsed = time.perf_counter() - start
        assert elapsed < 30.0


class TestThresholdEnvOverrides:
    def test_orphan_fail_pct_override(self, trade_log_path, monkeypatch):
        monkeypatch.setattr(dq, '_ORPHAN_FAIL_PCT', 0.01)
        rows = [{'timestamp': _recent_ts(1), 'symbol': f'S{i}', 'action': 'buy',
                 'order_id': f'o{i}', 'source': 'paper'} for i in range(9)]
        rows.append({'timestamp': _recent_ts(1), 'symbol': 'X', 'action': 'buy',
                     'order_id': 'ox', 'source': 'orphan'})
        _write_rows(trade_log_path, rows)
        result = dq._check_orphan_rate(trade_log_path, None)
        assert result['status'] == 'fail'

    def test_fallback_min_tagged_n_override(self, db_path, monkeypatch):
        monkeypatch.setattr(dq, '_FALLBACK_MIN_TAGGED_N', 2)
        _db.insert_training_example({
            'label': 'winner', 'input': 'p', 'output': _WINNER_OUT,
            'metadata': {'bot': 'stock', 'source': 'paper', 'symbol': 'A',
                         'prompt_hash': 'h1', 'label_source': 'realized'},
        }, source='paper', db_path=db_path)
        _db.insert_training_example({
            'label': 'winner', 'input': 'q', 'output': _WINNER_OUT,
            'metadata': {'bot': 'stock', 'source': 'paper', 'symbol': 'B',
                         'prompt_hash': 'h2', 'label_source': 'realized'},
        }, source='paper', db_path=db_path)
        result = dq._check_fallback_label_rate(db_path, 'paper')
        assert 'insufficient' not in result['detail']
