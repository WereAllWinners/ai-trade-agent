"""
tests/test_apply_r1_log_corrections.py — R1 WI-5/WI-8 shared correction tool

Covers: idempotency (re-run doesn't duplicate) and correct append behavior
for each of the four correction items, against tmp_path files only.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'tools'))

import apply_r1_log_corrections as corrections  # noqa: E402


def _patched_paths(tmp_path):
    return patch.multiple(
        corrections,
        _OPTIONS_TRADE_LOG=tmp_path / 'options_trade_log.jsonl',
        _ALERTS_LOG=tmp_path / 'alerts.jsonl',
        _ADDENDUM_FILE=tmp_path / 'addendum.md',
    )


class TestMsftBackfill:
    def test_apply_writes_both_files(self, tmp_path):
        with _patched_paths(tmp_path):
            assert corrections._msft_backfill_already_applied() is False
            corrections._msft_backfill_apply()
            assert corrections._msft_backfill_already_applied() is True

        trade_log = tmp_path / 'options_trade_log.jsonl'
        alerts_log = tmp_path / 'alerts.jsonl'
        trade_rec = json.loads(trade_log.read_text().splitlines()[0])
        alert_rec = json.loads(alerts_log.read_text().splitlines()[0])
        assert trade_rec['order_id'] == corrections._MSFT_ORDER_ID
        assert trade_rec['action'] == 'sell'
        assert trade_rec['exit_pl_pct'] == -0.3827  # corrected sign, not "profitable"
        assert alert_rec['level'] == 'AlertLevel.CRITICAL'

    def test_rerun_does_not_duplicate(self, tmp_path):
        with _patched_paths(tmp_path):
            corrections._msft_backfill_apply()
            corrections._msft_backfill_apply()
        trade_log = tmp_path / 'options_trade_log.jsonl'
        lines = [l for l in trade_log.read_text().splitlines() if l.strip()]
        assert len(lines) == 1


class TestCorrections:
    def test_nvda_correction_has_no_action_key(self, tmp_path):
        with _patched_paths(tmp_path):
            corrections._nvda_apply()
        trade_log = tmp_path / 'options_trade_log.jsonl'
        rec = json.loads(trade_log.read_text().splitlines()[0])
        assert rec['type'] == 'correction'
        assert 'action' not in rec
        assert rec['order_id'] == corrections._NVDA_ORDER_ID

    def test_aapl_correction_has_no_action_key(self, tmp_path):
        with _patched_paths(tmp_path):
            corrections._aapl_apply()
        trade_log = tmp_path / 'options_trade_log.jsonl'
        rec = json.loads(trade_log.read_text().splitlines()[0])
        assert rec['type'] == 'correction'
        assert 'action' not in rec
        assert rec['order_id'] == corrections._AAPL_ORDER_ID

    def test_nvda_and_aapl_corrections_independent(self, tmp_path):
        with _patched_paths(tmp_path):
            corrections._nvda_apply()
            corrections._aapl_apply()
        trade_log = tmp_path / 'options_trade_log.jsonl'
        recs = [json.loads(l) for l in trade_log.read_text().splitlines() if l.strip()]
        assert len(recs) == 2
        assert {r['order_id'] for r in recs} == {corrections._NVDA_ORDER_ID, corrections._AAPL_ORDER_ID}

    def test_correction_rerun_does_not_duplicate(self, tmp_path):
        with _patched_paths(tmp_path):
            corrections._nvda_apply()
            corrections._nvda_apply()
        trade_log = tmp_path / 'options_trade_log.jsonl'
        recs = [json.loads(l) for l in trade_log.read_text().splitlines() if l.strip()]
        assert len(recs) == 1


class TestAddendumErrata:
    def test_apply_appends_marker(self, tmp_path):
        addendum = tmp_path / 'addendum.md'
        addendum.write_text("# Original addendum\n\nOriginal content.\n")
        with _patched_paths(tmp_path):
            assert corrections._errata_already_applied() is False
            corrections._errata_apply()
            assert corrections._errata_already_applied() is True
        content = addendum.read_text()
        assert "Original content." in content  # original text untouched
        assert corrections._ERRATA_MARKER in content
        assert "-38.3%" in content

    def test_missing_file_errors_instead_of_creating(self, tmp_path):
        with _patched_paths(tmp_path):
            try:
                corrections._errata_apply()
                assert False, "expected SystemExit"
            except SystemExit as e:
                assert e.code == 1
        assert not (tmp_path / 'addendum.md').exists()

    def test_rerun_does_not_duplicate_marker(self, tmp_path):
        addendum = tmp_path / 'addendum.md'
        addendum.write_text("# Original\n")
        with _patched_paths(tmp_path):
            corrections._errata_apply()
            corrections._errata_apply()
        content = addendum.read_text()
        assert content.count(corrections._ERRATA_MARKER) == 1


class TestMainCli:
    def test_preview_mode_writes_nothing(self, tmp_path, capsys):
        with _patched_paths(tmp_path), \
             patch('sys.argv', ['apply_r1_log_corrections.py', '--preview', '--item', 'msft_backfill']):
            corrections.main()
        assert not (tmp_path / 'options_trade_log.jsonl').exists()

    def test_execute_single_item_only_applies_that_one(self, tmp_path):
        with _patched_paths(tmp_path), \
             patch('sys.argv', ['apply_r1_log_corrections.py', '--execute', '--item', 'nvda_correction']):
            corrections.main()
        trade_log = tmp_path / 'options_trade_log.jsonl'
        recs = [json.loads(l) for l in trade_log.read_text().splitlines() if l.strip()]
        assert len(recs) == 1
        assert recs[0]['order_id'] == corrections._NVDA_ORDER_ID
