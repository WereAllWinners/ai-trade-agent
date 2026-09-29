"""
tests/test_service_suffix.py — sprint02 D4.1 paper/live file isolation helper.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))

from service_suffix import service_suffix, suffixed_path


class TestServiceSuffix:
    def test_paper_default_is_empty_suffix(self, monkeypatch):
        monkeypatch.delenv('PAPER_TRADING', raising=False)
        assert service_suffix() == ''

    def test_explicit_paper_true_is_empty_suffix(self, monkeypatch):
        monkeypatch.setenv('PAPER_TRADING', 'true')
        assert service_suffix() == ''

    def test_live_is_live_suffix(self, monkeypatch):
        monkeypatch.setenv('PAPER_TRADING', 'false')
        assert service_suffix() == '_live'

    def test_case_insensitive_false(self, monkeypatch):
        monkeypatch.setenv('PAPER_TRADING', 'FALSE')
        assert service_suffix() == '_live'


class TestSuffixedPath:
    def test_paper_path_unchanged(self, monkeypatch):
        monkeypatch.setenv('PAPER_TRADING', 'true')
        p = suffixed_path(Path('logs/reconcile_status.json'))
        assert p == Path('logs/reconcile_status.json')

    def test_live_path_gets_suffix_before_extension(self, monkeypatch):
        monkeypatch.setenv('PAPER_TRADING', 'false')
        p = suffixed_path(Path('logs/reconcile_status.json'))
        assert p == Path('logs/reconcile_status_live.json')

    def test_jsonl_extension_preserved(self, monkeypatch):
        monkeypatch.setenv('PAPER_TRADING', 'false')
        p = suffixed_path(Path('logs/rotation_log.jsonl'))
        assert p == Path('logs/rotation_log_live.jsonl')
