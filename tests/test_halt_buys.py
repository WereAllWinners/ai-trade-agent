"""
tests/test_halt_buys.py — R1 WI-7

Covers: flag_path paper/live split, fail-closed semantics, stale-flag-
still-blocks (no TTL), write_halt_flag content, manual-clear-only (no
code path removes the flag).
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))

from halt_buys import flag_path, is_halt_active, write_halt_flag  # noqa: E402


class TestFlagPath:
    def test_paper_path(self, tmp_path):
        base = tmp_path / 'halt_buys.flag'
        assert flag_path(True, base=base) == base

    def test_live_path_is_distinct(self, tmp_path):
        base = tmp_path / 'halt_buys.flag'
        live = flag_path(False, base=base)
        assert live != base
        assert live.name == 'halt_buys_live.flag'

    def test_default_base_paper_vs_live_distinct(self):
        assert flag_path(True) != flag_path(False)


class TestIsHaltActive:
    def test_no_flag_is_not_halted(self, tmp_path):
        assert is_halt_active(tmp_path / 'does_not_exist.flag') is False

    def test_flag_present_is_halted(self, tmp_path):
        p = tmp_path / 'halt.flag'
        p.write_text('{}')
        assert is_halt_active(p) is True

    def test_fail_closed_on_exception(self):
        """Any error determining state -> True (halt), not False."""
        bad_path = Path('/dev/null/impossible/path.flag')  # .exists() raises OSError on some platforms; patch to be sure
        with patch.object(Path, 'exists', side_effect=OSError('permission denied')):
            assert is_halt_active(bad_path) is True

    def test_stale_flag_still_blocks_no_ttl(self, tmp_path):
        """A flag from days ago (no mtime manipulation needed -- the
        function never reads mtime/content, only existence) still blocks."""
        p = tmp_path / 'old.flag'
        p.write_text('{"written_at": "2020-01-01T00:00:00"}')
        assert is_halt_active(p) is True

    def test_content_never_parsed_corrupt_body_still_blocks(self, tmp_path):
        p = tmp_path / 'corrupt.flag'
        p.write_text('not even valid json {{{')
        assert is_halt_active(p) is True  # presence alone is authoritative


class TestWriteHaltFlag:
    def test_writes_json_payload(self, tmp_path):
        p = tmp_path / 'halt.flag'
        write_halt_flag(p, reason='cash negative', data={'cash': -100.0})
        assert p.exists()
        import json
        payload = json.loads(p.read_text())
        assert payload['reason'] == 'cash negative'
        assert payload['data']['cash'] == -100.0
        assert 'written_at' in payload

    def test_creates_parent_dir(self, tmp_path):
        p = tmp_path / 'nested' / 'dir' / 'halt.flag'
        write_halt_flag(p, reason='test')
        assert p.exists()

    def test_no_clear_function_exists(self):
        """Manual-clear-only is enforced by omission: there is no
        clear_halt_flag() or similar in this module at all."""
        import halt_buys
        public_names = [n for n in dir(halt_buys) if not n.startswith('_')]
        assert not any('clear' in n.lower() or 'remove' in n.lower() or 'delete' in n.lower()
                       for n in public_names)
