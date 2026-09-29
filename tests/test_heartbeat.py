"""Heartbeat write/read round-trip.

Regression guard for a six-month silent outage: trading_daemon._write_heartbeat
used `datetime.now(datetime.UTC)`, which raises AttributeError because the
modules import `datetime` the *class* (`from datetime import datetime`), and
`UTC` lives on the *module*. The write was wrapped in `except Exception: pass`,
so every heartbeat froze on 2026-03-22 while both daemons kept trading normally
and /health reported "down" with a 190-day-old timestamp.

These tests assert the write actually happens AND that health_server can parse
what was written — the two halves were never covered together, which is why the
break went unnoticed.
"""
import importlib
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'scripts'))


@pytest.mark.parametrize('module_name, daemon_name, hb_attr', [
    ('trading_daemon', 'stock',   '_HEARTBEAT_FILE'),
    ('options_daemon', 'options', '_HEARTBEAT_FILE'),
])
def test_write_heartbeat_emits_parseable_current_timestamp(
    module_name, daemon_name, hb_attr, tmp_path, monkeypatch
):
    mod = importlib.import_module(module_name)
    hb_path = tmp_path / f'heartbeat_{daemon_name}.json'
    monkeypatch.setattr(mod, hb_attr, hb_path)

    mod._write_heartbeat('running', True)

    # The file must exist: the old code raised before write_text and the bare
    # except hid it, leaving a stale file (or none) behind.
    assert hb_path.exists(), f'{module_name} wrote no heartbeat file'

    data = json.loads(hb_path.read_text())
    assert data['daemon'] == daemon_name
    assert data['status'] == 'running'
    assert data['market_open'] is True

    # Parse exactly the way health_server._read_heartbeat does.
    ts = datetime.fromisoformat(data['ts'].rstrip('Z')).replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - ts).total_seconds()
    assert 0 <= age < 60, f'heartbeat age {age}s is not current'


@pytest.mark.parametrize('module_name', ['trading_daemon', 'options_daemon'])
def test_write_heartbeat_logs_instead_of_swallowing_failures(module_name, tmp_path, caplog):
    """A failing write must warn, not pass silently."""
    import logging
    mod = importlib.import_module(module_name)

    bad = tmp_path / 'nope' / 'hb.json'
    # Make the parent a file so mkdir(parents=True) fails.
    (tmp_path / 'nope').write_text('not a directory')

    orig = getattr(mod, '_HEARTBEAT_FILE')
    setattr(mod, '_HEARTBEAT_FILE', bad)
    try:
        with caplog.at_level(logging.WARNING):
            mod._write_heartbeat('running', False)   # must not raise
        assert any('heartbeat write failed' in r.getMessage() for r in caplog.records), \
            'a failed heartbeat write must log a warning, not pass silently'
    finally:
        setattr(mod, '_HEARTBEAT_FILE', orig)


@pytest.mark.parametrize('module_name', ['trading_daemon', 'options_daemon'])
def test_health_server_reports_fresh_heartbeat_as_healthy(module_name, tmp_path, monkeypatch):
    """End-to-end: a just-written heartbeat must read back as healthy."""
    import health_server

    mod = importlib.import_module(module_name)
    daemon = 'stock' if module_name == 'trading_daemon' else 'options'
    hb_path = tmp_path / f'heartbeat_{daemon}.json'
    monkeypatch.setattr(mod, '_HEARTBEAT_FILE', hb_path)
    mod._write_heartbeat('running', False)

    result = health_server._read_heartbeat(daemon, hb_path)
    assert result['healthy'] is True, f'fresh heartbeat reported unhealthy: {result}'
    assert result['status'] == 'running'
    assert result['age_seconds'] < 60


@pytest.mark.parametrize('module_name', ['trading_daemon', 'options_daemon'])
class TestHeartbeatThread:
    """Liveness must not depend on what the main loop is doing.

    Chunking the idle sleeps was tried first and was not enough: the daemon
    still reported down at 443s while healthily analysing CCL, because a
    trading session occupies the loop for many minutes (one symbol's LLM
    inference takes ~2 minutes and a session walks dozens of symbols). A
    background thread stamps the heartbeat regardless of whether the loop is
    sleeping, working, or blocked on a subprocess.
    """

    def test_thread_stamps_heartbeat_on_its_interval(self, module_name, tmp_path, monkeypatch):
        mod = importlib.import_module(module_name)
        hb = tmp_path / 'hb.json'
        monkeypatch.setattr(mod, '_HEARTBEAT_FILE', hb)
        monkeypatch.setattr(mod, '_HEARTBEAT_INTERVAL_SECONDS', 0.05)
        monkeypatch.setattr(mod, '_heartbeat_stop', mod.threading.Event())

        t = mod.start_heartbeat_thread()
        try:
            deadline = time.time() + 3
            while not hb.exists() and time.time() < deadline:
                time.sleep(0.02)
            assert hb.exists(), 'thread should have written a heartbeat'
            first = json.loads(hb.read_text())['ts']
            deadline = time.time() + 3
            while time.time() < deadline:
                if json.loads(hb.read_text())['ts'] != first:
                    break
                time.sleep(0.02)
            assert json.loads(hb.read_text())['ts'] != first, \
                'heartbeat must be re-stamped on each interval'
        finally:
            mod._heartbeat_stop.set()
            t.join(timeout=2)

    def test_thread_keeps_stamping_while_the_main_loop_is_busy(self, module_name, tmp_path, monkeypatch):
        """The regression that matters: the loop is working, not sleeping."""
        mod = importlib.import_module(module_name)
        hb = tmp_path / 'hb.json'
        monkeypatch.setattr(mod, '_HEARTBEAT_FILE', hb)
        monkeypatch.setattr(mod, '_HEARTBEAT_INTERVAL_SECONDS', 0.05)
        monkeypatch.setattr(mod, '_heartbeat_stop', mod.threading.Event())

        t = mod.start_heartbeat_thread()
        try:
            busy_until = time.time() + 0.6          # simulate a long session
            while time.time() < busy_until:
                pass
            assert hb.exists(), 'heartbeat must appear even though the loop never slept'
            age = time.time() - hb.stat().st_mtime
            assert age < 0.5, f'heartbeat went stale during busy work (age {age:.2f}s)'
        finally:
            mod._heartbeat_stop.set()
            t.join(timeout=2)

    def test_thread_is_a_daemon_so_it_never_blocks_exit(self, module_name, tmp_path, monkeypatch):
        mod = importlib.import_module(module_name)
        monkeypatch.setattr(mod, '_HEARTBEAT_FILE', tmp_path / 'hb.json')
        monkeypatch.setattr(mod, '_HEARTBEAT_INTERVAL_SECONDS', 0.05)
        monkeypatch.setattr(mod, '_heartbeat_stop', mod.threading.Event())
        t = mod.start_heartbeat_thread()
        try:
            assert t.daemon is True
        finally:
            mod._heartbeat_stop.set()
            t.join(timeout=2)

    def test_thread_does_not_poll_the_clock_api(self, module_name):
        """is_market_open() is an Alpaca call; the thread reuses a cached value."""
        mod = importlib.import_module(module_name)
        assert 'is_market_open' not in mod._heartbeat_loop.__code__.co_names

    def test_market_state_setter_feeds_the_thread(self, module_name, tmp_path, monkeypatch):
        mod = importlib.import_module(module_name)
        hb = tmp_path / 'hb.json'
        monkeypatch.setattr(mod, '_HEARTBEAT_FILE', hb)
        monkeypatch.setattr(mod, '_heartbeat_market_open', False)
        mod._set_heartbeat_market_open(True)
        mod._write_heartbeat('running', mod._heartbeat_market_open)
        assert json.loads(hb.read_text())['market_open'] is True
        mod._set_heartbeat_market_open(False)
        mod._write_heartbeat('running', mod._heartbeat_market_open)
        assert json.loads(hb.read_text())['market_open'] is False


@pytest.mark.parametrize('module_name, base', [
    ('trading_daemon', 'heartbeat_stock.json'),
    ('options_daemon', 'heartbeat_options.json'),
])
class TestPaperLiveHeartbeatSeparation:
    """Paper and live share a WorkingDirectory, so an unsuffixed heartbeat path
    meant both wrote the same file last-writer-wins — a healthy paper bot could
    mask a dead live one, and health_server had no key to tell them apart."""

    def _reload(self, module_name, monkeypatch, paper: bool):
        monkeypatch.setenv('PAPER_TRADING', 'true' if paper else 'false')
        mod = importlib.import_module(module_name)
        return importlib.reload(mod)

    def test_paper_keeps_the_unsuffixed_name(self, module_name, base, monkeypatch):
        mod = self._reload(module_name, monkeypatch, paper=True)
        assert mod._HEARTBEAT_FILE.name == base

    def test_live_gets_its_own_file(self, module_name, base, monkeypatch):
        mod = self._reload(module_name, monkeypatch, paper=False)
        expected = base.replace('.json', '_live.json')
        assert mod._HEARTBEAT_FILE.name == expected

    def test_paper_and_live_never_collide(self, module_name, base, monkeypatch):
        paper = self._reload(module_name, monkeypatch, paper=True)._HEARTBEAT_FILE
        live = self._reload(module_name, monkeypatch, paper=False)._HEARTBEAT_FILE
        assert paper != live
        # leave the module in its default (paper) state for other tests
        self._reload(module_name, monkeypatch, paper=True)


class TestHealthServerCoversBothAccounts:
    def test_all_four_daemons_are_tracked(self):
        import health_server
        assert set(health_server.HEARTBEAT_FILES) == {
            'stock', 'options', 'stock_live', 'options_live'}

    def test_live_absence_does_not_degrade_the_verdict(self, tmp_path, monkeypatch):
        """Live is held off; a service that never ran must not read as a failure."""
        import health_server
        monkeypatch.setattr(health_server, '_LOGS_DIR', tmp_path)
        monkeypatch.setattr(health_server, 'HEARTBEAT_FILES', {
            'stock':      tmp_path / 'heartbeat_stock.json',
            'options':    tmp_path / 'heartbeat_options.json',
            'stock_live': tmp_path / 'heartbeat_stock_live.json',
        })
        now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        for n in ('heartbeat_stock.json', 'heartbeat_options.json'):
            (tmp_path / n).write_text(json.dumps(
                {'daemon': n, 'status': 'running', 'market_open': True, 'ts': now}))

        payload, code = health_server.build_health_payload()
        assert payload['status'] == 'ok', payload['daemons']
        assert code == 200
        assert payload['daemons']['stock_live']['status'] == 'missing', \
            'the live daemon should still be reported, just not judged'

    def test_a_live_daemon_that_ran_and_went_stale_is_caught(self, tmp_path, monkeypatch):
        import health_server
        monkeypatch.setattr(health_server, '_LOGS_DIR', tmp_path)
        monkeypatch.setattr(health_server, 'HEARTBEAT_FILES', {
            'stock':      tmp_path / 'heartbeat_stock.json',
            'stock_live': tmp_path / 'heartbeat_stock_live.json',
        })
        now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        stale = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat().replace('+00:00', 'Z')
        (tmp_path / 'heartbeat_stock.json').write_text(json.dumps(
            {'daemon': 'stock', 'status': 'running', 'market_open': True, 'ts': now}))
        (tmp_path / 'heartbeat_stock_live.json').write_text(json.dumps(
            {'daemon': 'stock_live', 'status': 'running', 'market_open': True, 'ts': stale}))

        payload, _ = health_server.build_health_payload()
        assert payload['status'] == 'degraded', \
            'a live daemon that has run before and gone stale must be flagged'
