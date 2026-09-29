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
from datetime import datetime, timezone
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
class TestSleepWithHeartbeat:
    """Long idle sleeps must keep re-stamping the heartbeat.

    The daemon sleeps 30 minutes between sessions and many hours overnight, but
    health_server treats a heartbeat older than HEALTH_MAX_AGE_SECONDS (300s) as
    unhealthy. Sleeping straight through reported the daemon "down" for ~25 of
    every 30 minutes while it was perfectly healthy — observed live at 306.8s
    age moments after a good heartbeat.
    """

    def test_long_sleep_writes_multiple_heartbeats(self, module_name, tmp_path, monkeypatch):
        mod = importlib.import_module(module_name)
        hb = tmp_path / 'hb.json'
        monkeypatch.setattr(mod, '_HEARTBEAT_FILE', hb)
        monkeypatch.setattr(mod, '_HEARTBEAT_INTERVAL_SECONDS', 60)

        slept, writes = [], []
        monkeypatch.setattr(mod.time, 'sleep', lambda s: slept.append(s))
        real_write = mod._write_heartbeat
        monkeypatch.setattr(mod, '_write_heartbeat',
                            lambda status, mo: (writes.append((status, mo)), real_write(status, mo))[1])

        mod._sleep_with_heartbeat(1800, market_open=True)   # a 30-minute wait

        assert sum(slept) == pytest.approx(1800), 'must sleep the full duration'
        assert len(writes) == 30, f'expected a heartbeat per 60s chunk, got {len(writes)}'
        assert max(slept) <= 60, 'no single sleep may exceed the heartbeat interval'
        assert all(mo is True for _, mo in writes)

    def test_never_exceeds_the_health_staleness_threshold(self, module_name, tmp_path, monkeypatch):
        """The chunk size must stay under health_server's 300s default."""
        mod = importlib.import_module(module_name)
        monkeypatch.setattr(mod, '_HEARTBEAT_FILE', tmp_path / 'hb.json')
        slept = []
        monkeypatch.setattr(mod.time, 'sleep', lambda s: slept.append(s))

        mod._sleep_with_heartbeat(7 * 3600, market_open=False)   # overnight

        import health_server
        assert max(slept) < health_server.MAX_AGE_SECONDS, (
            'a sleep chunk longer than the staleness threshold would let a '
            'healthy daemon report as down')

    def test_short_sleep_still_stamps_once(self, module_name, tmp_path, monkeypatch):
        mod = importlib.import_module(module_name)
        hb = tmp_path / 'hb.json'
        monkeypatch.setattr(mod, '_HEARTBEAT_FILE', hb)
        monkeypatch.setattr(mod.time, 'sleep', lambda s: None)
        mod._sleep_with_heartbeat(5, market_open=False)
        assert hb.exists()

    def test_zero_and_negative_durations_do_not_hang(self, module_name, tmp_path, monkeypatch):
        mod = importlib.import_module(module_name)
        monkeypatch.setattr(mod, '_HEARTBEAT_FILE', tmp_path / 'hb.json')
        slept = []
        monkeypatch.setattr(mod.time, 'sleep', lambda s: slept.append(s))
        mod._sleep_with_heartbeat(0, market_open=False)
        mod._sleep_with_heartbeat(-10, market_open=False)
        assert slept == []

    def test_market_open_is_not_re_polled_per_chunk(self, module_name, tmp_path, monkeypatch):
        """is_market_open() is an Alpaca API call; polling it per chunk would add
        hundreds of needless calls against the same rate limit."""
        mod = importlib.import_module(module_name)
        monkeypatch.setattr(mod, '_HEARTBEAT_FILE', tmp_path / 'hb.json')
        monkeypatch.setattr(mod.time, 'sleep', lambda s: None)
        # Inspect the compiled code, not the source: the docstring legitimately
        # mentions is_market_open when explaining why it is NOT called.
        referenced = mod._sleep_with_heartbeat.__code__.co_names
        assert 'is_market_open' not in referenced, \
            '_sleep_with_heartbeat must take a captured bool, not re-poll the clock API'
        import inspect
        assert 'market_open' in inspect.signature(mod._sleep_with_heartbeat).parameters
