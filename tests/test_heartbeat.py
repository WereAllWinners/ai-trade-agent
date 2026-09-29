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
