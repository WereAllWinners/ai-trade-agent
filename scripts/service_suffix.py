#!/usr/bin/env python3
"""
service_suffix.py — shared/not-shared state file isolation between the paper
and live services (sprint02 D4.1).

All four systemd services (paper/live stock, paper/live options) share the same
WorkingDirectory, so any code path that writes a relative `logs/<name>` path
was silently shared across services with no discriminator — last-writer-wins.
This derives a filename suffix from the exact same PAPER_TRADING check already
used identically at autonomous_agent.py, trading_daemon.py, and options_agent.py,
so paper and live never collide on the same state file again.

Paper keeps the unsuffixed filename (no external-tooling/dashboard breakage);
live gets a `_live` suffixed filename.
"""
import os


def service_suffix() -> str:
    """Return '' for paper, '_live' for live — derived from PAPER_TRADING."""
    paper = os.getenv('PAPER_TRADING', 'true').lower() != 'false'
    return '' if paper else '_live'


def suffixed_path(base_path):
    """Insert the service suffix before a pathlib.Path's file extension.

    suffixed_path(Path('logs/reconcile_status.json')) -> Path('logs/reconcile_status.json')  (paper)
                                                        -> Path('logs/reconcile_status_live.json')  (live)
    """
    suffix = service_suffix()
    if not suffix:
        return base_path
    return base_path.with_name(base_path.stem + suffix + base_path.suffix)
