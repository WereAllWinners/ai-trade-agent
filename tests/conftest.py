"""
pytest conftest — stubs out heavy optional dependencies that are not installed
in the lightweight test environment (no GPU, no large ML packages required).

These stubs are applied once, before any test module is imported, so
autonomous_agent.py / options_agent.py / model_inference_lora.py can all be
imported without error.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Heavy deps that may not be present in the test environment
_STUBS = [
    'ollama',
    'yfinance',
    'unsloth',
    'bitsandbytes',
    'torch',
    'transformers',
    'peft',
    'trl',
    'accelerate',
    'datasets',
]

for _pkg in _STUBS:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = MagicMock()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import halt_buys  # noqa: E402
import alerts  # noqa: E402


@pytest.fixture(autouse=True)
def _halt_buys_default_clear(monkeypatch):
    """R1 WI-7: autonomous_agent.py/options_agent.py check
    halt_buys.is_halt_active() before every BUY. Default every test to a
    clear flag so pre-existing tests (written before halt_buys existed, and
    not mocking it) never depend on real filesystem state — e.g. a real
    logs/halt_buys.flag written by the live position-watchdog timer on this
    dev machine, which otherwise makes BUY-path tests fail nondeterministically
    depending on production account state. Tests that specifically exercise
    the halt-active path override this locally (patch to return True)."""
    monkeypatch.setattr(halt_buys, 'is_halt_active', lambda path: False)


@pytest.fixture(autouse=True)
def _no_real_alerts(monkeypatch):
    """Keep the suite from emitting real alerts.

    send_alert appends to the production logs/alerts.jsonl AND attempts an SMTP
    send when SMTP_HOST is configured. Unit tests were doing both: a single
    12-minute window of test runs wrote 15 real alert records — including
    `trade_executed` and `circuit_breaker` — into the live alert log, where an
    operator would read them as genuine trading events.

    It also made the suite unusable during market hours. Outside trading hours
    most sessions short-circuit before alerting; with the market open the alert
    paths fire, and every blocked SMTP connect costs its full 10s timeout. One
    options-session test took 66s for that reason alone, and the full suite blew
    past a 900s budget it normally finishes in ~280s.

    Stubs the three side-effecting primitives rather than send_alert itself, so
    that alert *construction* (source tagging, dedup keys, record shape) is still
    exercised and tests which patch these individually — as tests/test_alerts.py
    already does — simply override this default.
    """
    monkeypatch.setattr(alerts, '_write_to_log', lambda record: None)
    monkeypatch.setattr(alerts, '_send_email',
                        lambda level, event, message, record: None)
    # _send_telegram is deliberately NOT stubbed: it returns early unless the
    # TELEGRAM_* vars are set, so it costs nothing, and the telegram tests drive
    # the real function while patching requests.post underneath it.
