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
