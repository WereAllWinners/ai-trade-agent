"""
pytest conftest — stubs out heavy optional dependencies that are not installed
in the lightweight test environment (no GPU, no large ML packages required).

These stubs are applied once, before any test module is imported, so
autonomous_agent.py / options_agent.py / model_inference_lora.py can all be
imported without error.
"""
import sys
from unittest.mock import MagicMock

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
