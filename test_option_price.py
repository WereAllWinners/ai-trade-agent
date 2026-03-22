#!/usr/bin/env python3
"""
Live test for get_option_price() against Alpaca paper API.
Run from the project root with the venv active:
  source venv/bin/activate
  python3 test_option_price.py
"""
import os, sys
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, 'scripts')

# Patch out the heavy ML imports in model_inference_lora so we can
# test options_agent without loading a 20GB model.
import types
dummy = types.ModuleType('model_inference_lora')
dummy.parse_decision = lambda r: {'decision': 'hold', 'confidence': 0.5, 'reasoning': r[:100]}
sys.modules['model_inference_lora'] = dummy

# Also stub out ollama
dummy_ollama = types.ModuleType('ollama')
dummy_ollama.generate = lambda **kw: {'response': 'HOLD. Confidence: 0.5.'}
sys.modules['ollama'] = dummy_ollama

from scripts.options_agent import OptionsAgent

print("Initializing OptionsAgent (paper=True)...")
agent = OptionsAgent()

# Active contracts confirmed via get_option_contracts (as of 2026-03-08)
# Format: UNDERLYING YYMMDD C/P STRIKE (8 digits, padded)
test_symbols = [
    'AAPL260316C00200000',  # AAPL Mar 16 2026 $200 Call
    'AAPL260316C00220000',  # AAPL Mar 16 2026 $220 Call
    'AAPL260316C00210000',  # AAPL Mar 16 2026 $210 Call
]

print()
for sym in test_symbols:
    price = agent.get_option_price(sym)
    if price is not None:
        print(f"  [PASS] {sym}: ${price:.2f}")
    else:
        print(f"  [WARN] {sym}: no price returned (may be expired/invalid or market closed)")

print()
print("If all show [WARN], confirm: (1) market is open, (2) symbols are active contracts.")
print("Done.")
