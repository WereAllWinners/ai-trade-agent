#!/usr/bin/env python3
"""
training_constants.py — sprint03 E4: single source of truth for the SFT
training-corpus thresholds shared across training_data_builder.py,
finetune_model.py, and data_quality_check.py.

Previously these were split across the two orchestration scripts, with one
importing the other's constants (finetune_model.py importing from
training_data_builder.py) — workable for two files, but awkward for a third
consumer that only needs the constants and would otherwise have to import a
whole orchestration script just to read a threshold.
"""
import os

# SFT HOLD-teaching share tripwire.
# (counterfactual + correct_hold + constraint_block) / total_sft > threshold → WARNING.
# The threshold is intentionally below 70% so the Session-4 state (94%) would have
# failed immediately and Session 3 would have run before any fine-tune.
_SFT_MAX_HOLD_SHARE: float = float(os.getenv('TDB_SFT_MAX_HOLD_SHARE', '0.65'))

# All labels that teach "decline / hold" rather than "take the trade".
# Used by the tripwire, the cadence gate, and the data-quality gate.
_HOLD_TEACHING_LABELS: frozenset[str] = frozenset({
    'counterfactual', 'correct_hold', 'constraint_block',
})

# Minimum take-trade exemplar count and total SFT size the cadence gate
# requires before a fine-tune is allowed to proceed.
_MIN_TAKE_TRADE = int(os.getenv('FINETUNE_MIN_TAKE_TRADE', '40'))
_MIN_TOTAL_SFT  = int(os.getenv('FINETUNE_MIN_TOTAL_SFT',  '200'))

# Labels that count as a genuine take-trade exemplar for the above minimum.
_TAKE_TRADE_LABELS: frozenset[str] = frozenset({'winner', 'strong_winner'})
