"""
Unit tests for decision_parser.parse_decision (sprint01 C1.1 consolidation).

decision_parser.py is now the single canonical implementation of parse_decision;
model_inference_lora.py re-exports it. This file gives decision_parser.py its
own direct test coverage — see tests/test_parse_decision.py for coverage via
the model_inference_lora import path (used by the actual agents).

Run with:
  python3 -m pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))

import decision_parser
from decision_parser import parse_decision


class TestDecisionParserBasics:
    def test_buy_decision(self):
        result = parse_decision("I recommend BUY with confidence: 0.82. Strong momentum.")
        assert result['decision'] == 'buy'

    def test_sell_decision(self):
        result = parse_decision("SELL this position. Confidence: 0.75. RSI overbought.")
        assert result['decision'] == 'sell'

    def test_hold_decision(self):
        result = parse_decision("HOLD for now. Confidence: 0.55. Mixed signals.")
        assert result['decision'] == 'hold'

    def test_no_signal_defaults_to_hold(self):
        result = parse_decision("Market conditions unclear. No clear direction.")
        assert result['decision'] == 'hold'

    def test_dont_buy_is_not_buy(self):
        result = parse_decision("Don't buy here. Confidence: 0.70.")
        assert result['decision'] != 'buy'

    def test_not_buy_is_not_buy(self):
        result = parse_decision("I would not buy this stock. Confidence: 0.65.")
        assert result['decision'] != 'buy'

    def test_dont_sell_is_not_sell(self):
        result = parse_decision("Don't sell yet. Confidence: 0.60.")
        assert result['decision'] != 'sell'


class TestDecisionParserAmbiguity:
    def test_both_buy_and_sell_present_is_ambiguous_hold(self):
        result = parse_decision(
            "The analyst said buy signals are weak but some might sell. Confidence: 0.60."
        )
        assert result['decision'] == 'hold'
        assert result['parse_failed'] is True

    def test_word_boundary_excludes_buyers_false_match(self):
        result = parse_decision("Buyers are exhausted, no clear signal. Confidence: 0.5.")
        assert result['decision'] != 'buy'

    def test_fallback_never_returns_buy_call_or_buy_put(self):
        """The free-text fallback must never produce buy_call/buy_put — those
        only come from the JSON block or a structured Decision: line."""
        result = parse_decision("I like this call option and that put too, but no clear buy/sell.")
        assert result['decision'] not in ('buy_call', 'buy_put')


class TestDecisionParserStructuredLine:
    def test_buy_call_line(self):
        result = parse_decision("Decision: BUY_CALL\nConfidence: 0.80\nReasoning: r")
        assert result['decision'] == 'buy_call'
        assert result['parse_method'] == 'structured_line'

    def test_buy_put_line(self):
        result = parse_decision("Decision: BUY_PUT\nConfidence: 0.70\nReasoning: r")
        assert result['decision'] == 'buy_put'

    def test_decision_line_wins_over_conflicting_reasoning(self):
        result = parse_decision(
            "Decision: HOLD\nConfidence: 0.55\nReasoning: technicals call for caution"
        )
        assert result['decision'] == 'hold'

    def test_put_simply_reasoning_does_not_flip_hold(self):
        result = parse_decision("Decision: HOLD\nReasoning: put simply, wait")
        assert result['decision'] == 'hold'


class TestDecisionParserJson:
    def test_json_buy(self):
        response = '{"decision": "buy", "confidence": 0.88, "reasoning": "Strong breakout"}'
        result = parse_decision(response)
        assert result['decision'] == 'buy'
        assert result['parse_method'] == 'json'

    def test_json_buy_put_accepted(self):
        response = '{"decision": "buy_put", "confidence": 0.72, "reasoning": "bearish"}'
        result = parse_decision(response)
        assert result['decision'] == 'buy_put'
        assert result['parse_method'] == 'json'

    def test_json_confidence_normalised_from_100_scale(self):
        response = '{"decision": "buy", "confidence": 82, "reasoning": "Momentum"}'
        result = parse_decision(response)
        assert abs(result['confidence'] - 0.82) < 0.01


class TestDecisionParserConfidenceContract:
    def test_missing_confidence_is_zero_not_stale_half(self):
        """Sprint01 A2: pin the contract at 0.0, not the older 0.5 default."""
        result = parse_decision("Decision: HOLD\nReasoning: no clear edge")
        assert result['confidence'] == 0.0

    def test_confidence_always_float(self):
        for response in (
            "The market conditions are complex. There is uncertainty.",
            "BUY based on technical analysis.",
            "BUY. Confidence: 0.80.",
        ):
            result = parse_decision(response)
            assert isinstance(result['confidence'], float)

    def test_parse_failed_true_only_on_total_failure(self):
        result = parse_decision("The market conditions are complex. There is uncertainty.")
        assert result['parse_failed'] is True
        assert result['confidence'] == 0.0


class TestDecisionParserLightweightImport:
    def test_failure_counter_lives_on_decision_parser(self):
        before = decision_parser._parse_failures_count
        parse_decision("The market conditions are complex. There is uncertainty.")
        after = decision_parser._parse_failures_count
        assert after > before
