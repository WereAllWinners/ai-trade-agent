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
from decision_parser import parse_decision, unwrap_reasoning


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


class TestUnwrapReasoning:
    """Direct coverage for unwrap_reasoning — the fix for the contamination loop.

    Models trained on contaminated examples regurgitate whole Decision/Confidence/
    Reasoning blocks inside the Reasoning field, often with the trade's outcome
    appended. The old `reasoning\\s*[:\\-]\\s*(.+)` capture was greedy to end of
    line, so it stored the entire nested mess into decisions.reasoning; the
    builder then wrapped that into a new training example, which taught the model
    to emit it again. 78.8% of stored decisions were affected.
    """

    def test_plain_reasoning_is_untouched(self):
        assert unwrap_reasoning('Strong momentum confirmed.') == 'Strong momentum confirmed.'

    def test_single_nested_block_is_unwrapped(self):
        got = unwrap_reasoning('Decision: BUY Confidence: 0.85 Reasoning: Strong momentum.')
        assert got == 'Strong momentum.'

    def test_double_nested_block_is_unwrapped(self):
        got = unwrap_reasoning(
            'Decision: BUY Confidence: 0.80 Reasoning: Decision: BUY Confidence: 0.80 '
            'Reasoning: Real analysis here.')
        assert got == 'Real analysis here.'

    def test_triple_nesting_is_unwrapped(self):
        got = unwrap_reasoning('Reasoning: ' * 3 + 'The actual text.')
        assert got == 'The actual text.'

    def test_outcome_trailer_is_stripped(self):
        """Future-outcome leakage: the model cannot know this at decision time."""
        got = unwrap_reasoning(
            'Strong momentum. Outcome: Small win (+4.2%) — room for improvement.')
        assert got == 'Strong momentum.'
        assert 'Outcome' not in got

    def test_reward_signal_trailer_is_stripped(self):
        got = unwrap_reasoning('Good setup. Reward signal: +0.0003')
        assert got == 'Good setup.'

    def test_result_trailer_is_stripped(self):
        assert unwrap_reasoning('Momentum play. Result: loss') == 'Momentum play.'

    def test_nesting_and_trailer_together(self):
        """The real production shape."""
        got = unwrap_reasoning(
            'Decision: BUY Confidence: 0.85 Reasoning: Strong oversold condition. '
            'Outcome: Minor loss (-2.2%). Reward signal: -0.0007')
        assert got == 'Strong oversold condition.'
        for marker in ('Decision:', 'Outcome:', 'Reward signal'):
            assert marker not in got

    def test_markdown_emphasis_is_stripped(self):
        """143 stored rows use **Decision:** style formatting."""
        got = unwrap_reasoning('**Decision: BUY** **Confidence: 0.75** **Reasoning:** RSI is oversold')
        assert got == 'RSI is oversold'

    def test_returns_empty_when_nothing_survives(self):
        """Must not invent text — this feeds the decisions.reasoning audit column."""
        assert unwrap_reasoning('Outcome: Small win (+3.2%). Reward signal: +0.0006') == ''
        assert unwrap_reasoning('') == ''
        assert unwrap_reasoning(None) == ''

    def test_internal_asterisks_are_preserved(self):
        got = unwrap_reasoning('MACD crossed the 2*sigma band')
        assert got == 'MACD crossed the 2*sigma band'

    def test_trailing_punctuation_is_preserved(self):
        """Stripping it produced ~11,800 cosmetic-only diffs across the corpus."""
        assert unwrap_reasoning('Momentum is strong.').endswith('.')


class TestExtractReasoningIntegration:
    """parse_decision must not emit contaminated reasoning for real response shapes."""

    def test_nested_response_yields_clean_reasoning(self):
        response = (' Decision: SELL\nConfidence: 0.90\n'
                    'Reasoning: Decision: SELL Confidence: 0.90 Reasoning: Oversold with '
                    'negative momentum.\nOutcome: Small win (-0.6%). Reward signal: -0.0004')
        result = parse_decision(response)
        assert result['decision'] == 'sell'
        assert result['reasoning'] == 'Oversold with negative momentum.'
        for marker in ('Decision:', 'Outcome:', 'Reward signal'):
            assert marker not in result['reasoning']

    def test_reasoning_never_contains_newlines(self):
        """_is_clean_ideal_output requires the assembled block to be exactly 3 lines."""
        response = 'Decision: BUY\nConfidence: 0.8\nReasoning: line one\nline two'
        assert '\n' not in parse_decision(response)['reasoning']

    def test_two_hundred_char_cap_is_preserved(self):
        response = 'Decision: BUY\nConfidence: 0.8\nReasoning: ' + 'x' * 500
        assert len(parse_decision(response)['reasoning']) <= 200

    def test_clean_response_is_unaffected(self):
        response = 'Decision: BUY\nConfidence: 0.85\nReasoning: Strong breakout on volume.'
        assert parse_decision(response)['reasoning'] == 'Strong breakout on volume.'
