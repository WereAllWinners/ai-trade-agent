"""
Unit tests for model_inference_lora.parse_decision (PR-7 E1)

Run with:
  python3 -m pytest tests/ -v
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))

# Remove any test stub registered by other test modules before importing the real module
sys.modules.pop('model_inference_lora', None)
import model_inference_lora as _mli
from model_inference_lora import parse_decision


class TestParseDecision:
    # ------------------------------------------------------------------
    # Basic action parsing
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Negation handling
    # ------------------------------------------------------------------

    def test_dont_buy_is_hold(self):
        result = parse_decision("Don't buy here. Confidence: 0.70.")
        assert result['decision'] != 'buy'

    def test_not_buy_is_hold(self):
        result = parse_decision("I would not buy this stock. Confidence: 0.65.")
        assert result['decision'] != 'buy'

    def test_dont_sell_is_not_sell(self):
        result = parse_decision("Don't sell yet. Confidence: 0.60.")
        assert result['decision'] != 'sell'

    # ------------------------------------------------------------------
    # Confidence parsing
    # ------------------------------------------------------------------

    def test_confidence_decimal(self):
        result = parse_decision("BUY. Confidence: 0.85.")
        assert abs(result['confidence'] - 0.85) < 0.01

    def test_confidence_percentage(self):
        # Model sometimes outputs confidence as a percentage
        result = parse_decision("BUY. Confidence: 85.")
        assert abs(result['confidence'] - 0.85) < 0.01

    def test_confidence_clamped_to_1(self):
        result = parse_decision("BUY. Confidence: 1.5.")
        assert result['confidence'] <= 1.0

    def test_confidence_clamped_to_0(self):
        # Regex matches only non-negative numbers; "-0.2" is unparseable → coerced to 0.0.
        result = parse_decision("BUY. Confidence: -0.2.")
        assert isinstance(result['confidence'], float)
        assert result['confidence'] >= 0.0

    def test_confidence_is_zero_when_missing(self):
        """No confidence keyword in response → coerced to 0.0 (parse_failed=False since BUY found)."""
        result = parse_decision("BUY based on technical analysis.")
        assert result['confidence'] == 0.0
        assert result['parse_failed'] is False

    # ------------------------------------------------------------------
    # Output structure
    # ------------------------------------------------------------------

    def test_output_has_required_keys(self):
        result = parse_decision("BUY. Confidence: 0.80.")
        assert {'decision', 'confidence', 'reasoning', 'raw_response'}.issubset(result.keys())

    def test_reasoning_is_string(self):
        result = parse_decision("SELL. Confidence: 0.70. Reasons: RSI overbought, MACD cross.")
        assert isinstance(result['reasoning'], str)

    def test_reasoning_max_200_chars(self):
        long_response = "BUY. " + "x" * 500
        result = parse_decision(long_response)
        assert len(result['reasoning']) <= 200

    def test_raw_response_preserved(self):
        raw = "HOLD. Confidence: 0.50."
        result = parse_decision(raw)
        assert result['raw_response'] == raw

    # ------------------------------------------------------------------
    # JSON-first parsing (E1)
    # ------------------------------------------------------------------

    def test_json_block_in_prose_parsed_first(self):
        """A JSON block embedded in prose should be parsed via JSON path, not regex."""
        response = (
            'Sure, here is my analysis:\n'
            '{"decision": "buy", "confidence": 0.88, "reasoning": "Strong breakout"}\n'
            'Let me know if you need more details.'
        )
        result = parse_decision(response)
        assert result['decision']   == 'buy'
        assert result['confidence'] == 0.88
        assert result['parse_method'] == 'json'

    def test_json_sell_decision(self):
        response = '{"decision": "SELL", "confidence": 0.75, "reasoning": "Overbought RSI"}'
        result = parse_decision(response)
        assert result['decision'] == 'sell'

    def test_json_hold_decision(self):
        response = '{"decision": "hold", "confidence": 0.60, "reasoning": "Neutral signals"}'
        result = parse_decision(response)
        assert result['decision'] == 'hold'

    def test_json_confidence_normalised_from_100_scale(self):
        """JSON confidence on 0-100 scale should be normalised to 0-1."""
        response = '{"decision": "buy", "confidence": 82, "reasoning": "Momentum"}'
        result = parse_decision(response)
        assert abs(result['confidence'] - 0.82) < 0.01

    def test_malformed_json_falls_back_to_regex(self):
        """Invalid JSON should fall through to the regex path without error."""
        response = '{"decision": "buy" confidence 0.80} BUY. Confidence: 0.80.'
        result = parse_decision(response)
        assert result['decision'] == 'buy'
        # confidence from regex fallback
        assert result['confidence'] == 0.80

    # ------------------------------------------------------------------
    # _parse_failures_count (E1)
    # ------------------------------------------------------------------

    def test_parse_failures_count_increments_on_ambiguous_response(self):
        """A totally ambiguous response should increment _parse_failures_count."""
        before = _mli._parse_failures_count
        # Completely ambiguous — no buy/sell/hold, no confidence
        parse_decision("The market conditions are complex. There is uncertainty.")
        after = _mli._parse_failures_count
        assert after > before

    # ------------------------------------------------------------------
    # Sprint01 C1 — decision-inversion hardening
    # ------------------------------------------------------------------

    def test_both_buy_and_sell_present_is_ambiguous_hold(self):
        """Free text containing both 'buy' and 'sell' with no Decision: line
        must not guess a direction — resolves to hold, parse_failed=True."""
        result = parse_decision(
            "The analyst said buy signals are weak but some might sell. Confidence: 0.60."
        )
        assert result['decision'] == 'hold'
        assert result['parse_failed'] is True

    def test_decision_line_wins_over_conflicting_reasoning_substrings(self):
        """A structured Decision: line is authoritative even if the reasoning
        text elsewhere contains the opposite action word."""
        result = parse_decision(
            "Decision: BUY\nConfidence: 0.80\n"
            "Reasoning: the technicals call for a sell-off eventually but not yet"
        )
        assert result['decision'] == 'buy'
        assert result['parse_method'] == 'structured_line'

    def test_word_boundary_excludes_buyers_false_match(self):
        """'Buyers' must not be treated as a 'buy' signal — \\bbuy\\b requires a
        real word boundary, and 'buy' immediately followed by 'ers' has none."""
        result = parse_decision("Buyers are exhausted, no clear signal. Confidence: 0.5.")
        assert result['decision'] != 'buy'

    def test_json_buy_put_accepted(self):
        """JSON path must accept buy_put directly (sprint01 A1 fix) rather than
        falling through to the regex fallback, which would silently drop it."""
        response = '{"decision": "buy_put", "confidence": 0.72, "reasoning": "bearish momentum"}'
        result = parse_decision(response)
        assert result['decision'] == 'buy_put'
        assert result['parse_method'] == 'json'

    def test_structured_line_buy_call(self):
        result = parse_decision("Decision: BUY_CALL\nConfidence: 0.80\nReasoning: r")
        assert result['decision'] == 'buy_call'
        assert result['parse_method'] == 'structured_line'

    def test_missing_confidence_is_always_zero_not_stale_default(self):
        """Sprint01 A2: the consolidated implementation must coerce missing
        confidence to 0.0, not the older 0.5 default some revisions used."""
        result = parse_decision("Decision: HOLD\nReasoning: no clear edge")
        assert result['confidence'] == 0.0
        assert isinstance(result['confidence'], float)


class TestParseDecisionSafety:
    """confidence is never None; parse_failed distinguishes total failure from partial parse."""

    def test_confidence_always_float(self):
        """parse_decision must never return confidence=None regardless of input."""
        ambiguous = "The market conditions are complex. There is uncertainty."
        partial   = "BUY based on technical analysis."
        clear     = "BUY. Confidence: 0.80."
        for response in (ambiguous, partial, clear):
            result = parse_decision(response)
            assert isinstance(result['confidence'], float), (
                f"confidence is not a float for: {response!r}"
            )

    def test_parse_failed_true_only_on_total_failure(self):
        """parse_failed=True when neither decision NOR confidence is parseable."""
        result = parse_decision("The market conditions are complex. There is uncertainty.")
        assert result['parse_failed'] is True
        assert result['confidence'] == 0.0

    def test_parse_failed_false_when_decision_found_without_confidence(self):
        """Decision found but no confidence keyword → parse_failed=False, confidence=0.0."""
        result = parse_decision("BUY based on technical analysis.")
        assert result['parse_failed'] is False
        assert result['confidence'] == 0.0

    def test_parse_failed_false_on_successful_parse(self):
        result = parse_decision("BUY. Confidence: 0.82.")
        assert result['parse_failed'] is False
        assert result['confidence'] == 0.82

    def test_parse_failed_false_on_json_parse(self):
        result = parse_decision('{"decision": "buy", "confidence": 0.88, "reasoning": "test"}')
        assert result['parse_failed'] is False

    def test_ambiguous_response_safe_in_session_loop_comparisons(self):
        """The returned dict from a total-failure parse must survive all agent comparisons."""
        result = parse_decision("The market conditions are complex. There is uncertainty.")
        # These are the exact comparisons that were crashing in autonomous_agent.py
        min_confidence = 0.60
        debate_threshold = 0.90
        _ = result['confidence'] >= min_confidence           # was TypeError
        _ = result['confidence'] >= debate_threshold         # was TypeError
        _ = f"{result['confidence']:.2f}"                   # was TypeError
        _ = f"{result['confidence']:.0%}"                   # was TypeError

    def test_ambiguous_response_safe_in_rebalance_path(self):
        """Arithmetic used in _maybe_rotate_portfolio must not raise on parse-failure output."""
        result = parse_decision("The market conditions are complex. There is uncertainty.")
        assumed_return = 0.05
        # autonomous_agent.py:959 — new_ev = new_decision['confidence'] * assumed_return
        new_ev = result['confidence'] * assumed_return      # was TypeError
        # autonomous_agent.py:980 — if new_decision['confidence'] <= weakest['confidence']
        _ = result['confidence'] <= result['confidence']    # was TypeError
        # autonomous_agent.py:983 — f"{new_decision['confidence']:.2f}"
        _ = f"{result['confidence']:.2f}"                  # was TypeError
        assert new_ev == 0.0  # 0.0 * anything = 0.0 → rotation will be rejected (correct)


class TestReasoningContaminationViaReexport:
    """Same contamination guard as tests/test_decision_parser.py, exercised through
    the model_inference_lora re-export path that the agents actually import.

    Kept in both files deliberately — see this file's docstring: the two import
    paths are covered separately so a regression in either is caught.
    """

    def test_nested_block_is_unwrapped(self):
        response = ('Decision: BUY\nConfidence: 0.85\n'
                    'Reasoning: Decision: BUY Confidence: 0.85 Reasoning: Momentum confirmed.')
        assert parse_decision(response)['reasoning'] == 'Momentum confirmed.'

    def test_outcome_leakage_is_stripped(self):
        """The model must not be trained to state outcomes it cannot know."""
        response = ('Decision: BUY\nConfidence: 0.85\n'
                    'Reasoning: Good setup. Outcome: Small win (+4.2%). Reward signal: +0.0003')
        reasoning = parse_decision(response)['reasoning']
        assert reasoning == 'Good setup.'
        assert 'Outcome' not in reasoning and 'Reward signal' not in reasoning

    def test_reasoning_still_capped_at_200(self):
        response = 'Decision: BUY\nConfidence: 0.8\nReasoning: ' + 'y' * 400
        assert len(parse_decision(response)['reasoning']) <= 200

    def test_reasoning_is_single_line(self):
        response = 'Decision: HOLD\nConfidence: 0.5\nReasoning: first\nsecond'
        assert '\n' not in parse_decision(response)['reasoning']
