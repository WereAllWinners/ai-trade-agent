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
        # Regex matches only non-negative numbers; "-0.2" is not parsed → None.
        # Downstream callers treat confidence=None as 0.0 via `confidence or 0.0`.
        result = parse_decision("BUY. Confidence: -0.2.")
        assert result['confidence'] is None or result['confidence'] >= 0.0

    def test_confidence_is_none_when_missing(self):
        """No confidence keyword in response → confidence should be None (not 0.5)."""
        result = parse_decision("BUY based on technical analysis.")
        assert result['confidence'] is None

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
