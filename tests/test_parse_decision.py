"""
Unit tests for model_inference_lora.parse_decision

Run with:
  python3 -m pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))

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
        result = parse_decision("BUY. Confidence: -0.2.")
        assert result['confidence'] >= 0.0

    def test_confidence_defaults_to_05_when_missing(self):
        result = parse_decision("BUY based on technical analysis.")
        assert result['confidence'] == 0.5

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
