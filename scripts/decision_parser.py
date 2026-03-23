#!/usr/bin/env python3
"""
Lightweight decision parser — no torch / transformers / peft / unsloth.

Import this instead of model_inference_lora when you only need parse_decision.
Agents use Ollama for live inference; the heavy LoRA stack is only needed at
fine-tuning time and must NOT be imported at agent startup.
"""
import re
import logging

def parse_decision(response: str) -> dict:
    """Parse a model response string into a structured trading decision.

    Returns a dict with keys: decision, confidence, reasoning, raw_response.
    """
    response_lower = response.lower()

    # Extract decision (BUY, SELL, HOLD)
    decision = "hold"
    if "buy" in response_lower and "don't buy" not in response_lower and "not buy" not in response_lower:
        decision = "buy"
    elif "sell" in response_lower and "don't sell" not in response_lower and "not sell" not in response_lower:
        decision = "sell"

    # Extract confidence — accept "0-1" or "0-100" scale
    confidence = 0.5
    confidence_match = re.search(r'confidence[:\s]+(\d*\.?\d+)', response_lower)
    if confidence_match:
        try:
            conf_val = float(confidence_match.group(1))
            if conf_val > 1:
                conf_val = conf_val / 100
            confidence = max(0.0, min(1.0, conf_val))
        except (ValueError, TypeError) as exc:
            logging.warning(
                f"Could not parse confidence value '{confidence_match.group(1)}': {exc}. "
                "Defaulting to 0.5"
            )
    else:
        logging.debug("No confidence value found in model response, defaulting to 0.5")

    # Extract reasoning (first 200 chars, single line)
    reasoning = response[:200].replace('\n', ' ').strip()

    return {
        'decision': decision,
        'confidence': confidence,
        'reasoning': reasoning,
        'raw_response': response,
    }
