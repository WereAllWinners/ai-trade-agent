#!/usr/bin/env python3
"""
Lightweight decision parser — no torch / transformers / peft / unsloth.

Import this instead of model_inference_lora when you only need parse_decision.
Agents use Ollama for live inference; the heavy LoRA stack is only needed at
fine-tuning time and must NOT be imported at agent startup.
"""
import re
import logging

# Matches:  "Confidence: 0.82"  /  "confidence: 82%"  /  "confidence: 82"
_CONF_STRUCTURED = re.compile(
    r'confidence\s*[:\-]\s*(\d*\.?\d+)\s*%?', re.IGNORECASE
)
# Matches prose like "I am 80% confident" / "85% sure" / "with 0.9 certainty"
_CONF_PROSE = re.compile(
    r'(\d{1,3})\s*%\s*(?:confident|confidence|sure|certain)|'
    r'(?:confidence|certainty)\s+of\s+(\d*\.?\d+)',
    re.IGNORECASE
)
# Matches structured "Decision: BUY" line
_DECISION_LINE = re.compile(r'decision\s*[:\-]\s*(BUY_CALL|BUY_PUT|BUY|SELL|HOLD)', re.IGNORECASE)
# Matches structured "Reasoning: ..." line
_REASONING_LINE = re.compile(r'reasoning\s*[:\-]\s*(.+)', re.IGNORECASE)


def parse_decision(response: str) -> dict:
    """Parse a model response string into a structured trading decision.

    Tries structured "Decision: / Confidence: / Reasoning:" format first,
    then falls back to keyword/prose scanning for backwards compatibility.

    Returns a dict with keys: decision, confidence, reasoning, raw_response.
    """
    response_lower = response.lower()

    # --- Decision ---
    decision = "hold"
    m = _DECISION_LINE.search(response)
    if m:
        decision = m.group(1).lower()
    else:
        # Keyword fallback
        if "buy_call" in response_lower:
            decision = "buy_call"
        elif "buy_put" in response_lower:
            decision = "buy_put"
        elif "buy" in response_lower and "don't buy" not in response_lower and "not buy" not in response_lower:
            decision = "buy"
        elif "sell" in response_lower and "don't sell" not in response_lower and "not sell" not in response_lower:
            decision = "sell"

    # --- Confidence ---
    confidence = 0.5
    cm = _CONF_STRUCTURED.search(response)
    if cm:
        try:
            val = float(cm.group(1))
            if val > 1:
                val /= 100
            confidence = max(0.0, min(1.0, val))
        except (ValueError, TypeError) as exc:
            logging.warning(f"Could not parse confidence '{cm.group(1)}': {exc}")
    else:
        pm = _CONF_PROSE.search(response)
        if pm:
            raw = pm.group(1) or pm.group(2)
            try:
                val = float(raw)
                if val > 1:
                    val /= 100
                confidence = max(0.0, min(1.0, val))
            except (ValueError, TypeError):
                pass
        else:
            logging.debug("No confidence value found in model response, defaulting to 0.5")

    # --- Reasoning ---
    rm = _REASONING_LINE.search(response)
    if rm:
        reasoning = rm.group(1).strip()[:300]
    else:
        reasoning = response[:200].replace('\n', ' ').strip()

    return {
        'decision': decision,
        'confidence': confidence,
        'reasoning': reasoning,
        'raw_response': response,
    }
