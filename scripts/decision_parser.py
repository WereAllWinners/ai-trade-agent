#!/usr/bin/env python3
"""
Lightweight decision parser — no torch / transformers / peft / unsloth.

Import this instead of model_inference_lora when you only need parse_decision.
Agents use Ollama for live inference; the heavy LoRA stack is only needed at
fine-tuning time and must NOT be imported at agent startup.

This is the single canonical implementation of parse_decision —
model_inference_lora.py re-exports from here (sprint01 C1.1 consolidation).

Parse priority, stopping at the first success:
  1. JSON block — {"decision": ..., "confidence": ..., "reasoning": ...}
  2. Structured "Decision: <X>" line — word-boundary anchored, line-scoped
  3. Free-text keyword fallback — word-boundary regex, never returns
     buy_call/buy_put (those only ever come from paths 1 or 2)
"""
import json
import re
import logging
from datetime import datetime as _dt
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent

# E1: parse failure tracking — incremented when the response is ambiguous
# (both buy and sell signals present) or totally unparseable (no decision
# signal and no confidence value).  Written to logs/parse_failures.json every
# 50 failures for Prometheus scraping.
_parse_failures_count: int = 0

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
# Matches a structured "Decision: BUY_CALL" line, anchored to the start of a
# line. BUY_CALL/BUY_PUT must precede BUY in the alternation since regex
# alternation is first-match, and the trailing \b prevents partial matches.
_DECISION_LINE = re.compile(
    r'^\s*decision\s*[:\-]\s*(BUY_CALL|BUY_PUT|BUY|SELL|HOLD)\b',
    re.IGNORECASE | re.MULTILINE,
)
# Matches a structured "Reasoning: ..." line
_REASONING_LINE = re.compile(r'reasoning\s*[:\-]\s*(.+)', re.IGNORECASE)

# Free-text fallback word-boundary patterns. Underscore counts as a \w
# character, so these never accidentally match inside "buy_call"/"buy_put"
# tokens, nor inside "buyers"/"buying"/"selling".
_WORD_BUY  = re.compile(r'\bbuy\b', re.IGNORECASE)
_WORD_SELL = re.compile(r'\bsell\b', re.IGNORECASE)
_NEG_BUY   = re.compile(r"\b(?:don'?t|do\s+not|not)\s+buy\b", re.IGNORECASE)
_NEG_SELL  = re.compile(r"\b(?:don'?t|do\s+not|not)\s+sell\b", re.IGNORECASE)


def _extract_confidence(response: str):
    """Return a float in [0, 1], or None if no confidence value was found."""
    cm = _CONF_STRUCTURED.search(response)
    if cm:
        try:
            val = float(cm.group(1))
            if val > 1:
                val /= 100
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError) as exc:
            logging.warning(f"Could not parse confidence '{cm.group(1)}': {exc}")
            return None
    pm = _CONF_PROSE.search(response)
    if pm:
        raw = pm.group(1) or pm.group(2)
        try:
            val = float(raw)
            if val > 1:
                val /= 100
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            return None
    return None


# Splits on every "Reasoning:" marker so nested blocks can be unwrapped. Kept
# separate from _REASONING_LINE, which only locates the first one.
_REASONING_SPLIT = re.compile(r'(?i)reasoning\s*[:\-]\s*')
# Trailer appended after the real reasoning by contaminated rows. "Outcome:" and
# "Reward signal:" are future information the model cannot know at decision time
# — leaving them in trains the model to predict its own label.
_REASONING_TRAILER = re.compile(r'\s*\n?(?:outcome|reward\s+signal|result)\s*[:\-]', re.IGNORECASE)


def unwrap_reasoning(text: str) -> str:
    """Strip nested Decision/Confidence/Reasoning blocks and outcome trailers.

    Models trained on contaminated examples regurgitate whole blocks inside the
    Reasoning field, sometimes several levels deep and with the trade's outcome
    appended:

        Reasoning: Decision: BUY Confidence: 0.85 Reasoning: <the real text>
                   Outcome: Small win (+4.2%). Reward signal: +0.0003

    Taking the RIGHTMOST "Reasoning:" segment unwraps every level at once,
    however deep, without needing to match the Decision/Confidence preamble —
    which matters because the preamble is written inconsistently (plain,
    markdown-bolded as ``**Decision:**``, and sometimes run straight on from the
    preceding word). Verified against all 37,975 stored raw responses: 0.16%
    retain any residual marker, versus 10% for a prefix-stripping approach.

    Returns '' when nothing survives. Callers that need a placeholder supply
    their own — this must never invent reasoning text, because it feeds the
    `decisions.reasoning` audit column.
    """
    if not text:
        return ''
    parts = _REASONING_SPLIT.split(text)
    actual = next((p.strip() for p in reversed(parts) if p.strip()), '')

    trailer = _REASONING_TRAILER.search(actual)
    if trailer:
        actual = actual[:trailer.start()]

    # Markdown emphasis left over from "**Reasoning:**"-style formatting. Only
    # stripped from the ends; asterisks inside the prose are left alone.
    return actual.strip().strip('*').strip()


def _extract_reasoning(response: str) -> str:
    rm = _REASONING_LINE.search(response)
    raw = rm.group(1) if rm else response
    # Collapse newlines: training_data_builder._is_clean_ideal_output requires
    # the assembled block to be exactly 3 lines, so an embedded newline here
    # would make every downstream example fail that check.
    return unwrap_reasoning(raw).replace('\n', ' ').strip()[:200]


def parse_decision(response: str) -> dict:
    """Parse a model response string into a structured trading decision.

    Returns a dict with keys: decision, confidence, reasoning, raw_response,
    parse_method ('json'|'structured_line'|'fallback'), parse_failed (bool).
    confidence is always a float, never None.
    """
    global _parse_failures_count

    # ── Attempt 1: JSON block ────────────────────────────────────────────
    json_start = response.find('{')
    json_end   = response.rfind('}')
    if json_start != -1 and json_end != -1 and json_end > json_start:
        try:
            blob = json.loads(response[json_start:json_end + 1])
            blob = {k.lower(): v for k, v in blob.items()}
            j_decision = str(blob.get('decision', '')).lower().strip()
            if j_decision in ('buy', 'buy_call', 'buy_put', 'sell', 'hold'):
                j_conf = blob.get('confidence')
                if j_conf is not None:
                    try:
                        j_conf = float(j_conf)
                        if j_conf > 1:
                            j_conf = j_conf / 100
                        j_conf = max(0.0, min(1.0, j_conf))
                    except (ValueError, TypeError):
                        j_conf = None
                j_reasoning = str(blob.get('reasoning', response[:200])).replace('\n', ' ').strip()
                return {
                    'decision':     j_decision,
                    'confidence':   j_conf if j_conf is not None else 0.0,
                    'reasoning':    j_reasoning[:200],
                    'raw_response': response,
                    'parse_method': 'json',
                    'parse_failed': False,
                }
        except (json.JSONDecodeError, Exception):
            pass

    # ── Attempt 2: structured "Decision:" line ───────────────────────────
    m = _DECISION_LINE.search(response)
    if m:
        decision = m.group(1).lower()
        confidence = _extract_confidence(response)
        return {
            'decision':     decision,
            'confidence':   confidence if confidence is not None else 0.0,
            'reasoning':    _extract_reasoning(response),
            'raw_response': response,
            'parse_method': 'structured_line',
            'parse_failed': False,
        }

    # ── Attempt 3: free-text keyword fallback ────────────────────────────
    # Never returns buy_call/buy_put — those only come from paths 1 and 2.
    has_buy  = bool(_WORD_BUY.search(response))  and not _NEG_BUY.search(response)
    has_sell = bool(_WORD_SELL.search(response)) and not _NEG_SELL.search(response)
    confidence = _extract_confidence(response)

    if has_buy and has_sell:
        # Ambiguous — both signals present, do not guess a direction.
        decision = 'hold'
        parse_failed = True
    elif has_buy:
        decision = 'buy'
        parse_failed = False
    elif has_sell:
        decision = 'sell'
        parse_failed = False
    else:
        decision = 'hold'
        parse_failed = confidence is None

    if parse_failed:
        _parse_failures_count += 1
        logging.debug(
            "parse_decision: ambiguous/unparseable response (failure #%d), returning hold/0.0",
            _parse_failures_count,
        )
        _maybe_write_parse_failures()

    return {
        'decision':     decision,
        'confidence':   confidence if confidence is not None else 0.0,
        'reasoning':    _extract_reasoning(response),
        'raw_response': response,
        'parse_method': 'fallback',
        'parse_failed': parse_failed,
    }


def _maybe_write_parse_failures() -> None:
    """Persist _parse_failures_count to logs/ every 50 failures for Prometheus."""
    if _parse_failures_count % 50 != 0:
        return
    try:
        # sprint02 D4.1: shared by both stock and options agents — suffix by
        # paper/live so the counter isn't a last-writer-wins race. Note this
        # does not separate stock vs options (both paper instances still
        # share one counter, same for live) — this module's counter was
        # never bot-specific to begin with, only paper/live isolation is
        # in this stage's approved scope.
        from service_suffix import service_suffix
        path = _SCRIPTS_DIR.parent / 'logs' / f'parse_failures{service_suffix()}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            'parse_failures_total': _parse_failures_count,
            'updated_at': _dt.now().isoformat(),
        }))
    except Exception as _e:
        logging.debug("Could not write parse_failures.json: %s", _e)
