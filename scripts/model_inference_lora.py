#!/usr/bin/env python3
"""
Model Inference with LoRA - OPTIMIZED with model caching
"""
import json
import os
import logging
import time
import warnings
import torch

# GB10 (sm_121) exceeds PyTorch's compiled max (sm_120); bitsandbytes handles it fine
warnings.filterwarnings("ignore", message=".*cuda capability.*", category=UserWarning)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from filelock import FileLock, Timeout as FileLockTimeout
import re
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent

# Cross-process GPU inference lock — prevents both agents from running generate()
# simultaneously, which would burst past 128 GB of unified memory on the GB10.
# Both models stay resident; only the generate() call is serialized.
_INFERENCE_LOCK_PATH = _SCRIPTS_DIR.parent / 'logs' / 'inference.lock'
_INFERENCE_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
_INFERENCE_LOCK = FileLock(str(_INFERENCE_LOCK_PATH), timeout=600)

# Global model cache - load once, use many times
_MODEL_CACHE = {
    'model': None,
    'tokenizer': None,
    'loaded': False
}

# E1: parse failure tracking — incremented when neither JSON nor regex can extract
# a valid decision from the model response.  Written to logs/parse_failures.json
# every 50 failures for Prometheus scraping.
_parse_failures_count: int = 0

# E2: base-model env vars — when set, debate_trade uses a different model tag
# (un-fine-tuned base) to provide diverse perspective.  Falls back to OLLAMA_MODEL /
# VLLM_MODEL when not configured so the change is a safe no-op by default.
_OLLAMA_BASE_MODEL = os.getenv('OLLAMA_BASE_MODEL', '')
_VLLM_BASE_MODEL   = os.getenv('VLLM_BASE_MODEL', '')


def load_model_once():
    """Load model into cache if not already loaded."""
    _backend = os.getenv('INFERENCE_BACKEND', 'direct').lower()
    if _backend in ('vllm', 'ollama'):
        from inference_client import warmup
        warmup()
        return None, None

    if _MODEL_CACHE['loaded']:
        print("✅ Using cached model (already loaded)")
        return _MODEL_CACHE['model'], _MODEL_CACHE['tokenizer']

    print("Loading base model and tokenizer...")
    base_model_path = os.getenv('BASE_MODEL', 'Qwen/Qwen2.5-32B-Instruct')
    lora_adapter_path = os.getenv(
        'LORA_ADAPTER_PATH',
        str(_SCRIPTS_DIR.parent / 'finetune' / 'finance_qwen_32b_lora_latest')
    )
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Load base model with quantization.
    # GB10 Grace Blackwell has 128 GB unified memory shared between CPU and GPU.
    # Cap each process at 45 GB so two agents can coexist (2×45 GB = 90 GB < 128 GB).
    # The 4-bit quantized 32B model uses ~18-20 GB in practice, so 45 GB is safe headroom.
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={0: "45GiB"},
        trust_remote_code=True
    )
    
    print("Loading LoRA adapter...")
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    
    # Cache the model
    _MODEL_CACHE['model'] = model
    _MODEL_CACHE['tokenizer'] = tokenizer
    _MODEL_CACHE['loaded'] = True
    
    print("✅ Model loaded and cached successfully!")
    return model, tokenizer

def get_trading_decision(prompt, max_new_tokens=200, temperature=0.7):
    """
    Get trading decision from the model.
    Uses cached model for fast inference.
    """
    _backend = os.getenv('INFERENCE_BACKEND', 'direct').lower()
    if _backend in ('vllm', 'ollama'):
        from inference_client import generate
        return generate(prompt, max_tokens=max_new_tokens, temperature=temperature)

    # Get cached model
    model, tokenizer = load_model_once()

    # Format prompt for Qwen — CPU work, no lock needed
    messages = [
        {"role": "system", "content": "You are a professional stock trading analyst. Provide clear, actionable trading decisions with confidence scores."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs_cpu = tokenizer([text], return_tensors="pt")

    # Serialize GPU work across all processes (trading-bot + options-bot share
    # 128 GB unified memory; simultaneous generate() calls burst past the limit).
    # Retry on PermissionError: filelock 3.25.x deletes the file on release, creating
    # a brief race window where os.open() can fail instead of returning EAGAIN.
    try:
        for _retry in range(5):
            try:
                with _INFERENCE_LOCK:
                    model_inputs = model_inputs_cpu.to(model.device)
                    with torch.no_grad():
                        _greedy = temperature == 0.0
                        generated_ids = model.generate(
                            **model_inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=not _greedy,
                            **({} if _greedy else {'temperature': temperature, 'top_p': 0.9}),
                            pad_token_id=tokenizer.eos_token_id
                        )
                    generated_ids = [
                        output_ids[len(input_ids):]
                        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                    ]
                    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                break
            except PermissionError:
                if _retry == 4:
                    raise
                logging.warning("inference.lock PermissionError (attempt %d/5), retrying in 1s", _retry + 1)
                time.sleep(1)
    except FileLockTimeout:
        raise RuntimeError(
            "Inference lock timeout (600 s) — the other agent may be hung. "
            "Check the trading-bot and options-bot services."
        )

    return response

def parse_decision(response: str) -> dict:
    """Parse the model's response into a structured decision.

    Attempt 1 — JSON block: look for ``{...}`` anywhere in the response.
      Accepts keys: decision, confidence, reasoning (case-insensitive).
    Attempt 2 — structured-line regex: existing Decision:/Confidence: pattern.
    On total failure: return hold with confidence=None and increment
      _parse_failures_count (caller can gate on ``confidence is None``).
    """
    global _parse_failures_count

    # ── Attempt 1: JSON block ─────────────────────────────────────────────────
    json_start = response.find('{')
    json_end   = response.rfind('}')
    if json_start != -1 and json_end != -1 and json_end > json_start:
        try:
            blob = json.loads(response[json_start:json_end + 1])
            # Normalise keys to lower-case for robustness
            blob = {k.lower(): v for k, v in blob.items()}
            j_decision = str(blob.get('decision', '')).lower().strip()
            if j_decision in ('buy', 'buy_call', 'sell', 'hold'):
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

    # ── Attempt 2: structured-line regex ─────────────────────────────────────
    response_lower = response.lower()

    decision = 'hold'
    found_decision = False
    if 'buy' in response_lower and "don't buy" not in response_lower and 'not buy' not in response_lower:
        decision = 'buy'
        found_decision = True
    elif 'sell' in response_lower and "don't sell" not in response_lower and 'not sell' not in response_lower:
        decision = 'sell'
        found_decision = True
    else:
        # 'hold' — only explicit
        for line in response_lower.splitlines():
            if 'decision:' in line and 'hold' in line:
                found_decision = True
                break

    confidence = None
    confidence_match = re.search(r'confidence[:\s]+(\d*\.?\d+)', response_lower)
    if confidence_match:
        try:
            conf_val = float(confidence_match.group(1))
            if conf_val > 1:
                conf_val = conf_val / 100
            confidence = max(0.0, min(1.0, conf_val))
        except (ValueError, TypeError) as _e:
            logging.warning(
                "Could not parse confidence value '%s': %s",
                confidence_match.group(1), _e,
            )

    reasoning = response[:200].replace('\n', ' ').strip()

    parse_failed = False
    if not found_decision and confidence is None:
        _parse_failures_count += 1
        parse_failed = True
        logging.debug(
            "parse_decision: ambiguous response (failure #%d), returning hold/0.0",
            _parse_failures_count,
        )
        _maybe_write_parse_failures()

    if confidence is None:
        confidence = 0.0

    return {
        'decision':     decision,
        'confidence':   confidence,
        'reasoning':    reasoning,
        'raw_response': response,
        'parse_method': 'regex',
        'parse_failed': parse_failed,
    }


def _maybe_write_parse_failures() -> None:
    """Persist _parse_failures_count to logs/ every 50 failures for Prometheus."""
    if _parse_failures_count % 50 != 0:
        return
    try:
        path = _SCRIPTS_DIR.parent / 'logs' / 'parse_failures.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        from datetime import datetime as _dt
        path.write_text(json.dumps({
            'parse_failures_total': _parse_failures_count,
            'updated_at': _dt.now().isoformat(),
        }))
    except Exception as _e:
        logging.debug("Could not write parse_failures.json: %s", _e)


def _generate_base_model(prompt: str, max_new_tokens: int = 200, temperature: float = 0.7) -> str:
    """Generate using the BASE model (un-fine-tuned) for diverse debate perspective.

    When OLLAMA_BASE_MODEL / VLLM_BASE_MODEL is not set, falls back silently to
    the fine-tuned model so the change is a safe no-op.

    For the direct GPU path (no client backend), running a second model would
    exhaust memory, so we also fall back to the primary model.
    """
    _backend = os.getenv('INFERENCE_BACKEND', 'direct').lower()

    if _backend == 'ollama':
        base_model = _OLLAMA_BASE_MODEL
        if not base_model:
            return get_trading_decision(prompt, max_new_tokens, temperature)
        import ollama
        response = ollama.generate(
            model=base_model,
            prompt=prompt,
            think=False,
            options={
                'temperature': temperature,
                'top_p': 0.9,
                'num_predict': max_new_tokens,
            },
        )
        return response['response']

    elif _backend == 'vllm':
        base_model = _VLLM_BASE_MODEL
        if not base_model:
            return get_trading_decision(prompt, max_new_tokens, temperature)
        from inference_client import VLLM_BASE_URL, VLLM_API_KEY, INFERENCE_TIMEOUT
        import requests
        payload = {
            'model':       base_model,
            'prompt':      prompt,
            'max_tokens':  max_new_tokens,
            'temperature': temperature,
            'top_p': 0.9,
        }
        resp = requests.post(
            f"{VLLM_BASE_URL}/completions",
            json=payload,
            headers={'Authorization': f"Bearer {VLLM_API_KEY}"},
            timeout=INFERENCE_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()['choices'][0]['text']

    else:
        # Direct GPU path: second model load would exceed memory — use fine-tuned model
        logging.debug("_generate_base_model: direct GPU backend has no separate base model; using primary")
        return get_trading_decision(prompt, max_new_tokens, temperature)

if __name__ == "__main__":
    # Test the model
    test_prompt = """Analyze AAPL for trading:

Current Price: $225.50
RSI (14): 45.2
MACD: 1.23
Volume Ratio: 1.2x average
Price Change (100 bars): +2.5%

Discovery Signals: Momentum +15%, Near 52W high

Based on this data, should we BUY, SELL, or HOLD? Provide your decision, confidence (0-1), and reasoning."""
    
    print("Testing model inference...")
    print("="*70)
    
    response = get_trading_decision(test_prompt)
    print("Raw Response:")
    print(response)
    print("="*70)
    
    decision = parse_decision(response)
    print("\nParsed Decision:")
    print(f"Decision: {decision['decision'].upper()}")
    print(f"Confidence: {decision['confidence']:.2f}")
    print(f"Reasoning: {decision['reasoning']}")
