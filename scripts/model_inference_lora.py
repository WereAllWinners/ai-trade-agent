#!/usr/bin/env python3
"""
Model Inference with LoRA - OPTIMIZED with model caching
"""
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
from pathlib import Path

import decision_parser
from decision_parser import parse_decision

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

# E2: base-model env vars — when set, debate_trade uses a different model tag
# (un-fine-tuned base) to provide diverse perspective.  Falls back to OLLAMA_MODEL /
# VLLM_MODEL when not configured so the change is a safe no-op by default.
_OLLAMA_BASE_MODEL = os.getenv('OLLAMA_BASE_MODEL', '')
_VLLM_BASE_MODEL   = os.getenv('VLLM_BASE_MODEL', '')


def __getattr__(name):
    """PEP 562 module proxy: _parse_failures_count now lives in decision_parser
    (sprint01 C1.1 consolidation). A plain re-export would go stale since ints
    rebind on increment; this keeps `model_inference_lora._parse_failures_count`
    reflecting the live value for existing callers/tests."""
    if name == '_parse_failures_count':
        return decision_parser._parse_failures_count
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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


# parse_decision is imported from decision_parser (sprint01 C1.1 consolidation) —
# see the module import at the top of this file. Kept as a distinct name here
# (rather than only in __all__) so `from model_inference_lora import parse_decision`
# continues to work for all existing importers unchanged.


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
