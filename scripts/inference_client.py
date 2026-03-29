#!/usr/bin/env python3
"""
inference_client.py — Unified LLM inference abstraction.

Supports two backends selectable via the INFERENCE_BACKEND env var:

  ollama (default)
    Talks to a local Ollama server.  Model stays resident in GPU/CPU memory
    between calls.  Lowest setup friction.
    Required: Ollama running + model pulled (ollama pull qwen3:8b)

  vllm
    Calls a running vLLM server's OpenAI-compatible /v1/completions API.
    Gives 3-5× faster tokens/s (PagedAttention + continuous batching) and
    supports AWQ / GGUF quantized weights without Ollama overhead.
    Required: vLLM server running (see scripts/start_vllm_server.sh)

Environment variables:
  INFERENCE_BACKEND  ollama | vllm         (default: ollama)
  OLLAMA_MODEL       model tag             (default: qwen3:8b)
  VLLM_BASE_URL      vLLM server URL       (default: http://localhost:8000/v1)
  VLLM_MODEL         HuggingFace model ID  (default: Qwen/Qwen2.5-7B-Instruct-AWQ)
  VLLM_API_KEY       bearer token          (default: token — vLLM default)
  INFERENCE_TIMEOUT  request timeout (s)   (default: 45)

Usage:
  from inference_client import generate, warmup, stop_for_finetuning
  response_text = generate(prompt, max_tokens=200)
"""
import logging
import os

INFERENCE_BACKEND: str = os.getenv('INFERENCE_BACKEND', 'ollama').lower()
OLLAMA_MODEL:      str = os.getenv('OLLAMA_MODEL', 'qwen3:8b')
VLLM_BASE_URL:     str = os.getenv('VLLM_BASE_URL', 'http://localhost:8000/v1')
VLLM_MODEL:        str = os.getenv('VLLM_MODEL', 'Qwen/Qwen2.5-7B-Instruct-AWQ')
VLLM_API_KEY:      str = os.getenv('VLLM_API_KEY', 'token')
INFERENCE_TIMEOUT: int = int(os.getenv('INFERENCE_TIMEOUT', '45'))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
    """
    Generate a completion for *prompt*.

    Returns the model's response as a plain string.
    Raises on unrecoverable errors so callers can catch and log them.
    """
    if INFERENCE_BACKEND == 'vllm':
        return _generate_vllm(prompt, max_tokens, temperature)
    return _generate_ollama(prompt, max_tokens, temperature)


def warmup() -> None:
    """
    Pre-load the model into GPU/CPU memory so the first real decision call
    is not delayed by a cold-start (typically 8-15 s for 7-32B models).
    Call once at daemon / agent startup.
    """
    if INFERENCE_BACKEND == 'vllm':
        _warmup_vllm()
    else:
        _warmup_ollama()


def stop_for_finetuning() -> None:
    """
    Release GPU VRAM before a fine-tuning job loads the base model.
    Ollama only — vLLM serves continuously and is not stopped for fine-tuning.
    (Fine-tuning uses a separate process with different model weights.)
    """
    if INFERENCE_BACKEND == 'ollama':
        _stop_ollama()
    else:
        logging.debug("inference_client: vLLM backend — skipping model unload (fine-tuning is separate)")


def backend_info() -> dict:
    """Return a dict describing the active backend (for /status endpoint)."""
    if INFERENCE_BACKEND == 'vllm':
        return {'backend': 'vllm', 'model': VLLM_MODEL, 'url': VLLM_BASE_URL}
    return {'backend': 'ollama', 'model': OLLAMA_MODEL}


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

def _generate_ollama(prompt: str, max_tokens: int, temperature: float) -> str:
    import ollama
    response = ollama.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        think=False,
        options={
            "temperature": temperature,
            "top_p": 0.9,
            "num_predict": max_tokens,
        },
    )
    return response['response']


def _warmup_ollama() -> None:
    try:
        import ollama
        ollama.generate(
            model=OLLAMA_MODEL,
            prompt="warmup",
            think=False,
            options={"num_predict": 1},
        )
        logging.info(f"🔥 Ollama model '{OLLAMA_MODEL}' warmed up")
    except Exception as e:
        logging.warning(f"⚠️  Ollama warm-up failed (will cold-start on first call): {e}")


def _stop_ollama() -> None:
    import subprocess
    try:
        result = subprocess.run(
            ['ollama', 'stop', OLLAMA_MODEL],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            logging.info(f"🛑 Ollama model '{OLLAMA_MODEL}' unloaded from GPU (pre-finetune)")
        else:
            logging.warning(f"⚠️  ollama stop returned {result.returncode}: {result.stderr.strip()}")
    except FileNotFoundError:
        logging.debug("ollama CLI not found — skipping model unload")
    except Exception as e:
        logging.warning(f"⚠️  Could not unload Ollama model: {e}")


# ---------------------------------------------------------------------------
# vLLM backend  (/v1/completions — OpenAI-compatible)
# ---------------------------------------------------------------------------

def _generate_vllm(prompt: str, max_tokens: int, temperature: float) -> str:
    import requests
    payload = {
        "model": VLLM_MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
    }
    headers = {"Authorization": f"Bearer {VLLM_API_KEY}"}
    try:
        resp = requests.post(
            f"{VLLM_BASE_URL}/completions",
            json=payload,
            headers=headers,
            timeout=INFERENCE_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()['choices'][0]['text']
    except Exception as e:
        logging.error(f"vLLM inference failed: {e} — falling back to Ollama")
        return _generate_ollama(prompt, max_tokens, temperature)


def _warmup_vllm() -> None:
    try:
        _generate_vllm("warmup", max_tokens=1, temperature=0.0)
        logging.info(f"🔥 vLLM model '{VLLM_MODEL}' warmed up at {VLLM_BASE_URL}")
    except Exception as e:
        logging.warning(f"⚠️  vLLM warm-up failed: {e}")
