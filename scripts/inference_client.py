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
  INFERENCE_BACKEND          ollama | vllm         (default: ollama)
  OLLAMA_MODEL               model tag             (default: qwen3:8b)
  VLLM_BASE_URL              vLLM server URL       (default: http://localhost:8000/v1)
  VLLM_MODEL                 HuggingFace model ID  (default: Qwen/Qwen2.5-7B-Instruct-AWQ)
  VLLM_API_KEY               bearer token          (default: token — vLLM default)
  INFERENCE_TIMEOUT          request timeout (s)   (default: 45)
  ALLOW_INTRADAY_VLLM_RESTART  set 'true' to allow lock-guarded restart during market hours

Usage:
  from inference_client import generate, warmup, stop_for_finetuning
  response_text = generate(prompt, max_tokens=200)
"""
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path

INFERENCE_BACKEND: str = os.getenv('INFERENCE_BACKEND', 'ollama').lower()
OLLAMA_MODEL:      str = os.getenv('OLLAMA_MODEL', 'qwen3:8b')
VLLM_BASE_URL:     str = os.getenv('VLLM_BASE_URL', 'http://localhost:8000/v1')
VLLM_MODEL:        str = os.getenv('VLLM_MODEL', 'Qwen/Qwen2.5-32B-Instruct')
VLLM_API_KEY:      str = os.getenv('VLLM_API_KEY', 'token')
INFERENCE_TIMEOUT: int = int(os.getenv('INFERENCE_TIMEOUT', '120'))

_ALLOW_INTRADAY_VLLM_RESTART: bool = (
    os.getenv('ALLOW_INTRADAY_VLLM_RESTART', 'false').lower() == 'true'
)

_LOG_DIR = Path(__file__).resolve().parent.parent / 'logs'
_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Single-owner file lock for all vLLM lifecycle operations.
# timeout=5: a second caller gives up in 5s and returns False — never queues another stop.
try:
    from filelock import FileLock, Timeout as _FileLockTimeout
    _LIFECYCLE_LOCK = FileLock(str(_LOG_DIR / '.vllm_lifecycle.lock'), timeout=5)
except ImportError:
    _LIFECYCLE_LOCK = None   # filelock not installed — degraded mode (no lock)
    _FileLockTimeout = Exception


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
        try:
            return _generate_vllm(prompt, max_tokens, temperature)
        except Exception as e:
            # Connection-refused class: vLLM may have died. Attempt a lock-guarded
            # start (never stop/restart) and retry once. Market-hours guard is
            # intentionally NOT applied here — starting an already-down server is safe.
            err_str = str(e).lower()
            if any(kw in err_str for kw in ('connection refused', 'connectionerror',
                                             'connect error', 'failed to establish')):
                logging.warning("⚠️  vLLM connection refused — attempting self-heal via ensure_vllm_running")
                try:
                    ensure_vllm_running()
                    return _generate_vllm(prompt, max_tokens, temperature)
                except Exception as heal_err:
                    raise RuntimeError(
                        f"vLLM inference failed and self-heal did not restore service: {heal_err}"
                    ) from e
            raise
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


def stop_for_finetuning() -> bool:
    """
    Release GPU VRAM before a fine-tuning job loads the base model.
    Fine-tuning Qwen 32B needs ~130 GB; vLLM holding the model concurrently
    overflows the GB10's 128 GB and causes hard NVRM OOM crashes.

    Returns True if the stop was issued, False if it was blocked (market hours
    or lock contention). Callers MUST abort the fine-tune when False is returned —
    the server is still live and serving.
    """
    if INFERENCE_BACKEND != 'vllm':
        _stop_ollama()
        return True

    if _is_market_hours() and not _ALLOW_INTRADAY_VLLM_RESTART:
        _send_warning(
            'vllm_stop_blocked_market_hours',
            'vLLM stop blocked during market hours. Fine-tune will run at scheduled time. '
            'Set ALLOW_INTRADAY_VLLM_RESTART=true to override.',
        )
        logging.warning("⚠️  stop_for_finetuning() blocked — market is open")
        return False

    if _LIFECYCLE_LOCK is None:
        _stop_vllm()
        return True

    try:
        with _LIFECYCLE_LOCK:
            _stop_vllm()
            return True
    except _FileLockTimeout:
        logging.warning(
            "⚠️  Lifecycle lock held by another process (5 s timeout) — "
            "skipping stop; another lifecycle operation is in progress"
        )
        return False


def start_after_finetuning() -> None:
    """
    Restart the inference server after fine-tuning so it picks up the new merged model.
    Called by the daemon AFTER the finetune_model subprocess (and all its children) have
    fully exited, ensuring their GPU memory is released before vLLM loads.

    Checks is-active first: if vLLM is already running (e.g. stop was blocked by market-
    hours guard and never actually fired), this is a no-op — issuing restart would contain
    a stop and bypass the guard that just protected the live server.
    """
    if INFERENCE_BACKEND != 'vllm':
        return

    r = subprocess.run(
        ['sudo', 'systemctl', 'is-active', 'ai-inference-server.service'],
        capture_output=True, text=True,
    )
    if r.stdout.strip() == 'active':
        logging.info("ℹ️  vLLM already active — skipping start_after_finetuning (server was never stopped)")
        return

    def _do_restart() -> None:
        import time as _t
        # Wait for the CUDA driver to consolidate memory freed by the fine-tuning
        # subprocess. Without this delay, vLLM starts on fragmented memory and the
        # CUDA graph profiler hits NVRM OOM (_memdescAllocInternal), which leaves the
        # GPU driver in an unstable state and causes a hard system crash hours later.
        logging.info("⏳ Waiting 90s for GPU memory to drain before restarting vLLM…")
        _t.sleep(90)
        _restart_vllm()
        _poll_vllm_ready(timeout_s=600)

    if _LIFECYCLE_LOCK is None:
        try:
            _do_restart()
        except Exception as e:
            _send_critical(
                'vllm_restart_failed',
                f'vLLM failed to restart after fine-tune: {e}. Manual intervention required.',
            )
            raise
        return

    try:
        with _LIFECYCLE_LOCK:
            _do_restart()
    except _FileLockTimeout:
        logging.warning("⚠️  Lifecycle lock held — skipping start_after_finetuning")
    except Exception as e:
        _send_critical(
            'vllm_restart_failed',
            f'vLLM failed to restart after fine-tune: {e}. Manual intervention required.',
        )
        raise


def ensure_vllm_running() -> None:
    """
    Self-heal: start vLLM if not currently active. Never issues stop or restart.
    Lock-guarded with a TOCTOU re-check under the lock.
    Safe to call during market hours — it only starts, never stops.
    """
    if INFERENCE_BACKEND != 'vllm':
        return

    r = subprocess.run(
        ['sudo', 'systemctl', 'is-active', 'ai-inference-server.service'],
        capture_output=True, text=True,
    )
    if r.stdout.strip() == 'active':
        return

    if _LIFECYCLE_LOCK is None:
        logging.warning("⚠️  vLLM inactive — issuing systemctl start (no stop/restart)")
        subprocess.run(
            ['sudo', 'systemctl', 'start', 'ai-inference-server.service'],
            capture_output=True, text=True, timeout=30,
        )
        _poll_vllm_ready(timeout_s=600)
        return

    try:
        with _LIFECYCLE_LOCK:
            # Re-check under lock to avoid TOCTOU (another process may have started it
            # in the window between our first check and acquiring the lock).
            r2 = subprocess.run(
                ['sudo', 'systemctl', 'is-active', 'ai-inference-server.service'],
                capture_output=True, text=True,
            )
            if r2.stdout.strip() == 'active':
                return
            logging.warning("⚠️  vLLM inactive — issuing systemctl start (no stop/restart)")
            subprocess.run(
                ['sudo', 'systemctl', 'start', 'ai-inference-server.service'],
                capture_output=True, text=True, timeout=30,
            )
            _poll_vllm_ready(timeout_s=600)
    except _FileLockTimeout:
        logging.warning("⚠️  Lifecycle lock held — ensure_vllm_running skipped")
    except Exception as e:
        _send_critical(
            'vllm_start_failed',
            f'ensure_vllm_running failed: {e}. Manual start required.',
        )


def backend_info() -> dict:
    """Return a dict describing the active backend (for /status endpoint)."""
    if INFERENCE_BACKEND == 'vllm':
        return {'backend': 'vllm', 'model': VLLM_MODEL, 'url': VLLM_BASE_URL}
    return {'backend': 'ollama', 'model': OLLAMA_MODEL}


# ---------------------------------------------------------------------------
# Alerts helpers (lazy import to avoid circular deps at module level)
# ---------------------------------------------------------------------------

def _send_warning(event: str, message: str) -> None:
    try:
        from alerts import send_alert, AlertLevel
        send_alert(AlertLevel.WARNING, event, message)
    except Exception:
        logging.warning("⚠️  [%s] %s", event, message)


def _send_critical(event: str, message: str) -> None:
    try:
        from alerts import send_alert, AlertLevel
        send_alert(AlertLevel.CRITICAL, event, message)
    except Exception:
        logging.critical("🚨 [%s] %s", event, message)


# ---------------------------------------------------------------------------
# Market-hours guard
# ---------------------------------------------------------------------------

def _is_market_hours() -> bool:
    """True between 9:25 AM and 4:05 PM EST on weekdays."""
    try:
        import pytz
        from datetime import time as _dt_time
        est = pytz.timezone('US/Eastern')
        now = datetime.now(est)
        if now.weekday() >= 5:
            return False
        return _dt_time(9, 25) <= now.time() <= _dt_time(16, 5)
    except Exception:
        return False   # if pytz unavailable, don't block lifecycle ops


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


def _stop_vllm() -> None:
    try:
        result = subprocess.run(
            ['sudo', 'systemctl', 'stop', 'ai-inference-server.service'],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            logging.info("🛑 vLLM inference server stopped (pre-finetune, frees ~20 GB VRAM)")
        else:
            logging.warning(f"⚠️  systemctl stop returned {result.returncode}: {result.stderr.strip()}")
    except Exception as e:
        logging.warning(f"⚠️  Could not stop vLLM server: {e}")


def _restart_vllm() -> None:
    try:
        result = subprocess.run(
            ['sudo', 'systemctl', 'restart', 'ai-inference-server.service'],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            logging.info("🔄 ai-inference-server restarting (new merged model loads in ~2-3 min)")
        else:
            raise RuntimeError(
                f"systemctl restart returned {result.returncode}: {result.stderr.strip()}"
            )
    except Exception:
        raise   # caller (_start_after_finetuning) handles with CRITICAL alert


def _poll_vllm_ready(timeout_s: int = 600) -> None:
    """Poll /v1/models until vLLM reports ready (up to timeout_s seconds)."""
    import requests, time as _t
    deadline = _t.time() + timeout_s
    while _t.time() < deadline:
        try:
            resp = requests.get(f"{VLLM_BASE_URL}/models", timeout=5)
            if resp.status_code == 200:
                logging.info("✅ vLLM is ready and accepting requests")
                return
        except Exception:
            pass
        _t.sleep(10)
    raise RuntimeError(f"vLLM did not become ready within {timeout_s}s after restart")


# ---------------------------------------------------------------------------
# vLLM backend  (/v1/completions — OpenAI-compatible)
# ---------------------------------------------------------------------------

def _generate_vllm(prompt: str, max_tokens: int, temperature: float) -> str:
    import requests, time as _time
    payload = {
        "model": VLLM_MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
    }
    headers = {"Authorization": f"Bearer {VLLM_API_KEY}"}
    # Retry up to 3 times with 10 s gaps to ride out transient vLLM restart windows
    # (e.g., the ~2-3 min reload after a nightly fine-tune promotes a new merged model).
    last_exc = None
    for attempt in range(1, 4):
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
            last_exc = e
            if attempt < 3:
                logging.warning(f"vLLM inference attempt {attempt}/3 failed: {e} — retrying in 10s")
                _time.sleep(10)
    raise RuntimeError(f"vLLM inference failed after 3 attempts: {last_exc}")


def _warmup_vllm() -> None:
    """Ping vLLM with a 1-token completion, retrying with backoff for the 2-3 min cold-start."""
    import time as _time
    for attempt in range(1, 21):   # up to ~4.5 min total
        try:
            _generate_vllm("warmup", max_tokens=1, temperature=0.0)
            logging.info(f"🔥 vLLM model '{VLLM_MODEL}' ready at {VLLM_BASE_URL} (attempt {attempt}/20)")
            return
        except Exception as e:
            if attempt == 20:
                logging.error(f"⚠️  vLLM warm-up failed after 20 attempts: {e}")
                return
            delay = min(5.0 * (1.5 ** (attempt - 1)), 16.0)
            logging.info(f"vLLM not ready yet ({attempt}/20), retrying in {delay:.1f}s...")
            _time.sleep(delay)
