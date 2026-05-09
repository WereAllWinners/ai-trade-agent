import time
import random
import functools
import logging

_MAX_RETRIES = 3
_RETRY_STATUS_CODES = frozenset({429} | set(range(500, 600)))


def _is_retryable(exc) -> bool:
    try:
        from alpaca.common.exceptions import APIError
        if isinstance(exc, APIError):
            sc = exc.status_code
            return sc is not None and sc in _RETRY_STATUS_CODES
    except ImportError:
        pass
    return False


def retry_on_rate_limit(func):
    """Decorator: retry up to _MAX_RETRIES times on Alpaca 429 / 5xx with exponential backoff."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                if attempt < _MAX_RETRIES and _is_retryable(exc):
                    wait = 2 ** (attempt + 1) + random.uniform(0, 1)  # ~2s, ~4s, ~8s
                    logging.warning(
                        "⚠️  Alpaca API %s on %s — retry %d/%d in %.1fs",
                        exc.status_code, func.__name__, attempt + 1, _MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                else:
                    raise
    return wrapper
