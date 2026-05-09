"""
Unit tests for utils/alpaca_retry.py — retry decorator for transient Alpaca API errors.

All tests patch time.sleep to a no-op so the suite stays fast.
"""
import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))


def _make_api_error(status_code: int):
    """Build an APIError whose .status_code property returns status_code."""
    from alpaca.common.exceptions import APIError
    resp = MagicMock()
    resp.status_code = status_code
    http_err = MagicMock()
    http_err.response = resp
    return APIError(json.dumps({'code': status_code, 'message': 'test'}), http_error=http_err)


class TestRetryOnRateLimit:
    def test_single_429_then_success_retries_and_returns(self):
        """A single 429 should be retried; the second call should return its value."""
        from utils.alpaca_retry import retry_on_rate_limit, _MAX_RETRIES

        call_count = [0]

        def flaky():
            call_count[0] += 1
            if call_count[0] == 1:
                raise _make_api_error(429)
            return 'ok'

        decorated = retry_on_rate_limit(flaky)

        with patch('utils.alpaca_retry.time.sleep'):
            result = decorated()

        assert result == 'ok'
        assert call_count[0] == 2

    def test_exhausted_retries_raises(self):
        """After _MAX_RETRIES retries the decorator must re-raise the original exception."""
        from alpaca.common.exceptions import APIError
        from utils.alpaca_retry import retry_on_rate_limit, _MAX_RETRIES

        call_count = [0]

        def always_429():
            call_count[0] += 1
            raise _make_api_error(429)

        decorated = retry_on_rate_limit(always_429)

        with patch('utils.alpaca_retry.time.sleep'):
            with pytest.raises(APIError):
                decorated()

        assert call_count[0] == _MAX_RETRIES + 1

    def test_non_retryable_exception_not_retried(self):
        """A 404 is not retryable — the decorator must raise immediately without retry."""
        from alpaca.common.exceptions import APIError
        from utils.alpaca_retry import retry_on_rate_limit

        call_count = [0]

        def not_found():
            call_count[0] += 1
            raise _make_api_error(404)

        decorated = retry_on_rate_limit(not_found)

        with patch('utils.alpaca_retry.time.sleep'):
            with pytest.raises(APIError):
                decorated()

        assert call_count[0] == 1
