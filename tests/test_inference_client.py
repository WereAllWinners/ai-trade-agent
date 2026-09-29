"""
Unit tests for inference_client's env-gated guided-decoding payload
(sprint01 C1.4, non-blocking parser hardening).

INFERENCE_GUIDED_DECODING defaults to false — these tests confirm the default
is off, and that when explicitly enabled the vLLM payload carries the nested
`structured_outputs: {"regex": ...}` shape confirmed against the deployed
vLLM 0.20.2 server (NOT the older top-level `guided_regex` key).

Run with:
  python3 -m pytest tests/ -v
"""
import sys
import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))


def _reload_inference_client(monkeypatch, guided_decoding=None):
    if guided_decoding is not None:
        monkeypatch.setenv('INFERENCE_GUIDED_DECODING', 'true' if guided_decoding else 'false')
    else:
        monkeypatch.delenv('INFERENCE_GUIDED_DECODING', raising=False)
    sys.modules.pop('inference_client', None)
    import inference_client
    return inference_client


class TestGuidedDecodingFlag:
    def test_defaults_to_false(self, monkeypatch):
        ic = _reload_inference_client(monkeypatch)
        assert ic.INFERENCE_GUIDED_DECODING is False

    def test_explicit_true(self, monkeypatch):
        ic = _reload_inference_client(monkeypatch, guided_decoding=True)
        assert ic.INFERENCE_GUIDED_DECODING is True

    def test_explicit_false(self, monkeypatch):
        ic = _reload_inference_client(monkeypatch, guided_decoding=False)
        assert ic.INFERENCE_GUIDED_DECODING is False


class TestVllmPayloadShape:
    def _mock_response(self, ic):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {'choices': [{'text': 'Decision: HOLD'}]}
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    def test_payload_omits_structured_outputs_when_disabled(self, monkeypatch):
        ic = _reload_inference_client(monkeypatch, guided_decoding=False)
        with patch('requests.post') as mock_post:
            mock_post.return_value = self._mock_response(ic)
            ic._generate_vllm("prompt", max_tokens=10, temperature=0.0)
        payload = mock_post.call_args.kwargs['json']
        assert 'structured_outputs' not in payload

    def test_payload_includes_structured_outputs_when_enabled(self, monkeypatch):
        ic = _reload_inference_client(monkeypatch, guided_decoding=True)
        with patch('requests.post') as mock_post:
            mock_post.return_value = self._mock_response(ic)
            ic._generate_vllm("prompt", max_tokens=10, temperature=0.0)
        payload = mock_post.call_args.kwargs['json']
        assert 'structured_outputs' in payload
        # Nested regex shape (vLLM 0.20.2), NOT the older top-level guided_regex key.
        assert 'guided_regex' not in payload
        assert 'regex' in payload['structured_outputs']
        assert payload['structured_outputs']['regex'] == ic._GUIDED_DECISION_REGEX
