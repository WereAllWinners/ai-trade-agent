"""
tests/test_debate_diversity.py — PR-7 E2: debate_trade base model routing

Verifies:
  - debate_trade() in stock agent calls _generate_base_model, not get_trading_decision
  - debate_trade() in options agent calls _generate_base_model
  - _debate_unavailable_count increments when _generate_base_model raises
  - _generate_base_model falls back when OLLAMA_BASE_MODEL not set
"""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / 'scripts'
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


class TestStockDebateDiversity:
    """Tests for debate_trade in autonomous_agent."""

    def test_debate_calls_generate_base_model_not_get_trading_decision(self):
        """debate_trade must call _generate_base_model, NOT get_trading_decision."""
        from agents.autonomous_agent import debate_trade

        with patch('agents.autonomous_agent._generate_base_model',
                   return_value='VERDICT: PROCEED') as mock_base, \
             patch('agents.autonomous_agent.get_trading_decision') as mock_primary:
            result = debate_trade('AAPL', 'buy', 0.92, 'Strong RSI breakout')

        mock_base.assert_called_once()
        mock_primary.assert_not_called()
        assert result['verdict'] in ('PROCEED', 'ABORT')

    def test_debate_abort_verdict_parsed(self):
        """ABORT verdict in response → debate_trade returns ABORT."""
        from agents.autonomous_agent import debate_trade

        with patch('agents.autonomous_agent._generate_base_model',
                   return_value='Risk outweighs reward.\nVERDICT: ABORT\n'):
            result = debate_trade('TSLA', 'buy', 0.95, 'High momentum')

        assert result['verdict'] == 'ABORT'

    def test_debate_proceed_verdict_parsed(self):
        """PROCEED verdict in response → debate_trade returns PROCEED."""
        from agents.autonomous_agent import debate_trade

        with patch('agents.autonomous_agent._generate_base_model',
                   return_value='Upside justifies risk.\nVERDICT: PROCEED\n'):
            result = debate_trade('NVDA', 'buy', 0.91, 'AI demand')

        assert result['verdict'] == 'PROCEED'

    def test_debate_unavailable_count_increments_on_base_model_failure(self):
        """When _generate_base_model raises, _debate_unavailable_count must increase."""
        import agents.autonomous_agent as _aa
        before = _aa._debate_unavailable_count

        with patch('agents.autonomous_agent._generate_base_model',
                   side_effect=Exception("Base model unavailable")):
            result = _aa.debate_trade('SPY', 'buy', 0.91, 'Trending up')

        assert _aa._debate_unavailable_count > before
        assert result['verdict'] == 'PROCEED'   # safe fallback

    def test_debate_defaults_to_proceed_on_no_verdict_line(self):
        """When model output has no VERDICT: line → default to PROCEED."""
        from agents.autonomous_agent import debate_trade

        with patch('agents.autonomous_agent._generate_base_model',
                   return_value='The trade looks risky but may be worth it.'):
            result = debate_trade('AAPL', 'buy', 0.90, 'Oversold bounce')

        assert result['verdict'] == 'PROCEED'


class TestOptionsDebateDiversity:
    """Tests for debate_trade in options_agent."""

    def test_options_debate_calls_generate_base_model(self):
        """options_agent debate_trade must use _generate_base_model."""
        from agents.options_agent import debate_trade

        with patch('agents.options_agent._generate_base_model',
                   return_value='VERDICT: PROCEED') as mock_base, \
             patch('agents.options_agent.get_trading_decision') as mock_primary:
            debate_trade('SPY', 'buy_call', 0.86, 'Call spread play')

        mock_base.assert_called_once()
        mock_primary.assert_not_called()

    def test_options_debate_unavailable_count_increments(self):
        """Options debate failure increments _debate_unavailable_count."""
        import agents.options_agent as _oa
        before = _oa._debate_unavailable_count

        with patch('agents.options_agent._generate_base_model',
                   side_effect=RuntimeError("timeout")):
            result = _oa.debate_trade('NVDA', 'buy_call', 0.87, 'Breakout call')

        assert _oa._debate_unavailable_count > before
        assert result['verdict'] == 'PROCEED'


class TestGenerateBaseModelFallback:
    """Tests for _generate_base_model fallback behaviour."""

    def test_ollama_without_base_model_env_falls_back_to_primary(self, monkeypatch):
        """When OLLAMA_BASE_MODEL is not set, _generate_base_model calls get_trading_decision."""
        monkeypatch.setenv('INFERENCE_BACKEND', 'ollama')
        monkeypatch.setenv('OLLAMA_BASE_MODEL', '')

        sys.modules.pop('model_inference_lora', None)
        with patch.dict('os.environ', {'INFERENCE_BACKEND': 'ollama', 'OLLAMA_BASE_MODEL': ''}):
            import model_inference_lora as _mli
            _mli._OLLAMA_BASE_MODEL = ''  # ensure runtime value is empty
            with patch.object(_mli, 'get_trading_decision',
                               return_value='BUY. Confidence: 0.80.') as mock_primary:
                result = _mli._generate_base_model('test prompt')

        mock_primary.assert_called_once()
        assert 'BUY' in result

    def test_direct_gpu_backend_falls_back_to_primary(self, monkeypatch):
        """Direct GPU path always falls back — second model would exhaust memory."""
        sys.modules.pop('model_inference_lora', None)
        with patch.dict('os.environ', {'INFERENCE_BACKEND': 'direct'}):
            import model_inference_lora as _mli
            with patch.object(_mli, 'get_trading_decision',
                               return_value='HOLD. Confidence: 0.55.') as mock_primary:
                result = _mli._generate_base_model('test prompt')

        mock_primary.assert_called_once()
