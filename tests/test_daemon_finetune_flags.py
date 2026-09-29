"""
tests/test_daemon_finetune_flags.py — sprint03 E1 weekend automation safety flags.

FINETUNE_ENABLED and STRATEGY_EVOLVER_ENABLED are read fresh (not cached) at the
top of run_finetuning()/run_strategy_evolver() in both daemons, so a flag flip
takes effect without a daemon restart. Both default to enabled ('true') — only
a literal 'false' disables.
"""
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))

import trading_daemon
import options_daemon


def _make_trading_daemon():
    return trading_daemon.TradingDaemon.__new__(trading_daemon.TradingDaemon)


def _make_options_daemon():
    return options_daemon.OptionsDaemon.__new__(options_daemon.OptionsDaemon)


class TestFinetuneEnabledFlagStockDaemon:
    def test_flag_false_skips_finetune_subprocess(self, monkeypatch, caplog):
        monkeypatch.setenv('FINETUNE_ENABLED', 'false')
        d = _make_trading_daemon()
        with patch.object(trading_daemon.TradingDaemon, 'run_market_research') as mock_research, \
             patch.object(trading_daemon.TradingDaemon, 'run_training_data_builder') as mock_build, \
             patch.object(trading_daemon.TradingDaemon, 'run_dpo_builder') as mock_dpo, \
             patch('trading_daemon.subprocess.run') as mock_subprocess, \
             caplog.at_level(logging.INFO):
            d.run_finetuning()
        mock_research.assert_not_called()
        mock_build.assert_not_called()
        mock_dpo.assert_not_called()
        mock_subprocess.assert_not_called()
        assert any('FINETUNE_ENABLED=false' in r.getMessage() for r in caplog.records)

    def test_flag_true_runs_unchanged(self, monkeypatch):
        monkeypatch.setenv('FINETUNE_ENABLED', 'true')
        d = _make_trading_daemon()
        with patch.object(trading_daemon.TradingDaemon, 'run_market_research') as mock_research, \
             patch.object(trading_daemon.TradingDaemon, 'run_training_data_builder') as mock_build, \
             patch.object(trading_daemon.TradingDaemon, 'run_dpo_builder') as mock_dpo, \
             patch('trading_daemon.inference_client') as mock_inference:
            mock_inference.stop_for_finetuning.return_value = False  # abort before subprocess launch
            d.run_finetuning()
        mock_research.assert_called_once()
        mock_build.assert_called_once()
        mock_dpo.assert_called_once()

    def test_flag_absent_defaults_to_enabled(self, monkeypatch):
        monkeypatch.delenv('FINETUNE_ENABLED', raising=False)
        d = _make_trading_daemon()
        with patch.object(trading_daemon.TradingDaemon, 'run_market_research') as mock_research, \
             patch.object(trading_daemon.TradingDaemon, 'run_training_data_builder') as mock_build, \
             patch.object(trading_daemon.TradingDaemon, 'run_dpo_builder') as mock_dpo, \
             patch('trading_daemon.inference_client') as mock_inference:
            mock_inference.stop_for_finetuning.return_value = False
            d.run_finetuning()
        mock_research.assert_called_once()
        mock_build.assert_called_once()
        mock_dpo.assert_called_once()

    def test_flag_false_blocks_sigusr1_offcycle_path_too(self, monkeypatch):
        """The SIGUSR1 handler calls this exact same run_finetuning() method —
        confirms the single insertion point covers both the scheduled and the
        manual off-cycle trigger, not just one."""
        monkeypatch.setenv('FINETUNE_ENABLED', 'false')
        d = _make_trading_daemon()
        with patch.object(trading_daemon.TradingDaemon, 'run_market_research') as mock_research:
            d.run_finetuning()
        mock_research.assert_not_called()


class TestFinetuneEnabledFlagOptionsDaemon:
    def test_flag_false_skips_finetune_subprocess(self, monkeypatch, caplog):
        monkeypatch.setenv('FINETUNE_ENABLED', 'false')
        d = _make_options_daemon()
        with patch.object(options_daemon.OptionsDaemon, 'run_market_research') as mock_research, \
             patch.object(options_daemon.OptionsDaemon, 'run_training_data_builder') as mock_build, \
             patch('options_daemon.subprocess.run') as mock_subprocess, \
             caplog.at_level(logging.INFO):
            d.run_finetuning()
        mock_research.assert_not_called()
        mock_build.assert_not_called()
        mock_subprocess.assert_not_called()
        assert any('FINETUNE_ENABLED=false' in r.getMessage() for r in caplog.records)

    def test_flag_true_runs_unchanged(self, monkeypatch):
        monkeypatch.setenv('FINETUNE_ENABLED', 'true')
        d = _make_options_daemon()
        with patch.object(options_daemon.OptionsDaemon, 'run_market_research') as mock_research, \
             patch.object(options_daemon.OptionsDaemon, 'run_training_data_builder') as mock_build, \
             patch('options_daemon.inference_client') as mock_inference:
            mock_inference.stop_for_finetuning.return_value = False
            d.run_finetuning()
        mock_research.assert_called_once()
        mock_build.assert_called_once()

    def test_flag_absent_defaults_to_enabled(self, monkeypatch):
        monkeypatch.delenv('FINETUNE_ENABLED', raising=False)
        d = _make_options_daemon()
        with patch.object(options_daemon.OptionsDaemon, 'run_market_research') as mock_research, \
             patch.object(options_daemon.OptionsDaemon, 'run_training_data_builder') as mock_build, \
             patch('options_daemon.inference_client') as mock_inference:
            mock_inference.stop_for_finetuning.return_value = False
            d.run_finetuning()
        mock_research.assert_called_once()
        mock_build.assert_called_once()

    def test_flag_false_blocks_sigusr1_offcycle_path_too(self, monkeypatch):
        monkeypatch.setenv('FINETUNE_ENABLED', 'false')
        d = _make_options_daemon()
        with patch.object(options_daemon.OptionsDaemon, 'run_market_research') as mock_research:
            d.run_finetuning()
        mock_research.assert_not_called()


class TestStrategyEvolverEnabledFlag:
    """trading_daemon.py only — options_daemon.py has no StrategyEvolver call."""

    def test_flag_false_skips_strategy_evolver_subprocess(self, monkeypatch, caplog):
        monkeypatch.setenv('STRATEGY_EVOLVER_ENABLED', 'false')
        d = _make_trading_daemon()
        with patch('trading_daemon.subprocess.run') as mock_subprocess, \
             caplog.at_level(logging.INFO):
            d.run_strategy_evolver()
        mock_subprocess.assert_not_called()
        assert any('STRATEGY_EVOLVER_ENABLED=false' in r.getMessage() for r in caplog.records)

    def test_flag_true_runs_unchanged(self, monkeypatch):
        monkeypatch.setenv('STRATEGY_EVOLVER_ENABLED', 'true')
        d = _make_trading_daemon()
        mock_result = MagicMock(returncode=0)
        with patch('trading_daemon.subprocess.run', return_value=mock_result) as mock_subprocess:
            d.run_strategy_evolver()
        mock_subprocess.assert_called_once()

    def test_flag_absent_defaults_to_enabled(self, monkeypatch):
        monkeypatch.delenv('STRATEGY_EVOLVER_ENABLED', raising=False)
        d = _make_trading_daemon()
        mock_result = MagicMock(returncode=0)
        with patch('trading_daemon.subprocess.run', return_value=mock_result) as mock_subprocess:
            d.run_strategy_evolver()
        mock_subprocess.assert_called_once()

    def test_weekend_strategist_unaffected_by_strategy_evolver_flag(self, monkeypatch):
        """run_weekend_strategist() is a separate method, called just before
        run_strategy_evolver() in the Saturday block — must keep running
        regardless of STRATEGY_EVOLVER_ENABLED."""
        monkeypatch.setenv('STRATEGY_EVOLVER_ENABLED', 'false')
        d = _make_trading_daemon()
        mock_result = MagicMock(returncode=0)
        with patch('trading_daemon.subprocess.run', return_value=mock_result) as mock_subprocess:
            d.run_weekend_strategist()
        mock_subprocess.assert_called_once()


class TestOnlineTrainingEnabledFlag:
    """ONLINE_TRAINING_ENABLED gates the threshold-triggered LoRA update.

    Added after run_online_training fired unprompted: it had no switch at all,
    so when the outcome-tracker fix landed a 3-month backlog of 337 outcomes at
    once, the 15-outcome threshold tripped with no way to hold it short of
    stopping the daemon.
    """

    def test_flag_false_skips_online_training(self, monkeypatch, caplog):
        monkeypatch.setenv('ONLINE_TRAINING_ENABLED', 'false')
        d = _make_trading_daemon()
        with patch('trading_daemon.subprocess.run') as mock_subprocess, \
             caplog.at_level(logging.INFO):
            d.run_online_training()
        mock_subprocess.assert_not_called()
        assert any('ONLINE_TRAINING_ENABLED=false' in r.getMessage() for r in caplog.records)

    def test_flag_true_runs_unchanged(self, monkeypatch):
        monkeypatch.setenv('ONLINE_TRAINING_ENABLED', 'true')
        d = _make_trading_daemon()
        with patch('trading_daemon.subprocess.run') as mock_subprocess:
            mock_subprocess.return_value = MagicMock(returncode=0)
            d.run_online_training()
        mock_subprocess.assert_called_once()

    def test_flag_absent_defaults_to_enabled(self, monkeypatch):
        monkeypatch.delenv('ONLINE_TRAINING_ENABLED', raising=False)
        d = _make_trading_daemon()
        with patch('trading_daemon.subprocess.run') as mock_subprocess:
            mock_subprocess.return_value = MagicMock(returncode=0)
            d.run_online_training()
        mock_subprocess.assert_called_once()

    def test_is_independent_of_finetune_enabled(self, monkeypatch):
        """The two switches must not be coupled: online training is the
        mechanism that replaces the nightly fine-tune when that is off."""
        monkeypatch.setenv('FINETUNE_ENABLED', 'false')
        monkeypatch.delenv('ONLINE_TRAINING_ENABLED', raising=False)
        d = _make_trading_daemon()
        with patch('trading_daemon.subprocess.run') as mock_subprocess:
            mock_subprocess.return_value = MagicMock(returncode=0)
            d.run_online_training()
        mock_subprocess.assert_called_once(), \
            'FINETUNE_ENABLED=false must not silently gate online training'
