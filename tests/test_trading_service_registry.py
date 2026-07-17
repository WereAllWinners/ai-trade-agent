"""
tests/test_trading_service_registry.py — shared unit registry (R1 WI-2)

Covers:
  - service_state: active / inactive / not_found (exit-code-3-vs-4 split)
  - bot_positions: options (len>10) vs stock (len<=10) symbol filter
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'tools'))

from trading_service_registry import TRADING_UNITS, bot_positions, service_state  # noqa: E402


def _make_subprocess_result(returncode=0, stdout='', stderr=''):
    m = MagicMock()
    m.returncode = returncode
    m.stdout = stdout
    m.stderr = stderr
    return m


class TestServiceState:
    def test_active_unit(self):
        with patch('subprocess.run', return_value=_make_subprocess_result(0, 'active\n')):
            assert service_state('ai-trading-bot.service') == 'active'

    def test_installed_but_inactive_unit_exit_code_3(self):
        # Confirmed empirically: an installed-but-stopped/disabled unit
        # returns stdout 'inactive' with exit code 3.
        with patch('subprocess.run', return_value=_make_subprocess_result(3, 'inactive\n')):
            assert service_state('ai-options-bot-live.service') == 'inactive'

    def test_nonexistent_unit_exit_code_4_is_not_found(self):
        # Confirmed empirically: a unit with no installed unit file also
        # prints 'inactive' to stdout, but with exit code 4 -- this is the
        # signal that distinguishes "never installed" from "stopped."
        with patch('subprocess.run', return_value=_make_subprocess_result(4, 'inactive\n')):
            assert service_state('does-not-exist.service') == 'not_found'

    def test_subprocess_exception_is_not_found_not_a_crash(self):
        with patch('subprocess.run', side_effect=Exception('systemctl missing')):
            assert service_state('ai-trading-bot.service') == 'not_found'

    def test_no_sudo_in_invocation(self):
        """Unattended timer-triggered scripts have no TTY -- an accidental
        sudo requirement would hang forever on a password prompt with no
        visible error, exactly the silent-failure class this brief exists
        to fix."""
        with patch('subprocess.run', return_value=_make_subprocess_result(0, 'active\n')) as mock_run:
            service_state('ai-trading-bot.service')
        cmd = mock_run.call_args[0][0]
        assert 'sudo' not in cmd


class TestBotPositions:
    def _pos(self, symbol):
        p = MagicMock()
        p.symbol = symbol
        return p

    def test_options_filter_keeps_long_occ_symbols(self):
        positions = [self._pos('SPY250328P00560000'), self._pos('AAPL'), self._pos('NVDA')]
        result = bot_positions(positions, 'options')
        assert [p.symbol for p in result] == ['SPY250328P00560000']

    def test_stock_filter_keeps_short_symbols(self):
        positions = [self._pos('SPY250328P00560000'), self._pos('AAPL'), self._pos('NVDA')]
        result = bot_positions(positions, 'stock')
        assert sorted(p.symbol for p in result) == ['AAPL', 'NVDA']

    def test_empty_list(self):
        assert bot_positions([], 'options') == []
        assert bot_positions([], 'stock') == []


class TestRegistryContents:
    def test_four_expected_units(self):
        units = {u['unit'] for u in TRADING_UNITS}
        assert units == {
            'ai-trading-bot.service', 'ai-trading-bot-live.service',
            'ai-options-bot.service', 'ai-options-bot-live.service',
        }

    def test_each_unit_has_bot_and_paper_fields(self):
        for u in TRADING_UNITS:
            assert u['bot'] in ('stock', 'options')
            assert isinstance(u['paper'], bool)
