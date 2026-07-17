"""
tests/test_pre_stop_check.py — R1 WI-9a

Covers:
  - no positions/orders -> safe regardless of watchdog
  - positions/orders held + watchdog active -> safe
  - positions/orders held + watchdog NOT active -> unsafe
  - main(): unknown unit, missing credentials, exit codes
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'tools'))

from pre_stop_check import check_custody_state, main  # noqa: E402


def _pos(symbol):
    p = MagicMock()
    p.symbol = symbol
    return p


class TestCheckCustodyState:
    def test_no_positions_or_orders_safe_regardless_of_watchdog(self):
        client = MagicMock()
        client.get_all_positions.return_value = []
        client.get_orders.return_value = []
        result = check_custody_state(client, 'options', watchdog_active=False)
        assert result.safe_to_stop is True
        assert result.info['position_count'] == 0

    def test_positions_held_watchdog_active_is_safe(self):
        client = MagicMock()
        client.get_all_positions.return_value = [_pos('SPY250328C00560000')]
        client.get_orders.return_value = []
        result = check_custody_state(client, 'options', watchdog_active=True)
        assert result.safe_to_stop is True
        assert result.info['position_count'] == 1

    def test_positions_held_watchdog_inactive_is_unsafe(self):
        client = MagicMock()
        client.get_all_positions.return_value = [_pos('SPY250328C00560000')]
        client.get_orders.return_value = []
        result = check_custody_state(client, 'options', watchdog_active=False)
        assert result.safe_to_stop is False
        assert 'UNSAFE' in result.reasons[0]

    def test_orders_only_no_positions_watchdog_inactive_is_unsafe(self):
        client = MagicMock()
        client.get_all_positions.return_value = []
        order = MagicMock()
        order.symbol = 'AAPL'
        client.get_orders.return_value = [order]
        result = check_custody_state(client, 'stock', watchdog_active=False)
        assert result.safe_to_stop is False

    def test_only_relevant_bot_instrument_class_counted(self):
        # A down OPTIONS check must not count STOCK positions.
        client = MagicMock()
        client.get_all_positions.return_value = [_pos('AAPL'), _pos('NVDA')]
        client.get_orders.return_value = []
        result = check_custody_state(client, 'options', watchdog_active=False)
        assert result.safe_to_stop is True  # nothing in the options class
        assert result.info['position_count'] == 0


class TestMain:
    def test_unknown_unit_exits_2(self):
        with patch('sys.argv', ['pre_stop_check.py', 'not-a-real-unit.service']):
            try:
                main()
                assert False, "expected SystemExit"
            except SystemExit as e:
                assert e.code == 2

    def test_no_args_exits_2(self):
        with patch('sys.argv', ['pre_stop_check.py']):
            try:
                main()
                assert False, "expected SystemExit"
            except SystemExit as e:
                assert e.code == 2

    def test_safe_case_exits_0(self):
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = []
        mock_client.get_orders.return_value = []
        with patch('sys.argv', ['pre_stop_check.py', 'ai-options-bot.service']), \
             patch('pre_stop_check._build_client', return_value=mock_client), \
             patch('pre_stop_check.service_state', return_value='active'):
            try:
                main()
                assert False, "expected SystemExit"
            except SystemExit as e:
                assert e.code == 0

    def test_unsafe_case_exits_1(self):
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = [_pos('SPY250328C00560000')]
        mock_client.get_orders.return_value = []
        with patch('sys.argv', ['pre_stop_check.py', 'ai-options-bot.service']), \
             patch('pre_stop_check._build_client', return_value=mock_client), \
             patch('pre_stop_check.service_state', return_value='inactive'):
            try:
                main()
                assert False, "expected SystemExit"
            except SystemExit as e:
                assert e.code == 1

    def test_missing_credentials_exits_2(self):
        with patch('sys.argv', ['pre_stop_check.py', 'ai-options-bot.service']), \
             patch('pre_stop_check._build_client', return_value=None):
            try:
                main()
                assert False, "expected SystemExit"
            except SystemExit as e:
                assert e.code == 2
