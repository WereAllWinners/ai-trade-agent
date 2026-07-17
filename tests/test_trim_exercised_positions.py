"""
tests/test_trim_exercised_positions.py — WI-1 Variant B script

Covers:
  - compute_sell_plan: named-symbols-first, descending order, exact partial
    qty (no int() truncation), fallback to remaining positions ascending,
    shortfall <= 0 returns empty plan
  - main(): cancels open orders before submitting sells, zero-buffer default
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'tools'))

from trim_exercised_positions import compute_sell_plan, main  # noqa: E402


def _pos(symbol: str, qty: float, market_value: float) -> MagicMock:
    p = MagicMock()
    p.symbol = symbol
    p.qty = str(qty)
    p.market_value = str(market_value)
    return p


class TestComputeSellPlan:
    def test_zero_or_negative_shortfall_returns_empty(self):
        positions = [_pos('AAPL', 200, 66960.0)]
        assert compute_sell_plan(positions, ['AAPL'], 0) == []
        assert compute_sell_plan(positions, ['AAPL'], -100) == []

    def test_single_named_symbol_partial_sell_exact_qty(self):
        # NVDA: 400 shares, $82,052 mkt val -> $205.13/share. Shortfall
        # $58,736.66 -> exact fractional qty, no int() truncation.
        positions = [
            _pos('NVDA', 400, 82052.0),
            _pos('AAPL', 200, 66960.0),
            _pos('MSFT', 100, 39857.0),
        ]
        plan = compute_sell_plan(positions, ['AAPL', 'NVDA', 'MSFT'], 58736.66)
        assert len(plan) == 1
        p, qty, proceeds = plan[0]
        assert p.symbol == 'NVDA'  # largest of the three named symbols first
        assert abs(proceeds - 58736.66) < 0.01
        assert qty < 400  # partial, not the whole position
        assert round(qty, 6) not in (round(qty), )  # genuinely fractional, not truncated to an int boundary

    def test_named_symbols_sorted_descending_by_value(self):
        positions = [
            _pos('MSFT', 100, 39857.0),
            _pos('NVDA', 400, 82052.0),
            _pos('AAPL', 200, 66960.0),
        ]
        # Shortfall bigger than MSFT alone but smaller than NVDA alone ->
        # descending order means NVDA (largest) is chosen alone, not MSFT+AAPL.
        plan = compute_sell_plan(positions, ['AAPL', 'NVDA', 'MSFT'], 50000.0)
        assert len(plan) == 1
        assert plan[0][0].symbol == 'NVDA'

    def test_named_symbols_insufficient_falls_back_ascending(self):
        positions = [
            _pos('NVDA', 400, 10000.0),   # named, small
            _pos('SMALL', 10, 5000.0),    # fallback candidate, smaller
            _pos('BIG', 10, 200000.0),    # fallback candidate, larger
        ]
        # Named (NVDA) alone doesn't cover a $50,000 shortfall -> falls back
        # to remaining positions ascending by value: SMALL before BIG.
        plan = compute_sell_plan(positions, ['NVDA'], 50000.0)
        symbols_in_order = [p.symbol for p, _, _ in plan]
        assert symbols_in_order[0] == 'NVDA'
        assert 'SMALL' in symbols_in_order
        assert symbols_in_order.index('SMALL') < (
            symbols_in_order.index('BIG') if 'BIG' in symbols_in_order else len(symbols_in_order)
        )

    def test_full_sell_when_position_value_below_remaining_need(self):
        # NVDA is the larger of the two named symbols, so descending sort
        # processes it first: fully sold (10000 < 50000 need), AAPL (smaller,
        # processed second) partially sold for the rest.
        positions = [_pos('NVDA', 400, 10000.0), _pos('AAPL', 200, 5000.0)]
        plan = compute_sell_plan(positions, ['NVDA', 'AAPL'], 12000.0)
        nvda_row = next(r for r in plan if r[0].symbol == 'NVDA')
        assert nvda_row[1] == 400.0  # full qty, not partial
        assert abs(nvda_row[2] - 10000.0) < 0.01
        aapl_row = next(r for r in plan if r[0].symbol == 'AAPL')
        assert aapl_row[1] < 200.0  # partial
        assert abs(aapl_row[2] - 2000.0) < 0.01


class TestMainExecution:
    def _run_main(self, argv, account_cash, account_equity, positions, submit_side_effect=None):
        mock_account = MagicMock(cash=str(account_cash), equity=str(account_equity))
        mock_client = MagicMock()
        mock_client.get_account.return_value = mock_account
        mock_client.get_all_positions.return_value = positions
        if submit_side_effect is not None:
            mock_client.submit_order.side_effect = submit_side_effect
        else:
            mock_client.submit_order.return_value = MagicMock(id='order-1')
        mock_order_status = MagicMock(
            symbol='NVDA', status=MagicMock(value='filled'),
            filled_qty='286.045875', filled_avg_price='205.33',
        )
        mock_client.get_order_by_id.return_value = mock_order_status

        with patch('trim_exercised_positions._build_client', return_value=mock_client), \
             patch('sys.argv', argv), \
             patch('time.sleep'):
            try:
                main()
            except SystemExit:
                pass
        return mock_client

    def test_execute_cancels_orders_before_submitting_sells(self):
        positions = [_pos('NVDA', 400, 82052.0)]
        client = self._run_main(
            ['trim_exercised_positions.py', '--paper', '--execute'],
            account_cash=-58736.66, account_equity=204832.91, positions=positions,
        )
        assert client.cancel_orders.called
        assert client.submit_order.called
        # cancel_orders must happen before submit_order
        cancel_call_index = list(client.method_calls).index(next(
            c for c in client.method_calls if c[0] == 'cancel_orders'
        ))
        submit_call_index = list(client.method_calls).index(next(
            c for c in client.method_calls if c[0] == 'submit_order'
        ))
        assert cancel_call_index < submit_call_index

    def test_preview_mode_does_not_cancel_or_submit(self):
        positions = [_pos('NVDA', 400, 82052.0)]
        client = self._run_main(
            ['trim_exercised_positions.py', '--paper'],
            account_cash=-58736.66, account_equity=204832.91, positions=positions,
        )
        assert not client.cancel_orders.called
        assert not client.submit_order.called

    def test_default_target_buffer_is_zero(self):
        # cash=-100, no --target-buffer flag -> shortfall must be exactly
        # $100 (price $205.13/sh -> qty ~0.4875), NOT inflated by a hidden
        # buffer like cover_cash_deficit.py's old hardcoded $500 (which would
        # imply a $600 shortfall, qty ~2.925 -- an order of magnitude off).
        positions = [_pos('NVDA', 400, 82052.0)]
        client = self._run_main(
            ['trim_exercised_positions.py', '--paper', '--execute'],
            account_cash=-100.0, account_equity=100000.0, positions=positions,
        )
        submitted_qty = float(client.submit_order.call_args[0][0].qty)
        assert abs(submitted_qty - (100.0 / (82052.0 / 400))) < 1e-6
        assert submitted_qty < 1.0  # sanity: nowhere near the $500-buffer qty (~2.925)

    def test_already_at_or_above_target_does_nothing(self):
        positions = [_pos('NVDA', 400, 82052.0)]
        mock_account = MagicMock(cash='100.0', equity='100000.0')
        mock_client = MagicMock()
        mock_client.get_account.return_value = mock_account
        mock_client.get_all_positions.return_value = positions
        with patch('trim_exercised_positions._build_client', return_value=mock_client), \
             patch('sys.argv', ['trim_exercised_positions.py', '--paper']), \
             patch('time.sleep'):
            try:
                main()
            except SystemExit as e:
                assert e.code == 0
        assert not mock_client.cancel_orders.called
        assert not mock_client.submit_order.called
