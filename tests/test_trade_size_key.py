"""Trade-log size-key compatibility between writers and the outcome reader.

Two writers feed logs/trade_log.jsonl. The agents write 'shares'; exit_reconciler
— which records broker-side bracket/OCO exits — wrote 'quantity', the options-log
key, into the stock log. outcome_tracker only read 'shares', so 342 of 534 stock
sell rows produced a size of 0.

Size feeds `realized_pnl = (exit - entry) * shares`, so a 0 collapsed
realized_pnl to 0 and therefore `won = realized_pnl > 0` to False — while
pnl_pct, computed from prices alone, stayed correct. 181 profitable trades were
recorded as losses, biased strictly one way. That is a corpus that teaches a
model profitable setups fail.

Fixed on both sides: the reader accepts either key (repairing rows already on
disk), and the writer now uses the right key per log.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'analysis'))

import outcome_tracker as ot  # noqa: E402


class TestReaderAcceptsEitherKey:
    def test_shares_key(self):
        assert ot._row_qty({'shares': 7}) == 7

    def test_quantity_key(self):
        """The broker-exit row shape that used to read as 0."""
        assert ot._row_qty({'quantity': 7}) == 7

    def test_shares_wins_when_both_present(self):
        assert ot._row_qty({'shares': 5, 'quantity': 9}) == 5

    def test_falls_through_zero_shares_to_quantity(self):
        """A present-but-zero 'shares' must not mask a real 'quantity'."""
        assert ot._row_qty({'shares': 0, 'quantity': 9}) == 9

    @pytest.mark.parametrize('row', [{}, {'shares': None}, {'quantity': None},
                                     {'shares': 'abc'}, {'shares': 0}, {'quantity': 0}])
    def test_missing_or_unusable_size_is_zero(self, row):
        assert ot._row_qty(row) == 0.0

    def test_string_numerics_are_accepted(self):
        assert ot._row_qty({'quantity': '3.0'}) == 3.0


class TestWonFlagIsCorrectForBrokerExits:
    """The end-to-end consequence: a profitable broker exit must read as a win."""

    def _pair(self, sell_row):
        entry = {'symbol': 'AMD', 'action': 'buy', 'shares': 10,
                 'order_id': 'b1', 'timestamp': '2026-06-01T10:00:00',
                 'source': 'paper', 'confidence': 0.8, 'reasoning': 'x'}
        return [entry, {'symbol': 'AMD', 'action': 'sell', 'order_id': 's1',
                        'timestamp': '2026-06-03T15:00:00', 'source': 'paper',
                        **sell_row}]

    def _run(self, trades, monkeypatch):
        tracker = ot.OutcomeTracker.__new__(ot.OutcomeTracker)   # no network in __init__
        tracker.paper = True
        tracker._bot_name = 'stock'
        prices = {'b1': 100.0, 's1': 110.0}
        tracker.get_fill_price = lambda oid: prices.get(oid)
        tracker.get_order_status = lambda oid: {
            'status': 'filled', 'filled_qty': 10, 'avg_price': prices.get(oid)}
        monkeypatch.setattr(ot, '_fetch_spy_return', lambda *a, **k: None)
        return tracker.match_and_calculate_pnl(trades)

    def test_quantity_style_exit_is_scored_as_a_win(self, monkeypatch):
        outcomes = self._run(self._pair({'quantity': 10}), monkeypatch)
        assert len(outcomes) == 1
        o = outcomes[0]
        assert o['shares'] == 10, 'broker-exit size must be read from quantity'
        assert o['realized_pnl'] == pytest.approx(100.0)
        assert o['won'] is True, \
            'a profitable exit recorded via quantity must not be labelled a loss'

    def test_shares_style_exit_still_works(self, monkeypatch):
        outcomes = self._run(self._pair({'shares': 10}), monkeypatch)
        assert outcomes[0]['won'] is True
        assert outcomes[0]['realized_pnl'] == pytest.approx(100.0)

    def test_pnl_pct_and_won_agree_in_sign(self, monkeypatch):
        """The specific inconsistency seen in production: pnl_pct > 0, won = 0."""
        for sell in ({'quantity': 10}, {'shares': 10}):
            o = self._run(self._pair(sell), monkeypatch)[0]
            assert o['pnl_pct'] > 0 and o['won'] is True, \
                f'pnl_pct and won disagree for sell row {sell}'


class TestWriterUsesTheRightKeyPerLog:
    def test_stock_log_gets_shares_and_options_log_gets_quantity(self):
        """Guard the writer side so the mismatch cannot be reintroduced."""
        src = (Path(__file__).resolve().parent.parent
               / 'scripts' / 'exit_reconciler.py').read_text()
        assert "('quantity' if bot == 'options' else 'shares'): filled_qty" in src, \
            ("exit_reconciler must write 'shares' for the stock log; writing "
             "'quantity' there is what zeroed 342 sell rows")
