#!/usr/bin/env python3
"""
Trade Outcome Tracker
Enriches the raw trade log with real Alpaca fill prices, matches buy/sell
pairs (FIFO), and writes realized P&L to logs/trade_outcomes.jsonl.

Run this before the nightly performance analysis so analyzers have full data.
"""
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
import db as _db

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class OutcomeTracker:
    def __init__(self, paper=None):
        if paper is None:
            paper = os.getenv('PAPER_TRADING', 'true').lower() != 'false'
        self.trading_client = TradingClient(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            paper=paper
        )
        self.trade_log_path = Path('logs/trade_log.jsonl')
        self.outcomes_path = Path('logs/trade_outcomes.jsonl')
        Path('logs').mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_raw_trades(self):
        """Load all entries from the trade log."""
        trades = []
        if not self.trade_log_path.exists():
            logging.warning("No trade log found at %s", self.trade_log_path)
            return trades

        with open(self.trade_log_path) as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    trades.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logging.warning("Skipping malformed line %d in trade log: %s", i, e)

        logging.info("Loaded %d raw trades", len(trades))
        return trades

    def load_already_tracked_ids(self):
        """Return set of buy_order_ids already written to outcomes log."""
        tracked = set()
        if not self.outcomes_path.exists():
            return tracked

        with open(self.outcomes_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    oid = record.get('buy_order_id')
                    if oid:
                        tracked.add(oid)
                except json.JSONDecodeError:
                    pass

        return tracked

    # ------------------------------------------------------------------
    # Alpaca data fetching
    # ------------------------------------------------------------------

    def get_fill_price(self, order_id):
        """Fetch average fill price for an order from Alpaca."""
        if not order_id:
            return None
        try:
            order = self.trading_client.get_order_by_id(order_id)
            if order.filled_avg_price:
                return float(order.filled_avg_price)
        except Exception as e:
            logging.debug("Could not fetch fill for order %s: %s", order_id, e)
        return None

    # ------------------------------------------------------------------
    # Pair matching and P&L calculation
    # ------------------------------------------------------------------

    def match_and_calculate_pnl(self, trades):
        """
        Match buy/sell pairs per symbol (FIFO order) and compute realized P&L.
        Returns a list of closed-trade outcome dicts.
        """
        # Group by symbol, sorted by time
        by_symbol = defaultdict(list)
        for t in trades:
            symbol = t.get('symbol')
            if symbol:
                by_symbol[symbol].append(t)

        outcomes = []

        for symbol, symbol_trades in by_symbol.items():
            symbol_trades.sort(key=lambda x: x['timestamp'])
            open_buys = []  # FIFO queue of enriched buy records

            for trade in symbol_trades:
                action = trade.get('action', '')

                if action == 'buy':
                    fill_price = self.get_fill_price(trade.get('order_id'))
                    open_buys.append({**trade, 'fill_price': fill_price})

                elif action == 'sell' and open_buys:
                    entry = open_buys.pop(0)
                    exit_fill = self.get_fill_price(trade.get('order_id'))

                    entry_price = entry.get('fill_price')
                    exit_price = exit_fill

                    if not entry_price or not exit_price:
                        logging.debug(
                            "Missing fill price for %s pair (entry=%s, exit=%s) — skipping P&L",
                            symbol, entry_price, exit_price
                        )
                        continue

                    shares = min(
                        entry.get('shares', 0),
                        trade.get('shares', 0)
                    )
                    realized_pnl = (exit_price - entry_price) * shares
                    pnl_pct = (exit_price - entry_price) / entry_price

                    entry_dt = datetime.fromisoformat(entry['timestamp'])
                    exit_dt = datetime.fromisoformat(trade['timestamp'])
                    hold_hours = (exit_dt - entry_dt).total_seconds() / 3600

                    outcomes.append({
                        'symbol': symbol,
                        'buy_order_id': entry.get('order_id'),
                        'sell_order_id': trade.get('order_id'),
                        'entry_timestamp': entry['timestamp'],
                        'exit_timestamp': trade['timestamp'],
                        'entry_price': round(entry_price, 4),
                        'exit_price': round(exit_price, 4),
                        'shares': shares,
                        'realized_pnl': round(realized_pnl, 2),
                        'pnl_pct': round(pnl_pct, 4),
                        'hold_hours': round(hold_hours, 1),
                        'entry_confidence': entry.get('confidence'),
                        'entry_reasoning': entry.get('reasoning', ''),
                        'won': realized_pnl > 0,
                    })

        return outcomes

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self):
        """Load trades, match pairs, write new outcomes to disk."""
        trades = self.load_raw_trades()
        if not trades:
            logging.info("No trades to process — done.")
            return

        already_tracked = self.load_already_tracked_ids()
        all_outcomes = self.match_and_calculate_pnl(trades)

        new_outcomes = [
            o for o in all_outcomes
            if o.get('buy_order_id') not in already_tracked
        ]

        if not new_outcomes:
            logging.info("No new closed trade outcomes to record.")
            return

        with open(self.outcomes_path, 'a') as f:
            for outcome in new_outcomes:
                f.write(json.dumps(outcome) + '\n')
        for outcome in new_outcomes:
            try:
                _db.insert_outcome(outcome)
            except Exception as e:
                logging.warning("Could not write outcome to DB: %s", e)

        wins = [o for o in new_outcomes if o['won']]
        losses = [o for o in new_outcomes if not o['won']]
        total_pnl = sum(o['realized_pnl'] for o in new_outcomes)
        win_rate = len(wins) / len(new_outcomes) if new_outcomes else 0

        logging.info(
            "Recorded %d new outcomes: %dW / %dL — Win rate: %.1f%% — P&L: $%+.2f",
            len(new_outcomes), len(wins), len(losses), win_rate * 100, total_pnl
        )


if __name__ == "__main__":
    tracker = OutcomeTracker()
    tracker.run()
