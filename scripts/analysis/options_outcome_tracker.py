#!/usr/bin/env python3
"""
Options Trade Outcome Tracker
Enriches the raw options trade log with real Alpaca fill prices, matches
buy/sell pairs (FIFO) by contract symbol, and writes realized P&L to
logs/options_trade_outcomes.jsonl.

Run this before the nightly options performance analysis so analyzers have
accurate data for the go-live readiness gate.

Key difference from stock outcome_tracker: each options contract represents
100 shares, so realized_pnl = (exit_price - entry_price) * contracts * 100.
"""
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _pathfix  # noqa: F401

from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
import db as _db

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Options contracts represent 100 shares each
CONTRACT_MULTIPLIER = 100


class OptionsOutcomeTracker:
    _bot_name = 'options'

    def __init__(self, paper=None):
        if paper is None:
            paper = os.getenv('PAPER_TRADING', 'true').lower() != 'false'
        self.trading_client = TradingClient(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            paper=paper
        )
        self.trade_log_path = Path('logs/options_trade_log.jsonl')
        self.outcomes_path = Path('logs/options_trade_outcomes.jsonl')
        Path('logs').mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_raw_trades(self):
        """Load all entries from the options trade log."""
        trades = []
        if not self.trade_log_path.exists():
            logging.warning("No options trade log found at %s", self.trade_log_path)
            return trades

        with open(self.trade_log_path) as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    trades.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logging.warning("Skipping malformed line %d in options trade log: %s", i, e)

        logging.info("Loaded %d raw options trades", len(trades))
        return trades

    def load_already_tracked_ids(self):
        """Return set of buy_order_ids already written to options outcomes log."""
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

    def get_order_status(self, order_id) -> dict:
        """Return status, filled_qty, and avg_price for an order."""
        if not order_id:
            return {'status': 'unknown', 'filled_qty': 0, 'avg_price': None}
        try:
            order = self.trading_client.get_order_by_id(order_id)
            return {
                'status':     order.status.value if hasattr(order.status, 'value') else str(order.status),
                'filled_qty': float(order.filled_qty) if order.filled_qty else 0,
                'avg_price':  float(order.filled_avg_price) if order.filled_avg_price else None,
            }
        except Exception as e:
            logging.debug("Could not fetch order status for %s: %s", order_id, e)
            return {'status': 'unknown', 'filled_qty': 0, 'avg_price': None}

    def get_fill_price(self, order_id):
        """Thin wrapper kept for backwards compatibility."""
        return self.get_order_status(order_id)['avg_price']

    # ------------------------------------------------------------------
    # Pair matching and P&L calculation
    # ------------------------------------------------------------------

    def match_and_calculate_pnl(self, trades):
        """
        Match buy/sell pairs per contract symbol (FIFO) and compute realized P&L.

        Options use the 'contract' field as the position key (e.g. AAPL240119C00185000).
        P&L accounts for the 100x contract multiplier.

        Returns a list of closed-trade outcome dicts.
        """
        # Group by contract symbol, sorted by time
        by_contract = defaultdict(list)
        for t in trades:
            contract = t.get('contract')
            if contract:
                by_contract[contract].append(t)

        outcomes = []

        for contract, contract_trades in by_contract.items():
            contract_trades.sort(key=lambda x: x['timestamp'])
            open_buys = []  # FIFO queue of enriched buy records

            for trade in contract_trades:
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
                        missing_side = 'missing_entry_fill' if not entry_price else 'missing_exit_fill'
                        missing_oid = entry.get('order_id') if not entry_price else trade.get('order_id')
                        oid_status = self.get_order_status(missing_oid)
                        logging.warning(
                            "⚠️  Unreconciled order %s for %s — status=%s reason=%s",
                            missing_oid, contract, oid_status['status'], missing_side
                        )
                        try:
                            _db.insert_unreconciled_order({
                                'recorded_at': datetime.now().isoformat(),
                                'order_id':    missing_oid or '',
                                'symbol':      contract,
                                'status':      oid_status['status'],
                                'reason':      missing_side,
                            }, bot=self._bot_name)
                        except Exception as db_err:
                            logging.debug("Could not write unreconciled order: %s", db_err)
                        continue

                    contracts = min(
                        entry.get('quantity', 0),
                        trade.get('quantity', 0)
                    )
                    # Each contract = 100 shares of the underlying
                    realized_pnl = (exit_price - entry_price) * contracts * CONTRACT_MULTIPLIER
                    pnl_pct = (exit_price - entry_price) / entry_price

                    entry_dt = datetime.fromisoformat(entry['timestamp'])
                    exit_dt = datetime.fromisoformat(trade['timestamp'])
                    hold_hours = (exit_dt - entry_dt).total_seconds() / 3600

                    outcomes.append({
                        'symbol': contract,           # contract symbol (DB uses 'symbol' column)
                        'source': 'paper' if self.paper else 'live',
                        'underlying': entry.get('underlying', ''),
                        'option_type': entry.get('type', ''),
                        'strike': entry.get('strike'),
                        'expiration': entry.get('expiration', ''),
                        'buy_order_id': entry.get('order_id'),
                        'sell_order_id': trade.get('order_id'),
                        'entry_timestamp': entry['timestamp'],
                        'exit_timestamp': trade['timestamp'],
                        'entry_price': round(entry_price, 4),
                        'exit_price': round(exit_price, 4),
                        'shares': contracts,          # contracts (DB 'shares' column)
                        'realized_pnl': round(realized_pnl, 2),
                        'pnl_pct': round(pnl_pct, 4),
                        'hold_hours': round(hold_hours, 1),
                        'entry_confidence': entry.get('confidence'),
                        'entry_reasoning': entry.get('reasoning', ''),
                        'exit_reason': trade.get('reason', ''),
                        'won': realized_pnl > 0,
                    })

        return outcomes

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self):
        """Load options trades, match pairs, write new outcomes to disk."""
        trades = self.load_raw_trades()
        if not trades:
            logging.info("No options trades to process — done.")
            return

        already_tracked = self.load_already_tracked_ids()
        all_outcomes = self.match_and_calculate_pnl(trades)

        new_outcomes = [
            o for o in all_outcomes
            if o.get('buy_order_id') not in already_tracked
        ]

        if not new_outcomes:
            logging.info("No new closed options outcomes to record.")
            return

        with open(self.outcomes_path, 'a') as f:
            for outcome in new_outcomes:
                f.write(json.dumps(outcome) + '\n')
        for outcome in new_outcomes:
            try:
                _db.insert_outcome(outcome, bot='options', source='paper' if self.paper else 'live')
            except Exception as e:
                logging.warning("Could not write options outcome to DB: %s", e)

        wins = [o for o in new_outcomes if o['won']]
        losses = [o for o in new_outcomes if not o['won']]
        total_pnl = sum(o['realized_pnl'] for o in new_outcomes)
        win_rate = len(wins) / len(new_outcomes) if new_outcomes else 0

        logging.info(
            "Recorded %d new options outcomes: %dW / %dL — Win rate: %.1f%% — P&L: $%+.2f",
            len(new_outcomes), len(wins), len(losses), win_rate * 100, total_pnl
        )


if __name__ == "__main__":
    tracker = OptionsOutcomeTracker()
    tracker.run()
