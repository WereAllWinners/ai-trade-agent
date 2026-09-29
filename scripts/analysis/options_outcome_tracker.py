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
from analysis.order_status_cache import OrderStatusCache
import db as _db

# Reuse the SPY benchmark helper from outcome_tracker to avoid duplication
from outcome_tracker import _fetch_spy_return

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
        self.paper = paper
        self.trading_client = TradingClient(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            paper=paper
        )
        # sprint03 E2.1: mirror outcome_tracker.py's retry wrapper — same
        # unretried-lookup bug shape, same fix.
        from utils.alpaca_retry import retry_on_rate_limit
        for _m in ('submit_order', 'get_account', 'get_all_positions', 'get_orders', 'get_order_by_id'):
            if hasattr(self.trading_client, _m):
                method = getattr(self.trading_client, _m)
                if not hasattr(method, '_mock_name'):
                    setattr(self.trading_client, _m, retry_on_rate_limit(method))
        self.trade_log_path = Path('logs/options_trade_log.jsonl')
        self.outcomes_path = Path('logs/options_trade_outcomes.jsonl')
        Path('logs').mkdir(exist_ok=True)
        # Same uncached-lookup stall as outcome_tracker.py — one Alpaca call per
        # trade-log row on every run. Shared cache implementation so the two
        # trackers cannot drift apart.
        from service_suffix import suffixed_path
        self._status_cache = OrderStatusCache(
            suffixed_path(Path('logs') / f'order_status_cache_{self._bot_name}.json')
        )

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
        """Return status, filled_qty, and avg_price for an order.

        Cached for terminal orders — see analysis/order_status_cache.py.
        """
        if not order_id:
            return {'status': 'unknown', 'filled_qty': 0, 'avg_price': None}
        return self._status_cache.get_or_fetch(order_id, lambda: self._fetch_order_status(order_id))

    def _fetch_order_status(self, order_id) -> dict:
        """Uncached single lookup against Alpaca."""
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
        # sprint02 D4.3: group by (contract, source), not contract alone — same
        # paper/live commingling fix as outcome_tracker.py (see its comment
        # for full rationale). Legacy rows without a source tag bucket under
        # source=None, preserving FIFO matching among historical pairs.
        by_contract = defaultdict(list)
        for t in trades:
            contract = t.get('contract')
            if contract:
                by_contract[(contract, t.get('source'))].append(t)

        outcomes = []

        for (contract, _bucket_source), contract_trades in by_contract.items():
            contract_trades.sort(key=lambda x: x['timestamp'])
            open_buys = []  # FIFO queue of enriched buy records

            for trade in contract_trades:
                action = trade.get('action', '')

                if action == 'buy':
                    fill_price = self.get_fill_price(trade.get('order_id'))
                    open_buys.append({**trade, 'fill_price': fill_price})

                elif action == 'sell' and open_buys:
                    # sprint03 E2.2: PEEK, don't pop yet — mirror
                    # outcome_tracker.py's fix (see its comment for rationale).
                    entry = open_buys[0]
                    exit_fill = self.get_fill_price(trade.get('order_id'))

                    entry_price = entry.get('fill_price')
                    exit_price = exit_fill

                    if not entry_price or not exit_price:
                        missing_side = 'missing_entry_fill' if not entry_price else 'missing_exit_fill'
                        missing_oid = entry.get('order_id') if not entry_price else trade.get('order_id')
                        oid_status = self.get_order_status(missing_oid)
                        try:
                            retry_count = _db.upsert_unreconciled_order({
                                'recorded_at': datetime.now().isoformat(),
                                'order_id':    missing_oid or '',
                                'symbol':      contract,
                                'status':      oid_status['status'],
                                'reason':      missing_side,
                            }, bot=self._bot_name, source='paper' if self.paper else 'live')
                        except Exception as db_err:
                            logging.debug("Could not write unreconciled order: %s", db_err)
                            retry_count = 0

                        _max_retries = int(os.getenv('UNRECONCILED_MAX_RETRIES', '10'))
                        if retry_count >= _max_retries:
                            logging.warning(
                                "⚠️  Unreconciled order %s for %s exceeded %d retries — "
                                "abandoning (status=%s reason=%s)",
                                missing_oid, contract, _max_retries, oid_status['status'], missing_side,
                            )
                            try:
                                _db.mark_unreconciled_order_abandoned(
                                    missing_oid or '', missing_side, bot=self._bot_name,
                                    source='paper' if self.paper else 'live')
                            except Exception as db_err:
                                logging.debug("Could not mark unreconciled order abandoned: %s", db_err)
                            open_buys.pop(0)  # only NOW does the buy leave the FIFO queue
                        else:
                            logging.warning(
                                "⚠️  Unreconciled order %s for %s — status=%s reason=%s "
                                "(retry %d/%d, will retry next pass)",
                                missing_oid, contract, oid_status['status'], missing_side,
                                retry_count, _max_retries,
                            )
                        continue

                    open_buys.pop(0)  # normal successful match — pop now
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

                    spy_ret = _fetch_spy_return(entry['timestamp'], trade['timestamp'])
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
                        'spy_return_pct':   round(spy_ret, 4) if spy_ret is not None else None,
                        'excess_return_pct': round(pnl_pct - spy_ret, 4) if spy_ret is not None else None,
                        'regime':           entry.get('regime'),
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
        try:
            all_outcomes = self.match_and_calculate_pnl(trades)
        finally:
            # Persist resolved lookups even on failure so a cold cache warms up
            # across runs rather than restarting from scratch each time.
            self._status_cache.flush()
            logging.info("%s", self._status_cache.summary())

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
