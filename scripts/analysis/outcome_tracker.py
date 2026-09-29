#!/usr/bin/env python3
"""
Trade Outcome Tracker
Enriches the raw trade log with real Alpaca fill prices, matches buy/sell
pairs (FIFO), and writes realized P&L to logs/trade_outcomes.jsonl.

Run this before the nightly performance analysis so analyzers have full data.
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


def _fetch_spy_return(entry_ts: str, exit_ts: str) -> float | None:
    """Return SPY price-return (as a fraction) over the hold window.

    Used to populate spy_return_pct on closed outcomes so we can measure alpha.
    Returns None silently on any fetch failure.
    """
    try:
        import yfinance as yf
        entry_dt = datetime.fromisoformat(entry_ts).date()
        exit_dt  = datetime.fromisoformat(exit_ts).date()
        if entry_dt >= exit_dt:
            return None
        spy = yf.download(
            'SPY', start=str(entry_dt), end=str(exit_dt + __import__('datetime').timedelta(days=1)),
            auto_adjust=True, progress=False
        )['Close']
        if spy.empty or len(spy) < 2:
            return None
        return float((spy.iloc[-1] - spy.iloc[0]) / spy.iloc[0])
    except Exception as e:
        logging.debug("_fetch_spy_return failed (non-fatal): %s", e)
        return None

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class OutcomeTracker:
    _bot_name = 'stock'

    def __init__(self, paper=None):
        if paper is None:
            paper = os.getenv('PAPER_TRADING', 'true').lower() != 'false'
        self.paper = paper
        self.trading_client = TradingClient(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            paper=paper
        )
        # sprint03 E2.1: this client's get_order_by_id lookups (via
        # get_order_status below) were previously unretried — a transient
        # 429/5xx during the nightly pass would misclassify a genuinely-filled
        # order as unreconciled (see docs/sprint02-d7-findings.md). Mirror the
        # same retry wrapper autonomous_agent.py already applies.
        from utils.alpaca_retry import retry_on_rate_limit
        for _m in ('submit_order', 'get_account', 'get_all_positions', 'get_orders', 'get_order_by_id'):
            if hasattr(self.trading_client, _m):
                method = getattr(self.trading_client, _m)
                if not hasattr(method, '_mock_name'):
                    setattr(self.trading_client, _m, retry_on_rate_limit(method))
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
        Match buy/sell pairs per symbol (FIFO order) and compute realized P&L.
        Returns a list of closed-trade outcome dicts.
        """
        # sprint02 D4.3: group by (symbol, source) rather than symbol alone.
        # trade_log.jsonl is shared between the paper and live services with
        # no discriminator on rows written before this fix — confirmed
        # empirically that paper and live order_ids were commingled in the
        # same file. Grouping purely by symbol let a paper BUY get FIFO-
        # matched against a live SELL (or vice versa), producing realized P&L
        # that mixes two different accounts.
        #
        # Rows now carry a 'source' field going forward. Legacy rows without
        # one are bucketed under source=None rather than assumed to be
        # 'paper' — per the D4.3 caveat, we cannot know which account they
        # belonged to without redoing the manual order-id lookup, and
        # guessing risks silently mislabeling a live trade as paper. Bucketing
        # by None preserves FIFO matching among legacy rows exactly as before
        # (no regression for historical pairs where both sides are legacy),
        # while still preventing any *new* tagged row from cross-matching a
        # different source. A legacy BUY whose SELL arrives after this fix
        # ships will show as unmatched (missing_entry_fill) until D4.4's
        # targeted backfill stamps a source onto it.
        by_symbol_source = defaultdict(list)
        for t in trades:
            symbol = t.get('symbol')
            if symbol:
                by_symbol_source[(symbol, t.get('source'))].append(t)

        outcomes = []

        for (symbol, _bucket_source), symbol_trades in by_symbol_source.items():
            symbol_trades.sort(key=lambda x: x['timestamp'])
            open_buys = []  # FIFO queue of enriched buy records

            for trade in symbol_trades:
                action = trade.get('action', '')

                if action == 'buy':
                    fill_price = self.get_fill_price(trade.get('order_id'))
                    open_buys.append({**trade, 'fill_price': fill_price})

                elif action == 'sell' and open_buys:
                    # sprint03 E2.2: PEEK, don't pop yet — a transient
                    # fill-price lookup failure must not permanently destroy
                    # this buy/sell pair. Only pop once the pair is resolved
                    # (matched) or abandoned after too many retries.
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
                                'symbol':      symbol,
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
                                missing_oid, symbol, _max_retries, oid_status['status'], missing_side,
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
                                missing_oid, symbol, oid_status['status'], missing_side,
                                retry_count, _max_retries,
                            )
                        continue

                    open_buys.pop(0)  # normal successful match — pop now
                    shares = min(
                        entry.get('shares', 0),
                        trade.get('shares', 0)
                    )
                    realized_pnl = (exit_price - entry_price) * shares
                    pnl_pct = (exit_price - entry_price) / entry_price

                    entry_dt = datetime.fromisoformat(entry['timestamp'])
                    exit_dt = datetime.fromisoformat(trade['timestamp'])
                    hold_hours = (exit_dt - entry_dt).total_seconds() / 3600

                    spy_ret = _fetch_spy_return(entry['timestamp'], trade['timestamp'])
                    outcomes.append({
                        'symbol': symbol,
                        'source': 'paper' if self.paper else 'live',
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
                        'spy_return_pct':   round(spy_ret, 4) if spy_ret is not None else None,
                        'excess_return_pct': round(pnl_pct - spy_ret, 4) if spy_ret is not None else None,
                        'regime':           entry.get('regime'),
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
                _db.insert_outcome(outcome, bot='stock', source='paper' if self.paper else 'live')
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
