#!/usr/bin/env python3
"""
bracket_probe.py — Paper-submission bracket variant matrix.

Submits 5 bracket variants to the Alpaca PAPER account to determine which
construction the API accepts. Forces paper=True explicitly; can never touch live.

Run: venv/bin/python scripts/tools/bracket_probe.py

Hypotheses:
  (a) GTC+limit bracket is the problem → variant 2 or 3 accepts
  (b) Leg spacing / limit interaction → variant 4 or 5 accepts
  (c) Something else → all reject, verbatim errors tell us what

The accepted variant defines the evidence-based fix for the production buy path.
"""
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

# ---------------------------------------------------------------------------
# Always paper — hard-coded, not env-driven
# ---------------------------------------------------------------------------
_API_KEY    = os.getenv('ALPACA_API_KEY', '')
_API_SECRET = os.getenv('ALPACA_SECRET_KEY', '')
if not _API_KEY or not _API_SECRET:
    sys.exit('ERROR: ALPACA_API_KEY / ALPACA_SECRET_KEY not set in .env')

_trading = TradingClient(_API_KEY, _API_SECRET, paper=True)   # paper=True, hard-coded
_data    = StockHistoricalDataClient(_API_KEY, _API_SECRET)

SYMBOL      = 'AAPL'
QTY         = 1           # minimum whole-share qty
STOP_LOSS   = -0.07       # matches live params
TAKE_PROFIT =  0.15       # matches live params


def _current_price() -> float:
    """Fetch the latest ask (or mid) price from the paper data feed."""
    req = StockLatestQuoteRequest(symbol_or_symbols=SYMBOL)
    quote = _data.get_stock_latest_quote(req)[SYMBOL]
    ask = getattr(quote, 'ask_price', None)
    bid = getattr(quote, 'bid_price', None)
    if ask and ask > 0:
        return float(ask)
    if bid and ask:
        return round((float(bid) + float(ask)) / 2, 2)
    return float(bid or 0) or None


def _cancel(order_id: str) -> str:
    """Cancel an order and return 'cancelled' or the error."""
    try:
        _trading.cancel_order_by_id(order_id)
        return 'cancelled'
    except Exception as e:
        return f'cancel-failed: {e}'


def _submit(label: str, order) -> dict:
    """Submit an order, report result, cancel if accepted."""
    try:
        result = _trading.submit_order(order)
        order_id = str(result.id)
        cancel_status = _cancel(order_id)
        return {
            'label':   label,
            'status':  'ACCEPTED',
            'id':      order_id,
            'cancel':  cancel_status,
            'details': (
                f"  limit_price  = {getattr(order, 'limit_price', 'N/A')}\n"
                f"  stop_price   = {order.stop_loss.stop_price if order.stop_loss else 'N/A'}\n"
                f"  target_price = {order.take_profit.limit_price if order.take_profit else 'N/A'}\n"
                f"  order_id     = {order_id}\n"
                f"  cancel       = {cancel_status}"
            ),
        }
    except Exception as e:
        err = str(e)
        return {
            'label':   label,
            'status':  'REJECTED',
            'error':   err,
            'details': f"  error = {err}",
        }


def run_probe():
    price = _current_price()
    if not price:
        sys.exit(f'ERROR: could not fetch current price for {SYMBOL}')

    print(f'\n{"="*60}')
    print(f'BRACKET PROBE — {SYMBOL}  paper=True (hard-coded)')
    print(f'Current price: ${price:.2f}')
    print(f'Stop loss: {STOP_LOSS:.0%}   Take profit: {TAKE_PROFIT:.0%}')
    print(f'{"="*60}\n')

    # Leg math exactly matching autonomous_agent.py:695-696
    stop_price   = round(price * (1 + STOP_LOSS),   2)
    target_price = round(price * (1 + TAKE_PROFIT), 2)
    limit_price  = round(price, 2)

    print(f'Leg prices (production formula):')
    print(f'  limit_price  = ${limit_price:.2f}  (round(price, 2))')
    print(f'  stop_price   = ${stop_price:.2f}  (round(price * {1+STOP_LOSS:.2f}, 2))')
    print(f'  target_price = ${target_price:.2f}  (round(price * {1+TAKE_PROFIT:.2f}, 2))')
    print()

    # Variant 4 uses wider legs
    stop_wide   = round(price * 0.90, 2)
    target_wide = round(price * 1.20, 2)

    # Variant 5 uses a marketable limit (slightly above ask)
    limit_marketable = round(price * 1.001, 2)
    stop_mkt   = round(limit_marketable * (1 + STOP_LOSS),   2)
    target_mkt = round(limit_marketable * (1 + TAKE_PROFIT), 2)

    variants = []

    # 1. Exact current construction (baseline)
    variants.append((
        '1. GTC LimitOrder bracket (current production)',
        LimitOrderRequest(
            symbol=SYMBOL, qty=QTY, side=OrderSide.BUY,
            limit_price=limit_price,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            stop_loss=StopLossRequest(stop_price=stop_price),
            take_profit=TakeProfitRequest(limit_price=target_price),
        )
    ))
    time.sleep(0.3)

    # 2. Market bracket (MarketOrderRequest with bracket legs)
    try:
        mkt_order = MarketOrderRequest(
            symbol=SYMBOL, qty=QTY, side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            stop_loss=StopLossRequest(stop_price=stop_price),
            take_profit=TakeProfitRequest(limit_price=target_price),
        )
        variants.append(('2. DAY MarketOrder bracket', mkt_order))
    except Exception as e:
        variants.append(('2. DAY MarketOrder bracket', _BuildError(str(e))))
    time.sleep(0.3)

    # 3. DAY tif instead of GTC (same limit price)
    variants.append((
        '3. DAY LimitOrder bracket',
        LimitOrderRequest(
            symbol=SYMBOL, qty=QTY, side=OrderSide.BUY,
            limit_price=limit_price,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            stop_loss=StopLossRequest(stop_price=stop_price),
            take_profit=TakeProfitRequest(limit_price=target_price),
        )
    ))
    time.sleep(0.3)

    # 4. Wider legs (target 1.20×, stop 0.90×)
    variants.append((
        '4. GTC LimitOrder bracket (wider legs: stop=0.90x, target=1.20x)',
        LimitOrderRequest(
            symbol=SYMBOL, qty=QTY, side=OrderSide.BUY,
            limit_price=limit_price,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            stop_loss=StopLossRequest(stop_price=stop_wide),
            take_profit=TakeProfitRequest(limit_price=target_wide),
        )
    ))
    time.sleep(0.3)

    # 5. Marketable limit (limit_price slightly above ask, leg math off this price)
    variants.append((
        f'5. GTC LimitOrder bracket (marketable limit=${limit_marketable:.2f} = price×1.001)',
        LimitOrderRequest(
            symbol=SYMBOL, qty=QTY, side=OrderSide.BUY,
            limit_price=limit_marketable,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            stop_loss=StopLossRequest(stop_price=stop_mkt),
            take_profit=TakeProfitRequest(limit_price=target_mkt),
        )
    ))

    results = []
    for label, order in variants:
        if isinstance(order, _BuildError):
            results.append({'label': label, 'status': 'BUILD-ERROR',
                           'error': order.msg, 'details': f'  build error: {order.msg}'})
        else:
            r = _submit(label, order)
            results.append(r)
        time.sleep(0.5)

    # ---------------------------------------------------------------------------
    # Print table
    # ---------------------------------------------------------------------------
    print(f'\n{"="*60}')
    print('RESULTS TABLE')
    print(f'{"="*60}')
    for r in results:
        status_icon = '✅' if r['status'] == 'ACCEPTED' else '❌'
        print(f"\n{status_icon} {r['label']}")
        print(f"   Status: {r['status']}")
        print(r['details'])

    accepted = [r for r in results if r['status'] == 'ACCEPTED']
    rejected = [r for r in results if r['status'] != 'ACCEPTED']
    print(f'\n{"="*60}')
    print(f'SUMMARY: {len(accepted)} accepted, {len(rejected)} rejected')
    if accepted:
        print(f'First accepted: {accepted[0]["label"]}')
        print('→ That variant defines the evidence-based production fix.')
    else:
        print('→ ALL REJECTED — escalate with verbatim errors above.')
    print(f'{"="*60}\n')


class _BuildError:
    """Sentinel for orders that fail Pydantic construction before submission."""
    def __init__(self, msg):
        self.msg = msg


if __name__ == '__main__':
    run_probe()
