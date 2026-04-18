"""
paper_market_simulator.py — Realistic fill simulation for paper trading.

Alpaca's paper trading fills market orders instantly at the last trade price
with no slippage, spread, or partial-fill risk.  This module intercepts order
submissions and injects the friction that will exist in live markets:

  1. Slippage / spread
       Buys fill above the reference price; sells fill below.
       Magnitude scales with order size relative to average daily volume (ADV)
       and with the stock's liquidity tier.

  2. Partial fills
       Large orders relative to current volume, or orders in thin stocks,
       may only partially fill.  Remaining shares are NOT resubmitted — the
       agent must re-evaluate on the next cycle, which trains it to size
       correctly.

  3. Non-fills (rare)
       A small probability of a complete non-fill, simulating a fast-moving
       market or a temporary halt.  Only triggered for illiquid stocks or when
       volume is very thin.

Only active when PAPER_TRADING=true.  When live, this module is a no-op.
"""

import logging
import math
import random
from dataclasses import dataclass
from typing import Literal

# ── Slippage model constants ──────────────────────────────────────────────────

# Base half-spread (one-way cost of crossing the spread)
_SPREAD_BPS_LIQUID   = 5    # 0.05%  — S&P 500 / high-cap NASDAQ names
_SPREAD_BPS_MID      = 12   # 0.12%  — mid-cap, ADV 500K–1M
_SPREAD_BPS_ILLIQUID = 25   # 0.25%  — small-cap, ADV < 500K

# Market-impact coefficient: extra bps per 1% of ADV ordered
# e.g. ordering 2% of ADV on a liquid stock → 2 * 3 = 6 extra bps
_IMPACT_BPS_PER_PCT_ADV_LIQUID   = 3
_IMPACT_BPS_PER_PCT_ADV_MID      = 8
_IMPACT_BPS_PER_PCT_ADV_ILLIQUID = 20

# ADV thresholds (shares/day) for liquidity tier classification
_ADV_LIQUID   = 1_000_000
_ADV_MID      = 500_000

# ── Partial-fill model constants ─────────────────────────────────────────────

# Probability of a partial fill given order size vs current session volume
# If order_qty / session_volume > threshold → apply partial fill probability
_PARTIAL_FILL_TRIGGER_PCT   = 0.02    # order > 2% of today's volume so far
_PARTIAL_FILL_PROBABILITY   = 0.45   # 45% chance of a partial fill in that case
_PARTIAL_FILL_MIN_FRACTION  = 0.50   # fill at least 50% of the order
_PARTIAL_FILL_MAX_FRACTION  = 0.85   # fill at most 85% of the order

# For illiquid stocks (ADV < 500K), partial-fill risk is higher
_ILLIQUID_PARTIAL_FILL_PROB = 0.65
_ILLIQUID_PARTIAL_FILL_MIN  = 0.35

# ── Non-fill (rejection) model ────────────────────────────────────────────────
_NONFILL_PROB_LIQUID   = 0.005   # 0.5% — essentially never on liquid names
_NONFILL_PROB_ILLIQUID = 0.025   # 2.5% — fast market / halt risk on thin names

# ── Options-specific constants ────────────────────────────────────────────────
# Options spreads are much wider than equities.
_OPTIONS_SPREAD_BPS_LIQUID   = 50    # 0.50% — high-OI chains (SPY, QQQ, AAPL)
_OPTIONS_SPREAD_BPS_ILLIQUID = 150   # 1.50% — low-OI contracts
_OPTIONS_ADV_LIQUID          = 500   # contracts/day threshold
_OPTIONS_PARTIAL_FILL_PROB   = 0.30
_OPTIONS_NONFILL_PROB        = 0.05


@dataclass
class SimulatedFill:
    """Result of a simulated fill."""
    fill_price:     float   # adjusted fill price (includes slippage)
    fill_qty:       int     # shares/contracts actually filled (may be < requested)
    requested_qty:  int     # original request
    partial:        bool    # True if fill_qty < requested_qty
    filled:         bool    # False = complete non-fill
    slippage_bps:   float   # total slippage applied in basis points
    note:           str     # human-readable description


class PaperMarketSimulator:
    """
    Wrap around order submission to inject realistic fill friction.

    Usage
    -----
    sim = PaperMarketSimulator(paper=True)

    # Before submitting an order:
    fill = sim.simulate_stock_fill(
        side='buy',
        price=142.50,
        qty=100,
        adv=8_500_000,        # average daily volume in shares
        session_volume=3_200_000,  # volume traded so far today
    )

    if not fill.filled:
        logging.warning(f"Order non-fill simulated — skipping trade")
        return False

    if fill.partial:
        qty = fill.fill_qty   # use reduced qty for order submission
        logging.info(f"Partial fill: {fill.fill_qty}/{fill.requested_qty} shares")

    # Submit a limit order at fill.fill_price with qty=fill.fill_qty
    """

    def __init__(self, paper: bool = True):
        self.paper = paper
        if paper:
            logging.info(
                "PaperMarketSimulator active — slippage, partial fills, "
                "and non-fill risk are enabled"
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    def simulate_stock_fill(
        self,
        side:           Literal['buy', 'sell'],
        price:          float,
        qty:            int,
        adv:            float,
        session_volume: float,
    ) -> SimulatedFill:
        """Simulate a stock market order fill."""
        if not self.paper:
            return SimulatedFill(price, qty, qty, False, True, 0.0, 'live mode — no simulation')

        tier     = self._stock_tier(adv)
        slip_bps = self._stock_slippage_bps(side, qty, price, adv, tier)
        fill_price = self._apply_slippage(side, price, slip_bps)

        # Non-fill check
        nonfill_prob = _NONFILL_PROB_ILLIQUID if tier == 'illiquid' else _NONFILL_PROB_LIQUID
        if random.random() < nonfill_prob:
            return SimulatedFill(
                fill_price, 0, qty, False, False, slip_bps,
                f"Non-fill simulated ({tier} stock, {nonfill_prob:.1%} probability)"
            )

        # Partial fill check
        fill_qty, partial, note = self._stock_partial_fill(qty, session_volume, tier)

        return SimulatedFill(fill_price, fill_qty, qty, partial, True, slip_bps, note)

    def simulate_options_fill(
        self,
        side:        Literal['buy', 'sell'],
        mid_price:   float,
        contracts:   int,
        open_interest: int,
        daily_volume:  int,
    ) -> SimulatedFill:
        """Simulate an options contract fill."""
        if not self.paper:
            return SimulatedFill(mid_price, contracts, contracts, False, True, 0.0, 'live mode — no simulation')

        liquid    = (open_interest >= _OPTIONS_ADV_LIQUID or daily_volume >= _OPTIONS_ADV_LIQUID)
        spread_bps = _OPTIONS_SPREAD_BPS_LIQUID if liquid else _OPTIONS_SPREAD_BPS_ILLIQUID
        fill_price = self._apply_slippage(side, mid_price, spread_bps)

        # Non-fill
        if random.random() < _OPTIONS_NONFILL_PROB:
            return SimulatedFill(
                fill_price, 0, contracts, False, False, spread_bps,
                f"Options non-fill simulated (5% probability)"
            )

        # Partial fill
        fill_qty = contracts
        partial  = False
        if random.random() < _OPTIONS_PARTIAL_FILL_PROB:
            fraction = random.uniform(0.40, 0.80)
            fill_qty = max(1, math.floor(contracts * fraction))
            partial  = fill_qty < contracts

        note = (
            f"{'Liquid' if liquid else 'Illiquid'} options — "
            f"spread {spread_bps}bps applied"
            + (f", partial fill {fill_qty}/{contracts} contracts" if partial else "")
        )
        return SimulatedFill(fill_price, fill_qty, contracts, partial, True, spread_bps, note)

    def log_fill(self, symbol: str, fill: SimulatedFill) -> None:
        """Log the simulation result at the appropriate level."""
        if not fill.filled:
            logging.warning(
                f"🚫 [PaperSim] {symbol} — ORDER NOT FILLED: {fill.note}"
            )
        elif fill.partial:
            logging.info(
                f"⚠️  [PaperSim] {symbol} — PARTIAL FILL: "
                f"{fill.fill_qty}/{fill.requested_qty} @ ${fill.fill_price:.4f} "
                f"(slippage {fill.slippage_bps:.1f}bps) — {fill.note}"
            )
        else:
            logging.info(
                f"📋 [PaperSim] {symbol} — FULL FILL: "
                f"{fill.fill_qty} @ ${fill.fill_price:.4f} "
                f"(slippage {fill.slippage_bps:.1f}bps)"
            )

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _stock_tier(adv: float) -> str:
        if adv >= _ADV_LIQUID:
            return 'liquid'
        if adv >= _ADV_MID:
            return 'mid'
        return 'illiquid'

    @staticmethod
    def _stock_slippage_bps(
        side: str, qty: int, price: float, adv: float, tier: str
    ) -> float:
        spread = {
            'liquid': _SPREAD_BPS_LIQUID,
            'mid':    _SPREAD_BPS_MID,
            'illiquid': _SPREAD_BPS_ILLIQUID,
        }[tier]
        impact_coeff = {
            'liquid':   _IMPACT_BPS_PER_PCT_ADV_LIQUID,
            'mid':      _IMPACT_BPS_PER_PCT_ADV_MID,
            'illiquid': _IMPACT_BPS_PER_PCT_ADV_ILLIQUID,
        }[tier]
        order_pct_adv = (qty / adv * 100) if adv > 0 else 0
        impact = order_pct_adv * impact_coeff
        # Add a small random component (±20% of total) to simulate intraday noise
        total   = spread + impact
        noise   = total * random.uniform(-0.20, 0.20)
        return max(1.0, total + noise)

    @staticmethod
    def _apply_slippage(side: str, price: float, bps: float) -> float:
        factor = bps / 10_000
        if side == 'buy':
            return round(price * (1 + factor), 4)
        else:
            return round(price * (1 - factor), 4)

    @staticmethod
    def _stock_partial_fill(
        qty: int, session_volume: float, tier: str
    ) -> tuple[int, bool, str]:
        """Return (fill_qty, is_partial, note)."""
        if session_volume > 0 and (qty / session_volume) > _PARTIAL_FILL_TRIGGER_PCT:
            prob = _ILLIQUID_PARTIAL_FILL_PROB if tier == 'illiquid' else _PARTIAL_FILL_PROBABILITY
            if random.random() < prob:
                lo = _ILLIQUID_PARTIAL_FILL_MIN if tier == 'illiquid' else _PARTIAL_FILL_MIN_FRACTION
                fraction = random.uniform(lo, _PARTIAL_FILL_MAX_FRACTION)
                fill_qty = max(1, math.floor(qty * fraction))
                return (
                    fill_qty, fill_qty < qty,
                    f"Partial fill {fill_qty}/{qty} — order was {qty/session_volume:.2%} of session volume"
                )
        return qty, False, "Full fill"
