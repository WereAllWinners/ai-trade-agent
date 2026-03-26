"""
fee_simulator.py — Simulates real Alpaca live-trading costs.

Even though paper trading is commission-free, we model the fees that will apply
in live trading so the rotation logic accounts for true opportunity cost:

  - SEC Section 31 fee  (sells only): $27.80 per $1M of principal
  - FINRA TAF           (sells only): $0.000166/share, capped at $8.30/trade
  - Bid-ask spread      (both sides): estimated from a liquidity tier based on
                                      average daily volume

Commissions: $0 (Alpaca is commission-free for equities).
"""

import logging

# ── Regulatory fee constants (current as of 2025) ────────────────────────────
_SEC_FEE_RATE   = 27.80 / 1_000_000   # 0.0000278 per $ of sale proceeds
_FINRA_TAF_RATE = 0.000166             # per share sold
_FINRA_TAF_CAP  = 8.30                 # max per trade

# ── Bid-ask spread estimates (one-way, applied to both buy and sell sides) ────
# These are conservative floor estimates for each liquidity tier.
# Liquid   : avg daily volume ≥ 1 M shares  (S&P 500, NASDAQ 100 names)
# Illiquid : avg daily volume <  1 M shares
_SPREAD_LIQUID   = 0.0005   # 0.05% one-way  →  0.10% round-trip
_SPREAD_ILLIQUID = 0.0010   # 0.10% one-way  →  0.20% round-trip


class FeeSimulator:
    """
    Estimate the all-in cost of a trade or a rotation (sell + buy pair).

    Parameters
    ----------
    paper : bool
        When True the simulator is in paper-trading mode. Fees are still
        calculated (so the rotation logic behaves identically), but log
        messages note that no fees are actually charged yet.
    """

    def __init__(self, paper: bool = True):
        self.paper = paper
        mode = "PAPER (fees simulated, not charged)" if paper else "LIVE (fees will be charged)"
        logging.debug(f"FeeSimulator initialised — {mode}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def estimate_sell_fees(
        self,
        price: float,
        shares: int,
        liquid: bool = True,
    ) -> dict:
        """
        Estimate costs for selling `shares` at `price`.

        Returns
        -------
        dict with keys:
          sec_fee       – SEC Section 31 fee ($)
          finra_taf     – FINRA Trading Activity Fee ($)
          spread_cost   – estimated slippage on the sell side ($)
          total         – sum of all costs ($)
          total_pct     – total as a fraction of trade value
        """
        trade_value  = price * shares
        sec_fee      = trade_value * _SEC_FEE_RATE
        finra_taf    = min(shares * _FINRA_TAF_RATE, _FINRA_TAF_CAP)
        spread_rate  = _SPREAD_LIQUID if liquid else _SPREAD_ILLIQUID
        spread_cost  = trade_value * spread_rate
        total        = sec_fee + finra_taf + spread_cost

        return {
            'sec_fee':    round(sec_fee, 6),
            'finra_taf':  round(finra_taf, 6),
            'spread_cost': round(spread_cost, 4),
            'total':       round(total, 4),
            'total_pct':   round(total / trade_value, 6) if trade_value else 0,
        }

    def estimate_buy_fees(
        self,
        price: float,
        shares: int,
        liquid: bool = True,
    ) -> dict:
        """
        Estimate costs for buying `shares` at `price`.

        Buys have no SEC fee or FINRA TAF — only spread slippage.

        Returns
        -------
        dict with keys:
          spread_cost   – estimated slippage on the buy side ($)
          total         – same as spread_cost ($)
          total_pct     – total as a fraction of trade value
        """
        trade_value = price * shares
        spread_rate = _SPREAD_LIQUID if liquid else _SPREAD_ILLIQUID
        spread_cost = trade_value * spread_rate
        return {
            'spread_cost': round(spread_cost, 4),
            'total':        round(spread_cost, 4),
            'total_pct':    round(spread_cost / trade_value, 6) if trade_value else 0,
        }

    def estimate_round_trip_cost(
        self,
        exit_price:   float,
        exit_shares:  int,
        entry_price:  float,
        entry_shares: int,
        liquid:       bool = True,
    ) -> dict:
        """
        Full round-trip cost: selling an existing position + buying a new one.

        Returns
        -------
        dict with keys:
          exit_fees      – result of estimate_sell_fees
          entry_fees     – result of estimate_buy_fees
          total_cost     – combined dollar cost ($)
          total_cost_pct – total_cost as a fraction of the *exit* position value
          mode           – 'paper' or 'live'
        """
        exit_fees  = self.estimate_sell_fees(exit_price, exit_shares, liquid)
        entry_fees = self.estimate_buy_fees(entry_price, entry_shares, liquid)

        total_cost     = exit_fees['total'] + entry_fees['total']
        exit_value     = exit_price * exit_shares
        total_cost_pct = round(total_cost / exit_value, 6) if exit_value else 0

        if self.paper:
            logging.debug(
                f"[FeeSimulator/paper] round-trip est: exit ${exit_fees['total']:.4f} "
                f"+ entry ${entry_fees['total']:.4f} = ${total_cost:.4f} "
                f"({total_cost_pct:.4%} of exit value) — NOT charged in paper mode"
            )
        else:
            logging.debug(
                f"[FeeSimulator/live] round-trip est: exit ${exit_fees['total']:.4f} "
                f"+ entry ${entry_fees['total']:.4f} = ${total_cost:.4f} "
                f"({total_cost_pct:.4%} of exit value)"
            )

        return {
            'exit_fees':       exit_fees,
            'entry_fees':      entry_fees,
            'total_cost':      round(total_cost, 4),
            'total_cost_pct':  total_cost_pct,
            'mode':            'paper' if self.paper else 'live',
        }
