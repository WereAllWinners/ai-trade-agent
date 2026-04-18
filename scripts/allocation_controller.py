#!/usr/bin/env python3
"""
allocation_controller.py — Gradual position-size ramp based on live performance.

The controller gates how large any single position can be as a fraction of
equity, based on the recent track record stored in the trade database.  It
prevents the agent from deploying full risk capital before it has demonstrated
stable performance.

Tiers
-----
  Tier 1 (default / cold-start)  — 1 % of equity per position
  Tier 2 (promising)             — 3 % of equity per position
  Tier 3 (proven)                — 5 % of equity per position  (production max)

Gate criteria
-------------
  Tier 2 requires ALL of:
    - ≥ MIN_TRADES_TIER2 completed trades (default 20)
    - Sharpe ratio  > SHARPE_THRESHOLD_TIER2 (default 1.0)
    - Max draw-down < MAX_DD_THRESHOLD_TIER2 (default 8 %)

  Tier 3 requires ALL of:
    - ≥ MIN_TRADES_TIER3 completed trades (default 50)
    - Sharpe ratio  > SHARPE_THRESHOLD_TIER3 (default 1.5)
    - Max draw-down < MAX_DD_THRESHOLD_TIER3 (default 5 %)

State persistence
-----------------
The current tier and the metrics used to compute it are written to
  logs/allocation_config.json
so the daemons can inspect them at any time without importing this module.

Usage
-----
    from allocation_controller import AllocationController
    import db as _db

    controller = AllocationController(_db)
    max_pct = controller.get_position_size_pct()   # e.g. 0.03
    position_usd = equity * max_pct
"""
import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Tunable constants (all overridable via env vars)
# ---------------------------------------------------------------------------

MIN_TRADES_TIER2:       int   = int(os.getenv('ALLOC_MIN_TRADES_T2',   '20'))
MIN_TRADES_TIER3:       int   = int(os.getenv('ALLOC_MIN_TRADES_T3',   '50'))
SHARPE_THRESHOLD_TIER2: float = float(os.getenv('ALLOC_SHARPE_T2',     '1.0'))
SHARPE_THRESHOLD_TIER3: float = float(os.getenv('ALLOC_SHARPE_T3',     '1.5'))
MAX_DD_THRESHOLD_TIER2: float = float(os.getenv('ALLOC_MAX_DD_T2',     '0.08'))
MAX_DD_THRESHOLD_TIER3: float = float(os.getenv('ALLOC_MAX_DD_T3',     '0.05'))

TIER1_PCT: float = float(os.getenv('ALLOC_TIER1_PCT', '0.01'))
TIER2_PCT: float = float(os.getenv('ALLOC_TIER2_PCT', '0.03'))
TIER3_PCT: float = float(os.getenv('ALLOC_TIER3_PCT', '0.05'))

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ALLOC_CONFIG  = _PROJECT_ROOT / 'logs' / 'allocation_config.json'
_RISK_FREE_RATE = 0.05  # annualised, used for Sharpe calculation


# ---------------------------------------------------------------------------
# AllocationController
# ---------------------------------------------------------------------------

class AllocationController:
    """
    Reads the trade history from *db_module* (the scripts/db.py module) and
    computes the appropriate position-size tier.

    Parameters
    ----------
    db_module : module
        The imported ``db`` module (passed in to keep this class testable
        without importing the real database).
    lookback_trades : int
        Number of most-recent closed trades to consider for Sharpe / DD.
    """

    def __init__(self, db_module, lookback_trades: int = 100) -> None:
        self._db = db_module
        self._lookback = lookback_trades
        self._cached_tier:    Optional[int]   = None
        self._cached_metrics: Optional[dict]  = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_position_size_pct(self, force_refresh: bool = False) -> float:
        """
        Return the maximum allowed position size as a fraction of equity.

        Caches the tier for the lifetime of this object unless
        *force_refresh* is True or ``refresh()`` is called explicitly.
        """
        if self._cached_tier is None or force_refresh:
            self.refresh()
        return {1: TIER1_PCT, 2: TIER2_PCT, 3: TIER3_PCT}[self._cached_tier]

    def current_tier(self, force_refresh: bool = False) -> int:
        """Return the current tier (1, 2, or 3)."""
        if self._cached_tier is None or force_refresh:
            self.refresh()
        return self._cached_tier

    def metrics(self) -> Optional[dict]:
        """Return the last-computed performance metrics dict."""
        return self._cached_metrics

    def refresh(self) -> None:
        """Recompute the tier from the latest DB data and persist to disk."""
        try:
            trades = self._load_closed_trades()
            m = self._compute_metrics(trades)
            self._cached_metrics = m
            self._cached_tier = self._evaluate_tier(m)
            self._persist(m)
        except Exception as e:
            logging.warning(f"allocation_controller: refresh failed: {e}")
            if self._cached_tier is None:
                self._cached_tier = 1  # safe fallback

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_closed_trades(self) -> list[dict]:
        """Return the N most-recent closed trades from the DB."""
        try:
            rows = self._db.get_recent_trades(limit=self._lookback)
            # Each row is a sqlite3.Row or dict; normalise to dict
            return [dict(r) for r in rows]
        except Exception as e:
            logging.warning(f"allocation_controller: could not load trades: {e}")
            return []

    def _compute_metrics(self, trades: list[dict]) -> dict:
        """
        Compute Sharpe ratio and maximum draw-down from *trades*.

        Returns a dict with keys:
          total_trades, win_rate, sharpe, max_dd, annualised_return
        """
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'sharpe': 0.0,
                'max_dd': 1.0,
                'annualised_return': 0.0,
            }

        pnl_pcts = []
        for t in trades:
            pnl = t.get('pnl_pct') or t.get('pnl') or 0.0
            try:
                pnl_pcts.append(float(pnl))
            except (ValueError, TypeError):
                pass

        if not pnl_pcts:
            return {
                'total_trades': len(trades),
                'win_rate': 0.0,
                'sharpe': 0.0,
                'max_dd': 1.0,
                'annualised_return': 0.0,
            }

        n = len(pnl_pcts)
        winners = sum(1 for p in pnl_pcts if p > 0)
        win_rate = winners / n

        # Per-trade mean and std
        mean_pnl = sum(pnl_pcts) / n
        variance = sum((p - mean_pnl) ** 2 for p in pnl_pcts) / max(n - 1, 1)
        std_pnl = math.sqrt(variance) if variance > 0 else 1e-9

        # Annualise assuming ~252 trades/year (conservative)
        trades_per_year = 252
        annualised_return = mean_pnl * trades_per_year
        annualised_std = std_pnl * math.sqrt(trades_per_year)
        sharpe = (annualised_return - _RISK_FREE_RATE) / annualised_std if annualised_std > 0 else 0.0

        # Max draw-down on cumulative equity curve (starting at 1.0)
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        for p in pnl_pcts:
            equity *= (1.0 + p)
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        return {
            'total_trades': n,
            'win_rate': round(win_rate, 4),
            'sharpe': round(sharpe, 4),
            'max_dd': round(max_dd, 4),
            'annualised_return': round(annualised_return, 4),
        }

    def _evaluate_tier(self, m: dict) -> int:
        n      = m['total_trades']
        sharpe = m['sharpe']
        max_dd = m['max_dd']

        if (
            n      >= MIN_TRADES_TIER3
            and sharpe >= SHARPE_THRESHOLD_TIER3
            and max_dd  < MAX_DD_THRESHOLD_TIER3
        ):
            return 3

        if (
            n      >= MIN_TRADES_TIER2
            and sharpe >= SHARPE_THRESHOLD_TIER2
            and max_dd  < MAX_DD_THRESHOLD_TIER2
        ):
            return 2

        return 1

    def check_live_readiness(
        self,
        trading_client,
        min_paper_days: int = 30,
        min_trades: int = 50,
        min_sharpe: float = 1.0,
        max_dd: float = 0.08,
        min_win_rate: float = 0.45,
        min_equity: float = 25_000.0,
    ) -> tuple[dict, float, int, list[str]]:
        """
        Evaluate whether this agent is ready for live trading.

        Returns (metrics, equity, paper_days, unmet_criteria).
        unmet_criteria is empty when all gates pass.
        """
        import os as _os
        from datetime import datetime as _dt, timezone as _tz

        self.refresh()
        m = self._cached_metrics or {}

        # Equity from Alpaca
        equity = 0.0
        try:
            account = trading_client.get_account()
            equity = float(getattr(account, 'equity', 0) or 0)
        except Exception as e:
            logging.warning(f"live_readiness: could not fetch equity: {e}")

        # Days of paper trading — earliest outcome timestamp
        paper_days = 0
        try:
            rows = self._db.get_recent_trades(limit=10_000)
            if rows:
                timestamps = [r.get('timestamp') or r.get('entry_timestamp') or '' for r in rows]
                timestamps = [t for t in timestamps if t]
                if timestamps:
                    earliest = min(timestamps)
                    earliest_dt = _dt.fromisoformat(earliest.replace('Z', '+00:00'))
                    if earliest_dt.tzinfo is None:
                        earliest_dt = earliest_dt.replace(tzinfo=_tz.utc)
                    paper_days = (_dt.now(_tz.utc) - earliest_dt).days
        except Exception as e:
            logging.warning(f"live_readiness: could not compute paper_days: {e}")

        unmet: list[str] = []
        if equity < min_equity:
            unmet.append(f"equity ${equity:,.0f} < ${min_equity:,.0f} (PDT minimum)")
        if paper_days < min_paper_days:
            unmet.append(f"paper trading {paper_days}d < {min_paper_days}d minimum")
        if m.get('total_trades', 0) < min_trades:
            unmet.append(f"trades {m.get('total_trades',0)} < {min_trades}")
        if m.get('sharpe', 0) < min_sharpe:
            unmet.append(f"Sharpe {m.get('sharpe',0):.2f} < {min_sharpe}")
        if m.get('max_dd', 1) > max_dd:
            unmet.append(f"max drawdown {m.get('max_dd',1):.1%} > {max_dd:.1%}")
        if m.get('win_rate', 0) < min_win_rate:
            unmet.append(f"win rate {m.get('win_rate',0):.1%} < {min_win_rate:.1%}")

        return m, equity, paper_days, unmet

    def _persist(self, metrics: dict) -> None:
        try:
            _ALLOC_CONFIG.parent.mkdir(parents=True, exist_ok=True)
            from datetime import timezone as _tz
            payload = {
                'computed_at':    datetime.now(_tz.utc).isoformat(),
                'tier':           self._cached_tier,
                'position_size_pct': {1: TIER1_PCT, 2: TIER2_PCT, 3: TIER3_PCT}[self._cached_tier],
                'gates': {
                    'tier2': {
                        'min_trades':    MIN_TRADES_TIER2,
                        'sharpe_min':    SHARPE_THRESHOLD_TIER2,
                        'max_dd_max':    MAX_DD_THRESHOLD_TIER2,
                    },
                    'tier3': {
                        'min_trades':    MIN_TRADES_TIER3,
                        'sharpe_min':    SHARPE_THRESHOLD_TIER3,
                        'max_dd_max':    MAX_DD_THRESHOLD_TIER3,
                    },
                },
                'current_metrics': metrics,
            }
            _ALLOC_CONFIG.write_text(json.dumps(payload, indent=2))
        except Exception as e:
            logging.debug(f"allocation_controller: could not persist config: {e}")
