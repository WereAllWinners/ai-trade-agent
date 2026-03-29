#!/usr/bin/env python3
"""
portfolio_overseer.py — Portfolio-level concentration guard.

Prevents both the stock and options agents from creating over-concentrated
positions by enforcing two hard limits before each buy:

  1. Sector cap  — no single GICS sector may exceed MAX_SECTOR_PCT of equity
  2. Correlation cap — no more than MAX_CORR_POSITIONS open positions from the
     same sector (a cheap proxy for correlation when real correlation matrices
     are unavailable at run-time).

Usage:
    from portfolio_overseer import PortfolioOverseer
    overseer = PortfolioOverseer(trading_client)
    allowed, reason = overseer.is_buy_allowed('AAPL', spend_usd=500, equity=20_000)
    if not allowed:
        logging.warning(f"Buy vetoed: {reason}")

The overseer caches open positions for CACHE_TTL_SECONDS (default 60 s) so
repeated checks within a session do not hammer the Alpaca API.

Constraints log is written to logs/portfolio_constraints.json after every veto.
"""
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_SECTOR_PCT:      float = float(__import__('os').getenv('MAX_SECTOR_PCT', '0.30'))
MAX_CORR_POSITIONS:  int   = int(__import__('os').getenv('MAX_CORR_POSITIONS', '4'))
CACHE_TTL_SECONDS:   int   = int(__import__('os').getenv('OVERSEER_CACHE_TTL', '60'))

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONSTRAINTS_LOG = _PROJECT_ROOT / 'logs' / 'portfolio_constraints.json'

# ---------------------------------------------------------------------------
# GICS-inspired sector map (common liquid tickers — extend as needed)
# ---------------------------------------------------------------------------

_SECTOR_MAP: dict[str, str] = {
    # Technology
    'AAPL': 'technology', 'MSFT': 'technology', 'GOOGL': 'technology',
    'GOOG': 'technology', 'META': 'technology', 'NVDA': 'technology',
    'AMD': 'technology', 'INTC': 'technology', 'AVGO': 'technology',
    'QCOM': 'technology', 'TXN': 'technology', 'MU': 'technology',
    'AMAT': 'technology', 'LRCX': 'technology', 'KLAC': 'technology',
    'CRM': 'technology', 'ORCL': 'technology', 'SAP': 'technology',
    'NOW': 'technology', 'SNOW': 'technology', 'PLTR': 'technology',
    'SHOP': 'technology', 'UBER': 'technology', 'LYFT': 'technology',
    'COIN': 'technology', 'MSTR': 'technology',
    # Communication Services
    'NFLX': 'communication', 'DIS': 'communication', 'CMCSA': 'communication',
    'T': 'communication', 'VZ': 'communication', 'TMUS': 'communication',
    'SNAP': 'communication', 'PINS': 'communication', 'SPOT': 'communication',
    'RBLX': 'communication', 'EA': 'communication', 'TTWO': 'communication',
    # Consumer Discretionary
    'AMZN': 'consumer_discretionary', 'TSLA': 'consumer_discretionary',
    'HD': 'consumer_discretionary', 'LOW': 'consumer_discretionary',
    'NKE': 'consumer_discretionary', 'SBUX': 'consumer_discretionary',
    'MCD': 'consumer_discretionary', 'CMG': 'consumer_discretionary',
    'TGT': 'consumer_discretionary', 'BKNG': 'consumer_discretionary',
    'ABNB': 'consumer_discretionary', 'MAR': 'consumer_discretionary',
    'F': 'consumer_discretionary', 'GM': 'consumer_discretionary',
    'RIVN': 'consumer_discretionary', 'LCID': 'consumer_discretionary',
    # Consumer Staples
    'WMT': 'consumer_staples', 'COST': 'consumer_staples',
    'PG': 'consumer_staples', 'KO': 'consumer_staples',
    'PEP': 'consumer_staples', 'PM': 'consumer_staples',
    'MO': 'consumer_staples', 'CL': 'consumer_staples',
    # Financials
    'JPM': 'financials', 'BAC': 'financials', 'WFC': 'financials',
    'GS': 'financials', 'MS': 'financials', 'C': 'financials',
    'BLK': 'financials', 'SCHW': 'financials', 'AXP': 'financials',
    'V': 'financials', 'MA': 'financials', 'PYPL': 'financials',
    'SQ': 'financials', 'AFRM': 'financials',
    # Healthcare
    'JNJ': 'healthcare', 'UNH': 'healthcare', 'PFE': 'healthcare',
    'ABBV': 'healthcare', 'MRK': 'healthcare', 'LLY': 'healthcare',
    'BMY': 'healthcare', 'AMGN': 'healthcare', 'GILD': 'healthcare',
    'BIIB': 'healthcare', 'REGN': 'healthcare', 'MRNA': 'healthcare',
    'CVS': 'healthcare', 'CI': 'healthcare', 'HUM': 'healthcare',
    'ISRG': 'healthcare', 'MDT': 'healthcare', 'ABT': 'healthcare',
    # Energy
    'XOM': 'energy', 'CVX': 'energy', 'COP': 'energy',
    'SLB': 'energy', 'EOG': 'energy', 'PXD': 'energy',
    'OXY': 'energy', 'MPC': 'energy', 'VLO': 'energy',
    # Industrials
    'BA': 'industrials', 'CAT': 'industrials', 'HON': 'industrials',
    'GE': 'industrials', 'MMM': 'industrials', 'UPS': 'industrials',
    'FDX': 'industrials', 'LMT': 'industrials', 'RTX': 'industrials',
    'DE': 'industrials', 'EMR': 'industrials',
    # Materials
    'LIN': 'materials', 'APD': 'materials', 'ECL': 'materials',
    'NEM': 'materials', 'FCX': 'materials', 'AA': 'materials',
    'NUE': 'materials', 'X': 'materials',
    # Real Estate
    'AMT': 'real_estate', 'PLD': 'real_estate', 'EQIX': 'real_estate',
    'CCI': 'real_estate', 'PSA': 'real_estate', 'O': 'real_estate',
    'SPG': 'real_estate',
    # Utilities
    'NEE': 'utilities', 'DUK': 'utilities', 'SO': 'utilities',
    'AEP': 'utilities', 'XEL': 'utilities', 'PCG': 'utilities',
    # ETFs / indices (treated as diversified)
    'SPY': 'diversified', 'QQQ': 'diversified', 'IWM': 'diversified',
    'DIA': 'diversified', 'VTI': 'diversified', 'VOO': 'diversified',
    'XLK': 'diversified', 'XLF': 'diversified', 'XLE': 'diversified',
    'XLV': 'diversified', 'XLI': 'diversified', 'GLD': 'diversified',
    'SLV': 'diversified', 'TLT': 'diversified', 'HYG': 'diversified',
}


def get_sector(symbol: str) -> str:
    """Return the sector for *symbol*, defaulting to 'unknown'."""
    return _SECTOR_MAP.get(symbol.upper(), 'unknown')


# ---------------------------------------------------------------------------
# PortfolioOverseer
# ---------------------------------------------------------------------------

class PortfolioOverseer:
    """
    Stateful overseer that caches open positions and enforces concentration limits.

    Parameters
    ----------
    trading_client : alpaca.trading.client.TradingClient
        Authenticated Alpaca client used to fetch open positions.
    """

    def __init__(self, trading_client) -> None:
        self._client = trading_client
        self._positions_cache: Optional[list] = None
        self._cache_timestamp: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_buy_allowed(
        self,
        symbol: str,
        spend_usd: float,
        equity: float,
    ) -> tuple[bool, str]:
        """
        Return (True, '') if the buy is within concentration limits, or
        (False, reason_string) if it would breach a cap.

        Parameters
        ----------
        symbol    : ticker being purchased
        spend_usd : USD value of the proposed purchase
        equity    : total account equity (used to compute sector %)
        """
        if equity <= 0:
            return True, ''

        positions = self._get_positions()
        sector = get_sector(symbol)

        # Diversified ETFs are never vetoed
        if sector == 'diversified':
            return True, ''

        # ------------------------------------------------------------------
        # 1. Sector value cap
        # ------------------------------------------------------------------
        sector_value = self._sector_market_value(positions, sector)
        projected_sector_pct = (sector_value + spend_usd) / equity

        if projected_sector_pct > MAX_SECTOR_PCT:
            reason = (
                f"sector cap breached: {sector} would be "
                f"{projected_sector_pct:.1%} of equity "
                f"(limit {MAX_SECTOR_PCT:.0%})"
            )
            self._log_veto(symbol, sector, reason)
            return False, reason

        # ------------------------------------------------------------------
        # 2. Correlation / concentration cap (same-sector position count)
        # ------------------------------------------------------------------
        same_sector_count = self._sector_position_count(positions, sector)
        # Don't count the symbol itself if already held (adding to position)
        already_held = any(
            p.symbol.upper() == symbol.upper() for p in positions
        )
        new_positions = 0 if already_held else 1

        if same_sector_count + new_positions > MAX_CORR_POSITIONS:
            reason = (
                f"correlation cap breached: {sector} already has "
                f"{same_sector_count} open position(s) "
                f"(limit {MAX_CORR_POSITIONS})"
            )
            self._log_veto(symbol, sector, reason)
            return False, reason

        return True, ''

    def sector_headroom(self, equity: float) -> dict[str, float]:
        """
        Return remaining USD headroom per sector before the cap is hit.
        Useful for the /status health endpoint.
        """
        if equity <= 0:
            return {}
        positions = self._get_positions()
        sectors: dict[str, float] = {}
        for p in positions:
            s = get_sector(p.symbol)
            mv = float(getattr(p, 'market_value', 0) or 0)
            sectors[s] = sectors.get(s, 0.0) + mv

        headroom = {}
        for sector, used in sectors.items():
            if sector in ('diversified', 'unknown'):
                continue
            cap_usd = equity * MAX_SECTOR_PCT
            headroom[sector] = max(0.0, cap_usd - used)
        return headroom

    def invalidate_cache(self) -> None:
        """Force a fresh positions fetch on the next check."""
        self._cache_timestamp = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_positions(self) -> list:
        now = time.monotonic()
        if (
            self._positions_cache is not None
            and now - self._cache_timestamp < CACHE_TTL_SECONDS
        ):
            return self._positions_cache

        try:
            positions = self._client.get_all_positions()
            self._positions_cache = list(positions)
            self._cache_timestamp = now
        except Exception as e:
            logging.warning(f"portfolio_overseer: could not fetch positions: {e}")
            self._positions_cache = self._positions_cache or []
        return self._positions_cache

    def _sector_market_value(self, positions: list, sector: str) -> float:
        total = 0.0
        for p in positions:
            if get_sector(p.symbol) == sector:
                try:
                    total += float(p.market_value or 0)
                except (ValueError, TypeError):
                    pass
        return total

    def _sector_position_count(self, positions: list, sector: str) -> int:
        return sum(1 for p in positions if get_sector(p.symbol) == sector)

    def _log_veto(self, symbol: str, sector: str, reason: str) -> None:
        try:
            _CONSTRAINTS_LOG.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                'timestamp': datetime.now(__import__('datetime').timezone.utc).isoformat(),
                'symbol': symbol,
                'sector': sector,
                'reason': reason,
            }
            existing: list = []
            if _CONSTRAINTS_LOG.exists():
                try:
                    existing = json.loads(_CONSTRAINTS_LOG.read_text())
                    if not isinstance(existing, list):
                        existing = []
                except Exception:
                    existing = []
            existing.append(entry)
            # Keep last 500 veto records
            _CONSTRAINTS_LOG.write_text(json.dumps(existing[-500:], indent=2))
        except Exception as e:
            logging.debug(f"portfolio_overseer: could not write veto log: {e}")
