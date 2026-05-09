"""
Unit tests for stock_discovery — passes_liquidity_filter and scan loop integration.

All tests are fully offline — no yfinance API calls, no network required.
"""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'data'))

import stock_discovery
from stock_discovery import passes_liquidity_filter, _MIN_AVG_DAILY_VOLUME, _MIN_MARKET_CAP


# ── passes_liquidity_filter ────────────────────────────────────────────────────

class TestPassesLiquidityFilter:
    def _good_info(self):
        return {
            'averageVolume': _MIN_AVG_DAILY_VOLUME + 1,
            'marketCap':     _MIN_MARKET_CAP + 1,
        }

    def test_passes_when_both_thresholds_met(self):
        assert passes_liquidity_filter('AAPL', self._good_info()) is True

    def test_fails_when_volume_below_threshold(self):
        info = self._good_info()
        info['averageVolume'] = _MIN_AVG_DAILY_VOLUME - 1
        assert passes_liquidity_filter('LOW', info) is False

    def test_fails_when_market_cap_below_threshold(self):
        info = self._good_info()
        info['marketCap'] = _MIN_MARKET_CAP - 1
        assert passes_liquidity_filter('TINY', info) is False

    def test_fails_when_volume_exactly_at_threshold_minus_one(self):
        info = self._good_info()
        info['averageVolume'] = _MIN_AVG_DAILY_VOLUME - 1
        assert passes_liquidity_filter('X', info) is False

    def test_passes_when_volume_exactly_at_threshold(self):
        info = self._good_info()
        info['averageVolume'] = _MIN_AVG_DAILY_VOLUME
        assert passes_liquidity_filter('X', info) is True

    def test_none_volume_treated_as_zero(self):
        info = {'averageVolume': None, 'marketCap': _MIN_MARKET_CAP + 1}
        assert passes_liquidity_filter('X', info) is False

    def test_none_market_cap_treated_as_zero(self):
        info = {'averageVolume': _MIN_AVG_DAILY_VOLUME + 1, 'marketCap': None}
        assert passes_liquidity_filter('X', info) is False

    def test_missing_keys_treated_as_zero(self):
        assert passes_liquidity_filter('X', {}) is False

    def test_both_none_fails(self):
        assert passes_liquidity_filter('X', {'averageVolume': None, 'marketCap': None}) is False

    def test_custom_threshold_via_env(self):
        """ENV overrides work — filter uses the patched module-level constant."""
        info = {'averageVolume': 500_000, 'marketCap': _MIN_MARKET_CAP + 1}
        original = stock_discovery._MIN_AVG_DAILY_VOLUME
        try:
            stock_discovery._MIN_AVG_DAILY_VOLUME = 500_000
            assert passes_liquidity_filter('X', info) is True
        finally:
            stock_discovery._MIN_AVG_DAILY_VOLUME = original


# ── Scan loop integration ─────────────────────────────────────────────────────

def _make_discovery():
    """Return a StockDiscovery with mocked delisted cache to avoid filesystem I/O."""
    with patch('stock_discovery._load_delisted_cache', return_value=set()):
        d = stock_discovery.StockDiscovery()
    return d


def _ticker_mock(volume_ratio: float = 3.0, has_info: bool = True,
                 avg_vol: int = None, mkt_cap: int = None):
    """Build a yf.Ticker mock that passes or fails the liquidity filter."""
    avg_vol = avg_vol if avg_vol is not None else _MIN_AVG_DAILY_VOLUME + 1
    mkt_cap = mkt_cap if mkt_cap is not None else _MIN_MARKET_CAP + 1

    import pandas as pd
    import numpy as np

    prices = [100.0] * 10
    # scan_unusual_volume uses iloc[-1] as today's volume, iloc[:-1].mean() as average
    volumes = [avg_vol] * 9 + [int(avg_vol * volume_ratio)]
    hist = pd.DataFrame({
        'Close':  prices,
        'High':   [p * 1.01 for p in prices],
        'Low':    [p * 0.99 for p in prices],
        'Open':   prices,
        'Volume': volumes,
    })

    ticker = MagicMock()
    ticker.history.return_value = hist
    ticker.info = {'averageVolume': avg_vol, 'marketCap': mkt_cap} if has_info else {}
    return ticker


class TestScanUnusualVolumeRespectLiquidity:
    def test_liquid_stock_is_included(self):
        d = _make_discovery()
        ticker = _ticker_mock(volume_ratio=3.0)
        with patch('yfinance.Ticker', return_value=ticker):
            result = d.scan_unusual_volume(['AAPL'], top_n=10)
        assert 'AAPL' in result

    def test_illiquid_stock_volume_is_excluded(self):
        """A stock that exceeds the volume ratio but fails market-cap gate is skipped."""
        d = _make_discovery()
        ticker = _ticker_mock(volume_ratio=3.0, mkt_cap=_MIN_MARKET_CAP - 1)
        with patch('yfinance.Ticker', return_value=ticker):
            result = d.scan_unusual_volume(['TINY'], top_n=10)
        assert 'TINY' not in result

    def test_illiquid_stock_avg_vol_is_excluded(self):
        """Stock below avg-volume threshold is excluded even with unusual spike."""
        d = _make_discovery()
        ticker = _ticker_mock(volume_ratio=3.0, avg_vol=_MIN_AVG_DAILY_VOLUME - 1)
        with patch('yfinance.Ticker', return_value=ticker):
            result = d.scan_unusual_volume(['LOW'], top_n=10)
        assert 'LOW' not in result

    def test_info_exception_excludes_stock(self):
        """If .info raises, the stock is excluded (safe default)."""
        d = _make_discovery()
        import pandas as pd
        volumes = [_MIN_AVG_DAILY_VOLUME * 3] + [_MIN_AVG_DAILY_VOLUME] * 9
        hist = pd.DataFrame({
            'Close':  [100.0] * 10,
            'High':   [101.0] * 10,
            'Low':    [99.0]  * 10,
            'Open':   [100.0] * 10,
            'Volume': volumes,
        })
        ticker = MagicMock()
        ticker.history.return_value = hist
        type(ticker).info = property(lambda self: (_ for _ in ()).throw(Exception('timeout')))
        with patch('yfinance.Ticker', return_value=ticker):
            result = d.scan_unusual_volume(['ERR'], top_n=10)
        # Info exception → empty info dict → liquidity filter fails → excluded
        assert 'ERR' not in result


class TestScanBreakoutsRespectLiquidity:
    def _breakout_ticker(self, liquid: bool = True):
        import pandas as pd
        avg_vol = _MIN_AVG_DAILY_VOLUME + 1
        mkt_cap = _MIN_MARKET_CAP + 1 if liquid else _MIN_MARKET_CAP - 1
        n = 260
        prices = [95.0] * (n - 1) + [100.0]
        hist = pd.DataFrame({
            'Close':  prices,
            'High':   [p * 1.001 for p in prices],
            'Low':    [p * 0.999 for p in prices],
            'Open':   prices,
            'Volume': [avg_vol] * n,
        })
        ticker = MagicMock()
        ticker.history.return_value = hist
        ticker.info = {'averageVolume': avg_vol, 'marketCap': mkt_cap}
        return ticker

    def test_liquid_breakout_included(self):
        d = _make_discovery()
        with patch('yfinance.Ticker', return_value=self._breakout_ticker(liquid=True)):
            result = d.scan_breakouts(['NVDA'], top_n=10)
        assert 'NVDA' in result

    def test_illiquid_breakout_excluded(self):
        d = _make_discovery()
        with patch('yfinance.Ticker', return_value=self._breakout_ticker(liquid=False)):
            result = d.scan_breakouts(['SMALL'], top_n=10)
        assert 'SMALL' not in result
