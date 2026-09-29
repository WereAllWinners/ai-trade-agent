"""
Unit tests for stock_discovery — passes_liquidity_filter and scan loop integration.

All tests are fully offline — no yfinance API calls, no network required.
"""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from collections import defaultdict

import pandas as pd

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
    with patch('stock_discovery._load_delisted_cache', return_value=set()), \
         patch('stock_discovery._load_fail_counts', return_value={}):
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
    """sprint04 F3: OHLCV now comes from _get_batch_history (yf.download),
    not yf.Ticker(...).history() — mock the batch map for history data,
    keep yfinance.Ticker mocked only for the still-per-symbol .info lookup."""

    def test_liquid_stock_is_included(self):
        d = _make_discovery()
        ticker = _ticker_mock(volume_ratio=3.0)
        with patch.object(d, '_get_batch_history', return_value={'AAPL': ticker.history.return_value}), \
             patch('yfinance.Ticker', return_value=ticker):
            result = d.scan_unusual_volume(['AAPL'], top_n=10)
        assert 'AAPL' in result

    def test_illiquid_stock_volume_is_excluded(self):
        """A stock that exceeds the volume ratio but fails market-cap gate is skipped."""
        d = _make_discovery()
        ticker = _ticker_mock(volume_ratio=3.0, mkt_cap=_MIN_MARKET_CAP - 1)
        with patch.object(d, '_get_batch_history', return_value={'TINY': ticker.history.return_value}), \
             patch('yfinance.Ticker', return_value=ticker):
            result = d.scan_unusual_volume(['TINY'], top_n=10)
        assert 'TINY' not in result

    def test_illiquid_stock_avg_vol_is_excluded(self):
        """Stock below avg-volume threshold is excluded even with unusual spike."""
        d = _make_discovery()
        ticker = _ticker_mock(volume_ratio=3.0, avg_vol=_MIN_AVG_DAILY_VOLUME - 1)
        with patch.object(d, '_get_batch_history', return_value={'LOW': ticker.history.return_value}), \
             patch('yfinance.Ticker', return_value=ticker):
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
        type(ticker).info = property(lambda self: (_ for _ in ()).throw(Exception('timeout')))
        with patch.object(d, '_get_batch_history', return_value={'ERR': hist}), \
             patch('yfinance.Ticker', return_value=ticker):
            result = d.scan_unusual_volume(['ERR'], top_n=10)
        # Info exception → empty info dict → liquidity filter fails → excluded
        assert 'ERR' not in result

    def test_delisted_symbol_excluded_before_batch_fetch(self):
        """sprint04 F4: a _delisted symbol is filtered out of candidates
        before the batch fetch is even requested."""
        d = _make_discovery()
        d._delisted.add('DEAD')
        with patch.object(d, '_get_batch_history') as mock_batch:
            mock_batch.return_value = {}
            result = d.scan_unusual_volume(['DEAD'], top_n=10)
        mock_batch.assert_called_once_with([], period='10d')
        assert result == []

    def test_empty_history_records_fetch_failure(self):
        """sprint04 F4: a genuinely empty DataFrame for a symbol calls
        _record_fetch_result(success=False) — the real per-symbol failure
        signal that eventually feeds D5a's delisted counter."""
        d = _make_discovery()
        import pandas as pd
        with patch.object(d, '_get_batch_history', return_value={'DEAD': pd.DataFrame()}), \
             patch.object(d, '_record_fetch_result') as mock_record:
            d.scan_unusual_volume(['DEAD'], top_n=10)
        mock_record.assert_called_once_with('DEAD', success=False)

    def test_chunk_failure_sentinel_skips_without_recording(self):
        """sprint04 F3/F4: a None result (chunk-level infrastructure
        failure) must NOT call _record_fetch_result at all — this is the
        regression test for the mass-false-delisting hazard."""
        d = _make_discovery()
        with patch.object(d, '_get_batch_history', return_value={'AAPL': None}), \
             patch.object(d, '_record_fetch_result') as mock_record:
            result = d.scan_unusual_volume(['AAPL'], top_n=10)
        mock_record.assert_not_called()
        assert result == []


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
        ticker = self._breakout_ticker(liquid=True)
        with patch.object(d, '_get_batch_history', return_value={'NVDA': ticker.history.return_value}), \
             patch('yfinance.Ticker', return_value=ticker):
            result = d.scan_breakouts(['NVDA'], top_n=10)
        assert 'NVDA' in result

    def test_illiquid_breakout_excluded(self):
        d = _make_discovery()
        ticker = self._breakout_ticker(liquid=False)
        with patch.object(d, '_get_batch_history', return_value={'SMALL': ticker.history.return_value}), \
             patch('yfinance.Ticker', return_value=ticker):
            result = d.scan_breakouts(['SMALL'], top_n=10)
        assert 'SMALL' not in result

    def test_chunk_failure_sentinel_skips_without_recording(self):
        d = _make_discovery()
        with patch.object(d, '_get_batch_history', return_value={'NVDA': None}), \
             patch.object(d, '_record_fetch_result') as mock_record:
            result = d.scan_breakouts(['NVDA'], top_n=10)
        mock_record.assert_not_called()
        assert result == []


# ── sprint04 F3: _fetch_batch_history / _get_batch_history ─────────────────────

def _multiindex_download_response(data: dict) -> pd.DataFrame:
    """Build a yf.download(group_by='ticker')-shaped MultiIndex DataFrame.
    `data`: {symbol: {'Open': [...], 'High': [...], 'Low': [...],
    'Close': [...], 'Volume': [...]}}. Symbols absent from `data` are
    omitted from the returned columns entirely — mirrors yf.download's real
    omit-on-failure behavior (not just NaN-filled)."""
    n = len(next(iter(data.values()))['Close'])
    dates = pd.date_range('2026-01-01', periods=n)
    frame_data = {}
    for sym, fields in data.items():
        for field, vals in fields.items():
            frame_data[(sym, field)] = vals
    df = pd.DataFrame(frame_data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _healthy_fields(n: int = 5, price: float = 100.0, volume: int = 1_000_000) -> dict:
    return {
        'Open': [price] * n, 'High': [price * 1.01] * n,
        'Low': [price * 0.99] * n, 'Close': [price] * n,
        'Volume': [volume] * n,
    }


def _all_nan_fields(n: int = 5) -> dict:
    import numpy as np
    return {
        'Open': [np.nan] * n, 'High': [np.nan] * n,
        'Low': [np.nan] * n, 'Close': [np.nan] * n,
        'Volume': [np.nan] * n,
    }


class TestFetchBatchHistory:
    def test_correct_per_symbol_slicing(self):
        resp = _multiindex_download_response({
            'AAA': _healthy_fields(price=100.0),
            'BBB': _healthy_fields(price=200.0),
        })
        with patch('stock_discovery.yf.download', return_value=resp):
            result = stock_discovery._fetch_batch_history(['AAA', 'BBB'], period='5d')
        assert result['AAA']['Close'].iloc[-1] == 100.0
        assert result['BBB']['Close'].iloc[-1] == 200.0

    def test_all_nan_column_in_healthy_chunk_is_empty_for_that_symbol_only(self):
        symbols = [f'S{i}' for i in range(12)]
        data = {s: _healthy_fields() for s in symbols[:-1]}
        data[symbols[-1]] = _all_nan_fields()
        resp = _multiindex_download_response(data)
        with patch('stock_discovery.yf.download', return_value=resp):
            result = stock_discovery._fetch_batch_history(symbols, period='5d')
        assert result[symbols[-1]] is not None
        assert result[symbols[-1]].empty
        for s in symbols[:-1]:
            assert result[s] is not None and not result[s].empty

    def test_single_symbol_chunk_branch(self):
        """yf.download does not return MultiIndex columns for a length-1
        tickers list, even with group_by='ticker'."""
        n = 5
        single_resp = pd.DataFrame(_healthy_fields(n=n, price=100.0),
                                    index=pd.date_range('2026-01-01', periods=n))
        with patch('stock_discovery.yf.download', return_value=single_resp):
            result = stock_discovery._fetch_batch_history(['AAA'], period='5d')
        assert result['AAA']['Close'].iloc[-1] == 100.0

    def test_chunk_boundary_594_symbols_yields_6_download_calls(self):
        symbols = [f'S{i}' for i in range(594)]

        def fake_download(tickers, **kwargs):
            return _multiindex_download_response({s: _healthy_fields() for s in tickers})

        with patch('stock_discovery.yf.download', side_effect=fake_download) as mock_dl:
            stock_discovery._fetch_batch_history(symbols, period='5d')
        assert mock_dl.call_count == 6

    def test_download_exception_maps_whole_chunk_to_none(self):
        with patch('stock_discovery.yf.download', side_effect=Exception('rate limited')):
            result = stock_discovery._fetch_batch_history(['AAA', 'BBB'], period='5d')
        assert result == {'AAA': None, 'BBB': None}


class TestChunkHealthCheck:
    """The most important regression class in sprint04 F3: collapsing
    'chunk-wide infrastructure failure' and 'isolated symbol death' into the
    same signal would let one bad rate-limited scan mass-delist a whole
    chunk of otherwise-healthy symbols via D5a's persisted counter."""

    def test_fully_empty_chunk_maps_every_symbol_to_none(self):
        symbols = [f'S{i}' for i in range(20)]
        resp = _multiindex_download_response({s: _all_nan_fields() for s in symbols})
        with patch('stock_discovery.yf.download', return_value=resp):
            result = stock_discovery._fetch_batch_history(symbols, period='5d')
        assert all(result[s] is None for s in symbols)

    def test_isolated_empty_symbol_among_healthy_chunk_recordable(self):
        symbols = [f'S{i}' for i in range(20)]
        data = {s: _healthy_fields() for s in symbols[:-1]}
        data[symbols[-1]] = _all_nan_fields()
        resp = _multiindex_download_response(data)
        with patch('stock_discovery.yf.download', return_value=resp):
            result = stock_discovery._fetch_batch_history(symbols, period='5d')
        assert result[symbols[-1]] is not None
        assert result[symbols[-1]].empty
        for s in symbols[:-1]:
            assert result[s] is not None and not result[s].empty

    def test_downstream_scan_method_skips_record_for_fully_empty_chunk(self):
        """Full regression test for the mass-false-delisting hazard: a
        mocked fully-empty chunk must not call _record_fetch_result at all
        for any of its symbols, at the scan_* method level."""
        d = _make_discovery()
        symbols = [f'S{i}' for i in range(15)]
        with patch.object(d, '_get_batch_history', return_value={s: None for s in symbols}), \
             patch.object(d, '_record_fetch_result') as mock_record:
            d.scan_unusual_volume(symbols, top_n=10)
        mock_record.assert_not_called()


class TestOmittedTickerCountsAsEmpty:
    def test_omitted_symbol_gets_real_empty_dataframe_not_none(self):
        """yf.download can omit a failed ticker from its response entirely
        (not just NaN-fill it). An omitted-but-genuinely-dead symbol inside
        an otherwise-healthy chunk must still be individually recordable —
        NOT silently fall through to the chunk-failure None default, which
        would make SQ/WBA-class symbols un-delistable again through this
        new fetch path."""
        symbols = [f'S{i}' for i in range(12)]
        healthy, omitted = symbols[:-1], symbols[-1]
        resp = _multiindex_download_response({s: _healthy_fields() for s in healthy})
        with patch('stock_discovery.yf.download', return_value=resp):
            result = stock_discovery._fetch_batch_history(symbols, period='5d')
        assert result[omitted] is not None
        assert result[omitted].empty
        for s in healthy:
            assert not result[s].empty


class TestTinyChunkNotClassifiedAsInfrastructureFailure:
    def test_tiny_chunk_majority_empty_classified_per_symbol(self):
        """A chunk of 3 symbols where 2 come back empty (empty_fraction=0.67
        > 0.5) must NOT trigger the chunk-level None sentinel — below the
        len(chunk) >= 10 floor, each symbol is classified individually."""
        resp = _multiindex_download_response({
            'A': _healthy_fields(), 'B': _all_nan_fields(), 'C': _all_nan_fields(),
        })
        with patch('stock_discovery.yf.download', return_value=resp):
            result = stock_discovery._fetch_batch_history(['A', 'B', 'C'], period='5d')
        assert result['A'] is not None and not result['A'].empty
        assert result['B'] is not None and result['B'].empty
        assert result['C'] is not None and result['C'].empty


class TestBatchHistoryMemoizationKeyOrder:
    def test_same_symbols_different_order_hits_cache(self):
        d = _make_discovery()
        with patch('stock_discovery._fetch_batch_history') as mock_fetch:
            mock_fetch.return_value = {'AAA': pd.DataFrame(), 'BBB': pd.DataFrame()}
            d._get_batch_history(['AAA', 'BBB'], period='5d')
            d._get_batch_history(['BBB', 'AAA'], period='5d')
        mock_fetch.assert_called_once()


# ── sprint04 F4: delisted-skip + _record_fetch_result wiring in scan_* ─────────

class TestScanMethodsDelistedWiring:
    def test_delisted_symbol_excluded_from_candidates_scan_momentum(self):
        d = _make_discovery()
        d._delisted.add('DEAD')
        with patch.object(d, '_get_batch_history') as mock_batch:
            mock_batch.return_value = {}
            d.scan_momentum(['DEAD'], top_n=10)
        mock_batch.assert_called_once_with([], period='1mo')

    def test_second_failure_on_fresh_instance_delists_via_scan_gap_moves(self, tmp_path):
        """Same cross-instance-persistence shape as TestDelistedFailureThreshold,
        exercised through an actual scan_* method rather than
        _record_fetch_result directly."""
        fake_cache_path = tmp_path / 'delisted_cache.json'
        with patch.object(stock_discovery, '_DELISTED_CACHE_PATH', fake_cache_path):
            d1 = stock_discovery.StockDiscovery()
            with patch.object(d1, '_get_batch_history', return_value={'ZZZZ': pd.DataFrame()}):
                d1.scan_gap_moves(['ZZZZ'], top_n=10)
            assert 'ZZZZ' not in d1._delisted

            d2 = stock_discovery.StockDiscovery()  # fresh instance, reloads from disk
            with patch.object(d2, '_get_batch_history', return_value={'ZZZZ': pd.DataFrame()}):
                d2.scan_gap_moves(['ZZZZ'], top_n=10)
            assert 'ZZZZ' in d2._delisted

            d3 = stock_discovery.StockDiscovery()  # third instance, reloads _delisted
            with patch.object(d3, '_get_batch_history') as mock_batch:
                mock_batch.return_value = {}
                d3.scan_gap_moves(['ZZZZ'], top_n=10)
            mock_batch.assert_called_once_with([], period='5d')

    def test_chunk_level_none_not_recorded_scan_oversold(self):
        d = _make_discovery()
        with patch.object(d, '_get_batch_history', return_value={'AAPL': None}), \
             patch.object(d, '_record_fetch_result') as mock_record:
            d.scan_oversold(['AAPL'], top_n=10)
        mock_record.assert_not_called()

    def test_wba_sq_shaped_regression_two_sessions_then_excluded(self, tmp_path):
        """A WBA/SQ-shaped fixture: a hardcoded-list symbol that always
        returns empty history. Delisted after the 2nd failing session; a
        third fresh instance's scan excludes it from candidates entirely —
        the actual regression this sprint's F4 fix targets."""
        fake_cache_path = tmp_path / 'delisted_cache.json'
        with patch.object(stock_discovery, '_DELISTED_CACHE_PATH', fake_cache_path):
            for _ in range(2):
                d = stock_discovery.StockDiscovery()
                with patch.object(d, '_get_batch_history', return_value={'WBA': pd.DataFrame()}):
                    d.scan_unusual_volume(['WBA'], top_n=10)

            d_final = stock_discovery.StockDiscovery()
            assert 'WBA' in d_final._delisted
            with patch.object(d_final, '_get_batch_history') as mock_batch:
                mock_batch.return_value = {}
                d_final.scan_unusual_volume(['WBA'], top_n=10)
            mock_batch.assert_called_once_with([], period='10d')


# ── Sprint02 D1 — pandas 3.0 read_html regression guard ────────────────────────
#
# pandas 3.0 raises FileNotFoundError when pd.read_html() is given a raw string
# (it's treated as a file path). Before the fix, get_index_constituents() and
# get_premarket_movers() silently swallowed that exception in their except blocks
# and returned []/empty results — so a naive "no exception raised" test would
# have passed on the broken code too. These tests assert non-empty parsed
# output specifically to catch that regression.

_CANNED_INDEX_HTML = """
<table>
<tr><th>Symbol</th><th>Name</th></tr>
<tr><td>AAPL</td><td>Apple</td></tr>
<tr><td>BRK.B</td><td>Berkshire</td></tr>
</table>
<table>
<tr><th>Ticker</th><th>Name</th></tr>
<tr><td>NVDA</td><td>Nvidia</td></tr>
</table>
"""


class TestReadHtmlWrapping:
    def test_get_index_constituents_returns_nonempty_under_pandas3(self):
        """S&P500 (table[0]['Symbol']), NASDAQ100 (table with 'Ticker' column),
        and Dow 30 (table with 'Symbol' column) must all parse successfully —
        the same canned HTML round-trips through all three lookups."""
        d = _make_discovery()
        mock_resp = MagicMock()
        mock_resp.text = _CANNED_INDEX_HTML
        mock_resp.status_code = 200
        with patch('stock_discovery.requests.get', return_value=mock_resp):
            constituents = d.get_index_constituents()
        # AAPL/BRK-B come from the S&P500 parse, NVDA from NASDAQ100, plus the
        # hardcoded 90-symbol popular-stocks list appended unconditionally.
        assert 'AAPL' in constituents
        assert 'BRK-B' in constituents  # '.' replaced with '-'
        assert 'NVDA' in constituents
        assert len(constituents) > 90  # more than just the popular-stocks fallback

    def test_get_premarket_movers_returns_nonempty_under_pandas3(self):
        d = _make_discovery()
        mock_resp = MagicMock()
        mock_resp.text = _CANNED_INDEX_HTML
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        with patch('stock_discovery.requests.get', return_value=mock_resp):
            movers = d.get_premarket_movers(top_n=10)
        assert movers == ['NVDA']

    def test_malformed_html_still_degrades_gracefully(self):
        """Negative path: an exception during fetch/parse must still be caught,
        not propagate — existing except-block behavior is unchanged by the fix."""
        d = _make_discovery()
        with patch('stock_discovery.requests.get', side_effect=Exception('network down')):
            constituents = d.get_index_constituents()
        # No index data parsed, but the popular-stocks fallback still appends.
        # sprint04 F4: 89, not 90 — WBA (went private) was removed with no
        # replacement; SQ was swapped in-place for XYZ (Block, Inc.).
        assert len(constituents) == 89


# ── Sprint02 D2 — empty/low-yield scan must not poison the TTL cache ───────────

class TestCacheGating:
    def test_below_floor_scan_not_cached(self, tmp_path):
        d = _make_discovery()
        d.discovered_stocks = ['AAPL', 'MSFT']  # 2 < default floor of 5
        fake_cache_path = tmp_path / 'discovery_cache.json'
        with patch.object(stock_discovery, '_DISCOVERY_CACHE_PATH', fake_cache_path), \
             patch('builtins.open', MagicMock()):
            d.save_opportunities()
        assert not fake_cache_path.exists()

    def test_at_or_above_floor_scan_is_cached(self, tmp_path):
        d = _make_discovery()
        d.discovered_stocks = ['A', 'B', 'C', 'D', 'E']  # == default floor
        fake_cache_path = tmp_path / 'discovery_cache.json'
        with patch.object(stock_discovery, '_DISCOVERY_CACHE_PATH', fake_cache_path), \
             patch('builtins.open', MagicMock()):
            d.save_opportunities()
        assert fake_cache_path.exists()

    def test_low_yield_warning_logged(self, tmp_path, caplog):
        import logging as _logging
        d = _make_discovery()
        d.discovered_stocks = ['AAPL']
        fake_cache_path = tmp_path / 'discovery_cache.json'
        with patch.object(stock_discovery, '_DISCOVERY_CACHE_PATH', fake_cache_path), \
             patch('builtins.open', MagicMock()), \
             caplog.at_level(_logging.WARNING):
            d.save_opportunities()
        assert any('Low-yield scan' in r.getMessage() for r in caplog.records)

    def test_env_override_respected(self, tmp_path):
        """A lower floor set via MIN_CACHEABLE_OPPORTUNITIES allows a smaller
        scan to still be cached."""
        original = stock_discovery._MIN_CACHEABLE_OPPORTUNITIES
        try:
            stock_discovery._MIN_CACHEABLE_OPPORTUNITIES = 1
            d = _make_discovery()
            d.discovered_stocks = ['AAPL']
            fake_cache_path = tmp_path / 'discovery_cache.json'
            with patch.object(stock_discovery, '_DISCOVERY_CACHE_PATH', fake_cache_path), \
                 patch('builtins.open', MagicMock()):
                d.save_opportunities()
            assert fake_cache_path.exists()
        finally:
            stock_discovery._MIN_CACHEABLE_OPPORTUNITIES = original

    def test_end_to_end_low_yield_scan_does_not_poison_next_read(self, tmp_path):
        """Amendment (required): exercise the actual read path, not just the
        write-skip, to catch a mismatch between the gated cache artifact and
        the unconditional human-readable log — a bug here would mean D2's fix
        doesn't actually stop the poisoning even though the write looks gated."""
        fake_cache_path = tmp_path / 'discovery_cache.json'
        with patch.object(stock_discovery, '_DISCOVERY_CACHE_PATH', fake_cache_path):
            d = _make_discovery()
            d.discovered_stocks = []  # 0 < floor
            with patch('builtins.open', MagicMock()):
                d.save_opportunities()

            # Simulate the next session: a fresh read of the cache the loader
            # actually consumes.
            result = stock_discovery._load_discovery_cache()
        assert result is None, (
            "the loader must report no valid cache after a low-yield scan — "
            "if this fails, the read path is consuming the wrong artifact"
        )


# ── D5a: delisted-cache consecutive-failure tracking ───────────────────────────

class TestDelistedFailureThreshold:
    def test_single_failure_does_not_delist(self, tmp_path):
        fake_cache_path = tmp_path / 'delisted_cache.json'
        with patch.object(stock_discovery, '_DELISTED_CACHE_PATH', fake_cache_path):
            d = stock_discovery.StockDiscovery()
            d._record_fetch_result('ZZZZ', success=False)
            assert 'ZZZZ' not in d._delisted

    def test_second_failure_on_fresh_instance_delists(self, tmp_path):
        """The counter must survive across separate StockDiscovery instances
        (simulating separate ~30-min sessions / process restarts) — an
        in-memory-only counter would never accumulate past 1 before the
        instance that saw it is discarded."""
        fake_cache_path = tmp_path / 'delisted_cache.json'
        with patch.object(stock_discovery, '_DELISTED_CACHE_PATH', fake_cache_path):
            d1 = stock_discovery.StockDiscovery()
            d1._record_fetch_result('ZZZZ', success=False)
            assert 'ZZZZ' not in d1._delisted

            d2 = stock_discovery.StockDiscovery()  # fresh instance, reloads from disk
            d2._record_fetch_result('ZZZZ', success=False)
            assert 'ZZZZ' in d2._delisted

    def test_success_resets_counter_and_reset_persists(self, tmp_path):
        """A prior failure, then a success, then one more failure on a new
        instance should NOT reach threshold — the reset-to-0 must itself
        persist, not just apply in-memory for the instance that saw the
        success."""
        fake_cache_path = tmp_path / 'delisted_cache.json'
        with patch.object(stock_discovery, '_DELISTED_CACHE_PATH', fake_cache_path):
            d1 = stock_discovery.StockDiscovery()
            d1._record_fetch_result('ZZZZ', success=False)  # count=1
            d1._record_fetch_result('ZZZZ', success=True)   # reset to 0, persisted

            d2 = stock_discovery.StockDiscovery()
            d2._record_fetch_result('ZZZZ', success=False)  # count=1 again, not 2
            assert 'ZZZZ' not in d2._delisted

    def test_threshold_env_override_respected(self, tmp_path):
        fake_cache_path = tmp_path / 'delisted_cache.json'
        original = stock_discovery._DELISTED_FAILURE_THRESHOLD
        try:
            stock_discovery._DELISTED_FAILURE_THRESHOLD = 1
            with patch.object(stock_discovery, '_DELISTED_CACHE_PATH', fake_cache_path):
                d = stock_discovery.StockDiscovery()
                d._record_fetch_result('ZZZZ', success=False)
                assert 'ZZZZ' in d._delisted
        finally:
            stock_discovery._DELISTED_FAILURE_THRESHOLD = original

    def test_fail_counts_sidecar_does_not_clobber_delisted_set(self, tmp_path):
        """_save_delisted_cache is also called by _mark_delisted with only the
        delisted set — must preserve the fail_counts sidecar already on disk
        rather than wiping it."""
        fake_cache_path = tmp_path / 'delisted_cache.json'
        with patch.object(stock_discovery, '_DELISTED_CACHE_PATH', fake_cache_path):
            d = stock_discovery.StockDiscovery()
            d._record_fetch_result('AAAA', success=False)  # count=1, persisted
            d._mark_delisted('BBBB')  # unrelated symbol, marked directly

            reloaded_counts = stock_discovery._load_fail_counts()
            reloaded_delisted = stock_discovery._load_delisted_cache()
        assert reloaded_counts.get('AAAA') == 1
        assert 'BBBB' in reloaded_delisted


# ── D6a: discovery-health alerts ────────────────────────────────────────────────

class TestDiscoveryHealthAlerts:
    def test_low_yield_scan_fires_alert(self, tmp_path):
        d = _make_discovery()
        d.discovered_stocks = ['AAPL']  # 1 < default floor of 5
        fake_cache_path = tmp_path / 'discovery_cache.json'
        with patch.object(stock_discovery, '_DISCOVERY_CACHE_PATH', fake_cache_path), \
             patch('builtins.open', MagicMock()), \
             patch('alerts.alert_discovery_low_yield') as mock_alert:
            d.save_opportunities()
        mock_alert.assert_called_once_with(1, stock_discovery._MIN_CACHEABLE_OPPORTUNITIES)

    def test_at_or_above_floor_scan_does_not_fire_alert(self, tmp_path):
        d = _make_discovery()
        d.discovered_stocks = ['A', 'B', 'C', 'D', 'E']  # == default floor
        fake_cache_path = tmp_path / 'discovery_cache.json'
        with patch.object(stock_discovery, '_DISCOVERY_CACHE_PATH', fake_cache_path), \
             patch('builtins.open', MagicMock()), \
             patch('alerts.alert_discovery_low_yield') as mock_alert:
            d.save_opportunities()
        mock_alert.assert_not_called()

    def test_all_curated_fetches_failing_fires_alert(self):
        """All three curated index fetches (S&P 500 / NASDAQ 100 / Dow 30)
        failing — universe falls back to the hardcoded popular-stocks list
        only — must fire the discovery_curated_fetch_failed alert."""
        d = _make_discovery()
        with patch('stock_discovery.requests.get', side_effect=Exception('network down')), \
             patch('alerts.alert_discovery_curated_fetch_failed') as mock_alert:
            constituents = d.get_index_constituents()
        assert len(constituents) == 89  # popular-stocks fallback only (sprint04 F4: WBA removed)
        mock_alert.assert_called_once()

    def test_partial_curated_fetch_success_does_not_fire_alert(self):
        """At least one curated index parsing successfully means it's not a
        full fallback-only universe — no alert."""
        d = _make_discovery()
        mock_resp = MagicMock()
        mock_resp.text = _CANNED_INDEX_HTML
        mock_resp.status_code = 200
        with patch('stock_discovery.requests.get', return_value=mock_resp), \
             patch('alerts.alert_discovery_curated_fetch_failed') as mock_alert:
            d.get_index_constituents()
        mock_alert.assert_not_called()


# ── sprint04 F2.1: funnel-stage logging ─────────────────────────────────────────

class TestRecordFunnelStage:
    def test_near_misses_sorted_higher_is_better(self):
        d = _make_discovery()
        evaluated = [('A', 1.5), ('B', 4.0), ('C', 2.5), ('D', 0.5)]
        passed = ['B']  # only B crossed the threshold
        d._record_funnel_stage('unusual_volume', universe_in=4, evaluated=evaluated,
                                passed=passed, higher_is_better=True)
        stage = d._funnel['unusual_volume']
        assert stage['universe_in'] == 4
        assert stage['evaluated'] == 4
        assert stage['passed'] == 1
        # near misses = evaluated minus passed, sorted descending (higher is better)
        assert [s for s, _ in stage['near_misses']] == ['C', 'A', 'D']

    def test_near_misses_sorted_lower_is_better(self):
        d = _make_discovery()
        evaluated = [('A', 35.0), ('B', 10.0), ('C', 28.0), ('D', 60.0)]
        passed = ['B']  # RSI < 30 passes; B=10 is the only passer
        d._record_funnel_stage('oversold', universe_in=4, evaluated=evaluated,
                                passed=passed, higher_is_better=False)
        stage = d._funnel['oversold']
        # near misses sorted ascending (lower is better) among non-passers
        assert [s for s, _ in stage['near_misses']] == ['C', 'A', 'D']

    def test_near_misses_capped_at_ten(self):
        d = _make_discovery()
        evaluated = [(f'S{i}', float(i)) for i in range(20)]
        d._record_funnel_stage('momentum', universe_in=20, evaluated=evaluated,
                                passed=[], higher_is_better=True)
        assert len(d._funnel['momentum']['near_misses']) == 10


class TestLogFunnelSummary:
    def test_one_info_line_covers_all_present_stages(self, caplog):
        import logging
        d = _make_discovery()
        d._record_funnel_stage('unusual_volume', 100, [('A', 1.0)], ['A'], higher_is_better=True)
        d._record_funnel_stage('breakout', 100, [('B', 1.0)], [], higher_is_better=True)
        with caplog.at_level(logging.INFO):
            d._log_funnel_summary()
        funnel_lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith('FUNNEL:')]
        assert len(funnel_lines) == 1
        assert 'unusual_volume=1' in funnel_lines[0]
        assert 'breakout=0' in funnel_lines[0]

    def test_debug_dump_scoped_to_lowest_survivor_stage(self, caplog):
        import logging
        d = _make_discovery()
        d._record_funnel_stage('unusual_volume', 100, [('A', 1.0), ('B', 2.0)], ['A', 'B'], higher_is_better=True)
        d._record_funnel_stage('breakout', 100, [('C', 1.0), ('D', 2.0)], [], higher_is_better=True)
        with caplog.at_level(logging.DEBUG):
            d._log_funnel_summary()
        debug_lines = [r.getMessage() for r in caplog.records
                       if r.getMessage().startswith('FUNNEL nearest-miss')]
        assert len(debug_lines) == 1
        assert '@ breakout' in debug_lines[0]  # breakout has 0 survivors, the collapse point

    def test_no_stages_recorded_is_a_noop(self, caplog):
        import logging
        d = _make_discovery()
        with caplog.at_level(logging.INFO):
            d._log_funnel_summary()
        assert not any(r.getMessage().startswith('FUNNEL:') for r in caplog.records)


class TestScanMethodsPopulateFunnel:
    def test_scan_unusual_volume_records_funnel_stage(self):
        d = _make_discovery()
        ticker = _ticker_mock(volume_ratio=3.0)
        with patch.object(d, '_get_batch_history', return_value={'AAPL': ticker.history.return_value}), \
             patch('yfinance.Ticker', return_value=ticker):
            d.scan_unusual_volume(['AAPL'], top_n=10)
        assert 'unusual_volume' in d._funnel
        assert d._funnel['unusual_volume']['universe_in'] == 1
        assert d._funnel['unusual_volume']['passed'] == 1

    def test_scan_breakouts_records_funnel_stage(self):
        d = _make_discovery()
        breakout_ticker = TestScanBreakoutsRespectLiquidity()._breakout_ticker(liquid=True)
        with patch.object(d, '_get_batch_history', return_value={'NVDA': breakout_ticker.history.return_value}), \
             patch('yfinance.Ticker', return_value=breakout_ticker):
            d.scan_breakouts(['NVDA'], top_n=10)
        assert 'breakout' in d._funnel
        assert d._funnel['breakout']['passed'] == 1


# ── sprint04 F2.2: shared pipeline refactor ─────────────────────────────────────

class TestScanPipelineRefactor:
    def test_discover_opportunities_delegates_to_build_and_run(self):
        """discover_opportunities must delegate to build_scan_universe() +
        run_scan_pipeline() rather than duplicating scan logic — this is
        what makes the offline replay tool trustworthy as a regression check."""
        d = _make_discovery()
        with patch.object(d, 'build_scan_universe', return_value=['AAPL']) as mock_build, \
             patch.object(d, 'run_scan_pipeline', return_value={
                 'universe_size': 1, 'ranked_stocks': ['AAPL'],
                 'opportunities': {}, 'funnel': {}}) as mock_run, \
             patch.object(d, 'save_opportunities'), \
             patch('stock_discovery._load_discovery_cache', return_value=None):
            d.discover_opportunities(max_stocks=10, use_cache=False)
        mock_build.assert_called_once_with(deep_scan=False)
        mock_run.assert_called_once_with(['AAPL'])

    def test_run_scan_pipeline_never_saves_or_caches(self):
        """run_scan_pipeline must have zero persisted side effects — this is
        the actual read-only guarantee the replay tool depends on."""
        d = _make_discovery()
        mock_resp = MagicMock()
        mock_resp.text = _CANNED_INDEX_HTML
        mock_resp.status_code = 200
        with patch('stock_discovery.requests.get', return_value=mock_resp), \
             patch('yfinance.Ticker', return_value=MagicMock(history=MagicMock(
                 return_value=__import__('pandas').DataFrame()))), \
             patch.object(d, 'save_opportunities') as mock_save, \
             patch('stock_discovery._save_discovery_cache') as mock_cache:
            d.run_scan_pipeline(['AAPL'])
        mock_save.assert_not_called()
        mock_cache.assert_not_called()

    def test_run_scan_pipeline_returns_expected_keys(self):
        d = _make_discovery()
        with patch.object(d, 'scan_unusual_volume', return_value=[]), \
             patch.object(d, 'scan_breakouts', return_value=[]), \
             patch.object(d, 'scan_oversold', return_value=[]), \
             patch.object(d, 'scan_momentum', return_value=[]), \
             patch.object(d, 'scan_gap_moves', return_value=[]), \
             patch.object(d, 'scan_mean_reversion', return_value=[]), \
             patch.object(d, 'get_premarket_movers', return_value=[]), \
             patch.object(d, 'filter_by_liquidity', return_value=[]), \
             patch.object(d, 'rank_opportunities', return_value=[]):
            result = d.run_scan_pipeline(['AAPL', 'MSFT'])
        assert set(result.keys()) == {'universe_size', 'ranked_stocks', 'opportunities', 'funnel'}
        assert result['universe_size'] == 2
        assert 'dedup' in result['funnel']
        assert 'final_liquidity' in result['funnel']
        assert 'ranked' in result['funnel']

    def test_get_premarket_movers_records_survivor_count_only(self):
        d = _make_discovery()
        mock_resp = MagicMock()
        mock_resp.text = _CANNED_INDEX_HTML
        mock_resp.status_code = 200
        with patch('stock_discovery.requests.get', return_value=mock_resp):
            d.get_premarket_movers(top_n=10)
        assert 'premarket' in d._funnel
        assert d._funnel['premarket']['near_misses'] == []
