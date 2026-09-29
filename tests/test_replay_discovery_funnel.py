"""
tests/test_replay_discovery_funnel.py — sprint04 F2.2 offline funnel replay
tool. The core guarantee under test: this tool must be genuinely read-only —
no delisted_cache.json mutation, no discovery cache write, no alerts.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'data'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'tools'))

import stock_discovery
import replay_discovery_funnel as rdf


def _make_discovery():
    with patch('stock_discovery._load_delisted_cache', return_value=set()), \
         patch('stock_discovery._load_fail_counts', return_value={}):
        return stock_discovery.StockDiscovery()


class TestReplayDiscoveryFunnelMain:
    def test_json_output_has_expected_keys(self, capsys, monkeypatch):
        monkeypatch.setattr(sys, 'argv', ['replay_discovery_funnel.py', '--json'])
        fake_disco = _make_discovery()
        with patch.object(rdf, 'StockDiscovery', return_value=fake_disco), \
             patch.object(fake_disco, 'build_scan_universe', return_value=['AAPL']), \
             patch.object(fake_disco, 'run_scan_pipeline', return_value={
                 'universe_size': 1, 'ranked_stocks': ['AAPL'],
                 'opportunities': {}, 'funnel': {}}):
            rdf.main()
        out = capsys.readouterr().out
        data = json.loads(out)
        assert set(data.keys()) == {'universe_size', 'ranked_stocks', 'opportunities', 'funnel'}

    def test_record_fetch_result_neutralized_no_cache_writes(self, monkeypatch):
        """The strongest proof of 'no cache touch': run a full pipeline pass
        including empty-history symbols (which would normally increment
        delisted_cache.json's fail_counts) and assert the save function is
        never called. OHLCV now comes from yf.download (sprint04 F3), not
        yf.Ticker(...).history() — mock that path directly rather than the
        now-dead-for-OHLCV Ticker mock."""
        monkeypatch.setattr(sys, 'argv', ['replay_discovery_funnel.py'])
        import pandas as pd
        fake_disco = _make_discovery()
        mock_resp = MagicMock()
        mock_resp.text = '<table><tr><th>Symbol</th></tr><tr><td>AAPL</td></tr></table>'
        mock_resp.status_code = 200

        # AAPL is entirely absent from the response (omitted, not just
        # NaN-filled) and DEAD is present but all-NaN — both count as empty.
        n = 5
        dates = pd.date_range('2026-01-01', periods=n)
        frame_data = {('DEAD', field): [None] * n
                      for field in ('Open', 'High', 'Low', 'Close', 'Volume')}
        empty_resp = pd.DataFrame(frame_data, index=dates)
        empty_resp.columns = pd.MultiIndex.from_tuples(empty_resp.columns)

        with patch.object(rdf, 'StockDiscovery', return_value=fake_disco), \
             patch.object(fake_disco, 'build_scan_universe', return_value=['AAPL', 'DEAD']), \
             patch('stock_discovery.requests.get', return_value=mock_resp), \
             patch('stock_discovery.yf.download', return_value=empty_resp), \
             patch('stock_discovery._save_delisted_cache') as mock_save:
            rdf.main()
        mock_save.assert_not_called()

    def test_save_opportunities_never_called(self, monkeypatch):
        monkeypatch.setattr(sys, 'argv', ['replay_discovery_funnel.py', '--json'])
        fake_disco = _make_discovery()
        with patch.object(rdf, 'StockDiscovery', return_value=fake_disco), \
             patch.object(fake_disco, 'build_scan_universe', return_value=['AAPL']), \
             patch.object(fake_disco, 'run_scan_pipeline', return_value={
                 'universe_size': 1, 'ranked_stocks': ['AAPL'],
                 'opportunities': {}, 'funnel': {}}), \
             patch.object(fake_disco, 'save_opportunities') as mock_save_opps, \
             patch('stock_discovery._save_discovery_cache') as mock_save_cache:
            rdf.main()
        mock_save_opps.assert_not_called()
        mock_save_cache.assert_not_called()

    def test_record_fetch_result_is_monkeypatched_to_noop(self, monkeypatch):
        monkeypatch.setattr(sys, 'argv', ['replay_discovery_funnel.py', '--json'])
        fake_disco = _make_discovery()
        with patch.object(rdf, 'StockDiscovery', return_value=fake_disco), \
             patch.object(fake_disco, 'build_scan_universe', return_value=[]), \
             patch.object(fake_disco, 'run_scan_pipeline', return_value={
                 'universe_size': 0, 'ranked_stocks': [],
                 'opportunities': {}, 'funnel': {}}):
            rdf.main()
        # after main() runs, the instance's _record_fetch_result must be the no-op
        assert fake_disco._record_fetch_result('AAPL', success=False) is None
