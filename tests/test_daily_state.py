"""
Unit tests for daily_state persistence in db.py.

Tests run fully offline against a temp SQLite file.
"""
import sys
from pathlib import Path
from unittest.mock import patch
from datetime import date

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))


@pytest.fixture()
def db_path(tmp_path):
    import db as _db
    p = tmp_path / 'test_trading.db'
    _db.init_db(p)
    return p


class TestDailyState:
    def test_save_and_load_returns_value(self, db_path):
        import db as _db
        _db.save_daily_start_equity('stock', 10_000.0, db_path=db_path)
        result = _db.load_daily_start_equity('stock', db_path=db_path)
        assert result == 10_000.0

    def test_load_returns_none_for_missing_bot(self, db_path):
        import db as _db
        _db.save_daily_start_equity('stock', 10_000.0, db_path=db_path)
        result = _db.load_daily_start_equity('options', db_path=db_path)
        assert result is None

    def test_load_returns_none_when_nothing_saved(self, db_path):
        import db as _db
        result = _db.load_daily_start_equity('stock', db_path=db_path)
        assert result is None

    def test_load_returns_none_for_different_date(self, db_path):
        import db as _db
        # Save under yesterday's date
        with patch('db.datetime') as mock_dt:
            mock_dt.now.return_value.date.return_value.isoformat.return_value = '2000-01-01'
            _db.save_daily_start_equity('stock', 10_000.0, db_path=db_path)
        # Load under today's date — should find nothing
        result = _db.load_daily_start_equity('stock', db_path=db_path)
        assert result is None

    def test_upsert_overwrites_same_day(self, db_path):
        import db as _db
        _db.save_daily_start_equity('stock', 10_000.0, db_path=db_path)
        _db.save_daily_start_equity('stock', 12_500.0, db_path=db_path)
        result = _db.load_daily_start_equity('stock', db_path=db_path)
        assert result == 12_500.0

    def test_stock_and_options_are_independent(self, db_path):
        import db as _db
        _db.save_daily_start_equity('stock',   10_000.0, db_path=db_path)
        _db.save_daily_start_equity('options',  5_000.0, db_path=db_path)
        assert _db.load_daily_start_equity('stock',   db_path=db_path) == 10_000.0
        assert _db.load_daily_start_equity('options', db_path=db_path) ==  5_000.0
