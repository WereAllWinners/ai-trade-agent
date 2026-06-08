#!/usr/bin/env python3
"""
indicators.py — Shared technical indicator utilities.

Provides MA/ATR calculation and consistent return-window metrics that can be
imported by both autonomous_agent.py and strategy_evolver.py without circular
imports.
"""
import logging
from datetime import datetime, timedelta

import pandas as pd


def get_daily_bars_for_ma(symbol: str, alpaca_client=None) -> pd.DataFrame | None:
    """
    Fetch ~300 daily bars for MA/ATR calculation.

    Primary: Alpaca daily bars (300-day window).
    Fallback: yfinance 1-year history.

    Returns None if fewer than 205 rows are available (stock too new for 200MA).
    """
    # Try Alpaca daily bars first
    if alpaca_client is not None:
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            end = datetime.now()
            start = end - timedelta(days=300)
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
            )
            bars_data = alpaca_client.get_stock_bars(req)
            df = bars_data.df
            if not df.empty:
                df = df.reset_index(level='symbol', drop=True) if 'symbol' in df.index.names else df
                df.columns = [c.lower() for c in df.columns]
                if len(df) >= 205:
                    return df
        except Exception as e:
            logging.debug(f"Alpaca daily bars (MA) failed for {symbol}: {e}")

    # Fallback: yfinance
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(period='1y')
        if df.empty:
            return None
        df = df.rename(columns={
            'Open': 'open', 'High': 'high',
            'Low': 'low', 'Close': 'close', 'Volume': 'volume',
        })
        if len(df) >= 205:
            return df
    except Exception as e:
        logging.debug(f"yfinance daily bars (MA) failed for {symbol}: {e}")

    return None


def is_daily_df(df: pd.DataFrame) -> bool:
    """Return True if the DataFrame's index has daily (not intraday) frequency."""
    if df is None or len(df) < 2:
        return False
    idx = df.index
    if hasattr(idx, 'tz_localize'):
        diffs = pd.Series(idx).diff().dropna()
        return diffs.median() >= pd.Timedelta(hours=20)
    return False


def calculate_ma_atr(df: pd.DataFrame, current_price: float) -> dict | None:
    """
    Compute 50MA, 200MA, price-to-MA %, ATR(14), and return windows from a
    daily bar DataFrame.

    Returns None if df is None or has fewer than 200 rows.
    Returns a dict with keys:
        ma50, ma200, pct_from_ma50, pct_from_ma200, atr_14,
        return_5d, return_20d
    Individual return fields may be None if not enough rows.
    """
    if df is None or len(df) < 200:
        return None

    close = df['close']

    ma50  = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1]

    if ma50 == 0 or ma200 == 0:
        return None

    pct_from_ma50  = (current_price - ma50)  / ma50  * 100
    pct_from_ma200 = (current_price - ma200) / ma200 * 100

    # ATR(14): True Range = max(H-L, |H-prev_close|, |L-prev_close|)
    high  = df['high']
    low   = df['low']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean().iloc[-1]

    # Consistent return windows from the same daily series
    return_5d  = _pct_change(close, lookback=5)
    return_20d = _pct_change(close, lookback=20)

    return {
        'ma50':          float(ma50),
        'ma200':         float(ma200),
        'pct_from_ma50':  float(pct_from_ma50),
        'pct_from_ma200': float(pct_from_ma200),
        'atr_14':        float(atr_14) if not pd.isna(atr_14) else None,
        'return_5d':     return_5d,
        'return_20d':    return_20d,
    }


def _pct_change(series: pd.Series, lookback: int) -> float | None:
    """Compute % change over the last `lookback` daily bars."""
    if len(series) < lookback + 1:
        return None
    end_price   = series.iloc[-1]
    start_price = series.iloc[-(lookback + 1)]
    if start_price == 0:
        return None
    return float((end_price - start_price) / start_price * 100)
