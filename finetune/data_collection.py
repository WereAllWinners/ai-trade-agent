#!/usr/bin/env python3
"""
Enterprise Financial Data Collection System - Fixed & Reliable Version
Handles yfinance flakiness with retries, pre-checks, logging
"""

import os
import json
import time
import random
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict

import pandas as pd
import numpy as np

from dotenv import load_dotenv
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Alpaca imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

load_dotenv()
warnings.filterwarnings('ignore')

class EnterpriseFinancialDataCollector:
    def __init__(self, output_dir='finetune/data/finance_tuning'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fail_log = self.output_dir / 'failed_symbols.log'

        # Alpaca
        try:
            self.alpaca = TradingClient(
                os.getenv('ALPACA_API_KEY'),
                os.getenv('ALPACA_SECRET_KEY'),
                paper=True
            )
            self.has_alpaca = True
            print("✅ Alpaca connected")
        except Exception:
            self.has_alpaca = False
            print("⚠️ Alpaca unavailable")

        # Base symbols (clean)
        self.base_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'AVGO', 'QCOM',
            'JPM', 'BAC', 'GS', 'UNH', 'JNJ', 'XOM', 'CVX', 'SPY', 'QQQ', 'IWM',
            'PLTR', 'HOOD', 'SOFI', 'COIN', 'RIVN', 'LCID', 'UPST', 'ARM', 'SNOW'
        ]

        self.portfolio_symbols = set()

    def log_failure(self, symbol: str, reason: str):
        timestamp = datetime.now().isoformat()
        with open(self.fail_log, 'a', encoding='utf-8') as f:
            f.write(f"{timestamp} | {symbol} | {reason}\n")

    def discover_trending_stocks(self, limit=100) -> List[str]:
        print("\n🔍 Discovering trending/active symbols...")
        trending = set()

        sp500_symbols = []
        sources = [
            "https://raw.githubusercontent.com/fja05680/sp500/master/sp500.csv",
            "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        ]

        for url in sources:
            try:
                df = pd.read_csv(url)
                col = next((c for c in df.columns if 'symbol' in c.lower()), None)
                if col:
                    sp500_symbols = df[col].str.replace('.', '-').dropna().unique().tolist()
                if len(sp500_symbols) > 400:
                    print(f"  ✓ Loaded {len(sp500_symbols)} symbols from source")
                    break
            except Exception as e:
                print(f"  Source failed: {e}")

        if not sp500_symbols:
            print("  ⚠️ Falling back to base + known symbols")
            sp500_symbols = self.base_symbols[:]

        candidates = sp500_symbols[:180]
        volume_data = []
        for sym in candidates:
            try:
                hist = yf.Ticker(sym).history(period='10d')
                if len(hist) >= 5:
                    avg_vol = hist['Volume'].mean()
                    last_vol = hist['Volume'].iloc[-1]
                    ratio = last_vol / avg_vol if avg_vol > 0 else 0
                    change = abs((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0])
                    volume_data.append({'symbol': sym, 'vol_ratio': ratio, 'change': change})
            except:
                continue

        volume_data.sort(key=lambda x: (x['vol_ratio'], x['change']), reverse=True)
        trending.update([x['symbol'] for x in volume_data[:50]])

        trending.update(self.base_symbols)
        # Simple junk filter
        clean_symbols = [s for s in trending if 1 < len(s) <= 6 and s.isalnum() or '.' in s[:5]]
        return clean_symbols[:limit]

    def analyze_portfolio_performance(self) -> Dict[str, Any]:
        if not self.has_alpaca:
            return {'symbols': [], 'lessons': []}

        print("\n📊 Analyzing Alpaca portfolio...")
        try:
            orders = self.alpaca.get_orders(GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=600))
            trades = defaultdict(list)
            for o in orders:
                if o.filled_at:
                    trades[o.symbol].append({'side': o.side.value, 'price': float(o.filled_avg_price or 0)})

            analysis = {'symbols': [], 'lessons': []}
            sector_perf = defaultdict(lambda: {'w': 0, 'l': 0})

            for sym, tlist in trades.items():
                buys = [t['price'] for t in tlist if t['side'] == 'buy']
                sells = [t['price'] for t in tlist if t['side'] == 'sell']
                if not buys or not sells: continue
                pnl = (np.mean(sells) - np.mean(buys)) / np.mean(buys)
                sector = 'Unknown'
                try:
                    sector = yf.Ticker(sym).info.get('sector', 'Unknown')
                except:
                    pass
                if pnl > 0.04:
                    sector_perf[sector]['w'] += 1
                elif pnl < -0.04:
                    sector_perf[sector]['l'] += 1
                analysis['symbols'].append(sym)

            for sec, perf in sector_perf.items():
                tot = perf['w'] + perf['l']
                if tot >= 3:
                    wr = perf['w'] / tot
                    if wr > 0.65: analysis['lessons'].append(f"Strong in {sec}")
                    if wr < 0.35: analysis['lessons'].append(f"Weak in {sec}")

            self.portfolio_symbols = set(analysis['symbols'])
            print(f"  {len(self.portfolio_symbols)} symbols, {len(analysis['lessons'])} lessons")
            return analysis
        except Exception as e:
            print(f"  Portfolio analysis failed: {e}")
            return {'symbols': [], 'lessons': []}

    def build_symbol_universe(self) -> List[str]:
        print("\n🌎 Building universe...")
        all_sym = set(self.base_symbols)
        all_sym.update(self.portfolio_symbols)
        trending = self.discover_trending_stocks(limit=140)
        all_sym.update(trending)
        symbols = sorted(list(all_sym))
        print(f"  🎯 {len(symbols)} symbols")
        return symbols

    def get_macro_context(self) -> Dict[str, Any]:
        try:
            vix_hist = yf.Ticker('^VIX').history(period='5d')
            if vix_hist.empty:
                raise ValueError("No VIX data")
            vix = float(vix_hist['Close'].iloc[-1])
            regime = 'high_vol' if vix > 25 else 'normal_vol' if vix > 15 else 'low_vol'
            return {'vix': vix, 'regime': regime}
        except Exception as e:
            print(f"  Macro fetch failed: {e} → using defaults")
            return {'vix': 18.0, 'regime': 'normal'}

    @retry(stop=stop_after_attempt(4),
           wait=wait_exponential(multiplier=1, min=4, max=45),
           retry=retry_if_exception_type(Exception))
    def safe_history(self, ticker: yf.Ticker, period: str = "max") -> pd.DataFrame:
        hist = ticker.history(period=period)
        if hist.empty:
            raise ValueError("Empty history returned")
        return hist

    def collect_stock_data_comprehensive(self, symbol: str, hist_df=None, asof=None) -> Optional[Dict]:
        try:
            ticker = yf.Ticker(symbol)

            # Pre-check: does this ticker return anything recent?
            test_hist = ticker.history(period="5d")
            if test_hist.empty:
                self.log_failure(symbol, "no recent 5d data")
                return None

            hist = hist_df if hist_df is not None else self.safe_history(ticker)
            if asof:
                hist = hist.loc[:pd.Timestamp(asof)]
            if len(hist) < 60:
                self.log_failure(symbol, f"insufficient bars ({len(hist)})")
                return None

            # Minimal technicals (expand as needed)
            close = hist['Close']
            tech = {
                'rsi14': 100 - (100 / (1 + (close.diff().clip(lower=0).rolling(14).mean() /
                                        -close.diff().clip(upper=0).rolling(14).mean()))).iloc[-1]
            }

            price = float(close.iloc[-1])
            ret_20 = (price - close.iloc[-21]) / close.iloc[-21] if len(close) >= 21 else 0.0
            vol_20 = close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) if len(close) >= 20 else 0.0

            return {
                'symbol': symbol,
                'timestamp': hist.index[-1].isoformat(),
                'price': price,
                'technicals': tech,
                'fundamentals': {},  # add your full version here
                'performance': {'ret_20d': ret_20, 'vol_20d': vol_20},
                'was_traded': symbol in self.portfolio_symbols
            }
        except Exception as e:
            self.log_failure(symbol, str(e)[:150])
            return None

    def generate_training_example(self, data: Dict, macro: Dict, portfolio: Dict) -> Optional[Dict]:
        # Minimal version - replace with your full prompt logic
        symbol = data['symbol']
        score = 0
        if data['technicals'].get('rsi14', 50) < 35: score += 2

        action = "BUY" if score > 1 else "HOLD"

        input_text = f"Analyze {symbol} for trading decision.\nPrice: ${data['price']:.2f}"
        output_text = f"Decision: {action}\nConfidence: 65%"

        return {
            'input': input_text,
            'output': output_text,
            'metadata': {'symbol': symbol, 'action': action}
        }

    def generate_historical_examples(self, symbol: str, target: int = 40) -> List[Dict]:
        examples = []
        try:
            full_hist = yf.Ticker(symbol).history(period="max")
            if len(full_hist) < 250:
                return []

            step = max(5, len(full_hist) // target)
            for i in range(180, len(full_hist)-20, step):
                snap_date = full_hist.index[i]
                snap_data = self.collect_stock_data_comprehensive(symbol, hist_df=full_hist, asof=snap_date)
                if snap_data:
                    ex = self.generate_training_example(snap_data, self.get_macro_context(), {})
                    if ex:
                        examples.append(ex)
                time.sleep(1.2)  # small delay inside historical loop
            return examples
        except Exception as e:
            self.log_failure(symbol, f"historical failed: {e}")
            return []

    def collect_all_data(self, historical_per_symbol: int = 40) -> Path:
        print("=" * 70)
        print(" ENTERPRISE FINANCIAL DATA COLLECTION - Fixed Version")
        print("=" * 70)

        portfolio_analysis = self.analyze_portfolio_performance()
        symbols = self.build_symbol_universe()
        macro = self.get_macro_context()

        print(f"\nCollecting for {len(symbols)} symbols (~{len(symbols) * (1 + historical_per_symbol):,} target examples)")

        training_examples = []

        for idx, sym in enumerate(symbols, 1):
            print(f"[{idx:3d}/{len(symbols):3d}] {sym:<8} ", end="", flush=True)

            latest = self.collect_stock_data_comprehensive(sym)
            if latest:
                ex = self.generate_training_example(latest, macro, portfolio_analysis)
                if ex:
                    training_examples.append(ex)
                    print("✓ ", end="")
            else:
                print("✗ ", end="")

            hist_exs = self.generate_historical_examples(sym, historical_per_symbol)
            training_examples.extend(hist_exs)
            print(f"({len(hist_exs)})")

            time.sleep(3.5 + random.uniform(0, 2.5))  # aggressive delay

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        out_path = self.output_dir / f"training_data_{timestamp}.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(training_examples, f, indent=2, ensure_ascii=False)

        print(f"\nDone! {len(training_examples):,} examples → {out_path}")
        if self.fail_log.exists():
            print(f"Check failures: {self.fail_log}")
        return out_path


if __name__ == "__main__":
    collector = EnterpriseFinancialDataCollector()
    collector.collect_all_data(historical_per_symbol=40)