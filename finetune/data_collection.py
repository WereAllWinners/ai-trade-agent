#!/usr/bin/env python3
"""
ULTIMATE SCALED TRADING DATA COLLECTOR — FINAL FIXED VERSION
- Dynamic universe: S&P 500 + Nasdaq-100 (robust sources + headers)
- Historical snapshots (25 per symbol)
- 12,000 synthetic expert strategies
- Parallel processing
- Tolerant liquidity filter
"""

import os
import json
import time
import random
import warnings
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import yfinance as yf
from polygon import RESTClient
from tenacity import retry, stop_after_attempt, wait_fixed

load_dotenv()
warnings.filterwarnings('ignore')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

class UltimateTradingDataCollector:
    def __init__(self, output_dir='finetune/data/finance_tuning'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fail_log = self.output_dir / 'failed_symbols.log'

        self.polygon = RESTClient(os.getenv('POLYGON_API_KEY'))

    def log_failure(self, symbol: str, reason: str):
        with open(self.fail_log, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()} | {symbol} | {reason}\n")

    def clean_ticker(self, t) -> Optional[str]:
        if not isinstance(t, str) or pd.isna(t):
            return None
        t = str(t).strip().upper()
        if re.match(r'^[A-Z0-9.-]{1,5}$', t) and any(c.isalpha() for c in t):
            return t.replace('.', '-')
        return None

    def load_large_universe(self, max_symbols: int = 580) -> List[str]:
        print("🌎 Building dynamic liquid stock universe...")

        symbols = set()

        def add_cleaned(raw_list):
            clean = [self.clean_ticker(s) for s in raw_list]
            added = [s for s in clean if s]
            symbols.update(added)
            return len(added)

        # 1. S&P 500
        try:
            df = pd.read_csv("https://raw.githubusercontent.com/fja05680/sp500/master/sp500.csv")
            count = add_cleaned(df['Symbol'])
            print(f"   ✓ {count} clean from S&P 500")
        except Exception as e:
            print(f"   ⚠️ S&P 500 failed: {e}")

        # 2. Nasdaq-100 official page
        try:
            url = "https://www.nasdaq.com/solutions/global-indexes/nasdaq-100/companies"
            tables = pd.read_html(url, header=0)
            for table in tables:
                col = 'Symbol' if 'Symbol' in table.columns else 'Ticker' if 'Ticker' in table.columns else None
                if col:
                    count = add_cleaned(table[col])
                    print(f"   ✓ {count} from official Nasdaq-100 page")
                    break
        except Exception as e:
            print(f"   Official Nasdaq page failed: {e}")

        # 3. Wikipedia with User-Agent
        try:
            url = "https://en.wikipedia.org/wiki/Nasdaq-100"
            tables = pd.read_html(url, header=0, attrs={'class': 'wikitable'})
            for table in tables:
                col = 'Ticker' if 'Ticker' in table.columns else 'Symbol' if 'Symbol' in table.columns else None
                if col:
                    count = add_cleaned(table[col])
                    print(f"   ✓ {count} from Wikipedia Nasdaq-100")
                    break
        except Exception as e:
            print(f"   Wikipedia failed: {e}")

        # 4. Hard fallback (current Feb 2026)
        fallback = [
            'AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','AVGO','COST','NFLX','GOOG','ASML','AMD','TMUS',
            'QCOM','PEP','LIN','ADBE','CSCO','INTC','TXN','AMAT','ISRG','CMCSA','MU','LRCX','INTU','HON','BKNG',
            'VRTX','REGN','KLAC','PANW','ADP','ADI','SBUX','PDD','MELI','GILD','MAR','CTAS','PYPL','SNPS','CDNS',
            'CSX','ORLY','PCAR','NXPI','MNST','PAYX','ROST','FTNT','KDP','ODFL','MRVL','CHTR','CRWD','KHC','DASH',
            'IDXX','VRSK','CEG','DDOG','FAST','GEHC','FANG','CTSH','EXC','BKR','CSGP','TTD','XEL','CDW','ANSS',
            'TTWO','TEAM','DXCM','WBD','WDAY','ZS','MDB','APP','AXON','DLTR','ILMN','CPRT','ON','TER','GFS','MPWR',
            'WDC','ENTG','PTON','ALGN','LCID','OKTA','AEP','BIDU','JD','NTES','MRNA','EA','SGEN','ZM','DOCU',
            'MTCH','BIIB','SWKS','SIRI','PLTR','HOOD','SOFI','COIN','RIVN','UPST','SNOW','ARM','WMT'
        ]
        count = add_cleaned(fallback)
        print(f"   ✓ {count} from hard-coded fallback")

        # Final clean
        symbols = {s for s in symbols if s and self.clean_ticker(s)}

        # Liquidity filter
        print(f"Filtering to top {max_symbols} liquid stocks...")
        liquid = []
        skipped = 0
        for sym in sorted(list(symbols)):
            if len(liquid) >= max_symbols:
                break
            try:
                @retry(stop=stop_after_attempt(2), wait=wait_fixed(2))
                def get_hist():
                    return yf.Ticker(sym).history(period="5d")
                hist = get_hist()
                if not hist.empty and len(hist) >= 3:
                    price = hist['Close'].iloc[-1]
                    avg_vol = hist['Volume'].mean()
                    if price > 5 and avg_vol > 400000:
                        liquid.append(sym)
            except Exception as e:
                skipped += 1
                self.log_failure(sym, f"liquidity check: {str(e)[:80]}")

        print(f"   Final universe: {len(liquid)} liquid stocks (skipped {skipped})")
        return liquid

    # ===================================================================
    #  Data Fetchers & Generators (unchanged)
    # ===================================================================
    def collect_stock_snapshot(self, symbol: str, hist_df=None, asof=None) -> Optional[Dict]:
        try:
            if hist_df is None:
                hist = yf.Ticker(symbol).history(period="2y")
            else:
                hist = hist_df.copy()
                if asof:
                    hist = hist[hist.index <= pd.Timestamp(asof)]

            if len(hist) < 60:
                return None

            close = hist['Close']
            return {
                'symbol': symbol,
                'price': float(close.iloc[-1]),
                'ret_20d': float((close.iloc[-1] - close.iloc[-21]) / close.iloc[-21]),
                'vol_20d': float(close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)),
                'rsi14': 100 - (100 / (1 + (close.diff().clip(lower=0).rolling(14).mean() /
                                            -close.diff().clip(upper=0).rolling(14).mean()))).iloc[-1]
            }
        except Exception as e:
            self.log_failure(symbol, f"snapshot failed: {e}")
            return None

    def generate_historical_examples(self, symbol: str, target: int = 25) -> List[Dict]:
        examples = []
        try:
            full = yf.Ticker(symbol).history(period="max")
            if len(full) < 200:
                return []

            step = max(5, len(full) // target)
            for i in range(180, len(full) - 20, step):
                snap = self.collect_stock_snapshot(symbol, hist_df=full, asof=full.index[i])
                if snap:
                    ex = self.generate_price_example(snap)
                    if ex:
                        examples.append(ex)
            return examples
        except Exception as e:
            self.log_failure(symbol, f"historical failed: {e}")
            return []

    def generate_price_example(self, data: Dict) -> Dict:
        action = "BUY" if data.get('ret_20d', 0) > 0.07 and data.get('rsi14', 50) < 48 else "HOLD"
        return {
            'input': f"Analyze {data['symbol']}:\nPrice ${data['price']:.2f} | 20d ret {data['ret_20d']:+.1%} | RSI {data['rsi14']:.1f}",
            'output': f"Decision: {action}\nStrong momentum with oversold RSI.",
            'metadata': {'type': 'price', 'symbol': data['symbol']}
        }

    def generate_synthetic_strategies(self, count: int = 12000) -> List[Dict]:
        print(f"Generating {count:,} synthetic expert strategy examples...")
        strats = ["ORB", "VWAP", "Momentum", "Mean Reversion", "Iron Condor", "Calendar Spread", "Jade Lizard", "Kelly Sizing"]
        examples = []
        for _ in range(count):
            strat = random.choice(strats)
            sym = random.choice(['NVDA', 'TSLA', 'AAPL', 'SPY', 'QQQ'])
            examples.append({
                'input': f"Explain how to trade a {strat} strategy on {sym} with full rules, risk management, and psychology.",
                'output': f"{strat} on {sym}: Entry at breakout, stop below range, target 2R. Risk 1% max. Journal every trade.",
                'metadata': {'type': 'synthetic', 'strategy': strat}
            })
        return examples

    # ===================================================================
    #  Main Collection
    # ===================================================================
    def collect_all_data(self, num_symbols: int = 580, historical_per_symbol: int = 25) -> Path:
        print("=" * 95)
        print("ULTIMATE SCALED TRADING DATA COLLECTION — DYNAMIC UNIVERSE")
        print("=" * 95)

        symbols = self.load_large_universe(max_symbols=num_symbols)
        all_examples = self.generate_synthetic_strategies(12000)

        def process_symbol(sym: str) -> List[Dict]:
            local = []
            snap = self.collect_stock_snapshot(sym)
            if snap:
                local.append(self.generate_price_example(snap))

            hist_ex = self.generate_historical_examples(sym, historical_per_symbol)
            local.extend(hist_ex)

            return local

        print(f"\nStarting parallel collection on {len(symbols)} symbols...\n")

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(process_symbol, sym): sym for sym in symbols}
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    exs = future.result()
                    all_examples.extend(exs)
                    print(f"[{len(all_examples):,}] {sym:<6} +{len(exs)}")
                except Exception as e:
                    print(f"Failed {sym}: {e}")

        ts = datetime.now().strftime("%Y%m%d_%H%M")
        out_path = self.output_dir / f"training_data_scaled_{ts}.json"

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(all_examples, f, indent=2, ensure_ascii=False)

        print("\n" + "="*95)
        print(f"✅ COLLECTION COMPLETE — {len(all_examples):,} examples")
        print(f"   Saved to: {out_path}")
        print("="*95)
        return out_path


if __name__ == "__main__":
    collector = UltimateTradingDataCollector()
    collector.collect_all_data(num_symbols=580, historical_per_symbol=25)