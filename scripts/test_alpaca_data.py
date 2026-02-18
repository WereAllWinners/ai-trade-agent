#!/usr/bin/env python3
import os
import sys
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv('/home/zgx/personal-projects/ai-trade-agent/.env')

api_key = os.getenv('ALPACA_API_KEY')
secret_key = os.getenv('ALPACA_SECRET_KEY')

data_client = StockHistoricalDataClient(api_key, secret_key)

for symbol in ["AAPL", "MSFT"]:
    print(f"\n{'='*50}")
    print(f"Testing {symbol}")
    print(f"{'='*50}")
    
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=datetime.now() - timedelta(days=30),
        end=datetime.now()
    )
    
    try:
        bars_data = data_client.get_stock_bars(request)
        
        # Method 1: Use bars_data.df directly
        if hasattr(bars_data, 'df'):
            df = bars_data.df
            print(f"✅ bars_data.df: {type(df)}")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {df.columns.tolist()}")
            if len(df) > 0:
                print(f"   Last close: ${df['close'].iloc[-1]:.2f}")
                print(f"   Index: {df.index.name}")
        
        # Method 2: Check .data structure
        print(f"\n.data type: {type(bars_data.data)}")
        print(f".data keys: {list(bars_data.data.keys())}")
        if symbol in bars_data.data:
            bars_list = bars_data.data[symbol]
            print(f"bars_data.data['{symbol}'] is a list with {len(bars_list)} items")
            if len(bars_list) > 0:
                print(f"First item type: {type(bars_list[0])}")
                print(f"First item: {bars_list[0]}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
