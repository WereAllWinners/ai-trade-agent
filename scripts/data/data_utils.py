import random
import logging
from datasets import Dataset
import yfinance as yf
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_politician_trades(politicians):
    """Fetch stocks traded by politicians (placeholder - implement with real data source)."""
    # TODO: Integrate with real politician trading data API
    # For now, return dummy data
    pol_stocks = ['NVDA', 'TSLA', 'AAPL', 'MSFT']  # Example
    logging.info(f"Fetched {len(pol_stocks)} politician-traded stocks")
    return pol_stocks

def discover_stocks():
    """Discover trending stocks."""
    # TODO: Implement real stock discovery logic
    # Could use: top gainers API, volume screeners, etc.
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    logging.info(f"Discovered {len(stocks)} stocks")
    return stocks

def discover_cryptos():
    """Discover trending cryptocurrencies."""
    # TODO: Implement crypto discovery
    cryptos = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    logging.info(f"Discovered {len(cryptos)} cryptos")
    return cryptos

def fetch_data(ticker, period='2y'):
    """Fetch historical data and calculate technical indicators."""
    try:
        df = yf.download(ticker, period=period, progress=False)
        if df.empty:
            return pd.DataFrame()
        
        # Calculate technical indicators
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Return
        df['Return'] = df['Close'].pct_change()
        
        df = df.dropna()
        return df
        
    except Exception as e:
        logging.error(f"Failed to fetch data for {ticker}: {e}")
        return pd.DataFrame()

def generate_trading_dataset(num_samples=10000):
    """Generate trading dataset with technical indicators."""
    logging.info("Generating trading dataset...")
    
    politicians = [
        {'name': 'Nancy Pelosi', 'average_return': 70.9}, 
        {'name': 'David Rouzer', 'average_return': 149.0}
    ]
    
    pol_stocks = fetch_politician_trades(politicians)
    tickers = list(set(discover_stocks() + discover_cryptos() + pol_stocks))  # Remove duplicates
    
    dataset = []
    
    for ticker in tickers:
        logging.info(f"Processing {ticker}...")
        data = fetch_data(ticker, period='2y')
        
        if data.empty or len(data) < 21:
            logging.warning(f"Insufficient data for {ticker}, skipping")
            continue
        
        for i in range(20, len(data) - 1):
            features = data.iloc[i][['MA5', 'MA20', 'Volatility', 'RSI', 'MACD_Hist', 'Volume']].to_dict()
            actual_return = data.iloc[i+1]['Return']
            action = 'buy' if actual_return > 0.01 else 'sell' if actual_return < -0.01 else 'hold'
            
            # Generate reasoning
            in_pol_stocks = "Yes" if ticker in pol_stocks else "No"
            reasoning = f"Based on RSI {features['RSI']:.2f} ({'overbought' if features['RSI'] > 70 else 'oversold' if features['RSI'] < 30 else 'neutral'}), MACD histogram {features['MACD_Hist']:.4f} ({'bullish' if features['MACD_Hist'] > 0 else 'bearish'}), and volatility {features['Volatility']:.4f}, the predicted return is {actual_return:.4f}, suggesting {action}. Politician-tracked: {in_pol_stocks}."
            
            prompt = f"Analyze ticker {ticker}: Features {features}. Politician stocks: {in_pol_stocks}. Output decision (buy/sell/hold), predicted return, and reasoning."
            completion = f"Decision: {action}. Predicted return: {actual_return:.4f}. Reasoning: {reasoning}"
            
            dataset.append({
                "messages": [
                    {"role": "system", "content": "You are a financial trading AI expert."},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion}
                ]
            })
    
    random.shuffle(dataset)
    dataset = dataset[:num_samples]
    
    logging.info(f"Dataset generated with {len(dataset)} samples")
    return dataset

def load_dataset(filepath):
    """Load dataset from JSONL file."""
    import json
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def clean_data(dataset):
    """Clean and validate dataset."""
    # Remove duplicates, invalid entries, etc.
    cleaned = [d for d in dataset if d.get('messages')]
    logging.info(f"Cleaned dataset: {len(cleaned)} examples")
    return cleaned

def format_for_finetune(dataset):
    """Format dataset for fine-tuning."""
    # Already in correct format from generate_training_dataset
    return dataset