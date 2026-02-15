#!/usr/bin/env python3
"""
Enterprise Financial Data Collection System with Portfolio Learning
Collects high-quality, diverse financial data for model fine-tuning

NEW FEATURES:
- Learns from actual Alpaca portfolio performance
- Dynamic symbol discovery (trending stocks, unusual volume, sector leaders)
- Adaptive sector weighting based on performance
"""

import os
import json
import time
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import defaultdict
from dotenv import load_dotenv

import yfinance as yf
import pandas as pd
import numpy as np

# Alpaca imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderSide, QueryOrderStatus

load_dotenv()
warnings.filterwarnings('ignore')

class EnterpriseFinancialDataCollector:
    """
    High-quality financial data collection with portfolio learning and dynamic discovery.
    """
    
    def __init__(self, output_dir='finetune/data/finance_tuning'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.output_dir / 'cache'
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize Alpaca client
        try:
            self.alpaca = TradingClient(
                os.getenv('ALPACA_API_KEY'),
                os.getenv('ALPACA_SECRET_KEY'),
                paper=True
            )
            self.has_alpaca = True
            print("✅ Alpaca API connected")
        except:
            self.has_alpaca = False
            print("⚠️  Alpaca API not available - skipping portfolio analysis")
        
        # Base symbol universe (fallback/minimum)
        self.base_symbols = {
            'mega_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'],
            'large_cap_tech': ['AMD', 'INTC', 'QCOM', 'AVGO', 'CSCO', 'ORCL', 'CRM'],
            'semiconductors': ['NVDA', 'AMD', 'INTC', 'TSM', 'QCOM', 'AVGO'],
            'ai_ml': ['NVDA', 'GOOGL', 'MSFT', 'META', 'PLTR'],
            'financials': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS'],
            'healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK'],
            'consumer': ['AMZN', 'WMT', 'HD', 'NKE', 'SBUX', 'MCD'],
            'energy': ['XOM', 'CVX', 'COP', 'SLB'],
            'industrials': ['BA', 'CAT', 'GE', 'HON', 'UNP'],
            'etfs': ['SPY', 'QQQ', 'IWM', 'DIA']
        }
        
        # Dynamic symbols (will be populated)
        self.dynamic_symbols = set()
        self.portfolio_symbols = set()
        
        print(f"📊 Initialized data collector")
    
    def discover_trending_stocks(self, limit=50) -> List[str]:
        """
        Discover trending/active stocks dynamically.
        """
        print("\n🔍 Discovering trending stocks...")
        trending = set()
        
        try:
            # Method 1: Most active stocks (high volume)
            print("  📊 Finding high-volume stocks...")
            
            # Get S&P 500 components
            sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            sp500_table = pd.read_html(sp500_url)[0]
            sp500_symbols = sp500_table['Symbol'].str.replace('.', '-').tolist()[:100]  # Top 100 by market cap
            
            # Check volume for each
            volume_data = []
            for symbol in sp500_symbols[:50]:  # Check first 50 to save time
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period='5d')
                    if not hist.empty:
                        avg_volume = hist['Volume'].mean()
                        recent_volume = hist['Volume'].iloc[-1]
                        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 0
                        
                        volume_data.append({
                            'symbol': symbol,
                            'volume_ratio': volume_ratio,
                            'avg_volume': avg_volume
                        })
                except:
                    continue
            
            # Sort by volume ratio and volume
            volume_data.sort(key=lambda x: (x['volume_ratio'], x['avg_volume']), reverse=True)
            high_volume_stocks = [x['symbol'] for x in volume_data[:20]]
            trending.update(high_volume_stocks)
            print(f"    ✓ Found {len(high_volume_stocks)} high-volume stocks")
            
            # Method 2: Biggest movers (price change)
            print("  📈 Finding biggest movers...")
            movers = []
            for symbol in sp500_symbols[:100]:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period='5d')
                    if len(hist) >= 5:
                        price_change = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]
                        movers.append({
                            'symbol': symbol,
                            'change': abs(price_change)
                        })
                except:
                    continue
            
            movers.sort(key=lambda x: x['change'], reverse=True)
            big_movers = [x['symbol'] for x in movers[:15]]
            trending.update(big_movers)
            print(f"    ✓ Found {len(big_movers)} big movers")
            
            # Method 3: Sector leaders
            print("  🏆 Finding sector leaders...")
            sectors = {
                'XLK': 'Technology',
                'XLF': 'Financials', 
                'XLV': 'Healthcare',
                'XLE': 'Energy',
                'XLI': 'Industrials',
                'XLC': 'Communications',
                'XLY': 'Consumer Discretionary',
                'XLP': 'Consumer Staples'
            }
            
            sector_etfs = list(sectors.keys())
            trending.update(sector_etfs)
            
            # Method 4: Recent IPOs and hot stocks (hardcoded but can be updated)
            hot_stocks = [
                'PLTR', 'SNOW', 'COIN', 'RBLX', 'U', 'DKNG', 'SOFI',
                'RIVN', 'LCID', 'NIO', 'GRAB', 'HOOD', 'UPST'
            ]
            trending.update(hot_stocks)
            print(f"    ✓ Added {len(hot_stocks)} hot/growth stocks")
            
            print(f"  ✅ Total trending stocks discovered: {len(trending)}")
            return list(trending)[:limit]
            
        except Exception as e:
            print(f"  ⚠️  Trending discovery failed: {e}")
            return []
    
    def analyze_portfolio_performance(self) -> Dict[str, Any]:
        """
        Analyze actual Alpaca portfolio performance and learn from it.
        """
        if not self.has_alpaca:
            return {'symbols': [], 'lessons': []}
        
        print("\n📊 Analyzing Alpaca portfolio performance...")
        
        try:
            # Get closed orders from last 90 days
            request = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                limit=500
            )
            orders = self.alpaca.get_orders(request)
            
            if not orders:
                print("  ℹ️  No closed orders found")
                return {'symbols': [], 'lessons': []}
            
            print(f"  📋 Analyzing {len(orders)} closed orders...")
            
            # Parse trade history
            trades = defaultdict(list)
            for order in orders:
                symbol = order.symbol
                
                trade = {
                    'symbol': symbol,
                    'side': order.side.value,
                    'qty': float(order.filled_qty) if order.filled_qty else 0,
                    'filled_price': float(order.filled_avg_price) if order.filled_avg_price else 0,
                    'timestamp': order.filled_at,
                    'order_type': order.type.value
                }
                
                trades[symbol].append(trade)
            
            # Analyze each symbol's performance
            analysis = {
                'symbols': [],
                'winners': [],
                'losers': [],
                'lessons': [],
                'sector_performance': defaultdict(lambda: {'wins': 0, 'losses': 0})
            }
            
            for symbol, symbol_trades in trades.items():
                # Match buys and sells
                buys = [t for t in symbol_trades if t['side'] == 'buy']
                sells = [t for t in symbol_trades if t['side'] == 'sell']
                
                if not buys or not sells:
                    continue
                
                # Calculate P&L
                avg_buy_price = np.mean([t['filled_price'] for t in buys])
                avg_sell_price = np.mean([t['filled_price'] for t in sells])
                pnl_percent = (avg_sell_price - avg_buy_price) / avg_buy_price
                
                # Get current data
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    hist = ticker.history(period='3mo')
                    
                    if hist.empty:
                        continue
                    
                    # Calculate what indicators looked like at buy time
                    buy_date = buys[0]['timestamp']
                    
                    trade_analysis = {
                        'symbol': symbol,
                        'pnl_percent': pnl_percent,
                        'num_trades': len(buys) + len(sells),
                        'sector': info.get('sector', 'Unknown'),
                        'avg_buy_price': avg_buy_price,
                        'avg_sell_price': avg_sell_price,
                        'outcome': 'winner' if pnl_percent > 0.05 else 'loser' if pnl_percent < -0.05 else 'neutral'
                    }
                    
                    analysis['symbols'].append(symbol)
                    
                    if pnl_percent > 0.05:
                        analysis['winners'].append(trade_analysis)
                        analysis['sector_performance'][trade_analysis['sector']]['wins'] += 1
                    elif pnl_percent < -0.05:
                        analysis['losers'].append(trade_analysis)
                        analysis['sector_performance'][trade_analysis['sector']]['losses'] += 1
                    
                except Exception as e:
                    print(f"    ⚠️  Could not analyze {symbol}: {e}")
                    continue
            
            # Generate lessons
            if analysis['winners']:
                print(f"  ✅ Winners: {len(analysis['winners'])} trades")
                top_winner = max(analysis['winners'], key=lambda x: x['pnl_percent'])
                analysis['lessons'].append(f"Best trade: {top_winner['symbol']} ({top_winner['sector']}) with {top_winner['pnl_percent']:.1%} gain")
            
            if analysis['losers']:
                print(f"  ❌ Losers: {len(analysis['losers'])} trades")
                worst_loser = min(analysis['losers'], key=lambda x: x['pnl_percent'])
                analysis['lessons'].append(f"Worst trade: {worst_loser['symbol']} ({worst_loser['sector']}) with {worst_loser['pnl_percent']:.1%} loss")
            
            # Sector analysis
            print("\n  📊 Sector Performance:")
            for sector, perf in analysis['sector_performance'].items():
                total = perf['wins'] + perf['losses']
                win_rate = perf['wins'] / total if total > 0 else 0
                print(f"    {sector}: {perf['wins']}W / {perf['losses']}L ({win_rate:.1%} win rate)")
                
                if win_rate > 0.6:
                    analysis['lessons'].append(f"Strength in {sector} sector ({win_rate:.1%} win rate)")
                elif win_rate < 0.4 and total >= 3:
                    analysis['lessons'].append(f"Avoid {sector} sector ({win_rate:.1%} win rate)")
            
            # Save portfolio symbols for inclusion
            self.portfolio_symbols = set(analysis['symbols'])
            print(f"\n  ✅ Portfolio analysis complete: {len(self.portfolio_symbols)} unique symbols traded")
            
            return analysis
            
        except Exception as e:
            print(f"  ⚠️  Portfolio analysis failed: {e}")
            return {'symbols': [], 'lessons': []}
    
    def build_symbol_universe(self) -> List[str]:
        """
        Build comprehensive symbol universe from multiple sources.
        """
        print("\n🌎 Building symbol universe...")
        
        all_symbols = set()
        
        # 1. Base symbols (guaranteed quality)
        base = [s for category in self.base_symbols.values() for s in category]
        all_symbols.update(base)
        print(f"  ✓ Base symbols: {len(base)}")
        
        # 2. Portfolio symbols (learn from experience)
        if self.portfolio_symbols:
            all_symbols.update(self.portfolio_symbols)
            print(f"  ✓ Portfolio symbols: {len(self.portfolio_symbols)}")
        
        # 3. Dynamic discovery (trending/active)
        trending = self.discover_trending_stocks(limit=50)
        all_symbols.update(trending)
        print(f"  ✓ Trending symbols: {len(trending)}")
        
        all_symbols = list(all_symbols)
        print(f"\n  🎯 Total symbol universe: {len(all_symbols)}")
        
        return all_symbols
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate comprehensive technical indicators.
        """
        try:
            close = df['Close']
            high = df['High']
            low = df['Low']
            volume = df['Volume']
            
            indicators = {}
            
            # Moving Averages
            indicators['sma_20'] = close.rolling(window=20).mean().iloc[-1]
            indicators['sma_50'] = close.rolling(window=50).mean().iloc[-1]
            indicators['sma_200'] = close.rolling(window=200).mean().iloc[-1]
            indicators['ema_12'] = close.ewm(span=12).mean().iloc[-1]
            indicators['ema_26'] = close.ewm(span=26).mean().iloc[-1]
            
            # Price position relative to MAs
            current_price = close.iloc[-1]
            indicators['price_vs_sma20'] = (current_price - indicators['sma_20']) / indicators['sma_20']
            indicators['price_vs_sma50'] = (current_price - indicators['sma_50']) / indicators['sma_50']
            indicators['price_vs_sma200'] = (current_price - indicators['sma_200']) / indicators['sma_200']
            
            # RSI (14-day)
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi_14'] = (100 - (100 / (1 + rs))).iloc[-1]
            
            # MACD
            indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
            indicators['macd_signal'] = pd.Series([indicators['macd']]).ewm(span=9).mean().iloc[-1]
            indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
            
            # Bollinger Bands
            bb_middle = close.rolling(window=20).mean()
            bb_std = close.rolling(window=20).std()
            indicators['bb_upper'] = (bb_middle + 2 * bb_std).iloc[-1]
            indicators['bb_lower'] = (bb_middle - 2 * bb_std).iloc[-1]
            indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / bb_middle.iloc[-1]
            indicators['bb_position'] = (current_price - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
            
            # ATR (Average True Range) - Volatility
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            indicators['atr_14'] = true_range.rolling(14).mean().iloc[-1]
            indicators['atr_percent'] = indicators['atr_14'] / current_price
            
            # Volume indicators
            indicators['volume_sma_20'] = volume.rolling(20).mean().iloc[-1]
            indicators['volume_ratio'] = volume.iloc[-1] / indicators['volume_sma_20']
            
            # On-Balance Volume (OBV)
            obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
            indicators['obv'] = obv.iloc[-1]
            indicators['obv_sma_20'] = obv.rolling(20).mean().iloc[-1]
            
            # Rate of Change
            indicators['roc_10'] = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]
            indicators['roc_20'] = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]
            
            # Stochastic Oscillator
            low_14 = low.rolling(14).min()
            high_14 = high.rolling(14).max()
            indicators['stoch_k'] = 100 * (close - low_14) / (high_14 - low_14)
            indicators['stoch_k'] = indicators['stoch_k'].iloc[-1]
            
            # Money Flow Index (MFI)
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
            mfi_ratio = positive_flow / negative_flow
            indicators['mfi'] = (100 - 100 / (1 + mfi_ratio)).iloc[-1]
            
            # Trend strength
            indicators['trend_strength'] = abs(indicators['price_vs_sma20'])
            
            return indicators
            
        except Exception as e:
            print(f"  ⚠️  Technical indicator calculation failed: {e}")
            return {}
    
    def get_fundamental_data(self, ticker: yf.Ticker, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive fundamental data.
        """
        try:
            info = ticker.info
            
            fundamentals = {
                # Valuation
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'price_to_book': info.get('priceToBook'),
                'price_to_sales': info.get('priceToSalesTrailing12Months'),
                'ev_to_revenue': info.get('enterpriseToRevenue'),
                'ev_to_ebitda': info.get('enterpriseToEbitda'),
                
                # Profitability
                'profit_margin': info.get('profitMargins'),
                'operating_margin': info.get('operatingMargins'),
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                
                # Growth
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth'),
                
                # Financial Health
                'current_ratio': info.get('currentRatio'),
                'debt_to_equity': info.get('debtToEquity'),
                'free_cash_flow': info.get('freeCashflow'),
                'operating_cash_flow': info.get('operatingCashflow'),
                
                # Market Data
                'market_cap': info.get('marketCap'),
                'enterprise_value': info.get('enterpriseValue'),
                'shares_outstanding': info.get('sharesOutstanding'),
                'float_shares': info.get('floatShares'),
                'shares_short': info.get('sharesShort'),
                'short_ratio': info.get('shortRatio'),
                'short_percent_float': info.get('shortPercentOfFloat'),
                
                # Analyst Data
                'target_price': info.get('targetMeanPrice'),
                'target_high': info.get('targetHighPrice'),
                'target_low': info.get('targetLowPrice'),
                'num_analysts': info.get('numberOfAnalystOpinions'),
                'recommendation': info.get('recommendationKey'),
                
                # Other
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'beta': info.get('beta'),
                '52w_high': info.get('fiftyTwoWeekHigh'),
                '52w_low': info.get('fiftyTwoWeekLow'),
            }
            
            # Calculate derived metrics
            current_price = info.get('currentPrice', info.get('regularMarketPrice'))
            if current_price:
                if fundamentals['52w_high']:
                    fundamentals['distance_from_52w_high'] = (current_price - fundamentals['52w_high']) / fundamentals['52w_high']
                if fundamentals['52w_low']:
                    fundamentals['distance_from_52w_low'] = (current_price - fundamentals['52w_low']) / fundamentals['52w_low']
                if fundamentals['target_price']:
                    fundamentals['upside_to_target'] = (fundamentals['target_price'] - current_price) / current_price
            
            return fundamentals
            
        except Exception as e:
            return {}
    
    def get_macro_context(self) -> Dict[str, Any]:
        """
        Get macroeconomic context and market regime.
        """
        try:
            macro = {}
            
            # VIX - Market fear gauge
            vix = yf.Ticker('^VIX')
            vix_hist = vix.history(period='1mo')
            if not vix_hist.empty:
                macro['vix'] = vix_hist['Close'].iloc[-1]
                macro['vix_20d_avg'] = vix_hist['Close'].mean()
                
                # Market regime based on VIX
                if macro['vix'] < 15:
                    macro['market_regime'] = 'low_volatility'
                elif macro['vix'] < 25:
                    macro['market_regime'] = 'normal_volatility'
                else:
                    macro['market_regime'] = 'high_volatility'
            
            # Treasury yields (10-year)
            tnx = yf.Ticker('^TNX')
            tnx_hist = tnx.history(period='1mo')
            if not tnx_hist.empty:
                macro['treasury_10y'] = tnx_hist['Close'].iloc[-1]
            
            # SPY as market benchmark
            spy = yf.Ticker('SPY')
            spy_hist = spy.history(period='6mo')
            if not spy_hist.empty:
                macro['spy_return_1m'] = (spy_hist['Close'].iloc[-1] - spy_hist['Close'].iloc[-20]) / spy_hist['Close'].iloc[-20]
                macro['spy_return_3m'] = (spy_hist['Close'].iloc[-1] - spy_hist['Close'].iloc[-60]) / spy_hist['Close'].iloc[-60]
                
                # Market trend
                sma_50 = spy_hist['Close'].rolling(50).mean().iloc[-1]
                sma_200 = spy_hist['Close'].rolling(200).mean().iloc[-1] if len(spy_hist) >= 200 else None
                
                if sma_200:
                    if spy_hist['Close'].iloc[-1] > sma_50 > sma_200:
                        macro['market_trend'] = 'strong_uptrend'
                    elif spy_hist['Close'].iloc[-1] > sma_50:
                        macro['market_trend'] = 'uptrend'
                    elif spy_hist['Close'].iloc[-1] < sma_50 < sma_200:
                        macro['market_trend'] = 'strong_downtrend'
                    else:
                        macro['market_trend'] = 'downtrend'
            
            return macro
            
        except Exception as e:
            return {}
    
    def collect_stock_data_comprehensive(self, symbol: str, period='1y') -> Optional[Dict[str, Any]]:
        """
        Collect comprehensive data for a single symbol.
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Historical data
            hist = ticker.history(period=period)
            
            if hist.empty or len(hist) < 50:
                return None
            
            # Calculate all indicators
            technicals = self.calculate_technical_indicators(hist)
            fundamentals = self.get_fundamental_data(ticker, symbol)
            
            # Recent performance
            current_price = hist['Close'].iloc[-1]
            performance = {
                'return_5d': (hist['Close'].iloc[-1] - hist['Close'].iloc[-5]) / hist['Close'].iloc[-5],
                'return_10d': (hist['Close'].iloc[-1] - hist['Close'].iloc[-10]) / hist['Close'].iloc[-10],
                'return_20d': (hist['Close'].iloc[-1] - hist['Close'].iloc[-20]) / hist['Close'].iloc[-20],
                'return_60d': (hist['Close'].iloc[-1] - hist['Close'].iloc[-60]) / hist['Close'].iloc[-60],
                'volatility_20d': hist['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252),
                'max_drawdown_20d': (hist['Close'].iloc[-20:].min() - hist['Close'].iloc[-20:].max()) / hist['Close'].iloc[-20:].max(),
            }
            
            # Volume analysis
            volume_analysis = {
                'avg_volume_20d': hist['Volume'].iloc[-20:].mean(),
                'volume_trend': 'increasing' if hist['Volume'].iloc[-5:].mean() > hist['Volume'].iloc[-20:-5].mean() else 'decreasing',
                'unusual_volume': hist['Volume'].iloc[-1] > hist['Volume'].rolling(20).mean().iloc[-1] * 2,
            }
            
            # Check if this was in our portfolio
            was_traded = symbol in self.portfolio_symbols
            
            # Combine all data
            stock_data = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'current_price': float(current_price),
                'technicals': technicals,
                'fundamentals': fundamentals,
                'performance': performance,
                'volume_analysis': volume_analysis,
                'was_traded': was_traded  # NEW: Track if we have experience with this stock
            }
            
            return stock_data
            
        except Exception as e:
            return None
    
    def generate_training_example(self, stock_data: Dict[str, Any], macro_context: Dict[str, Any], portfolio_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate training example with portfolio learning context.
        """
        try:
            symbol = stock_data['symbol']
            tech = stock_data['technicals']
            fund = stock_data['fundamentals']
            perf = stock_data['performance']
            vol = stock_data['volume_analysis']
            was_traded = stock_data.get('was_traded', False)
            
            # Determine ground truth action based on multiple factors
            score = 0
            reasons = []
            
            # Technical signals
            if tech.get('rsi_14', 50) < 30:
                score += 2
                reasons.append("RSI oversold (<30)")
            elif tech.get('rsi_14', 50) > 70:
                score -= 2
                reasons.append("RSI overbought (>70)")
            
            if tech.get('macd_histogram', 0) > 0:
                score += 1
                reasons.append("MACD positive")
            else:
                score -= 1
                reasons.append("MACD negative")
            
            if tech.get('price_vs_sma20', 0) > 0.02:
                score += 1
                reasons.append("Price >2% above SMA20")
            elif tech.get('price_vs_sma20', 0) < -0.02:
                score -= 1
                reasons.append("Price >2% below SMA20")
            
            if tech.get('bb_position', 0.5) < 0.2:
                score += 1
                reasons.append("Near lower Bollinger Band")
            elif tech.get('bb_position', 0.5) > 0.8:
                score -= 1
                reasons.append("Near upper Bollinger Band")
            
            # Fundamental signals
            pe = fund.get('pe_ratio')
            if pe and pe < 15:
                score += 1
                reasons.append("Low P/E ratio (<15)")
            elif pe and pe > 30:
                score -= 1
                reasons.append("High P/E ratio (>30)")
            
            if fund.get('revenue_growth', 0) > 0.15:
                score += 1
                reasons.append("Strong revenue growth (>15%)")
            
            if fund.get('debt_to_equity', 100) < 0.5:
                score += 1
                reasons.append("Low debt-to-equity (<0.5)")
            
            # Performance signals
            if perf.get('return_20d', 0) > 0.10:
                score += 1
                reasons.append("Strong 20-day momentum (+10%)")
            elif perf.get('return_20d', 0) < -0.10:
                score -= 1
                reasons.append("Weak 20-day momentum (-10%)")
            
            # Volume confirmation
            if vol.get('unusual_volume', False) and score > 0:
                score += 1
                reasons.append("Unusual volume confirms trend")
            
            # Portfolio learning (NEW)
            if was_traded and portfolio_analysis.get('lessons'):
                sector = fund.get('sector')
                # Check if we had success in this sector
                for lesson in portfolio_analysis['lessons']:
                    if sector and f"Strength in {sector}" in lesson:
                        score += 1
                        reasons.append(f"Historical success in {sector}")
                    elif sector and f"Avoid {sector}" in lesson:
                        score -= 1
                        reasons.append(f"Historical weakness in {sector}")
            
            # Market regime adjustment
            if macro_context.get('market_regime') == 'high_volatility' and abs(score) > 2:
                score = score * 0.5
                reasons.append("Market volatility reduces conviction")
            
            # Determine action and confidence
            if score >= 3:
                action = "BUY"
                sentiment = "bullish"
                confidence = min(0.85, 0.60 + (score - 3) * 0.05)
            elif score <= -3:
                action = "SELL"
                sentiment = "bearish"
                confidence = min(0.85, 0.60 + (abs(score) - 3) * 0.05)
            else:
                action = "HOLD"
                sentiment = "neutral"
                confidence = 0.50
            
            # Add portfolio context to input if available
            portfolio_context = ""
            if was_traded:
                portfolio_context = f"\n\nPortfolio History: Previously traded {symbol}"
                if portfolio_analysis.get('lessons'):
                    portfolio_context += f"\nLessons: {'; '.join(portfolio_analysis['lessons'][:2])}"
            
            # Create rich input prompt
            input_text = f"""Analyze {symbol} for trading decision.

Technical Analysis:
- Current Price: ${stock_data['current_price']:.2f}
- RSI(14): {tech.get('rsi_14', 0):.1f}
- MACD Histogram: {tech.get('macd_histogram', 0):.3f}
- Price vs SMA(20): {tech.get('price_vs_sma20', 0):.2%}
- Bollinger Band Position: {tech.get('bb_position', 0):.2f}
- ATR (Volatility): {tech.get('atr_percent', 0):.2%}

Fundamental Analysis:
- P/E Ratio: {fund.get('pe_ratio', 'N/A')}
- Revenue Growth: {fund.get('revenue_growth', 0):.1%}
- Debt/Equity: {fund.get('debt_to_equity', 'N/A')}
- Sector: {fund.get('sector', 'Unknown')}

Performance:
- 20-day Return: {perf.get('return_20d', 0):.2%}
- Volatility: {perf.get('volatility_20d', 0):.2%}
- Volume Trend: {vol.get('volume_trend', 'unknown')}

Market Context:
- VIX: {macro_context.get('vix', 'N/A')}
- Market Regime: {macro_context.get('market_regime', 'unknown')}
- Market Trend: {macro_context.get('market_trend', 'unknown')}{portfolio_context}

Provide trading recommendation with reasoning."""

            # Create rich output response
            output_text = f"""Decision: {action}
Confidence: {confidence:.0%}
Sentiment: {sentiment}

Analysis:
{' | '.join(reasons[:5])}

Risk Assessment: {"Low" if perf.get('volatility_20d', 0) < 0.3 else "Moderate" if perf.get('volatility_20d', 0) < 0.5 else "High"}
Market Context: {macro_context.get('market_regime', 'unknown').replace('_', ' ').title()}

Strategy: {"Accumulate on dips" if score > 2 else "Reduce position on rallies" if score < -2 else "Wait for clearer signals"}"""

            training_example = {
                'input': input_text,
                'output': output_text,
                'metadata': {
                    'symbol': symbol,
                    'action': action,
                    'confidence': confidence,
                    'score': score,
                    'sector': fund.get('sector'),
                    'was_traded': was_traded,
                    'timestamp': stock_data['timestamp']
                }
            }
            
            return training_example
            
        except Exception as e:
            return None
    
    def validate_dataset(self, examples: List[Dict]) -> Dict[str, Any]:
        """
        Validate dataset quality and diversity.
        """
        print("\n🔍 Validating dataset quality...")
        
        validation = {
            'total_examples': len(examples),
            'action_distribution': defaultdict(int),
            'sector_distribution': defaultdict(int),
            'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'avg_confidence': 0,
            'portfolio_examples': 0
        }
        
        confidences = []
        
        for ex in examples:
            action = ex['metadata']['action']
            validation['action_distribution'][action] += 1
            
            sector = ex['metadata'].get('sector', 'Unknown')
            validation['sector_distribution'][sector] += 1
            
            conf = ex['metadata']['confidence']
            confidences.append(conf)
            
            if conf > 0.75:
                validation['confidence_distribution']['high'] += 1
            elif conf > 0.60:
                validation['confidence_distribution']['medium'] += 1
            else:
                validation['confidence_distribution']['low'] += 1
            
            if ex['metadata'].get('was_traded'):
                validation['portfolio_examples'] += 1
        
        validation['avg_confidence'] = np.mean(confidences)
        
        # Quality checks
        print(f"  ✓ Total examples: {validation['total_examples']}")
        print(f"  ✓ Action distribution: {dict(validation['action_distribution'])}")
        print(f"  ✓ Avg confidence: {validation['avg_confidence']:.2%}")
        print(f"  ✓ Sector diversity: {len(validation['sector_distribution'])} sectors")
        print(f"  ✓ Portfolio-based examples: {validation['portfolio_examples']}")
        
        # Warnings
        if validation['action_distribution']['BUY'] / validation['total_examples'] > 0.7:
            print("  ⚠️  WARNING: Dataset heavily biased toward BUY actions")
        
        if validation['action_distribution']['SELL'] / validation['total_examples'] < 0.2:
            print("  ⚠️  WARNING: Insufficient SELL examples for training")
        
        if len(validation['sector_distribution']) < 5:
            print("  ⚠️  WARNING: Low sector diversity")
        
        return validation
    
    def collect_all_data(self) -> Path:
        """
        Main data collection workflow with portfolio learning.
        """
        print("=" * 70)
        print("🚀 ENTERPRISE FINANCIAL DATA COLLECTION")
        print("=" * 70)
        
        # 1. Analyze portfolio performance (learn from experience)
        portfolio_analysis = self.analyze_portfolio_performance()
        
        # 2. Build comprehensive symbol universe
        all_symbols = self.build_symbol_universe()
        
        # 3. Get macro context
        print("\n📊 Collecting macroeconomic context...")
        macro_context = self.get_macro_context()
        print(f"  ✓ Market regime: {macro_context.get('market_regime', 'unknown')}")
        print(f"  ✓ VIX: {macro_context.get('vix', 'N/A')}")
        
        # 4. Collect data for all symbols
        print(f"\n📈 Collecting data for {len(all_symbols)} symbols...")
        stock_data_list = []
        
        for i, symbol in enumerate(all_symbols, 1):
            print(f"  [{i}/{len(all_symbols)}] {symbol}...", end='', flush=True)
            data = self.collect_stock_data_comprehensive(symbol)
            if data:
                stock_data_list.append(data)
                print(" ✓")
            else:
                print(" ✗")
            
            # Rate limiting
            if i % 10 == 0:
                time.sleep(1)
        
        print(f"\n✅ Collected data for {len(stock_data_list)} symbols")
        
        # 5. Generate training examples
        print("\n🎓 Generating training examples...")
        training_examples = []
        
        for stock_data in stock_data_list:
            example = self.generate_training_example(stock_data, macro_context, portfolio_analysis)
            if example:
                training_examples.append(example)
        
        print(f"✅ Generated {len(training_examples)} training examples")
        
        # 6. Validate dataset
        validation = self.validate_dataset(training_examples)
        
        # 7. Save everything
        output_path = self.output_dir / 'training_data.json'
        with open(output_path, 'w') as f:
            json.dump(training_examples, f, indent=2)
        
        # Save validation report
        validation_path = self.output_dir / 'validation_report.json'
        with open(validation_path, 'w') as f:
            json.dump(validation, f, indent=2, default=str)
        
        # Save portfolio analysis
        if portfolio_analysis.get('lessons'):
            portfolio_path = self.output_dir / 'portfolio_analysis.json'
            with open(portfolio_path, 'w') as f:
                json.dump(portfolio_analysis, f, indent=2, default=str)
        
        print("\n" + "=" * 70)
        print("✅ DATA COLLECTION COMPLETE")
        print("=" * 70)
        print(f"📁 Training data: {output_path}")
        print(f"📁 Validation report: {validation_path}")
        print(f"📊 Total examples: {len(training_examples)}")
        print(f"🎯 Average confidence: {validation['avg_confidence']:.1%}")
        if portfolio_analysis.get('lessons'):
            print(f"🧠 Portfolio lessons learned: {len(portfolio_analysis['lessons'])}")
        print("=" * 70)
        
        return output_path

if __name__ == "__main__":
    collector = EnterpriseFinancialDataCollector()
    output_path = collector.collect_all_data()
    
    print(f"\n💡 Next steps:")
    print(f"   1. Review the data: cat {output_path} | less")
    print(f"   2. Fine-tune model: python3 finetune/fine_tune_llm.py --data {output_path}")
    print(f"   3. Test inference: python3 scripts/model_inference_lora.py")
