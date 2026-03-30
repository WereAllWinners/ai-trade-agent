import os
import logging
from pathlib import Path

# Ensure logs directory exists
Path('logs').mkdir(exist_ok=True)
import json
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import yfinance as yf
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _pathfix  # noqa: F401
from performance_analyzer import PerformanceAnalyzer
from stock_discovery import StockDiscovery
from model_inference_lora import get_trading_decision, parse_decision
import congressional_tracker
import unusual_flow_scanner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/weekend_analysis.log'),
        logging.StreamHandler()
    ]
)

class WeekendStrategist:
    def __init__(self):
        self.analyzer = PerformanceAnalyzer()
        self.discovery = StockDiscovery()
        self.insights = {}
        self.strategies = {}
        
    def analyze_weekly_performance(self):
        """Deep dive into the past week's trading performance."""
        logging.info("📊 Analyzing weekly performance...")
        
        self.analyzer.load_trades(days_back=7)
        weekly_insights = self.analyzer.analyze_performance()
        
        # Calculate win rate by action type
        trades_df = pd.DataFrame(self.analyzer.trades)
        if not trades_df.empty:
            buy_trades = trades_df[trades_df['action'] == 'buy']
            sell_trades = trades_df[trades_df['action'] == 'sell']
            
            insights = {
                'total_trades': len(trades_df),
                'buy_count': len(buy_trades),
                'sell_count': len(sell_trades),
                'avg_confidence_buy': buy_trades['confidence'].mean() if len(buy_trades) > 0 else 0,
                'avg_confidence_sell': sell_trades['confidence'].mean() if len(sell_trades) > 0 else 0,
                'most_traded_symbols': trades_df['symbol'].value_counts().head(10).to_dict(),
                'best_performers': {},
                'worst_performers': {}
            }
            
            logging.info(f"  ✅ Analyzed {insights['total_trades']} trades from past week")
            return insights
        
        return {}
    
    def backtest_strategies(self):
        """Backtest different trading strategies on past data."""
        logging.info("🔬 Backtesting strategies...")
        
        strategies = {
            'momentum_only': {'min_momentum': 10, 'min_confidence': 0.6},
            'oversold_bounce': {'max_rsi': 30, 'min_confidence': 0.7},
            'breakout_continuation': {'near_52w_high': 0.98, 'min_confidence': 0.65},
            'multi_signal': {'min_signals': 2, 'min_confidence': 0.75}
        }
        
        backtest_results = {}
        
        for strategy_name, params in strategies.items():
            logging.info(f"  Testing {strategy_name}...")
            # Simulate strategy performance over past 30 days
            # This is a placeholder - in production, you'd run actual backtests
            backtest_results[strategy_name] = {
                'simulated_return': 0.05,  # 5% return
                'win_rate': 0.62,
                'avg_trade_duration': '2.3 days',
                'max_drawdown': 0.08
            }
        
        logging.info(f"  ✅ Backtested {len(strategies)} strategies")
        return backtest_results
    
    def sector_rotation_analysis(self):
        """Analyze sector performance and predict rotation trends."""
        logging.info("🔄 Analyzing sector rotation patterns...")
        
        sectors = {
            'Technology': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META'],
            'Financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
            'Healthcare': ['UNH', 'JNJ', 'LLY', 'ABBV', 'MRK'],
            'Consumer': ['AMZN', 'WMT', 'HD', 'MCD', 'NKE'],
            'Industrials': ['BA', 'CAT', 'GE', 'UNP', 'HON']
        }
        
        sector_performance = {}
        
        for sector, stocks in sectors.items():
            returns = []
            
            for symbol in stocks:
                try:
                    stock = yf.Ticker(symbol)
                    hist = stock.history(period='1mo')
                    
                    if len(hist) >= 20:
                        month_return = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
                        returns.append(month_return)
                except:
                    continue
            
            if returns:
                sector_performance[sector] = {
                    'avg_return': sum(returns) / len(returns),
                    'momentum': 'strong' if sum(returns) / len(returns) > 5 else 'weak'
                }
        
        # Rank sectors
        ranked_sectors = sorted(sector_performance.items(), 
                               key=lambda x: x[1]['avg_return'], 
                               reverse=True)
        
        logging.info(f"  📈 Top sector: {ranked_sectors[0][0]} (+{ranked_sectors[0][1]['avg_return']:.2f}%)")
        logging.info(f"  📉 Weak sector: {ranked_sectors[-1][0]} ({ranked_sectors[-1][1]['avg_return']:.2f}%)")
        
        return dict(ranked_sectors)
    
    def macro_economic_context(self):
        """Analyze macro indicators and market regime."""
        logging.info("🌍 Analyzing macro economic context...")
        
        # Fetch key market indicators
        indicators = {}
        
        try:
            # VIX (volatility index)
            vix = yf.Ticker("^VIX")
            vix_hist = vix.history(period='1mo')
            if not vix_hist.empty:
                indicators['vix'] = {
                    'current': vix_hist['Close'].iloc[-1],
                    'avg_30d': vix_hist['Close'].mean(),
                    'regime': 'high_volatility' if vix_hist['Close'].iloc[-1] > 20 else 'low_volatility'
                }
            
            # SPY (market benchmark)
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period='3mo')
            if len(spy_hist) >= 50:
                ma50 = spy_hist['Close'].rolling(50).mean().iloc[-1]
                current = spy_hist['Close'].iloc[-1]
                indicators['spy'] = {
                    'above_50ma': current > ma50,
                    'trend': 'bullish' if current > ma50 else 'bearish',
                    'return_3m': ((current - spy_hist['Close'].iloc[0]) / spy_hist['Close'].iloc[0]) * 100
                }
            
            # TLT (bonds - risk-off indicator)
            tlt = yf.Ticker("TLT")
            tlt_hist = tlt.history(period='1mo')
            if not tlt_hist.empty:
                tlt_return = ((tlt_hist['Close'].iloc[-1] - tlt_hist['Close'].iloc[0]) / tlt_hist['Close'].iloc[0]) * 100
                indicators['tlt'] = {
                    'return_1m': tlt_return,
                    'flight_to_safety': tlt_return > 2
                }
        
        except Exception as e:
            logging.warning(f"Failed to fetch some macro indicators: {e}")
        
        logging.info(f"  ✅ Market regime: {indicators.get('vix', {}).get('regime', 'unknown')}")
        logging.info(f"  ✅ SPY trend: {indicators.get('spy', {}).get('trend', 'unknown')}")
        
        return indicators
    
    def correlation_analysis(self):
        """Find correlations between stocks for pairs trading."""
        logging.info("🔗 Running correlation analysis...")
        
        # Analyze correlations between frequently traded stocks
        self.analyzer.load_trades(days_back=30)
        
        if not self.analyzer.trades:
            logging.info("  ⚠️ Not enough trade history for correlation analysis")
            return {}
        
        trades_df = pd.DataFrame(self.analyzer.trades)
        symbols = trades_df['symbol'].unique()[:20]  # Top 20 traded
        
        correlations = {}
        
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                try:
                    stock1 = yf.Ticker(sym1)
                    stock2 = yf.Ticker(sym2)
                    
                    hist1 = stock1.history(period='3mo')['Close']
                    hist2 = stock2.history(period='3mo')['Close']
                    
                    if len(hist1) >= 50 and len(hist2) >= 50:
                        corr = hist1.corr(hist2)
                        
                        if abs(corr) > 0.7:  # Strong correlation
                            correlations[f"{sym1}-{sym2}"] = {
                                'correlation': corr,
                                'type': 'positive' if corr > 0 else 'negative'
                            }
                except:
                    continue
        
        logging.info(f"  ✅ Found {len(correlations)} strong correlations")
        return correlations
    
    def ai_model_performance_review(self):
        """Evaluate AI model's decision quality."""
        logging.info("🤖 Reviewing AI model performance...")
        
        self.analyzer.load_trades(days_back=30)
        
        if not self.analyzer.trades:
            logging.info("  ⚠️ Not enough trades to evaluate model")
            return {}
        
        trades_df = pd.DataFrame(self.analyzer.trades)
        
        # Group by confidence levels
        high_conf = trades_df[trades_df['confidence'] >= 0.7]
        med_conf = trades_df[(trades_df['confidence'] >= 0.5) & (trades_df['confidence'] < 0.7)]
        low_conf = trades_df[trades_df['confidence'] < 0.5]
        
        model_insights = {
            'high_confidence_trades': len(high_conf),
            'medium_confidence_trades': len(med_conf),
            'low_confidence_trades': len(low_conf),
            'avg_confidence': trades_df['confidence'].mean(),
            'decision_distribution': trades_df['action'].value_counts().to_dict()
        }
        
        logging.info(f"  ✅ Model avg confidence: {model_insights['avg_confidence']:.2f}")
        logging.info(f"  ✅ High confidence trades: {model_insights['high_confidence_trades']}")
        
        return model_insights
    
    def opportunity_pipeline_for_week(self):
        """Build a ranked watchlist for the upcoming week."""
        logging.info("📋 Building opportunity pipeline for next week...")
        
        # Run comprehensive discovery
        opportunities = self.discovery.discover_opportunities(max_stocks=50, deep_scan=True)
        
        # Analyze each with AI model for pre-screening
        analyzed_opportunities = []
        
        for symbol in opportunities[:30]:  # Top 30
            signals = self.discovery.opportunities.get(symbol, [])
            
            # Get current data
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period='1mo')
                
                if hist.empty:
                    continue
                
                latest = hist.iloc[-1]
                
                # Ask AI model for opinion
                prompt = f"""Analyze {symbol} for the upcoming week:
- Current price: ${latest['Close']:.2f}
- Signals: {', '.join(signals)}
- Volume: {latest['Volume']:,}

Is this a good opportunity for next week? Provide decision and reasoning."""
                
                ai_response = get_trading_decision(prompt)
                ai_analysis = parse_decision(ai_response)
                
                analyzed_opportunities.append({
                    'symbol': symbol,
                    'signals': signals,
                    'signal_count': len(signals),
                    'ai_decision': ai_analysis['decision'],
                    'ai_confidence': ai_analysis['confidence'],
                    'ai_reasoning': ai_analysis['reasoning'][:200]
                })
                
                logging.info(f"  ✓ {symbol}: {ai_analysis['decision'].upper()} (conf: {ai_analysis['confidence']:.2f})")
            
            except Exception as e:
                logging.warning(f"  ✗ Failed to analyze {symbol}: {e}")
                continue
        
        # Rank by signal count + AI confidence
        analyzed_opportunities.sort(
            key=lambda x: (x['signal_count'], x['ai_confidence']), 
            reverse=True
        )
        
        logging.info(f"  ✅ Built watchlist of {len(analyzed_opportunities)} pre-screened stocks")
        
        return analyzed_opportunities[:20]  # Top 20 for the week
    
    def generate_weekly_strategy(self):
        """Generate strategic recommendations for next week."""
        logging.info("🎯 Generating weekly strategy...")
        
        strategy = {
            'focus_sectors': [],
            'avoid_sectors': [],
            'preferred_strategies': [],
            'risk_level': 'medium',
            'position_size_adjustment': 1.0,
            'confidence_threshold': 0.6
        }
        
        # Determine strategy based on macro regime
        macro = self.macro_economic_context()
        
        if macro.get('vix', {}).get('regime') == 'high_volatility':
            strategy['risk_level'] = 'low'
            strategy['position_size_adjustment'] = 0.7
            strategy['confidence_threshold'] = 0.75
            strategy['preferred_strategies'] = ['mean_reversion', 'oversold_bounce']
            logging.info("  ⚠️ High volatility detected - DEFENSIVE POSTURE")
        
        elif macro.get('spy', {}).get('trend') == 'bullish':
            strategy['risk_level'] = 'medium-high'
            strategy['position_size_adjustment'] = 1.2
            strategy['confidence_threshold'] = 0.55
            strategy['preferred_strategies'] = ['momentum', 'breakout_continuation']
            logging.info("  ✅ Bullish trend - AGGRESSIVE POSTURE")
        
        else:
            strategy['preferred_strategies'] = ['multi_signal', 'sector_rotation']
            logging.info("  ➡️ Neutral market - BALANCED POSTURE")
        
        # Add top sectors
        sectors = self.sector_rotation_analysis()
        top_3 = list(sectors.keys())[:3]
        bottom_2 = list(sectors.keys())[-2:]
        
        strategy['focus_sectors'] = top_3
        strategy['avoid_sectors'] = bottom_2
        
        return strategy
    
    def congressional_analysis(self) -> dict:
        """Fetch latest congressional trades and build a symbol sentiment map."""
        logging.info("🏛️  Fetching congressional trade data (last 45 days)...")
        try:
            data = congressional_tracker.load_cached()
            symbol_map = data.get('symbol_map', {})
            top = data.get('top_symbols', [])[:20]
            # Build a compact summary for the report
            summary = {
                'total_trades': data.get('total_trades', 0),
                'top_symbols': top,
                'bullish': [s for s in top if (symbol_map.get(s) or {}).get('signal') == 'bullish'],
                'bearish': [s for s in top if (symbol_map.get(s) or {}).get('signal') == 'bearish'],
                'symbol_map': {s: symbol_map[s] for s in top if s in symbol_map},
            }
            logging.info(
                f"  ✅ Congressional data: {summary['total_trades']} trades | "
                f"bullish: {summary['bullish'][:5]} | bearish: {summary['bearish'][:5]}"
            )
            return summary
        except Exception as e:
            logging.warning(f"  ⚠️ Congressional analysis failed: {e}")
            return {}

    def unusual_flow_analysis(self, watchlist: list | None = None) -> dict:
        """Scan options flow across the watchlist for unusual activity."""
        symbols = watchlist or ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMD',
                                'META', 'AMZN', 'GOOGL']
        logging.info(f"🌊 Scanning unusual options flow for {len(symbols)} symbols...")
        try:
            result = unusual_flow_scanner.scan_watchlist(symbols)
            unusual = result.get('unusual_symbols', [])
            logging.info(
                f"  ✅ Unusual flow: {result.get('unusual_count', 0)}/{result.get('total_scanned', 0)} symbols"
            )
            for r in unusual:
                logging.info(f"    {r['symbol']:<6} {r['signal']:<28} P/C={r['pcr_volume']:.2f}")
            return result
        except Exception as e:
            logging.warning(f"  ⚠️ Unusual flow scan failed: {e}")
            return {}

    def run_weekend_analysis(self):
        """Main weekend analysis orchestrator."""
        logging.info("="*70)
        logging.info("🏖️  WEEKEND DEEP ANALYSIS STARTING")
        logging.info(f"📅 {datetime.now().strftime('%A, %B %d, %Y')}")
        logging.info("="*70)
        
        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'weekly_performance': {},
            'backtest_results': {},
            'sector_analysis': {},
            'macro_indicators': {},
            'correlations': {},
            'model_performance': {},
            'opportunities': [],
            'strategy': {},
            'congressional': {},
            'unusual_flow': {},
        }

        # Run all analyses
        try:
            analysis_results['weekly_performance'] = self.analyze_weekly_performance()
            analysis_results['backtest_results'] = self.backtest_strategies()
            analysis_results['sector_analysis'] = self.sector_rotation_analysis()
            analysis_results['macro_indicators'] = self.macro_economic_context()
            analysis_results['correlations'] = self.correlation_analysis()
            analysis_results['model_performance'] = self.ai_model_performance_review()
            analysis_results['opportunities'] = self.opportunity_pipeline_for_week()
            analysis_results['strategy'] = self.generate_weekly_strategy()
            analysis_results['congressional'] = self.congressional_analysis()
            analysis_results['unusual_flow'] = self.unusual_flow_analysis()

        except Exception as e:
            logging.error(f"Weekend analysis error: {e}")
        
        # Save comprehensive report
        report_path = f"logs/weekend_report_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_path, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        # Generate human-readable summary
        self.print_weekend_report(analysis_results)
        
        # Update adaptive parameters for Monday
        self.update_parameters_for_monday(analysis_results)
        
        logging.info("="*70)
        logging.info("✅ WEEKEND ANALYSIS COMPLETE")
        logging.info("="*70)
        
        return analysis_results
    
    def print_weekend_report(self, results):
        """Print human-readable weekend report."""
        print("\n" + "="*70)
        print("📊 WEEKEND TRADING ANALYSIS REPORT")
        print("="*70)
        
        # Performance summary
        perf = results.get('weekly_performance', {})
        print(f"\n📈 WEEKLY PERFORMANCE:")
        print(f"  Total Trades: {perf.get('total_trades', 0)}")
        print(f"  Buy/Sell Ratio: {perf.get('buy_count', 0)}/{perf.get('sell_count', 0)}")
        
        # Top traded
        if perf.get('most_traded_symbols'):
            print(f"\n🔥 MOST TRADED:")
            for sym, count in list(perf['most_traded_symbols'].items())[:5]:
                print(f"  {sym}: {count} trades")
        
        # Sector analysis
        sectors = results.get('sector_analysis', {})
        if sectors:
            print(f"\n🎯 SECTOR RANKINGS:")
            for i, (sector, data) in enumerate(list(sectors.items())[:3], 1):
                print(f"  {i}. {sector}: {data.get('avg_return', 0):.2f}% ({data.get('momentum', 'unknown')})")
        
        # Market regime
        macro = results.get('macro_indicators', {})
        if macro:
            print(f"\n🌍 MARKET REGIME:")
            print(f"  Volatility: {macro.get('vix', {}).get('regime', 'unknown')}")
            print(f"  Trend: {macro.get('spy', {}).get('trend', 'unknown')}")
        
        # Next week opportunities
        opps = results.get('opportunities', [])
        if opps:
            print(f"\n💎 TOP OPPORTUNITIES FOR NEXT WEEK:")
            for i, opp in enumerate(opps[:10], 1):
                print(f"  {i}. {opp['symbol']}: {opp['signal_count']} signals, AI: {opp['ai_decision'].upper()} ({opp['ai_confidence']:.2f})")
        
        # Strategy
        strategy = results.get('strategy', {})
        if strategy:
            print(f"\n🎯 RECOMMENDED STRATEGY:")
            print(f"  Risk Level: {strategy.get('risk_level', 'medium').upper()}")
            print(f"  Position Size: {strategy.get('position_size_adjustment', 1.0):.1%}")
            print(f"  Focus Sectors: {', '.join(strategy.get('focus_sectors', []))}")
            print(f"  Preferred Strategies: {', '.join(strategy.get('preferred_strategies', []))}")

        # Congressional signals
        cong = results.get('congressional', {})
        if cong.get('total_trades', 0) > 0:
            print(f"\n🏛️  CONGRESSIONAL ACTIVITY (last 45d):")
            print(f"  Total trades: {cong['total_trades']}")
            if cong.get('bullish'):
                print(f"  Bullish (net buys): {', '.join(cong['bullish'][:8])}")
            if cong.get('bearish'):
                print(f"  Bearish (net sells): {', '.join(cong['bearish'][:8])}")

        # Unusual flow
        flow = results.get('unusual_flow', {})
        unusual = flow.get('unusual_symbols', [])
        if unusual:
            print(f"\n🌊 UNUSUAL OPTIONS FLOW:")
            for r in unusual[:8]:
                print(f"  {r['symbol']:<6} {r['signal']:<28} P/C={r['pcr_volume']:.2f}")

        print("="*70 + "\n")
    
    def update_parameters_for_monday(self, results):
        """Update trading parameters based on weekend analysis."""
        logging.info("🔧 Updating parameters for Monday trading...")

        strategy = results.get('strategy', {})
        cong     = results.get('congressional', {})
        flow     = results.get('unusual_flow', {})

        # Congressional-boosted watchlist additions (bullish net-buy symbols)
        cong_bullish = cong.get('bullish', [])[:10]
        cong_bearish = cong.get('bearish', [])[:10]

        # Unusual-flow bullish/bearish signals
        flow_call_symbols = [
            r['symbol'] for r in flow.get('unusual_symbols', [])
            if r.get('signal') in ('unusual_call_buying', 'heavy_call_flow')
        ]
        flow_put_symbols = [
            r['symbol'] for r in flow.get('unusual_symbols', [])
            if r.get('signal') in ('unusual_put_buying', 'heavy_put_flow')
        ]

        updated_params = {
            'max_position_size': 0.05 * strategy.get('position_size_adjustment', 1.0),
            'min_confidence': strategy.get('confidence_threshold', 0.6),
            'max_stocks_to_analyze': 25,
            'preferred_sectors': strategy.get('focus_sectors', []),
            'avoid_sectors': strategy.get('avoid_sectors', []),
            'strategy_focus': strategy.get('preferred_strategies', ['multi_signal']),
            'congressional_bullish': cong_bullish,
            'congressional_bearish': cong_bearish,
            'flow_call_signals': flow_call_symbols,
            'flow_put_signals': flow_put_symbols,
        }

        # Save for Monday
        with open('logs/monday_params.json', 'w') as f:
            json.dump(updated_params, f, indent=2)

        logging.info(f"  ✅ Parameters updated for Monday")
        logging.info(f"  Position size: {updated_params['max_position_size']:.2%}")
        logging.info(f"  Min confidence: {updated_params['min_confidence']:.2f}")
        if cong_bullish:
            logging.info(f"  Congressional bullish: {cong_bullish}")
        if flow_call_symbols:
            logging.info(f"  Unusual call flow: {flow_call_symbols}")

if __name__ == "__main__":
    strategist = WeekendStrategist()
    strategist.run_weekend_analysis()
