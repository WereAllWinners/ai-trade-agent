import json
import pandas as pd
from datetime import datetime, timedelta
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)

class PerformanceAnalyzer:
    def __init__(self, log_file='logs/trade_log.jsonl'):
        self.log_file = log_file
        self.trades = []
        self.insights = {}
    
    def load_trades(self, days_back=7):
        """Load trades from the last N days."""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    trade = json.loads(line)
                    trade_date = datetime.fromisoformat(trade['timestamp'])
                    
                    if trade_date >= cutoff_date:
                        self.trades.append(trade)
            
            logging.info(f"Loaded {len(self.trades)} trades from last {days_back} days")
        except FileNotFoundError:
            logging.warning(f"No trade log found at {self.log_file}")
        except Exception as e:
            logging.error(f"Error loading trades: {e}")
    
    def analyze_performance(self):
        """Analyze trading performance and extract insights."""
        if not self.trades:
            logging.info("No trades to analyze")
            return {}
        
        df = pd.DataFrame(self.trades)
        
        # Calculate metrics
        total_trades = len(df)
        buy_trades = len(df[df['action'] == 'buy'])
        sell_trades = len(df[df['action'] == 'sell'])
        failed_trades = len(df[df['action'].str.contains('FAILED')])
        
        # Average confidence
        avg_confidence = df['confidence'].mean()
        
        # Most traded symbols
        symbol_counts = df['symbol'].value_counts().head(5).to_dict()
        
        # Action distribution
        action_dist = df['action'].value_counts().to_dict()
        
        # Calculate total capital deployed
        total_deployed = (df['qty'] * df['price']).sum()
        
        self.insights = {
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'failed_trades': failed_trades,
            'success_rate': ((total_trades - failed_trades) / total_trades * 100) if total_trades > 0 else 0,
            'avg_confidence': avg_confidence,
            'most_traded': symbol_counts,
            'action_distribution': action_dist,
            'total_capital_deployed': total_deployed,
            'analysis_date': datetime.now().isoformat()
        }
        
        return self.insights
    
    def generate_recommendations(self):
        """Generate trading recommendations based on past performance."""
        recommendations = []
        
        if not self.insights:
            self.analyze_performance()
        
        # Recommendation 1: Adjust confidence threshold
        avg_conf = self.insights.get('avg_confidence', 0.5)
        if avg_conf < 0.6:
            recommendations.append({
                'type': 'confidence_threshold',
                'action': 'increase',
                'value': 0.65,
                'reason': f'Average confidence is low ({avg_conf:.2f}), increase threshold to be more selective'
            })
        
        # Recommendation 2: Diversification
        most_traded = self.insights.get('most_traded', {})
        if most_traded and len(most_traded) < 5:
            recommendations.append({
                'type': 'diversification',
                'action': 'expand',
                'value': 15,
                'reason': 'Trading too few symbols, expand discovery to 15+ stocks'
            })
        
        # Recommendation 3: Position sizing
        total_deployed = self.insights.get('total_capital_deployed', 0)
        if total_deployed > 50000:
            recommendations.append({
                'type': 'position_size',
                'action': 'reduce',
                'value': 0.03,
                'reason': f'High capital deployment (${total_deployed:.0f}), reduce position size to 3%'
            })
        
        # Recommendation 4: Failed trades
        failed_rate = (self.insights.get('failed_trades', 0) / self.insights.get('total_trades', 1)) * 100
        if failed_rate > 10:
            recommendations.append({
                'type': 'execution',
                'action': 'improve',
                'value': 'add_retry_logic',
                'reason': f'High failure rate ({failed_rate:.1f}%), add retry logic for orders'
            })
        
        return recommendations
    
    def save_analysis(self, output_file='logs/daily_analysis.jsonl'):
        """Save analysis to file."""
        analysis_record = {
            'timestamp': datetime.now().isoformat(),
            'insights': self.insights,
            'recommendations': self.generate_recommendations()
        }
        
        with open(output_file, 'a') as f:
            f.write(json.dumps(analysis_record) + '\n')
        
        logging.info(f"Analysis saved to {output_file}")
    
    def print_report(self):
        """Print a human-readable report."""
        if not self.insights:
            self.analyze_performance()
        
        print("\n" + "="*60)
        print("📊 TRADING PERFORMANCE REPORT")
        print("="*60)
        print(f"Total Trades: {self.insights.get('total_trades', 0)}")
        print(f"  - Buys: {self.insights.get('buy_trades', 0)}")
        print(f"  - Sells: {self.insights.get('sell_trades', 0)}")
        print(f"  - Failed: {self.insights.get('failed_trades', 0)}")
        print(f"Success Rate: {self.insights.get('success_rate', 0):.1f}%")
        print(f"Avg Confidence: {self.insights.get('avg_confidence', 0):.2f}")
        print(f"Capital Deployed: ${self.insights.get('total_capital_deployed', 0):,.2f}")
        
        print("\nMost Traded Symbols:")
        for symbol, count in self.insights.get('most_traded', {}).items():
            print(f"  {symbol}: {count} trades")
        
        recommendations = self.generate_recommendations()
        if recommendations:
            print("\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec['reason']}")
        
        print("="*60 + "\n")

if __name__ == "__main__":
    analyzer = PerformanceAnalyzer()
    analyzer.load_trades(days_back=7)
    analyzer.print_report()
    analyzer.save_analysis()
