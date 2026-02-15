#!/usr/bin/env python3
"""
Options Performance Analyzer - Analyzes options trading performance
"""
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class OptionsPerformanceAnalyzer:
    def __init__(self):
        self.trading_client = TradingClient(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            paper=True
        )
        self.trade_log_path = 'logs/options_trade_log.jsonl'
        self.output_path = 'logs/options_performance_metrics.json'
        self.training_data_path = 'finetune/data/options_training_data.json'
        
        # Create directories
        Path('logs').mkdir(exist_ok=True)
        Path('finetune/data').mkdir(parents=True, exist_ok=True)
    
    def load_trade_history(self):
        """Load trade history from log."""
        trades = []
        
        if not os.path.exists(self.trade_log_path):
            return trades
        
        with open(self.trade_log_path, 'r') as f:
            for line in f:
                try:
                    trades.append(json.loads(line.strip()))
                except:
                    continue
        
        return trades
    
    def get_current_positions(self):
        """Get current options positions."""
        try:
            positions = self.trading_client.get_all_positions()
            options_positions = []
            
            for p in positions:
                # Options have long symbols
                if len(p.symbol) > 10:
                    options_positions.append({
                        'contract': p.symbol,
                        'qty': float(p.qty),
                        'avg_entry': float(p.avg_entry_price),
                        'current_price': float(p.current_price),
                        'unrealized_pl': float(p.unrealized_pl),
                        'unrealized_plpc': float(p.unrealized_plpc)
                    })
            
            return options_positions
            
        except Exception as e:
            logging.error(f"❌ Failed to get positions: {e}")
            return []
    
    def calculate_metrics(self, trades):
        """Calculate performance metrics."""
        if not trades:
            return None
        
        # Filter completed trades (buy + sell pairs)
        buy_trades = [t for t in trades if t.get('action') == 'buy']
        sell_trades = [t for t in trades if t.get('action') == 'sell']
        
        winners = [t for t in sell_trades if t.get('exit_pl_pct', 0) > 0]
        losers = [t for t in sell_trades if t.get('exit_pl_pct', 0) < 0]
        
        metrics = {
            'total_trades': len(buy_trades),
            'closed_trades': len(sell_trades),
            'open_trades': len(buy_trades) - len(sell_trades),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': len(winners) / len(sell_trades) if sell_trades else 0,
            'avg_return': sum(t['exit_pl_pct'] for t in sell_trades) / len(sell_trades) if sell_trades else 0,
            'best_trade': max((t['exit_pl_pct'] for t in sell_trades), default=0),
            'worst_trade': min((t['exit_pl_pct'] for t in sell_trades), default=0),
            'timestamp': datetime.now().isoformat()
        }
        
        return metrics
    
    def generate_training_data(self, trades, positions, metrics):
        """Generate training data from performance."""
        training_examples = []
        
        # Winners (profit > 30%)
        winners = [t for t in trades if t.get('action') == 'sell' and t.get('exit_pl_pct', 0) > 0.3]
        for trade in winners[-5:]:  # Last 5 winners
            training_examples.append({
                'input': f"Analyze options trade: {trade.get('underlying', 'UNKNOWN')}, confidence: {trade.get('confidence', 0):.2f}",
                'output': f"TRADE. Reasoning: {trade.get('reasoning', 'Good setup')}. Result: Profitable exit at {trade['exit_pl_pct']:+.1%}",
                'label': 'winner'
            })
        
        # Losers (loss < -30%)
        losers = [t for t in trades if t.get('action') == 'sell' and t.get('exit_pl_pct', 0) < -0.3]
        for trade in losers[-5:]:  # Last 5 losers
            training_examples.append({
                'input': f"Analyze options trade: {trade.get('underlying', 'UNKNOWN')}, confidence: {trade.get('confidence', 0):.2f}",
                'output': f"AVOID. Reasoning: {trade.get('reasoning', 'Poor setup')}. Result: Loss at {trade['exit_pl_pct']:+.1%}",
                'label': 'loser'
            })
        
        # Strong open positions (unrealized > 40%)
        strong_positions = [p for p in positions if p['unrealized_plpc'] > 0.4]
        for pos in strong_positions[:3]:
            training_examples.append({
                'input': f"Evaluate options position: {pos['contract'][:10]}...",
                'output': f"STRONG. Current unrealized P&L: {pos['unrealized_plpc']:+.1%}. Consider taking profits.",
                'label': 'strong_position'
            })
        
        # Save training data
        if training_examples:
            with open(self.training_data_path, 'w') as f:
                json.dump(training_examples, f, indent=2)
            logging.info(f"✅ Generated {len(training_examples)} training examples")
            return len(training_examples)
        
        return 0
    
    def run_analysis(self):
        """Run full performance analysis."""
        logging.info("📊 Starting Options Performance Analysis")
        
        # Load data
        trades = self.load_trade_history()
        positions = self.get_current_positions()
        
        # Calculate metrics
        metrics = self.calculate_metrics(trades)
        
        # Log summary
        if metrics:
            logging.info(f"📈 Total Options Trades: {metrics['total_trades']}")
            logging.info(f"✅ Closed: {metrics['closed_trades']}, Open: {metrics['open_trades']}")
            logging.info(f"🎯 Win Rate: {metrics['win_rate']:.1%}")
            logging.info(f"💰 Avg Return: {metrics['avg_return']:+.1%}")
            logging.info(f"🏆 Best: {metrics['best_trade']:+.1%}, Worst: {metrics['worst_trade']:+.1%}")
            
            # Save metrics
            with open(self.output_path, 'w') as f:
                json.dump(metrics, f, indent=2)
        
        # Log positions
        if positions:
            logging.info(f"📊 Current Options Positions: {len(positions)}")
            total_pl = sum(p['unrealized_pl'] for p in positions)
            logging.info(f"💰 Total Unrealized P&L: ${total_pl:,.2f}")
        
        # Generate training data
        num_examples = self.generate_training_data(trades, positions, metrics)
        
        if num_examples >= 3:
            logging.info(f"✅ Ready for fine-tuning with {num_examples} examples")
        else:
            logging.info(f"⏳ Need {3 - num_examples} more examples for fine-tuning")
        
        logging.info("✅ Options performance analysis complete")

if __name__ == "__main__":
    analyzer = OptionsPerformanceAnalyzer()
    analyzer.run_analysis()
