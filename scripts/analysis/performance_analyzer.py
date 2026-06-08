import json
import math
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from collections import defaultdict
from dotenv import load_dotenv
from alpaca.trading.requests import GetPortfolioHistoryRequest

load_dotenv()

logging.basicConfig(level=logging.INFO)

_RISK_FREE_ANNUAL = 0.05  # used for Sharpe calculations throughout

class PerformanceAnalyzer:
    def __init__(self, log_file='logs/trade_log.jsonl', paper=None):
        self.log_file = log_file
        self.outcomes_file = str(Path(log_file).parent / 'trade_outcomes.jsonl')
        self.trades = []
        self.insights = {}
        self.pnl_metrics = {}
        if paper is None:
            paper = os.getenv('PAPER_TRADING', 'true').lower() != 'false'
        self._paper = paper
        self._trading_client = None  # lazy-initialised on first equity-curve fetch
    
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
    
    def load_outcomes(self, days_back=None):
        """Load realized P&L records from trade_outcomes.jsonl."""
        outcomes = []
        try:
            cutoff = datetime.now() - timedelta(days=days_back) if days_back else None
            with open(self.outcomes_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        o = json.loads(line)
                        if cutoff:
                            exit_dt = datetime.fromisoformat(o['exit_timestamp'])
                            if exit_dt < cutoff:
                                continue
                        outcomes.append(o)
                    except Exception:
                        pass
        except FileNotFoundError:
            logging.info("No trade_outcomes.jsonl yet — outcome tracker hasn't run or no closed trades.")
        except Exception as e:
            logging.error(f"Error loading outcomes: {e}")
        return outcomes

    # ------------------------------------------------------------------
    # Equity-curve Sharpe (portfolio-level, accounts for position sizing)
    # ------------------------------------------------------------------

    def _get_trading_client(self):
        if self._trading_client is None:
            from alpaca.trading.client import TradingClient
            self._trading_client = TradingClient(
                os.getenv('ALPACA_API_KEY'),
                os.getenv('ALPACA_SECRET_KEY'),
                paper=self._paper,
            )
        return self._trading_client

    def get_equity_curve(self, days_back=30):
        """Fetch daily equity snapshots from Alpaca portfolio history."""
        try:
            req = GetPortfolioHistoryRequest(
                period=f"{min(days_back, 365)}D",
                timeframe="1D",
            )
            history = self._get_trading_client().get_portfolio_history(req)
            return [(ts, eq) for ts, eq in zip(history.timestamp, history.equity)
                    if eq is not None]
        except Exception as e:
            logging.warning("Could not fetch portfolio history: %s", e)
            return []

    def _equity_curve_sharpe(self, days_back=30):
        """
        Annualised Sharpe from the daily portfolio equity curve.
        Returns None when insufficient data is available.
        This is the portfolio-level Sharpe — it correctly reflects the 1–5%
        position-sizing used by the agent (unlike the per-trade formula).
        """
        curve = self.get_equity_curve(days_back)
        equities = [eq for _, eq in curve if eq and eq > 0]
        if len(equities) < 5:
            return None
        daily_returns = [
            (equities[i] - equities[i - 1]) / equities[i - 1]
            for i in range(1, len(equities))
        ]
        n = len(daily_returns)
        mean_r = sum(daily_returns) / n
        variance = sum((r - mean_r) ** 2 for r in daily_returns) / max(n - 1, 1)
        std_dev = math.sqrt(variance)
        if std_dev == 0:
            return None
        rf_daily = (1 + _RISK_FREE_ANNUAL) ** (1 / 252) - 1
        return round((mean_r - rf_daily) / std_dev * math.sqrt(252), 3)

    def analyze_pnl_metrics(self, days_back=30):
        """
        Compute real trading metrics from closed trade outcomes.

        Uses trade_outcomes.jsonl (written by outcome_tracker.py after each
        nightly run) which contains actual fill prices and realized P&L.

        Returns a dict with Sharpe, profit factor, win rate, and a
        go_live_ready boolean against the target thresholds.
        """
        outcomes = self.load_outcomes(days_back=days_back)
        if not outcomes:
            logging.info("No closed outcomes yet — keep paper trading.")
            return {}

        pnl_pcts    = [o['pnl_pct']     for o in outcomes]
        pnl_dollars = [o['realized_pnl'] for o in outcomes]
        winners     = [o for o in outcomes if o['won']]
        losers      = [o for o in outcomes if not o['won']]

        gross_profit = sum(o['realized_pnl'] for o in winners)
        gross_loss   = abs(sum(o['realized_pnl'] for o in losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        win_rate     = len(winners) / len(outcomes)
        avg_win_pct  = float(np.mean([o['pnl_pct'] for o in winners])) if winners else 0.0
        avg_loss_pct = float(np.mean([o['pnl_pct'] for o in losers]))  if losers  else 0.0
        total_pnl    = sum(pnl_dollars)

        # -----------------------------------------------------------------
        # Sharpe ratio — two methods, portfolio-level preferred
        # -----------------------------------------------------------------
        avg_hold_hours  = float(np.mean([o['hold_hours'] for o in outcomes]))
        # Floor at 0.5 trading days to prevent extreme annualization factors
        # from very short exits (stopped out within minutes).
        avg_hold_days   = max(avg_hold_hours / 6.5, 0.5)
        trades_per_year = 252 / avg_hold_days

        mean_r = float(np.mean(pnl_pcts))
        # Sample std (ddof=1) — unbiased for finite samples
        std_r  = float(np.std(pnl_pcts, ddof=1)) if len(pnl_pcts) > 1 else 0.0
        # Risk-free rate scaled to the average hold period
        rf_per_trade = (1 + _RISK_FREE_ANNUAL) ** (avg_hold_days / 252) - 1
        sharpe_per_trade = (
            (mean_r - rf_per_trade) / std_r * math.sqrt(trades_per_year)
            if std_r > 0 else 0.0
        )

        # Preferred: portfolio-level Sharpe from the daily equity curve.
        # This correctly reflects the 1–5% position sizing; the per-trade
        # formula does not.  Falls back to the corrected per-trade value
        # if Alpaca portfolio history is unavailable (e.g., paper account
        # < 5 days old or API unreachable).
        sharpe_equity = self._equity_curve_sharpe(days_back)
        sharpe = sharpe_equity if sharpe_equity is not None else sharpe_per_trade

        # Max consecutive losses
        max_consec = consec = 0
        for o in outcomes:
            if not o['won']:
                consec += 1
                max_consec = max(max_consec, consec)
            else:
                consec = 0

        # Expectancy per trade
        expectancy = (win_rate * avg_win_pct) + ((1 - win_rate) * avg_loss_pct)

        # Go-live gate (matches roadmap targets)
        go_live_ready = (
            win_rate      >= 0.55 and
            profit_factor >= 1.4  and
            sharpe        >= 0.8
        )

        self.pnl_metrics = {
            'total_closed_trades':    len(outcomes),
            'winners':                len(winners),
            'losers':                 len(losers),
            'win_rate':               round(win_rate, 4),
            'avg_win_pct':            round(avg_win_pct, 4),
            'avg_loss_pct':           round(avg_loss_pct, 4),
            'expectancy_per_trade':   round(expectancy, 4),
            'profit_factor':          round(profit_factor, 3),
            # Primary Sharpe — portfolio-level from equity curve when available,
            # otherwise corrected per-trade estimate.
            'sharpe_ratio':           round(sharpe, 3),
            'sharpe_method':          'equity_curve' if sharpe_equity is not None else 'per_trade_corrected',
            # Diagnostic: corrected per-trade Sharpe (inflated vs portfolio-level;
            # shown so you can see both numbers side by side).
            'sharpe_per_trade':       round(sharpe_per_trade, 3),
            'total_realized_pnl':     round(total_pnl, 2),
            'gross_profit':           round(gross_profit, 2),
            'gross_loss':             round(gross_loss, 2),
            'avg_hold_hours':         round(avg_hold_hours, 1),
            'max_consecutive_losses': max_consec,
            'go_live_ready':          go_live_ready,
            'metrics_period_days':    days_back,
            'targets': {
                'win_rate_target':      0.55,
                'profit_factor_target': 1.4,
                'sharpe_target':        0.8,
            },
            'computed_at': datetime.now().isoformat(),
        }
        return self.pnl_metrics

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
        failed_trades = len(df[df['action'].str.contains('FAILED', na=False)])

        # Average confidence
        avg_confidence = df['confidence'].mean() if 'confidence' in df.columns else 0.0

        # Most traded symbols
        symbol_counts = df['symbol'].value_counts().head(5).to_dict() if 'symbol' in df.columns else {}

        # Action distribution
        action_dist = df['action'].value_counts().to_dict()

        # Total shares traded (price not logged at entry — tracked separately by outcome tracker)
        total_shares = int(df['shares'].sum()) if 'shares' in df.columns else 0

        self.insights = {
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'failed_trades': failed_trades,
            'success_rate': ((total_trades - failed_trades) / total_trades * 100) if total_trades > 0 else 0,
            'avg_confidence': avg_confidence,
            'most_traded': symbol_counts,
            'action_distribution': action_dist,
            'total_shares_traded': total_shares,
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
        
        # Recommendation 3: Position sizing (based on trade frequency as a proxy)
        total_trades = self.insights.get('total_trades', 0)
        if total_trades > 50:
            recommendations.append({
                'type': 'position_size',
                'action': 'reduce',
                'value': 0.03,
                'reason': f'High trade volume ({total_trades} trades), consider reducing position size to 3% to limit risk'
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
        """Save analysis to rolling JSONL and export a standalone performance_metrics.json."""
        if not self.pnl_metrics:
            self.analyze_pnl_metrics(days_back=30)

        analysis_record = {
            'timestamp':       datetime.now().isoformat(),
            'insights':        self.insights,
            'pnl_metrics':     self.pnl_metrics,
            'recommendations': self.generate_recommendations(),
        }

        with open(output_file, 'a') as f:
            f.write(json.dumps(analysis_record) + '\n')
        logging.info(f"Analysis saved to {output_file}")

        # Standalone export for easy review (overwrites each run)
        metrics_path = Path(output_file).parent / 'performance_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(analysis_record, f, indent=2)
        logging.info(f"Performance metrics exported to {metrics_path}")
    
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
        print(f"Total Shares Traded: {self.insights.get('total_shares_traded', 0):,}")

        print("\nMost Traded Symbols:")
        for symbol, count in self.insights.get('most_traded', {}).items():
            print(f"  {symbol}: {count} trades")

        # P&L metrics section (requires closed trades from outcome_tracker)
        m = self.analyze_pnl_metrics(days_back=30)
        if m:
            targets = m.get('targets', {})
            print("\n" + "-"*60)
            print("💰 REALIZED P&L METRICS  (last 30 days, closed trades only)")
            print("-"*60)
            print(f"Closed Trades:    {m['total_closed_trades']}  "
                  f"({m['winners']}W / {m['losers']}L)")
            print(f"Win Rate:         {m['win_rate']:.1%}  "
                  f"(target ≥ {targets.get('win_rate_target', 0.55):.0%})")
            print(f"Profit Factor:    {m['profit_factor']:.2f}  "
                  f"(target ≥ {targets.get('profit_factor_target', 1.4):.1f})")
            sharpe_method = m.get('sharpe_method', 'unknown')
            print(f"Sharpe Ratio:     {m['sharpe_ratio']:.2f}  "
                  f"(target ≥ {targets.get('sharpe_target', 0.8):.1f})  [{sharpe_method}]")
            if 'sharpe_per_trade' in m and sharpe_method != 'per_trade_corrected':
                print(f"  ↳ per-trade est: {m['sharpe_per_trade']:.2f}  "
                      f"(diagnostic only — inflated vs portfolio-level)")
            print(f"Avg Win:          {m['avg_win_pct']:+.2%}    "
                  f"Avg Loss: {m['avg_loss_pct']:+.2%}")
            print(f"Expectancy/Trade: {m['expectancy_per_trade']:+.2%}")
            print(f"Total Realized:   ${m['total_realized_pnl']:+,.2f}")
            print(f"Avg Hold:         {m['avg_hold_hours']:.1f}h    "
                  f"Max Consec Losses: {m['max_consecutive_losses']}")
            gate = "✅ PASSES — consider go-live" if m['go_live_ready'] else "❌ FAILS — keep paper trading"
            print(f"\nGo-Live Gate:     {gate}")
        else:
            print("\n⏳ No closed trade outcomes yet — outcome tracker hasn't run or no completed trades.")

        try:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            import db as _db
            with _db.get_conn() as conn:
                unreconciled = conn.execute(
                    "SELECT COUNT(*) FROM unreconciled_orders WHERE recorded_at > ?",
                    [(datetime.now() - timedelta(days=30)).isoformat()]
                ).fetchone()[0]
            if unreconciled:
                print(f"\n⚠️  Unreconciled orders (last 30d): {unreconciled} "
                      f"— review unreconciled_orders table for manual action")
        except Exception:
            pass

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
