#!/usr/bin/env python3
"""
Options Performance Analyzer
Reads realized P&L from options_trade_outcomes.jsonl (written by
options_outcome_tracker.py) and computes the same metrics as the stock
performance analyzer — including a deterministic go-live readiness gate.

Options-specific go-live thresholds are slightly different from stocks:
  - win_rate      >= 0.45  (options buyers typically 40-50%; offset by larger winners)
  - profit_factor >= 1.4
  - sharpe        >= 0.7   (higher per-trade volatility means a lower bar here)
"""
import json
import math
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetPortfolioHistoryRequest

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

_RISK_FREE_ANNUAL = 0.05  # used for Sharpe calculations throughout


class OptionsPerformanceAnalyzer:
    def __init__(self):
        _paper = os.getenv('PAPER_TRADING', 'true').lower() != 'false'
        self.trading_client = TradingClient(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            paper=_paper
        )
        self.outcomes_file = Path('logs/options_trade_outcomes.jsonl')
        self.output_path = Path('logs/options_performance_metrics.json')
        self.training_data_path = Path('finetune/data/options_training_data.json')
        self.pnl_metrics: dict = {}

        Path('logs').mkdir(exist_ok=True)
        Path('finetune/data').mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_outcomes(self, days_back=None):
        """Load realized P&L records from options_trade_outcomes.jsonl."""
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
            logging.info(
                "No options_trade_outcomes.jsonl yet — "
                "outcome tracker hasn't run or no closed options trades."
            )
        except Exception as e:
            logging.error("Error loading options outcomes: %s", e)
        return outcomes

    def get_current_positions(self):
        """Get current open options positions from Alpaca."""
        try:
            positions = self.trading_client.get_all_positions()
            options_positions = []
            for p in positions:
                # Options contracts have long symbols (> 10 chars)
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
            logging.error("Failed to get options positions: %s", e)
            return []

    # ------------------------------------------------------------------
    # Equity-curve Sharpe (portfolio-level, accounts for position sizing)
    # ------------------------------------------------------------------

    def get_equity_curve(self, days_back=30):
        """Fetch daily equity snapshots from Alpaca portfolio history."""
        try:
            req = GetPortfolioHistoryRequest(
                period=f"{min(days_back, 365)}D",
                timeframe="1D",
            )
            history = self.trading_client.get_portfolio_history(req)
            return [(ts, eq) for ts, eq in zip(history.timestamp, history.equity)
                    if eq is not None]
        except Exception as e:
            logging.warning("Could not fetch portfolio history: %s", e)
            return []

    def _equity_curve_sharpe(self, days_back=30):
        """
        Annualised Sharpe from the daily portfolio equity curve.
        Returns None when insufficient data is available.
        This is the portfolio-level Sharpe — it correctly reflects the 1–3%
        position-sizing used by the options agent.
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

    # ------------------------------------------------------------------
    # Metrics calculation
    # ------------------------------------------------------------------

    def analyze_pnl_metrics(self, days_back=30):
        """
        Compute options trading metrics from closed outcomes.

        Uses options_trade_outcomes.jsonl (written by options_outcome_tracker.py)
        which contains actual Alpaca fill prices and realized P&L.

        Returns a dict with Sharpe, profit factor, win rate, and a
        go_live_ready boolean against options-specific thresholds.
        """
        outcomes = self.load_outcomes(days_back=days_back)
        if not outcomes:
            logging.info("No closed options outcomes yet — keep paper trading.")
            return {}

        pnl_pcts    = [o['pnl_pct']      for o in outcomes]
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
        avg_hold_days   = max(avg_hold_hours / 6.5, 0.5)  # floor at half a trading day
        trades_per_year = 252 / avg_hold_days

        mean_r = float(np.mean(pnl_pcts))
        std_r  = float(np.std(pnl_pcts, ddof=1)) if len(pnl_pcts) > 1 else 0.0
        rf_per_trade = (1 + _RISK_FREE_ANNUAL) ** (avg_hold_days / 252) - 1
        sharpe_per_trade = (
            (mean_r - rf_per_trade) / std_r * math.sqrt(trades_per_year)
            if std_r > 0 else 0.0
        )

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

        # Go-live gate — options-specific thresholds
        WIN_RATE_TARGET      = 0.45
        PROFIT_FACTOR_TARGET = 1.4
        SHARPE_TARGET        = 0.7

        go_live_ready = (
            win_rate      >= WIN_RATE_TARGET      and
            profit_factor >= PROFIT_FACTOR_TARGET and
            sharpe        >= SHARPE_TARGET
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
            'sharpe_ratio':           round(sharpe, 3),
            'sharpe_method':          'equity_curve' if sharpe_equity is not None else 'per_trade_corrected',
            'sharpe_per_trade':       round(sharpe_per_trade, 3),
            'total_realized_pnl':     round(total_pnl, 2),
            'gross_profit':           round(gross_profit, 2),
            'gross_loss':             round(gross_loss, 2),
            'avg_hold_hours':         round(avg_hold_hours, 1),
            'max_consecutive_losses': max_consec,
            'go_live_ready':          go_live_ready,
            'metrics_period_days':    days_back,
            'targets': {
                'win_rate_target':      WIN_RATE_TARGET,
                'profit_factor_target': PROFIT_FACTOR_TARGET,
                'sharpe_target':        SHARPE_TARGET,
            },
            'computed_at': datetime.now().isoformat(),
        }
        return self.pnl_metrics

    # ------------------------------------------------------------------
    # Training data generation
    # ------------------------------------------------------------------

    def generate_training_data(self, outcomes, positions):
        """Generate training examples from closed outcomes and open positions."""
        training_examples = []

        # Winners (profit > 30%)
        winners = [o for o in outcomes if o.get('won') and o.get('pnl_pct', 0) > 0.30]
        for o in winners[-5:]:
            training_examples.append({
                'input': (
                    f"Analyze options trade: {o.get('underlying', o['symbol'][:10])}, "
                    f"confidence: {o.get('entry_confidence', 0):.2f}"
                ),
                'output': (
                    f"TRADE. Reasoning: {o.get('entry_reasoning', 'Good setup')}. "
                    f"Result: Profitable exit at {o['pnl_pct']:+.1%}"
                ),
                'label': 'winner',
            })

        # Losers (loss > 30%)
        losers = [o for o in outcomes if not o.get('won') and o.get('pnl_pct', 0) < -0.30]
        for o in losers[-5:]:
            training_examples.append({
                'input': (
                    f"Analyze options trade: {o.get('underlying', o['symbol'][:10])}, "
                    f"confidence: {o.get('entry_confidence', 0):.2f}"
                ),
                'output': (
                    f"AVOID. Reasoning: {o.get('entry_reasoning', 'Poor setup')}. "
                    f"Result: Loss at {o['pnl_pct']:+.1%}"
                ),
                'label': 'loser',
            })

        # Strong open positions (unrealized > 40%)
        for pos in [p for p in positions if p['unrealized_plpc'] > 0.4][:3]:
            training_examples.append({
                'input': f"Evaluate options position: {pos['contract'][:10]}...",
                'output': (
                    f"STRONG. Current unrealized P&L: {pos['unrealized_plpc']:+.1%}. "
                    f"Consider taking profits."
                ),
                'label': 'strong_position',
            })

        if training_examples:
            with open(self.training_data_path, 'w') as f:
                json.dump(training_examples, f, indent=2)
            logging.info("Generated %d options training examples", len(training_examples))

        return len(training_examples)

    # ------------------------------------------------------------------
    # Save + print
    # ------------------------------------------------------------------

    def save_analysis(self, output_file='logs/options_daily_analysis.jsonl'):
        """Save analysis to rolling JSONL and export standalone options_performance_metrics.json."""
        if not self.pnl_metrics:
            self.analyze_pnl_metrics(days_back=30)

        analysis_record = {
            'timestamp':   datetime.now().isoformat(),
            'pnl_metrics': self.pnl_metrics,
        }

        with open(output_file, 'a') as f:
            f.write(json.dumps(analysis_record) + '\n')
        logging.info("Options analysis saved to %s", output_file)

        with open(self.output_path, 'w') as f:
            json.dump(analysis_record, f, indent=2)
        logging.info("Options performance metrics exported to %s", self.output_path)

    def print_report(self):
        """Print a human-readable options performance report with go-live gate."""
        m = self.analyze_pnl_metrics(days_back=30)

        print("\n" + "=" * 60)
        print("📊 OPTIONS PERFORMANCE REPORT")
        print("=" * 60)

        if not m:
            print("⏳ No closed options outcomes yet — keep paper trading.")
            print("=" * 60 + "\n")
            return

        targets = m.get('targets', {})
        print(f"Closed Trades:    {m['total_closed_trades']}  "
              f"({m['winners']}W / {m['losers']}L)")
        print(f"Win Rate:         {m['win_rate']:.1%}  "
              f"(target ≥ {targets.get('win_rate_target', 0.45):.0%})")
        print(f"Profit Factor:    {m['profit_factor']:.2f}  "
              f"(target ≥ {targets.get('profit_factor_target', 1.4):.1f})")
        print(f"Sharpe Ratio:     {m['sharpe_ratio']:.2f}  "
              f"(target ≥ {targets.get('sharpe_target', 0.7):.1f})")
        print(f"Avg Win:          {m['avg_win_pct']:+.2%}    "
              f"Avg Loss: {m['avg_loss_pct']:+.2%}")
        print(f"Expectancy/Trade: {m['expectancy_per_trade']:+.2%}")
        print(f"Total Realized:   ${m['total_realized_pnl']:+,.2f}")
        print(f"Avg Hold:         {m['avg_hold_hours']:.1f}h    "
              f"Max Consec Losses: {m['max_consecutive_losses']}")
        gate = "✅ PASSES — consider go-live" if m['go_live_ready'] else "❌ FAILS — keep paper trading"
        print(f"\nGo-Live Gate:     {gate}")
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_analysis(self):
        """Run full options performance analysis."""
        logging.info("📊 Starting Options Performance Analysis")

        outcomes  = self.load_outcomes(days_back=30)
        positions = self.get_current_positions()

        self.analyze_pnl_metrics(days_back=30)

        if self.pnl_metrics:
            m = self.pnl_metrics
            logging.info("Options closed trades: %d  (%dW / %dL)",
                         m['total_closed_trades'], m['winners'], m['losers'])
            logging.info("Win Rate: %.1f%%  Profit Factor: %.2f  Sharpe: %.2f",
                         m['win_rate'] * 100, m['profit_factor'], m['sharpe_ratio'])
            gate = "PASSES" if m['go_live_ready'] else "FAILS"
            logging.info("Go-Live Gate: %s", gate)

        if positions:
            total_pl = sum(p['unrealized_pl'] for p in positions)
            logging.info("Open options positions: %d  Unrealized P&L: $%+.2f",
                         len(positions), total_pl)

        num_examples = self.generate_training_data(outcomes, positions)
        if num_examples >= 3:
            logging.info("Ready for fine-tuning with %d examples", num_examples)
        else:
            logging.info("Need %d more examples for fine-tuning", max(0, 3 - num_examples))

        self.save_analysis()
        logging.info("✅ Options performance analysis complete")


if __name__ == "__main__":
    analyzer = OptionsPerformanceAnalyzer()
    analyzer.print_report()
    analyzer.run_analysis()
