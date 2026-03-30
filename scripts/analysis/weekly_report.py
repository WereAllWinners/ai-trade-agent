#!/usr/bin/env python3
"""
Weekly Performance Report Generator

Computes full trading statistics from closed trade outcomes:
  - Sharpe ratio and max drawdown (from Alpaca portfolio history)
  - Win rate, profit factor, win/loss ratio
  - Confidence calibration (does higher AI confidence → higher win rate?)
  - Per-symbol P&L breakdown
  - Go-live readiness gate evaluation

Usage:
  python3 scripts/weekly_report.py            # last 30 days
  python3 scripts/weekly_report.py --days 7   # last 7 days
  python3 scripts/weekly_report.py --all      # entire history
"""
import os
import json
import math
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Go-live thresholds — update these as confidence grows
GO_LIVE_GATES = {
    'min_trades':    {'threshold': 100,  'desc': 'Minimum closed trades'},
    'win_rate':      {'threshold': 0.55, 'desc': 'Win rate ≥ 55%'},
    'sharpe_ratio':  {'threshold': 1.00, 'desc': 'Sharpe ratio ≥ 1.0'},
    'max_drawdown':  {'threshold': 0.15, 'desc': 'Max drawdown ≤ 15%', 'invert': True},
    'profit_factor': {'threshold': 1.40, 'desc': 'Profit factor ≥ 1.4'},
    'calibration':   {'threshold': 1,    'desc': 'Confidence monotone with win rate'},
}


class WeeklyReporter:
    def __init__(self, paper=None):
        if paper is None:
            paper = os.getenv('PAPER_TRADING', 'true').lower() != 'false'
        self.trading_client = TradingClient(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            paper=paper
        )
        self.outcomes_path = Path('logs/trade_outcomes.jsonl')
        self.report_dir = Path('logs/reports')
        self.report_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_outcomes(self, days_back=None):
        """Load closed trade outcomes, optionally filtered to last N days."""
        outcomes = []
        if not self.outcomes_path.exists():
            logging.warning("No trade outcomes file found at %s", self.outcomes_path)
            return outcomes

        cutoff = datetime.now() - timedelta(days=days_back) if days_back else None

        with open(self.outcomes_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                    if cutoff:
                        trade_dt = datetime.fromisoformat(o['exit_timestamp'])
                        if trade_dt < cutoff:
                            continue
                    outcomes.append(o)
                except (json.JSONDecodeError, KeyError):
                    pass

        logging.info("Loaded %d closed trade outcomes", len(outcomes))
        return outcomes

    def get_equity_curve(self, days_back=30):
        """Fetch daily equity snapshots from Alpaca portfolio history."""
        try:
            period = f"{min(days_back, 365)}D"
            history = self.trading_client.get_portfolio_history(
                period=period,
                timeframe="1D"
            )
            pairs = list(zip(history.timestamp, history.equity))
            # Filter None equity values
            return [(ts, eq) for ts, eq in pairs if eq is not None]
        except Exception as e:
            logging.warning("Could not fetch portfolio history: %s", e)
            return []

    # ------------------------------------------------------------------
    # Statistical calculations
    # ------------------------------------------------------------------

    def calculate_sharpe(self, equity_curve, risk_free_annual=0.05):
        """Annualized Sharpe ratio from a daily equity curve."""
        equities = [eq for _, eq in equity_curve if eq and eq > 0]
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

        rf_daily = (1 + risk_free_annual) ** (1 / 252) - 1
        sharpe = (mean_r - rf_daily) / std_dev * math.sqrt(252)
        return round(sharpe, 3)

    def calculate_max_drawdown(self, equity_curve):
        """Maximum peak-to-trough drawdown as a fraction (e.g. 0.12 = 12%)."""
        equities = [eq for _, eq in equity_curve if eq and eq > 0]
        if len(equities) < 2:
            return None

        peak = equities[0]
        max_dd = 0.0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd

        return round(max_dd, 4)

    def calculate_trade_metrics(self, outcomes):
        """Core per-trade statistics."""
        if not outcomes:
            return {}

        wins = [o for o in outcomes if o['won']]
        losses = [o for o in outcomes if not o['won']]

        total = len(outcomes)
        win_rate = len(wins) / total

        avg_win_pct = sum(o['pnl_pct'] for o in wins) / len(wins) if wins else 0.0
        avg_loss_pct = sum(o['pnl_pct'] for o in losses) / len(losses) if losses else 0.0

        gross_profit = sum(o['realized_pnl'] for o in wins) if wins else 0.0
        gross_loss = abs(sum(o['realized_pnl'] for o in losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

        win_loss_ratio = (
            abs(avg_win_pct / avg_loss_pct) if avg_loss_pct != 0 else None
        )

        total_pnl = sum(o['realized_pnl'] for o in outcomes)
        avg_hold_hours = sum(o['hold_hours'] for o in outcomes) / total

        # Best / worst single trades
        best = max(outcomes, key=lambda o: o['pnl_pct'])
        worst = min(outcomes, key=lambda o: o['pnl_pct'])

        return {
            'total_trades': total,
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': round(win_rate, 4),
            'avg_win_pct': round(avg_win_pct, 4),
            'avg_loss_pct': round(avg_loss_pct, 4),
            'win_loss_ratio': round(win_loss_ratio, 3) if win_loss_ratio else None,
            'profit_factor': round(profit_factor, 3) if profit_factor else None,
            'total_realized_pnl': round(total_pnl, 2),
            'avg_hold_hours': round(avg_hold_hours, 1),
            'best_trade': {
                'symbol': best['symbol'],
                'pnl_pct': round(best['pnl_pct'], 4),
                'pnl': round(best['realized_pnl'], 2),
            },
            'worst_trade': {
                'symbol': worst['symbol'],
                'pnl_pct': round(worst['pnl_pct'], 4),
                'pnl': round(worst['realized_pnl'], 2),
            },
        }

    def calculate_confidence_calibration(self, outcomes):
        """
        Bin trades by AI entry confidence and measure actual win rate per bin.
        A well-calibrated model should show monotonically increasing win rate
        as confidence increases.
        """
        bins = {
            '0.60-0.70': [],
            '0.70-0.80': [],
            '0.80-0.90': [],
            '0.90-1.00': [],
        }

        uncalibrated_count = 0
        for o in outcomes:
            conf = o.get('entry_confidence')
            if conf is None:
                uncalibrated_count += 1
                continue
            if conf < 0.70:
                bins['0.60-0.70'].append(o['won'])
            elif conf < 0.80:
                bins['0.70-0.80'].append(o['won'])
            elif conf < 0.90:
                bins['0.80-0.90'].append(o['won'])
            else:
                bins['0.90-1.00'].append(o['won'])

        result = {}
        for label, results in bins.items():
            if results:
                result[label] = {
                    'count': len(results),
                    'win_rate': round(sum(results) / len(results), 4),
                }

        if uncalibrated_count:
            result['_missing_confidence'] = uncalibrated_count

        # Compute calibration score: correlation between midpoint confidence and win rate
        # Simple check: is it monotonically increasing?
        win_rates = [v['win_rate'] for v in result.values() if isinstance(v, dict)]
        if len(win_rates) >= 2:
            monotone = all(
                win_rates[i] <= win_rates[i + 1]
                for i in range(len(win_rates) - 1)
            )
            result['_is_monotone'] = monotone

        return result

    def calculate_symbol_performance(self, outcomes):
        """Per-symbol trade count, win rate, and total P&L."""
        by_symbol = defaultdict(list)
        for o in outcomes:
            by_symbol[o['symbol']].append(o)

        stats = {}
        for symbol, trades in by_symbol.items():
            wins = [t for t in trades if t['won']]
            stats[symbol] = {
                'trades': len(trades),
                'win_rate': round(len(wins) / len(trades), 4),
                'total_pnl': round(sum(t['realized_pnl'] for t in trades), 2),
                'avg_pnl_pct': round(
                    sum(t['pnl_pct'] for t in trades) / len(trades), 4
                ),
                'avg_hold_hours': round(
                    sum(t['hold_hours'] for t in trades) / len(trades), 1
                ),
            }

        # Sort by total P&L descending
        return dict(
            sorted(stats.items(), key=lambda x: x[1]['total_pnl'], reverse=True)
        )

    # ------------------------------------------------------------------
    # Go-live gate evaluation
    # ------------------------------------------------------------------

    def evaluate_go_live_gates(self, trade_metrics, sharpe, max_drawdown, calibration=None):
        """Check each go-live readiness gate and return pass/fail per gate."""
        # 1 = calibration passes (monotone), 0 = fails, None = not enough data
        calib_val = None
        if calibration:
            is_monotone = calibration.get('_is_monotone')
            if is_monotone is not None:
                calib_val = 1 if is_monotone else 0

        values = {
            'min_trades':   trade_metrics.get('total_trades', 0),
            'win_rate':     trade_metrics.get('win_rate', 0),
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'profit_factor': trade_metrics.get('profit_factor'),
            'calibration':  calib_val,
        }

        gates = {}
        for gate_name, config in GO_LIVE_GATES.items():
            val = values.get(gate_name)
            threshold = config['threshold']
            invert = config.get('invert', False)  # For max_drawdown: lower is better

            if val is None:
                passed = False
            elif invert:
                passed = val <= threshold
            else:
                passed = val >= threshold

            gates[gate_name] = {
                'description': config['desc'],
                'value': val,
                'threshold': threshold,
                'passed': passed,
            }

        gates['_all_passed'] = all(g['passed'] for g in gates.values())
        return gates

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self, days_back=None):
        """Build, save, and print the full weekly report."""
        label = f"last {days_back} days" if days_back else "all time"
        logging.info("Generating performance report (%s)...", label)

        outcomes = self.load_outcomes(days_back=days_back)
        equity_curve = self.get_equity_curve(days_back=days_back or 365)

        trade_metrics = self.calculate_trade_metrics(outcomes)
        sharpe = self.calculate_sharpe(equity_curve)
        max_drawdown = self.calculate_max_drawdown(equity_curve)
        calibration = self.calculate_confidence_calibration(outcomes)
        symbol_perf = self.calculate_symbol_performance(outcomes)
        gates = self.evaluate_go_live_gates(trade_metrics, sharpe, max_drawdown, calibration)

        account_equity = None
        try:
            acct = self.trading_client.get_account()
            account_equity = float(acct.equity)
        except Exception as e:
            logging.warning("Could not fetch account equity: %s", e)

        report = {
            'generated_at': datetime.now().isoformat(),
            'period': label,
            'account_equity': account_equity,
            'portfolio': {
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'equity_curve_points': len(equity_curve),
            },
            'trades': trade_metrics,
            'confidence_calibration': calibration,
            'symbol_performance': symbol_perf,
            'go_live_gates': gates,
        }

        # Persist
        date_str = datetime.now().strftime('%Y-%m-%d')
        report_path = self.report_dir / f'weekly_report_{date_str}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        with open(self.report_dir / 'latest_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        self._print_report(report)
        logging.info("Report saved to %s", report_path)
        return report

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _print_report(self, report):
        t = report['trades']
        p = report['portfolio']
        gates = report['go_live_gates']
        calib = report['confidence_calibration']

        W = 65
        print("\n" + "=" * W)
        print("  TRADING PERFORMANCE REPORT")
        print(f"  Period:  {report['period']}")
        print(f"  Generated: {report['generated_at'][:19]}")
        if report['account_equity']:
            print(f"  Account Equity: ${report['account_equity']:,.2f}")
        print("=" * W)

        print("\n── PORTFOLIO ──")
        sharpe = p.get('sharpe_ratio')
        dd = p.get('max_drawdown')
        print(f"  Sharpe Ratio : {sharpe if sharpe is not None else 'N/A (need more data)'}")
        print(f"  Max Drawdown : {dd:.1%}" if dd is not None else "  Max Drawdown : N/A")

        if t:
            print("\n── TRADE OUTCOMES ──")
            print(f"  Closed Trades  : {t.get('total_trades', 0)}")
            print(f"  Win Rate       : {t.get('win_rate', 0):.1%}  "
                  f"({t.get('wins', 0)}W / {t.get('losses', 0)}L)")
            print(f"  Avg Win        : {t.get('avg_win_pct', 0):+.2%}")
            print(f"  Avg Loss       : {t.get('avg_loss_pct', 0):+.2%}")
            wl = t.get('win_loss_ratio')
            pf = t.get('profit_factor')
            print(f"  Win/Loss Ratio : {wl:.2f}x" if wl else "  Win/Loss Ratio : N/A")
            print(f"  Profit Factor  : {pf:.2f}" if pf else "  Profit Factor  : N/A")
            print(f"  Total P&L      : ${t.get('total_realized_pnl', 0):+,.2f}")
            print(f"  Avg Hold Time  : {t.get('avg_hold_hours', 0):.1f} hrs")
            best = t.get('best_trade', {})
            worst = t.get('worst_trade', {})
            if best:
                print(f"  Best Trade     : {best.get('symbol')} "
                      f"{best.get('pnl_pct', 0):+.1%} (${best.get('pnl', 0):+,.2f})")
            if worst:
                print(f"  Worst Trade    : {worst.get('symbol')} "
                      f"{worst.get('pnl_pct', 0):+.1%} (${worst.get('pnl', 0):+,.2f})")
        else:
            print("\n  No closed trades yet — run outcome_tracker.py first.")

        print("\n── CONFIDENCE CALIBRATION ──")
        calibration_bins = {
            k: v for k, v in calib.items()
            if not k.startswith('_') and isinstance(v, dict)
        }
        if calibration_bins:
            for bin_label, stats in calibration_bins.items():
                wr = stats['win_rate']
                bar = '█' * int(wr * 20) + '░' * (20 - int(wr * 20))
                print(f"  {bin_label}: [{bar}] {wr:.1%}  (n={stats['count']})")
            if calib.get('_is_monotone') is False:
                print("  ! Confidence not correlated with outcomes — model may need retraining")
        else:
            print("  Not enough data yet.")

        print("\n── TOP SYMBOLS (by P&L) ──")
        sym_perf = report.get('symbol_performance', {})
        if sym_perf:
            for symbol, stats in list(sym_perf.items())[:10]:
                pnl_str = f"${stats['total_pnl']:+,.2f}"
                print(f"  {symbol:6s}  WR={stats['win_rate']:.0%}  "
                      f"{pnl_str:>10}  ({stats['trades']} trades)")
        else:
            print("  No data yet.")

        print("\n── GO-LIVE READINESS ──")
        gate_rows = [(k, v) for k, v in gates.items() if not k.startswith('_')]
        for name, gate in gate_rows:
            status = "PASS" if gate['passed'] else "FAIL"
            val = gate['value']
            thr = gate['threshold']

            def fmt(x):
                if x is None:
                    return "N/A"
                if isinstance(x, float):
                    return f"{x:.3f}"
                return str(x)

            print(f"  [{status}] {gate['description']:<28} "
                  f"value={fmt(val):>7}  threshold={fmt(thr)}")

        overall = gates.get('_all_passed', False)
        verdict = "READY FOR LIVE TRADING" if overall else "NOT YET READY"
        print(f"\n  Verdict: {verdict}")
        print("=" * W + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate weekly trading performance report")
    parser.add_argument('--days', type=int, default=30,
                        help='Days of history to include (default: 30)')
    parser.add_argument('--all', action='store_true',
                        help='Include entire trade history')
    args = parser.parse_args()

    days = None if args.all else args.days
    reporter = WeeklyReporter()
    reporter.generate_report(days_back=days)
