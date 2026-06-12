#!/usr/bin/env python3
"""
Trading Daemon - Runs stock trading bot 24/7
"""
import json
import os
import signal
import sys
import time
import logging
import subprocess
import traceback
from datetime import datetime, timedelta
from pathlib import Path
import pytz

_SCRIPTS_DIR = Path(__file__).resolve().parent
_HEARTBEAT_FILE = _SCRIPTS_DIR.parent / 'logs' / 'heartbeat_stock.json'
sys.path.append(str(_SCRIPTS_DIR))

import inference_client
from preflight_check import run_preflight

# When TRADING_LIVE_ONLY=1 this instance is the live stock execution engine.
# All self-improvement work (fine-tuning, analysis, weekend reports) is skipped
# so the live bot never triggers GPU-intensive background jobs or modifies the model.
_LIVE_ONLY = os.getenv('TRADING_LIVE_ONLY', '0') == '1'


def _write_heartbeat(status: str, market_open: bool) -> None:
    """Update heartbeat file so health_server.py can report daemon liveness."""
    try:
        _HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
        _HEARTBEAT_FILE.write_text(json.dumps({
            'daemon': 'stock',
            'status': status,
            'market_open': market_open,
            'ts': datetime.now(datetime.UTC).isoformat(),
        }))
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TradingDaemon:
    def __init__(self):
        self.est = pytz.timezone('US/Eastern')
        self.market_open = datetime.strptime('09:30', '%H:%M').time()
        self.market_close = datetime.strptime('16:00', '%H:%M').time()
        self.trading_interval = 30  # 30 minutes
        self.analysis_time = datetime.strptime('17:00', '%H:%M').time()  # 5:00 PM
        self.finetune_time = datetime.strptime('20:00', '%H:%M').time()  # 8:00 PM
        self.weekly_report_time = datetime.strptime('10:00', '%H:%M').time()  # Sat 10:00 AM
        self.weekend_strategist_time = datetime.strptime('11:00', '%H:%M').time()  # Sat 11:00 AM

        self._finetune_requested = False
        self.trading_client = None  # set by _run_preflight; used for clock API
        signal.signal(signal.SIGUSR1, self._handle_finetune_signal)

        logging.info("🤖 Trading Daemon Initialized")
        logging.info("⏰ Stock trading every 30 minutes")
        logging.info("📊 Performance analysis scheduled for 5:00 PM EST daily")
        logging.info("🔍 Market research + 🎓 Fine-tuning scheduled for 8:00 PM EST daily")
        logging.info("🏖️  Weekend deep analysis scheduled for Saturday 11:00 AM EST")
        logging.info("📬 Send SIGUSR1 to trigger an off-cycle fine-tune: systemctl kill -s SIGUSR1 ai-trading-bot.service")

        # Preflight: validate Alpaca account before any sessions start
        self._run_preflight()

        # NOTE: do NOT preload the model here. Every trading session runs as a
        # subprocess (autonomous_agent.py) with its own Python interpreter and its
        # own _MODEL_CACHE — so a model loaded in the daemon process is never shared
        # with or used by those subprocesses. Holding 45 GB in the daemon process
        # just steals memory from the nightly fine-tune subprocess (which needs it)
        # and from the other daemon's inference subprocess.
    
    def _run_preflight(self) -> None:
        """Validate the Alpaca account. Logs critical issues but never aborts the daemon."""
        try:
            from dotenv import load_dotenv
            from alpaca.trading.client import TradingClient
            load_dotenv()
            client = TradingClient(
                os.getenv('ALPACA_API_KEY'),
                os.getenv('ALPACA_SECRET_KEY'),
                paper=os.getenv('PAPER_TRADING', 'true').lower() != 'false',
            )
            self.trading_client = client
            result = run_preflight(client, require_options=False)
            if not result.ok:
                for issue in result.issues:
                    logging.critical(f"⛔ PREFLIGHT FAILED: {issue}")
                logging.critical(
                    "Daemon will continue but trading may be blocked by Alpaca. "
                    "Fix the issues above and restart."
                )
        except Exception as e:
            logging.warning(f"⚠️  Preflight check failed to run: {e}")

    def _handle_finetune_signal(self, signum, frame):
        """SIGUSR1 handler — set flag to trigger an off-cycle fine-tune."""
        logging.info("📬 SIGUSR1 received — off-cycle fine-tune requested")
        self._finetune_requested = True

    def is_market_open(self):
        """Check if market is currently open, using the Alpaca clock API when available."""
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except Exception as e:
            logging.warning(f"⚠️  Clock API unavailable, falling back to time-based check: {e}")
            now = datetime.now(self.est)
            if now.weekday() >= 5:
                return False
            return self.market_open <= now.time() <= self.market_close

    def get_next_market_open(self):
        """Return the next market open time, using the Alpaca clock API when available."""
        try:
            clock = self.trading_client.get_clock()
            return clock.next_open.astimezone(self.est)
        except Exception as e:
            logging.warning(f"⚠️  Clock API unavailable, falling back to time-based calculation: {e}")
            now = datetime.now(self.est)
            # If weekend, next Monday
            if now.weekday() >= 5:
                days_ahead = 7 - now.weekday()
                next_open = now + timedelta(days=days_ahead)
                next_open = next_open.replace(hour=9, minute=30, second=0, microsecond=0)
            # If before market open today
            elif now.time() < self.market_open:
                next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            # If after market close, next day
            else:
                next_open = now + timedelta(days=1)
                next_open = next_open.replace(hour=9, minute=30, second=0, microsecond=0)
                # Skip weekend
                if next_open.weekday() >= 5:
                    days_ahead = 7 - next_open.weekday()
                    next_open = next_open + timedelta(days=days_ahead)
            return next_open
    
    def run_trading_session(self):
        """Run stock trading session."""
        try:
            logging.info("======================================================================")
            logging.info("📈 STARTING TRADING SESSION")
            logging.info("======================================================================")
            logging.info("🔄 Initializing trading agent...")

            result = subprocess.run(
                [sys.executable, str(_SCRIPTS_DIR / 'agents' / 'autonomous_agent.py')],
                timeout=14400  # 4 hours — 35 stocks × ~6 min/inference
            )
            
            if result.returncode == 0:
                logging.info("✅ Trading session completed")
            else:
                logging.error(f"❌ Trading session failed with code: {result.returncode}")
                
        except subprocess.TimeoutExpired:
            logging.error("❌ Trading session timed out")
        except Exception as e:
            logging.error(f"❌ Trading session failed: {e}")
    
    def run_live_outcome_tracking(self):
        """Resolve fill prices and P&L for live trades so they enter the training pool.

        The paper bot's nightly fine-tune calls training_data_builder which queries
        all executed decisions regardless of source. Without outcomes in the DB, live
        decisions can't be labeled and contribute no training signal. This runs the
        same outcome_tracker subprocess as the paper bot, but inherits PAPER_TRADING=false
        from the live daemon's environment so it queries the live Alpaca account.
        """
        try:
            logging.info("📥 [live] Fetching live trade fill prices and computing P&L...")
            result = subprocess.run(
                [sys.executable, str(_SCRIPTS_DIR / 'analysis' / 'outcome_tracker.py')],
                timeout=120,
            )
            if result.returncode == 0:
                logging.info("✅ [live] Outcome tracking complete — live trades will feed tonight's fine-tune")
            else:
                logging.warning(f"⚠️ [live] Outcome tracker exited with code: {result.returncode}")
        except Exception as e:
            logging.warning(f"⚠️ [live] Outcome tracker failed: {e}")

    def run_performance_analysis(self):
        """Run outcome tracking then performance analysis."""
        # Step 1: Enrich trade log with fill prices and compute P&L
        try:
            logging.info("📥 Fetching trade fill prices and computing P&L...")
            result = subprocess.run(
                [sys.executable, str(_SCRIPTS_DIR / 'analysis' / 'outcome_tracker.py')],
                timeout=120
            )
            if result.returncode != 0:
                logging.warning(f"⚠️ Outcome tracker exited with code: {result.returncode}")
        except Exception as e:
            logging.warning(f"⚠️ Outcome tracker failed: {e}")

        # Step 2: Run core performance analysis
        try:
            logging.info("🔔 Time for performance analysis!")
            logging.info("======================================================================")
            logging.info("📊 RUNNING PERFORMANCE ANALYSIS")
            logging.info("======================================================================")

            result = subprocess.run(
                [sys.executable, str(_SCRIPTS_DIR / 'analysis' / 'performance_analyzer.py')],
                timeout=300
            )

            if result.returncode == 0:
                logging.info("✅ Performance analysis complete")
            else:
                logging.error(f"❌ Performance analysis failed with code: {result.returncode}")

        except Exception as e:
            logging.error(f"❌ Performance analysis failed: {e}")

        # Step 3: Send daily buy summary email
        try:
            from alerts import send_daily_buy_summary
            send_daily_buy_summary('StockAgent')
        except Exception as e:
            logging.warning(f"⚠️ Daily buy summary email failed: {e}")

        # Step 4: Live-trading readiness check (only meaningful in paper mode)
        if os.getenv('PAPER_TRADING', 'true').lower() != 'false':
            try:
                import db as _db
                from dotenv import load_dotenv
                from alpaca.trading.client import TradingClient
                from allocation_controller import AllocationController
                from alerts import alert_live_trading_ready
                load_dotenv()
                _client = TradingClient(
                    os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_SECRET_KEY'), paper=True
                )
                ctrl = AllocationController(_db)
                m, equity, paper_days, unmet = ctrl.check_live_readiness(_client)
                alert_live_trading_ready('stock-bot', m, equity, paper_days, unmet)
            except Exception as e:
                logging.warning(f"⚠️ Live readiness check failed: {e}")

    def run_weekly_report(self):
        """Generate the weekly performance report (runs Saturday morning)."""
        try:
            logging.info("📊 Running weekly performance report...")
            result = subprocess.run(
                [sys.executable, str(_SCRIPTS_DIR / 'analysis' / 'weekly_report.py'), '--days', '7'],
                timeout=300
            )
            if result.returncode == 0:
                logging.info("✅ Weekly report complete")
            else:
                logging.error(f"❌ Weekly report failed with code: {result.returncode}")
        except Exception as e:
            logging.error(f"❌ Weekly report failed: {e}")

    def run_market_research(self):
        """Run nightly market research before fine-tuning."""
        try:
            logging.info("🔍 Running nightly market research...")
            result = subprocess.run(
                [sys.executable, str(_SCRIPTS_DIR / 'data' / 'market_researcher.py'), '--bot', 'trading'],
                timeout=600
            )
            if result.returncode == 0:
                logging.info("✅ Market research complete")
            else:
                logging.warning(f"⚠️ Market research exited with code: {result.returncode}")
        except Exception as e:
            logging.warning(f"⚠️ Market research failed (continuing to fine-tune): {e}")

    def run_weekend_strategist(self):
        """Run deep weekend analysis (Saturday only)."""
        try:
            logging.info("======================================================================")
            logging.info("🏖️  RUNNING WEEKEND DEEP ANALYSIS")
            logging.info("======================================================================")
            result = subprocess.run(
                [sys.executable, str(_SCRIPTS_DIR / 'analysis' / 'weekend_strategist.py')],
                timeout=1800  # 30 min max
            )
            if result.returncode == 0:
                logging.info("✅ Weekend analysis complete")
            else:
                logging.error(f"❌ Weekend analysis failed with code: {result.returncode}")
        except Exception as e:
            logging.error(f"❌ Weekend analysis failed: {e}")

    def run_strategy_evolver(self):
        """Evolve trading strategies and generate synthetic training examples (Saturday only)."""
        try:
            logging.info("=" * 70)
            logging.info("🧬 RUNNING STRATEGY EVOLVER")
            logging.info("=" * 70)
            result = subprocess.run(
                [sys.executable, str(_SCRIPTS_DIR / 'analysis' / 'strategy_evolver.py')],
                timeout=1200,  # 20 min max
            )
            if result.returncode == 0:
                logging.info("✅ Strategy evolver complete")
            else:
                logging.error(f"❌ Strategy evolver failed with code: {result.returncode}")
        except subprocess.TimeoutExpired:
            logging.error("❌ Strategy evolver timed out (20 min limit)")
        except Exception as e:
            logging.error(f"❌ Strategy evolver failed: {e}")

    def run_training_data_builder(self):
        """Convert today's decisions + outcomes into labelled training examples."""
        try:
            logging.info("🏗️  Building training examples from live decisions...")
            result = subprocess.run(
                [sys.executable, str(_SCRIPTS_DIR / 'training' / 'training_data_builder.py'), '--bot', 'stock'],
                timeout=300
            )
            if result.returncode == 0:
                logging.info("✅ Training data builder complete")
            else:
                logging.warning(f"⚠️  Training data builder exited with code: {result.returncode}")
        except Exception as e:
            logging.warning(f"⚠️  Training data builder failed (continuing to fine-tune): {e}")

    def run_online_training(self) -> None:
        """Trigger lightweight LoRA update if enough new outcomes have closed."""
        try:
            logging.info("🔬 Checking online training threshold...")
            result = subprocess.run(
                [sys.executable, str(_SCRIPTS_DIR / 'training' / 'online_trainer.py')],
                timeout=1800,
            )
            if result.returncode == 0:
                logging.info("✅ Online training check complete")
            else:
                logging.warning(f"⚠️  Online trainer exited with code: {result.returncode}")
        except subprocess.TimeoutExpired:
            logging.error("❌ Online training timed out")
        except Exception as e:
            logging.warning(f"⚠️  Online training failed: {e}")

    def run_dpo_builder(self):
        """Build DPO preference pairs from this session's labelled outcomes."""
        dpo_script = _SCRIPTS_DIR.parent / 'finetune' / 'build_dpo_dataset.py'
        try:
            result = subprocess.run([sys.executable, str(dpo_script)], timeout=120)
            if result.returncode == 0:
                logging.info("✅ DPO pairs built")
            else:
                logging.warning(f"⚠️  DPO dataset build returned code {result.returncode}")
        except Exception as e:
            logging.warning(f"⚠️  DPO dataset build failed (non-fatal): {e}")

    def run_finetuning(self):
        """Run market research, build training data, then full fine-tune."""
        self.run_market_research()
        self.run_training_data_builder()
        self.run_dpo_builder()
        # Free GPU VRAM by unloading the inference model before the training job loads
        # the 32B base model.  Returns False if blocked (market hours / lock contention)
        # — in that case we abort: the server is still live and must not be disturbed.
        stopped = inference_client.stop_for_finetuning()
        if not stopped:
            from alerts import send_alert, AlertLevel
            send_alert(AlertLevel.WARNING, 'finetune_aborted_server_live',
                       'Fine-tune aborted: vLLM stop was blocked (market hours or lock '
                       'contention). Inference server remains live.')
            logging.warning("⚠️  Fine-tune aborted — inference server still live; skipping this cycle")
            return
        try:
            logging.info("🔔 Time for daily model fine-tuning!")
            logging.info("======================================================================")
            logging.info("🎓 RUNNING MODEL FINE-TUNING")
            logging.info("======================================================================")

            result = subprocess.run(
                [sys.executable, str(_SCRIPTS_DIR / 'training' / 'finetune_model.py')],
                # No timeout here — finetune_model.py manages its own internal timeouts
                # (3600s for training + 2400s for promotion eval). Killing it early
                # causes the orphan-process issue where promotion never runs.
            )

            if result.returncode == 0:
                logging.info("✅ Model fine-tuning complete")
            else:
                logging.error(f"❌ Fine-tuning failed with code: {result.returncode}")

        except Exception as e:
            logging.error(f"❌ Fine-tuning failed: {e}")
        finally:
            # Restart vLLM here, after finetune_model.py and all its children (training +
            # eval subprocesses) have fully exited and released their GPU memory.
            # start_after_finetuning() checks is-active first — no-ops if already up.
            inference_client.start_after_finetuning()
    
    def sleep_until_next_event(self):
        """Sleep until next scheduled event."""
        now = datetime.now(self.est)
        
        # Check for scheduled events today
        events = []
        
        # Market open
        if not self.is_market_open():
            next_open = self.get_next_market_open()
            events.append(('Market open', next_open))
        
        # Analysis time (5:00 PM)
        analysis_dt = now.replace(hour=17, minute=0, second=0, microsecond=0)
        if now < analysis_dt:
            events.append(('Performance Analysis', analysis_dt))
        
        # Fine-tuning time (8:00 PM)
        finetune_dt = now.replace(hour=20, minute=0, second=0, microsecond=0)
        if now < finetune_dt:
            events.append(('Model Fine-tuning', finetune_dt))
        
        # Find next event
        if events:
            events.sort(key=lambda x: x[1])
            event_name, event_time = events[0]
            sleep_seconds = (event_time - now).total_seconds()
            
            if sleep_seconds > 0:
                hours = sleep_seconds / 3600
                minutes = (sleep_seconds % 3600) / 60
                logging.info(f"💤 Next event: {event_name} in {hours:.1f} hours" if hours >= 1 else f"💤 Next event: {event_name} in {minutes:.1f} minutes")
                time.sleep(max(sleep_seconds, 60))  # Always sleep at least 60s to avoid tight spin
        else:
            # Sleep until tomorrow
            tomorrow = now + timedelta(days=1)
            tomorrow = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
            sleep_seconds = (tomorrow - now).total_seconds()
            logging.info(f"💤 Sleeping until tomorrow ({sleep_seconds/3600:.1f} hours)")
            time.sleep(sleep_seconds)
    
    def run(self):
        """Main daemon loop."""
        logging.info("🚀 Starting Trading Daemon - Running 24/7")
        logging.info("📈 Trading: 9:30 AM - 4:00 PM EST")
        logging.info("📊 Analysis: 5:00 PM EST (daily)")
        logging.info("🔍 Research + 🎓 Learning: 8:00 PM EST (daily)")
        logging.info("🏖️  Weekend analysis: Saturday 11:00 AM EST")

        last_trade_time = None
        # Initialise from wall-clock time so a restart mid-day doesn't
        # immediately re-run analysis or fine-tuning that already ran today.
        _startup_now = datetime.now(self.est)
        _startup = _startup_now.time()
        analysis_done_today = _startup >= self.analysis_time
        finetune_done_today = _startup >= self.finetune_time
        weekly_report_done_this_week = False
        weekend_strategist_done_this_week = False
        last_weekly_report_week = None
        last_reset_date = _startup_now.date()  # tracks the last date flags were reset

        while True:
            try:
                now = datetime.now(self.est)
                current_time = now.time()
                current_date = now.date()
                current_week = now.isocalendar()[1]

                # Off-cycle fine-tune triggered by SIGUSR1 (paper bot only)
                if self._finetune_requested:
                    self._finetune_requested = False
                    if not _LIVE_ONLY:
                        if self.is_market_open():
                            # Never fine-tune during market hours via SIGUSR1 — stop_for_finetuning
                            # would kill a live vLLM server and orphan active trading sessions.
                            from alerts import send_alert, AlertLevel
                            send_alert(AlertLevel.WARNING, 'sigusr1_deferred_market_hours',
                                       'Off-cycle fine-tune deferred: SIGUSR1 received during '
                                       'market hours. Fine-tune will run at the scheduled time.')
                            logging.warning(
                                "⚠️  SIGUSR1 fine-tune ignored — market is open. "
                                "Fine-tune will run at scheduled time after close."
                            )
                        else:
                            finetune_done_today = False
                            logging.info("🔔 Running off-cycle fine-tune (SIGUSR1)...")
                            self.run_finetuning()
                            finetune_done_today = True
                    else:
                        logging.info("📬 SIGUSR1 received but TRADING_LIVE_ONLY=1 — ignoring fine-tune request")

                # Reset daily flags at midnight — use a dedicated date tracker so
                # the reset fires exactly once per calendar day regardless of whether
                # any trades happened (the old last_trade_time approach missed resets
                # on weekends and caused double fine-tunes on market-open mornings).
                if current_date != last_reset_date:
                    analysis_done_today = False
                    finetune_done_today = False
                    last_reset_date = current_date

                # Reset weekly flags on new calendar week
                if last_weekly_report_week != current_week:
                    weekly_report_done_this_week = False
                    weekend_strategist_done_this_week = False

                logging.info(f"📅 Current time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                _write_heartbeat('running', self.is_market_open())
                
                # Weekly report on Saturday at 10:00 AM (paper bot only)
                if (not _LIVE_ONLY
                        and now.weekday() == 5  # Saturday
                        and not weekly_report_done_this_week
                        and current_time >= self.weekly_report_time):
                    self.run_weekly_report()
                    weekly_report_done_this_week = True
                    last_weekly_report_week = current_week

                # Weekend deep analysis + strategy evolution on Saturday at 11:00 AM (paper bot only)
                if (not _LIVE_ONLY
                        and now.weekday() == 5  # Saturday
                        and not weekend_strategist_done_this_week
                        and current_time >= self.weekend_strategist_time):
                    self.run_weekend_strategist()
                    self.run_strategy_evolver()   # evolve strategies, enrich training_data.json
                    weekend_strategist_done_this_week = True

                # Performance analysis at 5:00 PM (paper bot only)
                if (not _LIVE_ONLY
                        and not analysis_done_today
                        and current_time >= self.analysis_time):
                    self.run_performance_analysis()
                    self.run_online_training()
                    analysis_done_today = True

                # Live outcome tracking at 5:00 PM (live bot only)
                # Resolves fill prices and P&L for live trades so they can be
                # labeled and included in the paper bot's nightly fine-tune.
                if (_LIVE_ONLY
                        and not analysis_done_today
                        and current_time >= self.analysis_time):
                    self.run_live_outcome_tracking()
                    analysis_done_today = True

                # Fine-tuning at 8:00 PM (paper bot only) — never run during market hours
                if (not _LIVE_ONLY
                        and not finetune_done_today
                        and current_time >= self.finetune_time
                        and not self.is_market_open()):
                    self.run_finetuning()
                    finetune_done_today = True
                
                # Trading during market hours
                if self.is_market_open():
                    should_trade = False
                    
                    if last_trade_time is None:
                        should_trade = True
                    else:
                        time_since_trade = (now - last_trade_time).total_seconds()
                        if time_since_trade >= self.trading_interval * 60:
                            should_trade = True
                    
                    if should_trade:
                        logging.info("🟢 Market is OPEN - Running trading session")
                        self.run_trading_session()
                        last_trade_time = now
                        logging.info(f"⏰ Next session in {self.trading_interval} minutes")
                        time.sleep(self.trading_interval * 60)
                    else:
                        time.sleep(60)
                else:
                    logging.info("🔴 Market is CLOSED")
                    next_open = self.get_next_market_open()
                    hours_until = (next_open - now).total_seconds() / 3600
                    logging.info(f"⏰ Next market open: {next_open.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                    logging.info(f"⏰ Time until open: {hours_until:.1f} hours")
                    self.sleep_until_next_event()
                    
            except KeyboardInterrupt:
                logging.info("🛑 Trading daemon stopped by user")
                break
            except Exception as e:
                logging.error(f"❌ Daemon error: {e}")
                logging.error(traceback.format_exc())
                time.sleep(60)

if __name__ == "__main__":
    daemon = TradingDaemon()
    daemon.run()
