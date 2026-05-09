#!/usr/bin/env python3
"""
Options Trading Daemon - Runs options bot 24/7
"""
import json
import os
import signal
import sys
import time
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import pytz

_SCRIPTS_DIR = Path(__file__).resolve().parent
_HEARTBEAT_FILE = _SCRIPTS_DIR.parent / 'logs' / 'heartbeat_options.json'
sys.path.append(str(_SCRIPTS_DIR))

import inference_client
from preflight_check import run_preflight


def _write_heartbeat(status: str, market_open: bool) -> None:
    """Update heartbeat file so health_server.py can report daemon liveness."""
    try:
        _HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
        _HEARTBEAT_FILE.write_text(json.dumps({
            'daemon': 'options',
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

class OptionsDaemon:
    def __init__(self):
        self.est = pytz.timezone('US/Eastern')
        self.market_open = datetime.strptime('09:30', '%H:%M').time()
        self.market_close = datetime.strptime('16:00', '%H:%M').time()
        self.trading_interval = 60  # 60 minutes
        self.analysis_time = datetime.strptime('17:30', '%H:%M').time()  # 5:30 PM
        self.finetune_time = datetime.strptime('02:00', '%H:%M').time()  # 2:00 AM
        self.weekend_strategist_time = datetime.strptime('10:00', '%H:%M').time()  # Sat 10:00 AM

        self._finetune_requested = False
        self.trading_client = None  # set by _run_preflight; used for clock API
        signal.signal(signal.SIGUSR1, self._handle_finetune_signal)

        logging.info("🤖 Options Trading Daemon Initialized")
        logging.info("⏰ Options trading every 60 minutes")
        logging.info("📊 Performance analysis scheduled for 5:30 PM EST daily")
        logging.info("🔍 Market research + 🎓 Fine-tuning scheduled for 2:00 AM EST daily")
        logging.info("🏖️  Weekend deep analysis scheduled for Saturday 10:00 AM EST")
        logging.info("📬 Send SIGUSR1 to trigger an off-cycle fine-tune: systemctl kill -s SIGUSR1 ai-options-bot.service")

        # Preflight: validate Alpaca account + options approval level
        self._run_preflight()

        # NOTE: do NOT preload the model here. Every options session runs as a
        # subprocess (options_agent.py) with its own Python interpreter and its
        # own _MODEL_CACHE — so a model loaded in the daemon process is never shared
        # with or used by those subprocesses. Holding 45 GB in the daemon process
        # just steals memory from the nightly fine-tune subprocess.
    
    def _run_preflight(self) -> None:
        """Validate the Alpaca account including options approval level."""
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
            result = run_preflight(client, require_options=True)
            if not result.ok:
                for issue in result.issues:
                    logging.critical(f"⛔ PREFLIGHT FAILED: {issue}")
                logging.critical(
                    "Daemon will continue but options trading may be blocked. "
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
    
    def run_options_trading(self):
        """Run options trading session."""
        try:
            logging.info("======================================================================")
            logging.info("💰 RUNNING OPTIONS TRADING SESSION")
            logging.info("======================================================================")
            
            result = subprocess.run(
                [sys.executable, str(_SCRIPTS_DIR / 'agents' / 'options_agent.py')],
                timeout=2700
            )
            
            if result.returncode == 0:
                logging.info("✅ Options trading session completed successfully")
            else:
                logging.error(f"❌ Options trading session failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logging.error("❌ Options trading session timed out")
        except Exception as e:
            logging.error(f"❌ Options trading session failed: {e}")
    
    def run_outcome_tracker(self) -> None:
        """Fetch Alpaca fill prices and reconcile closed options positions into outcomes log."""
        try:
            logging.info("📥 Fetching options fill prices and computing P&L...")
            result = subprocess.run(
                [sys.executable, str(_SCRIPTS_DIR / 'analysis' / 'options_outcome_tracker.py')],
                timeout=120
            )
            if result.returncode != 0:
                logging.warning(f"⚠️ Options outcome tracker exited with code: {result.returncode}")
        except Exception as e:
            logging.warning(f"⚠️ Options outcome tracker failed: {e}")

    def run_performance_analysis(self):
        """Run outcome tracking then options performance analysis."""
        # Step 1: Reconcile fills and compute realized P&L
        self.run_outcome_tracker()

        # Step 2: Run performance analysis (reads from outcomes log)
        try:
            logging.info("🔔 Time for options performance analysis!")
            logging.info("======================================================================")
            logging.info("📊 RUNNING OPTIONS PERFORMANCE ANALYSIS")
            logging.info("======================================================================")

            result = subprocess.run(
                [sys.executable, str(_SCRIPTS_DIR / 'analysis' / 'options_performance_analyzer.py')],
                timeout=1200
            )

            if result.returncode == 0:
                logging.info("✅ Options performance analysis complete")
            else:
                logging.error(f"❌ Options performance analysis failed with code: {result.returncode}")

        except Exception as e:
            logging.error(f"❌ Options performance analysis failed: {e}")

        # Step 3: Live-trading readiness check (only meaningful in paper mode)
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
                alert_live_trading_ready('options-bot', m, equity, paper_days, unmet)
            except Exception as e:
                logging.warning(f"⚠️ Live readiness check failed: {e}")

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
    
    def run_market_research(self):
        """Run nightly market research before fine-tuning."""
        try:
            logging.info("🔍 Running nightly options market research...")
            result = subprocess.run(
                [sys.executable, str(_SCRIPTS_DIR / 'data' / 'market_researcher.py'), '--bot', 'options'],
                timeout=600
            )
            if result.returncode == 0:
                logging.info("✅ Market research complete")
            else:
                logging.warning(f"⚠️ Market research exited with code: {result.returncode}")
        except Exception as e:
            logging.warning(f"⚠️ Market research failed (continuing to fine-tune): {e}")

    def run_weekend_strategist(self):
        """Run deep weekend options analysis (Saturday only)."""
        try:
            logging.info("======================================================================")
            logging.info("🏖️  RUNNING OPTIONS WEEKEND ANALYSIS")
            logging.info("======================================================================")
            result = subprocess.run(
                [sys.executable, str(_SCRIPTS_DIR / 'analysis' / 'options_weekend_strategist.py')],
                timeout=1800  # 30 min max
            )
            if result.returncode == 0:
                logging.info("✅ Options weekend analysis complete")
            else:
                logging.error(f"❌ Options weekend analysis failed with code: {result.returncode}")
        except Exception as e:
            logging.error(f"❌ Options weekend analysis failed: {e}")

    def run_training_data_builder(self):
        """Convert today's options decisions + outcomes into labelled training examples."""
        try:
            logging.info("🏗️  Building options training examples from live decisions...")
            result = subprocess.run(
                [sys.executable, str(_SCRIPTS_DIR / 'training' / 'training_data_builder.py'), '--bot', 'options'],
                timeout=300
            )
            if result.returncode == 0:
                logging.info("✅ Options training data builder complete")
            else:
                logging.warning(f"⚠️  Options training data builder exited with code: {result.returncode}")
        except Exception as e:
            logging.warning(f"⚠️  Options training data builder failed (continuing to fine-tune): {e}")

    def run_finetuning(self):
        """Run market research, build training data, then fine-tune options model."""
        self.run_market_research()
        self.run_training_data_builder()
        # Free GPU VRAM before training job loads the 32B base model
        inference_client.stop_for_finetuning()
        try:
            training_data_path = str(_SCRIPTS_DIR.parent / 'finetune' / 'data' / 'options_training_data.json')

            if not os.path.exists(training_data_path):
                logging.info("🎓 No options training data yet - skipping fine-tuning")
                return

            logging.info("🔔 Time for options model fine-tuning!")
            logging.info("======================================================================")
            logging.info("🎓 RUNNING OPTIONS MODEL FINE-TUNING")
            logging.info("======================================================================")

            result = subprocess.run(
                [
                    sys.executable,
                    str(_SCRIPTS_DIR / 'training' / 'finetune_model.py'),
                    '--data', training_data_path,
                ],
                # No timeout — finetune_model.py manages its own internal timeouts
                # (3600s training + 2400s promotion eval).
            )

            if result.returncode == 0:
                logging.info("✅ Options fine-tuning complete")
            else:
                logging.error(f"❌ Options fine-tuning failed with code: {result.returncode}")

        except subprocess.TimeoutExpired:
            logging.error("❌ Options fine-tuning timed out after 1 hour")
        except Exception as e:
            logging.error(f"❌ Options fine-tuning failed: {e}")
    
    def sleep_until_next_event(self):
        """Sleep until next scheduled event."""
        now = datetime.now(self.est)
        
        # Check for scheduled events today
        events = []
        
        # Market open
        if not self.is_market_open():
            next_open = self.get_next_market_open()
            events.append(('Market open', next_open))
        
        # Weekend strategist (Saturday 10:00 AM only)
        if now.weekday() == 5:
            strategist_dt = now.replace(hour=10, minute=0, second=0, microsecond=0)
            if now < strategist_dt:
                events.append(('Options Weekend Analysis', strategist_dt))

        # Analysis time (5:30 PM)
        analysis_dt = now.replace(hour=17, minute=30, second=0, microsecond=0)
        if now < analysis_dt:
            events.append(('Options Analysis', analysis_dt))
        
        # Fine-tuning time (2:00 AM) - may be next calendar day
        finetune_dt = now.replace(hour=2, minute=0, second=0, microsecond=0)
        if now >= finetune_dt:
            finetune_dt += timedelta(days=1)
        events.append(('Options Fine-tuning', finetune_dt))
        
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
        logging.info("🚀 Starting Options Daemon - Running 24/7")
        logging.info("💰 Options Trading: 9:30 AM - 4:00 PM EST")
        logging.info("📊 Analysis: 5:30 PM EST (daily)")
        logging.info("🔍 Research + 🎓 Learning: 2:00 AM EST (daily)")
        logging.info("🏖️  Weekend analysis: Saturday 10:00 AM EST")

        last_trade_time = None
        # Initialise from wall-clock time so a restart mid-day doesn't
        # immediately re-run analysis or fine-tuning that already ran today.
        _startup_now = datetime.now(self.est)
        _startup = _startup_now.time()
        analysis_done_today = _startup >= self.analysis_time
        finetune_done_today = _startup >= self.finetune_time
        weekend_strategist_done_this_week = False
        last_weekend_week = None
        last_reset_date = _startup_now.date()  # tracks the last date flags were reset
        
        while True:
            try:
                now = datetime.now(self.est)
                current_time = now.time()
                current_date = now.date()
                current_week = now.isocalendar()[1]

                # Off-cycle fine-tune triggered by SIGUSR1
                if self._finetune_requested:
                    self._finetune_requested = False
                    finetune_done_today = False
                    logging.info("🔔 Running off-cycle fine-tune (SIGUSR1)...")
                    self.run_finetuning()
                    finetune_done_today = True

                # Reset daily flags at midnight — use a dedicated date tracker so
                # the reset fires exactly once per calendar day regardless of whether
                # any trades happened (the old last_trade_time approach missed resets
                # on weekends and caused double fine-tunes on market-open mornings).
                if current_date != last_reset_date:
                    analysis_done_today = False
                    finetune_done_today = False
                    last_reset_date = current_date

                # Reset weekly flag on new calendar week
                if last_weekend_week != current_week:
                    weekend_strategist_done_this_week = False

                logging.info(f"📅 Current time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                _write_heartbeat('running', self.is_market_open())
                
                # Weekend deep analysis on Saturday at 10:00 AM
                if (now.weekday() == 5  # Saturday
                        and not weekend_strategist_done_this_week
                        and current_time >= self.weekend_strategist_time):
                    self.run_weekend_strategist()
                    weekend_strategist_done_this_week = True
                    last_weekend_week = current_week

                # Performance analysis at 5:30 PM, followed by online training check
                if not analysis_done_today and current_time >= self.analysis_time:
                    self.run_performance_analysis()
                    self.run_online_training()
                    analysis_done_today = True
                
                # Fine-tuning at 2:00 AM — never run during market hours to avoid
                # OOM from simultaneous inference + training on the same GPU.
                if (not finetune_done_today
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
                        logging.info("🟢 Market is OPEN - Running options trading session")
                        self.run_options_trading()
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
                logging.info("🛑 Options daemon stopped by user")
                break
            except Exception as e:
                logging.error(f"❌ Options daemon error: {e}")
                time.sleep(60)

if __name__ == "__main__":
    daemon = OptionsDaemon()
    daemon.run()
