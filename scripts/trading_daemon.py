#!/usr/bin/env python3
"""
Trading Daemon - Runs stock trading bot 24/7
"""
import os
import sys
import time
import logging
import subprocess
import traceback
from datetime import datetime, timedelta
import pytz

sys.path.append('/home/zgx/personal-projects/ai-trade-agent/scripts')

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
        
        logging.info("🤖 Trading Daemon Initialized")
        logging.info("⏰ Stock trading every 30 minutes")
        logging.info("📊 Performance analysis scheduled for 5:00 PM EST daily")
        logging.info("🎓 Model fine-tuning scheduled for 8:00 PM EST daily")
    
    def is_market_open(self):
        """Check if market is currently open."""
        now = datetime.now(self.est)
        
        # Check if weekend
        if now.weekday() >= 5:
            return False
        
        current_time = now.time()
        return self.market_open <= current_time <= self.market_close
    
    def get_next_market_open(self):
        """Get next market open time."""
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
                ['python3', '/home/zgx/personal-projects/ai-trade-agent/scripts/autonomous_agent.py'],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                logging.info("✅ Trading session completed")
            else:
                logging.error(f"❌ Trading session failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logging.error("❌ Trading session timed out")
        except Exception as e:
            logging.error(f"❌ Trading session failed: {e}")
    
    def run_performance_analysis(self):
        """Run performance analysis."""
        try:
            logging.info("🔔 Time for performance analysis!")
            logging.info("======================================================================")
            logging.info("📊 RUNNING PERFORMANCE ANALYSIS")
            logging.info("======================================================================")
            
            result = subprocess.run(
                ['python3', '/home/zgx/personal-projects/ai-trade-agent/scripts/performance_analyzer.py'],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                logging.info("✅ Performance analysis complete")
            else:
                logging.error(f"❌ Performance analysis failed: {result.stderr}")
                
        except Exception as e:
            logging.error(f"❌ Performance analysis failed: {e}")
    
    def run_finetuning(self):
        """Run model fine-tuning."""
        try:
            logging.info("🔔 Time for daily model fine-tuning!")
            logging.info("======================================================================")
            logging.info("🎓 RUNNING MODEL FINE-TUNING")
            logging.info("======================================================================")
            
            result = subprocess.run(
                ['python3', '/home/zgx/personal-projects/ai-trade-agent/scripts/finetune_model.py'],
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode == 0:
                logging.info("✅ Model fine-tuning complete")
            else:
                logging.error(f"❌ Fine-tuning failed: {result.stderr}")
                
        except Exception as e:
            logging.error(f"❌ Fine-tuning failed: {e}")
    
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
                time.sleep(max(sleep_seconds - 60, 0))  # Wake up 1 min early
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
        logging.info("🎓 Learning: 8:00 PM EST (daily)")
        
        last_trade_time = None
        analysis_done_today = False
        finetune_done_today = False
        
        while True:
            try:
                now = datetime.now(self.est)
                current_time = now.time()
                current_date = now.date()
                
                # Reset daily flags
                if last_trade_time and last_trade_time.date() != current_date:
                    analysis_done_today = False
                    finetune_done_today = False
                
                logging.info(f"📅 Current time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                
                # Performance analysis at 5:00 PM
                if not analysis_done_today and current_time >= self.analysis_time:
                    self.run_performance_analysis()
                    analysis_done_today = True
                
                # Fine-tuning at 8:00 PM
                if not finetune_done_today and current_time >= self.finetune_time:
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
