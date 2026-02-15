#!/usr/bin/env python3
"""
AI Trading Daemon - Runs 24/7 with intelligent scheduling
TRADES CONTINUOUSLY EVERY 30 MIN DURING MARKET HOURS
"""
import os
import sys
import time
import logging
import traceback
from datetime import datetime, time as dt_time
import pytz

# Add project root to path
sys.path.append('/home/zgx/personal-projects/ai-trade-agent/scripts')

# Simple console logging - systemd will capture it
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Import after path and logging are set
from autonomous_agent import AutonomousAgent
from weekend_strategist import WeekendStrategist
from performance_analyzer import PerformanceAnalyzer

class TradingDaemon:
    def __init__(self):
        self.timezone = pytz.timezone('America/New_York')
        self.agent = None  # Initialize lazily on first trading session
        self.weekend_strategist = WeekendStrategist()
        self.analyzer = PerformanceAnalyzer()
        
        # Track last execution times
        self.last_trade_check = None
        self.last_nightly_analysis = None
        self.last_weekend_analysis_sat = None
        self.last_weekend_analysis_sun = None
        
        logging.info("🤖 Trading Daemon Initialized")
    
    def get_current_time(self):
        """Get current time in EST/EDT."""
        from datetime import timezone as tz
        # Get UTC time and convert to EST/EDT
        utc_now = datetime.now(tz.utc)
        est_now = utc_now.astimezone(self.timezone)
        return est_now
    
    def is_market_hours(self):
        """Check if it's market hours (9:30 AM - 4:00 PM EST, Mon-Fri)."""
        now = self.get_current_time()
        
        if now.weekday() >= 5:
            return False
        
        market_open = dt_time(9, 30)
        market_close = dt_time(16, 0)
        current_time = now.time()
        
        return market_open <= current_time <= market_close
    
    def should_run_trading(self):
        """Check if we should run trading logic (every 30 minutes during market hours)."""
        if not self.is_market_hours():
            return False
        
        now = self.get_current_time()
        
        if self.last_trade_check is None:
            return True
        
        time_since_last = (now - self.last_trade_check).total_seconds()
        return time_since_last >= 1800  # 30 minutes
    
    def should_run_nightly_analysis(self):
        """Check if we should run nightly analysis (6 PM on weekdays)."""
        now = self.get_current_time()
        
        if now.weekday() >= 5:
            return False
        
        if now.hour != 18:
            return False
        
        if self.last_nightly_analysis is not None:
            if self.last_nightly_analysis.date() == now.date():
                return False
        
        return True
    
    def should_run_weekend_analysis(self):
        """Check if we should run weekend analysis (Sat 10 AM, Sun 2 PM)."""
        now = self.get_current_time()
        
        if now.weekday() == 5:  # Saturday
            if now.hour == 10:
                if self.last_weekend_analysis_sat is None or \
                   self.last_weekend_analysis_sat.date() != now.date():
                    return True
        
        if now.weekday() == 6:  # Sunday
            if now.hour == 14:
                if self.last_weekend_analysis_sun is None or \
                   self.last_weekend_analysis_sun.date() != now.date():
                    return True
        
        return False
    
    def run_trading_session(self):
        """Execute trading logic."""
        logging.info("="*70)
        logging.info("📊 RUNNING TRADING SESSION")
        logging.info("="*70)
        
        try:
            # Initialize agent on first run (loads model once)
            if self.agent is None:
                logging.info("Initializing autonomous agent (and loading AI model)...")
                self.agent = AutonomousAgent()
            
            # Run trading session
            self.agent.run_trading_session()
            
            self.last_trade_check = self.get_current_time()
            logging.info("✅ Trading session complete")
            logging.info(f"⏰ Next session in 30 minutes")
        
        except Exception as e:
            logging.error(f"❌ Trading session failed: {e}")
            logging.error(traceback.format_exc())
    
    def run_nightly_analysis(self):
        """Execute nightly performance analysis."""
        logging.info("="*70)
        logging.info("🌙 RUNNING NIGHTLY ANALYSIS")
        logging.info("="*70)
        
        try:
            self.analyzer.load_trades(days_back=7)
            performance = self.analyzer.analyze_performance()
            recommendations = self.analyzer.generate_recommendations(performance)
            
            self.analyzer.save_analysis(performance, recommendations)
            self.analyzer.print_report(performance, recommendations)
            self.analyzer.apply_recommendations(recommendations)
            
            self.last_nightly_analysis = self.get_current_time()
            logging.info("✅ Nightly analysis complete")
        
        except Exception as e:
            logging.error(f"❌ Nightly analysis failed: {e}")
            logging.error(traceback.format_exc())
    
    def run_weekend_analysis(self):
        """Execute weekend deep analysis."""
        logging.info("="*70)
        logging.info("🏖️  RUNNING WEEKEND DEEP ANALYSIS")
        logging.info("="*70)
        
        try:
            self.weekend_strategist.run_weekend_analysis()
            
            now = self.get_current_time()
            if now.weekday() == 5:
                self.last_weekend_analysis_sat = now
            else:
                self.last_weekend_analysis_sun = now
            
            logging.info("✅ Weekend analysis complete")
        
        except Exception as e:
            logging.error(f"❌ Weekend analysis failed: {e}")
            logging.error(traceback.format_exc())
    
    def run_forever(self):
        """Main daemon loop - runs 24/7."""
        logging.info("🚀 Starting Trading Daemon - Running 24/7")
        logging.info(f"📅 Current time: {self.get_current_time()}")
        
        check_interval = 60  # Check every minute
        
        while True:
            try:
                now = self.get_current_time()
                
                # Hourly status update
                if now.minute == 0:
                    market_status = "OPEN" if self.is_market_hours() else "CLOSED"
                    logging.info(f"⏰ Hourly check - Market: {market_status} - {now.strftime('%Y-%m-%d %H:%M %Z')}")
                
                # Check what to run
                if self.should_run_trading():
                    self.run_trading_session()
                
                elif self.should_run_nightly_analysis():
                    self.run_nightly_analysis()
                
                elif self.should_run_weekend_analysis():
                    self.run_weekend_analysis()
                
                # Sleep until next check
                time.sleep(check_interval)
            
            except KeyboardInterrupt:
                logging.info("🛑 Daemon stopped by user")
                break
            
            except Exception as e:
                logging.error(f"❌ Daemon error: {e}")
                logging.error(traceback.format_exc())
                logging.info("⚠️  Continuing after error...")
                time.sleep(check_interval)

if __name__ == "__main__":
    daemon = TradingDaemon()
    daemon.run_forever()
