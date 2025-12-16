#!/usr/bin/env python3
"""
Trading-Sentiment Correlation Verifier
Monitors both sentiment changes and trading actions to verify correlation
ENHANCED VERSION: Added comprehensive logging for review and analysis
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List
import csv
import logging

class TradingCorrelationMonitor:
    def __init__(self):
        self.base_dir = "/root/arslan-chart/agentic-trading-dec2025/stocks"
        self.sentiment_file = os.path.join(self.base_dir, "container_output/final_score/final-weighted-scores.json")
        
        # ENHANCED: All logs in /logs folder for better organization
        os.makedirs(os.path.join(self.base_dir, "logs"), exist_ok=True)
        self.log_file = os.path.join(self.base_dir, "logs/trading_correlation.csv")  # MOVED to logs folder
        self.detailed_log_file = os.path.join(self.base_dir, "logs/detailed_correlation.log")
        self.session_log_file = os.path.join(self.base_dir, "logs/correlation_sessions.log")
        
        # Track previous state
        self.last_sentiment = None
        self.last_check_time = None
        
        # ENHANCED: Session tracking (MOVED before setup_enhanced_logging)
        self.session_start = datetime.now()
        self.events_logged = 0
        
        # ENHANCED: Setup comprehensive logging (MOVED after session_start is defined)
        self.setup_enhanced_logging()
        
        # Initialize log file
        self.init_log_file()
        
    # ENHANCED: Setup comprehensive logging system
    def setup_enhanced_logging(self):
        """Setup detailed logging configuration for correlation monitoring"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.base_dir, "logs", "correlation_monitor.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Enhanced correlation monitoring initialized - Session: {self.session_start.strftime('%Y%m%d_%H%M%S')}")
    
    def init_log_file(self):
        """Initialize CSV log file for correlation tracking"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'nvda_sentiment', 'sentiment_change', 
                    'expected_action', 'actual_action', 'correlation_verified'
                ])
            print(f"📊 Created correlation log: {self.log_file}")
        
        # ENHANCED: Initialize detailed log file
        self.logger.info(f"CSV log file ready: {self.log_file}")
        self.logger.info(f"Detailed log file: {self.detailed_log_file}")
        self.logger.info(f"Session summary file: {self.session_log_file}")
    
    def get_current_sentiment(self) -> Dict:
        """Get current sentiment from file"""
        try:
            if not os.path.exists(self.sentiment_file):
                return None
                
            with open(self.sentiment_file, 'r') as f:
                data = json.load(f)
            
            nvda_data = data.get('companies', {}).get('NVIDIA', {})
            main_timestamp = data.get('metadata', {}).get('timestamp', '')
            
            # ENHANCED: Calculate data age for freshness tracking
            try:
                if main_timestamp:
                    data_time = datetime.fromisoformat(main_timestamp.replace('Z', ''))
                    data_age_seconds = (datetime.now() - data_time).total_seconds()
                else:
                    data_age_seconds = float('inf')
            except:
                data_age_seconds = float('inf')
            
            return {
                'score': nvda_data.get('final_score', 5.0),
                'timestamp': nvda_data.get('timestamp', ''),
                'news_score': nvda_data.get('news_score', 5.0),
                # ENHANCED: Additional metadata for comprehensive logging
                'main_timestamp': main_timestamp,
                'data_age_seconds': data_age_seconds,
                'is_fresh': data_age_seconds <= 900,  # 15 minutes
                'calculation_method': nvda_data.get('calculation_method', 'Unknown')
            }
        except Exception as e:
            print(f"❌ Error reading sentiment: {e}")
            # ENHANCED: Log errors to file as well
            self.logger.error(f"Error reading sentiment data: {e}")
            return None
    
    def determine_expected_action(self, sentiment_score: float) -> str:
        """Determine what trading action should happen based on sentiment"""
        if sentiment_score >= 7.0:
            return "BUY"
        elif sentiment_score <= 4.0:
            return "SELL/SHORT"
        else:
            return "HOLD"
    
    def monitor_correlation(self, duration_minutes: int = 60):
        """Monitor sentiment-trading correlation for specified duration"""
        session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        print("🔍 ENHANCED CORRELATION MONITORING WITH COMPREHENSIVE LOGGING")
        print(f"   Duration: {duration_minutes} minutes")
        print(f"   Session ID: {session_id}")
        print(f"   📄 Log Files:")
        print(f"      • CSV Log: {self.log_file}")
        print(f"      • Detailed Log: {self.detailed_log_file}")
        print(f"      • Session Summary: {self.session_log_file}")
        print(f"   📁 Monitoring: {self.sentiment_file}")
        print("\n📋 What to watch for:")
        print("   • Sentiment >= 7.0 should trigger BUY signals")
        print("   • Sentiment <= 4.0 should trigger SELL/SHORT signals")
        print("   • Sentiment 4.1-6.9 should trigger HOLD")
        print("   • Data freshness warnings (>15 minutes = STALE)")
        print("\n⚠️  Please run your trading system in another terminal!")
        print("   Command: python3 auto_trader.py")
        
        end_time = time.time() + (duration_minutes * 60)
        check_interval = 30  # Check every 30 seconds
        
        while time.time() < end_time:
            current_sentiment = self.get_current_sentiment()
            
            if current_sentiment:
                # Check if sentiment changed significantly
                if self.last_sentiment is None:
                    self.last_sentiment = current_sentiment
                else:
                    score_change = abs(current_sentiment['score'] - self.last_sentiment['score'])
                    
                    if score_change >= 0.5:  # Significant change
                        expected_action = self.determine_expected_action(current_sentiment['score'])
                        
                        print(f"\n🚨 SENTIMENT CHANGE DETECTED!")
                        print(f"   Time: {datetime.now().strftime('%H:%M:%S')}")
                        print(f"   NVDA Score: {self.last_sentiment['score']:.1f} → {current_sentiment['score']:.1f}")
                        print(f"   Change: {score_change:+.1f}")
                        print(f"   Expected Action: {expected_action}")
                        # ENHANCED: Show data freshness
                        print(f"   Data Age: {current_sentiment['data_age_seconds']:.1f} seconds ({'FRESH' if current_sentiment['is_fresh'] else 'STALE'})")
                        
                        # Log the event (original CSV logging preserved)
                        with open(self.log_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                datetime.now().isoformat(),
                                current_sentiment['score'],
                                f"{score_change:+.1f}",
                                expected_action,
                                "MANUAL_VERIFY",  # User needs to verify actual action
                                "PENDING"
                            ])
                        
                        # ENHANCED: Detailed logging to file
                        self.log_detailed_event(current_sentiment, expected_action, score_change)
                        
                        print(f"   📝 Event #{self.events_logged} logged to correlation file and detailed log")
                        print(f"   👀 WATCH YOUR TRADING TERMINAL - Did it execute '{expected_action}'?")
                        
                        self.last_sentiment = current_sentiment
                        
            time.sleep(check_interval)
        
        print(f"\n✅ Monitoring completed. Check log: {self.log_file}")
        # ENHANCED: Create comprehensive session summary
        self.create_session_summary()
        self.show_correlation_summary()
    
    # ENHANCED: Detailed logging function
    def log_detailed_event(self, sentiment_data: Dict, expected_action: str, sentiment_change: float):
        """Log detailed event information to comprehensive log file"""
        self.events_logged += 1
        timestamp = datetime.now()
        session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        
        # Write detailed event log
        with open(self.detailed_log_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"CORRELATION EVENT #{self.events_logged} - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Session ID: {session_id}\n")
            f.write(f"NVIDIA Sentiment: {sentiment_data['score']}/10\n")
            f.write(f"News Score: {sentiment_data['news_score']}/10\n")
            f.write(f"Sentiment Change: {sentiment_change:+.1f}\n")
            f.write(f"Expected Action: {expected_action}\n")
            f.write(f"Data Age: {sentiment_data['data_age_seconds']:.1f} seconds\n")
            f.write(f"Data Freshness: {'FRESH' if sentiment_data['is_fresh'] else 'STALE'}\n")
            f.write(f"Calculation Method: {sentiment_data['calculation_method']}\n")
            f.write(f"Raw Timestamp: {sentiment_data['main_timestamp']}\n")
            f.write(f"Market Status: {'OPEN' if datetime.now().hour >= 4 and datetime.now().hour < 20 else 'CLOSED'}\n")
            f.write(f"Weekend: {'Yes' if datetime.now().weekday() >= 5 else 'No'}\n")
        
        # Log to structured logger as well
        self.logger.info(f"Event #{self.events_logged}: NVDA {sentiment_data['score']}/10 -> {expected_action} (Change: {sentiment_change:+.1f})")
        
    # ENHANCED: Session summary function
    def create_session_summary(self):
        """Create comprehensive session summary"""
        session_duration = datetime.now() - self.session_start
        session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        
        # Read events from CSV for this session (approximate by timestamp)
        session_events = []
        try:
            with open(self.log_file, 'r') as f:
                reader = csv.DictReader(f)
                session_start_str = self.session_start.strftime('%Y-%m-%d')
                session_events = [
                    row for row in reader 
                    if session_start_str in row['timestamp']
                ]
        except Exception as e:
            self.logger.error(f"Error reading session events: {e}")
        
        # Write session summary
        with open(self.session_log_file, 'a') as f:
            f.write(f"\n{'='*100}\n")
            f.write(f"CORRELATION MONITORING SESSION SUMMARY\n")
            f.write(f"{'='*100}\n")
            f.write(f"Session ID: {session_id}\n")
            f.write(f"Start Time: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {session_duration}\n")
            f.write(f"Total Events Logged: {self.events_logged}\n")
            
            if session_events:
                buy_signals = len([e for e in session_events if e['expected_action'] == 'BUY'])
                sell_signals = len([e for e in session_events if 'SELL' in e['expected_action']])
                hold_signals = len([e for e in session_events if e['expected_action'] == 'HOLD'])
                
                f.write(f"\nSignal Distribution:\n")
                f.write(f"  BUY Signals:  {buy_signals} ({buy_signals/len(session_events)*100:.1f}%)\n")
                f.write(f"  SELL Signals: {sell_signals} ({sell_signals/len(session_events)*100:.1f}%)\n")
                f.write(f"  HOLD Signals: {hold_signals} ({hold_signals/len(session_events)*100:.1f}%)\n")
                
                sentiments = [float(e['nvda_sentiment']) for e in session_events]
                f.write(f"\nSentiment Statistics:\n")
                f.write(f"  Range: {min(sentiments):.1f} - {max(sentiments):.1f}\n")
                f.write(f"  Average: {sum(sentiments)/len(sentiments):.1f}\n")
                
                f.write(f"\nEvent Timeline:\n")
                for i, event in enumerate(session_events, 1):
                    timestamp = event['timestamp'].split('T')[1].split('.')[0]
                    f.write(f"  {i:2d}. {timestamp} | NVDA: {event['nvda_sentiment']:>3} | {event['expected_action']:>4}\n")
            
            f.write(f"\nLog Files Generated:\n")
            f.write(f"  CSV Log: {self.log_file}\n")
            f.write(f"  Detailed Log: {self.detailed_log_file}\n")
            f.write(f"  Session Summary: {self.session_log_file}\n")
            f.write(f"  Monitor Log: {os.path.join(self.base_dir, 'logs', 'correlation_monitor.log')}\n")
        
        # Console summary
        print(f"\n📋 SESSION SUMMARY CREATED:")
        print(f"   Session ID: {session_id}")
        print(f"   Events Logged: {self.events_logged}")
        print(f"   Duration: {session_duration}")
        print(f"   📄 Detailed logs available in /logs/ directory")
    
    def show_correlation_summary(self):
        """Show summary of correlation events"""
        try:
            with open(self.log_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                events = list(reader)
            
            if events:
                print(f"\n📊 CORRELATION SUMMARY ({len(events)} events):")
                print("-" * 60)
                for event in events[-5:]:  # Show last 5 events
                    timestamp, sentiment, change, expected, actual, verified = event
                    print(f"   {timestamp[:19]} | Sentiment: {sentiment} | Expected: {expected}")
                print("-" * 60)
            else:
                print("📊 No correlation events recorded")
                
        except Exception as e:
            print(f"❌ Error reading log: {e}")

if __name__ == "__main__":
    monitor = TradingCorrelationMonitor()
    print("🔍 How long do you want to monitor? (default: 30 minutes)")
    try:
        duration = int(input("Enter duration in minutes: ") or 30)
    except:
        duration = 30
    
    monitor.monitor_correlation(duration)