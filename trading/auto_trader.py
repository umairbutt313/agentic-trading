#!/usr/bin/env python3
"""
Automated Trading System based on Sentiment Analysis
Connects your sentiment scores to Capital.com for real trading
"""

import json
import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.capital_trader import CapitalTrader

class SentimentAutoTrader:
    def __init__(self, config_file: str = ".env"):
        """Initialize the automated trading system"""
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.final_score_dir = os.path.join(self.base_dir, "container_output", "final_score")
        
        # Setup logging first
        self._setup_logging()
        
        # Load Capital.com credentials from .env
        self.load_config(config_file)
        
        # Initialize Capital.com trader
        self.trader = CapitalTrader(
            api_key=self.capital_api_key,
            password=self.capital_password,
            email=self.capital_email,
            demo='--live' not in sys.argv  # Use live mode if --live flag is present
        )
        
        # Trading configuration - NVIDIA ONLY
        self.companies = {
            "NVIDIA": "NVDA"
            # "APPLE": "AAPL",      # DISABLED
            # "MICROSOFT": "MSFT",  # DISABLED
            # "GOOGLE": "GOOG",     # DISABLED
            # "AMAZON": "AMZN",     # DISABLED
            # "TESLA": "TSLA",      # DISABLED
            # "INTEL": "INTC"       # DISABLED
        }
        
        # Trading thresholds - Adjusted for more active trading
        self.buy_threshold = 6.0   # Buy when sentiment >= 6.0 (more aggressive)
        self.sell_threshold = 5.0  # Sell when sentiment <= 5.0 (more aggressive)

        # Original conservative thresholds:
        # self.buy_threshold = 7.0   # Buy when sentiment >= 7.0
        # self.sell_threshold = 4.0  # Sell when sentiment <= 4.0
        
        # Log initialization
        logging.info("🤖 Sentiment Auto Trader initialized")
        logging.info(f"📈 Trading companies: {list(self.companies.keys())}")
        logging.info(f"📊 Buy threshold: {self.buy_threshold}, Sell threshold: {self.sell_threshold}")
    
    def _setup_logging(self):
        """Setup logging to save all output to logs directory"""
        logs_dir = os.path.join(self.base_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f"auto_trader_{timestamp}.log")
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ],
            force=True
        )
        
        logging.info(f"📝 Logging initialized. Output will be saved to: {log_file}")
        
    def load_config(self, config_file: str):
        """Load configuration from .env file"""
        env_path = os.path.join(self.base_dir, config_file)
        
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        # Remove inline comments
                        value = value.split('#')[0].strip()
                        
                        if key == 'CAPITAL_API_KEY':
                            self.capital_api_key = value
                        elif key == 'CAPITAL_PASSWORD':
                            self.capital_password = value
                        elif key == 'CAPITAL_EMAIL':
                            self.capital_email = value
        else:
            print("⚠️  Please add CAPITAL_API_KEY and CAPITAL_ACCOUNT_ID to .env file")
            self.capital_api_key = "YOUR_API_KEY"
            self.capital_account_id = "YOUR_ACCOUNT_ID"
    
    def load_sentiment_scores(self) -> Dict:
        """Load latest sentiment scores from your analysis system"""
        try:
            # Load weighted sentiment scores
            weighted_file = os.path.join(self.final_score_dir, "final-weighted-scores.json")
            with open(weighted_file, 'r') as f:
                weighted_data = json.load(f)
            
            # Load news sentiment scores (image sentiment disabled)
            news_data = {}
            news_file = os.path.join(self.final_score_dir, "news-sentiment-analysis.json")
            if os.path.exists(news_file):
                with open(news_file, 'r') as f:
                    news_data = json.load(f)
            
            # Combine all scores (without image sentiment)
            sentiment_data = {}
            
            # Extract companies data from the new structure
            companies_data = weighted_data.get("companies", weighted_data)
            
            for company in self.companies:
                if company in companies_data:
                    company_data = companies_data[company]
                    sentiment_data[company] = {
                        "weighted_score": company_data["final_score"],
                        "news_score": company_data.get("news_score", company_data["final_score"]),
                        "image_score": 5.0,  # Default neutral score since disabled
                        "timestamp": company_data.get("timestamp", weighted_data.get("metadata", {}).get("timestamp", ""))
                    }
            
            return sentiment_data
            
        except Exception as e:
            print(f"❌ Error loading sentiment scores: {e}")
            return {}
    
    def check_trading_hours(self) -> bool:
        """Check if market is open for trading
        - DEMO MODE: 24/7 trading allowed
        - LIVE MODE: NASDAQ hours in Germany time (CET: 15:30 - 22:00)
        """
        from datetime import datetime, timezone, timedelta

        # Demo accounts can trade 24/7
        if self.trader.demo:
            return True

        # LIVE MODE: Check NASDAQ hours in Germany time (CET)
        # Server is GMT (UTC+0), convert to CET (UTC+1)
        # GMT to CET: add 1 hour
        gmt_now = datetime.now()  # Server time in GMT
        cet_now = gmt_now + timedelta(hours=1)  # Convert GMT to CET

        weekday = cet_now.weekday()

        # Market closed on weekends
        if weekday >= 5:  # Saturday = 5, Sunday = 6
            return False

        # NASDAQ HOURS IN GERMANY TIME (CET): 15:30 - 22:00
        hour = cet_now.hour
        minute = cet_now.minute

        print(f"📅 Server GMT: {gmt_now.strftime('%H:%M:%S')} | CET (Germany): {cet_now.strftime('%Y-%m-%d %H:%M:%S')}")

        # NASDAQ Trading Hours in CET: 15:30 (3:30 PM) - 22:00 (10:00 PM)
        if hour >= 16 and hour < 22:  # 16:00 (4:00 PM) - 21:59 (9:59 PM) CET
            return True
        elif hour == 15 and minute >= 30:  # 15:30 (3:30 PM) or later
            return True
        else:  # Market closed (before 15:30 or after 22:00 CET)
            return False

        return False
    
    def execute_trading_cycle(self):
        """Execute one trading cycle based on current sentiment"""
        logging.info(f"\n{'='*60}")
        logging.info(f"🤖 TRADING CYCLE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"{'='*60}")

        # Check if market is open
        if not self.check_trading_hours():
            logging.info("🔒 Market is closed. Waiting for market hours...")
            return

        # Load latest sentiment scores
        sentiment_data = self.load_sentiment_scores()

        if not sentiment_data:
            logging.warning("⚠️  No sentiment data available. Run sentiment analysis first.")
            return

        # Execute trades for each company
        for company, symbol in self.companies.items():
            if company in sentiment_data:
                data = sentiment_data[company]

                logging.info(f"\n📊 {company} ({symbol}):")
                logging.info(f"   Weighted Score: {data['weighted_score']:.1f}/10")
                logging.info(f"   News Sentiment: {data['news_score']:.1f}/10")
                logging.info(f"   Chart Sentiment: {data['image_score']:.1f}/10")
                
                # Execute trade based on sentiment
                self.trader.execute_sentiment_trade(
                    symbol=symbol,
                    sentiment_score=data['weighted_score'],
                    news_sentiment=data['news_score'],
                    image_sentiment=data['image_score']
                )
            else:
                logging.info(f"\n📊 {company} ({symbol}):")
                logging.warning(f"   ⚠️  No sentiment data available - skipping")

        # Show portfolio status
        logging.info(f"\n📈 PORTFOLIO STATUS:")
        report = self.trader.monitor_positions()

        if report:
            # Safe formatting for complex balance objects
            balance = report.get('balance', 0)
            available = report.get('available', 0)
            profit_loss = report.get('profit_loss', 0)

            # Handle complex balance objects
            if isinstance(balance, dict):
                balance = balance.get('balance', 0)
            if isinstance(available, dict):
                available = available.get('available', 0)
            if isinstance(profit_loss, dict):
                profit_loss = profit_loss.get('profit_loss', 0)

            logging.info(f"   Balance: €{float(balance or 0):.2f}")
            logging.info(f"   Available: €{float(available or 0):.2f}")
            logging.info(f"   P&L: €{float(profit_loss or 0):.2f}")

            if report.get('positions'):
                logging.info(f"\n   Open Positions:")
                for pos in report['positions']:
                    logging.info(f"   - {pos.get('symbol', 'N/A')}: {pos.get('size', 0)} @ {pos.get('open_price', 0)}")
                    logging.info(f"     Current: {pos.get('current_price', 0)} | P&L: €{float(pos.get('profit_loss', 0)):.2f}")
        else:
            logging.info("   📭 No portfolio data available")
    
    def run_continuous_trading(self, interval_minutes: int = 5):
        """Run continuous trading with specified interval"""
        logging.info(f"🚀 Starting Automated Sentiment-Based Trading")
        logging.info(f"   Interval: {interval_minutes} minutes")
        logging.info(f"   Buy Threshold: {self.buy_threshold}/10")
        logging.info(f"   Sell Threshold: {self.sell_threshold}/10")
        logging.info(f"   Mode: {'DEMO' if self.trader.demo else 'LIVE'}")
        logging.info(f"   Trading Stocks: {', '.join(self.companies.values())}")
        logging.info(f"\n⚠️  Press Ctrl+C to stop trading\n")

        try:
            while True:
                self.execute_trading_cycle()

                # Wait for next cycle
                logging.info(f"\n⏰ Next cycle in {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            logging.info("\n\n🛑 Trading stopped by user")

            # Final portfolio report
            logging.info("\n📊 FINAL PORTFOLIO REPORT:")
            report = self.trader.monitor_positions()

            if report:
                # Handle complex balance objects (same as in execute_trading_cycle)
                balance = report.get('balance', 0)
                profit_loss = report.get('profit_loss', 0)

                if isinstance(balance, dict):
                    balance = balance.get('balance', 0)
                if isinstance(profit_loss, dict):
                    profit_loss = profit_loss.get('profit_loss', 0)

                logging.info(f"   Final Balance: €{float(balance or 0):.2f}")
                logging.info(f"   Total P&L: €{float(profit_loss or 0):.2f}")

                if report.get('positions'):
                    logging.info(f"   Open Positions: {len(report['positions'])}")


if __name__ == "__main__":
    # Safety check for live trading mode
    if '--live' in sys.argv:
        print("⚠️  WARNING: LIVE TRADING MODE WITH REAL MONEY!")
        print(f"   Account: awesome313@gmail.com")
        print(f"   Current balance: Check your Capital.com account")
        print(f"   Trading: NVIDIA sentiment-based automation")
        response = input("Are you sure you want to trade with REAL MONEY? (yes/no): ")
        if response.lower() != 'yes':
            print("❌ Live trading cancelled. Use without --live for demo mode.")
            sys.exit(0)
        print("✅ Live trading mode confirmed. Starting with real money...")
    
    # Create auto trader
    auto_trader = SentimentAutoTrader()

    # Run continuous trading
    # Options: 0.5 (30 sec), 1 (1 min), 5 (5 min), 15 (15 min), 40 (40 min), 60 (1 hr)
    # Changed from 0.5 (30-sec) to 40 minutes - spread economics require larger moves
    auto_trader.run_continuous_trading(interval_minutes=40)  # 40 minutes for viable spread economics