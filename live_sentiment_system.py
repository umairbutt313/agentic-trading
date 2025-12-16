#!/usr/bin/env python3
"""
Live Sentiment & Price System
Real-time data collection without using cached JSON files
"""

# import asyncio  # Not used - removed to fix lint warning
import threading
import time
import subprocess
import logging
import os
from datetime import datetime

# Configure logging - both console and file
log_dir = '/root/arslan-chart/agentic-trading-dec2025/stocks/logs'
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/live_sentiment_system.log'),
        logging.StreamHandler()  # Also show on console
    ]
)
logger = logging.getLogger('LiveSentimentSystem')

class LiveSentimentSystem:
    def __init__(self):
        self.running = False
        self.fresh_news_collected = True  # Start True so first analysis runs fresh
        self.last_news_fetch_time = 0  # Track last news fetch timestamp (Fix: modulo bug)

        # NEW: Create session timestamp ONCE per session to reuse same file
        # This prevents creating new JSON file every 5 minutes (was creating 12 files/hour)
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = f"raw-news-images_{self.session_timestamp}.json"
        logger.info(f"🆕 New session started - file: {self.session_file}")
        
    def fetch_fresh_news(self):
        """Fetch fresh news from NewsAPI"""
        logger.info("🌐 Fetching fresh news from NewsAPI...")
        try:
            # OLD: Created new timestamped file every 5 minutes
            # result = subprocess.run([
            #     'python3', 'news/news_dump.py'
            # ], capture_output=True, text=True, cwd='/root/arslan-chart/agentic-trading-dec2025/stocks')

            # NEW: Pass session filename to reuse same file during entire session
            # Only creates new file when script is restarted (Ctrl+C and run again)
            result = subprocess.run([
                'python3', 'news/news_dump.py', '--session-file', self.session_file
            ], capture_output=True, text=True, cwd='/root/arslan-chart/agentic-trading-dec2025/stocks')

            if result.returncode == 0:
                logger.info("✅ Fresh news data collected")
                self.fresh_news_collected = True  # Mark that we have fresh news to analyze
                return True
            else:
                logger.error(f"❌ News collection failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"❌ Error fetching news: {e}")
            return False
    
    def analyze_sentiment(self):
        """Analyze sentiment from fresh news data"""
        logger.info("🧠 Analyzing sentiment with GPT-4...")
        try:
            # DISABLED: This call was broken - sentiment_analyzer.py requires filename argument
            # First run sentiment analyzer on fresh data
            # result1 = subprocess.run([
            #     'python3', 'news/sentiment_analyzer.py'
            # ], capture_output=True, text=True, cwd='/root/arslan-chart/agentic-trading-dec2025/stocks')
            #
            # if result1.returncode != 0:
            #     logger.error(f"❌ Sentiment analysis failed: {result1.stderr}")
            #     return False

            # Then create weighted scores
            # OLD: --force-refresh caused duplicate JSON files every 10 seconds (should only fetch news every 5 min)
            # result2 = subprocess.run([
            #     'python3', 'news/weighted_sentiment_aggregator.py', '--force-refresh'
            # ], capture_output=True, text=True, cwd='/root/arslan-chart/agentic-trading-dec2025/stocks')

            # DISABLED: --quiet mode skips analysis if file exists (causes stale data)
            # NEW: Use --quiet to analyze existing news without re-fetching (prevents creating 360+ files/hour)
            # result2 = subprocess.run([
            #     'python3', 'news/weighted_sentiment_aggregator.py', '--quiet'
            # ], capture_output=True, text=True, cwd='/root/arslan-chart/agentic-trading-dec2025/stocks')

            # FIX (2025-11-17): Run weighted_sentiment_aggregator which properly calls sentiment_analyzer.py
            # with correct file argument (unlike the broken direct call above).
            # Use --force-refresh when we have fresh news, otherwise --quiet to avoid re-analyzing same data
            if self.fresh_news_collected:
                logger.info("🔥 Fresh news available - running full sentiment analysis with GPT-4")
                result = subprocess.run([
                    'python3', 'news/weighted_sentiment_aggregator.py', '--force-refresh'
                ], capture_output=True, text=True, cwd='/root/arslan-chart/agentic-trading-dec2025/stocks')
                self.fresh_news_collected = False  # Reset flag after analysis
            else:
                # Use --quiet for intermediate cycles (no new news to analyze)
                result = subprocess.run([
                    'python3', 'news/weighted_sentiment_aggregator.py', '--quiet'
                ], capture_output=True, text=True, cwd='/root/arslan-chart/agentic-trading-dec2025/stocks')

            if result.returncode == 0:
                logger.info("✅ Sentiment analysis completed")
                return True
            else:
                logger.error(f"❌ Weighted sentiment failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"❌ Error analyzing sentiment: {e}")
            return False
    
    def update_chart_data(self):
        """Update chart data files with latest sentiment + prices"""
        logger.info("📊 Updating chart data...")
        try:
            logger.info("🔧 Running data_aggregator.py subprocess...")
            result = subprocess.run([
                'python3', 'data_aggregator.py'
            ], capture_output=True, text=True, cwd='/root/arslan-chart/agentic-trading-dec2025/stocks', timeout=30)
            
            logger.info(f"🔍 Subprocess return code: {result.returncode}")
            if result.stdout:
                logger.info(f"📤 Subprocess stdout: {result.stdout[:200]}...")
            if result.stderr:
                logger.warning(f"📤 Subprocess stderr: {result.stderr[:200]}...")
            
            if result.returncode == 0:
                logger.info("✅ Chart data updated successfully")
                return True
            else:
                logger.error(f"❌ Chart data update failed with code {result.returncode}")
                logger.error(f"❌ Full stderr: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error("❌ Chart data update timed out after 30 seconds")
            return False
        except Exception as e:
            logger.error(f"❌ Error updating chart data: {e}")
            return False
    
    def start_price_collection(self):
        """Start real-time price collection in background"""
        logger.info("💰 Starting real-time price collection...")
        try:
            def run_price_collector():
                subprocess.run([
                    'python3', 'start_realtime_prices.py'
                ], cwd='/root/arslan-chart/agentic-trading-dec2025/stocks')
            
            price_thread = threading.Thread(target=run_price_collector, daemon=True)
            price_thread.start()
            logger.info("✅ Real-time price collection started")
        except Exception as e:
            logger.error(f"❌ Error starting price collection: {e}")
    
    def start_web_server(self):
        """Start chart web server in background"""
        logger.info("🌐 Starting chart web server...")
        try:
            def run_web_server():
                subprocess.run([
                    'python3', 'serve_charts.py'
                ], cwd='/root/arslan-chart/agentic-trading-dec2025/stocks')
            
            web_thread = threading.Thread(target=run_web_server, daemon=True)
            web_thread.start()
            logger.info("✅ Chart web server started at http://localhost:8080")
        except Exception as e:
            logger.error(f"❌ Error starting web server: {e}")
    
    def sentiment_loop(self):
        """Main sentiment analysis loop"""
        logger.info("🔄 Starting sentiment analysis loop...")
        cycle_count = 0
        
        while self.running:
            try:
                cycle_count += 1
                current_time = datetime.now()
                logger.info(f"📊 CYCLE #{cycle_count} - {current_time.strftime('%H:%M:%S')}")
                
                # Step 1: Fetch fresh news (every 5 minutes)
                # OLD BUG: if int(time.time()) % 300 == 0:  # Almost never triggers with 10s loop!
                # NEW FIX: Track last fetch time and check if 300 seconds elapsed
                current_time = time.time()
                if current_time - self.last_news_fetch_time >= 300:  # Every 5 minutes
                    logger.info("⏰ 5-minute interval reached - fetching fresh news")
                    if self.fetch_fresh_news():
                        logger.info("✅ Fresh news cycle completed successfully")
                        self.last_news_fetch_time = current_time  # Update last fetch time
                    else:
                        logger.warning("⚠️ Fresh news cycle failed")
                        # Still update timestamp to avoid rapid retries
                        self.last_news_fetch_time = current_time
                
                # Step 2: Analyze sentiment every minute
                logger.info("🧠 Starting sentiment analysis cycle...")
                start_time = time.time()
                
                if self.analyze_sentiment():
                    analysis_time = time.time() - start_time
                    logger.info(f"✅ Sentiment analysis completed in {analysis_time:.2f} seconds")
                    
                    # Step 3: Update chart data
                    chart_start = time.time()
                    if self.update_chart_data():
                        chart_time = time.time() - chart_start
                        logger.info(f"✅ Chart data updated in {chart_time:.2f} seconds")
                    else:
                        logger.warning("⚠️ Chart data update failed")
                else:
                    logger.error("❌ Sentiment analysis failed")
                
                total_cycle_time = time.time() - start_time
                logger.info(f"🏁 CYCLE #{cycle_count} completed in {total_cycle_time:.2f} seconds")
                logger.info("😴 Waiting 10 seconds for next sentiment cycle...")
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"❌ Error in sentiment loop cycle #{cycle_count}: {e}")
                logger.error("🔄 Continuing with next cycle in 10 seconds...")
                time.sleep(10)
    
    def start(self):
        """Start the complete live system"""
        self.running = True
        
        print("🚀 LIVE SENTIMENT & PRICE SYSTEM")
        print("=" * 50)
        print("📰 Fresh News: Every 5 minutes from NewsAPI")
        print("🧠 Sentiment Analysis: Every 10 seconds with GPT-4")
        print("💰 Real-time Prices: Continuous streaming")
        print("📊 Chart Updates: Every 10 seconds")
        print("🌐 Web Dashboard: http://localhost:8080")
        print("=" * 50)
        
        # Start background services
        self.start_price_collection()
        self.start_web_server()
        
        # Initial data fetch
        logger.info("🌟 Performing initial data collection...")
        self.fetch_fresh_news()
        self.analyze_sentiment()
        self.update_chart_data()
        
        # Start main sentiment loop
        try:
            self.sentiment_loop()
        except KeyboardInterrupt:
            logger.info("⛔ Stopping system (Ctrl+C received)")
            self.running = False
        except Exception as e:
            logger.error(f"❌ System error: {e}")
            self.running = False
        
        logger.info("🏁 Live Sentiment System stopped")

def main():
    """Entry point"""
    # Create logs directory
    os.makedirs('/root/arslan-chart/agentic-trading-dec2025/stocks/logs', exist_ok=True)
    
    # Create and start system
    system = LiveSentimentSystem()
    system.start()

if __name__ == "__main__":
    main()