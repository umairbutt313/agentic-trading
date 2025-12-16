#!/usr/bin/env python3
"""
Centralized task scheduler using APScheduler:
- Real-time prices: Every 1 second (market hours)
- News sentiment: Every 15 minutes
- TradingView data: Every 30 minutes
- Chart data generation: Every 60 seconds
- System health checks: Every 5 minutes
- Cleanup tasks: Daily at midnight
"""

import asyncio
import logging
import signal
import sys
import os
from datetime import datetime, time as dt_time
from typing import Dict, Any
import threading
import subprocess
import json

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

# Import our modules
from high_frequency_price_collector import HighFrequencyPriceCollector
from price_storage import PriceStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/production_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ProductionScheduler')

class ProductionScheduler:
    """Centralized production scheduler for all data collection tasks"""
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.price_collector = HighFrequencyPriceCollector(
            symbols=['NVDA', 'AAPL', 'INTC', 'MSFT', 'GOOG', 'TSLA'],
            interval=1
        )
        self.price_storage = PriceStorage()
        self.running = False
        self.health_status = {
            'price_collection': 'stopped',
            'sentiment_analysis': 'idle',
            'tradingview_data': 'idle',
            'chart_generation': 'idle',
            'last_health_check': None
        }
        
    def is_market_hours(self) -> bool:
        """Check if current time is during market hours"""
        now = datetime.now()
        # Skip weekends
        if now.weekday() >= 5:
            return False
        
        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = dt_time(9, 30)
        market_close = dt_time(16, 0)
        current_time = now.time()
        
        return market_open <= current_time <= market_close
    
    async def start_price_collection(self):
        """Start real-time price collection during market hours"""
        if self.is_market_hours() and not self.price_collector.running:
            logger.info("🚀 Starting price collection (market hours)")
            self.health_status['price_collection'] = 'running'
            
            # Start price collection in background thread
            def run_collector():
                asyncio.run(self.price_collector.start_streaming())
            
            price_thread = threading.Thread(target=run_collector, daemon=True)
            price_thread.start()
        elif not self.is_market_hours():
            logger.debug("Outside market hours, price collection not started")
            self.health_status['price_collection'] = 'market_closed'
        else:
            logger.debug("Price collection already running")
    
    async def stop_price_collection(self):
        """Stop price collection after market hours"""
        if not self.is_market_hours() and self.price_collector.running:
            logger.info("⏹️ Stopping price collection (market closed)")
            self.price_collector.running = False
            self.health_status['price_collection'] = 'market_closed'
    
    async def update_news_sentiment(self):
        """Update news sentiment analysis every 15 minutes"""
        logger.info("📰 Starting news sentiment update")
        self.health_status['sentiment_analysis'] = 'running'
        
        try:
            # Run weighted sentiment aggregator (removed --quiet to avoid image sentiment requirement)
            result = subprocess.run([
                'python3', 'news/weighted_sentiment_aggregator.py'
            ], capture_output=True, text=True, cwd='/root/arslan-chart/agentic-trading-dec2025/stocks')
            
            if result.returncode == 0:
                logger.info("✅ News sentiment analysis completed successfully")
                self.health_status['sentiment_analysis'] = 'completed'
            else:
                logger.error(f"❌ News sentiment analysis failed: {result.stderr}")
                self.health_status['sentiment_analysis'] = 'failed'
                
        except Exception as e:
            logger.error(f"❌ Error in news sentiment update: {e}")
            self.health_status['sentiment_analysis'] = 'error'
    
    async def update_tradingview_data(self):
        """Update TradingView data every 30 minutes"""
        logger.info("📊 Starting TradingView data update")
        self.health_status['tradingview_data'] = 'running'
        
        try:
            # Run TradingView scraper
            result = subprocess.run([
                'python3', 'news/playwright_tradingview_scraper.py', '--all'
            ], capture_output=True, text=True, cwd='/root/arslan-chart/agentic-trading-dec2025/stocks')
            
            if result.returncode == 0:
                logger.info("✅ TradingView data update completed")
                
                # Run sentiment analysis on the new data
                result2 = subprocess.run([
                    'python3', 'news/tradingview_sentiment_analyzer.py', '--latest'
                ], capture_output=True, text=True, cwd='/root/arslan-chart/agentic-trading-dec2025/stocks')
                
                if result2.returncode == 0:
                    logger.info("✅ TradingView sentiment analysis completed")
                    self.health_status['tradingview_data'] = 'completed'
                else:
                    logger.error(f"❌ TradingView sentiment analysis failed: {result2.stderr}")
                    self.health_status['tradingview_data'] = 'partial_success'
            else:
                logger.error(f"❌ TradingView data update failed: {result.stderr}")
                self.health_status['tradingview_data'] = 'failed'
                
        except Exception as e:
            logger.error(f"❌ Error in TradingView data update: {e}")
            self.health_status['tradingview_data'] = 'error'
    
    async def generate_chart_data(self):
        """Generate chart data files every 60 seconds"""
        logger.debug("📈 Generating chart data")
        self.health_status['chart_generation'] = 'running'
        
        try:
            # Run data aggregator
            result = subprocess.run([
                'python3', 'data_aggregator.py'
            ], capture_output=True, text=True, cwd='/root/arslan-chart/agentic-trading-dec2025/stocks')
            
            if result.returncode == 0:
                logger.debug("✅ Chart data generation completed")
                self.health_status['chart_generation'] = 'completed'
            else:
                logger.warning(f"⚠️ Chart data generation issues: {result.stderr}")
                self.health_status['chart_generation'] = 'warning'
                
        except Exception as e:
            logger.error(f"❌ Error in chart data generation: {e}")
            self.health_status['chart_generation'] = 'error'
    
    async def health_check(self):
        """Perform system health checks every 5 minutes"""
        logger.debug("🏥 Performing system health check")
        
        # Update health check timestamp
        self.health_status['last_health_check'] = datetime.now().isoformat()
        
        # Check price storage statistics
        try:
            stats = self.price_storage.get_statistics()
            logger.info(f"💾 Storage: {stats['total_records']} records, {stats['file_size_mb']:.1f} MB")
            
            # Check data freshness (within last 10 minutes during market hours)
            if self.is_market_hours() and stats['latest_timestamp']:
                latest = datetime.fromisoformat(stats['latest_timestamp'].replace('Z', '+00:00'))
                now = datetime.now()
                minutes_old = (now - latest).total_seconds() / 60
                
                if minutes_old > 10:
                    logger.warning(f"⚠️ Price data is {minutes_old:.1f} minutes old")
                else:
                    logger.debug(f"✅ Price data is fresh ({minutes_old:.1f} minutes old)")
            
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")
        
        # Log current health status
        logger.info(f"🏥 Health Status: {self.health_status}")
    
    async def daily_cleanup(self):
        """Daily cleanup tasks at midnight"""
        logger.info("🧹 Starting daily cleanup tasks")
        
        try:
            # Cleanup old price data
            await self.price_storage.cleanup_old_data()
            logger.info("✅ Price data cleanup completed")
            
            # Cleanup old log files (keep last 7 days)
            log_cleanup_result = subprocess.run([
                'find', 'logs/', '-name', '*.log', '-mtime', '+7', '-delete'
            ], capture_output=True, text=True, cwd='/root/arslan-chart/agentic-trading-dec2025/stocks')
            
            if log_cleanup_result.returncode == 0:
                logger.info("✅ Log cleanup completed")
            else:
                logger.warning("⚠️ Log cleanup had issues")
                
            # Archive old container output (optional)
            archive_result = subprocess.run([
                'python3', '-c', 
                """
import os
import shutil
from datetime import datetime, timedelta

# Archive files older than 7 days
archive_dir = 'container_output/archive'
os.makedirs(archive_dir, exist_ok=True)

cutoff_date = datetime.now() - timedelta(days=7)
for root, dirs, files in os.walk('container_output'):
    if 'archive' in root:
        continue
    for file in files:
        filepath = os.path.join(root, file)
        if os.path.getmtime(filepath) < cutoff_date.timestamp():
            dest = os.path.join(archive_dir, file)
            shutil.move(filepath, dest)
                """
            ], capture_output=True, text=True, cwd='/root/arslan-chart/agentic-trading-dec2025/stocks')
            
            logger.info("✅ Daily cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Daily cleanup error: {e}")
    
    async def start(self):
        """Start the production scheduler"""
        logger.info("🚀 Starting Production Scheduler")
        logger.info("=" * 60)
        
        # Schedule price collection (market hours only)
        self.scheduler.add_job(
            self.start_price_collection,
            IntervalTrigger(minutes=1),
            id='start_price_collection',
            replace_existing=True
        )
        
        # Schedule price collection stop (after hours)
        self.scheduler.add_job(
            self.stop_price_collection,
            IntervalTrigger(minutes=5),
            id='stop_price_collection', 
            replace_existing=True
        )
        
        # Schedule news sentiment updates (every 15 minutes)
        self.scheduler.add_job(
            self.update_news_sentiment,
            IntervalTrigger(minutes=1),  # 1 minute for active trading
            id='news_sentiment',
            replace_existing=True
        )
        
        # Schedule TradingView data updates (every 30 minutes)
        self.scheduler.add_job(
            self.update_tradingview_data,
            IntervalTrigger(minutes=5),  # 5 minutes for TradingView data
            id='tradingview_data',
            replace_existing=True
        )
        
        # Schedule chart data generation (every 60 seconds during market hours)
        self.scheduler.add_job(
            self.generate_chart_data,
            IntervalTrigger(seconds=60),
            id='chart_generation',
            replace_existing=True
        )
        
        # Schedule health checks (every 5 minutes)
        self.scheduler.add_job(
            self.health_check,
            IntervalTrigger(minutes=5),
            id='health_check',
            replace_existing=True
        )
        
        # Schedule daily cleanup (midnight)
        self.scheduler.add_job(
            self.daily_cleanup,
            CronTrigger(hour=0, minute=0),
            id='daily_cleanup',
            replace_existing=True
        )
        
        # Start scheduler
        self.scheduler.start()
        self.running = True
        
        logger.info("✅ Production Scheduler started with jobs:")
        for job in self.scheduler.get_jobs():
            logger.info(f"  - {job.id}: {job.trigger}")
    
    async def stop(self):
        """Stop the production scheduler"""
        logger.info("⏹️ Stopping Production Scheduler")
        
        self.running = False
        
        # Stop price collector
        self.price_collector.running = False
        
        # Shutdown scheduler
        self.scheduler.shutdown(wait=True)
        
        logger.info("✅ Production Scheduler stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status"""
        jobs_info = []
        for job in self.scheduler.get_jobs():
            jobs_info.append({
                'id': job.id,
                'next_run': job.next_run_time.isoformat() if job.next_run_time else None,
                'trigger': str(job.trigger)
            })
        
        return {
            'running': self.running,
            'market_hours': self.is_market_hours(),
            'health_status': self.health_status,
            'scheduled_jobs': jobs_info,
            'scheduler_state': 'running' if self.scheduler.running else 'stopped'
        }

async def main():
    """Main entry point"""
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Create scheduler
    scheduler = ProductionScheduler()
    
    # Handle signals for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        asyncio.create_task(scheduler.stop())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start scheduler
        await scheduler.start()
        
        logger.info("🎯 Production Scheduler is running")
        logger.info("📊 Monitoring all data collection tasks")
        logger.info("💡 Press Ctrl+C to stop")
        
        # Keep running
        while scheduler.running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Scheduler error: {e}")
    finally:
        await scheduler.stop()

if __name__ == "__main__":
    asyncio.run(main())