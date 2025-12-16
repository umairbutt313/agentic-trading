#!/usr/bin/env python3
"""
Price Storage System
Thread-safe storage and retrieval of real-time stock price data
"""

import json
import sqlite3
import threading
import logging
import fcntl
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union
import asyncio
from contextlib import asynccontextmanager, contextmanager

logger = logging.getLogger('PriceStorage')

class PriceStorage:
    """Thread-safe and process-safe price data storage with JSON and SQLite backends"""

    def __init__(self, storage_type: str = 'json', data_dir: str = 'container_output/realtime_data'):
        self.storage_type = storage_type.lower()

        # Convert to absolute path to prevent working directory issues
        self.data_dir = Path(data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Thread safety (within process)
        self.lock = threading.RLock()

        # Process safety (across processes) - lock file
        self.lock_file = self.data_dir / 'price_storage.lock'

        # Storage paths (absolute)
        self.json_file = self.data_dir / 'price_history.json'
        self.db_file = self.data_dir / 'price_history.db'
        
        # Data retention settings
        self.max_age_hours = 48  # Keep last 48 hours of data
        self.cleanup_interval = 3600  # Clean up every hour
        self.last_cleanup = datetime.now()
        
        # Initialize storage
        self._initialize_storage()
        
        logger.info(f"📁 Price storage initialized: {storage_type} backend in {data_dir}")

    @contextmanager
    def _file_lock(self, timeout=5.0):
        """
        Cross-process file locking using fcntl
        Prevents race conditions when multiple processes access the same file
        """
        lock_fd = None
        acquired = False
        start_time = time.time()

        try:
            # Create lock file if it doesn't exist
            lock_fd = open(self.lock_file, 'w')

            # Try to acquire lock with timeout
            while time.time() - start_time < timeout:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except IOError:
                    time.sleep(0.01)  # Wait 10ms before retry

            if not acquired:
                raise TimeoutError(f"Could not acquire file lock within {timeout}s")

            yield  # Lock acquired, execute protected code

        finally:
            # Release lock
            if lock_fd and acquired:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except:
                    pass
            if lock_fd:
                try:
                    lock_fd.close()
                except:
                    pass

    def _initialize_storage(self):
        """Initialize the storage backend"""
        if self.storage_type == 'sqlite':
            self._initialize_sqlite()
        else:
            self._initialize_json()
    
    def _initialize_sqlite(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_file) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS price_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    INDEX(symbol, timestamp)
                )
            ''')
            conn.commit()
    
    def _initialize_json(self):
        """Initialize JSON file storage"""
        if not self.json_file.exists():
            with open(self.json_file, 'w') as f:
                json.dump({}, f, indent=2)
    
    async def store_price(self, price_data: Dict[str, Union[str, float]]):
        """Store a single price update (async)"""
        await asyncio.get_event_loop().run_in_executor(
            None, self._store_price_sync, price_data
        )
    
    def _store_price_sync(self, price_data: Dict[str, Union[str, float]]):
        """Store a single price update (synchronous) with cross-process locking"""
        with self.lock:  # Thread safety
            with self._file_lock():  # Process safety
                if self.storage_type == 'sqlite':
                    self._store_price_sqlite(price_data)
                else:
                    self._store_price_json(price_data)

                # Periodic cleanup
                if (datetime.now() - self.last_cleanup).seconds > self.cleanup_interval:
                    self._cleanup_old_data()
                    self.last_cleanup = datetime.now()
    
    def _store_price_sqlite(self, price_data: Dict[str, Union[str, float]]):
        """Store price data in SQLite"""
        try:
            with sqlite3.connect(self.db_file) as conn:
                conn.execute(
                    'INSERT INTO price_data (timestamp, symbol, price) VALUES (?, ?, ?)',
                    (price_data['timestamp'], price_data['symbol'], price_data['price'])
                )
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ Failed to store price in SQLite: {e}")
    
    def _store_price_json(self, price_data: Dict[str, Union[str, float]]):
        """Store price data in JSON with validation and recovery"""
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Load existing data with validation
                data = self._load_json_with_recovery()

                symbol = price_data['symbol']
                if symbol not in data:
                    data[symbol] = []

                # Add new price point
                data[symbol].append({
                    'timestamp': price_data['timestamp'],
                    'price': price_data['price']
                })

                # Limit data points per symbol (keep last 2000 points)
                if len(data[symbol]) > 2000:
                    data[symbol] = data[symbol][-2000:]

                # Save updated data with atomic write
                self._save_json_atomically(data)
                return  # Success - exit

            except json.JSONDecodeError as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"❌ Failed to store price after {max_retries} retries: {e}")
                    logger.info("🔄 Triggering recovery and retry...")
                    # Force recovery and try one more time
                    data = self._recover_json()
                    symbol = price_data['symbol']
                    if symbol not in data:
                        data[symbol] = []
                    data[symbol].append({
                        'timestamp': price_data['timestamp'],
                        'price': price_data['price']
                    })
                    self._save_json_atomically(data)
                else:
                    import time
                    time.sleep(0.1 * retry_count)  # Exponential backoff

            except Exception as e:
                logger.error(f"❌ Failed to store price in JSON: {e}")
                break
    
    def _load_json_with_recovery(self) -> Dict:
        """Load JSON with automatic corruption recovery and cross-process locking"""
        try:
            # Note: File lock is acquired by caller (_store_price_sync)
            # No need for double-locking here when called from write path
            with open(self.json_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"⚠️ JSON corruption detected, attempting recovery: {e}")
            return self._recover_json()
    
    def _recover_json(self) -> Dict:
        """Recover from corrupted JSON file"""
        backup_file = self.data_dir / f'{self.json_file.stem}.backup'
        
        # Try to load from backup first
        if backup_file.exists():
            try:
                with open(backup_file, 'r') as f:
                    data = json.load(f)
                logger.info("✅ Recovered from backup file")
                return data
            except json.JSONDecodeError:
                logger.warning("⚠️ Backup file also corrupted")
        
        # Create fresh structure
        logger.info("🔄 Creating fresh JSON structure")
        return {
            "NVDA": [],
            "AAPL": [], 
            "INTC": [],
            "MSFT": [],
            "GOOG": [],
            "TSLA": []
        }
    
    def _save_json_atomically(self, data: Dict):
        """Save JSON atomically to prevent corruption"""
        # Ensure directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Use parent directory for temp file to ensure same filesystem
        temp_file = self.data_dir / f'{self.json_file.stem}.tmp'
        backup_file = self.data_dir / f'{self.json_file.stem}.backup'

        try:
            # Write to temporary file first
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)

            # Validate the temporary file
            with open(temp_file, 'r') as f:
                json.load(f)

            # Create backup of current file
            if self.json_file.exists():
                import shutil
                shutil.copy2(self.json_file, backup_file)

            # Atomically replace the original file
            temp_file.replace(self.json_file)

        except Exception as e:
            # Clean up temporary file on error
            if temp_file.exists():
                temp_file.unlink()
            raise e
    
    def get_price_history(self, symbol: str, hours_back: int = 24) -> List[Dict]:
        """Get price history for a symbol with cross-process locking"""
        with self.lock:  # Thread safety
            with self._file_lock():  # Process safety
                if self.storage_type == 'sqlite':
                    return self._get_price_history_sqlite(symbol, hours_back)
                else:
                    return self._get_price_history_json(symbol, hours_back)
    
    def _get_price_history_sqlite(self, symbol: str, hours_back: int) -> List[Dict]:
        """Get price history from SQLite"""
        try:
            cutoff_time = (datetime.now() - timedelta(hours=hours_back)).isoformat()
            
            with sqlite3.connect(self.db_file) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    'SELECT timestamp, price FROM price_data WHERE symbol = ? AND timestamp >= ? ORDER BY timestamp',
                    (symbol, cutoff_time)
                )
                
                return [{'timestamp': row['timestamp'], 'price': row['price']} 
                       for row in cursor.fetchall()]
                       
        except Exception as e:
            logger.error(f"❌ Failed to get price history from SQLite: {e}")
            return []
    
    def _get_price_history_json(self, symbol: str, hours_back: int) -> List[Dict]:
        """Get price history from JSON"""
        try:
            with open(self.json_file, 'r') as f:
                data = json.load(f)
            
            if symbol not in data:
                return []
            
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            # Filter data by time
            filtered_data = []
            for point in data[symbol]:
                try:
                    point_time = datetime.fromisoformat(point['timestamp'].replace('Z', '+00:00'))
                    if point_time >= cutoff_time:
                        filtered_data.append(point)
                except ValueError:
                    continue
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"❌ Failed to get price history from JSON: {e}")
            return []
    
    def get_latest_price(self, symbol: str) -> Optional[Dict]:
        """Get the most recent price for a symbol"""
        history = self.get_price_history(symbol, hours_back=1)
        return history[-1] if history else None
    
    def get_all_symbols(self) -> List[str]:
        """Get list of all symbols with stored data"""
        with self.lock:
            if self.storage_type == 'sqlite':
                return self._get_all_symbols_sqlite()
            else:
                return self._get_all_symbols_json()
    
    def _get_all_symbols_sqlite(self) -> List[str]:
        """Get all symbols from SQLite"""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.execute('SELECT DISTINCT symbol FROM price_data')
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"❌ Failed to get symbols from SQLite: {e}")
            return []
    
    def _get_all_symbols_json(self) -> List[str]:
        """Get all symbols from JSON"""
        try:
            with open(self.json_file, 'r') as f:
                data = json.load(f)
            return list(data.keys())
        except Exception as e:
            logger.error(f"❌ Failed to get symbols from JSON: {e}")
            return []
    
    def _cleanup_old_data(self):
        """Remove old data beyond retention period"""
        cutoff_time = datetime.now() - timedelta(hours=self.max_age_hours)
        
        if self.storage_type == 'sqlite':
            self._cleanup_sqlite(cutoff_time)
        else:
            self._cleanup_json(cutoff_time)
        
        logger.info(f"🧹 Cleaned up price data older than {self.max_age_hours} hours")
    
    def _cleanup_sqlite(self, cutoff_time: datetime):
        """Clean up old SQLite data"""
        try:
            with sqlite3.connect(self.db_file) as conn:
                conn.execute(
                    'DELETE FROM price_data WHERE timestamp < ?',
                    (cutoff_time.isoformat(),)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"❌ Failed to cleanup SQLite data: {e}")
    
    def _cleanup_json(self, cutoff_time: datetime):
        """Clean up old JSON data"""
        try:
            with open(self.json_file, 'r') as f:
                data = json.load(f)
            
            for symbol in data:
                data[symbol] = [
                    point for point in data[symbol]
                    if datetime.fromisoformat(point['timestamp'].replace('Z', '+00:00')) >= cutoff_time
                ]
            
            with open(self.json_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Failed to cleanup JSON data: {e}")
    
    def get_statistics(self) -> Dict:
        """Get storage statistics"""
        stats = {
            'storage_type': self.storage_type,
            'data_dir': str(self.data_dir),
            'symbols': self.get_all_symbols(),
            'total_symbols': len(self.get_all_symbols())
        }
        
        # Add per-symbol statistics
        symbol_stats = {}
        for symbol in stats['symbols']:
            history = self.get_price_history(symbol, hours_back=24)
            if history:
                prices = [point['price'] for point in history]
                symbol_stats[symbol] = {
                    'data_points': len(history),
                    'latest_price': history[-1]['price'],
                    'latest_time': history[-1]['timestamp'],
                    'price_range': f"{min(prices):.2f} - {max(prices):.2f}"
                }
        
        stats['symbol_details'] = symbol_stats
        return stats
    
    def export_to_csv(self, symbol: str, output_file: Optional[str] = None) -> str:
        """Export symbol data to CSV format"""
        history = self.get_price_history(symbol, hours_back=48)
        
        if not output_file:
            output_file = self.data_dir / f"{symbol.lower()}_price_history.csv"
        
        with open(output_file, 'w') as f:
            f.write("timestamp,symbol,price\n")
            for point in history:
                f.write(f"{point['timestamp']},{symbol},{point['price']}\n")
        
        logger.info(f"📊 Exported {len(history)} price points for {symbol} to {output_file}")
        return str(output_file)


# Convenience functions for easy integration
def store_price(symbol: str, price: float, timestamp: str = None):
    """Convenience function to store a single price"""
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    storage = PriceStorage()
    price_data = {
        'timestamp': timestamp,
        'symbol': symbol,
        'price': price
    }
    
    # Use synchronous version for convenience
    storage._store_price_sync(price_data)

def get_latest_prices(symbols: List[str] = None) -> Dict[str, Dict]:
    """Convenience function to get latest prices for symbols"""
    storage = PriceStorage()
    
    if symbols is None:
        symbols = storage.get_all_symbols()
    
    latest_prices = {}
    for symbol in symbols:
        latest = storage.get_latest_price(symbol)
        if latest:
            latest_prices[symbol] = latest
    
    return latest_prices

def main():
    """CLI interface for price storage management"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Price storage management')
    parser.add_argument('--stats', action='store_true', help='Show storage statistics')
    parser.add_argument('--export', type=str, help='Export symbol data to CSV')
    parser.add_argument('--cleanup', action='store_true', help='Clean up old data')
    parser.add_argument('--test', action='store_true', help='Test storage system')
    args = parser.parse_args()
    
    storage = PriceStorage()
    
    if args.stats:
        stats = storage.get_statistics()
        print(json.dumps(stats, indent=2))
    
    elif args.export:
        csv_file = storage.export_to_csv(args.export.upper())
        print(f"Exported to: {csv_file}")
    
    elif args.cleanup:
        storage._cleanup_old_data()
        print("Cleanup completed")
    
    elif args.test:
        # Test storage operations
        test_data = {
            'timestamp': datetime.now().isoformat(),
            'symbol': 'TEST',
            'price': 123.45
        }
        
        storage._store_price_sync(test_data)
        retrieved = storage.get_latest_price('TEST')
        
        if retrieved and abs(retrieved['price'] - 123.45) < 0.01:
            print("✅ Storage test passed")
        else:
            print("❌ Storage test failed")
    
    else:
        print("Use --help for available options")

if __name__ == "__main__":
    main()