#!/usr/bin/env python3
"""
SQLite Position Persistence Layer for Trading System
Fixes orphaned position bug by persisting all position metadata to database

# ==============================================================================
# FIX ATTEMPT [2025-12-15 18:00:00]
# ==============================================================================
# ISSUE: Orphaned positions when system restarts - no local tracking of broker positions
# ISSUE_HASH: position_persistence_sqlite_001
# PREVIOUS ATTEMPTS: None
#
# MATHEMATICAL CHECK:
#   Win Rate: N/A - Infrastructure fix, not strategy change
#   Spread Impact: N/A - No trading logic change
#   Timeframe: N/A - Persistence layer supports all timeframes
#
# LIANG WENFENG REASONING:
#   1. Market Context: System restart causes all in-memory position data loss
#      Broker positions remain open but we have no trailing stops, no take profits
#   2. Signal Interpretation: Need write-ahead logging pattern (PENDING → OPEN → CLOSED)
#      Database must be written BEFORE broker API call (not after)
#   3. Alternative Evaluation:
#      - JSON files: No ACID guarantees, race conditions on concurrent writes
#      - Redis: Requires separate service, overkill for 10-100 positions
#      - SQLite + WAL: ACID guarantees, no dependencies, battle-tested
#   4. Risk Management: WAL mode prevents corruption on crashes
#      PENDING status allows recovery of partial failures
#   5. Reflection: Production trading bots (Freqtrade, Jesse) all use SQLite
#      This is proven architecture for retail trading systems
#
# SOLUTION: SQLite database with write-ahead logging
#   - Create PENDING position BEFORE broker call
#   - Update to OPEN after broker confirms
#   - Update to CLOSED after position exits
#   - Reconcile on startup to recover from crashes
#
# VALIDATION:
#   1. Create PENDING position → crash system → verify in DB
#   2. Open position → restart system → verify position tracked
#   3. Run reconcile_positions() → verify orphaned positions recovered
# ==============================================================================
"""

import sqlite3
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import asdict
from contextlib import contextmanager
from pathlib import Path

# NOTE: Do not import from position_manager to avoid circular import
# Position objects will be passed in as parameters


class PositionDatabaseManager:
    """
    SQLite database manager for position persistence
    Uses write-ahead logging (WAL) mode for crash safety
    """

    def __init__(self, db_path: str = "data/positions.db"):
        """
        Initialize database manager

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path

        # Ensure data directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database schema
        self._init_database()

        logging.info(f"📊 Position database initialized: {db_path}")

    def _init_database(self):
        """Initialize database schema and enable WAL mode"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Enable WAL mode for crash safety and concurrent reads
            cursor.execute("PRAGMA journal_mode=WAL")

            # Create positions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL CHECK(direction IN ('LONG', 'SHORT')),
                    strategy TEXT,
                    quantity REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_time REAL NOT NULL,
                    order_id TEXT,
                    position_id TEXT UNIQUE,
                    exit_price REAL,
                    exit_time REAL,
                    exit_reason TEXT,
                    stop_loss REAL,
                    take_profit REAL,
                    trailing_stop_distance REAL,
                    max_hold_time REAL DEFAULT 14400.0,
                    use_trailing_stop INTEGER DEFAULT 1,
                    entry_spread REAL DEFAULT 0.0,
                    highest_price REAL DEFAULT 0.0,
                    lowest_price REAL DEFAULT 999999.0,
                    unrealized_pnl REAL DEFAULT 0.0,
                    realized_pnl REAL DEFAULT 0.0,
                    status TEXT NOT NULL DEFAULT 'PENDING' CHECK(status IN ('PENDING', 'OPEN', 'CLOSING', 'CLOSED', 'CANCELLED')),
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)

            # Create indexes for fast lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_positions_status
                ON positions(status)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_positions_symbol
                ON positions(symbol, status)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_positions_order_id
                ON positions(order_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_positions_position_id
                ON positions(position_id)
            """)

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        try:
            yield conn
        finally:
            conn.close()

    def create_pending_position(self, position: Any) -> bool:
        """
        Create PENDING position in database BEFORE broker call
        This is the write-ahead pattern to prevent orphaned positions

        Args:
            position: Position object to persist

        Returns:
            True if created successfully, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                now = time.time()

                cursor.execute("""
                    INSERT INTO positions (
                        symbol, direction, strategy, quantity, entry_price, entry_time,
                        position_id, stop_loss, take_profit, trailing_stop_distance,
                        max_hold_time, use_trailing_stop, entry_spread,
                        highest_price, lowest_price, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    position.symbol,
                    position.direction,
                    position.strategy,
                    position.quantity,
                    position.entry_price,
                    position.entry_time,
                    position.position_id,
                    position.stop_loss,
                    position.take_profit,
                    position.trailing_stop_distance,
                    position.max_hold_time,
                    1 if position.use_trailing_stop else 0,
                    position.entry_spread,
                    position.highest_price,
                    position.lowest_price,
                    'PENDING',  # Always create as PENDING first
                    now,
                    now
                ))

                conn.commit()

                logging.debug(f"📝 Created PENDING position in DB: {position.position_id}")
                return True

        except sqlite3.IntegrityError as e:
            logging.error(f"❌ Position already exists in DB: {position.position_id}: {e}")
            return False
        except Exception as e:
            logging.error(f"❌ Failed to create PENDING position: {e}")
            return False

    def update_position_opened(self, position_id: str, order_id: str, status: str = 'OPEN') -> bool:
        """
        Update position to OPEN status after broker confirms

        Args:
            position_id: Unique position identifier
            order_id: Broker order/deal ID
            status: Status to set (default: OPEN)

        Returns:
            True if updated successfully, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    UPDATE positions
                    SET order_id = ?, status = ?, updated_at = ?
                    WHERE position_id = ?
                """, (order_id, status, time.time(), position_id))

                conn.commit()

                if cursor.rowcount == 0:
                    logging.error(f"❌ Position not found for update: {position_id}")
                    return False

                logging.debug(f"✅ Updated position to {status}: {position_id} (order_id: {order_id})")
                return True

        except Exception as e:
            logging.error(f"❌ Failed to update position to OPEN: {e}")
            return False

    def update_position_metrics(self, position_id: str, highest_price: float = None,
                               lowest_price: float = None, unrealized_pnl: float = None) -> bool:
        """
        Update position performance metrics

        Args:
            position_id: Unique position identifier
            highest_price: Highest price reached (optional)
            lowest_price: Lowest price reached (optional)
            unrealized_pnl: Current unrealized P&L (optional)

        Returns:
            True if updated successfully, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                updates = []
                params = []

                if highest_price is not None:
                    updates.append("highest_price = ?")
                    params.append(highest_price)

                if lowest_price is not None:
                    updates.append("lowest_price = ?")
                    params.append(lowest_price)

                if unrealized_pnl is not None:
                    updates.append("unrealized_pnl = ?")
                    params.append(unrealized_pnl)

                if not updates:
                    return True  # Nothing to update

                updates.append("updated_at = ?")
                params.append(time.time())
                params.append(position_id)

                query = f"UPDATE positions SET {', '.join(updates)} WHERE position_id = ?"
                cursor.execute(query, params)

                conn.commit()
                return True

        except Exception as e:
            logging.error(f"❌ Failed to update position metrics: {e}")
            return False

    def update_position_closed(self, position_id: str, exit_price: float,
                              exit_reason: str, realized_pnl: float) -> bool:
        """
        Update position to CLOSED status after exit

        Args:
            position_id: Unique position identifier
            exit_price: Exit price
            exit_reason: Reason for exit (ExitReason enum value)
            realized_pnl: Realized P&L

        Returns:
            True if updated successfully, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                now = time.time()

                cursor.execute("""
                    UPDATE positions
                    SET exit_price = ?, exit_time = ?, exit_reason = ?,
                        realized_pnl = ?, status = 'CLOSED', updated_at = ?
                    WHERE position_id = ?
                """, (exit_price, now, exit_reason, realized_pnl, now, position_id))

                conn.commit()

                if cursor.rowcount == 0:
                    logging.error(f"❌ Position not found for close: {position_id}")
                    return False

                logging.debug(f"✅ Updated position to CLOSED: {position_id}")
                return True

        except Exception as e:
            logging.error(f"❌ Failed to update position to CLOSED: {e}")
            return False

    def get_open_positions(self) -> List[Dict]:
        """
        Get all OPEN positions from database

        Returns:
            List of position dictionaries
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT * FROM positions
                    WHERE status = 'OPEN'
                    ORDER BY created_at DESC
                """)

                rows = cursor.fetchall()
                return [dict(row) for row in rows]

        except Exception as e:
            logging.error(f"❌ Failed to get open positions: {e}")
            return []

    def get_pending_positions(self) -> List[Dict]:
        """
        Get all PENDING positions from database (for recovery)

        Returns:
            List of position dictionaries
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT * FROM positions
                    WHERE status = 'PENDING'
                    ORDER BY created_at DESC
                """)

                rows = cursor.fetchall()
                return [dict(row) for row in rows]

        except Exception as e:
            logging.error(f"❌ Failed to get pending positions: {e}")
            return []

    def get_position_by_id(self, position_id: str) -> Optional[Dict]:
        """
        Get position by position_id

        Args:
            position_id: Unique position identifier

        Returns:
            Position dictionary or None if not found
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT * FROM positions
                    WHERE position_id = ?
                """, (position_id,))

                row = cursor.fetchone()
                return dict(row) if row else None

        except Exception as e:
            logging.error(f"❌ Failed to get position by ID: {e}")
            return None

    def get_position_status(self, position_id: str) -> Optional[str]:
        """
        Get only the status of a position (lightweight query for duplicate close prevention)

        Args:
            position_id: Unique position identifier

        Returns:
            Position status string ('PENDING', 'OPEN', 'CLOSING', 'CLOSED', 'CANCELLED') or None
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT status FROM positions
                    WHERE position_id = ?
                """, (position_id,))

                row = cursor.fetchone()
                return row['status'] if row else None

        except Exception as e:
            logging.error(f"❌ Failed to get position status: {e}")
            return None

    def get_position_by_order_id(self, order_id: str) -> Optional[Dict]:
        """
        Get position by broker order_id

        Args:
            order_id: Broker order/deal ID

        Returns:
            Position dictionary or None if not found
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT * FROM positions
                    WHERE order_id = ?
                """, (order_id,))

                row = cursor.fetchone()
                return dict(row) if row else None

        except Exception as e:
            logging.error(f"❌ Failed to get position by order ID: {e}")
            return None

    def get_all_positions(self, limit: int = 100) -> List[Dict]:
        """
        Get all positions (for reconciliation and reporting)

        Args:
            limit: Maximum number of positions to return

        Returns:
            List of position dictionaries
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT * FROM positions
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))

                rows = cursor.fetchall()
                return [dict(row) for row in rows]

        except Exception as e:
            logging.error(f"❌ Failed to get all positions: {e}")
            return []

    def recover_pending_positions(self) -> List[Dict]:
        """
        Recover PENDING positions on startup
        These are positions that were created but broker call may have failed

        Returns:
            List of PENDING position dictionaries
        """
        pending = self.get_pending_positions()

        if pending:
            logging.warning(f"⚠️ Found {len(pending)} PENDING positions on startup")
            logging.warning("   These positions may need reconciliation with broker")

            for pos in pending:
                logging.warning(f"   - {pos['symbol']} {pos['direction']} @ ${pos['entry_price']:.2f}")
                logging.warning(f"     Position ID: {pos['position_id']}")
                logging.warning(f"     Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(pos['created_at']))}")

        return pending

    def reconcile_with_broker(self, broker_positions: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Reconcile database positions with broker positions

        Args:
            broker_positions: List of positions from broker API

        Returns:
            Tuple of (orphaned_positions, untracked_positions)
            - orphaned_positions: In DB but not on broker (need cleanup)
            - untracked_positions: On broker but not in DB (need tracking)
        """
        try:
            # Get all open positions from database
            db_positions = self.get_open_positions()

            # Extract broker deal IDs
            broker_deal_ids = set()
            for pos in broker_positions:
                if 'position' in pos and 'dealId' in pos['position']:
                    broker_deal_ids.add(pos['position']['dealId'])

            # Extract database order IDs
            db_order_ids = {pos['order_id'] for pos in db_positions if pos.get('order_id')}

            # Find orphaned positions (in DB but not on broker)
            orphaned = []
            for pos in db_positions:
                if pos.get('order_id') and pos['order_id'] not in broker_deal_ids:
                    orphaned.append(pos)

            # Find untracked positions (on broker but not in DB)
            untracked_broker_positions = []
            for pos in broker_positions:
                deal_id = pos.get('position', {}).get('dealId')
                if deal_id and deal_id not in db_order_ids:
                    untracked_broker_positions.append(pos)

            if orphaned:
                logging.warning(f"🔍 Found {len(orphaned)} orphaned positions (in DB but not on broker)")
                for pos in orphaned:
                    logging.warning(f"   - {pos['symbol']} {pos['direction']} (order_id: {pos['order_id']})")

            if untracked_broker_positions:
                logging.warning(f"🔍 Found {len(untracked_broker_positions)} untracked positions (on broker but not in DB)")
                for pos in untracked_broker_positions:
                    deal_id = pos.get('position', {}).get('dealId')
                    symbol = pos.get('market', {}).get('epic', 'UNKNOWN')
                    logging.warning(f"   - {symbol} (dealId: {deal_id})")

            return orphaned, untracked_broker_positions

        except Exception as e:
            logging.error(f"❌ Failed to reconcile with broker: {e}")
            return [], []

    def cancel_pending_position(self, position_id: str, reason: str = "Broker call failed") -> bool:
        """
        Cancel a PENDING position (broker call failed)

        Args:
            position_id: Unique position identifier
            reason: Reason for cancellation

        Returns:
            True if cancelled successfully, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    UPDATE positions
                    SET status = 'CANCELLED', exit_reason = ?, updated_at = ?
                    WHERE position_id = ? AND status = 'PENDING'
                """, (reason, time.time(), position_id))

                conn.commit()

                if cursor.rowcount == 0:
                    logging.error(f"❌ Position not found or not PENDING: {position_id}")
                    return False

                logging.debug(f"✅ Cancelled PENDING position: {position_id}")
                return True

        except Exception as e:
            logging.error(f"❌ Failed to cancel pending position: {e}")
            return False

    def get_statistics(self) -> Dict:
        """
        Get database statistics for monitoring

        Returns:
            Dictionary with position counts by status
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT status, COUNT(*) as count
                    FROM positions
                    GROUP BY status
                """)

                stats = {row['status']: row['count'] for row in cursor.fetchall()}

                cursor.execute("SELECT COUNT(*) as total FROM positions")
                stats['total'] = cursor.fetchone()['total']

                return stats

        except Exception as e:
            logging.error(f"❌ Failed to get statistics: {e}")
            return {}
