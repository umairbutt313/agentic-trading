#!/usr/bin/env python3
"""
Test script for SQLite position persistence layer
Validates database operations and crash recovery

# ==============================================================================
# FIX ATTEMPT [2025-12-15 18:00:00]
# ==============================================================================
# ISSUE: Test suite for position persistence layer
# ISSUE_HASH: position_persistence_sqlite_001
# SOLUTION: Comprehensive test script to validate all database operations
# ==============================================================================
"""

import sys
import time
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from trading.position_database import PositionDatabaseManager
from trading.position_manager import Position, PositionStatus, ExitReason

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def test_database_initialization():
    """Test 1: Database creation and initialization"""
    print("\n" + "="*80)
    print("TEST 1: Database Initialization")
    print("="*80)

    try:
        db = PositionDatabaseManager(db_path="data/test_positions.db")
        stats = db.get_statistics()
        print(f"✅ Database initialized successfully")
        print(f"   Total positions: {stats.get('total', 0)}")
        return True
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False


def test_create_pending_position():
    """Test 2: Create PENDING position (write-ahead pattern)"""
    print("\n" + "="*80)
    print("TEST 2: Create PENDING Position")
    print("="*80)

    try:
        db = PositionDatabaseManager(db_path="data/test_positions.db")

        # Create test position
        position_id = f"NVDA_TEST_{int(time.time() * 1000)}"
        position = Position(
            symbol="NVDA",
            direction="LONG",
            entry_price=150.00,
            quantity=10.0,
            entry_time=time.time(),
            stop_loss=148.00,
            take_profit=153.00,
            strategy="TEST_STRATEGY",
            status=PositionStatus.PENDING,
            position_id=position_id,
            entry_spread=0.10,
            max_hold_time=14400.0
        )

        # Create PENDING in database
        success = db.create_pending_position(position)

        if success:
            print(f"✅ PENDING position created: {position_id}")

            # Verify it's in database
            retrieved = db.get_position_by_id(position_id)
            if retrieved:
                print(f"   ✅ Position verified in database")
                print(f"      Status: {retrieved['status']}")
                print(f"      Symbol: {retrieved['symbol']}")
                print(f"      Entry: ${retrieved['entry_price']:.2f}")
                return True, position_id
            else:
                print(f"   ❌ Position not found in database")
                return False, None
        else:
            print(f"❌ Failed to create PENDING position")
            return False, None

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False, None


def test_update_position_opened(position_id):
    """Test 3: Update PENDING → OPEN"""
    print("\n" + "="*80)
    print("TEST 3: Update Position to OPEN")
    print("="*80)

    try:
        db = PositionDatabaseManager(db_path="data/test_positions.db")

        # Update to OPEN with mock order_id
        order_id = f"DEAL_{int(time.time())}"
        success = db.update_position_opened(position_id, order_id, 'OPEN')

        if success:
            print(f"✅ Position updated to OPEN: {position_id}")

            # Verify update
            retrieved = db.get_position_by_id(position_id)
            if retrieved and retrieved['status'] == 'OPEN':
                print(f"   ✅ Status verified: {retrieved['status']}")
                print(f"      Order ID: {retrieved['order_id']}")
                return True
            else:
                print(f"   ❌ Status not updated correctly")
                return False
        else:
            print(f"❌ Failed to update position to OPEN")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_get_open_positions():
    """Test 4: Retrieve all OPEN positions"""
    print("\n" + "="*80)
    print("TEST 4: Get All OPEN Positions")
    print("="*80)

    try:
        db = PositionDatabaseManager(db_path="data/test_positions.db")

        open_positions = db.get_open_positions()
        print(f"✅ Found {len(open_positions)} OPEN positions")

        for pos in open_positions:
            print(f"   - {pos['symbol']} {pos['direction']} @ ${pos['entry_price']:.2f}")
            print(f"     Position ID: {pos['position_id']}")
            print(f"     Order ID: {pos['order_id']}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_update_position_closed(position_id):
    """Test 5: Update OPEN → CLOSED"""
    print("\n" + "="*80)
    print("TEST 5: Update Position to CLOSED")
    print("="*80)

    try:
        db = PositionDatabaseManager(db_path="data/test_positions.db")

        # Update to CLOSED with mock exit data
        exit_price = 152.00
        exit_reason = ExitReason.TAKE_PROFIT.value
        realized_pnl = 20.00

        success = db.update_position_closed(
            position_id,
            exit_price,
            exit_reason,
            realized_pnl
        )

        if success:
            print(f"✅ Position updated to CLOSED: {position_id}")

            # Verify update
            retrieved = db.get_position_by_id(position_id)
            if retrieved and retrieved['status'] == 'CLOSED':
                print(f"   ✅ Status verified: {retrieved['status']}")
                print(f"      Exit Price: ${retrieved['exit_price']:.2f}")
                print(f"      Exit Reason: {retrieved['exit_reason']}")
                print(f"      Realized P&L: ${retrieved['realized_pnl']:.2f}")
                return True
            else:
                print(f"   ❌ Status not updated correctly")
                return False
        else:
            print(f"❌ Failed to update position to CLOSED")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_cancel_pending_position():
    """Test 6: Cancel PENDING position (broker call failed)"""
    print("\n" + "="*80)
    print("TEST 6: Cancel PENDING Position")
    print("="*80)

    try:
        db = PositionDatabaseManager(db_path="data/test_positions.db")

        # Create another test position
        position_id = f"NVDA_CANCEL_{int(time.time() * 1000)}"
        position = Position(
            symbol="NVDA",
            direction="SHORT",
            entry_price=150.00,
            quantity=5.0,
            entry_time=time.time(),
            stop_loss=152.00,
            take_profit=147.00,
            strategy="TEST_CANCEL",
            status=PositionStatus.PENDING,
            position_id=position_id,
            entry_spread=0.10
        )

        db.create_pending_position(position)
        print(f"   Created PENDING position: {position_id}")

        # Cancel it
        success = db.cancel_pending_position(position_id, "Test cancellation")

        if success:
            print(f"✅ Position cancelled: {position_id}")

            # Verify cancellation
            retrieved = db.get_position_by_id(position_id)
            if retrieved and retrieved['status'] == 'CANCELLED':
                print(f"   ✅ Status verified: {retrieved['status']}")
                print(f"      Exit Reason: {retrieved['exit_reason']}")
                return True
            else:
                print(f"   ❌ Status not updated correctly")
                return False
        else:
            print(f"❌ Failed to cancel position")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_database_statistics():
    """Test 7: Database statistics"""
    print("\n" + "="*80)
    print("TEST 7: Database Statistics")
    print("="*80)

    try:
        db = PositionDatabaseManager(db_path="data/test_positions.db")

        stats = db.get_statistics()
        print(f"✅ Database statistics:")
        print(f"   Total positions: {stats.get('total', 0)}")
        print(f"   PENDING: {stats.get('PENDING', 0)}")
        print(f"   OPEN: {stats.get('OPEN', 0)}")
        print(f"   CLOSED: {stats.get('CLOSED', 0)}")
        print(f"   CANCELLED: {stats.get('CANCELLED', 0)}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_crash_recovery_simulation():
    """Test 8: Simulate crash and recovery"""
    print("\n" + "="*80)
    print("TEST 8: Crash Recovery Simulation")
    print("="*80)

    try:
        # Simulate creating a position before crash
        db1 = PositionDatabaseManager(db_path="data/test_positions.db")

        position_id = f"NVDA_CRASH_{int(time.time() * 1000)}"
        position = Position(
            symbol="NVDA",
            direction="LONG",
            entry_price=151.00,
            quantity=8.0,
            entry_time=time.time(),
            stop_loss=149.00,
            take_profit=154.00,
            strategy="CRASH_TEST",
            status=PositionStatus.PENDING,
            position_id=position_id,
            entry_spread=0.10
        )

        db1.create_pending_position(position)
        print(f"   Created position before 'crash': {position_id}")

        # Simulate crash by creating new database instance (like system restart)
        del db1
        print(f"   Simulated system crash...")

        # Simulate recovery (new database instance)
        db2 = PositionDatabaseManager(db_path="data/test_positions.db")
        print(f"   System restarted...")

        # Check if position survived
        recovered = db2.get_position_by_id(position_id)
        if recovered:
            print(f"✅ Position recovered after crash!")
            print(f"   Symbol: {recovered['symbol']}")
            print(f"   Status: {recovered['status']}")
            print(f"   Entry: ${recovered['entry_price']:.2f}")

            # Clean up - update to OPEN then CLOSED
            db2.update_position_opened(position_id, "CRASH_TEST_DEAL", 'OPEN')
            db2.update_position_closed(position_id, 153.00, ExitReason.TAKE_PROFIT.value, 16.00)

            return True
        else:
            print(f"❌ Position not recovered after crash")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_reconciliation():
    """Test 9: Broker reconciliation"""
    print("\n" + "="*80)
    print("TEST 9: Broker Reconciliation")
    print("="*80)

    try:
        db = PositionDatabaseManager(db_path="data/test_positions.db")

        # Create mock broker positions
        mock_broker_positions = [
            {
                'position': {'dealId': 'DEAL_12345'},
                'market': {'epic': 'NVDA'},
            },
            {
                'position': {'dealId': 'DEAL_67890'},
                'market': {'epic': 'AAPL'},
            }
        ]

        orphaned, untracked = db.reconcile_with_broker(mock_broker_positions)

        print(f"✅ Reconciliation completed:")
        print(f"   Orphaned positions (in DB, not on broker): {len(orphaned)}")
        print(f"   Untracked positions (on broker, not in DB): {len(untracked)}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def run_all_tests():
    """Run all validation tests"""
    print("\n" + "="*80)
    print("POSITION PERSISTENCE LAYER VALIDATION TESTS")
    print("="*80)

    results = []

    # Test 1: Database initialization
    results.append(("Database Initialization", test_database_initialization()))

    # Test 2: Create PENDING position
    success, position_id = test_create_pending_position()
    results.append(("Create PENDING Position", success))

    if success and position_id:
        # Test 3: Update to OPEN
        results.append(("Update to OPEN", test_update_position_opened(position_id)))

        # Test 4: Get OPEN positions
        results.append(("Get OPEN Positions", test_get_open_positions()))

        # Test 5: Update to CLOSED
        results.append(("Update to CLOSED", test_update_position_closed(position_id)))

    # Test 6: Cancel PENDING position
    results.append(("Cancel PENDING Position", test_cancel_pending_position()))

    # Test 7: Database statistics
    results.append(("Database Statistics", test_database_statistics()))

    # Test 8: Crash recovery
    results.append(("Crash Recovery", test_crash_recovery_simulation()))

    # Test 9: Reconciliation
    results.append(("Broker Reconciliation", test_reconciliation()))

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print("="*80)
    print(f"TOTAL: {passed}/{total} tests passed")
    print("="*80)

    if passed == total:
        print("\n🎉 All tests passed! Position persistence layer is working correctly.")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Review output above.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
