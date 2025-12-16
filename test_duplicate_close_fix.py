#!/usr/bin/env python3
"""
Test script to validate duplicate close prevention fix
(ISSUE_HASH: duplicate_close_prevention_001)

This script verifies:
1. Database status check prevents duplicate close attempts
2. Only ONE broker API call is made per position close
3. No "error.not-found.dealId" errors occur
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from trading.position_manager import PositionManager, Position, PositionStatus, ExitReason
from trading.position_database import PositionDatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('logs/test_duplicate_close.log'),
        logging.StreamHandler()
    ]
)


class MockTrader:
    """Mock trader for testing without hitting real API"""

    def __init__(self):
        self.cst = "mock_session"
        self.close_count = 0
        self.positions_closed = set()

    def close_position(self, order_id: str):
        """Mock close - succeeds first time, fails with 404 after"""
        self.close_count += 1

        if order_id in self.positions_closed:
            # Simulate broker returning 404 for already-closed position
            raise RuntimeError('Failed to close position: {"errorCode":"error.not-found.dealId"}')

        # First close succeeds
        self.positions_closed.add(order_id)
        return {"dealReference": f"p_{order_id}"}

    def get_positions(self):
        """Mock get_positions"""
        return {"positions": []}

    async def get_positions_async(self):
        """Mock async get_positions"""
        return {"positions": []}


async def test_duplicate_close_prevention():
    """Test that database check prevents duplicate close attempts"""

    print("\n" + "="*70)
    print("TEST: Duplicate Close Prevention")
    print("="*70 + "\n")

    # Create mock trader
    mock_trader = MockTrader()

    # Create position manager with mock trader (it will create its own test DB)
    pm = PositionManager(mock_trader)

    # Use the position manager's database instance
    db = pm.db

    # Create test position
    position = Position(
        symbol="NVDA",
        direction="LONG",
        entry_price=150.0,
        quantity=1.0,
        entry_time=1234567890.0,
        stop_loss=149.0,
        take_profit=151.0,
        strategy="test",
        position_id="TEST_POSITION_001",
        order_id="test_deal_123",
        status=PositionStatus.OPEN
    )

    # Create PENDING in database first
    db.create_pending_position(position)

    # Update to OPEN (simulate successful broker open)
    db.update_position_opened(position.position_id, position.order_id, 'OPEN')

    print(f"✅ Created test position: {position.position_id}")
    print(f"   Order ID: {position.order_id}")
    print(f"   Database status: {db.get_position_status(position.position_id)}")
    print()

    # Add to position manager's tracking
    pm.positions[position.position_id] = position

    # TEST 1: First close should succeed
    print("TEST 1: First close attempt (should succeed)")
    print("-" * 70)

    await pm.close_position(position, ExitReason.STOP_LOSS, 149.0)

    print(f"   Broker close count: {mock_trader.close_count}")
    print(f"   Database status: {db.get_position_status(position.position_id)}")
    print(f"   In-memory status: {position.status}")
    print()

    # Verify first close succeeded
    assert mock_trader.close_count == 1, f"Expected 1 close call, got {mock_trader.close_count}"
    assert db.get_position_status(position.position_id) == 'CLOSED', "Database should show CLOSED"
    print("✅ TEST 1 PASSED: First close succeeded\n")

    # TEST 2: Second close should be skipped (duplicate prevention)
    print("TEST 2: Second close attempt (should be skipped)")
    print("-" * 70)

    # Reset position status to OPEN in-memory (simulate concurrent task)
    position.status = PositionStatus.OPEN
    print(f"   Simulated concurrent task: position.status = {position.status}")

    await pm.close_position(position, ExitReason.STOP_LOSS, 149.0)

    print(f"   Broker close count: {mock_trader.close_count}")
    print(f"   Database status: {db.get_position_status(position.position_id)}")
    print(f"   In-memory status: {position.status}")
    print()

    # Verify second close was skipped
    assert mock_trader.close_count == 1, f"Expected 1 close call (no duplicate), got {mock_trader.close_count}"
    assert db.get_position_status(position.position_id) == 'CLOSED', "Database should still show CLOSED"
    print("✅ TEST 2 PASSED: Duplicate close was prevented\n")

    # TEST 3: Multiple concurrent attempts (simulate reconciliation bug)
    print("TEST 3: Multiple concurrent close attempts (simulate bug scenario)")
    print("-" * 70)

    # Create new test position
    position2 = Position(
        symbol="NVDA",
        direction="SHORT",
        entry_price=150.0,
        quantity=1.0,
        entry_time=1234567891.0,
        stop_loss=151.0,
        take_profit=149.0,
        strategy="test",
        position_id="TEST_POSITION_002",
        order_id="test_deal_456",
        status=PositionStatus.OPEN
    )

    db.create_pending_position(position2)
    db.update_position_opened(position2.position_id, position2.order_id, 'OPEN')
    pm.positions[position2.position_id] = position2

    print(f"   Created position: {position2.position_id}")
    print(f"   Simulating 10 concurrent close attempts...")
    print()

    # Launch 10 concurrent close tasks (simulate the bug)
    tasks = []
    for i in range(10):
        # Create copy with OPEN status (simulate concurrent tasks seeing stale state)
        pos_copy = Position(
            symbol=position2.symbol,
            direction=position2.direction,
            entry_price=position2.entry_price,
            quantity=position2.quantity,
            entry_time=position2.entry_time,
            stop_loss=position2.stop_loss,
            take_profit=position2.take_profit,
            strategy=position2.strategy,
            position_id=position2.position_id,
            order_id=position2.order_id,
            status=PositionStatus.OPEN  # Stale in-memory state
        )
        pm.positions[pos_copy.position_id] = pos_copy
        tasks.append(pm.close_position(pos_copy, ExitReason.TRAILING_STOP, 149.5))

    # Run all tasks concurrently
    await asyncio.gather(*tasks)

    print(f"   Broker close count: {mock_trader.close_count}")
    print(f"   Database status: {db.get_position_status(position2.position_id)}")
    print()

    # Verify only ONE broker API call was made
    assert mock_trader.close_count == 2, f"Expected 2 total close calls (1 from test 1), got {mock_trader.close_count}"
    assert db.get_position_status(position2.position_id) == 'CLOSED', "Database should show CLOSED"
    print("✅ TEST 3 PASSED: Only ONE close succeeded, 9 duplicates prevented\n")

    print("="*70)
    print("ALL TESTS PASSED ✅")
    print("="*70)
    print()
    print("Fix validated:")
    print("  - Database status check prevents duplicate broker API calls")
    print("  - No 'error.not-found.dealId' errors occur")
    print("  - Concurrent close attempts are safely handled")
    print()


if __name__ == "__main__":
    asyncio.run(test_duplicate_close_prevention())
