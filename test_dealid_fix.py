#!/usr/bin/env python3
"""
Validation Test for DealId Mismatch Fix (ISSUE_HASH: dealid_mismatch_close_failure_001)

This test verifies:
1. Position opens successfully
2. Correct dealId is stored (from affectedDeals, not root)
3. Position can be closed without "error.not-found.dealId"
4. Broker confirms position is actually closed

Run this test BEFORE deploying the fix to production.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading.capital_trader import CapitalTrader
from trading.position_manager import PositionManager
from trading.scalping_strategies import ScalpingStrategy
from trading.models import Signal, SignalType
from price_fetcher.realtime_price_storage import RealtimePriceStorage

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/dealid_fix_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

async def test_dealid_fix():
    """
    Test the dealId fix by opening and closing a test position
    """
    print("=" * 80)
    print("DEALID FIX VALIDATION TEST")
    print("=" * 80)
    print()

    # Initialize components
    print("📋 Step 1: Initializing components...")
    trader = CapitalTrader(
        api_key=os.getenv('CAPITAL_API_KEY'),
        password=os.getenv('CAPITAL_PASSWORD'),
        email=os.getenv('CAPITAL_EMAIL'),
        demo=True  # ALWAYS use demo for testing!
    )

    # Create session
    trader.create_session()
    print("✅ Trading session created")

    # Initialize price storage and position manager
    price_storage = RealtimePriceStorage(backend='json')
    position_manager = PositionManager(trader, price_storage)

    print()
    print("📋 Step 2: Creating test BUY signal...")

    # Create a simple test signal (NVDA long)
    test_signal = Signal(
        symbol='NVDA',
        signal_type=SignalType.BUY,
        strategy='TEST_DEALID_FIX',
        confidence=0.75,
        entry_price=177.50,
        stop_loss=176.00,
        take_profit=179.00,
        position_size=0.1,  # Minimal size for testing
        metadata={
            'test': True,
            'issue_hash': 'dealid_mismatch_close_failure_001'
        }
    )

    print(f"✅ Test signal created: {test_signal.symbol} {test_signal.signal_type}")
    print()

    # Step 3: Open position
    print("📋 Step 3: Opening test position...")
    print("   This will test dealId extraction from Capital.com API response")
    print()

    position = await position_manager.open_position(test_signal)

    if not position:
        print("❌ FAILED: Could not open position")
        return False

    print(f"✅ Position opened: {position.position_id}")
    print(f"   Order ID (dealId): {position.order_id}")
    print()

    # Step 4: Verify dealId format
    print("📋 Step 4: Verifying dealId format...")
    if not position.order_id:
        print("❌ FAILED: No order_id stored!")
        return False

    if position.order_id.startswith('o_'):
        print(f"❌ FAILED: order_id is dealReference, not dealId: {position.order_id}")
        print("   This means the fix didn't work - still using root dealId!")
        return False

    # Capital.com dealIds have format: XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
    if len(position.order_id) == 36 and position.order_id.count('-') == 4:
        print(f"✅ DealId format valid: {position.order_id}")
        print("   (36 chars with 4 hyphens = UUID format)")
    else:
        print(f"⚠️  WARNING: Unexpected dealId format: {position.order_id}")

    print()

    # Step 5: Check broker position
    print("📋 Step 5: Verifying position exists on broker...")
    broker_positions = await trader.get_positions_async()
    broker_position = None

    for pos in broker_positions.get('positions', []):
        pos_data = pos.get('position', {})
        if pos_data.get('dealId') == position.order_id:
            broker_position = pos
            break

    if not broker_position:
        print(f"❌ FAILED: Position {position.order_id} not found on broker!")
        print("   Stored dealId doesn't match any broker position")
        return False

    print(f"✅ Position confirmed on broker: {position.order_id}")
    print()

    # Step 6: Close position (THE CRITICAL TEST)
    print("📋 Step 6: Closing position (CRITICAL TEST)...")
    print(f"   Using dealId: {position.order_id}")
    print()

    try:
        # Get current price for close
        current_price = await trader.get_current_price_async(position.symbol)

        # Close the position
        from trading.models import ExitReason
        await position_manager.close_position(
            position=position,
            exit_reason=ExitReason.MANUAL_EXIT,
            exit_price=current_price
        )

        print("✅ SUCCESS: Position closed without errors!")
        print()

    except Exception as e:
        error_msg = str(e)
        if "error.not-found.dealId" in error_msg:
            print(f"❌ FAILED: Got 'error.not-found.dealId' error!")
            print(f"   Error: {error_msg}")
            print("   This means the fix didn't work - wrong dealId being used!")
            return False
        else:
            print(f"❌ FAILED: Unexpected error closing position: {e}")
            return False

    # Step 7: Verify broker position is closed
    print("📋 Step 7: Verifying position closed on broker...")
    await asyncio.sleep(2)  # Wait for broker to process

    broker_positions_after = await trader.get_positions_async()
    position_still_open = False

    for pos in broker_positions_after.get('positions', []):
        pos_data = pos.get('position', {})
        if pos_data.get('dealId') == position.order_id:
            position_still_open = True
            break

    if position_still_open:
        print(f"❌ FAILED: Position {position.order_id} still open on broker!")
        print("   Close command didn't actually close the position")
        return False

    print(f"✅ Position confirmed closed on broker")
    print()

    # All tests passed!
    print("=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print()
    print("Summary:")
    print("  ✅ Position opened successfully")
    print("  ✅ Correct dealId stored (affectedDeals, not root)")
    print("  ✅ Position closed without 'error.not-found.dealId'")
    print("  ✅ Broker confirmed position actually closed")
    print()
    print("The dealId fix is WORKING CORRECTLY!")
    print()

    return True


if __name__ == "__main__":
    try:
        result = asyncio.run(test_dealid_fix())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
