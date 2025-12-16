#!/usr/bin/env python3
"""
Test script for Official Capital.com /confirms endpoint implementation

Tests the complete position lifecycle:
1. Open position → Get dealReference
2. Confirm deal → Get dealId from affectedDeals
3. Close position → Use dealId

This validates the fix for the dealReference vs dealId bug.
"""

import os
import sys
import asyncio
import logging
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading.capital_trader import CapitalTrader

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def test_confirms_endpoint():
    """Test the official Capital.com confirms endpoint workflow"""

    logging.info("=" * 80)
    logging.info("TESTING OFFICIAL CAPITAL.COM /confirms ENDPOINT")
    logging.info("=" * 80)

    # Initialize trader
    trader = CapitalTrader(
        api_key=os.getenv('CAPITAL_API_KEY'),
        password=os.getenv('CAPITAL_PASSWORD'),
        email=os.getenv('CAPITAL_EMAIL'),
        demo=True
    )

    try:
        # Step 1: Create session
        logging.info("\n📡 Step 1: Creating session...")
        session = trader.create_session()
        logging.info(f"✅ Session created")

        # Step 2: Check existing positions and close all
        logging.info("\n🧹 Step 2: Cleaning up existing positions...")
        positions = await trader.get_positions_async()
        position_count = len(positions.get('positions', []))

        if position_count > 0:
            logging.warning(f"⚠️ Found {position_count} existing positions - closing all...")
            for pos in positions.get('positions', []):
                deal_id = pos['position']['dealId']
                symbol = pos['market']['epic']
                logging.info(f"   Closing {symbol} (dealId: {deal_id})")
                trader.close_position(deal_id)
            logging.info(f"✅ Closed {position_count} positions")
        else:
            logging.info(f"✅ No existing positions")

        # Step 3: Open a test position
        logging.info("\n📈 Step 3: Opening test position...")
        order_data = {
            "epic": "NVDA",
            "direction": "BUY",
            "size": 0.1,  # Minimum size
            "guaranteedStop": False,
            "trailingStop": False
        }

        logging.info(f"   Symbol: NVDA")
        logging.info(f"   Direction: BUY")
        logging.info(f"   Size: 0.1")

        # Place order
        order_result = trader.place_order(
            symbol="NVDA",
            direction="BUY",
            size=0.1
        )

        # Extract dealReference
        deal_reference = order_result.get('dealReference')

        if not deal_reference:
            logging.error(f"❌ FAILED: No dealReference in response: {order_result}")
            return False

        logging.info(f"✅ Position opened")
        logging.info(f"   dealReference: {deal_reference}")

        # Step 4: Confirm deal using official endpoint
        logging.info(f"\n🔍 Step 4: Confirming deal using /confirms endpoint...")
        confirm_result = await trader.confirm_deal_async(deal_reference)

        # Extract dealId from affectedDeals
        affected_deals = confirm_result.get('affectedDeals', [])

        if not affected_deals:
            logging.error(f"❌ FAILED: No affectedDeals in response: {confirm_result}")
            return False

        deal_id = affected_deals[0].get('dealId')
        deal_status = confirm_result.get('dealStatus')
        position_status = confirm_result.get('status')

        if not deal_id:
            logging.error(f"❌ FAILED: No dealId in affectedDeals: {affected_deals}")
            return False

        logging.info(f"✅ Deal confirmed")
        logging.info(f"   dealReference: {deal_reference}")
        logging.info(f"   dealId: {deal_id}")
        logging.info(f"   dealStatus: {deal_status}")
        logging.info(f"   positionStatus: {position_status}")

        # Step 5: Verify position on broker
        logging.info(f"\n🔎 Step 5: Verifying position on broker...")
        positions = await trader.get_positions_async()

        found = False
        for pos in positions.get('positions', []):
            pos_deal_id = pos['position']['dealId']
            if pos_deal_id == deal_id:
                found = True
                logging.info(f"✅ Position found on broker")
                logging.info(f"   dealId: {pos_deal_id}")
                logging.info(f"   Symbol: {pos['market']['epic']}")
                logging.info(f"   Direction: {pos['position']['direction']}")
                logging.info(f"   Size: {pos['position']['size']}")
                break

        if not found:
            logging.warning(f"⚠️ Position not found on broker (may be already closed)")

        # Step 6: Close position using dealId
        logging.info(f"\n🔒 Step 6: Closing position using dealId...")
        close_result = trader.close_position(deal_id)

        logging.info(f"✅ Position closed successfully")
        logging.info(f"   Response: {close_result}")

        # Step 7: Verify position is closed
        logging.info(f"\n✔️ Step 7: Verifying position is closed...")
        positions = await trader.get_positions_async()

        still_open = False
        for pos in positions.get('positions', []):
            if pos['position']['dealId'] == deal_id:
                still_open = True
                break

        if still_open:
            logging.error(f"❌ FAILED: Position still open on broker!")
            return False
        else:
            logging.info(f"✅ Position confirmed closed on broker")

        # Success!
        logging.info("\n" + "=" * 80)
        logging.info("✅ ALL TESTS PASSED!")
        logging.info("=" * 80)
        logging.info("\nSummary:")
        logging.info(f"  1. ✅ Session created")
        logging.info(f"  2. ✅ Test position opened (dealReference: {deal_reference})")
        logging.info(f"  3. ✅ Deal confirmed via /confirms endpoint (dealId: {deal_id})")
        logging.info(f"  4. ✅ Position closed successfully using dealId")
        logging.info(f"  5. ✅ Position confirmed closed on broker")
        logging.info("\n🎉 Official Capital.com workflow is working correctly!")

        return True

    except Exception as e:
        logging.error(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_confirms_endpoint())
    sys.exit(0 if result else 1)
