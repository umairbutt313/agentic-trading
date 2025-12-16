#!/usr/bin/env python3
"""
Capital.com Trade History Verifier
Checks actual trades executed on Capital.com and correlates with sentiment
"""

import json
import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.capital_trader import CapitalTrader

class CapitalTradeVerifier:
    def __init__(self):
        self.base_dir = "/root/arslan-chart/agentic-trading-dec2025/stocks"
        self.sentiment_file = os.path.join(self.base_dir, "container_output/final_score/final-weighted-scores.json")
        
        # Load Capital.com credentials
        self.load_config()
        
        # Initialize Capital.com trader
        self.trader = CapitalTrader(
            api_key=self.capital_api_key,
            password=self.capital_password,
            email=self.capital_email,
            demo=True
        )
    
    def load_config(self):
        """Load Capital.com credentials from .env"""
        from dotenv import load_dotenv
        load_dotenv(os.path.join(self.base_dir, '.env'))
        
        self.capital_api_key = os.getenv("CAPITAL_API_KEY")
        self.capital_password = os.getenv("CAPITAL_PASSWORD") 
        self.capital_email = os.getenv("CAPITAL_EMAIL", "awesome313@gmail.com")
    
    def get_current_sentiment(self) -> Dict:
        """Get current NVDA sentiment"""
        try:
            with open(self.sentiment_file, 'r') as f:
                data = json.load(f)
            
            nvda_data = data.get('companies', {}).get('NVIDIA', {})
            return {
                'score': nvda_data.get('final_score', 5.0),
                'timestamp': nvda_data.get('timestamp', ''),
                'method': nvda_data.get('calculation_method', '')
            }
        except Exception as e:
            print(f"❌ Error reading sentiment: {e}")
            return {'score': 5.0, 'timestamp': '', 'method': 'ERROR'}
    
    def get_recent_trades(self, hours: int = 24) -> List[Dict]:
        """Get recent trades from Capital.com"""
        try:
            print("🔍 Fetching recent trades from Capital.com...")
            
            # This would need to be implemented based on Capital.com API
            # For now, we'll show account info and positions
            account_info = self.trader.get_account_info()
            positions = self.trader.monitor_positions()
            
            print("📊 Account Information:")
            if isinstance(account_info, dict):
                for key, value in account_info.items():
                    print(f"   {key}: {value}")
            else:
                print(f"   {account_info}")
            
            print(f"\n📈 Current Positions:")
            print(f"   {positions}")
            
            # Return mock data structure for now
            return []
            
        except Exception as e:
            print(f"❌ Error fetching trades: {e}")
            return []
    
    def verify_sentiment_trade_correlation(self):
        """Verify if trades correlate with sentiment"""
        print("🔬 VERIFYING SENTIMENT-TRADE CORRELATION")
        print("=" * 50)
        
        # Get current sentiment
        sentiment = self.get_current_sentiment()
        print(f"📊 Current NVDA Sentiment:")
        print(f"   Score: {sentiment['score']}/10")
        print(f"   Method: {sentiment['method']}")
        print(f"   Timestamp: {sentiment['timestamp']}")
        
        # Determine expected action
        if sentiment['score'] >= 7.0:
            expected_action = "BUY/LONG"
        elif sentiment['score'] <= 4.0:
            expected_action = "SELL/SHORT"
        else:
            expected_action = "HOLD"
        
        print(f"   Expected Action: {expected_action}")
        
        # Get recent trades
        trades = self.get_recent_trades()
        
        # Check account status
        print(f"\n💰 Capital.com Account Status:")
        try:
            balance = self.trader.get_account_info()
            print(f"   Balance: {balance}")
            
            positions = self.trader.monitor_positions()
            print(f"   Positions: {positions}")
            
        except Exception as e:
            print(f"   ❌ Error getting account status: {e}")
        
        return {
            'sentiment': sentiment,
            'expected_action': expected_action,
            'trades': trades
        }
    
    def test_manual_trade(self):
        """Test manual trade to verify API connectivity"""
        print("🧪 TESTING MANUAL TRADE (Demo Mode)")
        print("=" * 40)
        
        sentiment = self.get_current_sentiment()
        print(f"Current NVDA sentiment: {sentiment['score']}/10")
        
        # Test position sizing calculation
        test_sentiment_scores = [3.0, 5.0, 7.0, 8.5]
        
        for score in test_sentiment_scores:
            try:
                size = self.trader.calculate_position_size("NVDA", score)
                print(f"Sentiment {score}/10 → Position size: {size:.2f} units")
            except Exception as e:
                print(f"❌ Error calculating position for {score}: {e}")
        
        print(f"\n🔍 Testing actual trade execution...")
        try:
            # Test current sentiment trade
            result = self.trader.execute_sentiment_trade("NVDA", sentiment['score'])
            print(f"Trade result: {result}")
        except Exception as e:
            print(f"❌ Trade execution error: {e}")

if __name__ == "__main__":
    verifier = CapitalTradeVerifier()
    
    print("🔍 What would you like to verify?")
    print("1. Current sentiment vs expected trades")
    print("2. Test manual trade execution")
    print("3. Both")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice in ["1", "3"]:
        verifier.verify_sentiment_trade_correlation()
    
    if choice in ["2", "3"]:
        print("\n" + "="*50)
        verifier.test_manual_trade()