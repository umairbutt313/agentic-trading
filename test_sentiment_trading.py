#!/usr/bin/env python3
"""
Sentiment Trading Verification Test
Creates controlled sentiment changes to verify trading system response
"""

import json
import os
import time
from datetime import datetime
from trading.auto_trader import SentimentAutoTrader

class SentimentTradingVerifier:
    def __init__(self):
        self.base_dir = "/root/arslan-chart/agentic-trading-dec2025/stocks"
        self.final_score_path = os.path.join(self.base_dir, "container_output/final_score/final-weighted-scores.json")
        self.backup_path = self.final_score_path + ".backup"
        
        # Create trader instance
        self.trader = SentimentAutoTrader()
        
    def backup_current_sentiment(self):
        """Backup current sentiment data"""
        if os.path.exists(self.final_score_path):
            os.system(f"cp {self.final_score_path} {self.backup_path}")
            print("✅ Backed up current sentiment data")
    
    def restore_sentiment(self):
        """Restore original sentiment data"""
        if os.path.exists(self.backup_path):
            os.system(f"cp {self.backup_path} {self.final_score_path}")
            print("✅ Restored original sentiment data")
    
    def create_test_sentiment(self, nvda_score: float, test_name: str):
        """Create artificial sentiment data for testing"""
        test_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_companies": 1,
                "companies_with_image_data": 0,
                "calculation_method": "TEST_MODE",
                "test_name": test_name
            },
            "companies": {
                "NVIDIA": {
                    "final_score": nvda_score,
                    "news_score": nvda_score,
                    "image_score": None,
                    "calculation_method": f"TEST: {test_name}",
                    "calculation_details": f"Artificial test score: {nvda_score}",
                    "timestamp": datetime.now().isoformat()
                }
            }
        }
        
        with open(self.final_score_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        print(f"🧪 Created test sentiment: {test_name} (NVDA: {nvda_score}/10)")
        return test_data
    
    def run_verification_tests(self):
        """Run comprehensive trading verification tests"""
        print("🔬 STARTING SENTIMENT TRADING VERIFICATION")
        print("=" * 60)
        
        # Backup current data
        self.backup_current_sentiment()
        
        tests = [
            {"score": 8.5, "name": "STRONG_BUY_TEST", "expected": "BUY signal"},
            {"score": 3.0, "name": "STRONG_SELL_TEST", "expected": "SELL/SHORT signal"},
            {"score": 5.5, "name": "NEUTRAL_TEST", "expected": "HOLD signal"},
            {"score": 7.0, "name": "BUY_THRESHOLD_TEST", "expected": "BUY signal"},
            {"score": 4.0, "name": "SELL_THRESHOLD_TEST", "expected": "SELL signal"}
        ]
        
        try:
            for i, test in enumerate(tests, 1):
                print(f"\n🧪 TEST {i}/{len(tests)}: {test['name']}")
                print(f"   Setting NVDA sentiment to: {test['score']}/10")
                print(f"   Expected result: {test['expected']}")
                
                # Create test sentiment
                self.create_test_sentiment(test['score'], test['name'])
                
                # Wait a moment for file system
                time.sleep(2)
                
                print(f"   🤖 Executing trading cycle...")
                
                # Execute one trading cycle
                self.trader.execute_trading_cycle()
                
                input(f"\n   📋 Did the system show '{test['expected']}'? Press Enter to continue...")
                
        finally:
            # Always restore original data
            print(f"\n🔄 Restoring original sentiment data...")
            self.restore_sentiment()
            
        print(f"\n✅ VERIFICATION TESTS COMPLETED")

if __name__ == "__main__":
    verifier = SentimentTradingVerifier()
    verifier.run_verification_tests()