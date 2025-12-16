#!/usr/bin/env python3
"""
Simple continuous runner - keeps running test_all.sh until API credits exhausted
"""

import subprocess
import time
import sys

def run_test_all():
    """Run test_all.sh and return True if successful"""
    try:
        # Run and wait for completion (subprocess.run waits by default)
        print("⏳ Running test_all.sh (this may take several minutes)...")
        result = subprocess.run(['./test_all.sh'], capture_output=True, text=True)
        
        # Check for API rate limit or credit errors in both stderr and stdout
        output_text = result.stdout + result.stderr
        if "insufficient_quota" in output_text or "rate_limit_exceeded" in output_text or "quota exceeded" in output_text.lower():
            print("❌ API credits exhausted or rate limit hit")
            return False
        
        if result.returncode == 0:
            print("✅ test_all.sh completed successfully")
        else:
            print("⚠️ test_all.sh finished with errors (continuing anyway)")
            
        # Always return True unless API limits hit - continue on test failures
        return True
    except Exception as e:
        print(f"Error running test_all.sh: {e}")
        return False

def main():
    print("🚀 Starting continuous data collection...")
    print("This will run until API credits are exhausted")
    print("Press Ctrl+C to stop manually\n")
    
    run_count = 0
    
    while True:
        run_count += 1
        print(f"\n{'='*60}")
        print(f"🔄 Run #{run_count} starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # Run the test script
        if not run_test_all():
            print("\n❌ Script failed or API exhausted. Stopping.")
            break
            
        # Wait 10 seconds before next run
        print(f"\n✅ Run #{run_count} completed. Waiting 10 seconds...")
        time.sleep(10)  # 10 seconds
    
    print(f"\n🏁 Completed {run_count} runs total")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⛔ Stopped by user (Ctrl+C)")
        sys.exit(0)