#!/usr/bin/env python3
"""
Correlation Log Analysis Script
Analyzes and displays correlation monitoring results in a readable format
"""

import csv
import os
import json
from datetime import datetime
from typing import Dict, List

def analyze_csv_log(log_file: str):
    """Analyze CSV correlation log"""
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        return
    
    print("📊 CSV CORRELATION LOG ANALYSIS")
    print("=" * 60)
    
    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        events = list(reader)
    
    if not events:
        print("📭 No events found in log")
        return
    
    print(f"📈 Total Events: {len(events)}")
    print(f"📅 Time Range: {events[0]['timestamp'][:19]} to {events[-1]['timestamp'][:19]}")
    print()
    
    # Signal analysis
    buy_signals = len([e for e in events if e['expected_action'] == 'BUY'])
    sell_signals = len([e for e in events if 'SELL' in e['expected_action']])
    hold_signals = len([e for e in events if e['expected_action'] == 'HOLD'])
    
    print("🎯 SIGNAL DISTRIBUTION:")
    print(f"   BUY Signals:  {buy_signals:2d} ({buy_signals/len(events)*100:5.1f}%)")
    print(f"   SELL Signals: {sell_signals:2d} ({sell_signals/len(events)*100:5.1f}%)")
    print(f"   HOLD Signals: {hold_signals:2d} ({hold_signals/len(events)*100:5.1f}%)")
    print()
    
    # Sentiment statistics
    sentiments = [float(e['nvda_sentiment']) for e in events]
    print("📊 SENTIMENT STATISTICS:")
    print(f"   Range: {min(sentiments):.1f} - {max(sentiments):.1f}")
    print(f"   Average: {sum(sentiments)/len(sentiments):.1f}")
    print(f"   Most Common: {max(set(sentiments), key=sentiments.count):.1f}")
    print()
    
    # Event timeline
    print("📅 EVENT TIMELINE (Latest 10):")
    print("-" * 60)
    for i, event in enumerate(events[-10:], 1):
        timestamp = event['timestamp'].split('T')[1].split('.')[0]
        sentiment = event['nvda_sentiment']
        change = event['sentiment_change']
        action = event['expected_action']
        verified = event['correlation_verified']
        print(f"   {i:2d}. {timestamp} | NVDA: {sentiment:>4} | {change:>5} | {action:>4} | {verified}")
    print("-" * 60)

def analyze_session_logs(session_log_file: str):
    """Show session summaries"""
    if not os.path.exists(session_log_file):
        print(f"❌ Session log file not found: {session_log_file}")
        return
        
    print("\n📋 SESSION SUMMARIES")
    print("=" * 60)
    
    with open(session_log_file, 'r') as f:
        content = f.read()
        sessions = content.split('CORRELATION MONITORING SESSION SUMMARY')
        
    for session in sessions[1:]:  # Skip first empty split
        if 'Session ID:' in session:
            lines = session.strip().split('\n')
            for line in lines[:10]:  # Show first 10 lines of each session
                if line.strip() and not line.startswith('='):
                    print(f"   {line}")
            print("-" * 40)

def show_recent_sentiment_data():
    """Show current sentiment data for context"""
    sentiment_file = "/root/arslan-chart/agentic-trading-dec2025/stocks/container_output/final_score/final-weighted-scores.json"
    
    if not os.path.exists(sentiment_file):
        print("❌ Sentiment data file not found")
        return
        
    try:
        with open(sentiment_file, 'r') as f:
            data = json.load(f)
        
        nvidia_data = data['companies']['NVIDIA']
        main_timestamp = data['metadata']['timestamp']
        
        # Calculate data age
        try:
            data_time = datetime.fromisoformat(main_timestamp.replace('Z', ''))
            data_age_seconds = (datetime.now() - data_time).total_seconds()
            age_minutes = data_age_seconds / 60
        except:
            age_minutes = float('inf')
        
        print("\n🎯 CURRENT SENTIMENT CONTEXT")
        print("=" * 60)
        print(f"NVIDIA Sentiment: {nvidia_data['final_score']}/10")
        print(f"News Score: {nvidia_data['news_score']}/10")
        print(f"Data Age: {age_minutes:.1f} minutes ({'FRESH' if age_minutes <= 15 else 'STALE'})")
        print(f"Last Updated: {main_timestamp}")
        print(f"Expected Action: {'BUY' if nvidia_data['final_score'] >= 7.0 else 'SELL' if nvidia_data['final_score'] <= 4.0 else 'HOLD'}")
        
    except Exception as e:
        print(f"❌ Error reading sentiment data: {e}")

def main():
    """Main analysis function"""
    base_dir = "/root/arslan-chart/agentic-trading-dec2025/stocks"
    csv_log = os.path.join(base_dir, "logs/trading_correlation.csv")  # UPDATED: Now in logs folder
    session_log = os.path.join(base_dir, "logs/correlation_sessions.log")
    detailed_log = os.path.join(base_dir, "logs/detailed_correlation.log")
    
    print("🔍 TRADING CORRELATION LOG ANALYZER")
    print("=" * 80)
    print(f"📁 Analyzing logs from: {base_dir}")
    print()
    
    # Analyze CSV log
    analyze_csv_log(csv_log)
    
    # Show session summaries
    analyze_session_logs(session_log)
    
    # Show current sentiment context
    show_recent_sentiment_data()
    
    # File status
    print(f"\n📄 LOG FILE STATUS:")
    files_to_check = [
        ("CSV Log", csv_log),
        ("Detailed Log", detailed_log),
        ("Session Log", session_log),
        ("Monitor Log", os.path.join(base_dir, "logs/correlation_monitor.log"))
    ]
    
    for name, filepath in files_to_check:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
            print(f"   ✅ {name}: {size:,} bytes (Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
        else:
            print(f"   ❌ {name}: Not found")
    
    print(f"\n💡 To view detailed logs manually:")
    print(f"   cat {detailed_log}")
    print(f"   tail -50 {csv_log}")

if __name__ == "__main__":
    main()