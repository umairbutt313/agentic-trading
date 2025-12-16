#!/usr/bin/env python3
"""
Migrate nvidia_score_price_dump.txt to new 5-column format
Converts: timestamp,score,price
To: timestamp,score_tradingview,score_sa_news,score_sa_image,price
"""

import csv
import os

def migrate_nvidia_file():
    """Migrate nvidia dump file to new format"""
    input_file = "web/nvidia_score_price_dump.txt"
    backup_file = "web/nvidia_score_price_dump_backup.txt"
    
    # Create backup
    if os.path.exists(input_file):
        os.rename(input_file, backup_file)
        print(f"Created backup: {backup_file}")
    
    # Read backup and convert to new format
    with open(backup_file, 'r') as infile, open(input_file, 'w') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Write new header
        writer.writerow(['timestamp', 'score_tradingview', 'score_sa_news', 'score_sa_image', 'price'])
        
        # Skip old header
        next(reader)
        
        # Convert data rows
        for row in reader:
            if len(row) >= 3:  # timestamp, score, price
                timestamp, old_score, price = row[0], row[1], row[2]
                # Use old score for all sentiment types (placeholder)
                new_row = [timestamp, old_score, old_score, old_score, price]
                writer.writerow(new_row)
    
    print(f"✅ Migrated {input_file} to new 5-column format")

if __name__ == "__main__":
    migrate_nvidia_file()