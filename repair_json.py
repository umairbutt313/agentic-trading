#!/usr/bin/env python3
"""
JSON Repair Script for Price History
Repairs corrupted price_history.json file
"""

import json
import re
from pathlib import Path

def repair_price_json(file_path: str):
    """Repair corrupted price history JSON file"""
    print(f"🔧 Repairing JSON file: {file_path}")
    
    # Read the corrupted file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the last complete JSON object
    # Look for pattern: "},\n    {\n" followed by incomplete data
    pattern = r'(.*},\s*{\s*$)'
    match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
    
    if match:
        # Truncate at the last complete object
        good_content = content[:match.start(1)] + content[match.start(1):match.end(1)]
        # Remove the incomplete object starter
        good_content = good_content.rstrip().rstrip(',').rstrip()
        
        # Find the last symbol section and add proper closing
        # Look for the structure to understand nesting
        try:
            # Try to parse what we have so far
            lines = good_content.split('\n')
            
            # Find where we are in the structure
            brace_count = 0
            bracket_count = 0
            last_symbol_line = -1
            
            for i, line in enumerate(lines):
                # Count braces and brackets to understand nesting
                brace_count += line.count('{') - line.count('}')
                bracket_count += line.count('[') - bracket_count.count(']')
                
                # Track symbol sections
                if '"NVDA"' in line or '"AAPL"' in line or '"INTC"' in line or '"MSFT"' in line or '"GOOG"' in line or '"TSLA"' in line:
                    last_symbol_line = i
            
            # Rebuild the JSON structure
            result_lines = []
            in_array = False
            
            for i, line in enumerate(lines):
                if line.strip().endswith('['):
                    in_array = True
                    result_lines.append(line)
                elif in_array and (line.strip() == '{' or '"timestamp"' in line or '"price"' in line):
                    # We're in a price object
                    result_lines.append(line)
                elif in_array and line.strip().startswith('}'):
                    result_lines.append(line)
                    # Check if this closes the array
                    if i < len(lines) - 1 and not lines[i+1].strip().startswith(','):
                        in_array = False
                else:
                    result_lines.append(line)
            
            # Close any open structures
            if in_array:
                # Close the current price array
                result_lines.append('    ]')
            
            # Add final closing brace
            if not result_lines[-1].strip() == '}':
                result_lines.append('}')
            
            repaired_content = '\n'.join(result_lines)
            
        except Exception as e:
            print(f"❌ Complex repair failed: {e}")
            # Fallback: create minimal valid JSON
            repaired_content = '{\n  "NVDA": [],\n  "AAPL": [],\n  "INTC": [],\n  "MSFT": [],\n  "GOOG": [],\n  "TSLA": []\n}'
    
    else:
        print("❌ Could not find repair point, creating fresh JSON")
        repaired_content = '{\n  "NVDA": [],\n  "AAPL": [],\n  "INTC": [],\n  "MSFT": [],\n  "GOOG": [],\n  "TSLA": []\n}'
    
    # Validate the repaired JSON
    try:
        json.loads(repaired_content)
        print("✅ Repaired JSON is valid")
    except json.JSONDecodeError as e:
        print(f"❌ Repaired JSON still invalid: {e}")
        # Final fallback
        repaired_content = '{\n  "NVDA": [],\n  "AAPL": [],\n  "INTC": [],\n  "MSFT": [],\n  "GOOG": [],\n  "TSLA": []\n}'
    
    # Write the repaired content
    with open(file_path, 'w') as f:
        f.write(repaired_content)
    
    print(f"🎯 JSON repair completed")
    return True

if __name__ == "__main__":
    repair_price_json('/root/arslan-chart/agentic-trading-dec2025/stocks/container_output/realtime_data/price_history.json')