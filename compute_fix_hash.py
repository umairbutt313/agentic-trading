#!/usr/bin/env python3
import hashlib

# Fix notification event key fields
event_type = 'MANUAL_TRIGGER'
fix_applied = 'true'
issue_hash = '3e8f7a5b'
file = 'trading/position_manager.py'

# Create hash input by concatenating key fields
hash_input = f'{event_type}:{fix_applied}:{issue_hash}:{file}'.lower()

# Generate MD5 hash
md5_hash = hashlib.md5(hash_input.encode()).hexdigest()

print('Hash Input String:', hash_input)
print()
print('MD5 Hash (first 8 chars):', md5_hash[:8])
print()
print('Full MD5:', md5_hash)
