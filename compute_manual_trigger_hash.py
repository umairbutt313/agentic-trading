#!/usr/bin/env python3
import hashlib

# Key fields for the manual trigger event
event_type = 'MANUAL_TRIGGER'
description = 'Analyze latest trading performance from logs'
context_request = 'full_log_analysis'

# Create hash input by concatenating key fields
hash_input = f'{event_type}:{description}:{context_request}'

# Generate MD5 hash
md5_hash = hashlib.md5(hash_input.encode()).hexdigest()

print('Hash Input String:', hash_input)
print()
print('MD5 Hash (first 8 chars):', md5_hash[:8])
print()
print('Full MD5:', md5_hash)
