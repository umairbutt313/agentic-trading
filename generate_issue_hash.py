#!/usr/bin/env python3
import hashlib

# ERROR event components - Sentiment analyzer data availability issue
event_type = 'ERROR'
error_message = 'No articles found in dump file'
source = 'sentiment_analyzer.log'
error_category = 'Data availability issue'

# Create a deterministic hash input by combining the key components
hash_input = f'{event_type}:{error_message}:{source}:{error_category}'.lower()

# Generate MD5 hash (good for non-cryptographic deduplication)
md5_hash = hashlib.md5(hash_input.encode()).hexdigest()

# Generate SHA256 hash (more collision-resistant)
sha256_hash = hashlib.sha256(hash_input.encode()).hexdigest()

print('Hash Input String:', hash_input)
print()
print('MD5 Hash (8 chars):', md5_hash[:8])
print('MD5 Hash (12 chars):', md5_hash[:12])
print()
print('SHA256 Hash (8 chars):', sha256_hash[:8])
print('SHA256 Hash (12 chars):', sha256_hash[:12])
print()
print('Full MD5:', md5_hash)
print('Full SHA256:', sha256_hash)
