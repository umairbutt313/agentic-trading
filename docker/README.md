# Docker Configuration

This folder contains all Docker-related files for the Stock Sentiment Analysis System.

## Files

- `Dockerfile` - Main Docker container configuration
- `docker-compose.yml` - Development environment setup
- `docker-compose.prod.yml` - Production environment setup
- `docker-entrypoint.sh` - Container startup script
- `docker-prebuild.sh` - Pre-build setup script
- `build_clean.sh` - Clean build artifacts
- `clean_volumes.sh` - Clean Docker volumes
- `dockerfileplan.md` - Docker implementation planning notes

## Usage

From the project root directory:

```bash
# Development
docker-compose -f docker/docker-compose.yml up

# Production
docker-compose -f docker/docker-compose.prod.yml up

# Build from Dockerfile
docker build -f docker/Dockerfile -t stock-sentiment-analysis .
```

## Notes

All Docker commands should be run from the project root directory, referencing the files in this docker/ folder.