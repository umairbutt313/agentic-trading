# Multi-stage Dockerfile for Stock Sentiment Analysis System
# Optimized for production with Python + Node.js + Playwright
# UPDATED: Includes Playwright migration and all recent changes

# Build argument to trigger host volume cleaning
ARG CLEAN_VOLUMES=true

# ================================
# Stage 1: Base System Setup
# ================================
FROM python:3.11-slim AS base

# Set environment variables (let Playwright manage browser paths)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CHROME_BIN=/usr/bin/chromium \
    PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=0 \
    DOCKER_CONTAINER=true \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Chrome
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools for pandas-ta
    gcc \
    g++ \
    make \
    build-essential \
    # Chrome and dependencies for Playwright/Selenium
    chromium \
    chromium-driver \
    # Additional Chrome dependencies
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libc6 \
    libcairo2 \
    libcups2 \
    libdbus-1-3 \
    libexpat1 \
    libfontconfig1 \
    libgbm1 \
    libgcc1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libstdc++6 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxrandr2 \
    libxrender1 \
    libxss1 \
    libxtst6 \
    # Playwright additional dependencies
    libwoff1 \
    libopus0 \
    libwebp7 \
    libwebpdemux2 \
    libenchant-2-2 \
    libgudev-1.0-0 \
    libsecret-1-0 \
    libhyphen0 \
    libgdk-pixbuf2.0-0 \
    libegl1 \
    libnotify4 \
    # Utilities
    curl \
    wget \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Verify Chrome installation
RUN chromium --version

# ================================
# Stage 2: Node.js + Playwright Environment
# ================================
FROM base AS nodejs

# Install Node.js from official repository (most reliable)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_18.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Verify Node.js installation
RUN node --version && npm --version

# Set working directory
WORKDIR /app

# Note: Playwright will manage browser installation automatically

# Copy package.json and install Node.js dependencies
COPY playwright_scrapers/package*.json ./playwright_scrapers/
RUN cd playwright_scrapers && npm ci --omit=dev \
    && npm cache clean --force

# Install Playwright browsers (minimal, headless-only)
RUN cd playwright_scrapers && \
    echo "Installing Playwright browsers (headless chromium only)..." && \
    ./node_modules/.bin/playwright install chromium && \
    echo "Cleaning up unnecessary browser files..." && \
    find ~/.cache/ms-playwright -name "*.so" -size +50M -delete 2>/dev/null || true && \
    echo "Browser installation completed"

# Verify Playwright installation
RUN cd playwright_scrapers && \
    node -e "console.log('Playwright version:', require('./node_modules/playwright/package.json').version)"

# ================================
# Stage 3: Python Dependencies
# ================================
FROM nodejs AS python-deps

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Install Python dependencies with multiple fallback strategies
RUN pip install --no-cache-dir -r requirements.txt || \
    (echo "First install attempt failed, trying individual packages..." && \
     pip install --no-cache-dir transformers pandas numpy scipy yfinance torch selenium praw schedule requests beautifulsoup4 pyyaml python-dotenv openai aiofiles) && \
    # Try pandas-ta with multiple fallback strategies
    (pip install --no-cache-dir pandas-ta || \
     pip install --no-cache-dir git+https://github.com/twopirllc/pandas-ta || \
     pip install --no-cache-dir ta-lib || \
     echo "WARNING: pandas-ta installation failed, technical analysis features may be limited") && \
    # Verify critical packages are installed
    python -c "import aiofiles; print('✅ aiofiles installed successfully')" && \
    python -c "import pandas; print('✅ pandas installed successfully')" && \
    python -c "import requests; print('✅ requests installed successfully')" && \
    python -c "import openai; print('✅ openai installed successfully')" && \
    python -c "import selenium; print('✅ selenium installed successfully')" && \
    python -c "import praw; print('✅ praw installed successfully')" && \
    python -c "import transformers; print('✅ transformers installed successfully')"

# ================================
# Stage 4: Application Runtime
# ================================
FROM python-deps AS runtime

# Create application user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create required directories
RUN mkdir -p /app/container_output /app/logs \
    && mkdir -p /app/container_output/news /app/container_output/tradingview /app/container_output/images /app/container_output/final_score /app/container_output/charts

# Copy application code
COPY --chown=appuser:appuser . .

# Ensure logs and container_output folders are always empty on build
RUN rm -rf /app/logs/* && mkdir -p /app/logs && \
    rm -rf /app/container_output/* && mkdir -p /app/container_output/news /app/container_output/tradingview /app/container_output/images /app/container_output/final_score /app/container_output/charts

# Copy Node.js dependencies from previous stage
COPY --from=nodejs --chown=appuser:appuser /app/playwright_scrapers/node_modules ./playwright_scrapers/node_modules

# Setup Playwright browser directory (browsers will be installed on first run)
RUN mkdir -p /home/appuser/.cache/ms-playwright

# Copy and make scripts executable
COPY --chown=appuser:appuser docker-entrypoint.sh /usr/local/bin/
COPY --chown=appuser:appuser test_all.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh && \
    chmod +x /usr/local/bin/test_all.sh

# Set correct permissions
RUN chown -R appuser:appuser /app \
    && chmod -R 755 /app \
    && chmod -R 777 /app/container_output /app/logs \
    && chown -R appuser:appuser /home/appuser/.cache

# Switch to non-root user
USER appuser

# Set working directory
WORKDIR /app

# Expose port (if needed for future API endpoints)
EXPOSE 8000

# Health check - Updated to include Playwright
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD chromium --version && python3 -c "import requests; print('Health check passed')" && \
    cd playwright_scrapers && node -e "const { chromium } = require('playwright'); console.log('Playwright OK')" || exit 1

# Set default entry point
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Default command - run automated sentiment analysis pipeline
CMD ["automated"]