#!/bin/bash
set -e

# Docker Entry Point Script for Stock Sentiment Analysis System
# Supports multiple execution modes for flexibility

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check required environment variables
check_env_vars() {
    local missing_vars=()
    
    # Check for required API keys
    if [[ -z "${NEWS_API_KEY:-}" ]]; then
        missing_vars+=("NEWS_API_KEY")
    fi
    
    if [[ -z "${REDDIT_CLIENT_ID:-}" ]]; then
        missing_vars+=("REDDIT_CLIENT_ID")
    fi
    
    if [[ -z "${REDDIT_CLIENT_SECRET:-}" ]]; then
        missing_vars+=("REDDIT_CLIENT_SECRET")
    fi
    
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
        missing_vars+=("OPENAI_API_KEY")
    fi
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        print_error "Missing required environment variables:"
        for var in "${missing_vars[@]}"; do
            echo "  - $var"
        done
        print_info "Please set these environment variables or mount a .env file"
        print_info "Example: docker run --env-file .env stock-sentiment-analyzer"
        return 1
    fi
    
    return 0
}

# Function to install Playwright browsers if needed
install_playwright_browsers() {
    print_info "Checking Playwright browser installation..."
    
    cd /app/playwright_scrapers
    
    # Test if browsers are already installed
    if node -e "const { chromium } = require('playwright'); chromium.launch({ headless: true }).then(() => console.log('OK')).catch(() => process.exit(1))" &> /dev/null; then
        print_success "Playwright browsers already installed"
        return 0
    fi
    
    print_info "Installing Playwright browsers..."
    if ./node_modules/.bin/playwright install chromium; then
        print_success "Playwright browsers installed successfully"
        return 0
    else
        print_error "Failed to install Playwright browsers"
        return 1
    fi
}

# Function to check system dependencies
check_dependencies() {
    print_info "Checking system dependencies..."
    
    # Check Chrome
    if ! command -v chromium &> /dev/null; then
        print_error "Chromium not found"
        return 1
    fi
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        print_error "Node.js not found"
        return 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 not found"
        return 1
    fi
    
    # Check key Python packages
    if ! python3 -c "import requests, pandas, numpy, openai, selenium, praw, transformers" &> /dev/null; then
        print_error "Key Python packages not available"
        return 1
    fi
    
    # Install Playwright browsers if needed
    if ! install_playwright_browsers; then
        return 1
    fi
    
    print_success "All dependencies available"
    return 0
}

# Function to check file permissions
check_permissions() {
    print_info "Checking file permissions..."
    
    # Check container_output directory
    if [[ ! -w "/app/container_output" ]]; then
        print_error "Container output directory not writable"
        return 1
    fi
    
    # Check logs directory
    if [[ ! -w "/app/logs" ]]; then
        print_error "Logs directory not writable"
        return 1
    fi
    
    print_success "File permissions OK"
    return 0
}

# Function to run health checks
run_health_checks() {
    print_info "Running health checks..."
    
    if ! check_dependencies; then
        return 1
    fi
    
    if ! check_permissions; then
        return 1
    fi
    
    # Test Chrome functionality
    if ! chromium --version &> /dev/null; then
        print_error "Chrome health check failed"
        return 1
    fi
    
    print_success "All health checks passed"
    return 0
}

# Function to display help
show_help() {
    cat << EOF
Stock Sentiment Analysis System - Docker Container

Usage: docker run [OPTIONS] stock-sentiment-analyzer [COMMAND]

Commands:
  automated    Run complete sentiment analysis pipeline (default)
  news         Run news sentiment analysis only
  images       Run image sentiment analysis only  
  tradingview  Run TradingView scraper and analysis
  test         Run system tests and health checks
  test-all     Run comprehensive test suite for all components
  shell        Start interactive bash shell
  help         Show this help message

Examples:
  # Run complete pipeline
  docker run --env-file .env stock-sentiment-analyzer

  # Run with custom options
  docker run --env-file .env stock-sentiment-analyzer automated --force-refresh

  # Run individual components
  docker run --env-file .env stock-sentiment-analyzer news
  docker run --env-file .env stock-sentiment-analyzer tradingview

  # Interactive debugging
  docker run -it --env-file .env stock-sentiment-analyzer shell

  # Test system health
  docker run --env-file .env stock-sentiment-analyzer test

Environment Variables:
  NEWS_API_KEY         - NewsAPI key (required)
  REDDIT_CLIENT_ID     - Reddit client ID (required)
  REDDIT_CLIENT_SECRET - Reddit client secret (required)
  OPENAI_API_KEY       - OpenAI API key (required)

Volume Mounts:
  /app/container_output  - Analysis results and data
  /app/logs              - Application logs

EOF
}

# Main execution logic
main() {
    local command="${1:-automated}"
    
    print_info "Starting Stock Sentiment Analysis System..."
    print_info "Command: $command"
    
    # Handle help command
    if [[ "$command" == "help" ]]; then
        show_help
        exit 0
    fi
    
    # Handle test command
    if [[ "$command" == "test" ]]; then
        print_info "Running system tests..."
        if run_health_checks; then
            print_success "All tests passed!"
            exit 0
        else
            print_error "Tests failed!"
            exit 1
        fi
    fi
    
    # Handle test-all command
    if [[ "$command" == "test-all" ]]; then
        print_info "Running comprehensive test suite..."
        exec /usr/local/bin/test_all.sh
    fi
    
    # Handle shell command
    if [[ "$command" == "shell" ]]; then
        print_info "Starting interactive shell..."
        # Check if we're in interactive mode
        if [[ -t 0 ]]; then
            exec /bin/bash
        else
            # Non-interactive mode - just keep container running
            print_info "Container running in background mode. Use 'docker exec -it stock-sentiment-app bash' to access shell."
            exec tail -f /dev/null
        fi
    fi
    
    # For all other commands, check environment and dependencies
    if ! check_env_vars; then
        print_error "Environment check failed"
        exit 1
    fi
    
    if ! run_health_checks; then
        print_error "Health check failed"
        exit 1
    fi
    
    print_success "System initialized successfully"
    
    # Execute the requested command
    case "$command" in
        "automated")
            print_info "Running automated sentiment analysis pipeline..."
            cd /app && exec python3 news/weighted_sentiment_aggregator.py --force-refresh
            ;;
        "news")
            print_info "Running news sentiment analysis..."
            cd /app && exec python3 news/sentiment_analyzer.py
            ;;
        "images")
            print_info "Running image sentiment analysis..."
            cd /app && exec python3 news/image_sentiment_analyzer.py
            ;;
        "tradingview")
            print_info "Running TradingView scraper and analysis..."
            cd /app && python3 news/tradingview_scraper.py
            cd /app && exec python3 news/tradingview_sentiment_analyzer.py --latest
            ;;
        "charts")
            print_info "Generating TradingView charts..."
            cd /app && exec python3 utils/headless_charts.py
            ;;
        "main")
            print_info "Running main trading system..."
            cd /app && exec python3 main.py
            ;;
        *)
            print_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"