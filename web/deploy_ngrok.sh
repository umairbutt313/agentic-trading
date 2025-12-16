#!/bin/bash
# Deploy NVIDIA chart to internet using ngrok

echo "Setting up internet access via ngrok..."

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo "Installing ngrok..."
    # For macOS with Homebrew
    if command -v brew &> /dev/null; then
        brew install ngrok/ngrok/ngrok
    else
        echo "Please install ngrok from https://ngrok.com/download"
        exit 1
    fi
fi

echo "Starting local server..."
python3 server.py &
SERVER_PID=$!

sleep 3

echo "Exposing to internet via ngrok..."
ngrok http 8080 --domain=stock.stk.name

# Kill server when ngrok stops
kill $SERVER_PID