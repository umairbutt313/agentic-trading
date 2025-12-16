#!/bin/bash
# Setup script for stock.stk.name domain mapping

echo "Setting up stock.stk.name domain mapping..."

# Check if running as root for /etc/hosts modification
if [[ $EUID -eq 0 ]]; then
    echo "127.0.0.1 stock.stk.name" >> /etc/hosts
    echo "Domain mapping added to /etc/hosts"
else
    echo "Adding domain mapping to /etc/hosts (requires sudo):"
    echo "127.0.0.1 stock.stk.name" | sudo tee -a /etc/hosts
fi

echo "Domain setup complete!"
echo "Start the server with: python3 server.py"
echo "Then access: http://stock.stk.name:8080/"