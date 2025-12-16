#!/usr/bin/env python3
"""
HTTPS server for multi-stock chart viewer with SSL/TLS support
Serves the HTML file on configurable ports with environment variable support
"""

import http.server
import socketserver
import ssl
import os
import sys
import threading
from dotenv import load_dotenv

# Change to the directory containing the HTML file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, '.env')
load_dotenv(env_path)

# Configuration from environment variables with defaults
HTTP_PORT = int(os.getenv('HTTP_PORT', '8080'))
HTTPS_PORT = int(os.getenv('HTTPS_PORT', '8443'))
HOST = os.getenv('SERVER_HOST', '0.0.0.0')  # Allow external connections
DOMAIN_NAME = os.getenv('DOMAIN_NAME', 'stock.stk.name')

# SSL certificate paths (configurable via environment)
CERT_PATH = os.getenv('SSL_CERT_PATH', f'/etc/letsencrypt/live/{DOMAIN_NAME}/fullchain.pem')
KEY_PATH = os.getenv('SSL_KEY_PATH', f'/etc/letsencrypt/live/{DOMAIN_NAME}/privkey.pem')

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers for external access
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_GET(self):
        # Handle API endpoints for stock data
        if self.path == '/api/data':
            self.serve_csv_data('nvidia')
            return
        elif self.path == '/api/data/nvidia':
            self.serve_csv_data('nvidia')  
            return
        elif self.path == '/api/data/apple':
            self.serve_csv_data('apple')
            return
        elif self.path == '/api/data/intel':
            self.serve_csv_data('intel')
            return
        
        # Serve the multi-stock chart viewer for root path
        if self.path == '/' or self.path == '/index.html':
            self.path = '/multi_stock_chart_viewer.html'
        super().do_GET()
    
    def serve_csv_data(self, stock='nvidia'):
        """Serve live CSV data from the actual data file for specified stock"""
        try:
            # Map stock names to file names
            stock_files = {
                'nvidia': 'nvidia_score_price_dump.txt',
                'apple': 'apple_score_price_dump.txt', 
                'intel': 'intel_score_price_dump.txt'
            }
            
            if stock not in stock_files:
                raise ValueError(f"Unknown stock: {stock}")
            
            filename = stock_files[stock]
            
            # Try multiple possible locations for the data file (prioritize web folder)
            possible_paths = [
                filename,  # Current web folder (highest priority)
                f'../news/{filename}',  # Fallback to news folder
                f'../../news/{filename}',
                f'/Users/uab/Desktop/Arsalan-Project/chartfunction1/stocks/web/{filename}',
                f'/Users/uab/Desktop/Arsalan-Project/chartfunction1/stocks/news/{filename}'
            ]
            
            csv_data = None
            for file_path in possible_paths:
                try:
                    if os.path.exists(file_path):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            csv_data = f.read()
                        break
                except Exception:
                    continue
            
            if csv_data is None:
                raise FileNotFoundError(f"CSV data file not found for {stock}")
            
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.end_headers()
            self.wfile.write(csv_data.encode('utf-8'))
            
        except Exception as e:
            print(f"Error serving CSV data for {stock}: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            error_response = f'{{"error": "Unable to load data file for {stock}"}}'
            self.wfile.write(error_response.encode('utf-8'))

def start_http_server():
    """Start HTTP server on port 8080"""
    with socketserver.TCPServer((HOST, HTTP_PORT), CustomHTTPRequestHandler) as httpd:
        print(f"HTTP Server running at http://{HOST}:{HTTP_PORT}/")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass

def start_https_server():
    """Start HTTPS server on port 8443"""
    try:
        # Create SSL context
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(CERT_PATH, KEY_PATH)
        
        with socketserver.TCPServer((HOST, HTTPS_PORT), CustomHTTPRequestHandler) as httpd:
            httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
            print(f"HTTPS Server running at https://{HOST}:{HTTPS_PORT}/")
            print(f"SSL Certificate: {CERT_PATH}")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                pass
    except FileNotFoundError:
        print(f"SSL certificate files not found at {CERT_PATH} or {KEY_PATH}")
        print("Running HTTP server only.")
        return False
    except Exception as e:
        print(f"Error starting HTTPS server: {e}")
        print("Running HTTP server only.")
        return False
    return True

if __name__ == "__main__":
    print("Starting dual HTTP/HTTPS server for NVIDIA Chart Viewer...")
    print(f"Access HTTP at: http://stock.stk.name:{HTTP_PORT}/")
    print(f"Access HTTPS at: https://stock.stk.name:{HTTPS_PORT}/")
    print("Press Ctrl+C to stop both servers")
    
    # Start HTTPS server in a separate thread
    https_started = False
    try:
        https_thread = threading.Thread(target=start_https_server, daemon=True)
        https_thread.start()
        https_started = True
    except Exception as e:
        print(f"Could not start HTTPS server: {e}")
    
    # Start HTTP server in main thread
    try:
        start_http_server()
    except KeyboardInterrupt:
        print("\nServers stopped.")
        sys.exit(0)