#!/usr/bin/env python3
"""
Production HTTP/WebSocket server:
- HTTPS support with SSL certificates
- WebSocket broadcasting for real-time updates
- Static file serving with caching headers
- API endpoints for data access
- Authentication middleware (optional)
- Rate limiting and DDoS protection
"""

import asyncio
import json
import logging
import os
import ssl
import mimetypes
import gzip
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading
import time
from collections import defaultdict, deque
import urllib.parse
import hashlib

# HTTP Server imports
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
from io import BytesIO

# WebSocket imports
import websockets

# Import our modules
from price_storage import PriceStorage
from unified_data_service import UnifiedDataService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ProductionWebServer')

class RateLimiter:
    """Simple rate limiter for DDoS protection"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.clients = defaultdict(lambda: deque())
        
    def is_allowed(self, client_ip: str) -> bool:
        """Check if client is within rate limits"""
        now = time.time()
        client_requests = self.clients[client_ip]
        
        # Remove old requests outside the window
        while client_requests and client_requests[0] < now - self.window_seconds:
            client_requests.popleft()
        
        # Check if under limit
        if len(client_requests) >= self.max_requests:
            return False
        
        # Add current request
        client_requests.append(now)
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics"""
        now = time.time()
        active_clients = 0
        total_requests = 0
        
        for client_ip, requests in self.clients.items():
            # Clean old requests
            while requests and requests[0] < now - self.window_seconds:
                requests.popleft()
            
            if requests:
                active_clients += 1
                total_requests += len(requests)
        
        return {
            'active_clients': active_clients,
            'total_recent_requests': total_requests,
            'window_seconds': self.window_seconds,
            'max_requests_per_client': self.max_requests
        }

class ProductionWebHandler(BaseHTTPRequestHandler):
    """Enhanced HTTP handler with security and performance features"""
    
    def __init__(self, data_service: UnifiedDataService, rate_limiter: RateLimiter, *args, **kwargs):
        self.data_service = data_service
        self.rate_limiter = rate_limiter
        self.web_root = Path(__file__).parent / 'web'
        super().__init__(*args, **kwargs)
    
    def version_string(self):
        """Override server version for security"""
        return "ProductionWebServer/1.0"
    
    def do_GET(self):
        """Handle GET requests with security and caching"""
        try:
            # Get client IP
            client_ip = self.get_client_ip()
            
            # Rate limiting check
            if not self.rate_limiter.is_allowed(client_ip):
                self.send_error(429, "Too Many Requests")
                return
            
            # Parse URL
            parsed_url = urllib.parse.urlparse(self.path)
            path = parsed_url.path
            query_params = urllib.parse.parse_qs(parsed_url.query)
            
            # Security headers
            self.send_security_headers()
            
            # Handle different endpoints
            if path.startswith('/api/'):
                self.handle_api_request(path, query_params)
            elif path == '/' or path == '/dashboard':
                self.serve_file('multi_stock_chart_viewer.html')
            elif path.startswith('/web/'):
                # Serve files from web directory
                file_path = path[5:]  # Remove /web/ prefix
                self.serve_file(file_path)
            else:
                # Try to serve static files directly
                self.serve_file(path.lstrip('/'))
                
        except Exception as e:
            logger.error(f"Request handler error: {e}")
            self.send_error(500, "Internal Server Error")
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()
    
    def get_client_ip(self) -> str:
        """Get client IP address"""
        # Check for forwarded headers first (reverse proxy)
        forwarded_for = self.headers.get('X-Forwarded-For')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = self.headers.get('X-Real-IP')
        if real_ip:
            return real_ip
        
        # Fallback to client address
        return self.client_address[0]
    
    def send_security_headers(self):
        """Send security headers"""
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('X-XSS-Protection', '1; mode=block')
        self.send_header('Referrer-Policy', 'strict-origin-when-cross-origin')
        self.send_header('Content-Security-Policy', 
                        "default-src 'self' 'unsafe-inline' 'unsafe-eval' https://unpkg.com; "
                        "connect-src 'self' ws: wss:; "
                        "img-src 'self' data:;")
    
    def send_cors_headers(self):
        """Send CORS headers"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    
    def handle_api_request(self, path: str, query_params: Dict):
        """Handle API endpoints"""
        path_parts = path.strip('/').split('/')
        
        if len(path_parts) >= 3 and path_parts[1] == 'unified':
            # API endpoint: /api/unified/{symbol}
            symbol = path_parts[2].upper()
            hours_back = int(query_params.get('hours', [1])[0])
            compress = query_params.get('compress', ['false'])[0].lower() == 'true'
            
            data = self.data_service.get_unified_data(symbol, hours_back)
            self.send_json_response(data, compress=compress)
            
        elif len(path_parts) >= 2 and path_parts[1] == 'symbols':
            # API endpoint: /api/symbols
            symbols_data = {
                'symbols': self.data_service.symbols,
                'last_update': datetime.now().isoformat()
            }
            self.send_json_response(symbols_data)
            
        elif len(path_parts) >= 2 and path_parts[1] == 'health':
            # API endpoint: /api/health
            health_data = {
                'status': 'healthy',
                'service': 'production_web_server',
                'cache_size': self.data_service.cache.size(),
                'rate_limiting': self.rate_limiter.get_stats(),
                'timestamp': datetime.now().isoformat()
            }
            self.send_json_response(health_data)
            
        elif len(path_parts) >= 2 and path_parts[1] == 'stats':
            # API endpoint: /api/stats
            storage = PriceStorage()
            stats_data = storage.get_statistics()
            self.send_json_response(stats_data)
            
        else:
            # 404 Not Found
            self.send_error(404, "API endpoint not found")
    
    def serve_file(self, file_path: str):
        """Serve static files with caching headers"""
        try:
            # Security check - prevent path traversal
            if '..' in file_path or file_path.startswith('/'):
                self.send_error(403, "Forbidden")
                return
            
            # Map common paths
            if not file_path or file_path == 'index.html':
                file_path = 'multi_stock_chart_viewer.html'
            
            full_path = self.web_root / file_path
            
            # Check if file exists
            if not full_path.exists() or not full_path.is_file():
                # Try without extension for data files
                txt_path = self.web_root / f"{file_path}.txt"
                if txt_path.exists():
                    full_path = txt_path
                else:
                    self.send_error(404, "File Not Found")
                    return
            
            # Get file stats for caching
            stat_result = full_path.stat()
            file_size = stat_result.st_size
            last_modified = datetime.fromtimestamp(stat_result.st_mtime)
            
            # Generate ETag
            etag = hashlib.md5(f"{full_path}{stat_result.st_mtime}".encode()).hexdigest()
            
            # Check if client has cached version
            client_etag = self.headers.get('If-None-Match')
            if client_etag and client_etag.strip('"') == etag:
                self.send_response(304)
                self.end_headers()
                return
            
            # Read and serve file
            with open(full_path, 'rb') as f:
                content = f.read()
            
            # Determine content type
            content_type, _ = mimetypes.guess_type(str(full_path))
            if content_type is None:
                if full_path.suffix == '.txt':
                    content_type = 'text/plain'
                else:
                    content_type = 'application/octet-stream'
            
            # Check if content should be compressed
            compress = (content_type.startswith('text/') or 
                       content_type.startswith('application/json') or
                       content_type.startswith('application/javascript')) and len(content) > 1000
            
            if compress and 'gzip' in self.headers.get('Accept-Encoding', ''):
                content = gzip.compress(content)
                self.send_header('Content-Encoding', 'gzip')
            
            # Send response
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('ETag', f'"{etag}"')
            self.send_header('Last-Modified', last_modified.strftime('%a, %d %b %Y %H:%M:%S GMT'))
            
            # Caching headers
            if file_path.endswith(('.js', '.css', '.png', '.jpg', '.jpeg', '.gif', '.ico')):
                # Static assets - cache for 1 hour
                self.send_header('Cache-Control', 'public, max-age=3600')
            else:
                # HTML and data files - cache for 5 minutes
                self.send_header('Cache-Control', 'public, max-age=300')
            
            self.send_cors_headers()
            self.end_headers()
            
            # Send content
            self.wfile.write(content)
            
            logger.debug(f"Served {file_path} ({len(content)} bytes, {content_type})")
            
        except Exception as e:
            logger.error(f"Error serving file {file_path}: {e}")
            self.send_error(500, "Internal Server Error")
    
    def send_json_response(self, data: Dict, compress: bool = False):
        """Send JSON response with optional compression"""
        try:
            json_str = json.dumps(data, indent=2 if not compress else None)
            content = json_str.encode('utf-8')
            
            # Compress if requested and beneficial
            if compress and len(content) > 1000 and 'gzip' in self.headers.get('Accept-Encoding', ''):
                content = gzip.compress(content)
                self.send_header('Content-Encoding', 'gzip')
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(content)))
            self.send_cors_headers()
            self.end_headers()
            
            self.wfile.write(content)
            
        except Exception as e:
            logger.error(f"Error sending JSON response: {e}")
            self.send_error(500, "Internal Server Error")
    
    def log_message(self, format, *args):
        """Custom log message format"""
        client_ip = self.get_client_ip()
        logger.info(f"{client_ip} - {format % args}")

class ProductionWebServer:
    """Main production web server with HTTPS and WebSocket support"""
    
    def __init__(self, port: int = 8080, ssl_port: int = 8443):
        self.port = port
        self.ssl_port = ssl_port
        self.data_service = UnifiedDataService(port=8090)
        self.rate_limiter = RateLimiter(max_requests=200, window_seconds=60)
        
        # SSL configuration
        self.ssl_context = None
        self.setup_ssl()
        
    def setup_ssl(self):
        """Setup SSL context if certificates are available"""
        cert_file = 'ssl/server.crt'
        key_file = 'ssl/server.key'
        
        if os.path.exists(cert_file) and os.path.exists(key_file):
            try:
                self.ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                self.ssl_context.load_cert_chain(cert_file, key_file)
                logger.info("SSL certificates loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load SSL certificates: {e}")
                self.ssl_context = None
        else:
            logger.info("SSL certificates not found - HTTPS disabled")
    
    def create_handler_class(self):
        """Create handler class with dependency injection"""
        data_service = self.data_service
        rate_limiter = self.rate_limiter
        
        class Handler(ProductionWebHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(data_service, rate_limiter, *args, **kwargs)
        
        return Handler
    
    def start_http_server(self):
        """Start HTTP server"""
        try:
            handler_class = self.create_handler_class()
            httpd = HTTPServer(('', self.port), handler_class)
            
            logger.info(f"HTTP server starting on port {self.port}")
            logger.info(f"Dashboard URL: http://localhost:{self.port}/")
            logger.info(f"API endpoints: http://localhost:{self.port}/api/")
            
            httpd.serve_forever()
            
        except Exception as e:
            logger.error(f"HTTP server error: {e}")
    
    def start_https_server(self):
        """Start HTTPS server"""
        if not self.ssl_context:
            logger.info("HTTPS server not started - SSL not configured")
            return
        
        try:
            handler_class = self.create_handler_class()
            httpd = HTTPServer(('', self.ssl_port), handler_class)
            httpd.socket = self.ssl_context.wrap_socket(httpd.socket, server_side=True)
            
            logger.info(f"HTTPS server starting on port {self.ssl_port}")
            logger.info(f"Secure dashboard URL: https://localhost:{self.ssl_port}/")
            
            httpd.serve_forever()
            
        except Exception as e:
            logger.error(f"HTTPS server error: {e}")
    
    async def start_unified_data_service(self):
        """Start the unified data service in background"""
        try:
            self.data_service.running = True
            logger.info("Starting unified data service...")
            
            # Import and run the unified data service
            from unified_data_service import main as unified_main
            await unified_main()
            
        except Exception as e:
            logger.error(f"Unified data service error: {e}")
    
    def start(self):
        """Start all services"""
        logger.info("🚀 Starting Production Web Server")
        logger.info("=" * 60)
        logger.info(f"HTTP Port: {self.port}")
        logger.info(f"HTTPS Port: {self.ssl_port} {'(enabled)' if self.ssl_context else '(disabled)'}")
        logger.info(f"Rate Limiting: {self.rate_limiter.max_requests} requests/{self.rate_limiter.window_seconds}s per IP")
        logger.info(f"WebRoot: {Path(__file__).parent / 'web'}")
        
        try:
            # Start unified data service in background thread
            data_service_thread = threading.Thread(
                target=lambda: asyncio.run(self.start_unified_data_service()),
                daemon=True
            )
            data_service_thread.start()
            
            # Start HTTPS server in background thread if available
            if self.ssl_context:
                https_thread = threading.Thread(target=self.start_https_server, daemon=True)
                https_thread.start()
            
            # Start HTTP server in main thread
            self.start_http_server()
            
        except KeyboardInterrupt:
            logger.info("Shutting down Production Web Server")
            self.data_service.running = False

def generate_ssl_certificates():
    """Generate self-signed SSL certificates for testing"""
    try:
        import subprocess
        
        # Create ssl directory
        os.makedirs('ssl', exist_ok=True)
        
        # Generate self-signed certificate
        cmd = [
            'openssl', 'req', '-x509', '-newkey', 'rsa:4096', '-keyout', 'ssl/server.key',
            '-out', 'ssl/server.crt', '-days', '365', '-nodes',
            '-subj', '/C=US/ST=State/L=City/O=Organization/CN=localhost'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Self-signed SSL certificates generated successfully")
            return True
        else:
            logger.warning(f"Failed to generate SSL certificates: {result.stderr}")
            return False
            
    except FileNotFoundError:
        logger.warning("OpenSSL not found - SSL certificates not generated")
        return False

def main():
    """Main entry point"""
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Generate SSL certificates if not present
    if not os.path.exists('ssl/server.crt'):
        generate_ssl_certificates()
    
    # Create and start server
    server = ProductionWebServer(port=8080, ssl_port=8443)
    server.start()

if __name__ == "__main__":
    main()