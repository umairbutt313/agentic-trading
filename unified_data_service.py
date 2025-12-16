#!/usr/bin/env python3
"""
Unified Data Service
Single endpoint for price + sentiment data:
- REST API: GET /api/unified/{symbol}
- WebSocket: ws://localhost:8090/stream/{symbol}
- Data format: {timestamp, price, news_sentiment, tradingview_sentiment}
- Caching with simple in-memory cache for performance
- Data compression for bandwidth efficiency
"""

import asyncio
import json
import logging
import os
import gzip
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading
import time
from collections import defaultdict

# HTTP and WebSocket imports
from http.server import HTTPServer, BaseHTTPRequestHandler
import websockets
import urllib.parse

# Import our modules
from price_storage import PriceStorage
from data_aggregator import SentimentDataAggregator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('UnifiedDataService')

class DataCache:
    """Simple in-memory cache for data with TTL"""
    
    def __init__(self, default_ttl: int = 60):
        self.cache = {}
        self.timestamps = {}
        self.default_ttl = default_ttl
        
    def get(self, key: str) -> Optional[Any]:
        """Get cached data if not expired"""
        if key not in self.cache:
            return None
        
        # Check if expired
        if time.time() - self.timestamps[key] > self.default_ttl:
            del self.cache[key]
            del self.timestamps[key]
            return None
        
        return self.cache[key]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached data with TTL"""
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def clear(self) -> None:
        """Clear all cached data"""
        self.cache.clear()
        self.timestamps.clear()
    
    def size(self) -> int:
        """Get cache size"""
        return len(self.cache)

class UnifiedDataService:
    """Main unified data service"""
    
    def __init__(self, port: int = 8090):
        self.port = port
        self.price_storage = PriceStorage()
        self.sentiment_aggregator = SentimentDataAggregator()
        self.cache = DataCache(default_ttl=30)  # 30 second cache
        
        # WebSocket clients tracking
        self.websocket_clients = defaultdict(set)  # symbol -> set of websockets
        self.running = False
        
        # Company symbol mapping
        self.symbols = ['NVDA', 'AAPL', 'INTC', 'MSFT', 'GOOG', 'TSLA']
        
    def get_unified_data(self, symbol: str, hours_back: int = 1) -> Dict[str, Any]:
        """Get unified price + sentiment data for a symbol"""
        cache_key = f"unified_{symbol}_{hours_back}"
        
        # Try cache first
        cached_data = self.cache.get(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # Get price history
            price_history = self.price_storage.get_price_history(symbol, hours_back=hours_back)
            
            if not price_history:
                logger.warning(f"No price data available for {symbol}")
                return {
                    'symbol': symbol,
                    'status': 'no_data',
                    'data_points': [],
                    'last_update': datetime.now().isoformat()
                }
            
            # Get latest sentiment data
            sentiment_data = self._get_sentiment_data(symbol)
            
            # Combine price and sentiment data
            data_points = []
            for price_point in price_history:
                data_points.append({
                    'timestamp': price_point['timestamp'],
                    'price': float(price_point['price']),
                    'news_sentiment': sentiment_data['news_sentiment'],
                    'tradingview_sentiment': sentiment_data['tradingview_sentiment']
                })
            
            unified_data = {
                'symbol': symbol,
                'status': 'success',
                'data_points': data_points,
                'sentiment_summary': sentiment_data,
                'stats': {
                    'total_points': len(data_points),
                    'time_range_hours': hours_back,
                    'latest_price': data_points[-1]['price'] if data_points else None,
                    'price_change_24h': self._calculate_price_change(data_points)
                },
                'last_update': datetime.now().isoformat()
            }
            
            # Cache the result
            self.cache.set(cache_key, unified_data)
            
            return unified_data
            
        except Exception as e:
            logger.error(f"Error getting unified data for {symbol}: {e}")
            return {
                'symbol': symbol,
                'status': 'error',
                'error': str(e),
                'data_points': [],
                'last_update': datetime.now().isoformat()
            }
    
    def _get_sentiment_data(self, symbol: str) -> Dict[str, float]:
        """Get current sentiment data for a symbol"""
        try:
            # Load sentiment data sources
            self.sentiment_aggregator.news_data = self.sentiment_aggregator.load_news_sentiment()
            self.sentiment_aggregator.tradingview_data = self.sentiment_aggregator.load_tradingview_sentiment()
            
            # Map symbol to company name
            company_map = {v: k for k, v in self.sentiment_aggregator.company_symbols.items()}
            company = company_map.get(symbol, symbol)
            
            # Get sentiment scores
            news_sentiment = self.sentiment_aggregator.news_data.get(company, {}).get('score', 5.0)
            tradingview_sentiment = self.sentiment_aggregator.tradingview_data.get(company, {}).get('score', 5.0)
            
            return {
                'news_sentiment': float(news_sentiment),
                'tradingview_sentiment': float(tradingview_sentiment),
                'overall_sentiment': (float(news_sentiment) + float(tradingview_sentiment)) / 2
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment data for {symbol}: {e}")
            return {
                'news_sentiment': 5.0,
                'tradingview_sentiment': 5.0,
                'overall_sentiment': 5.0
            }
    
    def _calculate_price_change(self, data_points: List[Dict]) -> Optional[float]:
        """Calculate 24h price change percentage"""
        if len(data_points) < 2:
            return None
        
        try:
            current_price = data_points[-1]['price']
            old_price = data_points[0]['price']
            return ((current_price - old_price) / old_price) * 100
        except (KeyError, ZeroDivisionError):
            return None
    
    def compress_data(self, data: Dict) -> bytes:
        """Compress JSON data using gzip"""
        json_str = json.dumps(data)
        return gzip.compress(json_str.encode('utf-8'))
    
    def decompress_data(self, compressed_data: bytes) -> Dict:
        """Decompress gzip data to JSON"""
        json_str = gzip.decompress(compressed_data).decode('utf-8')
        return json.loads(json_str)

class UnifiedDataHTTPHandler(BaseHTTPRequestHandler):
    """HTTP handler for REST API endpoints"""
    
    def __init__(self, data_service: UnifiedDataService, *args, **kwargs):
        self.data_service = data_service
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        try:
            # Parse URL
            parsed_url = urllib.parse.urlparse(self.path)
            path_parts = parsed_url.path.strip('/').split('/')
            
            # CORS headers
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            
            # Handle different endpoints
            if len(path_parts) >= 3 and path_parts[0] == 'api' and path_parts[1] == 'unified':
                # GET /api/unified/{symbol}
                symbol = path_parts[2].upper()
                
                # Parse query parameters
                query_params = urllib.parse.parse_qs(parsed_url.query)
                hours_back = int(query_params.get('hours', [1])[0])
                compress = query_params.get('compress', ['false'])[0].lower() == 'true'
                
                # Get unified data
                data = self.data_service.get_unified_data(symbol, hours_back)
                
                # Send response
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                
                if compress:
                    compressed_data = self.data_service.compress_data(data)
                    self.send_header('Content-Encoding', 'gzip')
                    self.send_header('Content-Length', str(len(compressed_data)))
                    self.end_headers()
                    self.wfile.write(compressed_data)
                else:
                    response_data = json.dumps(data, indent=2)
                    self.send_header('Content-Length', str(len(response_data)))
                    self.end_headers()
                    self.wfile.write(response_data.encode('utf-8'))
                    
            elif path_parts[0] == 'api' and path_parts[1] == 'symbols':
                # GET /api/symbols
                symbols_data = {
                    'symbols': self.data_service.symbols,
                    'last_update': datetime.now().isoformat()
                }
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                response_data = json.dumps(symbols_data, indent=2)
                self.send_header('Content-Length', str(len(response_data)))
                self.end_headers()
                self.wfile.write(response_data.encode('utf-8'))
                
            elif path_parts[0] == 'api' and path_parts[1] == 'health':
                # GET /api/health
                health_data = {
                    'status': 'healthy',
                    'service': 'unified_data_service',
                    'cache_size': self.data_service.cache.size(),
                    'websocket_clients': sum(len(clients) for clients in self.data_service.websocket_clients.values()),
                    'timestamp': datetime.now().isoformat()
                }
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                response_data = json.dumps(health_data, indent=2)
                self.send_header('Content-Length', str(len(response_data)))
                self.end_headers()
                self.wfile.write(response_data.encode('utf-8'))
                
            else:
                # 404 Not Found
                self.send_response(404)
                self.send_header('Content-Type', 'application/json')
                error_response = json.dumps({'error': 'Endpoint not found'})
                self.send_header('Content-Length', str(len(error_response)))
                self.end_headers()
                self.wfile.write(error_response.encode('utf-8'))
                
        except Exception as e:
            logger.error(f"HTTP handler error: {e}")
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            error_response = json.dumps({'error': 'Internal server error'})
            self.send_header('Content-Length', str(len(error_response)))
            self.end_headers()
            self.wfile.write(error_response.encode('utf-8'))
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def log_message(self, format, *args):
        """Custom log message format"""
        logger.info(f"HTTP: {format % args}")

async def websocket_handler(websocket, path, data_service: UnifiedDataService):
    """Handle WebSocket connections for real-time updates"""
    try:
        # Parse path to get symbol
        path_parts = path.strip('/').split('/')
        if len(path_parts) >= 2 and path_parts[0] == 'stream':
            symbol = path_parts[1].upper()
            
            # Register client
            data_service.websocket_clients[symbol].add(websocket)
            logger.info(f"WebSocket client connected for {symbol}")
            
            # Send initial data
            initial_data = data_service.get_unified_data(symbol, hours_back=1)
            await websocket.send(json.dumps(initial_data))
            
            # Keep connection alive and handle messages
            async for message in websocket:
                try:
                    request = json.loads(message)
                    if request.get('type') == 'get_data':
                        hours_back = request.get('hours_back', 1)
                        data = data_service.get_unified_data(symbol, hours_back)
                        await websocket.send(json.dumps(data))
                except Exception as e:
                    logger.error(f"WebSocket message error: {e}")
                    
    except Exception as e:
        logger.error(f"WebSocket handler error: {e}")
    finally:
        # Unregister client
        if 'symbol' in locals():
            data_service.websocket_clients[symbol].discard(websocket)
            logger.info(f"WebSocket client disconnected from {symbol}")

async def broadcast_updates(data_service: UnifiedDataService):
    """Broadcast updates to WebSocket clients periodically"""
    while data_service.running:
        try:
            for symbol in data_service.symbols:
                clients = data_service.websocket_clients[symbol]
                if clients:
                    # Get latest data
                    data = data_service.get_unified_data(symbol, hours_back=1)
                    message = json.dumps(data)
                    
                    # Send to all clients
                    disconnected_clients = set()
                    for client in clients:
                        try:
                            await client.send(message)
                        except websockets.exceptions.ConnectionClosed:
                            disconnected_clients.add(client)
                    
                    # Remove disconnected clients
                    for client in disconnected_clients:
                        clients.discard(client)
            
            # Wait before next broadcast
            await asyncio.sleep(10)  # Broadcast every 10 seconds
            
        except Exception as e:
            logger.error(f"Broadcast error: {e}")
            await asyncio.sleep(5)

def start_http_server(data_service: UnifiedDataService, port: int):
    """Start HTTP server in a separate thread"""
    
    class DataServiceHTTPHandler(UnifiedDataHTTPHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(data_service, *args, **kwargs)
    
    server = HTTPServer(('', port), DataServiceHTTPHandler)
    logger.info(f"HTTP server starting on port {port}")
    server.serve_forever()

async def start_websocket_server(data_service: UnifiedDataService, port: int):
    """Start WebSocket server"""
    
    async def handler(websocket, path):
        await websocket_handler(websocket, path, data_service)
    
    server = websockets.serve(handler, "localhost", port + 1)  # WebSocket on port + 1
    logger.info(f"WebSocket server starting on port {port + 1}")
    await server

async def main():
    """Main entry point"""
    # Create unified data service
    data_service = UnifiedDataService(port=8090)
    data_service.running = True
    
    logger.info("🚀 Starting Unified Data Service")
    logger.info("=" * 60)
    logger.info("📡 REST API endpoints:")
    logger.info("  GET /api/unified/{symbol}?hours=1&compress=false")
    logger.info("  GET /api/symbols")
    logger.info("  GET /api/health")
    logger.info("🔌 WebSocket endpoint:")
    logger.info("  ws://localhost:8091/stream/{symbol}")
    logger.info("💾 Cache TTL: 30 seconds")
    logger.info("🔄 WebSocket broadcast: every 10 seconds")
    
    try:
        # Start HTTP server in thread
        http_thread = threading.Thread(
            target=start_http_server,
            args=(data_service, 8090),
            daemon=True
        )
        http_thread.start()
        
        # Start WebSocket server and broadcast service
        await asyncio.gather(
            start_websocket_server(data_service, 8090),
            broadcast_updates(data_service)
        )
        
    except KeyboardInterrupt:
        logger.info("Shutting down Unified Data Service")
        data_service.running = False

if __name__ == "__main__":
    asyncio.run(main())