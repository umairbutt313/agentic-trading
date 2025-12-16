#!/usr/bin/env python3
"""
WebSocket Real-Time Data Feed for High-Frequency Scalping
Provides sub-second price updates directly from Capital.com streaming API
"""

import asyncio
import websockets
import json
import time
import logging
from typing import Dict, Callable, Optional, List
from dataclasses import dataclass
import threading
from collections import deque

@dataclass
class TickData:
    symbol: str
    bid: float
    ask: float
    mid: float
    spread: float
    timestamp: float
    volume: int = 0

@dataclass
class OrderBookLevel:
    price: float
    volume: float

@dataclass
class OrderBookSnapshot:
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: float

class CapitalWebSocketClient:
    """
    Real-time WebSocket client for Capital.com streaming API
    Provides millisecond-level price updates for scalping
    """
    
    def __init__(self, trader, symbols: List[str]):
        self.trader = trader
        self.symbols = symbols
        
        # WebSocket configuration
        self.ws_url = "wss://demo-api-capital.backend-capital.com/streaming" if trader.demo else \
                     "wss://api-capital.backend-capital.com/streaming"
        
        # Connection state
        self.websocket = None
        self.is_connected = False
        self.is_authenticated = False
        self.is_subscribed = False
        
        # Data storage
        self.tick_data = {symbol: deque(maxlen=1000) for symbol in symbols}
        self.orderbook_data = {}
        self.last_tick = {}
        
        # Callbacks
        self.tick_callbacks = []
        self.orderbook_callbacks = []
        self.connection_callbacks = []
        
        # Statistics
        self.messages_received = 0
        self.last_heartbeat = time.time()
        self.latency_samples = deque(maxlen=100)
        
        # Connection monitoring
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 5
        
    async def connect(self):
        """Connect to Capital.com WebSocket"""
        try:
            logging.info(f"🔗 Connecting to WebSocket: {self.ws_url}")
            
            # Ensure we have valid session tokens
            if not self.trader.cst or not self.trader.x_security_token:
                logging.info("🔑 Creating trading session for WebSocket...")
                session_result = self.trader.create_session()
                if not session_result or not self.trader.cst:
                    raise Exception("Failed to create valid session tokens")
            
            # Connect to WebSocket
            self.websocket = await websockets.connect(
                self.ws_url,
                extra_headers={
                    'X-CAP-API-KEY': self.trader.api_key,
                    'CST': self.trader.cst,
                    'X-SECURITY-TOKEN': self.trader.x_security_token
                },
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            
            self.is_connected = True
            self.reconnect_attempts = 0
            
            logging.info("✅ WebSocket connected successfully")
            
            # Start authentication
            await self.authenticate()
            
            # Start message handler
            asyncio.create_task(self.message_handler())
            
            # Start heartbeat monitor
            asyncio.create_task(self.heartbeat_monitor())
            
            # Notify connection callbacks
            for callback in self.connection_callbacks:
                try:
                    await callback("CONNECTED")
                except:
                    pass
                    
        except Exception as e:
            logging.error(f"❌ WebSocket connection failed: {e}")
            self.is_connected = False
            await self.handle_reconnection()
    
    async def authenticate(self):
        """Authenticate WebSocket connection"""
        try:
            auth_message = {
                "destination": "connection.authenticate",
                "correlationId": int(time.time() * 1000),
                "cst": self.trader.cst,
                "securityToken": self.trader.x_security_token
            }
            
            await self.websocket.send(json.dumps(auth_message))
            logging.info("🔐 Authentication request sent")
            
        except Exception as e:
            logging.error(f"❌ Authentication failed: {e}")
            raise
    
    async def subscribe_to_symbols(self):
        """Subscribe to real-time price feeds for symbols"""
        try:
            subscription_message = {
                "destination": "marketData.subscribe",
                "correlationId": int(time.time() * 1000),
                "payload": {
                    "epics": self.symbols,
                    "fields": [
                        "BID", "OFFER", "CHANGE", "CHANGE_PCT", 
                        "HIGH", "LOW", "LAST_TRADED", "UPDATE_TIME"
                    ]
                }
            }
            
            await self.websocket.send(json.dumps(subscription_message))
            logging.info(f"📊 Subscribed to symbols: {', '.join(self.symbols)}")
            
            self.is_subscribed = True
            
        except Exception as e:
            logging.error(f"❌ Subscription failed: {e}")
            raise
    
    async def subscribe_to_orderbook(self, symbols: List[str] = None):
        """Subscribe to Level 2 order book data"""
        try:
            symbols_to_subscribe = symbols or self.symbols
            
            orderbook_message = {
                "destination": "orderBook.subscribe",
                "correlationId": int(time.time() * 1000),
                "payload": {
                    "epics": symbols_to_subscribe,
                    "levels": 10  # Top 10 levels
                }
            }
            
            await self.websocket.send(json.dumps(orderbook_message))
            logging.info(f"📋 Subscribed to order book: {', '.join(symbols_to_subscribe)}")
            
        except Exception as e:
            logging.error(f"❌ Order book subscription failed: {e}")
    
    async def message_handler(self):
        """Handle incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                try:
                    self.messages_received += 1
                    data = json.loads(message)
                    
                    await self.process_message(data)
                    
                except json.JSONDecodeError:
                    logging.warning(f"⚠️ Invalid JSON received: {message[:100]}")
                except Exception as e:
                    logging.error(f"❌ Error processing message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logging.warning("⚠️ WebSocket connection closed")
            self.is_connected = False
            await self.handle_reconnection()
        except Exception as e:
            logging.error(f"❌ Message handler error: {e}")
            self.is_connected = False
            await self.handle_reconnection()
    
    async def process_message(self, data: Dict):
        """Process different types of WebSocket messages"""
        message_type = data.get("destination", "")
        
        if message_type == "connection.authenticate":
            await self.handle_auth_response(data)
        elif message_type == "marketData.subscribe":
            await self.handle_subscription_response(data)
        elif message_type == "marketData.price":
            await self.handle_price_update(data)
        elif message_type == "orderBook.update":
            await self.handle_orderbook_update(data)
        elif message_type == "heartbeat":
            self.last_heartbeat = time.time()
        else:
            logging.debug(f"📨 Unhandled message type: {message_type}")
    
    async def handle_auth_response(self, data: Dict):
        """Handle authentication response"""
        status = data.get("status", "")
        
        if status == "OK":
            logging.info("✅ WebSocket authenticated successfully")
            self.is_authenticated = True
            
            # Subscribe to market data
            await self.subscribe_to_symbols()
            await self.subscribe_to_orderbook()
            
        else:
            error_msg = data.get("errorCode", "Unknown authentication error")
            logging.error(f"❌ Authentication failed: {error_msg}")
            raise Exception(f"Authentication failed: {error_msg}")
    
    async def handle_subscription_response(self, data: Dict):
        """Handle market data subscription response"""
        status = data.get("status", "")
        
        if status == "OK":
            logging.info("✅ Market data subscription successful")
        else:
            error_msg = data.get("errorCode", "Unknown subscription error")
            logging.error(f"❌ Subscription failed: {error_msg}")
    
    async def handle_price_update(self, data: Dict):
        """Handle real-time price updates"""
        try:
            payload = data.get("payload", {})
            symbol = payload.get("epic", "")
            
            if symbol in self.symbols:
                # Extract price data
                bid = float(payload.get("BID", 0))
                offer = float(payload.get("OFFER", 0))
                timestamp = time.time()
                
                # Calculate derived values
                mid = (bid + offer) / 2 if bid and offer else 0
                spread = offer - bid if bid and offer else 0
                
                # Create tick data
                tick = TickData(
                    symbol=symbol,
                    bid=bid,
                    ask=offer,
                    mid=mid,
                    spread=spread,
                    timestamp=timestamp,
                    volume=0  # Volume not available in basic feed
                )
                
                # Store tick data
                self.tick_data[symbol].append(tick)
                self.last_tick[symbol] = tick
                
                # Calculate latency (approximate)
                update_time = payload.get("UPDATE_TIME")
                if update_time:
                    try:
                        latency = timestamp - (update_time / 1000)
                        self.latency_samples.append(latency * 1000)  # Convert to ms
                    except:
                        pass
                
                # Notify callbacks
                for callback in self.tick_callbacks:
                    try:
                        await callback(tick)
                    except Exception as e:
                        logging.error(f"❌ Tick callback error: {e}")
                        
        except Exception as e:
            logging.error(f"❌ Error processing price update: {e}")
    
    async def handle_orderbook_update(self, data: Dict):
        """Handle order book updates"""
        try:
            payload = data.get("payload", {})
            symbol = payload.get("epic", "")
            
            if symbol in self.symbols:
                # Parse bid/ask levels
                bids = []
                asks = []
                
                for bid_data in payload.get("bids", []):
                    bids.append(OrderBookLevel(
                        price=float(bid_data.get("price", 0)),
                        volume=float(bid_data.get("size", 0))
                    ))
                
                for ask_data in payload.get("offers", []):
                    asks.append(OrderBookLevel(
                        price=float(ask_data.get("price", 0)),
                        volume=float(ask_data.get("size", 0))
                    ))
                
                # Create order book snapshot
                orderbook = OrderBookSnapshot(
                    symbol=symbol,
                    bids=sorted(bids, key=lambda x: x.price, reverse=True),
                    asks=sorted(asks, key=lambda x: x.price),
                    timestamp=time.time()
                )
                
                # Store order book
                self.orderbook_data[symbol] = orderbook
                
                # Notify callbacks
                for callback in self.orderbook_callbacks:
                    try:
                        await callback(orderbook)
                    except Exception as e:
                        logging.error(f"❌ OrderBook callback error: {e}")
                        
        except Exception as e:
            logging.error(f"❌ Error processing order book update: {e}")
    
    async def heartbeat_monitor(self):
        """Monitor connection health and send heartbeats"""
        while self.is_connected:
            try:
                # Check if we've received recent heartbeat
                time_since_heartbeat = time.time() - self.last_heartbeat
                
                if time_since_heartbeat > 60:  # 60 seconds timeout
                    logging.warning("⚠️ Heartbeat timeout, reconnecting...")
                    await self.handle_reconnection()
                    break
                
                # Send heartbeat
                if self.websocket and not self.websocket.closed:
                    heartbeat_msg = {
                        "destination": "heartbeat",
                        "correlationId": int(time.time() * 1000)
                    }
                    await self.websocket.send(json.dumps(heartbeat_msg))
                
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
                
            except Exception as e:
                logging.error(f"❌ Heartbeat error: {e}")
                break
    
    async def handle_reconnection(self):
        """Handle WebSocket reconnection"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logging.error(f"❌ Max reconnection attempts reached ({self.max_reconnect_attempts})")
            return
        
        self.is_connected = False
        self.is_authenticated = False
        self.is_subscribed = False
        
        self.reconnect_attempts += 1
        
        logging.info(f"🔄 Reconnecting... (Attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})")
        
        await asyncio.sleep(self.reconnect_delay)
        
        try:
            await self.connect()
        except Exception as e:
            logging.error(f"❌ Reconnection failed: {e}")
            await self.handle_reconnection()
    
    def add_tick_callback(self, callback: Callable):
        """Add callback for tick data updates"""
        self.tick_callbacks.append(callback)
    
    def add_orderbook_callback(self, callback: Callable):
        """Add callback for order book updates"""
        self.orderbook_callbacks.append(callback)
    
    def add_connection_callback(self, callback: Callable):
        """Add callback for connection events"""
        self.connection_callbacks.append(callback)
    
    def get_latest_tick(self, symbol: str) -> Optional[TickData]:
        """Get latest tick for symbol"""
        return self.last_tick.get(symbol)
    
    def get_tick_history(self, symbol: str, count: int = 100) -> List[TickData]:
        """Get recent tick history for symbol"""
        if symbol in self.tick_data:
            return list(self.tick_data[symbol])[-count:]
        return []
    
    def get_orderbook(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """Get latest order book for symbol"""
        return self.orderbook_data.get(symbol)
    
    def get_statistics(self) -> Dict:
        """Get connection statistics"""
        avg_latency = sum(self.latency_samples) / len(self.latency_samples) if self.latency_samples else 0
        
        return {
            'connected': self.is_connected,
            'authenticated': self.is_authenticated,
            'subscribed': self.is_subscribed,
            'messages_received': self.messages_received,
            'reconnect_attempts': self.reconnect_attempts,
            'avg_latency_ms': round(avg_latency, 2),
            'symbols_subscribed': len(self.symbols),
            'last_heartbeat': self.last_heartbeat
        }
    
    async def close(self):
        """Close WebSocket connection"""
        self.is_connected = False
        
        if self.websocket:
            await self.websocket.close()
            
        logging.info("🔌 WebSocket connection closed")


class WebSocketDataProvider:
    """
    High-level wrapper for WebSocket data access
    Provides simplified interface for scalping algorithms
    """
    
    def __init__(self, trader, symbols: List[str]):
        self.client = CapitalWebSocketClient(trader, symbols)
        self.symbols = symbols
        
        # Data caches for fast access
        self.latest_prices = {}
        self.price_streams = {symbol: deque(maxlen=500) for symbol in symbols}
        
        # Setup callbacks
        self.client.add_tick_callback(self._on_tick_update)
    
    async def start(self):
        """Start WebSocket connection"""
        await self.client.connect()
    
    async def _on_tick_update(self, tick: TickData):
        """Handle tick updates"""
        # Cache latest price
        self.latest_prices[tick.symbol] = {
            'bid': tick.bid,
            'ask': tick.ask,
            'mid': tick.mid,
            'spread': tick.spread,
            'timestamp': tick.timestamp
        }
        
        # Add to price stream
        self.price_streams[tick.symbol].append(tick.mid)
    
    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """Get current price data for symbol"""
        return self.latest_prices.get(symbol)
    
    def get_price_history(self, symbol: str, periods: int = 50) -> List[float]:
        """Get recent price history"""
        if symbol in self.price_streams:
            return list(self.price_streams[symbol])[-periods:]
        return []
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected"""
        return self.client.is_connected and self.client.is_authenticated
    
    async def close(self):
        """Close connection"""
        await self.client.close()


# Example usage for scalping integration
async def scalping_websocket_example():
    """Example of how to integrate WebSocket with scalping"""
    
    # This would be integrated into the main scalper
    from trading.capital_trader import CapitalTrader
    
    trader = CapitalTrader(
        api_key="your_key",
        password="your_password",
        email="your_email",
        demo=True
    )
    
    # OLD: Multiple symbols
    # symbols = ['NVDA', 'AAPL']
    # NEW: Only NVIDIA enabled (matches companies.yaml configuration)
    symbols = ['NVDA']
    ws_provider = WebSocketDataProvider(trader, symbols)
    
    # Start connection
    await ws_provider.start()
    
    print("🔥 WebSocket scalping data feed active")
    
    # Example scalping loop with WebSocket data
    for i in range(100):
        for symbol in symbols:
            current_price = ws_provider.get_current_price(symbol)
            price_history = ws_provider.get_price_history(symbol, 20)
            
            if current_price and len(price_history) > 10:
                print(f"💹 {symbol}: ${current_price['mid']:.2f} (Spread: ${current_price['spread']:.3f})")
                
                # Here you would run your scalping analysis
                # signal = analyze_scalping_opportunity(symbol, current_price, price_history)
                # if signal: execute_trade(signal)
        
        await asyncio.sleep(1)  # Check every second
    
    await ws_provider.close()


if __name__ == "__main__":
    # Test WebSocket connection
    asyncio.run(scalping_websocket_example())