#!/usr/bin/env python3
"""
HTTPS-enabled server for NVIDIA chart viewer
Creates self-signed certificate and serves over HTTPS
"""

import http.server
import socketserver
import ssl
import os
import sys
from pathlib import Path

# Change to the directory containing the HTML file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

PORT = 8443  # Standard HTTPS port
HOST = "0.0.0.0"
CERT_FILE = "server.crt"
KEY_FILE = "server.key"

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers for external access
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_GET(self):
        # Handle API endpoints
        if self.path == '/api/data':
            self.serve_csv_data()
            return
        
        # Serve the dashboard index for root path
        if self.path == '/':
            self.path = '/index.html'
        super().do_GET()
    
    def serve_csv_data(self):
        """Serve live CSV data from the actual data file"""
        try:
            # Try multiple possible locations for the data file (prioritize web folder)
            possible_paths = [
                'nvidia_score_price_dump.txt',  # Current web folder (highest priority)
                '../news/nvidia_score_price_dump.txt',  # Fallback to news folder
                '../../news/nvidia_score_price_dump.txt',
                '/Users/uab/Desktop/Arsalan-Project/chartfunction1/stocks/web/nvidia_score_price_dump.txt',
                '/Users/uab/Desktop/Arsalan-Project/chartfunction1/stocks/news/nvidia_score_price_dump.txt'
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
                raise FileNotFoundError("CSV data file not found")
            
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.end_headers()
            self.wfile.write(csv_data.encode('utf-8'))
            
        except Exception as e:
            print(f"Error serving CSV data: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            error_response = '{"error": "Unable to load data file"}'
            self.wfile.write(error_response.encode('utf-8'))

def create_self_signed_cert():
    """Create a self-signed certificate for local HTTPS"""
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import datetime
        
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        
        # Create certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Local"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Local"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Stock Analysis"),
            x509.NameAttribute(NameOID.COMMON_NAME, "stock.local"),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.DNSName("stock.local"),
                x509.DNSName("127.0.0.1"),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256())
        
        # Write private key
        with open(KEY_FILE, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        # Write certificate
        with open(CERT_FILE, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        
        print(f"✅ Created self-signed certificate: {CERT_FILE}, {KEY_FILE}")
        return True
        
    except ImportError:
        print("⚠️  cryptography library not available. Install with: pip install cryptography")
        return False
    except Exception as e:
        print(f"❌ Error creating certificate: {e}")
        return False

def create_simple_cert():
    """Create certificate using OpenSSL command (fallback)"""
    try:
        import subprocess
        
        # Create self-signed certificate using openssl
        cmd = [
            'openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-keyout', KEY_FILE,
            '-out', CERT_FILE, '-days', '365', '-nodes', '-subj',
            '/C=US/ST=Local/L=Local/O=Stock Analysis/CN=stock.local'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Created certificate using OpenSSL: {CERT_FILE}, {KEY_FILE}")
            return True
        else:
            print(f"❌ OpenSSL error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("⚠️  OpenSSL not found. Please install OpenSSL or use HTTP instead.")
        return False
    except Exception as e:
        print(f"❌ Error creating certificate with OpenSSL: {e}")
        return False

if __name__ == "__main__":
    # Check if certificate files exist
    if not (os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE)):
        print("🔐 Creating SSL certificate for HTTPS...")
        
        # Try cryptography library first, then OpenSSL
        if not create_self_signed_cert():
            if not create_simple_cert():
                print("❌ Could not create SSL certificate.")
                print("💡 Falling back to HTTP server on port 8080...")
                print("   Run: python3 server.py")
                sys.exit(1)
    
    try:
        # Create HTTPS server
        with socketserver.TCPServer((HOST, PORT), CustomHTTPRequestHandler) as httpd:
            # Wrap with SSL
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(CERT_FILE, KEY_FILE)
            httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
            
            print(f"🔒 HTTPS Server running at https://{HOST}:{PORT}/")
            print(f"📊 Access your chart at:")
            print(f"   https://localhost:{PORT}/")
            print(f"   https://stock.local:{PORT}/")
            print(f"   https://127.0.0.1:{PORT}/")
            print("")
            print("⚠️  You may see a security warning - click 'Advanced' → 'Proceed to localhost'")
            print("   This is normal for self-signed certificates.")
            print("")
            print("🛑 Press Ctrl+C to stop the server")
            
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n🛑 Server stopped.")
                sys.exit(0)
                
    except Exception as e:
        print(f"❌ Failed to start HTTPS server: {e}")
        print("💡 Try the regular HTTP server instead: python3 server.py")
        sys.exit(1)