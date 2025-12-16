from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import os
import yaml
import sys
import platform
from datetime import datetime

def get_chrome_driver():
    """Get Chrome driver with cross-platform compatibility"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-logging")
    chrome_options.add_argument("--silent")
    chrome_options.add_argument("--window-size=1920,1080")
    
    # Platform-specific Chrome/Chromium configuration
    system = platform.system().lower()
    arch = platform.machine().lower()
    
    print(f"[INFO] Detected system: {system}, architecture: {arch}")
    
    # Docker environment detection
    is_docker = os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER") == "true"
    
    if is_docker:
        print("[INFO] Running in Docker container")
        # Docker environment - use installed Chromium
        chrome_options.binary_location = "/usr/bin/chromium"
        
        # Try different driver paths in Docker
        driver_paths = [
            "/usr/bin/chromedriver",
            "/usr/bin/chromium-driver",
            "/usr/local/bin/chromedriver"
        ]
        
        driver = None
        for driver_path in driver_paths:
            if os.path.exists(driver_path):
                print(f"[INFO] Using Chrome driver: {driver_path}")
                try:
                    service = Service(driver_path)
                    driver = webdriver.Chrome(service=service, options=chrome_options)
                    break
                except Exception as e:
                    print(f"[WARNING] Failed to use driver {driver_path}: {e}")
                    continue
        
        if not driver:
            print("[INFO] Trying default Chrome driver setup")
            try:
                driver = webdriver.Chrome(options=chrome_options)
            except Exception as e:
                raise Exception(f"Failed to initialize Chrome driver in Docker: {e}")
                
    else:
        # Native environment
        print("[INFO] Running in native environment")
        
        # Native Mac/Linux environment
        if system == "darwin":  # macOS
            # Try common Chrome paths on Mac
            chrome_paths = [
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "/Applications/Chromium.app/Contents/MacOS/Chromium"
            ]
        else:  # Linux
            chrome_paths = [
                "/usr/bin/google-chrome",
                "/usr/bin/chromium-browser",
                "/usr/bin/chromium",
                "/snap/bin/chromium"
            ]
        
        # Find available Chrome binary
        for chrome_path in chrome_paths:
            if os.path.exists(chrome_path):
                chrome_options.binary_location = chrome_path
                print(f"[INFO] Using Chrome binary: {chrome_path}")
                break
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
        except Exception as e:
            raise Exception(f"Failed to initialize Chrome driver: {e}")
    
    return driver

def screenshot_tradingview_nvda(out_file="nvda_chart.png"):
    """Take screenshot of NVDA TradingView chart"""
    try:
        driver = get_chrome_driver()
        
        # Load the *advanced* chart URL for NVDA
        driver.get("https://www.tradingview.com/chart/?symbol=NASDAQ%3ANVDA")
        
        # Wait a bit for page elements to render (increase if needed)
        time.sleep(5)
        
        # Save screenshot
        driver.save_screenshot(out_file)
        driver.quit()
        print(f"[INFO] Screenshot saved: {out_file}")
        
    except Exception as e:
        print(f"[ERROR] Failed to take screenshot: {e}")
        raise

def load_companies_config(config_path: str = "companies.yaml"):
    """Load companies configuration from YAML file"""
    try:
        # Try relative path from current directory first
        if os.path.exists(config_path):
            config_file = config_path
        else:
            # Try from parent directory (for utils/ subfolder)
            parent_config = os.path.join("..", config_path)
            if os.path.exists(parent_config):
                config_file = parent_config
            else:
                print(f"[ERROR] Cannot find companies config file: {config_path}")
                return {}
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config.get('companies', {})
    except Exception as e:
        print(f"[ERROR] Error loading companies config: {e}")
        return {}

def screenshot_tradingview_all_companies(output_dir="container_output/images"):
    """
    Take TradingView screenshots for all companies defined in companies.yaml
    
    Args:
        output_dir: Directory to save screenshots
    """
    # Load companies configuration
    companies = load_companies_config()
    if not companies:
        print("[ERROR] No companies found in configuration")
        return []
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    screenshots_taken = []
    
    try:
        driver = get_chrome_driver()
        
        for company_key, company_data in companies.items():
            try:
                symbol = company_data.get('symbol', '').upper()
                name = company_data.get('name', '').upper()
                
                if not symbol:
                    print(f"[WARNING] No symbol found for {company_key}, skipping")
                    continue
                
                # Determine exchange based on symbol (default to NASDAQ)
                if symbol in ['NVDA', 'AAPL', 'MSFT', 'GOOG', 'GOOGL', 'AMZN', 'META', 'TSLA']:
                    exchange = "NASDAQ"
                else:
                    exchange = "NASDAQ"  # Default
                
                # Create TradingView URL
                url = f"https://www.tradingview.com/chart/?symbol={exchange}%3A{symbol}"
                
                # Generate filename with new naming convention
                filename = f"raw-chart-{symbol.lower()}_{timestamp}.png"
                filepath = os.path.join(output_dir, filename)
                
                print(f"[INFO] Taking screenshot for {name} ({symbol})...")
                
                # Load page
                driver.get(url)
                
                # Wait for page to load
                time.sleep(7)  # Slightly longer wait for complex charts
                
                # Save screenshot
                driver.save_screenshot(filepath)
                screenshots_taken.append(filepath)
                
                print(f"[INFO] ✓ Screenshot saved: {filename}")
                
                # Brief delay between requests to be respectful
                time.sleep(2)
                
            except Exception as e:
                print(f"[ERROR] Failed to screenshot {company_key}: {e}")
                continue
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize Chrome driver: {e}")
    finally:
        try:
            driver.quit()
        except:
            pass
    
    print(f"[INFO] Completed TradingView screenshots: {len(screenshots_taken)} files")
    return screenshots_taken

if __name__ == "__main__":
    # Take screenshots for all companies
    screenshots = screenshot_tradingview_all_companies()
    print(f"[INFO] Total screenshots taken: {len(screenshots)}")
    
    # Also take the original NVDA screenshot for compatibility
    screenshot_tradingview_nvda("nvda_chart_bar.png")
