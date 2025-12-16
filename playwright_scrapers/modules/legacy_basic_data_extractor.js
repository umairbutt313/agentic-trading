#!/usr/bin/env node

/**
 * Enhanced Basic Data Extractor - Playwright Implementation
 * 
 * Enhanced basic data extraction featuring:
 * - Price extraction timing/DOM loading issues
 * - Daily high/low not extracting  
 * - Volume not finding data
 * - Post-market data not extracting
 * 
 * Features:
 * - Playwright with stealth techniques
 * - Advanced waiting strategies for dynamic content
 * - Multiple extraction approaches with smart fallbacks
 * - Real-time and after-hours data extraction
 * - Enhanced error handling and retry mechanisms
 * - Market cap, volume, and price range extraction
 */

const { chromium } = require('playwright');
const fs = require('fs-extra');
const moment = require('moment');

class BasicDataExtractor {
  constructor(options = {}) {
    this.options = {
      headless: true,
      timeout: 60000,
      retryAttempts: 3,
      waitTime: 8000,
      userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
      viewport: { width: 1920, height: 1080 },
      ...options
    };
    
    this.debug = options.debug || false;
    this.extractedData = {};
  }

  async launch() {
    this.browser = await chromium.launch({
      headless: this.options.headless,
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-blink-features=AutomationControlled',
        '--disable-features=VizDisplayCompositor',
        '--disable-dev-shm-usage',
        '--disable-extensions',
        '--no-first-run',
        '--disable-default-apps',
        '--disable-background-timer-throttling',
        '--disable-backgrounding-occluded-windows',
        '--disable-renderer-backgrounding',
        '--disable-field-trial-config',
        '--disable-back-forward-cache',
        '--disable-hang-monitor',
        '--disable-prompt-on-repost',
        '--disable-sync',
        '--disable-translate',
        '--metrics-recording-only',
        '--no-default-browser-check',
        '--no-pings',
        '--password-store=basic',
        '--use-mock-keychain',
        '--disable-component-extensions-with-background-pages',
        '--disable-background-networking',
        '--disable-component-update',
        '--disable-client-side-phishing-detection',
        '--disable-default-apps',
        '--disable-domain-reliability',
        '--disable-features=AudioServiceOutOfProcess',
        '--disable-features=VizDisplayCompositor',
        '--disable-ipc-flooding-protection',
        '--disable-features=TranslateUI',
        '--disable-features=BlinkGenPropertyTrees',
        '--disable-renderer-backgrounding',
        '--disable-backgrounding-occluded-windows'
      ]
    });
  }

  async createPage() {
    const page = await this.browser.newPage();
    
    // Set viewport using setViewportSize
    await page.setViewportSize(this.options.viewport);
    
    // Enhanced stealth techniques
    await page.addInitScript(() => {
      // Override webdriver detection
      Object.defineProperty(navigator, 'webdriver', {
        get: () => undefined,
      });
      
      // Override Chrome runtime
      window.chrome = {
        runtime: {},
      };
      
      // Override permissions
      const originalQuery = window.navigator.permissions.query;
      window.navigator.permissions.query = (parameters) => (
        parameters.name === 'notifications' ?
          Promise.resolve({ state: Notification.permission }) :
          originalQuery(parameters)
      );
    });
    
    // Block unnecessary resources for faster loading
    await page.route('**/*', (route) => {
      const request = route.request();
      const resourceType = request.resourceType();
      
      if (['image', 'stylesheet', 'font', 'media'].includes(resourceType)) {
        route.abort();
      } else {
        route.continue();
      }
    });
    
    return page;
  }

  async waitForPriceData(page) {
    if (this.debug) console.log('Waiting for price data to load...');
    
    // Multiple strategies to wait for price data
    const strategies = [
      // Strategy 1: Wait for chart container (chart page)
      async () => {
        await page.waitForSelector('.chart-container', { timeout: 15000 });
        await page.waitForTimeout(5000); // Wait for content to load
        const chartContainer = await page.$('.chart-container');
        if (chartContainer) {
          const text = await chartContainer.textContent();
          if (text.includes('O') && text.includes('H') && text.includes('L')) {
            return true;
          }
        }
        throw new Error('Chart container OHLC data not found');
      },
      
      // Strategy 2: Wait for js-symbol-last (symbols page)
      async () => {
        await page.waitForSelector('.js-symbol-last', { timeout: 15000 });
        await page.waitForTimeout(3000); // Wait for content to load
        const priceElement = await page.$('.js-symbol-last');
        if (priceElement) {
          const text = await priceElement.textContent();
          if (text && text.trim() !== '') {
            return true;
          }
        }
        throw new Error('Price element not found or empty');
      },
      
      // Strategy 3: Wait for price in title
      async () => {
        await page.waitForTimeout(3000); // Wait for title to load
        const title = await page.title();
        if (title && title.includes('$') && /\d+\.\d+/.test(title)) {
          return true;
        }
        throw new Error('Price not found in title');
      }
    ];
    
    for (const strategy of strategies) {
      try {
        await strategy();
        if (this.debug) console.log('Price data loaded successfully');
        return true;
      } catch (error) {
        if (this.debug) console.log(`Strategy failed: ${error.message}`);
        continue;
      }
    }
    
    throw new Error('Failed to load price data with all strategies');
  }

  async extractCurrentPrice(page) {
    if (this.debug) console.log('Extracting current price...');
    
    // First try to extract from chart container (chart page)
    try {
      const chartContainer = await page.$('.chart-container');
      if (chartContainer) {
        const text = await chartContainer.textContent();
        // Parse OHLC format: O171.19H172.40L169.20C170.70
        const closeMatch = text.match(/C(\d+\.\d+)/);
        if (closeMatch) {
          const price = parseFloat(closeMatch[1]);
          if (this.debug) console.log(`Found price from chart container: ${price}`);
          return price;
        }
      }
    } catch (error) {
      if (this.debug) console.log(`Chart container extraction failed: ${error.message}`);
    }
    
    // Try symbol page selectors
    const priceSelectors = [
      '.js-symbol-last',
      '.js-symbol-change-pt'
    ];
    
    for (const selector of priceSelectors) {
      try {
        const priceElement = await page.$(selector);
        if (priceElement) {
          const priceText = await priceElement.textContent();
          if (priceText && priceText.trim() !== '') {
            const price = priceText.replace(/[^\d.,]/g, '');
            if (price && !isNaN(parseFloat(price))) {
              if (this.debug) console.log(`Found price: ${price} using selector: ${selector}`);
              return parseFloat(price);
            }
          }
        }
      } catch (error) {
        if (this.debug) console.log(`Selector ${selector} failed: ${error.message}`);
        continue;
      }
    }
    
    // Fallback: Extract from page title
    try {
      const title = await page.title();
      const priceMatch = title.match(/\$(\d+\.\d+)/);
      if (priceMatch) {
        const price = parseFloat(priceMatch[1]);
        if (this.debug) console.log(`Found price in title: ${price}`);
        return price;
      }
    } catch (error) {
      if (this.debug) console.log(`Title extraction failed: ${error.message}`);
    }
    
    return null;
  }

  async extractPriceChange(page) {
    if (this.debug) console.log('Extracting price change...');
    
    // First try to extract from chart container
    try {
      const chartContainer = await page.$('.chart-container');
      if (chartContainer) {
        const text = await chartContainer.textContent();
        // Look for change pattern: +6.63 (+4.04%)
        const changeMatch = text.match(/([+-]\d+\.\d+)\s*\([+-]\d+\.\d+%\)/);
        if (changeMatch) {
          const change = parseFloat(changeMatch[1]);
          if (this.debug) console.log(`Found price change from chart: ${change}`);
          return change;
        }
      }
    } catch (error) {
      if (this.debug) console.log(`Chart container change extraction failed: ${error.message}`);
    }
    
    // Try symbol page selectors
    const changeSelectors = [
      '.js-symbol-ext-hrs-change',
      '.js-symbol-change-pt'
    ];
    
    for (const selector of changeSelectors) {
      try {
        const changeElement = await page.$(selector);
        if (changeElement) {
          const changeText = await changeElement.textContent();
          if (changeText && changeText.trim() !== '') {
            const changeMatch = changeText.match(/([+-]?\d+\.?\d*)/);
            if (changeMatch) {
              const change = parseFloat(changeMatch[1]);
              if (this.debug) console.log(`Found price change: ${change}`);
              return change;
            }
          }
        }
      } catch (error) {
        if (this.debug) console.log(`Change selector ${selector} failed: ${error.message}`);
        continue;
      }
    }
    
    return null;
  }

  async extractDailyHigh(page) {
    if (this.debug) console.log('Extracting daily high...');
    
    try {
      const chartContainer = await page.$('.chart-container');
      if (chartContainer) {
        const text = await chartContainer.textContent();
        // Parse OHLC format: O171.19H172.40L169.20C170.70
        const highMatch = text.match(/H(\d+\.\d+)/);
        if (highMatch) {
          const high = parseFloat(highMatch[1]);
          if (this.debug) console.log(`Found daily high: ${high}`);
          return high;
        }
      }
    } catch (error) {
      if (this.debug) console.log(`Daily high extraction failed: ${error.message}`);
    }
    
    return null;
  }

  async extractDailyLow(page) {
    if (this.debug) console.log('Extracting daily low...');
    
    try {
      const chartContainer = await page.$('.chart-container');
      if (chartContainer) {
        const text = await chartContainer.textContent();
        // Parse OHLC format: O171.19H172.40L169.20C170.70
        const lowMatch = text.match(/L(\d+\.\d+)/);
        if (lowMatch) {
          const low = parseFloat(lowMatch[1]);
          if (this.debug) console.log(`Found daily low: ${low}`);
          return low;
        }
      }
    } catch (error) {
      if (this.debug) console.log(`Daily low extraction failed: ${error.message}`);
    }
    
    return null;
  }

  async extractVolume(page) {
    if (this.debug) console.log('Extracting volume...');
    
    try {
      const chartContainer = await page.$('.chart-container');
      if (chartContainer) {
        const text = await chartContainer.textContent();
        // Parse volume format: Vol229.9 M
        const volumeMatch = text.match(/Vol(\d+\.?\d*)\s*([MBK])?/);
        if (volumeMatch) {
          let volume = parseFloat(volumeMatch[1]);
          const unit = volumeMatch[2];
          
          // Convert to actual number
          if (unit === 'M') {
            volume = volume * 1000000;
          } else if (unit === 'B') {
            volume = volume * 1000000000;
          } else if (unit === 'K') {
            volume = volume * 1000;
          }
          
          if (this.debug) console.log(`Found volume: ${volume} (${volumeMatch[1]}${unit || ''})`);
          return volume;
        }
      }
    } catch (error) {
      if (this.debug) console.log(`Volume extraction failed: ${error.message}`);
    }
    
    return null;
  }

  async extractDailyRange(page) {
    if (this.debug) console.log('Extracting daily range...');
    
    const rangeSelectors = [
      '[data-field="daily_high"]',
      '[data-field="daily_low"]',
      '.js-symbol-high',
      '.js-symbol-low',
      '[data-symbol-high]',
      '[data-symbol-low]'
    ];
    
    let high = null, low = null;
    
    // Try to find high and low values
    for (const selector of rangeSelectors) {
      try {
        const elements = await page.$$(selector);
        for (const element of elements) {
          const text = await element.textContent();
          if (text && text.trim() !== '') {
            const value = parseFloat(text.replace(/[^\d.,]/g, ''));
            if (!isNaN(value)) {
              if (selector.includes('high')) {
                high = value;
              } else if (selector.includes('low')) {
                low = value;
              }
            }
          }
        }
      } catch (error) {
        if (this.debug) console.log(`Range selector ${selector} failed: ${error.message}`);
        continue;
      }
    }
    
    // Alternative approach: Look for range pattern in text
    if (!high || !low) {
      try {
        const bodyText = await page.textContent('body');
        const rangeMatch = bodyText.match(/(\d+\.\d+)\s*-\s*(\d+\.\d+)/);
        if (rangeMatch) {
          low = parseFloat(rangeMatch[1]);
          high = parseFloat(rangeMatch[2]);
          if (this.debug) console.log(`Found range from text: ${low} - ${high}`);
        }
      } catch (error) {
        if (this.debug) console.log(`Range text extraction failed: ${error.message}`);
      }
    }
    
    return { high, low };
  }

  async extractVolume(page) {
    if (this.debug) console.log('Extracting volume...');
    
    const volumeSelectors = [
      '[data-field="volume"]',
      '.js-symbol-volume',
      '[data-symbol-volume]',
      '.tv-symbol-price-quote__volume'
    ];
    
    for (const selector of volumeSelectors) {
      try {
        const volumeElement = await page.$(selector);
        if (volumeElement) {
          const volumeText = await volumeElement.textContent();
          if (volumeText && volumeText.trim() !== '') {
            const volume = this.parseVolumeString(volumeText);
            if (volume) {
              if (this.debug) console.log(`Found volume: ${volume}`);
              return volume;
            }
          }
        }
      } catch (error) {
        if (this.debug) console.log(`Volume selector ${selector} failed: ${error.message}`);
        continue;
      }
    }
    
    // Search for volume in page text
    try {
      const bodyText = await page.textContent('body');
      const volumeMatch = bodyText.match(/Volume[:\s]*(\d+\.?\d*[KMB]?)/i);
      if (volumeMatch) {
        const volume = this.parseVolumeString(volumeMatch[1]);
        if (this.debug) console.log(`Found volume in text: ${volume}`);
        return volume;
      }
    } catch (error) {
      if (this.debug) console.log(`Volume text extraction failed: ${error.message}`);
    }
    
    return null;
  }

  parseVolumeString(volumeStr) {
    if (!volumeStr) return null;
    
    const cleanStr = volumeStr.trim().toUpperCase();
    const numMatch = cleanStr.match(/(\d+\.?\d*)/);
    if (!numMatch) return null;
    
    const baseNum = parseFloat(numMatch[1]);
    
    if (cleanStr.includes('K')) {
      return baseNum * 1000;
    } else if (cleanStr.includes('M')) {
      return baseNum * 1000000;
    } else if (cleanStr.includes('B')) {
      return baseNum * 1000000000;
    }
    
    return baseNum;
  }

  async extractPostMarketData(page) {
    if (this.debug) console.log('Extracting post-market data...');
    
    const postMarketSelectors = [
      '[data-field="post_market_price"]',
      '[data-field="pre_market_price"]',
      '.js-symbol-postmarket',
      '.js-symbol-premarket',
      '[data-symbol-postmarket]',
      '[data-symbol-premarket]'
    ];
    
    let postMarketPrice = null;
    let postMarketChange = null;
    
    for (const selector of postMarketSelectors) {
      try {
        const element = await page.$(selector);
        if (element) {
          const text = await element.textContent();
          if (text && text.trim() !== '') {
            const price = parseFloat(text.replace(/[^\d.,]/g, ''));
            if (!isNaN(price)) {
              postMarketPrice = price;
              if (this.debug) console.log(`Found post-market price: ${price}`);
              break;
            }
          }
        }
      } catch (error) {
        if (this.debug) console.log(`Post-market selector ${selector} failed: ${error.message}`);
        continue;
      }
    }
    
    return { postMarketPrice, postMarketChange };
  }

  async extractMarketCap(page) {
    if (this.debug) console.log('Extracting market cap...');
    
    const marketCapSelectors = [
      '[data-field="market_cap"]',
      '.js-symbol-market-cap',
      '[data-symbol-market-cap]'
    ];
    
    for (const selector of marketCapSelectors) {
      try {
        const element = await page.$(selector);
        if (element) {
          const text = await element.textContent();
          if (text && text.trim() !== '') {
            const marketCap = this.parseMarketCapString(text);
            if (marketCap) {
              if (this.debug) console.log(`Found market cap: ${marketCap}`);
              return marketCap;
            }
          }
        }
      } catch (error) {
        if (this.debug) console.log(`Market cap selector ${selector} failed: ${error.message}`);
        continue;
      }
    }
    
    return null;
  }

  parseMarketCapString(capStr) {
    if (!capStr) return null;
    
    const cleanStr = capStr.trim().toUpperCase();
    const numMatch = cleanStr.match(/(\d+\.?\d*)/);
    if (!numMatch) return null;
    
    const baseNum = parseFloat(numMatch[1]);
    
    if (cleanStr.includes('T')) {
      return `${baseNum.toFixed(2)}T`;
    } else if (cleanStr.includes('B')) {
      return `${baseNum.toFixed(2)}B`;
    } else if (cleanStr.includes('M')) {
      return `${baseNum.toFixed(2)}M`;
    }
    
    return baseNum.toString();
  }

  async extractBasicData(symbol) {
    if (this.debug) console.log(`Starting basic data extraction for ${symbol}`);
    
    // Launch browser if not already launched
    if (!this.browser) {
      await this.launch();
    }
    
    const page = await this.createPage();
    const url = `https://www.tradingview.com/chart/?symbol=NASDAQ:${symbol}`;
    
    try {
      if (this.debug) console.log(`Navigating to: ${url}`);
      await page.goto(url, { waitUntil: 'networkidle', timeout: this.options.timeout });
      
      // Wait for page to load and price data to be available
      await this.waitForPriceData(page);
      
      // Add additional wait for dynamic content
      await page.waitForTimeout(this.options.waitTime);
      
      // Extract all data
      const currentPrice = await this.extractCurrentPrice(page);
      const priceChange = await this.extractPriceChange(page);
      const dailyHigh = await this.extractDailyHigh(page);
      const dailyLow = await this.extractDailyLow(page);
      const volume = await this.extractVolume(page);
      const postMarketData = await this.extractPostMarketData(page);
      const marketCap = await this.extractMarketCap(page);
      
      const basicData = {
        symbol: symbol,
        current_price: currentPrice,
        price_change: priceChange,
        daily_high: dailyHigh,
        daily_low: dailyLow,
        volume: volume,
        post_market_price: postMarketData.postMarketPrice,
        post_market_change: postMarketData.postMarketChange,
        market_cap: marketCap,
        extraction_time: moment().format('YYYY-MM-DD HH:mm:ss'),
        url: url
      };
      
      if (this.debug) {
        console.log('Extracted basic data:', JSON.stringify(basicData, null, 2));
      }
      
      return basicData;
      
    } catch (error) {
      console.error(`Error extracting basic data for ${symbol}:`, error.message);
      return {
        symbol: symbol,
        error: error.message,
        extraction_time: moment().format('YYYY-MM-DD HH:mm:ss'),
        url: url
      };
    } finally {
      await page.close();
    }
  }

  async close() {
    if (this.browser) {
      await this.browser.close();
    }
  }
}

module.exports = BasicDataExtractor;

// Test if called directly
if (require.main === module) {
  (async () => {
    const extractor = new BasicDataExtractor({ debug: true });
    
    try {
      await extractor.launch();
      const data = await extractor.extractBasicData('NVDA');
      console.log('Final result:', JSON.stringify(data, null, 2));
    } catch (error) {
      console.error('Test failed:', error);
    } finally {
      await extractor.close();
    }
  })();
} 