#!/usr/bin/env node

/**
 * Chart Data Extractor - Specialized TradingView Chart Page Scraper
 * 
 * Extracts OHLC data and price information from TradingView chart page including:
 * - Open, High, Low, Close prices
 * - Volume data
 * - Price changes and percentages
 * - Real-time and post-market data
 */

const { chromium } = require('playwright');
const fs = require('fs-extra');
const moment = require('moment');

class ChartDataExtractor {
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
    this.browser = null;
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
        '--disable-extensions'
      ]
    });
    
    if (this.debug) console.log('Chart data browser launched successfully');
  }

  async createPage() {
    const page = await this.browser.newPage();
    
    await page.setExtraHTTPHeaders({
      'User-Agent': this.options.userAgent
    });
    await page.setViewportSize(this.options.viewport);
    
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

  async extractOHLCData(page) {
    const ohlcData = {};
    
    try {
      // Wait for chart container to load
      await page.waitForSelector('.chart-container', { timeout: 15000 });
      await page.waitForTimeout(3000);
      
      // Extract OHLC data from chart legend
      const chartContainer = page.locator('.chart-container');
      const chartText = await chartContainer.textContent();
      
      if (chartText) {
        if (this.debug) console.log('Chart text:', chartText.substring(0, 500));
        
        // Parse OHLC format: O171.19H172.40L169.20C170.70
        const ohlcMatch = chartText.match(/O(\d+\.?\d*)H(\d+\.?\d*)L(\d+\.?\d*)C(\d+\.?\d*)/);
        if (ohlcMatch) {
          ohlcData.open = parseFloat(ohlcMatch[1]);
          ohlcData.high = parseFloat(ohlcMatch[2]);
          ohlcData.low = parseFloat(ohlcMatch[3]);
          ohlcData.close = parseFloat(ohlcMatch[4]);
          ohlcData.current_price = ohlcData.close;
          
          if (this.debug) {
            console.log(`OHLC extracted: O:${ohlcData.open}, H:${ohlcData.high}, L:${ohlcData.low}, C:${ohlcData.close}`);
          }
        }
        
        // Alternative OHLC pattern matching
        if (!ohlcMatch) {
          const altOhlcMatch = chartText.match(/Open[\s\S]*?(\d+\.?\d*)[\s\S]*?High[\s\S]*?(\d+\.?\d*)[\s\S]*?Low[\s\S]*?(\d+\.?\d*)[\s\S]*?Close[\s\S]*?(\d+\.?\d*)/i);
          if (altOhlcMatch) {
            ohlcData.open = parseFloat(altOhlcMatch[1]);
            ohlcData.high = parseFloat(altOhlcMatch[2]);
            ohlcData.low = parseFloat(altOhlcMatch[3]);
            ohlcData.close = parseFloat(altOhlcMatch[4]);
            ohlcData.current_price = ohlcData.close;
          }
        }
      }
      
      // Try alternative selector for OHLC data
      if (!ohlcData.open) {
        const ohlcElements = await page.locator('[data-name="legend-source-item"]').all();
        for (const element of ohlcElements) {
          const text = await element.textContent();
          if (text && text.includes('O') && text.includes('H') && text.includes('L') && text.includes('C')) {
            const match = text.match(/O(\d+\.?\d*)H(\d+\.?\d*)L(\d+\.?\d*)C(\d+\.?\d*)/);
            if (match) {
              ohlcData.open = parseFloat(match[1]);
              ohlcData.high = parseFloat(match[2]);
              ohlcData.low = parseFloat(match[3]);
              ohlcData.close = parseFloat(match[4]);
              ohlcData.current_price = ohlcData.close;
              break;
            }
          }
        }
      }
      
    } catch (error) {
      if (this.debug) console.log(`OHLC extraction failed: ${error.message}`);
    }
    
    return ohlcData;
  }

  async extractVolumeData(page) {
    const volumeData = {};
    
    try {
      // Extract volume from chart container
      const chartContainer = page.locator('.chart-container');
      const chartText = await chartContainer.textContent();
      
      if (chartText) {
        // Parse volume format: Vol229.9 M
        const volumeMatch = chartText.match(/Vol(\d+\.?\d*)\s*([MBK])?/);
        if (volumeMatch) {
          let volume = parseFloat(volumeMatch[1]);
          const unit = volumeMatch[2];
          
          // Convert to actual number
          if (unit === 'M') {
            volume = volume * 1000000;
            volumeData.volume_formatted = volumeMatch[1] + 'M';
          } else if (unit === 'B') {
            volume = volume * 1000000000;
            volumeData.volume_formatted = volumeMatch[1] + 'B';
          } else if (unit === 'K') {
            volume = volume * 1000;
            volumeData.volume_formatted = volumeMatch[1] + 'K';
          } else {
            volumeData.volume_formatted = volumeMatch[1];
          }
          
          volumeData.volume = volume;
          
          if (this.debug) {
            console.log(`Volume extracted: ${volume} (${volumeData.volume_formatted})`);
          }
        }
      }
      
      // Try alternative volume extraction
      if (!volumeData.volume) {
        const volumeElements = await page.locator(':text("Vol")').all();
        for (const element of volumeElements) {
          const text = await element.textContent();
          if (text) {
            const match = text.match(/Vol\s*(\d+\.?\d*)\s*([MBK])?/);
            if (match) {
              let volume = parseFloat(match[1]);
              const unit = match[2];
              
              if (unit === 'M') volume *= 1000000;
              else if (unit === 'B') volume *= 1000000000;
              else if (unit === 'K') volume *= 1000;
              
              volumeData.volume = volume;
              volumeData.volume_formatted = match[1] + (unit || '');
              break;
            }
          }
        }
      }
      
    } catch (error) {
      if (this.debug) console.log(`Volume extraction failed: ${error.message}`);
    }
    
    return volumeData;
  }

  async extractPriceChangeData(page) {
    const priceChangeData = {};
    
    try {
      // Extract price change from chart container
      const chartContainer = page.locator('.chart-container');
      const chartText = await chartContainer.textContent();
      
      if (chartText) {
        // Parse price change format: +6.63 (+4.04%)
        const changeMatch = chartText.match(/([+-]\d+\.?\d*)\s*\(([+-]\d+\.?\d*)%\)/);
        if (changeMatch) {
          priceChangeData.price_change = parseFloat(changeMatch[1]);
          priceChangeData.price_change_percent = changeMatch[2] + '%';
          
          if (this.debug) {
            console.log(`Price change extracted: ${priceChangeData.price_change} (${priceChangeData.price_change_percent})`);
          }
        }
        
        // Extract post-market data if available
        const postMarketMatch = chartText.match(/Post-market[\s\S]*?([+-]\d+\.?\d*)[\s\S]*?([+-]\d+\.?\d*)%/);
        if (postMarketMatch) {
          priceChangeData.post_market_change = parseFloat(postMarketMatch[1]);
          priceChangeData.post_market_change_percent = postMarketMatch[2] + '%';
          
          if (this.debug) {
            console.log(`Post-market change extracted: ${priceChangeData.post_market_change} (${priceChangeData.post_market_change_percent})`);
          }
        }
      }
      
    } catch (error) {
      if (this.debug) console.log(`Price change extraction failed: ${error.message}`);
    }
    
    return priceChangeData;
  }

  async extractChartData(symbol) {
    if (this.debug) console.log(`Starting chart data extraction for ${symbol}`);
    
    if (!this.browser) {
      await this.launch();
    }
    
    const page = await this.createPage();
    const url = `https://www.tradingview.com/chart/?symbol=NASDAQ:${symbol}`;
    
    try {
      await page.goto(url, { waitUntil: 'networkidle', timeout: this.options.timeout });
      await page.waitForTimeout(this.options.waitTime);
      
      // Extract different types of chart data in parallel
      const [ohlcData, volumeData, priceChangeData] = await Promise.all([
        this.extractOHLCData(page),
        this.extractVolumeData(page),
        this.extractPriceChangeData(page)
      ]);
      
      const chartData = {
        symbol: symbol,
        url: url,
        extraction_time: moment().format('YYYY-MM-DD HH:mm:ss'),
        ...ohlcData,
        ...volumeData,
        ...priceChangeData
      };
      
      if (this.debug) {
        console.log('Chart data extracted:', JSON.stringify(chartData, null, 2));
      }
      
      return chartData;
      
    } catch (error) {
      console.error(`Chart data extraction failed for ${symbol}:`, error.message);
      return {
        symbol: symbol,
        url: url,
        error: error.message,
        extraction_time: moment().format('YYYY-MM-DD HH:mm:ss')
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

module.exports = ChartDataExtractor;

// Test if called directly
if (require.main === module) {
  (async () => {
    const extractor = new ChartDataExtractor({ debug: true });
    
    try {
      const data = await extractor.extractChartData('NVDA');
      console.log('Final result:', JSON.stringify(data, null, 2));
    } catch (error) {
      console.error('Test failed:', error);
    } finally {
      await extractor.close();
    }
  })();
} 