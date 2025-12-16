#!/usr/bin/env node

/**
 * Technical Indicators Extractor - Playwright Implementation
 * 
 * Enhanced technical indicators extraction featuring:
 * - Better navigation to technicals page
 * - Improved extraction of technical analysis data
 * - More robust selector handling
 * - Enhanced error handling
 * 
 * Features:
 * - Playwright with enhanced stealth
 * - Technical analysis summary extraction
 * - Oscillators and moving averages data
 * - Multiple fallback strategies
 */

const { chromium } = require('playwright');
const moment = require('moment');

class TechnicalIndicatorsExtractor {
  constructor(options = {}) {
    this.options = {
      headless: true,
      timeout: 60000,
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
    
    if (this.debug) console.log('Technical indicators browser launched successfully');
  }

  async close() {
    if (this.browser) {
      await this.browser.close();
      this.browser = null;
    }
  }

  async createPage(browser) {
    const page = await browser.newPage();
    
    // Set user agent and viewport
    await page.setExtraHTTPHeaders({
      'User-Agent': this.options.userAgent
    });
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

  async waitForTechnicalData(page) {
    if (this.debug) console.log('Waiting for technical data to load...');
    
    const strategies = [
      // Strategy 1: Wait for technical analysis summary
      async () => {
        await page.waitForSelector('[data-name="technical-analysis-summary"]', { timeout: 15000 });
        await page.waitForFunction(() => {
          const summaryElement = document.querySelector('[data-name="technical-analysis-summary"]');
          return summaryElement && summaryElement.textContent.trim() !== '';
        }, { timeout: 10000 });
      },
      
      // Strategy 2: Wait for oscillators section
      async () => {
        await page.waitForSelector('[data-name="oscillators"]', { timeout: 15000 });
        await page.waitForFunction(() => {
          const oscillatorsElement = document.querySelector('[data-name="oscillators"]');
          return oscillatorsElement && oscillatorsElement.textContent.trim() !== '';
        }, { timeout: 10000 });
      },
      
      // Strategy 3: Wait for moving averages
      async () => {
        await page.waitForSelector('[data-name="moving-averages"]', { timeout: 15000 });
        await page.waitForFunction(() => {
          const maElement = document.querySelector('[data-name="moving-averages"]');
          return maElement && maElement.textContent.trim() !== '';
        }, { timeout: 10000 });
      },
      
      // Strategy 4: Generic wait for technical content
      async () => {
        await page.waitForFunction(() => {
          const bodyText = document.body.textContent || '';
          return bodyText.includes('Technical Analysis') || 
                 bodyText.includes('Oscillators') || 
                 bodyText.includes('Moving Averages');
        }, { timeout: 15000 });
      }
    ];
    
    for (const strategy of strategies) {
      try {
        await strategy();
        if (this.debug) console.log('Technical data loaded successfully');
        return true;
      } catch (error) {
        if (this.debug) console.log(`Technical data strategy failed: ${error.message}`);
        continue;
      }
    }
    
    throw new Error('Failed to load technical data with all strategies');
  }

  async extractTechnicalSummary(page) {
    if (this.debug) console.log('Extracting technical summary...');
    
    const summarySelectors = [
      '[data-name="technical-analysis-summary"]',
      '.speedometer-sum',
      '.speedometer-wrapper',
      '.technical-analysis-summary',
      '[data-field="technical_summary"]'
    ];
    
    for (const selector of summarySelectors) {
      try {
        const element = await page.$(selector);
        if (element) {
          const text = await element.textContent();
          if (text && text.trim() !== '') {
            const summary = this.parseTechnicalSummary(text);
            if (summary) {
              if (this.debug) console.log(`Found technical summary: ${summary}`);
              return summary;
            }
          }
        }
      } catch (error) {
        if (this.debug) console.log(`Technical summary selector ${selector} failed: ${error.message}`);
        continue;
      }
    }
    
    // Search for technical signals in page text
    try {
      const bodyText = await page.textContent('body');
      const signalMatch = bodyText.match(/(Strong\s+(?:Buy|Sell)|Buy|Sell|Neutral)/i);
      if (signalMatch) {
        const signal = signalMatch[1].trim();
        if (this.debug) console.log(`Found technical signal in text: ${signal}`);
        return signal;
      }
    } catch (error) {
      if (this.debug) console.log(`Technical summary text extraction failed: ${error.message}`);
    }
    
    return null;
  }

  parseTechnicalSummary(text) {
    if (!text) return null;
    
    const cleanText = text.trim().toLowerCase();
    
    if (cleanText.includes('strong buy')) return 'Strong Buy';
    if (cleanText.includes('strong sell')) return 'Strong Sell';
    if (cleanText.includes('buy')) return 'Buy';
    if (cleanText.includes('sell')) return 'Sell';
    if (cleanText.includes('neutral')) return 'Neutral';
    
    return null;
  }

  async extractOscillators(page) {
    if (this.debug) console.log('Extracting oscillators...');
    
    const oscillatorSelectors = [
      '[data-name="oscillators"]',
      '.oscillators-section',
      '.oscillators-wrapper',
      '[data-field="oscillators"]'
    ];
    
    for (const selector of oscillatorSelectors) {
      try {
        const element = await page.$(selector);
        if (element) {
          const text = await element.textContent();
          if (text && text.trim() !== '') {
            const oscillators = this.parseOscillators(text);
            if (oscillators) {
              if (this.debug) console.log(`Found oscillators: ${oscillators}`);
              return oscillators;
            }
          }
        }
      } catch (error) {
        if (this.debug) console.log(`Oscillators selector ${selector} failed: ${error.message}`);
        continue;
      }
    }
    
    // Search for oscillator data in page text
    try {
      const bodyText = await page.textContent('body');
      const oscillatorMatch = bodyText.match(/Oscillators?[:\s]*(Strong\s+(?:Buy|Sell)|Buy|Sell|Neutral)/i);
      if (oscillatorMatch) {
        const oscillators = oscillatorMatch[1].trim();
        if (this.debug) console.log(`Found oscillators in text: ${oscillators}`);
        return oscillators;
      }
    } catch (error) {
      if (this.debug) console.log(`Oscillators text extraction failed: ${error.message}`);
    }
    
    return null;
  }

  parseOscillators(text) {
    if (!text) return null;
    
    const cleanText = text.trim().toLowerCase();
    
    if (cleanText.includes('strong buy')) return 'Strong Buy';
    if (cleanText.includes('strong sell')) return 'Strong Sell';
    if (cleanText.includes('buy')) return 'Buy';
    if (cleanText.includes('sell')) return 'Sell';
    if (cleanText.includes('neutral')) return 'Neutral';
    
    return null;
  }

  async extractMovingAverages(page) {
    if (this.debug) console.log('Extracting moving averages...');
    
    const maSelectors = [
      '[data-name="moving-averages"]',
      '.moving-averages-section',
      '.moving-averages-wrapper',
      '[data-field="moving_averages"]'
    ];
    
    for (const selector of maSelectors) {
      try {
        const element = await page.$(selector);
        if (element) {
          const text = await element.textContent();
          if (text && text.trim() !== '') {
            const movingAverages = this.parseMovingAverages(text);
            if (movingAverages) {
              if (this.debug) console.log(`Found moving averages: ${movingAverages}`);
              return movingAverages;
            }
          }
        }
      } catch (error) {
        if (this.debug) console.log(`Moving averages selector ${selector} failed: ${error.message}`);
        continue;
      }
    }
    
    // Search for moving averages data in page text
    try {
      const bodyText = await page.textContent('body');
      const maMatch = bodyText.match(/Moving\s+Averages?[:\s]*(Strong\s+(?:Buy|Sell)|Buy|Sell|Neutral)/i);
      if (maMatch) {
        const movingAverages = maMatch[1].trim();
        if (this.debug) console.log(`Found moving averages in text: ${movingAverages}`);
        return movingAverages;
      }
    } catch (error) {
      if (this.debug) console.log(`Moving averages text extraction failed: ${error.message}`);
    }
    
    return null;
  }

  parseMovingAverages(text) {
    if (!text) return null;
    
    const cleanText = text.trim().toLowerCase();
    
    if (cleanText.includes('strong buy')) return 'Strong Buy';
    if (cleanText.includes('strong sell')) return 'Strong Sell';
    if (cleanText.includes('buy')) return 'Buy';
    if (cleanText.includes('sell')) return 'Sell';
    if (cleanText.includes('neutral')) return 'Neutral';
    
    return null;
  }

  async extractTechnicalData(symbol) {
    if (this.debug) console.log(`Starting technical indicators extraction for ${symbol}`);
    
    if (!this.browser) {
      await this.launch();
    }
    
    const page = await this.createPage(this.browser);
    const url = `https://www.tradingview.com/symbols/${symbol}/technicals/`;
    
    try {
      if (this.debug) console.log(`Navigating to: ${url}`);
      await page.goto(url, { waitUntil: 'networkidle', timeout: this.options.timeout });
      
      // Wait for technical data to load
      await this.waitForTechnicalData(page);
      
      // Add additional wait for dynamic content
      await page.waitForTimeout(this.options.waitTime);
      
      // Extract technical indicators
      const technicalSummary = await this.extractTechnicalSummary(page);
      const oscillators = await this.extractOscillators(page);
      const movingAverages = await this.extractMovingAverages(page);
      
      const technicalData = {
        symbol: symbol,
        technical_summary: technicalSummary,
        oscillators: oscillators,
        moving_averages: movingAverages,
        extraction_time: moment().format('YYYY-MM-DD HH:mm:ss'),
        url: url
      };
      
      if (this.debug) {
        console.log('Extracted technical data:', JSON.stringify(technicalData, null, 2));
      }
      
      return technicalData;
      
    } catch (error) {
      console.error(`Error extracting technical indicators for ${symbol}:`, error.message);
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
}

module.exports = TechnicalIndicatorsExtractor;

// Test if called directly
if (require.main === module) {
  (async () => {
    const extractor = new TechnicalIndicatorsExtractor({ debug: true });
    
    try {
      const data = await extractor.extractTechnicalData('NVDA');
      console.log('Final result:', JSON.stringify(data, null, 2));
    } catch (error) {
      console.error('Test failed:', error);
    } finally {
      await extractor.close();
    }
  })();
} 