#!/usr/bin/env node

/**
 * Enhanced TradingView Scraper - Playwright Implementation
 * 
 * Advanced TradingView data extraction featuring:
 * - Price extraction timing/DOM loading issues
 * - Daily high/low not extracting
 * - Volume not finding data
 * - Post-market data not extracting
 * - Performance improvements with better concurrency
 * 
 * Features:
 * - Playwright with enhanced stealth techniques
 * - Modular architecture with separate extractors
 * - Improved error handling and retry mechanisms
 * - Better waiting strategies for dynamic content
 * - Enhanced data extraction with multiple fallbacks
 * - Optimized performance with resource blocking
 */

const { chromium } = require('playwright');
const fs = require('fs-extra');
const path = require('path');
const moment = require('moment');

// Import modules (now that we're in the modules folder, require directly)
const BasicDataExtractor = require('./legacy_basic_data_extractor');
const TechnicalIndicatorsExtractor = require('./technical_indicators_extractor');

class TradingViewPlaywrightScraper {
  constructor(options = {}) {
    this.options = {
      headless: true,
      timeout: 60000,
      concurrency: 2, // Number of parallel extractions
      debug: false,
      outputDir: '../container_output',
      ...options
    };
    
    this.debug = this.options.debug;
    this.browser = null;
    this.companies = [];
    this.results = {};
  }

  async initialize() {
    if (this.debug) console.log('Initializing Playwright scraper...');
    
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
        '--use-mock-keychain'
      ]
    });

    if (this.debug) console.log('Browser launched successfully');
  }

  async loadCompanies() {
    if (this.debug) console.log('Loading companies configuration...');
    
    try {
      // Try to load companies from YAML file
      const companiesPath = path.join(__dirname, '../companies.yaml');
      if (await fs.pathExists(companiesPath)) {
        const yaml = require('js-yaml');
        const yamlContent = await fs.readFile(companiesPath, 'utf8');
        const data = yaml.load(yamlContent);
        
        // Convert YAML object structure to array
        if (data.companies) {
          this.companies = Object.values(data.companies).map(company => ({
            symbol: company.symbol,
            name: company.name
          }));
        } else {
          this.companies = [];
        }
      } else {
        // Fallback to default companies
        this.companies = [
          { symbol: 'NVDA', name: 'NVIDIA Corporation' },
          { symbol: 'AAPL', name: 'Apple Inc.' },
          { symbol: 'MSFT', name: 'Microsoft Corporation' },
          { symbol: 'GOOGL', name: 'Alphabet Inc.' },
          { symbol: 'AMZN', name: 'Amazon.com Inc.' },
          { symbol: 'TSLA', name: 'Tesla Inc.' }
        ];
      }
      
      if (this.debug) console.log(`Loaded ${this.companies.length} companies`);
    } catch (error) {
      console.error('Error loading companies:', error.message);
      // Use default companies on error
      this.companies = [
        { symbol: 'NVDA', name: 'NVIDIA Corporation' },
        { symbol: 'AAPL', name: 'Apple Inc.' },
        { symbol: 'MSFT', name: 'Microsoft Corporation' },
        { symbol: 'GOOGL', name: 'Alphabet Inc.' },
        { symbol: 'AMZN', name: 'Amazon.com Inc.' },
        { symbol: 'TSLA', name: 'Tesla Inc.' }
      ];
    }
  }

  async scrapeSymbol(symbol) {
    if (this.debug) console.log(`Starting extraction for ${symbol}`);
    
    const symbolData = {
      symbol: symbol,
      extraction_time: moment().format('YYYY-MM-DD HH:mm:ss'),
      basic_data: null,
      technical_indicators: null,
      errors: []
    };

    try {
      // Create extractors
      const basicExtractor = new BasicDataExtractor({
        debug: this.debug,
        timeout: this.options.timeout
      });

      const technicalExtractor = new TechnicalIndicatorsExtractor({
        debug: this.debug,
        timeout: this.options.timeout
      });

      // Extract basic data
      try {
        if (this.debug) console.log(`Extracting basic data for ${symbol}`);
        symbolData.basic_data = await basicExtractor.extractBasicData(symbol);
        
        if (symbolData.basic_data.error) {
          symbolData.errors.push(`Basic data extraction: ${symbolData.basic_data.error}`);
        }
      } catch (error) {
        const errorMsg = `Basic data extraction failed: ${error.message}`;
        symbolData.errors.push(errorMsg);
        console.error(errorMsg);
      }

      // Extract technical indicators
      try {
        if (this.debug) console.log(`Extracting technical indicators for ${symbol}`);
        symbolData.technical_indicators = await technicalExtractor.extractTechnicalIndicators(this.browser, symbol);
        
        if (symbolData.technical_indicators.error) {
          symbolData.errors.push(`Technical indicators extraction: ${symbolData.technical_indicators.error}`);
        }
      } catch (error) {
        const errorMsg = `Technical indicators extraction failed: ${error.message}`;
        symbolData.errors.push(errorMsg);
        console.error(errorMsg);
      }

      // Log success metrics
      const successfulExtractions = [];
      if (symbolData.basic_data && !symbolData.basic_data.error) {
        successfulExtractions.push('basic_data');
      }
      if (symbolData.technical_indicators && !symbolData.technical_indicators.error) {
        successfulExtractions.push('technical_indicators');
      }

      if (this.debug) {
        console.log(`${symbol} extraction completed. Success: ${successfulExtractions.join(', ')}`);
        if (symbolData.errors.length > 0) {
          console.log(`${symbol} errors: ${symbolData.errors.join(', ')}`);
        }
      }

      return symbolData;

    } catch (error) {
      console.error(`Error scraping ${symbol}:`, error.message);
      symbolData.errors.push(`General extraction error: ${error.message}`);
      return symbolData;
    }
  }

  async scrapeAll() {
    if (this.debug) console.log('Starting scraping process...');
    
    await this.initialize();
    await this.loadCompanies();

    const startTime = Date.now();
    
    try {
      // Process companies with controlled concurrency
      const results = [];
      const symbols = this.companies.map(c => c.symbol);
      
      if (this.debug) console.log(`Processing ${symbols.length} symbols with concurrency ${this.options.concurrency}`);
      
      // Process in batches to control concurrency
      for (let i = 0; i < symbols.length; i += this.options.concurrency) {
        const batch = symbols.slice(i, i + this.options.concurrency);
        
        if (this.debug) console.log(`Processing batch: ${batch.join(', ')}`);
        
        const batchPromises = batch.map(symbol => this.scrapeSymbol(symbol));
        const batchResults = await Promise.allSettled(batchPromises);
        
        batchResults.forEach((result, index) => {
          if (result.status === 'fulfilled') {
            results.push(result.value);
          } else {
            console.error(`Batch processing failed for ${batch[index]}:`, result.reason);
            results.push({
              symbol: batch[index],
              extraction_time: moment().format('YYYY-MM-DD HH:mm:ss'),
              errors: [`Batch processing failed: ${result.reason.message}`]
            });
          }
        });
      }

      // Compile final results
      this.results = {
        extraction_metadata: {
          timestamp: moment().format('YYYY-MM-DD HH:mm:ss'),
          total_symbols: symbols.length,
          processing_time_ms: Date.now() - startTime,
          scraper_version: '2.0.0-playwright',
          success_rate: this.calculateSuccessRate(results)
        },
        companies: results
      };

      // Save results
      await this.saveResults();
      
      if (this.debug) {
        console.log(`Scraping completed in ${Date.now() - startTime}ms`);
        console.log(`Success rate: ${this.results.extraction_metadata.success_rate}%`);
      }

      return this.results;

    } catch (error) {
      console.error('Scraping process failed:', error);
      throw error;
    } finally {
      await this.close();
    }
  }

  calculateSuccessRate(results) {
    if (results.length === 0) return 0;
    
    const successfulCount = results.filter(result => {
      const hasBasicData = result.basic_data && !result.basic_data.error;
      const hasTechnicalData = result.technical_indicators && !result.technical_indicators.error;
      return hasBasicData || hasTechnicalData;
    }).length;
    
    return ((successfulCount / results.length) * 100).toFixed(2);
  }

  formatForTradingView() {
    if (this.debug) console.log('Formatting results for TradingView compatibility...');
    
    const trading_data = this.results.companies.map(company => {
      const timestamp = new Date().getTime();
      const companyName = company.symbol === 'NVDA' ? 'NVIDIA' : 
                         company.symbol === 'AAPL' ? 'APPLE' :
                         company.symbol === 'MSFT' ? 'MICROSOFT' :
                         company.symbol === 'GOOGL' ? 'GOOGLE' :
                         company.symbol === 'AMZN' ? 'AMAZON' :
                         company.symbol === 'TSLA' ? 'TESLA' : company.symbol;
      
      return {
        id: `${company.symbol.toLowerCase()}_tradingview_${timestamp}`,
        company: companyName,
        symbol: company.symbol,
        exchange: "NASDAQ",
        price_data: {
          current_price: company.basic_data?.current_price || "N/A",
          price_change: company.basic_data?.price_change || "N/A",
          price_change_percent: company.basic_data?.price_change_percent || "N/A",
          daily_high: company.basic_data?.daily_high || "N/A",
          daily_low: company.basic_data?.daily_low || "N/A",
          volume: company.basic_data?.volume || "N/A",
          market_cap: company.basic_data?.market_cap || "N/A"
        },
        technical_indicators: {
          rsi: company.technical_indicators?.rsi || "N/A",
          macd_signal: company.technical_indicators?.macd_signal || "N/A",
          ma_20: company.technical_indicators?.ma_20 || "N/A",
          ma_50: company.technical_indicators?.ma_50 || "N/A",
          ma_200: company.technical_indicators?.ma_200 || "N/A",
          indicators_found: company.technical_indicators?.indicators_found || 0
        },
        sentiment_indicators: {
          ideas_count: company.sentiment_indicators?.ideas_count || 0,
          social_sentiment: company.sentiment_indicators?.social_sentiment || "neutral",
          analyst_rating: company.sentiment_indicators?.analyst_rating || "N/A",
          page_title: company.sentiment_indicators?.page_title || `${company.symbol} Chart`
        },
        scraped_at: new Date().toISOString(),
        url: `https://www.tradingview.com/chart/?symbol=NASDAQ:${company.symbol}`
      };
    });

    return {
      fetch_metadata: {
        companies: this.companies.map(c => c.name.toLowerCase()),
        fetch_timestamp: new Date().toISOString(),
        total_companies: this.companies.length,
        source: "TradingView.com",
        scraper_version: "2.0.0-playwright"
      },
      trading_data: trading_data
    };
  }

  async saveResults() {
    if (this.debug) console.log('Saving results...');
    
    try {
      // Save to Output/tradingview directory
      const outputDir = path.resolve(__dirname, '../../Output/tradingview');
      await fs.ensureDir(outputDir);
      
      // Use the same filename format as existing scraper
      const timestamp = moment().format('YYYYMMDD_HHmmss');
      const filename = `raw-tradingview_${timestamp}.json`;
      const filepath = path.join(outputDir, filename);
      
      // Transform results to match existing format
      const formattedResults = this.formatForTradingView();
      
      await fs.writeJson(filepath, formattedResults, { spaces: 2 });
      
      if (this.debug) console.log(`Results saved to: ${filepath}`);
      
      return filepath;
    } catch (error) {
      console.error('Error saving results:', error);
      throw error;
    }
  }

  async close() {
    if (this.browser) {
      await this.browser.close();
      this.browser = null;
      if (this.debug) console.log('Browser closed');
    }
  }

  // Method to scrape single symbol (for testing)
  async scrapeSingle(symbol) {
    if (this.debug) console.log(`Scraping single symbol: ${symbol}`);
    
    await this.initialize();
    
    try {
      const result = await this.scrapeSymbol(symbol);
      return result;
    } finally {
      await this.close();
    }
  }
}

// CLI functionality
if (require.main === module) {
  const args = process.argv.slice(2);
  const symbol = args[0];
  
  if (symbol && symbol !== '--all') {
    // Single symbol scraping
    (async () => {
      const scraper = new TradingViewPlaywrightScraper({ debug: true });
      
      try {
        const result = await scraper.scrapeSingle(symbol.toUpperCase());
        console.log('\n=== SCRAPING RESULTS ===');
        console.log(JSON.stringify(result, null, 2));
      } catch (error) {
        console.error('Scraping failed:', error);
        process.exit(1);
      }
    })();
  } else {
    // All symbols scraping
    (async () => {
      const scraper = new TradingViewPlaywrightScraper({ debug: true });
      
      try {
        const results = await scraper.scrapeAll();
        console.log('\n=== SCRAPING COMPLETED ===');
        console.log(`Total symbols: ${results.extraction_metadata.total_symbols}`);
        console.log(`Processing time: ${results.extraction_metadata.processing_time_ms}ms`);
        console.log(`Success rate: ${results.extraction_metadata.success_rate}%`);
        
        // Print summary
        results.companies.forEach(company => {
          const price = company.basic_data?.current_price || 'N/A';
          const technical = company.technical_indicators?.technical_summary || 'N/A';
          const errors = company.errors.length > 0 ? ` (${company.errors.length} errors)` : '';
          console.log(`${company.symbol}: $${price} - ${technical}${errors}`);
        });
      } catch (error) {
        console.error('Scraping failed:', error);
        process.exit(1);
      }
    })();
  }
}

module.exports = TradingViewPlaywrightScraper; 