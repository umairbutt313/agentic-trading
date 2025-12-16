#!/usr/bin/env node

/**
 * Fixed TradingView Scraper - Complete Solution
 * 
 * This script addresses all identified issues in the original TradingView scraper:
 * - Uses enhanced data extraction with accessibility tree parsing
 * - Implements modern stealth techniques from Context7 research
 * - Provides robust fallback mechanisms for data extraction
 * - Enhanced error handling and retry logic
 * - Compatible output format with existing pipeline
 */

const { chromium } = require('playwright');
const fs = require('fs-extra');
const path = require('path');
const moment = require('moment');
const yaml = require('js-yaml');

// Import our enhanced extractor
const EnhancedDataExtractor = require('./playwright_scrapers/modules/enhanced_data_extractor');

class FixedTradingViewScraper {
  constructor(options = {}) {
    this.options = {
      headless: true,
      timeout: 60000,
      concurrency: 2, // Reduced for better reliability
      debug: false,
      outputDir: '../../Output/tradingview',
      ...options
    };
    
    this.debug = this.options.debug;
    this.companies = [];
    this.enhancedExtractor = new EnhancedDataExtractor({ 
      debug: this.debug,
      ...this.options 
    });
  }

  async initialize() {
    if (this.debug) console.log('Initializing fixed TradingView scraper...');
    
    await this.enhancedExtractor.launch();
    
    if (this.debug) console.log('Enhanced extractor initialized successfully');
  }

  async loadCompanies() {
    if (this.debug) console.log('Loading companies configuration...');
    
    try {
      // Try to load companies from YAML file
      const companiesPath = path.join(__dirname, 'companies.yaml');
      if (await fs.pathExists(companiesPath)) {
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
    if (this.debug) console.log(`Starting fixed extraction for ${symbol}`);
    
    const startTime = Date.now();
    
    try {
      // Use enhanced extractor for reliable data extraction
      const extractedData = await this.enhancedExtractor.extractEnhancedData(symbol);
      
      // Transform to expected format
      const combinedData = {
        symbol: symbol,
        extraction_time: moment().format('YYYY-MM-DD HH:mm:ss'),
        processing_time_ms: Date.now() - startTime,
        sources: extractedData.sources || []
      };
      
      const errors = [];
      
      // Map extracted data to standard format
      if (extractedData.error) {
        errors.push(`Enhanced extraction: ${extractedData.error}`);
      } else {
        // Price data mapping
        combinedData.price_data = {
          current_price: extractedData.current_price || extractedData.close || "N/A",
          price_change: extractedData.price_change || "N/A",
          price_change_percent: extractedData.price_change_percent || "N/A",
          daily_high: extractedData.high || "N/A",
          daily_low: extractedData.low || "N/A",
          open: extractedData.open || "N/A",
          close: extractedData.close || "N/A",
          volume: extractedData.volume || "N/A",
          volume_formatted: extractedData.volume_formatted || "N/A"
        };
        
        // Financial data mapping
        combinedData.financial_data = {
          market_cap: extractedData.market_cap || "N/A",
          pe_ratio: "N/A", // Not extracted in this version
          dividend_yield: "N/A", // Not extracted in this version
          eps: "N/A", // Not extracted in this version
          employees: "N/A", // Not extracted in this version
          founded: "N/A", // Not extracted in this version
          ceo: "N/A" // Not extracted in this version
        };
        
        // Technical indicators mapping
        combinedData.technical_indicators = {
          technical_summary: extractedData.technical_indicators?.technical_summary || "N/A",
          oscillators: "N/A", // Not extracted in this version
          moving_averages: "N/A", // Not extracted in this version
          rsi: "N/A", // Not extracted in this version
          macd_signal: "N/A", // Not extracted in this version
          ma_20: "N/A", // Not extracted in this version
          ma_50: "N/A", // Not extracted in this version
          ma_200: "N/A", // Not extracted in this version
          indicators_found: 0
        };
      }
      
      // Add sentiment indicators (placeholder for compatibility)
      combinedData.sentiment_indicators = {
        ideas_count: 0,
        social_sentiment: "neutral",
        analyst_rating: "N/A",
        page_title: `${symbol} Analysis`
      };
      
      combinedData.errors = errors;
      combinedData.success = errors.length === 0 && combinedData.price_data.current_price !== "N/A";
      
      if (this.debug) {
        console.log(`${symbol} extraction completed. Success: ${combinedData.success}`);
        if (errors.length > 0) {
          console.log(`${symbol} errors: ${errors.join(', ')}`);
        }
      }
      
      return combinedData;
      
    } catch (error) {
      console.error(`Symbol processing failed for ${symbol}:`, error);
      return {
        symbol: symbol,
        extraction_time: moment().format('YYYY-MM-DD HH:mm:ss'),
        processing_time_ms: Date.now() - startTime,
        error: error.message,
        success: false,
        sources: [],
        errors: [error.message]
      };
    }
  }

  async scrapeAll() {
    if (this.debug) console.log('Starting fixed scraping for all companies...');
    
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
              processing_time_ms: 0,
              error: result.reason.message,
              success: false,
              sources: [],
              errors: [result.reason.message]
            });
          }
        });
      }

      // Compile final results in TradingView format
      const tradingViewData = this.formatForTradingView(results);
      
      // Save results
      const outputPath = await this.saveResults(tradingViewData);
      
      if (this.debug) {
        console.log(`Fixed scraping completed in ${Date.now() - startTime}ms`);
        console.log(`Results saved to: ${outputPath}`);
        console.log(`Success rate: ${this.calculateSuccessRate(results)}%`);
      }

      return tradingViewData;

    } catch (error) {
      console.error('Fixed scraping process failed:', error);
      throw error;
    } finally {
      await this.close();
    }
  }

  formatForTradingView(results) {
    if (this.debug) console.log('Formatting results for TradingView compatibility...');
    
    const trading_data = results.map(result => {
      const timestamp = new Date().getTime();
      const companyName = result.symbol === 'NVDA' ? 'NVIDIA' : 
                         result.symbol === 'AAPL' ? 'APPLE' :
                         result.symbol === 'MSFT' ? 'MICROSOFT' :
                         result.symbol === 'GOOGL' ? 'GOOGLE' :
                         result.symbol === 'AMZN' ? 'AMAZON' :
                         result.symbol === 'TSLA' ? 'TESLA' : result.symbol;
      
      return {
        id: `${result.symbol.toLowerCase()}_tradingview_${timestamp}`,
        company: companyName,
        symbol: result.symbol,
        exchange: "NASDAQ",
        price_data: result.price_data || {
          current_price: "N/A",
          price_change: "N/A",
          price_change_percent: "N/A",
          daily_high: "N/A",
          daily_low: "N/A",
          volume: "N/A",
          market_cap: "N/A"
        },
        technical_indicators: result.technical_indicators || {
          rsi: "N/A",
          macd_signal: "N/A",
          ma_20: "N/A",
          ma_50: "N/A",
          ma_200: "N/A",
          indicators_found: 0
        },
        sentiment_indicators: result.sentiment_indicators || {
          ideas_count: 0,
          social_sentiment: "neutral",
          analyst_rating: "N/A",
          page_title: `${result.symbol} Analysis`
        },
        financial_data: result.financial_data || {},
        scraped_at: new Date().toISOString(),
        url: `https://www.tradingview.com/chart/?symbol=NASDAQ:${result.symbol}`,
        sources: result.sources || [],
        errors: result.errors || [],
        success: result.success || false
      };
    });

    return {
      fetch_metadata: {
        companies: this.companies.map(c => c.name.toLowerCase()),
        fetch_timestamp: new Date().toISOString(),
        total_companies: this.companies.length,
        source: "TradingView.com",
        scraper_version: "4.0.0-fixed-enhanced",
        extraction_method: "enhanced-accessibility-dom"
      },
      trading_data: trading_data
    };
  }

  calculateSuccessRate(results) {
    if (results.length === 0) return 0;
    
    const successfulCount = results.filter(result => result.success).length;
    return ((successfulCount / results.length) * 100).toFixed(2);
  }

  async saveResults(results) {
    if (this.debug) console.log('Saving fixed results...');
    
    try {
      // Save to Output/tradingview directory
      const outputDir = path.resolve(__dirname, this.options.outputDir);
      await fs.ensureDir(outputDir);
      
      // Use the same filename format as existing scraper
      const timestamp = moment().format('YYYYMMDD_HHmmss');
      const filename = `raw-tradingview_${timestamp}.json`;
      const filepath = path.join(outputDir, filename);
      
      await fs.writeJson(filepath, results, { spaces: 2 });
      
      if (this.debug) console.log(`Fixed results saved to: ${filepath}`);
      
      return filepath;
    } catch (error) {
      console.error('Error saving results:', error);
      throw error;
    }
  }

  async close() {
    if (this.debug) console.log('Closing fixed scraper...');
    
    await this.enhancedExtractor.close();
    
    if (this.debug) console.log('Fixed scraper closed');
  }

  // Method to scrape single symbol (for testing)
  async scrapeSingle(symbol) {
    if (this.debug) console.log(`Scraping single symbol with fixed scraper: ${symbol}`);
    
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
      const scraper = new FixedTradingViewScraper({ debug: true });
      
      try {
        const result = await scraper.scrapeSingle(symbol.toUpperCase());
        console.log('\n=== FIXED SCRAPING RESULTS ===');
        console.log(JSON.stringify(result, null, 2));
      } catch (error) {
        console.error('Fixed scraping failed:', error);
        process.exit(1);
      }
    })();
  } else {
    // All symbols scraping
    (async () => {
      const scraper = new FixedTradingViewScraper({ debug: true });
      
      try {
        const results = await scraper.scrapeAll();
        console.log('\n=== FIXED SCRAPING COMPLETED ===');
        console.log(`Total symbols: ${results.fetch_metadata.total_companies}`);
        console.log(`Extraction method: ${results.fetch_metadata.extraction_method}`);
        console.log(`Scraper version: ${results.fetch_metadata.scraper_version}`);
        
        // Print summary
        results.trading_data.forEach(company => {
          const price = company.price_data?.current_price || 'N/A';
          const change = company.price_data?.price_change || 'N/A';
          const sources = company.sources.join(', ') || 'none';
          const errors = company.errors.length > 0 ? ` (${company.errors.length} errors)` : '';
          console.log(`${company.symbol}: $${price} (${change}) [${sources}]${errors}`);
        });
      } catch (error) {
        console.error('Fixed scraping failed:', error);
        process.exit(1);
      }
    })();
  }
}

module.exports = FixedTradingViewScraper;