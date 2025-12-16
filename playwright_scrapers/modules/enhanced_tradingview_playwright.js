#!/usr/bin/env node

/**
 * Enhanced TradingView Scraper - Comprehensive Playwright Implementation
 * 
 * Multi-Section Data Extraction:
 * - Chart data (OHLC, volume, price changes)
 * - Financial data (market cap, P/E ratio, EPS)
 * - Technical analysis (indicators, signals)
 * - Statements data (revenue, profits)
 * 
 * Features:
 * - Parallel extraction from multiple TradingView sections
 * - Modern Playwright selectors and techniques
 * - Comprehensive error handling and retry mechanisms
 * - Proper data aggregation and formatting
 * - Enhanced performance with optimized strategies
 */

const { chromium } = require('playwright');
const fs = require('fs-extra');
const path = require('path');
const moment = require('moment');

// Import specialized extractors (now in same directory)
const ChartDataExtractor = require('./chart_data_extractor');
const FinancialsExtractor = require('./financials_extractor');
const TechnicalIndicatorsExtractor = require('./technical_indicators_extractor');

class EnhancedTradingViewScraper {
  constructor(options = {}) {
    this.options = {
      headless: true,
      timeout: 60000,
      concurrency: 3, // Number of parallel extractors
      debug: false,
      outputDir: '../../Output/tradingview',
      ...options
    };
    
    this.debug = this.options.debug;
    this.companies = [];
    this.results = {};
    
    // Initialize extractors
    this.chartExtractor = new ChartDataExtractor({ debug: this.debug });
    this.financialsExtractor = new FinancialsExtractor({ debug: this.debug });
    this.technicalExtractor = new TechnicalIndicatorsExtractor({ debug: this.debug });
  }

  async initialize() {
    if (this.debug) console.log('Initializing enhanced TradingView scraper...');
    
    // Initialize all extractors in parallel
    await Promise.all([
      this.chartExtractor.launch(),
      this.financialsExtractor.launch(),
      this.technicalExtractor.launch()
    ]);
    
    if (this.debug) console.log('All extractors initialized successfully');
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
    if (this.debug) console.log(`Starting comprehensive extraction for ${symbol}`);
    
    const startTime = Date.now();
    
    try {
      // Run all extractors in parallel for maximum efficiency
      const extractionPromises = [
        this.chartExtractor.extractChartData(symbol),
        this.financialsExtractor.extractFinancialData(symbol),
        this.technicalExtractor.extractTechnicalData(symbol)
      ];
      
      const results = await Promise.allSettled(extractionPromises);
      
      // Process results and combine data
      const combinedData = {
        symbol: symbol,
        extraction_time: moment().format('YYYY-MM-DD HH:mm:ss'),
        processing_time_ms: Date.now() - startTime,
        sources: []
      };
      
      const errors = [];
      
      // Process chart data
      if (results[0].status === 'fulfilled' && !results[0].value.error) {
        const chartData = results[0].value;
        combinedData.sources.push('chart');
        
        // Map chart data to standard format
        combinedData.price_data = {
          current_price: chartData.current_price || chartData.close || "N/A",
          price_change: chartData.price_change || "N/A",
          price_change_percent: chartData.price_change_percent || "N/A",
          daily_high: chartData.high || "N/A",
          daily_low: chartData.low || "N/A",
          open: chartData.open || "N/A",
          close: chartData.close || "N/A",
          volume: chartData.volume || "N/A",
          volume_formatted: chartData.volume_formatted || "N/A"
        };
        
        if (chartData.post_market_change) {
          combinedData.price_data.post_market_change = chartData.post_market_change;
          combinedData.price_data.post_market_change_percent = chartData.post_market_change_percent;
        }
      } else {
        errors.push(`Chart extraction: ${results[0].reason?.message || results[0].value?.error || 'Unknown error'}`);
      }
      
      // Process financial data
      if (results[1].status === 'fulfilled' && !results[1].value.error) {
        const financialData = results[1].value;
        combinedData.sources.push('financials');
        
        // Map financial data to standard format
        combinedData.financial_data = {
          market_cap: financialData.key_facts?.market_cap || "N/A",
          pe_ratio: financialData.key_facts?.pe_ratio || "N/A",
          dividend_yield: financialData.key_facts?.dividend_yield || "N/A",
          eps: financialData.key_facts?.eps || "N/A",
          employees: financialData.key_facts?.employees || "N/A",
          founded: financialData.key_facts?.founded || "N/A",
          ceo: financialData.key_facts?.ceo || "N/A"
        };
        
        if (financialData.financial_statements) {
          combinedData.financial_data.latest_revenue = financialData.financial_statements.latest_revenue || "N/A";
          combinedData.financial_data.gross_profit = financialData.financial_statements.gross_profit || "N/A";
          combinedData.financial_data.operating_income = financialData.financial_statements.operating_income || "N/A";
        }
      } else {
        errors.push(`Financial extraction: ${results[1].reason?.message || results[1].value?.error || 'Unknown error'}`);
      }
      
      // Process technical data
      if (results[2].status === 'fulfilled' && !results[2].value.error) {
        const technicalData = results[2].value;
        combinedData.sources.push('technical');
        
        // Map technical data to standard format
        combinedData.technical_indicators = {
          technical_summary: technicalData.technical_summary || "N/A",
          oscillators: technicalData.oscillators || "N/A",
          moving_averages: technicalData.moving_averages || "N/A",
          rsi: technicalData.rsi || "N/A",
          macd_signal: technicalData.macd_signal || "N/A",
          ma_20: technicalData.ma_20 || "N/A",
          ma_50: technicalData.ma_50 || "N/A",
          ma_200: technicalData.ma_200 || "N/A",
          indicators_found: technicalData.indicators_found || 0
        };
      } else {
        errors.push(`Technical extraction: ${results[2].reason?.message || results[2].value?.error || 'Unknown error'}`);
      }
      
      // Add sentiment indicators (placeholder for compatibility)
      combinedData.sentiment_indicators = {
        ideas_count: 0,
        social_sentiment: "neutral",
        analyst_rating: "N/A",
        page_title: `${symbol} Analysis`
      };
      
      combinedData.errors = errors;
      combinedData.success = errors.length < 3; // Success if at least one extractor worked
      
      if (this.debug) {
        console.log(`${symbol} extraction completed. Sources: ${combinedData.sources.join(', ')}`);
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
    if (this.debug) console.log('Starting comprehensive scraping for all companies...');
    
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
        console.log(`Comprehensive scraping completed in ${Date.now() - startTime}ms`);
        console.log(`Results saved to: ${outputPath}`);
        console.log(`Success rate: ${this.calculateSuccessRate(results)}%`);
      }

      return tradingViewData;

    } catch (error) {
      console.error('Comprehensive scraping process failed:', error);
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
        scraper_version: "3.0.0-enhanced-playwright",
        extraction_method: "multi-section-parallel"
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
    if (this.debug) console.log('Saving comprehensive results...');
    
    try {
      // Save to Output/tradingview directory
      const outputDir = path.resolve(__dirname, this.options.outputDir);
      await fs.ensureDir(outputDir);
      
      // Use the same filename format as existing scraper
      const timestamp = moment().format('YYYYMMDD_HHmmss');
      const filename = `raw-tradingview_${timestamp}.json`;
      const filepath = path.join(outputDir, filename);
      
      await fs.writeJson(filepath, results, { spaces: 2 });
      
      if (this.debug) console.log(`Results saved to: ${filepath}`);
      
      return filepath;
    } catch (error) {
      console.error('Error saving results:', error);
      throw error;
    }
  }

  async close() {
    if (this.debug) console.log('Closing all extractors...');
    
    await Promise.all([
      this.chartExtractor.close(),
      this.financialsExtractor.close(),
      this.technicalExtractor.close()
    ]);
    
    if (this.debug) console.log('All extractors closed');
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
      const scraper = new EnhancedTradingViewScraper({ debug: true });
      
      try {
        const result = await scraper.scrapeSingle(symbol.toUpperCase());
        console.log('\n=== COMPREHENSIVE SCRAPING RESULTS ===');
        console.log(JSON.stringify(result, null, 2));
      } catch (error) {
        console.error('Scraping failed:', error);
        process.exit(1);
      }
    })();
  } else {
    // All symbols scraping
    (async () => {
      const scraper = new EnhancedTradingViewScraper({ debug: true });
      
      try {
        const results = await scraper.scrapeAll();
        console.log('\n=== COMPREHENSIVE SCRAPING COMPLETED ===');
        console.log(`Total symbols: ${results.fetch_metadata.total_companies}`);
        console.log(`Extraction method: ${results.fetch_metadata.extraction_method}`);
        console.log(`Scraper version: ${results.fetch_metadata.scraper_version}`);
        
        // Print summary
        results.trading_data.forEach(company => {
          const price = company.price_data?.current_price || 'N/A';
          const technical = company.technical_indicators?.technical_summary || 'N/A';
          const sources = company.sources.join(', ') || 'none';
          const errors = company.errors.length > 0 ? ` (${company.errors.length} errors)` : '';
          console.log(`${company.symbol}: $${price} - ${technical} [${sources}]${errors}`);
        });
      } catch (error) {
        console.error('Comprehensive scraping failed:', error);
        process.exit(1);
      }
    })();
  }
}

module.exports = EnhancedTradingViewScraper; 