#!/usr/bin/env node

/**
 * Financials Extractor - Specialized TradingView Financials Page Scraper
 * 
 * Extracts financial data from TradingView financials page including:
 * - Market capitalization
 * - Price to earnings ratio
 * - Dividend yield
 * - Basic EPS
 * - Revenue data
 * - Profit margins
 * - Key financial metrics
 */

const { chromium } = require('playwright');
const fs = require('fs-extra');
const moment = require('moment');

class FinancialsExtractor {
  constructor(options = {}) {
    this.options = {
      headless: true,
      timeout: 60000,
      retryAttempts: 3,
      waitTime: 5000,
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
    
    if (this.debug) console.log('Financials browser launched successfully');
  }

  async createPage() {
    const page = await this.browser.newPage();
    
    await page.setExtraHTTPHeaders({
      'User-Agent': this.options.userAgent
    });
    await page.setViewportSize(this.options.viewport);
    
    // Block unnecessary resources
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

  async extractKeyFacts(page) {
    const keyFacts = {};
    
    try {
      // Wait for key facts section to load
      await page.waitForSelector('h2:has-text("Key facts")', { timeout: 15000 }).catch(() => {});
      
      // Extract market capitalization
      const marketCapLocator = page.getByText('Market capitalization');
      if (await marketCapLocator.count() > 0) {
        try {
          const parentElement = marketCapLocator.first().locator('..');
          const text = await parentElement.textContent();
          const match = text.match(/(\d+\.?\d*)\s*([TBM])\s*USD/);
          if (match) {
            keyFacts.market_cap = match[1] + match[2] + ' USD';
            keyFacts.market_cap_value = this.parseMarketCapValue(match[1], match[2]);
          }
        } catch (error) {
          if (this.debug) console.log(`Market cap extraction failed: ${error.message}`);
        }
      }
      
      // Extract dividend yield
      const dividendLocator = page.getByText('Dividend yield');
      if (await dividendLocator.count() > 0) {
        try {
          const parentElement = dividendLocator.first().locator('..');
          const text = await parentElement.textContent();
          const match = text.match(/(\d+\.?\d*)%/);
          if (match) {
            keyFacts.dividend_yield = match[1] + '%';
            keyFacts.dividend_yield_value = parseFloat(match[1]);
          }
        } catch (error) {
          if (this.debug) console.log(`Dividend yield extraction failed: ${error.message}`);
        }
      }
      
      // Extract P/E ratio
      const peRatioLocator = page.getByText('Price to earnings ratio');
      if (await peRatioLocator.count() > 0) {
        try {
          const parentElement = peRatioLocator.first().locator('..');
          const text = await parentElement.textContent();
          const match = text.match(/(\d+\.?\d*)/);
          if (match) {
            keyFacts.pe_ratio = parseFloat(match[1]);
          }
        } catch (error) {
          if (this.debug) console.log(`P/E ratio extraction failed: ${error.message}`);
        }
      }
      
      // Extract Basic EPS
      const epsLocator = page.getByText('Basic EPS');
      if (await epsLocator.count() > 0) {
        try {
          const parentElement = epsLocator.first().locator('..');
          const text = await parentElement.textContent();
          const match = text.match(/(\d+\.?\d*)\s*USD/);
          if (match) {
            keyFacts.eps = parseFloat(match[1]);
          }
        } catch (error) {
          if (this.debug) console.log(`EPS extraction failed: ${error.message}`);
        }
      }
      
      // Extract employees count
      const employeesLocator = page.getByText('Employees');
      if (await employeesLocator.count() > 0) {
        try {
          const parentElement = employeesLocator.first().locator('..');
          const text = await parentElement.textContent();
          const match = text.match(/(\d+)\s*([K])?/);
          if (match) {
            keyFacts.employees = match[1] + (match[2] || '');
            keyFacts.employees_value = parseInt(match[1]) * (match[2] === 'K' ? 1000 : 1);
          }
        } catch (error) {
          if (this.debug) console.log(`Employees extraction failed: ${error.message}`);
        }
      }
      
      // Extract founded year
      const foundedLocator = page.getByText('Founded');
      if (await foundedLocator.count() > 0) {
        try {
          const parentElement = foundedLocator.first().locator('..');
          const text = await parentElement.textContent();
          const match = text.match(/(\d{4})/);
          if (match) {
            keyFacts.founded = parseInt(match[1]);
          }
        } catch (error) {
          if (this.debug) console.log(`Founded year extraction failed: ${error.message}`);
        }
      }
      
      // Extract CEO name
      const ceoLocator = page.getByText('CEO');
      if (await ceoLocator.count() > 0) {
        try {
          const parentElement = ceoLocator.first().locator('..');
          const text = await parentElement.textContent();
          const match = text.match(/CEO\s+(.+)/);
          if (match) {
            keyFacts.ceo = match[1].trim();
          }
        } catch (error) {
          if (this.debug) console.log(`CEO extraction failed: ${error.message}`);
        }
      }
      
    } catch (error) {
      if (this.debug) console.log(`Key facts extraction failed: ${error.message}`);
    }
    
    return keyFacts;
  }

  async extractFinancialStatements(page) {
    const statements = {};
    
    try {
      // Navigate to statements tab
      const statementsTab = page.getByRole('tab', { name: 'Statements' });
      if (await statementsTab.count() > 0) {
        await statementsTab.click();
        await page.waitForTimeout(3000);
      }
      
      // Extract revenue data
      const revenueLocator = page.getByText('Total revenue');
      if (await revenueLocator.count() > 0) {
        try {
          const parentRow = revenueLocator.first().locator('..').locator('..');
          const text = await parentRow.textContent();
          
          // Extract latest quarter revenue
          const revenueMatch = text.match(/(\d+\.?\d*)\s*B/);
          if (revenueMatch) {
            statements.latest_revenue = revenueMatch[1] + 'B';
            statements.latest_revenue_value = parseFloat(revenueMatch[1]) * 1000000000;
          }
        } catch (error) {
          if (this.debug) console.log(`Revenue extraction failed: ${error.message}`);
        }
      }
      
      // Extract gross profit
      const grossProfitLocator = page.getByText('Gross profit');
      if (await grossProfitLocator.count() > 0) {
        try {
          const parentRow = grossProfitLocator.first().locator('..').locator('..');
          const text = await parentRow.textContent();
          
          const profitMatch = text.match(/(\d+\.?\d*)\s*B/);
          if (profitMatch) {
            statements.gross_profit = profitMatch[1] + 'B';
            statements.gross_profit_value = parseFloat(profitMatch[1]) * 1000000000;
          }
        } catch (error) {
          if (this.debug) console.log(`Gross profit extraction failed: ${error.message}`);
        }
      }
      
      // Extract operating income
      const operatingIncomeLocator = page.getByText('Operating income');
      if (await operatingIncomeLocator.count() > 0) {
        try {
          const parentRow = operatingIncomeLocator.first().locator('..').locator('..');
          const text = await parentRow.textContent();
          
          const incomeMatch = text.match(/(\d+\.?\d*)\s*B/);
          if (incomeMatch) {
            statements.operating_income = incomeMatch[1] + 'B';
            statements.operating_income_value = parseFloat(incomeMatch[1]) * 1000000000;
          }
        } catch (error) {
          if (this.debug) console.log(`Operating income extraction failed: ${error.message}`);
        }
      }
      
    } catch (error) {
      if (this.debug) console.log(`Financial statements extraction failed: ${error.message}`);
    }
    
    return statements;
  }

  parseMarketCapValue(value, unit) {
    const numValue = parseFloat(value);
    switch (unit) {
      case 'T': return numValue * 1000000000000;
      case 'B': return numValue * 1000000000;
      case 'M': return numValue * 1000000;
      default: return numValue;
    }
  }

  async extractFinancialData(symbol) {
    if (this.debug) console.log(`Starting financial data extraction for ${symbol}`);
    
    if (!this.browser) {
      await this.launch();
    }
    
    const page = await this.createPage();
    const url = `https://www.tradingview.com/symbols/${symbol}/financials/`;
    
    try {
      await page.goto(url, { waitUntil: 'networkidle', timeout: this.options.timeout });
      await page.waitForTimeout(this.options.waitTime);
      
      // Extract key facts and financial statements in parallel
      const [keyFacts, statements] = await Promise.all([
        this.extractKeyFacts(page),
        this.extractFinancialStatements(page)
      ]);
      
      const financialData = {
        symbol: symbol,
        url: url,
        extraction_time: moment().format('YYYY-MM-DD HH:mm:ss'),
        key_facts: keyFacts,
        financial_statements: statements
      };
      
      if (this.debug) {
        console.log('Financial data extracted:', JSON.stringify(financialData, null, 2));
      }
      
      return financialData;
      
    } catch (error) {
      console.error(`Financial data extraction failed for ${symbol}:`, error.message);
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

module.exports = FinancialsExtractor;

// Test if called directly
if (require.main === module) {
  (async () => {
    const extractor = new FinancialsExtractor({ debug: true });
    
    try {
      const data = await extractor.extractFinancialData('NVDA');
      console.log('Final result:', JSON.stringify(data, null, 2));
    } catch (error) {
      console.error('Test failed:', error);
    } finally {
      await extractor.close();
    }
  })();
} 