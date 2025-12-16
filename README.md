# Stock Sentiment Analysis

## 🎯 Automated Workflow (Recommended)

### Weighted Sentiment Score Aggregation System

**Important Note**

Before running `weighted_sentiment_aggregator.py`, you must first generate fresh data:

- Run the news sentiment analysis script → this creates `combine-sentiment_analysis.json`
- Run the image sentiment analysis script → this creates `combine-image-sentimental_analysis.json`

Do not run `weighted_sentiment_aggregator.py` before generating these files, as it depends on them.

**One-command solution** - automatically handles everything:
```bash
python3 weighted_sentiment_aggregator.py
```

**Options:**
```bash
python3 weighted_sentiment_aggregator.py --force-refresh  # Force fresh data
python3 weighted_sentiment_aggregator.py --quiet         # Use existing data
```

## 📈 Enhanced Workflow (TradingView Charts)

For improved image sentiment analysis with professional chart data:
```bash
# Generate TradingView chart screenshots for all companies
python3 utils/headless_charts.py

# Then run automated workflow
python3 weighted_sentiment_aggregator.py
```

**Benefits of TradingView charts:**
- 📊 Professional technical analysis (typically 6-8/10 sentiment)
- 🎯 Balanced perspective vs Reddit images (typically 3-4/10 sentiment)  
- 📈 Chart-based sentiment for all companies (not just those with Reddit images)

## 📋 Manual Workflow (If Needed)

### Step 1: Collect Data
```bash
cd news && python3 news_dump.py
```

### Step 2: Generate TradingView Charts (Optional but Recommended)
```bash
python3 utils/headless_charts.py
```
*Requires Chrome/Chromium browser installed*

### Step 3: Analyze Sentiment  
```bash
cd news && python3 sentiment_analyzer.py dumps/combined_news_dump_YYYYMMDD_HHMMSS.json
```

### Step 4: Analyze Images
```bash
cd news && python3 image_sentiment_analyzer.py
```
*Analyzes both Reddit images and TradingView charts*

### Step 5: Calculate Weighted Scores
```bash
python3 weighted_sentiment_aggregator.py
```

## 📊 Results
- `final_weighted_scores.json` - **Main output** with weighted sentiment scores
- `news/combine-sentiment_analysis.json` - News sentiment scores (1-10 scale)
- `news/combine-image-sentimental_analysis.json` - Image sentiment scores