#!/usr/bin/env python3
"""
Combined Sentiment Analysis Generator
Combines data from all 4 sentiment analysis files into a single comprehensive file.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

class SentimentCombiner:
    def __init__(self, base_path: str = None):
        if base_path is None:
            # Use absolute path from project root
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            self.base_path = os.path.join(project_root, "container_output", "final_score")
        else:
            self.base_path = base_path
        self.combined_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "source_files": [
                    "final-weighted-scores.json",
                    "news-sentiment-analysis.json", 
                    "image-sentiment-analysis.json",
                    "tradingview-sentiment-analysis.json",
                    "raw-tradingview_2025-07-17_0756.json"
                ],
                "combination_method": "comprehensive_merge",
                "total_companies": 0
            },
            "companies": {}
        }
        
    def load_file(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load a JSON file and return its contents."""
        filepath = os.path.join(self.base_path, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️  Warning: {filename} not found")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ Error reading {filename}: {e}")
            return None
    
    def combine_all_data(self):
        """Combine data from all sentiment analysis files."""
        
        # Load all files
        final_scores = self.load_file("final-weighted-scores.json")
        news_sentiment = self.load_file("news-sentiment-analysis.json")
        image_sentiment = self.load_file("image-sentiment-analysis.json")
        
        # Load TradingView sentiment from its specific location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        tradingview_path = os.path.join(project_root, "container_output", "tradingview", "tradingview-sentiment-analysis.json")
        try:
            with open(tradingview_path, 'r', encoding='utf-8') as f:
                tradingview_sentiment = json.load(f)
        except FileNotFoundError:
            print(f"⚠️  Warning: TradingView sentiment file not found at {tradingview_path}")
            tradingview_sentiment = None
        except json.JSONDecodeError as e:
            print(f"❌ Error reading TradingView sentiment file: {e}")
            tradingview_sentiment = None
        
        # Load raw TradingView data file
        raw_tradingview_path = os.path.join(project_root, "container_output", "tradingview", "raw-tradingview_2025-07-17_0756.json")
        try:
            with open(raw_tradingview_path, 'r', encoding='utf-8') as f:
                raw_tradingview_data = json.load(f)
        except FileNotFoundError:
            print(f"⚠️  Warning: Raw TradingView data file not found at {raw_tradingview_path}")
            raw_tradingview_data = None
        except json.JSONDecodeError as e:
            print(f"❌ Error reading raw TradingView data file: {e}")
            raw_tradingview_data = None
        
        # Get list of all companies
        all_companies = set()
        
        if final_scores and "companies" in final_scores:
            all_companies.update(final_scores["companies"].keys())
        if news_sentiment and "companies" in news_sentiment:
            all_companies.update(news_sentiment["companies"].keys())
        if image_sentiment and "company_results" in image_sentiment:
            all_companies.update(image_sentiment["company_results"].keys())
        if tradingview_sentiment and "companies" in tradingview_sentiment:
            all_companies.update(tradingview_sentiment["companies"].keys())
        
        print(f"📊 Found {len(all_companies)} companies: {', '.join(sorted(all_companies))}")
        
        # Combine data for each company
        for company in sorted(all_companies):
            self.combined_data["companies"][company] = self.combine_company_data(
                company, final_scores, news_sentiment, image_sentiment, tradingview_sentiment, raw_tradingview_data
            )
        
        self.combined_data["metadata"]["total_companies"] = len(all_companies)
        
    def combine_company_data(self, company: str, final_scores: Dict, news_sentiment: Dict, 
                           image_sentiment: Dict, tradingview_sentiment: Dict, raw_tradingview_data: Dict) -> Dict[str, Any]:
        """Combine all sentiment data for a single company."""
        
        company_data = {
            "company_name": company,
            "final_weighted_score": None,
            "sentiment_sources": {
                "news": None,
                "images": None,
                "tradingview": None
            },
            "detailed_analysis": {
                "news_data": None,
                "image_data": None,
                "tradingview_data": None
            },
            "summary": {
                "total_data_points": 0,
                "available_sources": [],
                "sentiment_trend": "neutral",
                "confidence_level": "medium"
            }
        }
        
        # Extract final weighted score
        if final_scores and "companies" in final_scores and company in final_scores["companies"]:
            fs_data = final_scores["companies"][company]
            company_data["final_weighted_score"] = fs_data.get("final_score")
            company_data["sentiment_sources"]["news"] = fs_data.get("news_score")
            company_data["sentiment_sources"]["images"] = fs_data.get("image_score")
        
        # Extract news sentiment
        if news_sentiment and "companies" in news_sentiment and company in news_sentiment["companies"]:
            ns_data = news_sentiment["companies"][company]
            company_data["sentiment_sources"]["news"] = ns_data.get("score")
            company_data["detailed_analysis"]["news_data"] = {
                "score": ns_data.get("score"),
                "articles_analyzed": ns_data.get("articles_analyzed", 0),
                "news_articles": ns_data.get("source_breakdown", {}).get("news_articles", 0),
                "reddit_mentions": ns_data.get("source_breakdown", {}).get("reddit_mentions", 0),
                "reddit_images": ns_data.get("source_breakdown", {}).get("reddit_images", 0),
                "last_analysis": ns_data.get("last_analysis_date"),
                "source": ns_data.get("source", "Combined (NewsAPI + Reddit)")
            }
            company_data["summary"]["total_data_points"] += ns_data.get("articles_analyzed", 0)
            company_data["summary"]["available_sources"].append("news")
        
        # Extract image sentiment
        if image_sentiment and "company_results" in image_sentiment and company in image_sentiment["company_results"]:
            is_data = image_sentiment["company_results"][company]
            company_data["sentiment_sources"]["images"] = is_data.get("average_sentiment")
            company_data["detailed_analysis"]["image_data"] = {
                "average_score": is_data.get("average_sentiment"),
                "total_images": is_data.get("total_images", 0),
                "sentiment_scores": is_data.get("sentiment_scores", []),
                "tradingview_images": is_data.get("tradingview_images", 0),
                "reddit_images": is_data.get("reddit_images", 0),
                "other_images": is_data.get("other_images", 0)
            }
            company_data["summary"]["total_data_points"] += is_data.get("total_images", 0)
            company_data["summary"]["available_sources"].append("images")
        
        # Extract TradingView sentiment and raw data
        if tradingview_sentiment and "companies" in tradingview_sentiment and company in tradingview_sentiment["companies"]:
            tv_data = tradingview_sentiment["companies"][company]
            company_data["sentiment_sources"]["tradingview"] = tv_data.get("score")
            
            # Find corresponding raw TradingView data
            raw_tv_data = None
            if raw_tradingview_data and "trading_data" in raw_tradingview_data:
                for raw_company in raw_tradingview_data["trading_data"]:
                    if raw_company.get("symbol") == tv_data.get("symbol"):
                        raw_tv_data = raw_company
                        break
            
            company_data["detailed_analysis"]["tradingview_data"] = {
                "sentiment_analysis": {
                    "score": tv_data.get("score"),
                    "symbol": tv_data.get("symbol"),
                    "data_points_analyzed": tv_data.get("data_points_analyzed", 0),
                    "sentiment_breakdown": tv_data.get("sentiment_breakdown", {}),
                    "analysis_source": tv_data.get("analysis_source", "TradingView Data + GPT-4 Analysis")
                },
                "price_data": tv_data.get("price_data", {}),
                "raw_trading_data": raw_tv_data.get("price_data", {}) if raw_tv_data else {},
                "technical_indicators": raw_tv_data.get("technical_indicators", {}) if raw_tv_data else {},
                "sentiment_indicators": raw_tv_data.get("sentiment_indicators", {}) if raw_tv_data else {},
                "extraction_metadata": raw_tv_data.get("extraction_metadata", {}) if raw_tv_data else {}
            }
            company_data["summary"]["total_data_points"] += tv_data.get("data_points_analyzed", 0)
            company_data["summary"]["available_sources"].append("tradingview")
            
            # Add price data to top level if available
            if tv_data.get("price_data"):
                company_data["price_data"] = tv_data.get("price_data")
        
        # Calculate summary metrics
        company_data["summary"] = self.calculate_summary_metrics(company_data)
        
        return company_data
    
    def calculate_summary_metrics(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary metrics for a company."""
        summary = company_data["summary"]
        
        # Calculate sentiment trend
        scores = []
        final_score = company_data.get("final_weighted_score")
        if final_score:
            scores.append(final_score)
        
        for source in ["news", "images", "tradingview"]:
            score = company_data["sentiment_sources"].get(source)
            if score:
                scores.append(score)
        
        if scores:
            avg_score = sum(scores) / len(scores)
            if avg_score >= 7.5:
                summary["sentiment_trend"] = "very_bullish"
            elif avg_score >= 6.5:
                summary["sentiment_trend"] = "bullish"
            elif avg_score >= 5.5:
                summary["sentiment_trend"] = "neutral_positive"
            elif avg_score >= 4.5:
                summary["sentiment_trend"] = "neutral"
            elif avg_score >= 3.5:
                summary["sentiment_trend"] = "neutral_negative"
            elif avg_score >= 2.5:
                summary["sentiment_trend"] = "bearish"
            else:
                summary["sentiment_trend"] = "very_bearish"
        
        # Calculate confidence level
        source_count = len(summary["available_sources"])
        if source_count >= 3:
            summary["confidence_level"] = "high"
        elif source_count >= 2:
            summary["confidence_level"] = "medium"
        else:
            summary["confidence_level"] = "low"
        
        return summary
    
    def save_combined_file(self, output_path: str = None):
        """Save the combined sentiment analysis to a file."""
        if output_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            output_path = os.path.join(project_root, "container_output", "final_score", "combine-tradingview-sentiment-analysis.json")
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.combined_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Combined sentiment analysis saved to: {output_path}")
            print(f"📊 Total companies: {self.combined_data['metadata']['total_companies']}")
            
            # Display summary
            self.display_summary()
            
        except Exception as e:
            print(f"❌ Error saving combined file: {e}")
            raise

    def display_summary(self):
        """Display a summary of the combined analysis."""
        print("\n" + "="*80)
        print("📈 COMBINED SENTIMENT ANALYSIS SUMMARY")
        print("="*80)
        
        for company, data in self.combined_data["companies"].items():
            final_score = data.get("final_weighted_score", "N/A")
            sources = ", ".join(data["summary"]["available_sources"])
            trend = data["summary"]["sentiment_trend"].replace("_", " ").title()
            confidence = data["summary"]["confidence_level"].title()
            
            print(f"\n🏢 {company}")
            print(f"   Final Score: {final_score}/10")
            print(f"   Sources: {sources}")
            print(f"   Trend: {trend}")
            print(f"   Confidence: {confidence}")
            print(f"   Total Data Points: {data['summary']['total_data_points']}")

def create_final_sentiment_analysis():
    """Create final comprehensive sentiment analysis using all combined files."""
    print("🎯 Starting FINAL Comprehensive Sentiment Analysis Generation...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    final_score_dir = os.path.join(project_root, "container_output", "final_score")
    
    # Load all analysis files
    source_files = {
        "combine_tradingview": "combine-tradingview-sentiment-analysis.json",
        "image_sentiment": "image-sentiment-analysis.json", 
        "news_sentiment": "news-sentiment-analysis.json",
        "weighted_scores": "final-weighted-scores.json"
    }
    
    loaded_data = {}
    for key, filename in source_files.items():
        filepath = os.path.join(final_score_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_data[key] = json.load(f)
            print(f"✅ Loaded: {filename}")
        except FileNotFoundError:
            print(f"⚠️  Warning: {filename} not found")
            loaded_data[key] = None
        except json.JSONDecodeError as e:
            print(f"❌ Error reading {filename}: {e}")
            loaded_data[key] = None
    
    # Create final comprehensive structure
    final_analysis = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "analysis_type": "final_comprehensive_sentiment",
            "source_files": list(source_files.values()),
            "total_companies": 0,
            "analysis_version": "2.0.0"
        },
        "companies": {}
    }
    
    # Get all companies from all sources
    all_companies = set()
    
    if loaded_data["combine_tradingview"] and "companies" in loaded_data["combine_tradingview"]:
        all_companies.update(loaded_data["combine_tradingview"]["companies"].keys())
    if loaded_data["news_sentiment"] and "companies" in loaded_data["news_sentiment"]:
        all_companies.update(loaded_data["news_sentiment"]["companies"].keys())
    if loaded_data["image_sentiment"] and "company_results" in loaded_data["image_sentiment"]:
        all_companies.update(loaded_data["image_sentiment"]["company_results"].keys())
    if loaded_data["weighted_scores"] and "companies" in loaded_data["weighted_scores"]:
        all_companies.update(loaded_data["weighted_scores"]["companies"].keys())
    
    print(f"📊 Found {len(all_companies)} companies for final analysis: {', '.join(sorted(all_companies))}")
    
    # Process each company
    for company in sorted(all_companies):
        final_analysis["companies"][company] = create_final_company_analysis(company, loaded_data)
    
    final_analysis["metadata"]["total_companies"] = len(all_companies)
    
    # Save final analysis
    output_path = os.path.join(final_score_dir, "final-comprehensive-sentiment-analysis.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_analysis, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Final comprehensive sentiment analysis saved to: {output_path}")
    print(f"📊 Total companies: {len(all_companies)}")
    
    # Display final summary
    display_final_summary(final_analysis)
    
    return final_analysis

def create_final_company_analysis(company: str, loaded_data: Dict) -> Dict:
    """Create final comprehensive analysis for a single company."""
    
    company_analysis = {
        "company_name": company,
        "final_scores": {
            "comprehensive_score": None,
            "weighted_score": None,
            "combined_tradingview_score": None
        },
        "all_sentiment_sources": {
            "news": None,
            "images": None,
            "tradingview": None,
            "combined_tradingview": None
        },
        "comprehensive_data": {
            "news_analysis": None,
            "image_analysis": None,
            "tradingview_raw_data": None,
            "tradingview_sentiment": None,
            "weighted_breakdown": None
        },
        "final_metrics": {
            "total_data_points": 0,
            "confidence_level": "low",
            "sentiment_trend": "neutral",
            "data_sources_count": 0,
            "recommendation": "hold"
        }
    }
    
    data_sources = []
    all_scores = []
    
    # Extract from combine-tradingview analysis
    if loaded_data["combine_tradingview"] and company in loaded_data["combine_tradingview"].get("companies", {}):
        ctv_data = loaded_data["combine_tradingview"]["companies"][company]
        company_analysis["final_scores"]["combined_tradingview_score"] = ctv_data.get("final_weighted_score")
        company_analysis["all_sentiment_sources"]["combined_tradingview"] = ctv_data.get("final_weighted_score")
        
        # Extract detailed data
        if "detailed_analysis" in ctv_data:
            company_analysis["comprehensive_data"]["news_analysis"] = ctv_data["detailed_analysis"].get("news_data")
            company_analysis["comprehensive_data"]["image_analysis"] = ctv_data["detailed_analysis"].get("image_data")
            
            # Safely extract TradingView data with null checks
            tradingview_data = ctv_data["detailed_analysis"].get("tradingview_data")
            if tradingview_data and isinstance(tradingview_data, dict):
                company_analysis["comprehensive_data"]["tradingview_raw_data"] = tradingview_data.get("raw_trading_data")
                company_analysis["comprehensive_data"]["tradingview_sentiment"] = tradingview_data.get("sentiment_analysis")
            else:
                company_analysis["comprehensive_data"]["tradingview_raw_data"] = None
                company_analysis["comprehensive_data"]["tradingview_sentiment"] = None
        
        # Extract price data if available
        if "price_data" in ctv_data:
            company_analysis["price_data"] = ctv_data["price_data"]
        
        company_analysis["final_metrics"]["total_data_points"] += ctv_data.get("summary", {}).get("total_data_points", 0)
        data_sources.append("combined_tradingview")
        if ctv_data.get("final_weighted_score"):
            all_scores.append(ctv_data.get("final_weighted_score"))
    
    # Extract from weighted scores
    if loaded_data["weighted_scores"] and company in loaded_data["weighted_scores"].get("companies", {}):
        ws_data = loaded_data["weighted_scores"]["companies"][company]
        company_analysis["final_scores"]["weighted_score"] = ws_data.get("final_score")
        company_analysis["comprehensive_data"]["weighted_breakdown"] = ws_data
        data_sources.append("weighted_scores")
        if ws_data.get("final_score"):
            all_scores.append(ws_data.get("final_score"))
    
    # Extract individual sentiment sources for cross-reference
    if loaded_data["news_sentiment"] and company in loaded_data["news_sentiment"].get("companies", {}):
        ns_data = loaded_data["news_sentiment"]["companies"][company]
        company_analysis["all_sentiment_sources"]["news"] = ns_data.get("score")
        data_sources.append("news_sentiment")
    
    if loaded_data["image_sentiment"] and company in loaded_data["image_sentiment"].get("company_results", {}):
        is_data = loaded_data["image_sentiment"]["company_results"][company]
        company_analysis["all_sentiment_sources"]["images"] = is_data.get("average_sentiment")
        data_sources.append("image_sentiment")
    
    # Calculate final comprehensive score
    if all_scores:
        company_analysis["final_scores"]["comprehensive_score"] = sum(all_scores) / len(all_scores)
    
    # Calculate final metrics
    company_analysis["final_metrics"]["data_sources_count"] = len(data_sources)
    
    # Determine confidence level
    if len(data_sources) >= 3:
        company_analysis["final_metrics"]["confidence_level"] = "high"
    elif len(data_sources) >= 2:
        company_analysis["final_metrics"]["confidence_level"] = "medium"
    else:
        company_analysis["final_metrics"]["confidence_level"] = "low"
    
    # Determine sentiment trend and recommendation
    final_score = company_analysis["final_scores"]["comprehensive_score"] or 5.0
    
    if final_score >= 8.0:
        company_analysis["final_metrics"]["sentiment_trend"] = "very_bullish"
        company_analysis["final_metrics"]["recommendation"] = "strong_buy"
    elif final_score >= 7.0:
        company_analysis["final_metrics"]["sentiment_trend"] = "bullish"
        company_analysis["final_metrics"]["recommendation"] = "buy"
    elif final_score >= 6.0:
        company_analysis["final_metrics"]["sentiment_trend"] = "neutral_positive"
        company_analysis["final_metrics"]["recommendation"] = "hold_positive"
    elif final_score >= 5.0:
        company_analysis["final_metrics"]["sentiment_trend"] = "neutral"
        company_analysis["final_metrics"]["recommendation"] = "hold"
    elif final_score >= 4.0:
        company_analysis["final_metrics"]["sentiment_trend"] = "neutral_negative"
        company_analysis["final_metrics"]["recommendation"] = "hold_negative"
    elif final_score >= 3.0:
        company_analysis["final_metrics"]["sentiment_trend"] = "bearish"
        company_analysis["final_metrics"]["recommendation"] = "sell"
    else:
        company_analysis["final_metrics"]["sentiment_trend"] = "very_bearish"
        company_analysis["final_metrics"]["recommendation"] = "strong_sell"
    
    return company_analysis

def display_final_summary(final_analysis: Dict):
    """Display summary of the final comprehensive analysis."""
    print("\n" + "="*90)
    print("🎯 FINAL COMPREHENSIVE SENTIMENT ANALYSIS SUMMARY")
    print("="*90)
    
    for company, data in final_analysis["companies"].items():
        final_score = data["final_scores"]["comprehensive_score"]
        confidence = data["final_metrics"]["confidence_level"]
        trend = data["final_metrics"]["sentiment_trend"].replace("_", " ").title()
        recommendation = data["final_metrics"]["recommendation"].replace("_", " ").title()
        sources = data["final_metrics"]["data_sources_count"]
        
        print(f"\n🏢 {company}")
        print(f"   Final Score: {final_score:.1f}/10" if final_score else "   Final Score: N/A")
        print(f"   Trend: {trend}")
        print(f"   Recommendation: {recommendation}")
        print(f"   Confidence: {confidence.title()} ({sources} sources)")

def main():
    """Main execution function."""
    print("🎯 Starting Combined Sentiment Analysis Generation...")
    
    combiner = SentimentCombiner()
    combiner.combine_all_data()
    combiner.save_combined_file()
    
    print("\n✅ Combined sentiment analysis generation completed!")
    
    # Create final comprehensive analysis
    print("\n" + "="*50)
    create_final_sentiment_analysis()
    
    print("\n✅ Final comprehensive sentiment analysis completed!")

if __name__ == "__main__":
    main()