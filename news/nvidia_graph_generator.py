#!/usr/bin/env python3
"""
NVIDIA Advanced Graph Generator
Creates comprehensive financial visualization dashboard using Context 7 best practices.
Generates multiple chart types with seaborn and matplotlib for professional analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class NvidiaAdvancedVisualizer:
  def __init__(self, csv_path: str = None, output_dir: str = None):
    if csv_path is None:
      script_dir = os.path.dirname(os.path.abspath(__file__))
      project_root = os.path.dirname(script_dir)
      self.csv_path = os.path.join(project_root, "web", "nvidia_score_price_dump.txt")
    else:
      self.csv_path = csv_path
    
    if output_dir is None:
      script_dir = os.path.dirname(os.path.abspath(__file__))
      project_root = os.path.dirname(script_dir)
      self.output_dir = os.path.join(project_root, "container_output", "graphs")
    else:
      self.output_dir = output_dir
    
    # Ensure output directory exists
    os.makedirs(self.output_dir, exist_ok=True)
    
    self.setup_professional_style()
  
  def setup_professional_style(self):
    """Setup Context 7 professional financial visualization style."""
    # Set seaborn theme with financial-appropriate styling
    sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
    
    # Configure matplotlib for high-quality financial charts
    plt.rcParams.update({
      'figure.figsize': (16, 12),
      'figure.dpi': 100,
      'savefig.dpi': 300,
      'font.size': 11,
      'axes.titlesize': 14,
      'axes.labelsize': 12,
      'xtick.labelsize': 10,
      'ytick.labelsize': 10,
      'legend.fontsize': 11,
      'figure.titlesize': 16,
      'lines.linewidth': 2.5,
      'grid.alpha': 0.3,
      'figure.facecolor': 'white',
      'axes.facecolor': 'white',
      'savefig.facecolor': 'white',
      'axes.spines.top': False,
      'axes.spines.right': False
    })
    
    # Define professional financial color palette
    self.colors = {
      'bullish': '#2E8B57',     # Sea Green
      'neutral': '#4169E1',     # Royal Blue
      'bearish': '#DC143C',     # Crimson
      'price': '#1f77b4',       # Blue
      'sentiment': '#ff7f0e',   # Orange
      'ma_short': '#9467bd',    # Purple
      'ma_long': '#8c564b',     # Brown
      'volume': '#7f7f7f',      # Gray
      'grid': '#e0e0e0'         # Light Gray
    }
  
  def load_and_enhance_data(self) -> Optional[pd.DataFrame]:
    """Load CSV data and add advanced technical indicators."""
    try:
      if not os.path.exists(self.csv_path):
        print(f"❌ Data file not found: {self.csv_path}")
        return None
      
      # Read CSV with proper handling
      df = pd.read_csv(self.csv_path, skipinitialspace=True)
      df.columns = [col.strip().lower() for col in df.columns]
      
      print(f"✅ Loaded {len(df)} records from {self.csv_path}")
      
      # Parse timestamp
      if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
      
      # Validate required columns
      if not all(col in df.columns for col in ['score', 'price']):
        print("❌ Missing required columns (score, price)")
        return None
      
      # Add comprehensive technical indicators
      df = self.add_advanced_indicators(df)
      
      return df
      
    except Exception as e:
      print(f"❌ Error loading data: {e}")
      return None
  
  def add_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive technical analysis indicators."""
    if len(df) < 2:
      return df
    
    # Moving averages for both price and sentiment
    for window in [3, 5, 7]:
      if len(df) >= window:
        df[f'price_ma{window}'] = df['price'].rolling(window=window, min_periods=1).mean()
        df[f'score_ma{window}'] = df['score'].rolling(window=window, min_periods=1).mean()
    
    # Price volatility (rolling standard deviation)
    if len(df) >= 5:
      df['price_volatility'] = df['price'].rolling(window=5, min_periods=1).std()
      df['score_volatility'] = df['score'].rolling(window=5, min_periods=1).std()
    
    # Price and sentiment momentum (rate of change)
    df['price_momentum'] = df['price'].pct_change().fillna(0)
    df['score_momentum'] = df['score'].diff().fillna(0)
    
    # Bollinger Bands for price
    if len(df) >= 7:
      rolling_mean = df['price'].rolling(window=7).mean()
      rolling_std = df['price'].rolling(window=7).std()
      df['price_upper_bb'] = rolling_mean + (rolling_std * 2)
      df['price_lower_bb'] = rolling_mean - (rolling_std * 2)
    
    # Sentiment zones
    df['sentiment_zone'] = df['score'].apply(
      lambda x: 'Bullish' if x >= 7.0 else ('Neutral' if x >= 5.0 else 'Bearish')
    )
    
    # Price trend classification
    if len(df) >= 3:
      df['price_trend'] = 'Flat'
      for i in range(2, len(df)):
        if df.loc[i, 'price'] > df.loc[i-1, 'price'] > df.loc[i-2, 'price']:
          df.loc[i, 'price_trend'] = 'Uptrend'
        elif df.loc[i, 'price'] < df.loc[i-1, 'price'] < df.loc[i-2, 'price']:
          df.loc[i, 'price_trend'] = 'Downtrend'
    
    return df
  
  def create_comprehensive_dashboard(self, df: pd.DataFrame) -> str:
    """Create a comprehensive 6-panel financial dashboard."""
    output_path = os.path.join(self.output_dir, 'nvidia_comprehensive_dashboard.png')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Panel 1: Main Price & Sentiment Chart (spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    self.plot_main_price_sentiment(df, ax1)
    
    # Panel 2: Correlation Matrix
    ax2 = fig.add_subplot(gs[0, 2])
    self.plot_correlation_heatmap(df, ax2)
    
    # Panel 3: Price Distribution & Statistics
    ax3 = fig.add_subplot(gs[1, 0])
    self.plot_price_distribution(df, ax3)
    
    # Panel 4: Sentiment Distribution & Zones
    ax4 = fig.add_subplot(gs[1, 1])
    self.plot_sentiment_distribution(df, ax4)
    
    # Panel 5: Technical Analysis (Moving Averages)
    ax5 = fig.add_subplot(gs[1, 2])
    self.plot_technical_analysis(df, ax5)
    
    # Panel 6: Momentum & Volatility Analysis (spans 3 columns)
    ax6 = fig.add_subplot(gs[2, :])
    self.plot_momentum_volatility(df, ax6)
    
    # Add main title
    fig.suptitle('NVIDIA Stock & Sentiment Comprehensive Analysis Dashboard\n' + 
                 f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | ' +
                 f'Data Points: {len(df)}', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Comprehensive dashboard saved: {output_path}")
    
    return output_path
  
  def plot_main_price_sentiment(self, df: pd.DataFrame, ax):
    """Plot main price and sentiment chart with advanced styling."""
    # Determine x-axis
    if 'timestamp' in df.columns:
      x_data = df['timestamp']
      x_label = 'Time'
    else:
      x_data = df.index
      x_label = 'Entry Number'
    
    # Plot price with trend coloring
    price_line = ax.plot(x_data, df['price'], color=self.colors['price'], 
                        linewidth=3, label='NVIDIA Price', alpha=0.8)
    
    # Add moving average if available
    if 'price_ma5' in df.columns:
      ax.plot(x_data, df['price_ma5'], color=self.colors['ma_short'], 
              linewidth=2, linestyle='--', alpha=0.7, label='Price MA(5)')
    
    # Create twin axis for sentiment
    ax2 = ax.twinx()
    
    # Plot sentiment with color coding
    for i in range(len(df) - 1):
      score = df.iloc[i]['score']
      color = self.colors['bullish'] if score >= 7 else (
        self.colors['neutral'] if score >= 5 else self.colors['bearish'])
      
      if 'timestamp' in df.columns:
        x_segment = [df.iloc[i]['timestamp'], df.iloc[i+1]['timestamp']]
      else:
        x_segment = [i, i+1]
      y_segment = [df.iloc[i]['score'], df.iloc[i+1]['score']]
      
      ax2.plot(x_segment, y_segment, color=color, linewidth=3, alpha=0.9)
    
    # Add sentiment markers
    for i, (_, row) in enumerate(df.iterrows()):
      color = self.colors['bullish'] if row['score'] >= 7 else (
        self.colors['neutral'] if row['score'] >= 5 else self.colors['bearish'])
      x_pos = row['timestamp'] if 'timestamp' in df.columns else i
      ax2.scatter(x_pos, row['score'], color=color, s=60, alpha=0.8, zorder=5)
    
    # Styling
    ax.set_xlabel(x_label, fontweight='bold')
    ax.set_ylabel('Stock Price ($)', fontweight='bold', color=self.colors['price'])
    ax2.set_ylabel('Sentiment Score', fontweight='bold', color=self.colors['sentiment'])
    ax.tick_params(axis='y', labelcolor=self.colors['price'])
    ax2.tick_params(axis='y', labelcolor=self.colors['sentiment'])
    ax2.set_ylim(0, 10)
    
    # Add reference lines
    ax2.axhline(y=7, color=self.colors['bullish'], linestyle=':', alpha=0.5)
    ax2.axhline(y=5, color=self.colors['neutral'], linestyle=':', alpha=0.5)
    ax2.axhline(y=3, color=self.colors['bearish'], linestyle=':', alpha=0.5)
    
    ax.set_title('Price & Sentiment Trend Analysis', fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    # Legend
    handles1, labels1 = ax.get_legend_handles_labels()
    sentiment_handles = [
      plt.Line2D([0], [0], color=self.colors['bullish'], linewidth=3, label='Bullish (≥7.0)'),
      plt.Line2D([0], [0], color=self.colors['neutral'], linewidth=3, label='Neutral (5.0-6.9)'),
      plt.Line2D([0], [0], color=self.colors['bearish'], linewidth=3, label='Bearish (<5.0)')
    ]
    ax.legend(handles1 + sentiment_handles, labels1 + [h.get_label() for h in sentiment_handles], 
              loc='upper left', frameon=True, fancybox=True)
  
  def plot_correlation_heatmap(self, df: pd.DataFrame, ax):
    """Plot correlation heatmap of key metrics."""
    # Select numeric columns for correlation
    numeric_cols = ['price', 'score']
    if 'price_ma5' in df.columns:
      numeric_cols.extend(['price_ma5', 'score_ma5'])
    if 'price_volatility' in df.columns:
      numeric_cols.extend(['price_volatility', 'score_volatility'])
    
    corr_data = df[numeric_cols].corr()
    
    # Create heatmap
    sns.heatmap(corr_data, annot=True, cmap='RdBu_r', center=0, 
                square=True, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Correlation Matrix', fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
  
  def plot_price_distribution(self, df: pd.DataFrame, ax):
    """Plot price distribution with statistics."""
    sns.histplot(data=df, x='price', kde=True, ax=ax, alpha=0.7, color=self.colors['price'])
    
    # Add statistical lines
    mean_price = df['price'].mean()
    median_price = df['price'].median()
    
    ax.axvline(mean_price, color='red', linestyle='--', alpha=0.7, label=f'Mean: ${mean_price:.2f}')
    ax.axvline(median_price, color='orange', linestyle='--', alpha=0.7, label=f'Median: ${median_price:.2f}')
    
    ax.set_title('Price Distribution', fontweight='bold')
    ax.set_xlabel('Stock Price ($)')
    ax.set_ylabel('Frequency')
    ax.legend()
  
  def plot_sentiment_distribution(self, df: pd.DataFrame, ax):
    """Plot sentiment distribution by zones."""
    # Create sentiment zone counts
    zone_counts = df['sentiment_zone'].value_counts()
    colors = [self.colors['bullish'], self.colors['neutral'], self.colors['bearish']][:len(zone_counts)]
    
    bars = ax.bar(zone_counts.index, zone_counts.values, color=colors, alpha=0.7)
    
    # Add percentage labels
    total = len(df)
    for bar, count in zip(bars, zone_counts.values):
      height = bar.get_height()
      ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
              f'{count}\n({count/total*100:.1f}%)', 
              ha='center', va='bottom', fontweight='bold')
    
    ax.set_title('Sentiment Zone Distribution', fontweight='bold')
    ax.set_xlabel('Sentiment Zone')
    ax.set_ylabel('Count')
    ax.set_ylim(0, max(zone_counts.values) * 1.2)
  
  def plot_technical_analysis(self, df: pd.DataFrame, ax):
    """Plot technical analysis with moving averages."""
    x_data = df.index
    
    # Plot price and moving averages
    ax.plot(x_data, df['price'], color=self.colors['price'], linewidth=2, label='Price', alpha=0.8)
    
    if 'price_ma3' in df.columns:
      ax.plot(x_data, df['price_ma3'], color=self.colors['ma_short'], 
              linewidth=1.5, linestyle='--', alpha=0.7, label='MA(3)')
    
    if 'price_ma7' in df.columns:
      ax.plot(x_data, df['price_ma7'], color=self.colors['ma_long'], 
              linewidth=1.5, linestyle=':', alpha=0.7, label='MA(7)')
    
    # Add Bollinger Bands if available
    if 'price_upper_bb' in df.columns:
      ax.fill_between(x_data, df['price_lower_bb'], df['price_upper_bb'], 
                      alpha=0.1, color=self.colors['price'], label='Bollinger Bands')
    
    ax.set_title('Technical Analysis', fontweight='bold')
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Price ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
  
  def plot_momentum_volatility(self, df: pd.DataFrame, ax):
    """Plot momentum and volatility analysis."""
    x_data = df.index
    
    # Create twin axes for momentum and volatility
    ax2 = ax.twinx()
    
    # Plot price momentum
    momentum_colors = ['green' if x > 0 else 'red' for x in df['price_momentum']]
    bars = ax.bar(x_data, df['price_momentum'], color=momentum_colors, alpha=0.6, label='Price Momentum')
    
    # Plot volatility if available
    if 'price_volatility' in df.columns:
      ax2.plot(x_data, df['price_volatility'], color='purple', linewidth=2, 
               label='Price Volatility', alpha=0.8)
      ax2.set_ylabel('Volatility', color='purple')
      ax2.tick_params(axis='y', labelcolor='purple')
    
    ax.set_title('Momentum & Volatility Analysis', fontweight='bold')
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Price Momentum (%)', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax.grid(True, alpha=0.3)
  
  def create_individual_charts(self, df: pd.DataFrame) -> List[str]:
    """Create individual high-quality charts for detailed analysis."""
    output_paths = []
    
    # 1. Enhanced Price-Sentiment Chart
    fig, ax = plt.subplots(figsize=(14, 8))
    self.plot_main_price_sentiment(df, ax)
    path1 = os.path.join(self.output_dir, 'nvidia_price_sentiment_enhanced.png')
    plt.savefig(path1, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    output_paths.append(path1)
    
    # 2. Statistical Analysis Chart
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Scatter plot with regression
    sns.scatterplot(data=df, x='price', y='score', ax=ax1, s=100, alpha=0.7)
    sns.regplot(data=df, x='price', y='score', ax=ax1, scatter=False, color='red')
    correlation = df['price'].corr(df['score'])
    ax1.set_title(f'Price-Sentiment Correlation (r={correlation:.3f})', fontweight='bold')
    
    # Price over time
    if 'timestamp' in df.columns:
      ax2.plot(df['timestamp'], df['price'], color=self.colors['price'], linewidth=2, marker='o')
      ax2.set_title('Price Trend Over Time', fontweight='bold')
      ax2.tick_params(axis='x', rotation=45)
    else:
      ax2.plot(df.index, df['price'], color=self.colors['price'], linewidth=2, marker='o')
      ax2.set_title('Price Trend', fontweight='bold')
    
    # Sentiment over time with zones
    if 'timestamp' in df.columns:
      x_data = df['timestamp']
    else:
      x_data = df.index
    
    for i, (_, row) in enumerate(df.iterrows()):
      color = self.colors['bullish'] if row['score'] >= 7 else (
        self.colors['neutral'] if row['score'] >= 5 else self.colors['bearish'])
      ax3.scatter(x_data.iloc[i], row['score'], color=color, s=80, alpha=0.8)
    
    ax3.plot(x_data, df['score'], color='gray', linewidth=1, alpha=0.5)
    ax3.axhline(y=7, color=self.colors['bullish'], linestyle='--', alpha=0.5)
    ax3.axhline(y=5, color=self.colors['neutral'], linestyle='--', alpha=0.5)
    ax3.set_title('Sentiment Trend with Zones', fontweight='bold')
    ax3.set_ylim(0, 10)
    
    # Box plot by sentiment zone
    if len(df['sentiment_zone'].unique()) > 1:
      sns.boxplot(data=df, x='sentiment_zone', y='price', ax=ax4, 
                  palette=[self.colors['bearish'], self.colors['neutral'], self.colors['bullish']])
      ax4.set_title('Price Distribution by Sentiment Zone', fontweight='bold')
    else:
      ax4.text(0.5, 0.5, 'Insufficient data\nfor zone analysis', 
               ha='center', va='center', transform=ax4.transAxes, fontsize=12)
      ax4.set_title('Price by Sentiment Zone', fontweight='bold')
    
    plt.tight_layout()
    path2 = os.path.join(self.output_dir, 'nvidia_statistical_analysis.png')
    plt.savefig(path2, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    output_paths.append(path2)
    
    return output_paths
  
  def generate_detailed_report(self, df: pd.DataFrame) -> str:
    """Generate comprehensive statistical report."""
    report_path = os.path.join(self.output_dir, 'nvidia_detailed_report.txt')
    
    report = []
    report.append("📊 NVIDIA COMPREHENSIVE ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Data Source: {self.csv_path}")
    report.append(f"Total Records: {len(df)}")
    
    if 'timestamp' in df.columns and len(df) > 1:
      time_span = df['timestamp'].max() - df['timestamp'].min()
      report.append(f"Time Span: {time_span}")
      report.append(f"Data Frequency: {len(df)} points over {time_span}")
    
    report.append("\n📈 PRICE ANALYSIS")
    report.append("-" * 50)
    report.append(f"Latest Price: ${df['price'].iloc[-1]:.2f}")
    report.append(f"Average Price: ${df['price'].mean():.2f}")
    report.append(f"Median Price: ${df['price'].median():.2f}")
    report.append(f"Price Range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    report.append(f"Standard Deviation: ${df['price'].std():.2f}")
    report.append(f"Coefficient of Variation: {(df['price'].std()/df['price'].mean())*100:.2f}%")
    
    if len(df) > 1:
      price_change = df['price'].iloc[-1] - df['price'].iloc[0]
      price_change_pct = (price_change / df['price'].iloc[0]) * 100
      report.append(f"Total Price Change: ${price_change:.2f} ({price_change_pct:+.2f}%)")
    
    report.append("\n🎯 SENTIMENT ANALYSIS")
    report.append("-" * 50)
    report.append(f"Latest Sentiment: {df['score'].iloc[-1]:.1f}/10")
    report.append(f"Average Sentiment: {df['score'].mean():.1f}/10")
    report.append(f"Median Sentiment: {df['score'].median():.1f}/10")
    report.append(f"Sentiment Range: {df['score'].min():.1f} - {df['score'].max():.1f}")
    report.append(f"Sentiment Std Dev: {df['score'].std():.2f}")
    
    # Sentiment zone analysis
    zone_counts = df['sentiment_zone'].value_counts()
    report.append(f"\nSentiment Zone Distribution:")
    for zone, count in zone_counts.items():
      percentage = (count / len(df)) * 100
      report.append(f"  {zone}: {count} records ({percentage:.1f}%)")
    
    # Correlation analysis
    if len(df) > 2:
      correlation = df['price'].corr(df['score'])
      report.append(f"\n🔗 CORRELATION ANALYSIS")
      report.append("-" * 50)
      report.append(f"Price-Sentiment Correlation: {correlation:.4f}")
      
      if abs(correlation) > 0.7:
        strength = "Strong"
      elif abs(correlation) > 0.3:
        strength = "Moderate"
      else:
        strength = "Weak"
      
      direction = "Positive" if correlation > 0 else "Negative"
      report.append(f"Relationship Strength: {strength} {direction}")
      
      # Statistical significance (basic)
      from scipy.stats import pearsonr
      try:
        _, p_value = pearsonr(df['price'], df['score'])
        significance = "Significant" if p_value < 0.05 else "Not Significant"
        report.append(f"Statistical Significance: {significance} (p={p_value:.4f})")
      except:
        report.append("Statistical Significance: Cannot calculate")
    
    # Technical analysis
    if 'price_momentum' in df.columns:
      report.append(f"\n📊 TECHNICAL INDICATORS")
      report.append("-" * 50)
      avg_momentum = df['price_momentum'].mean()
      report.append(f"Average Price Momentum: {avg_momentum:.4f}")
      
      if 'price_volatility' in df.columns:
        avg_volatility = df['price_volatility'].mean()
        report.append(f"Average Price Volatility: ${avg_volatility:.2f}")
      
      # Trend analysis
      if 'price_trend' in df.columns:
        trend_counts = df['price_trend'].value_counts()
        report.append(f"\nPrice Trend Analysis:")
        for trend, count in trend_counts.items():
          percentage = (count / len(df)) * 100
          report.append(f"  {trend}: {count} periods ({percentage:.1f}%)")
    
    # Risk assessment
    report.append(f"\n⚠️ RISK ASSESSMENT")
    report.append("-" * 50)
    
    # Price volatility risk
    volatility_risk = "High" if df['price'].std() > df['price'].mean() * 0.1 else (
      "Medium" if df['price'].std() > df['price'].mean() * 0.05 else "Low")
    report.append(f"Price Volatility Risk: {volatility_risk}")
    
    # Sentiment consistency
    sentiment_consistency = "High" if df['score'].std() < 1.0 else (
      "Medium" if df['score'].std() < 2.0 else "Low")
    report.append(f"Sentiment Consistency: {sentiment_consistency}")
    
    # Overall assessment
    latest_sentiment = df['score'].iloc[-1]
    latest_price = df['price'].iloc[-1]
    avg_price = df['price'].mean()
    
    if latest_sentiment >= 7.0 and latest_price >= avg_price:
      outlook = "Positive - High sentiment with above-average price"
    elif latest_sentiment <= 3.0 and latest_price <= avg_price:
      outlook = "Negative - Low sentiment with below-average price"
    elif latest_sentiment >= 7.0 and latest_price <= avg_price:
      outlook = "Mixed - High sentiment but below-average price (potential opportunity)"
    elif latest_sentiment <= 3.0 and latest_price >= avg_price:
      outlook = "Warning - Low sentiment with above-average price (potential risk)"
    else:
      outlook = "Neutral - Moderate sentiment and price levels"
    
    report.append(f"\n📋 OVERALL ASSESSMENT")
    report.append("-" * 50)
    report.append(f"Market Outlook: {outlook}")
    
    # Save report
    with open(report_path, 'w', encoding='utf-8') as f:
      f.write('\n'.join(report))
    
    return report_path
  
  def generate_all_visualizations(self) -> bool:
    """Generate complete set of visualizations and reports."""
    print("🎯 Starting NVIDIA Advanced Visualization Generation...")
    
    # Load data
    df = self.load_and_enhance_data()
    if df is None or len(df) == 0:
      print("❌ No data available for visualization")
      return False
    
    try:
      # Generate comprehensive dashboard
      dashboard_path = self.create_comprehensive_dashboard(df)
      
      # Generate individual charts
      individual_paths = self.create_individual_charts(df)
      
      # Generate detailed report
      report_path = self.generate_detailed_report(df)
      
      # Summary
      print("\n✅ All visualizations generated successfully!")
      print(f"📊 Dashboard: {os.path.basename(dashboard_path)}")
      for path in individual_paths:
        print(f"📈 Chart: {os.path.basename(path)}")
      print(f"📋 Report: {os.path.basename(report_path)}")
      print(f"📁 Output directory: {self.output_dir}")
      
      return True
      
    except Exception as e:
      print(f"❌ Error generating visualizations: {e}")
      return False

def main():
  """Main execution function."""
  # Check if seaborn and scipy are available
  try:
    import seaborn as sns
    from scipy.stats import pearsonr
    print("✅ All required libraries available")
  except ImportError as e:
    print(f"⚠️ Missing library: {e}")
    print("Consider installing: pip install seaborn scipy")
  
  visualizer = NvidiaAdvancedVisualizer()
  success = visualizer.generate_all_visualizations()
  
  return success

if __name__ == "__main__":
  import sys
  success = main()
  sys.exit(0 if success else 1)