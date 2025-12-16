#!/usr/bin/env python3
"""
Enhanced Stock Price and Sentiment Score Visualization
Applies Context 7 best practices for financial data visualization using matplotlib and seaborn.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Path to your data file
DATA_PATH = '/root/2/stocks/web/nvidia_score_price_dump.txt'

class NvidiaStockVisualizer:
  def __init__(self, data_path: str = DATA_PATH):
    self.data_path = data_path
    self.setup_style()
  
  def setup_style(self):
    """Apply Context 7 best practices for professional financial visualization."""
    # Set seaborn theme for professional appearance
    sns.set_theme(style="whitegrid", palette="deep")
    
    # Configure matplotlib for high-quality financial charts
    plt.rcParams.update({
      'figure.figsize': (14, 10),
      'font.size': 11,
      'axes.titlesize': 14,
      'axes.labelsize': 12,
      'xtick.labelsize': 10,
      'ytick.labelsize': 10,
      'legend.fontsize': 11,
      'figure.titlesize': 16,
      'lines.linewidth': 2,
      'grid.alpha': 0.3
    })
  
  def load_and_prepare_data(self) -> Optional[pd.DataFrame]:
    """Load CSV data with proper datetime parsing and validation."""
    try:
      if not os.path.exists(self.data_path):
        print(f"❌ Data file not found: {self.data_path}")
        return None
      
      # Read CSV with proper handling
      df = pd.read_csv(self.data_path, skipinitialspace=True)
      
      # Normalize column names
      df.columns = [col.strip().lower() for col in df.columns]
      
      print(f"✅ Loaded data with columns: {df.columns.tolist()}")
      print(f"📊 Total records: {len(df)}")
      
      # Parse timestamp if it exists
      if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        print("✅ Timestamps parsed and sorted")
      
      # Validate required columns
      required_columns = ['score', 'price']
      missing_columns = [col for col in required_columns if col not in df.columns]
      if missing_columns:
        print(f"❌ Missing required columns: {missing_columns}")
        return None
      
      # Add technical indicators
      df = self.add_technical_indicators(df)
      
      return df
      
    except Exception as e:
      print(f"❌ Error loading data: {e}")
      return None
  
  def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    """Add moving averages and technical indicators."""
    if len(df) >= 3:
      # Add moving averages for price and score
      df['price_ma3'] = df['price'].rolling(window=3, min_periods=1).mean()
      df['score_ma3'] = df['score'].rolling(window=3, min_periods=1).mean()
      
      if len(df) >= 5:
        df['price_ma5'] = df['price'].rolling(window=5, min_periods=1).mean()
        df['score_ma5'] = df['score'].rolling(window=5, min_periods=1).mean()
    
    return df
  
  def get_sentiment_color(self, score: float) -> str:
    """Return color based on sentiment score using financial color coding."""
    if score >= 7.0:
      return '#2E8B57'  # Sea Green (bullish)
    elif score >= 5.0:
      return '#4169E1'  # Royal Blue (neutral)
    else:
      return '#DC143C'  # Crimson (bearish)
  
  def plot_dual_axis_chart(self, df: pd.DataFrame, output_path: str = 'nvidia_enhanced_chart.png'):
    """Create enhanced dual-axis chart with Context 7 best practices."""
    fig, ax1 = plt.subplots(figsize=(14, 10))
    
    # Determine x-axis (use timestamp if available, otherwise index)
    if 'timestamp' in df.columns:
      x_data = df['timestamp']
      x_label = 'Time'
    else:
      x_data = df.index
      x_label = 'Entry Number'
    
    # Plot price on primary axis with enhanced styling
    line1 = ax1.plot(x_data, df['price'], color='#1f77b4', linewidth=2.5, 
                     label='NVIDIA Price', marker='o', markersize=4, alpha=0.8)
    
    # Add price moving average if available
    if 'price_ma3' in df.columns:
      ax1.plot(x_data, df['price_ma3'], color='#1f77b4', linewidth=1.5, 
               linestyle='--', alpha=0.6, label='Price MA(3)')
    
    ax1.set_xlabel(x_label, fontweight='bold')
    ax1.set_ylabel('Stock Price ($)', fontweight='bold', color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.grid(True, alpha=0.3)
    
    # Create secondary y-axis for sentiment score
    ax2 = ax1.twinx()
    
    # Color-code sentiment line based on values
    for i in range(len(df) - 1):
      score = df.iloc[i]['score']
      color = self.get_sentiment_color(score)
      
      if 'timestamp' in df.columns:
        x_segment = [df.iloc[i]['timestamp'], df.iloc[i+1]['timestamp']]
      else:
        x_segment = [i, i+1]
      y_segment = [df.iloc[i]['score'], df.iloc[i+1]['score']]
      
      ax2.plot(x_segment, y_segment, color=color, linewidth=2.5, alpha=0.8)
    
    # Add sentiment moving average if available
    if 'score_ma3' in df.columns:
      ax2.plot(x_data, df['score_ma3'], color='#ff7f0e', linewidth=1.5, 
               linestyle=':', alpha=0.7, label='Sentiment MA(3)')
    
    # Add markers for sentiment data points
    for i, (_, row) in enumerate(df.iterrows()):
      color = self.get_sentiment_color(row['score'])
      if 'timestamp' in df.columns:
        x_pos = row['timestamp']
      else:
        x_pos = i
      ax2.scatter(x_pos, row['score'], color=color, s=50, alpha=0.8, zorder=5)
    
    ax2.set_ylabel('Sentiment Score (1-10)', fontweight='bold', color='#ff7f0e')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    ax2.set_ylim(0, 10)
    
    # Add horizontal reference lines for sentiment zones
    ax2.axhline(y=7, color='green', linestyle='--', alpha=0.3, label='Bullish Zone')
    ax2.axhline(y=5, color='blue', linestyle='--', alpha=0.3, label='Neutral Zone')
    ax2.axhline(y=3, color='red', linestyle='--', alpha=0.3, label='Bearish Zone')
    
    # Enhanced title with statistics
    correlation = df['price'].corr(df['score']) if len(df) > 1 else 0
    title = f'NVIDIA Stock Price vs Sentiment Analysis\n'
    title += f'Records: {len(df)} | Correlation: {correlation:.3f} | '
    title += f'Avg Price: ${df["price"].mean():.2f} | Avg Sentiment: {df["score"].mean():.1f}'
    
    plt.title(title, fontweight='bold', pad=20)
    
    # Create custom legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    
    # Add sentiment color legend
    sentiment_handles = [
      plt.Line2D([0], [0], color='#2E8B57', linewidth=3, label='Bullish (≥7.0)'),
      plt.Line2D([0], [0], color='#4169E1', linewidth=3, label='Neutral (5.0-6.9)'),
      plt.Line2D([0], [0], color='#DC143C', linewidth=3, label='Bearish (<5.0)')
    ]
    
    all_handles = handles1 + handles2 + sentiment_handles
    all_labels = labels1 + labels2 + [h.get_label() for h in sentiment_handles]
    
    plt.legend(all_handles, all_labels, loc='upper left', bbox_to_anchor=(0, 1), 
               frameon=True, fancybox=True, shadow=True)
    
    # Format x-axis for timestamps
    if 'timestamp' in df.columns:
      plt.xticks(rotation=45)
      fig.autofmt_xdate()
    
    plt.tight_layout()
    
    # Save with high resolution
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'✅ Enhanced chart saved to {output_path}')
    
    return fig, (ax1, ax2)
  
  def plot_correlation_analysis(self, df: pd.DataFrame, output_path: str = 'nvidia_correlation.png'):
    """Create correlation analysis visualization."""
    if len(df) < 2:
      print("⚠️ Not enough data for correlation analysis")
      return None
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Scatter plot with trend line
    sns.scatterplot(data=df, x='price', y='score', ax=ax1, s=100, alpha=0.7)
    sns.regplot(data=df, x='price', y='score', ax=ax1, scatter=False, color='red')
    correlation = df['price'].corr(df['score'])
    ax1.set_title(f'Price vs Sentiment Correlation\nr = {correlation:.3f}', fontweight='bold')
    ax1.set_xlabel('Stock Price ($)')
    ax1.set_ylabel('Sentiment Score')
    
    # 2. Price distribution
    sns.histplot(data=df, x='price', kde=True, ax=ax2, alpha=0.7)
    ax2.set_title('Price Distribution', fontweight='bold')
    ax2.axvline(df['price'].mean(), color='red', linestyle='--', label=f'Mean: ${df["price"].mean():.2f}')
    ax2.legend()
    
    # 3. Sentiment distribution
    sns.histplot(data=df, x='score', kde=True, ax=ax3, alpha=0.7)
    ax3.set_title('Sentiment Distribution', fontweight='bold')
    ax3.axvline(df['score'].mean(), color='red', linestyle='--', label=f'Mean: {df["score"].mean():.1f}')
    ax3.set_xlim(0, 10)
    ax3.legend()
    
    # 4. Time series if timestamp available
    if 'timestamp' in df.columns:
      ax4_twin = ax4.twinx()
      ax4.plot(df['timestamp'], df['price'], color='blue', label='Price', marker='o')
      ax4_twin.plot(df['timestamp'], df['score'], color='orange', label='Sentiment', marker='s')
      ax4.set_title('Price & Sentiment Over Time', fontweight='bold')
      ax4.set_xlabel('Time')
      ax4.set_ylabel('Price ($)', color='blue')
      ax4_twin.set_ylabel('Sentiment Score', color='orange')
      ax4.tick_params(axis='x', rotation=45)
      
      # Combine legends
      lines1, labels1 = ax4.get_legend_handles_labels()
      lines2, labels2 = ax4_twin.get_legend_handles_labels()
      ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
      # Box plot for sentiment ranges
      df['sentiment_category'] = df['score'].apply(lambda x: 'Bullish' if x >= 7 else ('Neutral' if x >= 5 else 'Bearish'))
      sns.boxplot(data=df, x='sentiment_category', y='price', ax=ax4)
      ax4.set_title('Price by Sentiment Category', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'✅ Correlation analysis saved to {output_path}')
    
    return fig
  
  def generate_comprehensive_report(self, df: pd.DataFrame) -> str:
    """Generate statistical summary report."""
    report = []
    report.append("📊 NVIDIA STOCK & SENTIMENT ANALYSIS REPORT")
    report.append("=" * 60)
    report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total Data Points: {len(df)}")
    
    if 'timestamp' in df.columns and len(df) > 1:
      time_range = df['timestamp'].max() - df['timestamp'].min()
      report.append(f"Time Range: {time_range}")
    
    report.append("\n📈 PRICE STATISTICS")
    report.append("-" * 30)
    report.append(f"Current Price: ${df['price'].iloc[-1]:.2f}")
    report.append(f"Average Price: ${df['price'].mean():.2f}")
    report.append(f"Price Range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    report.append(f"Price Volatility (std): ${df['price'].std():.2f}")
    
    report.append("\n🎯 SENTIMENT STATISTICS")
    report.append("-" * 30)
    report.append(f"Current Sentiment: {df['score'].iloc[-1]:.1f}/10")
    report.append(f"Average Sentiment: {df['score'].mean():.1f}/10")
    report.append(f"Sentiment Range: {df['score'].min():.1f} - {df['score'].max():.1f}")
    
    # Sentiment distribution
    bullish_count = len(df[df['score'] >= 7])
    neutral_count = len(df[(df['score'] >= 5) & (df['score'] < 7)])
    bearish_count = len(df[df['score'] < 5])
    
    report.append(f"Bullish Periods: {bullish_count} ({bullish_count/len(df)*100:.1f}%)")
    report.append(f"Neutral Periods: {neutral_count} ({neutral_count/len(df)*100:.1f}%)")
    report.append(f"Bearish Periods: {bearish_count} ({bearish_count/len(df)*100:.1f}%)")
    
    if len(df) > 1:
      correlation = df['price'].corr(df['score'])
      report.append(f"\n🔗 CORRELATION ANALYSIS")
      report.append("-" * 30)
      report.append(f"Price-Sentiment Correlation: {correlation:.3f}")
      
      if abs(correlation) > 0.7:
        strength = "Strong"
      elif abs(correlation) > 0.3:
        strength = "Moderate"
      else:
        strength = "Weak"
      
      direction = "Positive" if correlation > 0 else "Negative"
      report.append(f"Relationship: {strength} {direction}")
    
    return "\n".join(report)

def plot_score_and_price(data_path=DATA_PATH, output_path='nvidia_enhanced_chart.png'):
  """Main plotting function with Context 7 enhanced visualization."""
  visualizer = NvidiaStockVisualizer(data_path)
  
  # Load and prepare data
  df = visualizer.load_and_prepare_data()
  if df is None or len(df) == 0:
    print("❌ No data available for plotting")
    return False
  
  # Generate main chart
  try:
    fig, axes = visualizer.plot_dual_axis_chart(df, output_path)
    
    # Generate correlation analysis if enough data
    if len(df) >= 3:
      base_name = output_path.rsplit('.', 1)[0]
      correlation_path = f"{base_name}_correlation.png"
      visualizer.plot_correlation_analysis(df, correlation_path)
    
    # Generate and print statistical report
    report = visualizer.generate_comprehensive_report(df)
    print("\n" + report)
    
    # Save report to file
    report_path = output_path.rsplit('.', 1)[0] + '_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
      f.write(report)
    print(f"\n✅ Statistical report saved to {report_path}")
    
    return True
    
  except Exception as e:
    print(f"❌ Error generating visualizations: {e}")
    return False

if __name__ == '__main__':
  success = plot_score_and_price(DATA_PATH)
  if not success:
    print("❌ Visualization generation failed")
    exit(1)
  else:
    print("✅ All visualizations generated successfully!")
    # Optionally show the plot
    plt.show()