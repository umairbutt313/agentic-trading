# omnialpha/data_fusion/data_loader.py

import pandas as pd
import numpy as np
from typing import Dict, List

import yfinance as yf

# Optional import for pandas-ta
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    print("⚠️  pandas-ta not installed. Technical indicators will be limited.")

class DataFusionEngine:
    """
    A refined DataFusionEngine that fetches fundamental data via yfinance,
    calculates basic technical indicators, and includes placeholders
    for alternative data. The final DataFrame can be further processed
    or fed into your pipeline.
    """
    def __init__(self):
        # Dictionary of "data source" methods. Each returns a DataFrame
        # keyed by ticker or with ticker as an index.
        self.data_sources = {
            'fundamental': self._load_fundamental_data,
            'technical': self._load_technical_data,
            'alternative': self._load_alternative_data
        }

    def fuse_data(self, tickers: List[str]) -> pd.DataFrame:
        """
        Multi-modal data fusion pipeline: 
        1. Loads fundamental, technical, alternative data.
        2. Concatenates everything into a single DataFrame.
        3. Applies simple feature engineering (e.g., gamma_exposure).
        4. Returns the final fused DataFrame (one row per ticker).
        """
        # We create a container DataFrame whose index = tickers
        # so that each data source can be joined by index.
        fused_data = pd.DataFrame(index=tickers)

        for source_name, loader_func in self.data_sources.items():
            # Load data from this source
            source_df = loader_func(tickers)

            # Ensure the source_df aligns with the main index
            # We'll join on index (which should be the ticker)
            fused_data = fused_data.join(source_df, how='left')

        # Optional: apply any feature engineering logic after loading
        fused_data = self._apply_feature_engineering(fused_data)

        return fused_data

    def _load_fundamental_data(self, tickers: List[str]) -> pd.DataFrame:
        """
        Example: Use yfinance to grab fundamental info. Real usage might:
        - Connect to your data vendor or internal DB 
        - Parse statements for Piotroski or Altman Z
        - Etc.
        Returns a DataFrame with fundamental columns, indexed by ticker.
        """
        rows = []
        for tkr in tickers:
            info = {}
            try:
                # Quick fetch via yfinance
                data = yf.Ticker(tkr).info

                info['ticker'] = tkr
                # Market cap, trailing PE, and forward PE often used
                info['marketCap'] = data.get('marketCap', np.nan)
                info['trailingPE'] = data.get('trailingPE', np.nan)
                info['forwardPE'] = data.get('forwardPE', np.nan)

                # For demonstration, let's pretend Piotroski or Altman are placeholders
                info['piotroski_f_score'] = np.random.randint(1, 9)
                info['altman_z_score'] = np.random.uniform(1, 4)

            except Exception as e:
                # If data fails to load, fill with NaN
                info['ticker'] = tkr
                info['marketCap'] = np.nan
                info['trailingPE'] = np.nan
                info['forwardPE'] = np.nan
                info['piotroski_f_score'] = np.nan
                info['altman_z_score'] = np.nan

            rows.append(info)

        df = pd.DataFrame(rows).set_index('ticker')
        return df

    def _load_technical_data(self, tickers: List[str]) -> pd.DataFrame:
        """
        Fetch daily price data for each ticker, compute some
        technical indicators (via pandas-ta or custom logic),
        and return a DataFrame of key metrics.
        """
        tech_frames = []
        for tkr in tickers:
            try:
                # Download ~6 months of daily data
                df = yf.download(tkr, period='6mo', interval='1d', progress=False)
                if df.empty:
                    # If no data returned, create a single-row fallback
                    tech_frames.append(pd.DataFrame({
                        'ticker': [tkr],
                        'FAMA_50d': [np.nan],
                        'VW_MACD': [np.nan],
                        'volume': [np.nan],
                        'adv_20d': [np.nan],
                        'returns': [np.nan]
                    }).set_index('ticker'))
                    continue

                # Example: "Fractal Adaptive Moving Average" is not in standard pandas-ta,
                # but we can approximate. Let's just use a 50-day EMA for demonstration:
                df['FAMA_50d'] = df['Close'].ewm(span=50).mean()  # placeholder

                # Example: "VW-MACD" => we can do normal MACD, but weighting by volume is more advanced.
                # We'll do a standard MACD from pandas-ta (if available):
                if HAS_PANDAS_TA:
                    macd = ta.macd(df['Close'])
                    if macd is not None and not macd.empty:
                        df['MACD'] = macd['MACD_12_26_9']
                    else:
                        df['MACD'] = np.nan
                else:
                    # Simple MACD approximation without pandas-ta
                    ema_12 = df['Close'].ewm(span=12).mean()
                    ema_26 = df['Close'].ewm(span=26).mean()
                    df['MACD'] = ema_12 - ema_26

                # Typical volume + average daily volume
                df['adv_20d'] = df['Volume'].rolling(20).mean()

                # Quick daily returns (closing basis)
                df['returns'] = df['Close'].pct_change().fillna(0)

                # We'll pick the last row as "current" state
                last_row = df.iloc[-1]
                row_dict = {
                    'ticker': tkr,
                    'FAMA_50d': last_row['FAMA_50d'],
                    'VW_MACD': last_row['MACD'],
                    'volume': last_row['Volume'],
                    'adv_20d': last_row['adv_20d'],
                    'returns': last_row['returns']
                }
                tech_frames.append(pd.DataFrame([row_dict]).set_index('ticker'))

            except Exception as e:
                # If for some reason data fails, fallback
                tech_frames.append(pd.DataFrame({
                    'ticker': [tkr],
                    'FAMA_50d': [np.nan],
                    'VW_MACD': [np.nan],
                    'volume': [np.nan],
                    'adv_20d': [np.nan],
                    'returns': [np.nan]
                }).set_index('ticker'))

        if tech_frames:
            tech_df = pd.concat(tech_frames)
        else:
            # If no data at all, return empty
            columns = ['FAMA_50d','VW_MACD','volume','adv_20d','returns']
            tech_df = pd.DataFrame(index=tickers, columns=columns)

        return tech_df

    def _load_alternative_data(self, tickers: List[str]) -> pd.DataFrame:
        """
        Placeholder for alternative data such as:
        - CEO Sentiment, Satellite Imagery, etc.
        For demonstration, we just generate random or dummy data.
        """
        rows = []
        for tkr in tickers:
            row = {
                'ticker': tkr,
                'ceo_stress': np.random.uniform(0, 1),
                'dark_pool_volume': np.random.randint(1000, 50000),
                'meme_social_index': np.random.uniform(0, 1),
            }
            rows.append(row)

        df = pd.DataFrame(rows).set_index('ticker')
        return df

    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        # Example 1: liquidity_pressure = tanh(volume / adv_20d)
        if 'volume' in df.columns and 'adv_20d' in df.columns:
            df.columns = [
                "_".join(col).strip() if isinstance(col, tuple) else col
                for col in df.columns
            ]

            # Convert adv_20d and volume to numeric
            df['adv_20d'] = pd.to_numeric(df['adv_20d'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

            mask = df['adv_20d'] > 0
            df['liquidity_pressure'] = np.nan
            df.loc[mask, 'liquidity_pressure'] = np.tanh(
                df.loc[mask, 'volume'] / df.loc[mask, 'adv_20d']
            )

        # Example 2: gamma_exposure = (oi_call - oi_put), placeholders for demonstration
        df['oi_call'] = np.random.randint(1000, 10000, size=len(df))
        df['oi_put']  = np.random.randint(1000, 10000, size=len(df))
        df['gamma_exposure'] = df['oi_call'] - df['oi_put']

        return df




def main():
    # 1. Data Fusion
    fusion_engine = DataFusionEngine()
    universe = ['NVDA', 'TSLA', 'AMZN']  # Example tickers
    fused_data = fusion_engine.fuse_data(universe)
    fused_data.to_csv('fused_data.csv')
    fused_data.head()
    fused_data.tail()
    fused_data.describe()
    fused_data.info()
    fused_data.columns
    fused_data.index
    fused_data.shape
    fused_data.dtypes
    fused_data.isnull().sum()
    fused_data.dropna()
    fused_data.fillna(0)
    
    pass

if __name__ == "__main__":
    main()