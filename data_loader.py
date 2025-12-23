import yfinance as yf
import numpy as np
import pandas as pd
import os

class MarketData:
    def __init__(self, ticker="^NSEI", window_size=21, cache_dir="data"):
        self.ticker = ticker
        self.window_size = window_size
        self.cache_dir = cache_dir
        self.file_path = os.path.join(self.cache_dir, f"{ticker.replace('^', '')}_data.csv")
        
        # Create data directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def get_processed_data(self, start="2015-01-01", end="2023-12-31"):
        # Check if local cache exists
        if os.path.exists(self.file_path):
            print(f"Loading data from local cache: {self.file_path}")
            data = pd.read_csv(self.file_path, index_col=0, parse_dates=True)
        else:
            print(f"Downloading data for {self.ticker}...")
            data = yf.download(
                self.ticker, 
                start=start, 
                end=end, 
                auto_adjust=True, 
                multi_level_index=False
            )
            # Save to CSV for next time
            data.to_csv(self.file_path)

        # Pre-processing
        data['Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data = data.dropna()

        windows = []
        indices = []
        
        for i in range(len(data) - self.window_size):
            window = data['Returns'].iloc[i:i + self.window_size].values
            windows.append(np.sort(window)) 
            indices.append(data.index[i + self.window_size])
            
        return data, np.array(windows), indices