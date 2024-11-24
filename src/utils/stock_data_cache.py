import os
import pandas as pd
import yfinance as yf
from datetime import datetime
import pickle

class StockDataCache:
    def __init__(self, stock_symbol, start_date, end_date):
        self.stock_symbol = stock_symbol
        self.start_date = start_date
        self.end_date = end_date
        self.cache_file = f"cache/{self.stock_symbol}_{self.start_date}_{self.end_date}.pkl"
        os.makedirs("cache", exist_ok=True) 
        self.data = None

    def get_data(self, show: bool = True):
        # Check if the data is already cached
        if self.data is not None:
            return self.data
            
        if os.path.exists(self.cache_file):
            if show:
                print(f"Loading data from cache for {self.stock_symbol}...")
            with open(self.cache_file, 'rb') as f:
                self.data = pickle.load(f)
                return self.data
        else:
            if show:
                print(f"Downloading data for {self.stock_symbol}...")
            ticker = yf.Ticker(self.stock_symbol)
            df = ticker.history(start=self.start_date, end=self.end_date)
            # df = df.rename_axis("Date").reset_index()            
            
            df_data = df.astype('float')

            # Save to cache
            with open(self.cache_file, 'wb') as f:
                pickle.dump(df_data, f)
            
            df_data = df_data.astype('float')

            return df_data
