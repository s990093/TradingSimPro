import pandas as pd
import numpy as np
import yfinance as yf

from mult_strategy.strategies.base.base_strategy import BaseStrategy
from utils.stock_data_cache import StockDataCache

class ReferenceStrategy:
    """Generic class to handle reference data and indicator calculations."""
    
    def __init__(self, symbol, start_date, end_date, ma_window=50, rsi_period=14):
        # Store the symbol and sanitize it
        self.symbol = symbol
        self.stock_name = symbol.replace('^', '').replace('.', '_')  # Remove or replace problematic characters
        
        # Download market data for the given symbol
        self.data = StockDataCache(symbol, start_date, end_date).get_data()
        
        
        self.ma_window = ma_window
        self.rsi_period = rsi_period

    def calculate_rsi(self, prices):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def apply_indicators(self):
        # Calculate moving average and RSI for the reference symbol
        self.data['ma'] = self.data['Close'].rolling(window=self.ma_window).mean()
        self.data['rsi'] = self.calculate_rsi(self.data['Close'])
        return self.data[['ma', 'rsi']]


class MultiReferenceStrategy(BaseStrategy):
    signal_name = "multi_reference_signal"
    def __init__(self, start_date, end_date, references):
        super().__init__()
        self.references = [ReferenceStrategy(**ref, start_date=start_date, end_date=end_date) for ref in references]

    def apply_strategy(self, df):
        scores = []
        
        for ref in self.references:
            ref_data = ref.apply_indicators()
            df = df.join(ref_data[['ma', 'rsi']], rsuffix=f"_{ref.stock_name}", how='left')

            df[f'ma_{ref.stock_name}'] = df[f'ma'].ffill() 
            df[f'rsi_{ref.stock_name}'] = df[f'rsi'].ffill() 

            # Score calculation based on the distance from thresholds
            ma_score = (df['Close'] - df[f'ma_{ref.stock_name}']) / df[f'ma_{ref.stock_name}']
            rsi_score = (df[f'rsi_{ref.stock_name}'] - 50) / 50  # Normalizing RSI around 50

            # Combine scores, giving equal weight to MA and RSI deviations
            combined_score = ma_score - rsi_score
            scores.append(combined_score)

        # Average the scores across the references
        df['average_score'] = np.mean(scores, axis=0)

        # Convert the average score into a signal
        df['multi_reference_signal'] = np.where(
            df['average_score'] > 0.05,  # Strong positive signal
            self.TradingSignals.BUY,
            np.where(
                df['average_score'] < -0.05,  # Strong negative signal
                self.TradingSignals.SELL,
                self.TradingSignals.HOLD  # Neutral signal
            )
        )

        # Store the signals and dataframe for visualization
        self.signal = df['multi_reference_signal']
        # self.df = df

        # # Call the visualize method to plot the signals
        # self.visualize()

        return df['multi_reference_signal']
