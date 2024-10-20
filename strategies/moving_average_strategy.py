import numpy as np
from .base_strategy import BaseStrategy
import talib as ta

class MovingAverageStrategy(BaseStrategy):
    def __init__(self, short_window=50, long_window=200):
        self.short_window = short_window
        self.long_window = long_window

    def apply_strategy(self, df):
        df['short_ma'] = ta.SMA(df['Close'], timeperiod=self.short_window)
        df['long_ma'] = ta.SMA(df['Close'], timeperiod=self.long_window)
        df['ma_signal'] = 0
        df.iloc[self.short_window:, df.columns.get_loc('ma_signal')] = np.where(
            df['short_ma'].iloc[self.short_window:] > df['long_ma'].iloc[self.short_window:], 1, 0
        )
        df['ma_positions'] = df['ma_signal'].diff()
        return df['ma_signal']