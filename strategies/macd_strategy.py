import numpy as np
from .base_strategy import BaseStrategy
import talib as ta


class MACDStrategy(BaseStrategy):
    def apply_strategy(self, df):
        df['macd'], df['macdsignal'], df['macdhist'] = ta.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd_signal'] = np.where(df['macd'] > df['macdsignal'], 1, 0)
        df['macd_positions'] = df['macd_signal'].diff()
        return df