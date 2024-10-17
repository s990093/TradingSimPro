import numpy as np
from .base_strategy import BaseStrategy
import talib as ta

class RSIStrategy(BaseStrategy):
    def __init__(self, rsi_period=14, overbought=70, oversold=30):
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold

    def apply_strategy(self, df):
        df['RSI'] = ta.RSI(df['Close'], timeperiod=self.rsi_period)
        df['rsi_signal'] = np.where(df['RSI'] < self.oversold, 1, np.where(df['RSI'] > self.overbought, -1, 0))
        df['rsi_positions'] = df['rsi_signal'].diff()
        return df