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
        df['rsi_signal'] = np.where(df['RSI'] > self.overbought, self.TradingSignals.SELL
                                    , np.where(df['RSI'] < self.oversold, self.TradingSignals.BUY, self.TradingSignals.HOLD)) 
        
        df['rsi_positions'] = df['rsi_signal'].diff()
        return df['rsi_signal']