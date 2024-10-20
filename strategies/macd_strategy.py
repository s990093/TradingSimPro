import numpy as np
from .base_strategy import BaseStrategy
import talib as ta


class MACDStrategy(BaseStrategy):
    def __init__(self, fast_period=26, slow_period=50, signal_period=18):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def apply_strategy(self, df):
        df['macd'], df['signal_line'], _ = ta.MACD(df['Close'], fastperiod=self.fast_period, slowperiod=self.slow_period, signalperiod=self.signal_period)
        df['macd_signal'] = np.where(df['macd'] > df['signal_line'], self.TradingSignals.BUY, self.TradingSignals.SELL)  # MACD 向上突破訊號線買入，反之賣出
        self.signal = df['macd_signal']
        return self.signal
