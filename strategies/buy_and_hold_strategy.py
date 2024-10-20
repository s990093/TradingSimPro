import numpy as np
from .base_strategy import BaseStrategy
import talib as ta

class BuyAndHoldStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()

    def apply_strategy(self, df):
        df['buy_hold_signal'] = self.TradingSignals.BUY  # 代表買入，之後不會再變
        self.signal = df['buy_hold_signal']
        return self.signal
