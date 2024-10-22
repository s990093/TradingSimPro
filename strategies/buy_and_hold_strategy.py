import numpy as np
from .base_strategy import BaseStrategy
import talib as ta

class BuyAndHoldStrategy(BaseStrategy):
    name = "buy_hold_signal"
    def __init__(self):
        super().__init__()

    def apply_strategy(self, df):
        df['buy_hold_signal'] = self.TradingSignals.BUY
        return df['buy_hold_signal']
