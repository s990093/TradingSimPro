import numpy as np
from .base_strategy import BaseStrategy


class MomentumStrategy(BaseStrategy):
    def __init__(self, window=10):
        self.window = window

    def apply_strategy(self, df):
        df['momentum'] = df['Close'].diff(self.window)
        df['momentum_signal'] = np.where(df['momentum'] > 0, 1, 0)
        df['momentum_positions'] = df['momentum_signal'].diff()
        return df