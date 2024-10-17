# 布林帶策略
import numpy as np
from .base_strategy import BaseStrategy


class BollingerBandsStrategy(BaseStrategy):
    def __init__(self, window=20, no_of_std=2):
        self.window = window
        self.no_of_std = no_of_std

    def apply_strategy(self, df):
        df['middle_band'] = df['Close'].rolling(self.window).mean()
        df['std_dev'] = df['Close'].rolling(self.window).std()
        df['upper_band'] = df['middle_band'] + (df['std_dev'] * self.no_of_std)
        df['lower_band'] = df['middle_band'] - (df['std_dev'] * self.no_of_std)
        df['bb_signal'] = np.where(df['Close'] < df['lower_band'], 1, np.where(df['Close'] > df['upper_band'], -1, 0))
        df['bb_positions'] = df['bb_signal'].diff()
        return df