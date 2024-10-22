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
        
        # 使用 TradingSignals 类中的信号常量来替代数字
        df['bb_signal'] = np.where(df['Close'] < df['lower_band'], 
                                   self.TradingSignals.BUY, 
                                   np.where(df['Close'] > df['upper_band'], 
                                            self.TradingSignals.SELL, 
                                            self.TradingSignals.HOLD))
        
        
        df['bb_positions'] = df['bb_signal'].diff() 
        return df['bb_signal']