import numpy as np
from .base_strategy import BaseStrategy


class BollingerBandsStrategy(BaseStrategy): 
    def __init__(self, window=20, no_of_std=2):
        self.window = window
        self.no_of_std = no_of_std

    def apply_strategy(self, df):
        # 使用 talib 來計算布林帶
        upper_band, middle_band, lower_band  = self.talib.BBANDS(df['Close'], 
                                                                             timeperiod=self.window, 
                                                                             nbdevup=self.no_of_std, 
                                                                             nbdevdn=self.no_of_std, 
                                                                             matype=0)
        
        
        df['upper_band'] = upper_band
        df['middle_band'] = middle_band
        df['lower_band'] = lower_band

        # 使用 TradingSignals 类中的信号常量来替代数字
        df['bb_signal'] = np.where(df['Close'] < lower_band,
                                   self.TradingSignals.BUY,
                                   np.where(df['Close'] > upper_band,
                                            self.TradingSignals.SELL,
                                            self.TradingSignals.HOLD))
        
        df['bb_positions'] = df['bb_signal'].diff()
        return df['bb_signal']
