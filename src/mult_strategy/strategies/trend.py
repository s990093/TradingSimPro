import numpy as np
import talib
import pandas as pd
from .base.base_strategy import BaseStrategy

class ADXTrendStrategy(BaseStrategy):
    def __init__(self, timeperiod=28):
        self.timeperiod = timeperiod

    def apply_strategy(self, df):
        adx = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=self.timeperiod)
        trend_signal = np.where(adx > 25, 1, -1)
        
        # Use pd.concat to add new columns at once
        df = pd.concat([df, pd.DataFrame({'adx': adx, 'trend_signal': trend_signal})], axis=1)
        
        return df['trend_signal']

class IchimokuStrategy(BaseStrategy):
    def apply_strategy(self, df):
        tenkan_period = 9
        kijun_period = 26
        senkou_period = 52
        
        # 計算轉換線（Tenkan-sen）和基準線（Kijun-sen）
        high_values = df['High'].rolling(window=tenkan_period).max()
        low_values = df['Low'].rolling(window=tenkan_period).min()
        df['tenkan_sen'] = (high_values + low_values) / 2
        
        high_values = df['High'].rolling(window=kijun_period).max()
        low_values = df['Low'].rolling(window=kijun_period).min()
        df['kijun_sen'] = (high_values + low_values) / 2
        
        df['trend_signal'] = np.where(df['tenkan_sen'] > df['kijun_sen'], 1, -1)
        return df['trend_signal']

class SuperTrendStrategy(BaseStrategy):
    def __init__(self, period=10, multiplier=3):
        self.period = period
        self.multiplier = multiplier

    def apply_strategy(self, df):
        # 計算 ATR
        df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=self.period)
        
        # 計算基本帶
        df['basic_upper'] = ((df['High'] + df['Low']) / 2) + (self.multiplier * df['atr'])
        df['basic_lower'] = ((df['High'] + df['Low']) / 2) - (self.multiplier * df['atr'])
        
        df['trend_signal'] = np.where(df['Close'] > df['basic_upper'], 1, 
                                    np.where(df['Close'] < df['basic_lower'], -1, 0))
        return df['trend_signal']

class PABOLStrategy(BaseStrategy):
    def __init__(self, period=20, deviation=2):
        self.period = period
        self.deviation = deviation

    def apply_strategy(self, df):
        # 計算移動平均和標準差
        df['ma'] = talib.SMA(df['Close'], timeperiod=self.period)
        df['std'] = df['Close'].rolling(window=self.period).std()
        
        # 計算上下軌
        df['upper'] = df['ma'] + (self.deviation * df['std'])
        df['lower'] = df['ma'] - (self.deviation * df['std'])
        
        df['trend_signal'] = np.where(df['Close'] > df['upper'], 1,
                                    np.where(df['Close'] < df['lower'], -1, 0))
        return df['trend_signal']

class VortexStrategy(BaseStrategy):
    def __init__(self, timeperiod=14):
        self.timeperiod = timeperiod

    def apply_strategy(self, df):
        df['vi_plus'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=self.timeperiod)
        df['vi_minus'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=self.timeperiod)
        df['trend_signal'] = np.where(df['vi_plus'] > df['vi_minus'], 1, -1)
        return df['trend_signal']

class TTMTrendStrategy(BaseStrategy):
    def apply_strategy(self, df):
        # 假設 TTM Trend 是基於某些指標的計算
        # 這裡需要具體的計算方法
        df['trend_signal'] = np.where(df['Close'] > df['Close'].shift(1), 1, -1)
        return df['trend_signal']

class KeltnerChannelStrategy(BaseStrategy):
    def __init__(self, period=20, multiplier=2):
        self.period = period
        self.multiplier = multiplier

    def apply_strategy(self, df):
        df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=self.period)
        df['ma'] = talib.SMA(df['Close'], timeperiod=self.period)
        df['upper'] = df['ma'] + (self.multiplier * df['atr'])
        df['lower'] = df['ma'] - (self.multiplier * df['atr'])
        df['trend_signal'] = np.where(df['Close'] > df['upper'], 1, 
                                    np.where(df['Close'] < df['lower'], -1, 0))
        return df['trend_signal']

class MAEnvelopeStrategy(BaseStrategy):
    def __init__(self, period=20, deviation=0.02):
        self.period = period
        self.deviation = deviation

    def apply_strategy(self, df):
        df['ma'] = talib.SMA(df['Close'], timeperiod=self.period)
        df['upper'] = df['ma'] * (1 + self.deviation)
        df['lower'] = df['ma'] * (1 - self.deviation)
        df['trend_signal'] = np.where(df['Close'] > df['upper'], 1,
                                    np.where(df['Close'] < df['lower'], -1, 0))
        return df['trend_signal']

class PSARStrategy(BaseStrategy):
    def apply_strategy(self, df):
        df['psar'] = talib.SAR(df['High'], df['Low'])
        df['trend_signal'] = np.where(df['Close'] > df['psar'], 1, -1)
        return df['trend_signal']

strategy_mapping = {
    "ADXTrendStrategy": ADXTrendStrategy,
    "IchimokuStrategy": IchimokuStrategy,
    "SuperTrendStrategy": SuperTrendStrategy,
    "PABOLStrategy": PABOLStrategy,
    'VortexStrategy': VortexStrategy,              # Vortex Indicator
    'TTMTrendStrategy': TTMTrendStrategy,          # TTM Trend
    'KeltnerChannelStrategy': KeltnerChannelStrategy,  # Keltner Channel
    'MAEnvelopeStrategy': MAEnvelopeStrategy,      # Moving Average Envelope
    'PSARStrategy': PSARStrategy,                  # Parabolic SAR
} 