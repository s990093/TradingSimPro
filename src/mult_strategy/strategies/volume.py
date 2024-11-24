from .base.base_strategy import BaseStrategy
import numpy as np
import talib

class OBVStrategy(BaseStrategy):
    def apply_strategy(self, df):
        df['obv'] = talib.OBV(df['Close'], df['Volume'])
        df['obv_ma'] = talib.SMA(df['obv'], timeperiod=20)
        
        df['vol_signal'] = np.where(df['obv'] > df['obv_ma'], 1, -1)
        return df['vol_signal']

class ChaikinMoneyFlowStrategy(BaseStrategy):
    def __init__(self, timeperiod=20):
        self.timeperiod = timeperiod

    def apply_strategy(self, df):
        df['cmf'] = talib.ADOSC(df['High'], df['Low'], df['Close'], df['Volume'],
                               fastperiod=3, slowperiod=10)
        
        df['vol_signal'] = np.where(df['cmf'] > 0, 1, -1)
        return df['vol_signal']

class VolumeRateOfChangeStrategy(BaseStrategy):
    def __init__(self, timeperiod=25):
        self.timeperiod = timeperiod

    def apply_strategy(self, df):
        df['vroc'] = talib.ROC(df['Volume'], timeperiod=self.timeperiod)
        df['vol_signal'] = np.where(df['vroc'] > 20, 1,
                                  np.where(df['vroc'] < -20, -1, 0))
        return df['vol_signal']

class PriceVolumeTrendStrategy(BaseStrategy):
    def apply_strategy(self, df):
        # 計算價格變動百分比
        df['price_change'] = df['Close'].pct_change()
        
        # 計算 PVT
        df['pvt'] = (df['price_change'] * df['Volume']).cumsum()
        df['pvt_ma'] = talib.SMA(df['pvt'], timeperiod=20)
        
        df['vol_signal'] = np.where(df['pvt'] > df['pvt_ma'], 1, -1)
        return df['vol_signal']

class EMVStrategy(BaseStrategy):
    def apply_strategy(self, df):
        # 計算 Ease of Movement (EMV)
        distance_moved = ((df['High'] + df['Low']) / 2) - ((df['High'].shift(1) + df['Low'].shift(1)) / 2)
        box_ratio = (df['Volume'] / 100000000) / (df['High'] - df['Low'])
        df['emv'] = distance_moved / box_ratio
        df['emv_ma'] = talib.SMA(df['emv'], timeperiod=14)
        
        df['vol_signal'] = np.where(df['emv'] > df['emv_ma'], 1, -1)
        return df['vol_signal']

class FIStrategy(BaseStrategy):
    def apply_strategy(self, df):
        # 計算 Force Index (FI)
        df['fi'] = talib.ROC(df['Close'], timeperiod=1) * df['Volume']
        df['fi_ma'] = talib.SMA(df['fi'], timeperiod=13)
        
        df['vol_signal'] = np.where(df['fi'] > df['fi_ma'], 1, -1)
        return df['vol_signal']

class VWAPMomentumStrategy(BaseStrategy):
    def apply_strategy(self, df):
        # 計算 VWAP
        df['vwap'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        
        df['vol_signal'] = np.where(df['Close'] > df['vwap'], 1, -1)
        return df['vol_signal']

class KVOStrategy(BaseStrategy):
    def apply_strategy(self, df):
        # 計算 Klinger Volume Oscillator (KVO)
        df['kvo'] = talib.ADOSC(df['High'], df['Low'], df['Close'], df['Volume'], fastperiod=34, slowperiod=55)
        
        df['vol_signal'] = np.where(df['kvo'] > 0, 1, -1)
        return df['vol_signal']

class MFVStrategy(BaseStrategy):
    def apply_strategy(self, df):
        # 計算 Money Flow Volume (MFV)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        df['mfv'] = money_flow.rolling(window=14).sum()
        
        df['vol_signal'] = np.where(df['mfv'] > 0, 1, -1)
        return df['vol_signal']

strategy_mapping = {
    "OBVStrategy": OBVStrategy,
    "ChaikinMoneyFlowStrategy": ChaikinMoneyFlowStrategy,
    "VolumeRateOfChangeStrategy": VolumeRateOfChangeStrategy,
    "PriceVolumeTrendStrategy": PriceVolumeTrendStrategy,
    'EMVStrategy': EMVStrategy,                    # Ease of Movement
    'FIStrategy': FIStrategy,                      # Force Index
    'VWAPMomentumStrategy': VWAPMomentumStrategy,  # VWAP Momentum
    'KVOStrategy': KVOStrategy,                    # Klinger Volume Oscillator
    'MFVStrategy': MFVStrategy,     
}