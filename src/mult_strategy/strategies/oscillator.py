import numpy as np
import talib
from .base.base_strategy import BaseStrategy

class StochasticOscillatorStrategy(BaseStrategy):
    def __init__(self, k_period=14, d_period=3):
        self.k_period = k_period
        self.d_period = d_period

    def apply_strategy(self, df):
        df['slowk'], df['slowd'] = talib.STOCH(df['High'], df['Low'], df['Close'],
                                              fastk_period=self.k_period,
                                              slowk_period=3,
                                              slowd_period=self.d_period)
        
        df['osc_signal'] = np.where((df['slowk'] < 20) & (df['slowd'] < 20), 1,
                                  np.where((df['slowk'] > 80) & (df['slowd'] > 80), -1, 0))
        return df['osc_signal']

class WilliamsRStrategy(BaseStrategy):
    def __init__(self, timeperiod=14):
        self.timeperiod = timeperiod

    def apply_strategy(self, df):
        df['willr'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=self.timeperiod)
        df['osc_signal'] = np.where(df['willr'] < -80, 1,
                                  np.where(df['willr'] > -20, -1, 0))
        return df['osc_signal']

class UltimateOscillatorStrategy(BaseStrategy):
    def apply_strategy(self, df):
        df['ultosc'] = talib.ULTOSC(df['High'], df['Low'], df['Close'],
                                   timeperiod1=7, timeperiod2=14, timeperiod3=28)
        
        df['osc_signal'] = np.where(df['ultosc'] < 30, 1,
                                  np.where(df['ultosc'] > 70, -1, 0))
        return df['osc_signal']

class KDJStrategy(BaseStrategy):
    def __init__(self, k_period=14, d_period=3, j_period=3):
        self.k_period = k_period
        self.d_period = d_period
        self.j_period = j_period

    def apply_strategy(self, df):
        df['slowk'], df['slowd'] = talib.STOCH(df['High'], df['Low'], df['Close'],
                                               fastk_period=self.k_period,
                                               slowk_period=self.d_period,
                                               slowd_period=self.j_period)
        df['slowj'] = 3 * df['slowk'] - 2 * df['slowd']
        df['osc_signal'] = np.where((df['slowk'] < 20) & (df['slowd'] < 20) & (df['slowj'] < 20), 1,
                                    np.where((df['slowk'] > 80) & (df['slowd'] > 80) & (df['slowj'] > 80), -1, 0))
        return df['osc_signal']

class ElderRayStrategy(BaseStrategy):
    def apply_strategy(self, df):
        df['bull_power'] = df['High'] - talib.EMA(df['Close'], timeperiod=13)
        df['bear_power'] = df['Low'] - talib.EMA(df['Close'], timeperiod=13)
        df['osc_signal'] = np.where(df['bull_power'] > 0, 1,
                                    np.where(df['bear_power'] < 0, -1, 0))
        return df['osc_signal']

class ChaikinOscillatorStrategy(BaseStrategy):
    def apply_strategy(self, df):
        ad = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
        df['chaikin_osc'] = talib.ADOSC(df['High'], df['Low'], df['Close'], df['Volume'],
                                        fastperiod=3, slowperiod=10)
        df['osc_signal'] = np.where(df['chaikin_osc'] > 0, 1,
                                    np.where(df['chaikin_osc'] < 0, -1, 0))
        return df['osc_signal']

class CoppockCurveStrategy(BaseStrategy):
    def apply_strategy(self, df):
        roc1 = talib.ROC(df['Close'], timeperiod=14)
        roc2 = talib.ROC(df['Close'], timeperiod=11)
        df['coppock_curve'] = talib.WMA(roc1 + roc2, timeperiod=10)
        df['osc_signal'] = np.where(df['coppock_curve'] > 0, 1,
                                    np.where(df['coppock_curve'] < 0, -1, 0))
        return df['osc_signal']

class RVIStrategy(BaseStrategy):
    def apply_strategy(self, df):
        self.df = df
        close = df['Close']
        open_ = df['Open']
        high = df['High']
        low = df['Low']
        
        # Manually calculate RVI
        numerator = (close - open_) + 2 * (close.shift(1) - open_.shift(1)) + 2 * (close.shift(2) - open_.shift(2)) + (close.shift(3) - open_.shift(3))
        denominator = (high - low) + 2 * (high.shift(1) - low.shift(1)) + 2 * (high.shift(2) - low.shift(2)) + (high.shift(3) - low.shift(3))
        
        rvi = numerator.rolling(window=4).mean() / denominator.rolling(window=4).mean()
        
        # Generate trading signals based on RVI
        signals = []
        for value in rvi:
            if value > 0:
                signals.append(self.TradingSignals.BUY)
            elif value < 0:
                signals.append(self.TradingSignals.SELL)
            else:
                signals.append(self.TradingSignals.HOLD)
        
        df['rvi_signal'] = signals
        return df['rvi_signal']

strategy_mapping = {
    "StochasticOscillatorStrategy": StochasticOscillatorStrategy,
    "WilliamsRStrategy": WilliamsRStrategy,
    "UltimateOscillatorStrategy": UltimateOscillatorStrategy,
    'KDJStrategy': KDJStrategy,                    # KDJ指标
    'ElderRayStrategy': ElderRayStrategy,          # Elder Ray Index
    'ChaikinOscillatorStrategy': ChaikinOscillatorStrategy,  # Chaikin Oscillator
    'CoppockCurveStrategy': CoppockCurveStrategy,  # Coppock Curve
    'RVIStrategy': RVIStrategy,                    # Relative Vigor Index
} 