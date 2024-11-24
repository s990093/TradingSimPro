import numpy as np
import talib
from .base.base_strategy import BaseStrategy

# BBands Strategy
class BollingerBandsStrategy(BaseStrategy):
    def __init__(self, timeperiod=40, nbdevup=2, nbdevdn=2, matype=0):
        self.timeperiod = timeperiod
        self.nbdevup = nbdevup
        self.nbdevdn = nbdevdn
        self.matype = matype

    def apply_strategy(self, df):
        df['upperband'], df['middleband'], df['lowerband'] = talib.BBANDS(
            df['Close'], timeperiod=self.timeperiod, nbdevup=self.nbdevup, nbdevdn=self.nbdevdn, matype=self.matype
        )
        df['bollinger_signal'] = np.where(df['Close'] > df['upperband'], -1,  
                                          np.where(df['Close'] < df['lowerband'], 1, 0))
        return df['bollinger_signal']


# DEMA Strategy
class DEMAStrategy(BaseStrategy):
    def __init__(self, timeperiod=30):
        self.timeperiod = timeperiod

    def apply_strategy(self, df):
        df['dema'] = talib.DEMA(df['Close'], timeperiod=self.timeperiod)
        df['dema_signal'] = np.where(df['Close'] > df['dema'], 1, -1)
        return df['dema_signal']


# EMA Strategy
class EMAStrategy(BaseStrategy):
    def __init__(self, timeperiod=30):
        self.timeperiod = timeperiod

    def apply_strategy(self, df):
        df['ema'] = talib.EMA(df['Close'], timeperiod=self.timeperiod)
        df['ema_signal'] = np.where(df['Close'] > df['ema'], 1, -1)
        return df['ema_signal']


# HT Trendline Strategy
class HilbertTransformTrendlineStrategy(BaseStrategy):
    def apply_strategy(self, df):
        df['ht_trendline'] = talib.HT_TRENDLINE(df['Close'])
        df['ht_trendline_signal'] = np.where(df['Close'] > df['ht_trendline'], 1, -1)
        return df['ht_trendline_signal']


# KAMA Strategy
class KAMAStrategy(BaseStrategy):
    def __init__(self, timeperiod=30):
        self.timeperiod = timeperiod

    def apply_strategy(self, df):
        df['kama'] = talib.KAMA(df['Close'], timeperiod=self.timeperiod)
        df['kama_signal'] = np.where(df['Close'] > df['kama'], 1, -1)
        return df['kama_signal']


# MA Strategy
class MovingAverageStrategy(BaseStrategy):
    def __init__(self, timeperiod=30, matype=0):
        self.timeperiod = timeperiod
        self.matype = matype

    def apply_strategy(self, df):
        df['ma'] = talib.MA(df['Close'], timeperiod=self.timeperiod, matype=self.matype)
        df['ma_signal'] = np.where(df['Close'] > df['ma'], 1, -1)
        return df['ma_signal']


# MAMA Strategy
class MAMAStrategy(BaseStrategy):
    def __init__(self, fastlimit=0.5, slowlimit=0.05):
        self.fastlimit = fastlimit
        self.slowlimit = slowlimit

    def apply_strategy(self, df):
        df['mama'], df['fama'] = talib.MAMA(df['Close'], fastlimit=self.fastlimit, slowlimit=self.slowlimit)
        df['mama_signal'] = np.where(df['Close'] > df['mama'], 1, -1)
        return df['mama_signal']


# MAVP Strategy
class MAVPStrategy(BaseStrategy):
    def __init__(self, periods=30, minperiod=2, maxperiod=30, matype=0):
        self.periods = periods
        self.minperiod = minperiod
        self.maxperiod = maxperiod
        self.matype = matype

    def apply_strategy(self, df):
        # Ensure 'Close' is of type float64
        df['Close'] = df['Close'].astype(np.float64)

        # Check for NaN values and handle them (e.g., by dropping or filling)
        if df['Close'].isnull().any():
            df['Close'].fillna(method='ffill', inplace=True)  # Forward fill or use another method

        # Create an array of periods
        periods_array = np.arange(self.minperiod, self.maxperiod + 1)

        # Calculate MAVP
        df['mavp'] = talib.MAVP(df['Close'], periods_array, matype=self.matype)

        # Generate signals based on MAVP
        df['mavp_signal'] = np.where(df['Close'] > df['mavp'], 1, -1)
        return df['mavp_signal']
    
# MidPoint Strategy
class MidPointStrategy(BaseStrategy):
    def __init__(self, timeperiod=14):
        self.timeperiod = timeperiod

    def apply_strategy(self, df):
        df['midpoint'] = talib.MIDPOINT(df['Close'], timeperiod=self.timeperiod)
        df['midpoint_signal'] = np.where(df['Close'] > df['midpoint'], 1, -1)
        return df['midpoint_signal']


# MidPrice Strategy
class MidPriceStrategy(BaseStrategy):
    def __init__(self, timeperiod=14):
        self.timeperiod = timeperiod

    def apply_strategy(self, df):
        df['midprice'] = talib.MIDPRICE(df['High'], df['Low'], timeperiod=self.timeperiod)
        df['midprice_signal'] = np.where(df['Close'] > df['midprice'], 1, -1)
        return df['midprice_signal']


# SAR Strategy
class SARStrategy(BaseStrategy):
    def __init__(self, acceleration=0.02, maximum=0.2):
        self.acceleration = acceleration
        self.maximum = maximum

    def apply_strategy(self, df):
        df['sar'] = talib.SAR(df['High'], df['Low'], acceleration=self.acceleration, maximum=self.maximum)
        df['sar_signal'] = np.where(df['Close'] > df['sar'], 1, -1)
        return df['sar_signal']


# SMA Strategy
class SimpleMovingAverageStrategy(BaseStrategy):
    def __init__(self, timeperiod=30):
        self.timeperiod = timeperiod

    def apply_strategy(self, df):
        df['sma'] = talib.SMA(df['Close'], timeperiod=self.timeperiod)
        df['sma_signal'] = np.where(df['Close'] > df['sma'], 1, -1)
        return df['sma_signal']


# T3 Strategy
class T3Strategy(BaseStrategy):
    def __init__(self, timeperiod=10, vfactor=0.7):
        self.timeperiod = timeperiod
        self.vfactor = vfactor

    def apply_strategy(self, df):
        df['t3'] = talib.T3(df['Close'], timeperiod=self.timeperiod, vfactor=self.vfactor)
        df['t3_signal'] = np.where(df['Close'] > df['t3'], 1, -1)
        return df['t3_signal']


# WMA Strategy
class WeightedMovingAverageStrategy(BaseStrategy):
    def __init__(self, timeperiod=30):
        self.timeperiod = timeperiod

    def apply_strategy(self, df):
        df['wma'] = talib.WMA(df['Close'], timeperiod=self.timeperiod)
        df['wma_signal'] = np.where(df['Close'] > df['wma'], 1, -1)
        return df['wma_signal']


# TEMA Strategy
class TEMAStrategy(BaseStrategy):
    def __init__(self, timeperiod=30):
        self.timeperiod = timeperiod

    def apply_strategy(self, df):
        df['tema'] = talib.TEMA(df['Close'], timeperiod=self.timeperiod)
        df['tema_signal'] = np.where(df['Close'] > df['tema'], 1, -1)
        return df['tema_signal']


# TRIMA Strategy
class TRIMAStrategy(BaseStrategy):
    def __init__(self, timeperiod=30):
        self.timeperiod = timeperiod

    def apply_strategy(self, df):
        df['trima'] = talib.TRIMA(df['Close'], timeperiod=self.timeperiod)
        df['trima_signal'] = np.where(df['Close'] > df['trima'], 1, -1)
        return df['trima_signal']


# VWAP Strategy
class VWAPStrategy(BaseStrategy):
    def apply_strategy(self, df):
        # Calculate VWAP
        df['vwap'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        df['vwap_signal'] = np.where(df['Close'] > df['vwap'], 1, -1)
        return df['vwap_signal']

# HMA Strategy
class HMAStrategy(BaseStrategy):
    def __init__(self, timeperiod=30):
        self.timeperiod = timeperiod

    def apply_strategy(self, df):
        # Calculate HMA
        half_length = int(self.timeperiod / 2)
        sqrt_length = int(np.sqrt(self.timeperiod))
        df['wma_half'] = talib.WMA(df['Close'], timeperiod=half_length)
        df['wma_full'] = talib.WMA(df['Close'], timeperiod=self.timeperiod)
        df['hma'] = talib.WMA(2 * df['wma_half'] - df['wma_full'], timeperiod=sqrt_length)
        df['hma_signal'] = np.where(df['Close'] > df['hma'], 1, -1)
        return df['hma_signal']

# ZLEMA Strategy
class ZLEMAStrategy(BaseStrategy):
    def __init__(self, timeperiod=30):
        self.timeperiod = timeperiod

    def apply_strategy(self, df):
        # Calculate ZLEMA
        lag = (self.timeperiod - 1) / 2
        df['ema'] = talib.EMA(df['Close'], timeperiod=self.timeperiod)
        df['zlema'] = df['Close'] + (df['Close'] - df['Close'].shift(int(lag)))
        df['zlema_signal'] = np.where(df['Close'] > df['zlema'], 1, -1)
        return df['zlema_signal']

# JMA Strategy
class JMAStrategy(BaseStrategy):
    def __init__(self, timeperiod=30, phase=0):
        self.timeperiod = timeperiod
        self.phase = phase

    def apply_strategy(self, df):
        # Placeholder for JMA calculation
        # JMA is a proprietary indicator, so this is a simplified version
        df['jma'] = talib.EMA(df['Close'], timeperiod=self.timeperiod)  # Simplified
        df['jma_signal'] = np.where(df['Close'] > df['jma'], 1, -1)
        return df['jma_signal']

# Mapping of all strategy classes to string identifiers
strategy_mapping = {
    "BollingerBandsStrategy": BollingerBandsStrategy,
    "DEMAStrategy": DEMAStrategy,
    "EMAStrategy": EMAStrategy,
    "HilbertTransformTrendlineStrategy": HilbertTransformTrendlineStrategy,
    "KAMAStrategy": KAMAStrategy,
    "MovingAverageStrategy": MovingAverageStrategy,
    "MAMAStrategy": MAMAStrategy,
    "MAVPStrategy": MAVPStrategy,
    "MidPointStrategy": MidPointStrategy,
    "MidPriceStrategy": MidPriceStrategy,
    "SARStrategy": SARStrategy,
    "SimpleMovingAverageStrategy": SimpleMovingAverageStrategy,
    "T3Strategy": T3Strategy,
    "WeightedMovingAverageStrategy": WeightedMovingAverageStrategy,
    "TEMAStrategy": TEMAStrategy,
    "TRIMAStrategy": TRIMAStrategy,
    'VWAPStrategy': VWAPStrategy,                  # Volume Weighted Average Price
    'HMAStrategy': HMAStrategy,                    # Hull Moving Average
    'ZLEMAStrategy': ZLEMAStrategy,                # Zero-Lag Exponential Moving Average
    'JMAStrategy': JMAStrategy,                    # Jurik Moving Average
}
