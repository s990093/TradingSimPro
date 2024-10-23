from strategies.base_strategy import BaseStrategy

class ADXStrategy(BaseStrategy):
    def apply_strategy(self, df):
        self.df = df
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # ADX计算
        adx = self.talib.ADX(high, low, close, timeperiod=14)
        
        # 根据ADX值生成交易信号
        signals = []
        for value in adx:
            if value > 25:
                signals.append(self.TradingSignals.BUY)
            elif value < 20:
                signals.append(self.TradingSignals.SELL)
            else:
                signals.append(self.TradingSignals.HOLD)

        # 将信号添加到 DataFrame
        df['adx_signal'] = signals
        return df['adx_signal']

class ADXRStrategy(BaseStrategy):
    def apply_strategy(self, df):
        self.df = df
        high = df['High']  # 修改为 'High'
        low = df['Low']    # 修改为 'Low'
        close = df['Close'] # 修改为 'Close'

        # ADXR计算
        adxr = self.talib.ADXR(high, low, close, timeperiod=14)
        
        # 根据ADXR值生成交易信号
        signals = []
        for value in adxr:
            if value > 25:
                signals.append(self.TradingSignals.BUY)
            elif value < 20:
                signals.append(self.TradingSignals.SELL)
            else:
                signals.append(self.TradingSignals.HOLD)

        # 将信号添加到 DataFrame
        df['adxr_signal'] = signals
        return df['adxr_signal']

class APOCStrategy(BaseStrategy):
    def apply_strategy(self, df):
        self.df = df
        close = df['Close']  # 修改为 'Close'
        
        # APO计算
        apo = self.talib.APO(close, fastperiod=12, slowperiod=26, matype=0)

        # 根据APO值生成交易信号
        signals = []
        for value in apo:
            if value > 0:
                signals.append(self.TradingSignals.BUY)
            elif value < 0:
                signals.append(self.TradingSignals.SELL)
            else:
                signals.append(self.TradingSignals.HOLD)

        # 将信号添加到 DataFrame
        df['apo_signal'] = signals
        return   df['apo_signal']

class AROONStrategy(BaseStrategy):
    def apply_strategy(self, df):
        self.df = df
        high = df['High']  # 修改为 'High'
        low = df['Low']    # 修改为 'Low'

        # Aroon计算
        aroondown, aroonup = self.talib.AROON(high, low, timeperiod=14)

        # 根据Aroon Up/Down生成交易信号
        signals = []
        for up, down in zip(aroonup, aroondown):
            if up > down:
                signals.append(self.TradingSignals.BUY)
            elif up < down:
                signals.append(self.TradingSignals.SELL)
            else:
                signals.append(self.TradingSignals.HOLD)

        # 将信号添加到 DataFrame
        df['aroon_signal'] = signals
        return    df['aroon_signal']

# 下面的策略类同样进行类似修改
class MACDStrategy(BaseStrategy):
    def apply_strategy(self, df):
        self.df = df
        close = df['Close']  # 修改为 'Close'
        
        # MACD计算
        macd, macdsignal, macdhist = self.talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

        # 根据MACD差值生成交易信号
        signals = []
        for macd_value, signal_value in zip(macd, macdsignal):
            if macd_value > signal_value:
                signals.append(self.TradingSignals.BUY)
            elif macd_value < signal_value:
                signals.append(self.TradingSignals.SELL)
            else:
                signals.append(self.TradingSignals.HOLD)

        # 将信号添加到 DataFrame
        df['macd_signal'] = signals
        return     df['macd_signal']

class MACDEXTStrategy(BaseStrategy):
    def apply_strategy(self, df):
        self.df = df
        close = df['Close']  # 修改为 'Close'
        
        # MACDEXT计算
        macd, macdsignal, macdhist = self.talib.MACDEXT(
            close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0
        )
        
        # 根据MACD值生成交易信号
        signals = []
        for macd_value, signal_value in zip(macd, macdsignal):
            if macd_value > signal_value:
                signals.append(self.TradingSignals.BUY)
            elif macd_value < signal_value:
                signals.append(self.TradingSignals.SELL)
            else:
                signals.append(self.TradingSignals.HOLD)
        
        # 将信号添加到 DataFrame
        df['macdext_signal'] = signals
        return  df['macdext_signal']

class MFIStrategy(BaseStrategy):
    def apply_strategy(self, df):
        self.df = df
        high = df['High']  # 修改为 'High'
        low = df['Low']    # 修改为 'Low'
        close = df['Close'] # 修改为 'Close'
        volume = df['Volume'] # 修改为 'Volume'
        
        # MFI计算
        mfi = self.talib.MFI(high, low, close, volume, timeperiod=14)
        
        # 根据MFI值生成交易信号
        signals = []
        for value in mfi:
            if value > 80:
                signals.append(self.TradingSignals.SELL)
            elif value < 20:
                signals.append(self.TradingSignals.BUY)
            else:
                signals.append(self.TradingSignals.HOLD)
        
        # 将信号添加到 DataFrame
        df['mfi_signal'] = signals
        return    df['mfi_signal']

class RSIStrategy(BaseStrategy):
    def apply_strategy(self, df):
        self.df = df
        close = df['Close']  # 修改为 'Close'
        
        # RSI计算
        rsi = self.talib.RSI(close, timeperiod=14)
        
        # 根据RSI值生成交易信号
        signals = []
        for value in rsi:
            if value > 70:
                signals.append(self.TradingSignals.SELL)
            elif value < 30:
                signals.append(self.TradingSignals.BUY)
            else:
                signals.append(self.TradingSignals.HOLD)
        
        # 将信号添加到 DataFrame
        df['rsi_signal'] = signals
        return     df['rsi_signal'] 

class STOCHStrategy(BaseStrategy):
    def apply_strategy(self, df):
        self.df = df
        high = df['High']  # 修改为 'High'
        low = df['Low']    # 修改为 'Low'
        close = df['Close'] # 修改为 'Close'
        
        # STOCH计算
        slowk, slowd = self.talib.STOCH(
            high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0
        )
        
        # 根据STOCH值生成交易信号
        signals = []
        for k, d in zip(slowk, slowd):
            if k > d:
                signals.append(self.TradingSignals.BUY)
            elif k < d:
                signals.append(self.TradingSignals.SELL)
            else:
                signals.append(self.TradingSignals.HOLD)
        
        # 将信号添加到 DataFrame
        df['stoch_signal'] = signals
        return    df['stoch_signal'] 

class WILLRStrategy(BaseStrategy):
    def apply_strategy(self, df):
        self.df = df
        high = df['High']  # 修改为 'High'
        low = df['Low']    # 修改为 'Low'
        close = df['Close'] # 修改为 'Close'
        
        # WILLR计算
        willr = self.talib.WILLR(high, low, close, timeperiod=14)
        
        # 根据WILLR值生成交易信号
        signals = []
        for value in willr:
            if value < -80:
                signals.append(self.TradingSignals.BUY)
            elif value > -20:
                signals.append(self.TradingSignals.SELL)
            else:
                signals.append(self.TradingSignals.HOLD)
        
        # 将信号添加到 DataFrame
        df['willr_signal'] = signals
        return  df['willr_signal']


strategy_mapping = {
    'ADXStrategy': ADXStrategy,
    'ADXRStrategy': ADXRStrategy,
    'APOCStrategy': APOCStrategy,
    'AROONStrategy': AROONStrategy,
    'MACDStrategy': MACDStrategy,
    'MACDEXTStrategy': MACDEXTStrategy,
    'MFIStrategy': MFIStrategy,
    'RSIStrategy': RSIStrategy,
    'STOCHStrategy': STOCHStrategy,
    'WILLRStrategy': WILLRStrategy,
}
