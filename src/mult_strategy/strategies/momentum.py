from mult_strategy.strategies.base.base_strategy import BaseStrategy


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
        rsi = self.talib.RSI(close, timeperiod=50)
        
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
        return df['rsi_signal'] 

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

class CCIStrategy(BaseStrategy):
    def apply_strategy(self, df):
        self.df = df
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # CCI計算
        cci = self.talib.CCI(high, low, close, timeperiod=14)
        
        # 根據CCI值生成交易信號
        signals = []
        for value in cci:
            if value > 100:
                signals.append(self.TradingSignals.SELL)
            elif value < -100:
                signals.append(self.TradingSignals.BUY)
            else:
                signals.append(self.TradingSignals.HOLD)
        
        df['cci_signal'] = signals
        return df['cci_signal']

class ROCStrategy(BaseStrategy):
    def apply_strategy(self, df):
        self.df = df
        close = df['Close']
        
        # ROC計算
        roc = self.talib.ROC(close, timeperiod=10)
        
        # 根據ROC值生成交易信號
        signals = []
        for value in roc:
            if value > 2:
                signals.append(self.TradingSignals.SELL)
            elif value < -2:
                signals.append(self.TradingSignals.BUY)
            else:
                signals.append(self.TradingSignals.HOLD)
        
        df['roc_signal'] = signals
        return df['roc_signal']

class CMOStrategy(BaseStrategy):
    def apply_strategy(self, df):
        self.df = df
        close = df['Close']
        
        # CMO計算
        cmo = self.talib.CMO(close, timeperiod=14)
        
        # 根據CMO值生成交易信號
        signals = []
        for value in cmo:
            if value > 50:
                signals.append(self.TradingSignals.SELL)
            elif value < -50:
                signals.append(self.TradingSignals.BUY)
            else:
                signals.append(self.TradingSignals.HOLD)
        
        df['cmo_signal'] = signals
        return df['cmo_signal']

class PPOStrategy(BaseStrategy):
    def apply_strategy(self, df):
        self.df = df
        close = df['Close']
        
        # PPO計算
        ppo = self.talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)
        
        # 根據PPO值生成交易信號
        signals = []
        for value in ppo:
            if value > 1:
                signals.append(self.TradingSignals.SELL)
            elif value < -1:
                signals.append(self.TradingSignals.BUY)
            else:
                signals.append(self.TradingSignals.HOLD)
        
        df['ppo_signal'] = signals
        return df['ppo_signal']

class KSTStrategy(BaseStrategy):
    def apply_strategy(self, df):
        self.df = df
        close = df['Close']
        
        # Manually calculate KST
        roc1 = close.pct_change(periods=10)
        roc2 = close.pct_change(periods=15)
        roc3 = close.pct_change(periods=20)
        roc4 = close.pct_change(periods=30)
        
        smaroc1 = roc1.rolling(window=10).mean()
        smaroc2 = roc2.rolling(window=10).mean()
        smaroc3 = roc3.rolling(window=10).mean()
        smaroc4 = roc4.rolling(window=15).mean()
        
        kst = smaroc1 + 2 * smaroc2 + 3 * smaroc3 + 4 * smaroc4
        
        # Generate trading signals based on KST
        signals = []
        for value in kst:
            if value > 0:
                signals.append(self.TradingSignals.BUY)
            elif value < 0:
                signals.append(self.TradingSignals.SELL)
            else:
                signals.append(self.TradingSignals.HOLD)
        
        df['kst_signal'] = signals
        return df['kst_signal']

class TRIXStrategy(BaseStrategy):
    def apply_strategy(self, df):
        self.df = df
        close = df['Close']
        
        # TRIX計算
        trix = self.talib.TRIX(close, timeperiod=30)
        
        # 根據TRIX值生成交易信號
        signals = []
        for value in trix:
            if value > 0:
                signals.append(self.TradingSignals.BUY)
            elif value < 0:
                signals.append(self.TradingSignals.SELL)
            else:
                signals.append(self.TradingSignals.HOLD)
        
        df['trix_signal'] = signals
        return df['trix_signal']

class DPOStrategy(BaseStrategy):
    def apply_strategy(self, df):
        self.df = df
        close = df['Close']
        
        # Manually calculate DPO
        timeperiod = 20
        sma = close.rolling(window=timeperiod).mean()
        dpo = close.shift(int((timeperiod / 2) + 1)) - sma
        
        # Generate trading signals based on DPO
        signals = []
        for value in dpo:
            if value > 0:
                signals.append(self.TradingSignals.BUY)
            elif value < 0:
                signals.append(self.TradingSignals.SELL)
            else:
                signals.append(self.TradingSignals.HOLD)
        
        df['dpo_signal'] = signals
        return df['dpo_signal']

class BullPowerStrategy(BaseStrategy):
    def apply_strategy(self, df):
        self.df = df
        high = df['High']
        close = df['Close']
        
        # Bull Power計算
        bull_power = high - self.talib.EMA(close, timeperiod=13)
        
        # 根據Bull Power值生成交易信號
        signals = []
        for value in bull_power:
            if value > 0:
                signals.append(self.TradingSignals.BUY)
            else:
                signals.append(self.TradingSignals.SELL)
        
        df['bull_power_signal'] = signals
        return df['bull_power_signal']

class BearPowerStrategy(BaseStrategy):
    def apply_strategy(self, df):
        self.df = df
        low = df['Low']
        close = df['Close']
        
        # Bear Power計算
        bear_power = low - self.talib.EMA(close, timeperiod=13)
        
        # 根據Bear Power值生成交易信號
        signals = []
        for value in bear_power:
            if value < 0:
                signals.append(self.TradingSignals.BUY)
            else:
                signals.append(self.TradingSignals.SELL)
        
        df['bear_power_signal'] = signals
        return df['bear_power_signal']

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
    'CCIStrategy': CCIStrategy,
    'ROCStrategy': ROCStrategy,
    'CMOStrategy': CMOStrategy,
    'PPOStrategy': PPOStrategy,
    'KSTStrategy': KSTStrategy,
    'TRIXStrategy': TRIXStrategy,
    'DPOStrategy': DPOStrategy,
    'BullPowerStrategy': BullPowerStrategy,
    'BearPowerStrategy': BearPowerStrategy,
}
