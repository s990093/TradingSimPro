# 布林帶策略
import numpy as np
from .base_strategy import BaseStrategy


class ChannelBreakoutStrategy(BaseStrategy):
    def __init__(self, lookback_period=100):
        super().__init__()
        self.lookback_period = lookback_period

    def apply_strategy(self, df):
        # 计算最高高点和最低低点
        df['highest_high'] = df['Close'].rolling(window=self.lookback_period).max()
        df['lowest_low'] = df['Close'].rolling(window=self.lookback_period).min()

        # 生成突破信号
        df['channel_breakout_signal'] = np.where(
            df['Close'] > df['highest_high'].shift(1),
            self.TradingSignals.BUY,
            np.where(
                df['Close'] < df['lowest_low'].shift(1),
                self.TradingSignals.SELL,
                self.TradingSignals.HOLD
            )
        )

        # 将信号存储到类变量中
        self.signal = df['channel_breakout_signal']
        return self.signal
