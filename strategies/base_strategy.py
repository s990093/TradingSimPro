from abc import ABC, abstractmethod

class TradingSignals:
    BUY = 1      # 买入信号
    SELL = -1    # 卖出信号
    HOLD = 0     # 无操作信号

class BaseStrategy(ABC):
    TradingSignals = TradingSignals

    def __init__(self, *args, **kwargs):
        self.signal = None
        
    @abstractmethod
    def apply_strategy(self, df):
        """
        每個具體策略的核心邏輯。
        必須在每個策略類中實現此方法。
        """
        pass
