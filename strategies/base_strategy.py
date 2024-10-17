from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    @abstractmethod
    def apply_strategy(self, df):
        """
        每個具體策略的核心邏輯。
        必須在每個策略類中實現此方法。
        """
        pass
