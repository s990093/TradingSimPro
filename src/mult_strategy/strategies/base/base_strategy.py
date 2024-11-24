import matplotlib.pyplot as plt
import pandas as pd
from abc import ABC, abstractmethod
import talib

class TradingSignals:
    BUY = 1      # 买入信号
    SELL = -1    # 卖出信号
    HOLD = 0     # 无操作信号

class BaseStrategy(ABC):
    TradingSignals = TradingSignals  # Reference to your TradingSignals class
    talib = talib

    def __init__(self, *args, **kwargs):
        self.signal = None  # Store signals for the strategy
        self.df = None  #

    def apply_strategy(self, df):
        raise NotImplementedError("Subclasses should implement this!")

