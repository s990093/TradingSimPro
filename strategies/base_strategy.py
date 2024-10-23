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
        self.df = None  # Store the dataframe used

    def apply_strategy(self, df):
        raise NotImplementedError("Subclasses should implement this!")
    
    def visualize(self):
        """
        Visualizes the strategy signals on a price chart with buy/sell markers.
        """
        if self.df is None or self.signal is None:
            raise ValueError("Strategy has not been applied to a dataframe yet.")

        plt.figure(figsize=(12, 6))

        # Plot the closing prices
        plt.plot(self.df['Close'], label='Close Price', color='blue', alpha=0.6)

        # Buy signals
        buy_signals = self.df[self.signal == self.TradingSignals.BUY]
        plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy Signal', alpha=1)

        # Sell signals
        sell_signals = self.df[self.signal == self.TradingSignals.SELL]
        plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell Signal', alpha=1)

        # Add titles and labels
        plt.title('Trading Strategy Signals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()

        # Display the plot
        plt.grid(True)
        plt.show()

