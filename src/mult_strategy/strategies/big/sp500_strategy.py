import pandas as pd
import numpy as np
import yfinance as yf

from mult_strategy.strategies.base.base_strategy import BaseStrategy



class SP500ReferenceStrategy(BaseStrategy):
    def __init__(self, start_date, end_date, ma_window=50, rsi_period=14):
        super().__init__()
        
        # 下载 S&P 500 数据
        self.sp500_data = yf.download('^GSPC', start=start_date, end=end_date)
        
        # 计算移动平均线和 RSI 参数
        self.ma_window = ma_window
        self.rsi_period = rsi_period

    def calculate_rsi(self, prices):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    
    def apply_strategy(self, df):
        # 计算 S&P 500 的移动平均线
        self.sp500_data['sp500_ma'] = self.sp500_data['Close'].rolling(window=self.ma_window).mean()

        # 计算 S&P 500 的 RSI
        self.sp500_data['sp500_rsi'] = self.calculate_rsi(self.sp500_data['Close'])

        # 合并数据
        df = df.join(self.sp500_data[['sp500_ma', 'sp500_rsi']], how='left')

        # 基于 S&P 500 的指标生成信号
        df['sp500_signal'] = np.where(
            (df['Close'] > df['sp500_ma']) & (df['sp500_rsi'] < 30),  # 买入信号条件
            self.TradingSignals.BUY,
            np.where(
                (df['Close'] < df['sp500_ma']) & (df['sp500_rsi'] > 70),  # 卖出信号条件
                self.TradingSignals.SELL,
                self.TradingSignals.HOLD  # 默认保持持有
            )
        )

        return df['sp500_signal']