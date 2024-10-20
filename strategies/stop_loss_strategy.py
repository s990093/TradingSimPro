from .base_strategy import BaseStrategy

class StopLossStrategy(BaseStrategy):
    def __init__(self, stop_loss_percent=0.1):
        self.stop_loss_percent = stop_loss_percent

    def apply_strategy(self, df):
        df['stop_loss_signal'] = 0
        buy_price = None

        for i in range(1, len(df)):
            if df['stop_loss_signal'].iloc[i - 1] == 1:  # 持仓中
                buy_price = df['Close'].iloc[i - 1]
                # 检查是否触发止损
                if df['Close'].iloc[i] < buy_price * (1 - self.stop_loss_percent):
                    df.at[i, 'stop_loss_signal'] = -1  # 止损，卖出
            else:
                # 其他进场策略，例如 MA 交叉等
                if self.enter_buy_signal_condition(df, i):  # 自定义进场条件
                    df.at[i, 'stop_loss_signal'] = 1  # 买入信号

        return df['stop_loss_signal']

    def enter_buy_signal_condition(self, df, index):
        # 这里可以添加您自己的入场条件逻辑，例如 MA 交叉
        # 这只是一个示例，您可以根据实际情况修改
        return df['Close'].iloc[index] > df['Close'].rolling(window=5).mean().iloc[index]
