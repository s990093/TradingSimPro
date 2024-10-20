from .base_strategy import BaseStrategy

class MeanReversionStrategy(BaseStrategy):
    def apply_strategy(self, df):
        # 計算20日移動平均和標準差
        mean_price = df['Close'].rolling(window=20).mean()
        std_dev = df['Close'].rolling(window=20).std()
        
        # 初始化信號列
        df['mean_reversion_signal'] = 0
        
        # 設定買入和賣出信號
        df.loc[df['Close'] < mean_price - 2 * std_dev, 'mean_reversion_signal'] = 1  # 買入
        df.loc[df['Close'] > mean_price + 2 * std_dev, 'mean_reversion_signal'] = -1 # 賣出
        
        return df["mean_reversion_signal"]
