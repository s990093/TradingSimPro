from strategies.base_strategy import BaseStrategy

class BreakoutStrategy(BaseStrategy):
    def apply_strategy(self, df):
        # 確認 DataFrame 包含必要的列
        required_columns = ['High', 'Low', 'Close']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame is missing required columns: {required_columns}")

        df['breakout_signal'] = 0
        df['high_20'] = df['High'].rolling(window=20).max()
        df['low_20'] = df['Low'].rolling(window=20).min()
        
        # 使用 .loc 設定信號值
        df.loc[df['Close'] > df['high_20'], 'breakout_signal'] = 1  # 突破，買入
        df.loc[df['Close'] < df['low_20'], 'breakout_signal'] = -1  # 跌破，賣出
        
        # df['breakout_signal'] = df['signal'].diff()
        
        # self.signal = df['breakout_signal']        

        return df["breakout_signal"]
