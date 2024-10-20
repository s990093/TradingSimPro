from .base_strategy import BaseStrategy

class TrendFollowingStrategy(BaseStrategy):
    def __init__(self, window = 50, *args, **kwargs):
        self.window = window
        
    def apply_strategy(self, df):
        df['trend_following_signal'] = 0
        # 計算移動平均
        moving_avg = df['Close'].rolling(window=self.window).mean()
        
        # 設定買入和賣出信號
        df.loc[df['Close'] > moving_avg, 'trend_following_signal'] = 1  # 上升趨勢，買入
        df.loc[df['Close'] < moving_avg, 'trend_following_signal'] = -1 # 下降趨勢，賣出

        return df["trend_following_signal"]
