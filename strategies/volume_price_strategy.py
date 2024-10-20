from .base_strategy import BaseStrategy

class VolumePriceStrategy(BaseStrategy):
    def apply_strategy(self, df):
        df['volume_price_signal'] = 0
        avg_volume = df['Volume'].rolling(window=20).mean()  # 計算平均成交量

        # 設定買入信號
        df.loc[(df['Close'] > df['Close'].shift(1)) & (df['Volume'] > avg_volume), 'volume_price_signal'] = 1  # 買入

        # 設定賣出信號，避免 SettingWithCopyWarning
        df.loc[(df['Close'] < df['Close'].shift(1)) & (df['Volume'] > avg_volume), 'volume_price_signal'] = -1  # 賣出
        
        return df['volume_price_signal'] 
