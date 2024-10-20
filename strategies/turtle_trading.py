from .base_strategy import BaseStrategy

class TurtleTradingStrategy(BaseStrategy):
    def apply_strategy(self, df):
        df['high_55'] = df['High'].rolling(window=55).max()
        df['low_55'] = df['Low'].rolling(window=55).min()
        
        df['turtle_trading_signal'] = 0
        
        # 設定買入和賣出信號
        df.loc[df['Close'] > df['high_55'], 'turtle_trading_signal'] = 1   # 突破55天高點，買入
        df.loc[df['Close'] < df['low_55'], 'turtle_trading_signal'] = -1   # 跌破55天低點，賣出

        return df["turtle_trading_signal"]
