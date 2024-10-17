import numpy as np
import talib as ta
from .base_strategy import BaseStrategy


class StochasticOscillatorStrategy(BaseStrategy):
    def apply_strategy(self, df):
        df['k'], df['d'] = ta.STOCH(df['High'], df['Low'], df['Close'])
        df['stochastic_signal'] = np.where(df['k'] > df['d'], 1, -1)
        df['stochastic_positions'] = df['stochastic_signal'].diff()
        return df