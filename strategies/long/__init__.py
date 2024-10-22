import numpy as np
import pandas as pd
from strategies.base_strategy import BaseStrategy


class ValueInvestingStrategy(BaseStrategy):
    def __init__(self, valuation_metric="P/E", target_return=10):
        super().__init__()
        self.valuation_metric = valuation_metric
        self.target_return = target_return

    def apply_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        # 假設 df 包含市盈率（P/E）等基本面指標
        df['value_signal'] = np.where((df[self.valuation_metric] < self.target_return), 
                                      self.TradingSignals.BUY, 
                                      self.TradingSignals.HOLD)
        return df
