import pandas as pd
import pickle

class StrategyManager:
    def __init__(self, strategies):
        self.strategies = strategies
        self.strategy_params = {}
        self.signal_columns = []
        
    def get_signal_columns(self):
        return self.signal_columns
                    
    def apply_all_strategies(self, df):
        results = []
        for strategy in self.strategies:
            _df = strategy.apply_strategy(df)
            
            self.signal_columns.append(_df.name)
                        
            results.append(_df)  

        final_df = pd.concat(results, axis=1)
        
        return final_df