from mult_strategy.utility.calculate_returns_jit import fitness

class StrategyEvaluator:
    def __init__(self, df_strategy_np, df_data_open, signal_columns):
        self.df_strategy_np = df_strategy_np
        self.df_data_open = df_data_open
        self.signal_columns = signal_columns

    def evaluate(self, params) -> float:
        total_return, _ = fitness(
            params, 
            1.0, 
            1.0, 
            self.df_strategy_np, 
            self.df_data_open, 
            self.signal_columns
        )
        return -total_return 
    
    def evaluate_with_thresholds(self, params):
        return  fitness(
            params, 
            1.0, 
            1.0, 
            self.df_strategy_np, 
            self.df_data_open, 
            self.signal_columns
        )
