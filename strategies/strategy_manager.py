class StrategyManager:
    def __init__(self, strategies):
        self.strategies = strategies

    def apply_all_strategies(self, df):
        for strategy in self.strategies:
            df = strategy.apply_strategy(df)
        return df