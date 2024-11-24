import json

import numpy as np
from mult_strategy.strategies import create_strategies
from mult_strategy.utility.calculate_returns_jit import fitness
from mult_strategy.utility.helper.tool import trades_to_dataframe
from config import StrategyConfig
from multi_weight_strategy.evaluator.strategy_evaluator import StrategyEvaluator
from multi_weight_strategy.visualization.plotter import StrategyPlotter
from utils.stock_data_cache import StockDataCache

def backtest(best_solution):
    # Data preparation
    df_data = StockDataCache(
        StrategyConfig.target_stock, 
        StrategyConfig.start_date, 
        StrategyConfig.end_date
    ).get_data()
    
    strategy_manager = create_strategies()
    df_strategy = strategy_manager.apply_all_strategies(df_data)
    
    signal_columns = strategy_manager.get_signal_columns()
    df_strategy_np = df_strategy.to_numpy()
    df_data_open = df_data.Open.to_numpy()
    
    evaluator = StrategyEvaluator(df_strategy_np, df_data_open, signal_columns)
    
    total_return, best_trades_np = evaluator.evaluate_with_thresholds(np.array(best_solution))
    
    best_trades_df = trades_to_dataframe(best_trades_np, df_data)
    
    
    # Plot results
    save_path = f'res/{StrategyConfig.target_stock}/{StrategyConfig.target_stock}_backtest_analysis.png'
    StrategyPlotter.plot_strategy_analysis(df_data, best_trades_df, save_path)

if __name__ == "__main__":
    # Load the best solution from the JSON file
    save_path = f'res/{StrategyConfig.target_stock}/{StrategyConfig.target_stock}_strategy_parameters.json'
    with open(save_path, 'r') as json_file:
        data = json.load(json_file)
        best_solution = data["best_solution"]
        
    
    backtest(best_solution)