import numpy as np
import json
from rich import print as rprint
from alog.pso import pso
from rich.progress import Progress
from multiprocessing import Process, Manager
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import StrategyConfig
from mult_strategy.utility.helper.tool import trades_to_dataframe
from mult_strategy.utility.calculate_returns_jit import fitness
from utils.stock_data_cache import StockDataCache
from mult_strategy.strategies import create_strategies
from .model.rf_trainer import RFTrainer
from .evaluator.strategy_evaluator import StrategyEvaluator
from .visualization.plotter import StrategyPlotter

def optimize_process(evaluator, lower_bounds, upper_bounds):
    current_solution, current_fitness = pso(
        evaluator.evaluate, 
        lower_bounds, 
        upper_bounds, 
        swarmsize=200,
        maxiter=250,
    )
    return current_solution, current_fitness

def main():
    
    best_solution = None
    best_fitness = float('inf')

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
    
    # PSO optimization setup
    D = len(signal_columns) 


    # Strategy evaluation
    evaluator = StrategyEvaluator(df_strategy_np, df_data_open, signal_columns)
    
    max_retries = 150

    with Progress() as progress:
        task = progress.add_task("[green]Optimizing...", total=max_retries)

        with ProcessPoolExecutor(max_workers=8) as executor:  # Set max_workers as needed
            futures = []
            for _ in range(max_retries):
                lower_bounds = [-1.0] * D
                upper_bounds = [1.0] * D
                futures.append(executor.submit(optimize_process, evaluator, lower_bounds, upper_bounds))

            for future in as_completed(futures):
                current_solution, current_fitness = future.result()
                if current_fitness < best_fitness:
                    best_solution = current_solution
                    best_fitness = current_fitness

                    # Calculate best trades and total return
                    total_return, best_trades_np = fitness(
                        best_solution, 1.0, 1.0, 
                        df_strategy_np, df_data_open, 
                        signal_columns
                    )

                progress.update(task, advance=1)

    # Calculate best trades and total return
    total_return, best_trades_np = fitness(
        best_solution, 1.0, 1.0, 
        df_strategy_np, df_data_open, 
        signal_columns
    )

    best_trades_df = trades_to_dataframe(best_trades_np, df_data)
    
    # Print results
    rprint(f"[bold green]Best Fitness Value:[/bold green] {best_fitness}")
    rprint(f"[bold green]Best Weights:[/bold green] {best_solution}")
    
    # Plot results
    save_path = f'res/{StrategyConfig.target_stock}/{StrategyConfig.target_stock}_strategy_analysis.png'
    StrategyPlotter.plot_strategy_analysis(df_data, best_trades_df, save_path)

    # Save parameters
    data = {
        "best_solution": best_solution.tolist(),
        "signal_columns": signal_columns,
        "total_return": total_return   
    }
    
    save_path = f'res/{StrategyConfig.target_stock}/{StrategyConfig.target_stock}_strategy_parameters.json'
    with open(save_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

if __name__ == "__main__":
    main()