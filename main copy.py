import yfinance as yf
from datetime import datetime
import click
from utility.helper.stock_data_cache import StockDataCache
from utility.qlearning import ABCQlearningAlgorithmManager
from utility.abc_algorithm import ABCAlgorithmManager
# from utility.reward_func import dqn_algorithm
from utility.stock_plotter import plot_trades
from strategies import *
from ENV import Environment
from utility.print import display_results
import traceback
import os



@click.command()
@click.option('--algorithm', type=click.Choice(['abc', 'dqn', 'q'], case_sensitive=False), default='abc', help='Choose the algorithm to run.')
def main(algorithm):
    """
    Main function that applies the selected algorithm (ABC or DQN) to the stock data.
    """
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    print(f"Selected algorithm: {algorithm.upper()}")
    
    df_data = StockDataCache(Environment.target_stock, Environment.start_date, Environment.end_date).get_data()
    benchmark_df = StockDataCache('^GSPC', Environment.start_date, Environment.end_date).get_data()


    initial_price = df_data.iloc[0]['Open']

    # Create and apply all strategies to the data
    strategy_manager = create_strategies()
    df_strategy = strategy_manager.apply_all_strategies(df_data)
    signal_columns = strategy_manager.get_signal_columns()

    # Display configuration
    Environment.display_config(signal_columns)

    # Start timer
    start_time = datetime.now()

    # Execute selected algorithm
    if algorithm == 'abc':
        print("Running ABC Algorithm...")
        
        abc = ABCAlgorithmManager(
            df_strategy,
            df_data,
            Environment.CS,
            Environment.MCN,
            Environment.limit,
            Environment.weights_range,
            Environment.x_range,
            signal_columns,
            Environment.restart_threshold,
            Environment.max_restarts
        )
        
        abc.run_algorithm()
        abc.store_results_pickle()
        best_bee, best_fitness, best_trades_df = abc.get_res()
        
    elif algorithm == 'q':            
        abc_q = ABCQlearningAlgorithmManager(
                df_strategy,
                df_data,
                Environment.CS,
                Environment.MCN,
                Environment.limit,
                Environment.weights_range,
                Environment.x_range,
                signal_columns
            )
        abc_q.run_algorithm()
        
        abc_q.store_results_pickle()
        
        best_bee, best_fitness, best_trades_df = abc_q.get_res()
        
    # elif algorithm == 'dqn':
    #     print("Running DQN Algorithm...")
    #     best_bee, best_fitness, best_trades_df = dqn_algorithm(
    #         df_strategy,
    #         df_data,
    #         Environment.MCN,
    #         Environment.weights_range,
    #         Environment.x_range,
    #         signal_columns
    #     )
    else:
        raise ValueError("Unknown algorithm selected.")

    # End timer
    end_time = datetime.now()
    duration = end_time - start_time

    # Calculate profit ratio
    profit_ratio = (best_fitness + initial_price / best_fitness)

    # Display results
    display_results(algorithm, best_bee, best_fitness, duration, profit_ratio, best_trades_df)

    # Plot the trades
    plot_trades(algorithm, df_data, best_trades_df, best_fitness, 1000, benchmark_df)
    
    

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
