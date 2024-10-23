import json
import yfinance as yf
from datetime import datetime
import click
from utility.qlearning import abc_algorithm_qlearning
from utility.abc_algorithm import abc_algorithm
from utility.reward_func import dqn_algorithm
from utility.func import adjust_weights
from utility.stock_plotter import plot_trades
from utility.calculate_returns import calculate_trading_signals
from strategies import *
from ENV import Environment
from utility.print import print_df_strategy, display_results
import traceback


@click.command()
@click.option('--algorithm', type=click.Choice(['abc', 'dqn', 'q'], case_sensitive=False), default='abc', help='Choose the algorithm to run.')
def main(algorithm):
    """
    Main function that applies the selected algorithm (ABC or DQN) to the stock data.
    """
    
    print(f"Selected algorithm: {algorithm.upper()}")

    # Download stock data for the target stock and benchmark (e.g., S&P 500)
    df_data = yf.download(Environment.target_stock, start=Environment.start_date, end=Environment.end_date)
    
    df_data = df_data.astype('float')

    initial_price = df_data.iloc[0]['Open']
    benchmark_df = yf.download('^GSPC', start=Environment.start_date, end=Environment.end_date)

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
        best_bee, best_fitness, best_trades_df = abc_algorithm(
            df_strategy,
            df_data,
            Environment.CS,
            Environment.MCN,
            Environment.limit,
            Environment.weights_range,
            Environment.x_range,
            signal_columns
        )
    elif algorithm == 'q':            
     best_bee, best_fitness, best_trades_df = abc_algorithm_qlearning(
            df_strategy,
            df_data,
            Environment.CS,
            Environment.MCN,
            Environment.limit,
            Environment.weights_range,
            Environment.x_range,
            signal_columns
        )
    elif algorithm == 'dqn':
        print("Running DQN Algorithm...")
        best_bee, best_fitness, best_trades_df = dqn_algorithm(
            df_strategy,
            df_data,
            Environment.MCN,
            Environment.weights_range,
            Environment.x_range,
            signal_columns
        )
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
    plot_trades(df_data, best_trades_df, best_fitness, 1000, benchmark_df)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
