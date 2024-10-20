import json
import yfinance as yf
from datetime import datetime
from utility.func import adjust_weights
from utility.stock_plotter import plot_trades
from utility.calculate_returns import calculate_trading_signals
from utility.abc_algorithm import abc_algorithm
from strategies import *
from ENV import Environment
from utility.print import print_df_strategy, display_results
import traceback

def main():
    Environment.display_config()
    
    df_data = yf.download(Environment.target_stock, start=Environment.start_date, end=Environment.end_date)
    
    initial_price = df_data.iloc[0]['Open'] 

    benchmark_df = yf.download('^GSPC', start=Environment.start_date, end=Environment.end_date)  # 调整日期范围

    strategy_manager = create_strategies()
    
    # Apply all strategies to the data
    df_strategy = strategy_manager.apply_all_strategies(df_data)
    
    # print_df_strategy(df_strategy.columns)
    start_time = datetime.now()


   # Execute ABC algorithm
    best_bee, best_fitness, best_trades_df = abc_algorithm(
        df_strategy, df_data, 
        Environment.population_size, 
        Environment.max_iter, 
        Environment.weights_range, 
        Environment.x_range, 
        # Environment.MAX_PROCESS_WORKERS,
        # Environment.MAX_THREAD_WORKERS,
    )    
    
    end_time = datetime.now()
    duration = end_time - start_time

    profit_ratio = (best_fitness + initial_price / best_fitness) 
    

    display_results(best_bee, best_fitness, duration, profit_ratio)
    
    # adjust_weights()
    
    # total_return, trades_df = calculate_trading_signals(df_data, best_bee['weights'], best_bee['x'], Environment.signal_columns, df_data)
    
    # results = {
    #     "target_stock": Environment.target_stock,  # Ensure this variable is defined in your code
    #     "best_weights": list(best_bee['weights']),  # Save as a list
    #     "best_threshold": float(best_bee['x']),  # Ensure it's a float for JSON serialization
    #     "best_fitness": best_fitness,  # Ensure this variable is defined
    #     # Convert DataFrame to a dictionary for JSON serialization
    #     "df_data": df_data.to_dict(orient='records'),  # Converts DataFrame to a list of records
    # }
    
    # # Save results to JSON file
    # with open('strategy_results.json', 'w') as json_file:
    #     json.dump(results, json_file, indent=4)


    # plot_trading_signals(df_data, best_bee['weights'], best_bee['x'])
    # plot_backtest_performance(df_res, benchmark_df)
    
    plot_trades(df_data, best_trades_df, best_fitness, 1000, benchmark_df)

    

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc() 