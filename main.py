import json
import yfinance as yf
from datetime import datetime
from utility.abc_algorithm import abc_algorithm
from utility.reward_func import  dqn_algorithm
from utility.func import adjust_weights
from utility.stock_plotter import plot_trades
from utility.calculate_returns import calculate_trading_signals
from strategies import *
from ENV import Environment
from utility.print import print_df_strategy, display_results
import traceback

def main():
    
    df_data = yf.download(Environment.target_stock, start=Environment.start_date, end=Environment.end_date)
    
    initial_price = df_data.iloc[0]['Open'] 

    benchmark_df = yf.download('^GSPC', start=Environment.start_date, end=Environment.end_date) 

    strategy_manager = create_strategies()
    
    # Apply all strategies to the data
    df_strategy = strategy_manager.apply_all_strategies(df_data)
    
    signal_columns = strategy_manager.get_signal_columns()
    
    
    Environment.display_config(signal_columns)

    
    # print_df_strategy(df_strategy.columns)
    start_time = datetime.now()


#    Execute ABC algorithm
    # best_bee, best_fitness, best_trades_df = abc_algorithm(
    #     df_strategy,
    #     df_data, 
    #     Environment.CS, 
    #     Environment.MCN,
    #     Environment.limit, 
    #     Environment.weights_range, 
    #     Environment.x_range, 
    #     signal_columns
    #     # Environment.MAX_PROCESS_WORKERS,
    #     # Environment.MAX_THREAD_WORKERS,
    # )    
    
    best_bee, best_fitness, best_trades_df =  dqn_algorithm(df_strategy, df_data, 
                                                            Environment.MCN,
                                                            Environment.weights_range,
                                                            Environment.x_range,
                                                            signal_columns)
    
    end_time = datetime.now()
    duration = end_time - start_time

    profit_ratio = (best_fitness + initial_price / best_fitness) 
        

    display_results(best_bee, best_fitness, duration, profit_ratio, best_trades_df)
        
    
    # results = {
    #     "target_stock": Environment.target_stock,  # Ensure this variable is defined in your code
    #     "best_weights": list(best_bee['weights']),  # Save as a list
    #     "buy_threshold": float(best_bee['buy_threshold']),  # Ensure it's a float for JSON serialization
    #     "sell_threshold": float(best_bee['sell_threshold']),  # Ensure it's a float for JSON serialization
    #     "best_fitness": best_fitness,  # Ensure this variable is defined
    # }
    
    # # Save results to JSON file
    # with open('strategy_results.json', 'w') as json_file:
    #     json.dump(results, json_file, indent=4)
    
    
    plot_trades(df_data, best_trades_df, best_fitness, 1000, benchmark_df)

    

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc() 