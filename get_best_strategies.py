import yfinance as yf
from utility.helper.stock_data_cache import StockDataCache
from utility.print import display_best_strategies
from utility.best_strategies import select_best_strategies
from strategies import *
from ENV import Environment
import traceback
from rich.traceback import install

install()

def main():
    
    df_data = StockDataCache(Environment.target_stock, Environment.start_date, Environment.end_date).get_data()

    
    initial_price = df_data.iloc[0]['Open'] 



    strategy_manager = create_strategies()

    
    # Apply all strategies to the data
    df_strategy = strategy_manager.apply_all_strategies(df_data)
    
    signal_columns = strategy_manager.get_signal_columns()
    
    Environment.display_config(signal_columns)
    
    best_strategies, best_performance = select_best_strategies(df_strategy, df_data, signal_columns)
    
    display_best_strategies(best_strategies, best_performance, len(signal_columns), len(best_strategies))
        

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc() 