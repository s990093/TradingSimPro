from matplotlib import pyplot as plt
import yfinance as yf
from datetime import datetime
from stock_plotter import drawStockSeries
from strategies.strategy import *
from rich.console import Console
from rich.table import Table

def print_trade_dates(buy_signals, sell_signals, df):
    console = Console()
    table = Table(title="Buy and Sell Dates")

    table.add_column("Date", justify="center")
    table.add_column("Action", justify="center")
    table.add_column("Price", justify="center")

    # 打印買入信號
    for i, signal in enumerate(buy_signals):
        if signal > 0:
            date = df.index[i].strftime('%Y-%m-%d')
            price = df['Close'][i]
            table.add_row(date, "Buy", f"{price:.2f}")

    # 打印賣出信號
    for i, signal in enumerate(sell_signals):
        if signal > 0:
            date = df.index[i].strftime('%Y-%m-%d')
            price = df['Close'][i]
            table.add_row(date, "Sell", f"{price:.2f}")

    console.print(table)

def main():
    # 下載股票數據
    target_stock = "2498.TW"
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2021, 7, 19)

    df = yf.download(target_stock, start=start_date, end=end_date)


    
    # 定義各個策略
    moving_average_strategy = MovingAverageStrategy()
    rsi_strategy = RSIStrategy()
    macd_strategy = MACDStrategy()
    bollinger_bands_strategy = BollingerBandsStrategy()
    momentum_strategy = MomentumStrategy()
    stochastic_strategy = StochasticOscillatorStrategy()


    # 將策略添加到策略管理器
    strategy_manager = StrategyManager([
        moving_average_strategy, 
        rsi_strategy, 
        macd_strategy, 
        bollinger_bands_strategy,
        momentum_strategy,
        stochastic_strategy
    ])
    
    
    # 將所有策略應用到數據
    df = strategy_manager.apply_all_strategies(df)

    # 組合策略信號：這裡取所有策略信號的平均值

    # 組合策略信號：這裡取所有策略信號的平均值
    df['combined_signal'] = (
        df['ma_signal'] + df['rsi_signal'] + df['macd_signal'] + df['bb_signal'] +
        df['momentum_signal'] + df['stochastic_signal']
    ) / 6
    
    df['combined_positions'] = df['combined_signal'].diff()
    
    buy_signals =   df['combined_positions'].apply(lambda x: 1 if x > 0 else 0)  # 信號大於 0 就買入
    sell_signals =  df['combined_positions'].apply(lambda x: 1 if x < 0 else 0)  # 信號大於 0 就買入 
    print_trade_dates(buy_signals, sell_signals, df)




    # 畫圖
    plt.figure(figsize=(12, 6))

    # 繪製收盤價
    plt.plot(df.index, df['Close'], label='Close Price', color='blue')

    # 繪製買入信號
    plt.plot(df.index[buy_signals > 0], df['Close'][buy_signals > 0], '^', markersize=10, color='green', label='Buy Signal', alpha=1)

    # 繪製賣出信號
    plt.plot(df.index[sell_signals > 0], df['Close'][sell_signals > 0], 'v', markersize=10, color='red', label='Sell Signal', alpha=1)

    # 設定標題和標籤
    plt.title(f'{target_stock} Price and Trade Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()
    
if __name__ == "__main__":
    main()
