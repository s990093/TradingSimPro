import numba
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from random import choice
from rich import print
from rich.table import Table
from rich.console import Console

from utility.print import display_results
from utility.helper.tool import trades_to_dataframe
from utility.stock_plotter import plot_trades
from utility.calculate_returns_jit import fitness
from utility.helper.stock_data_cache import StockDataCache
from strategies import create_strategies
from ENV import Environment


best_trades_df_read = pd.read_pickle('best_trades.pkl')
df_data = StockDataCache(Environment.target_stock, Environment.start_date, Environment.end_date).get_data()

print(best_trades_df_read)

def calculate_returns(buy_trades, sell_trades, df, account_balance=10000, risk_per_trade=0.01, stop_loss_pct=0.02, take_profit_pct=0.05):
    total_return = 0.0
    trades = []
    holding_times = []
    cumulative_returns = [account_balance]
    dates = [df.index[0]]

    current_balance = account_balance

    for buy_index, buy_row in buy_trades.iterrows():
        buy_price = buy_row['Price']
        buy_date = buy_row['Date']
        
        # 計算止損價格和獲利價格
        stop_loss_price = buy_price * (1 - stop_loss_pct)
        take_profit_price = buy_price * (1 + take_profit_pct)
        
        # 計算每次交易的風險和持倉數量
        max_risk = current_balance * risk_per_trade
        risk_per_share = abs(buy_price - stop_loss_price)
        
        if risk_per_share > 0:  # 確保風險計算合理
            position_size = int(max_risk / risk_per_share)  # 計算可以購買的股票數量
            position_size = min(position_size, current_balance // buy_price)  # 確保不超過可用餘額

            for sell_index, sell_row in sell_trades.iterrows():
                sell_price = sell_row['Price']
                sell_date = sell_row['Date']

                if sell_date > buy_date:
                    df_slice = df[(df.index > buy_date) & (df.index <= sell_date)]

                    for day_date, day_row in df_slice.iterrows():
                        day_price = day_row['Close']
                        
                        # 檢查是否觸發止損
                        if day_price <= stop_loss_price:
                            profit = (stop_loss_price - buy_price) * position_size
                            holding_time = (day_date - buy_date).days
                            holding_times.append(holding_time)
                            current_balance += profit
                            total_return += profit
                            cumulative_returns.append(current_balance)
                            dates.append(day_date)
                            trades.append({
                                'Buy Date': buy_date,
                                'Sell Date': day_date,
                                'Buy Price': buy_price,
                                'Sell Price': stop_loss_price,
                                'Profit': profit,
                                'Position Size': position_size
                            })
                            break
                        # 檢查是否觸發獲利
                        elif day_price >= take_profit_price:
                            profit = (take_profit_price - buy_price) * position_size
                            holding_time = (day_date - buy_date).days
                            holding_times.append(holding_time)
                            current_balance += profit
                            total_return += profit
                            cumulative_returns.append(current_balance)
                            dates.append(day_date)
                            trades.append({
                                'Buy Date': buy_date,
                                'Sell Date': day_date,
                                'Buy Price': buy_price,
                                'Sell Price': take_profit_price,
                                'Profit': profit,
                                'Position Size': position_size
                            })
                            break
                    break

    avg_holding_time = sum(holding_times) / len(holding_times) if holding_times else 0
    max_holding_time = max(holding_times) if holding_times else 0

    return total_return, trades, avg_holding_time, max_holding_time, current_balance, dates, cumulative_returns


buy_trades = best_trades_df_read[best_trades_df_read['Action'] == 'Buy']
stop_trades = best_trades_df_read[best_trades_df_read['Action'] == 'Stop']
sell_trades = best_trades_df_read[best_trades_df_read['Action'] == 'Sell']  # 如果有 'Sell' 行为



# Display results with rich
def display_results(total_return, trades, avg_holding_time, max_holding_time, initial_balance=10000):
    console = Console()
    percentage_return = (total_return / initial_balance) * 100

    # Summary Table
   # Display results with rich
    console.print(f"[bold]总收益:[/bold] ${total_return:.2f}")
    console.print(f"[bold]收益百分比:[/bold] {percentage_return:.2f}%")
    console.print(f"[bold]初始余额:[/bold] ${initial_balance:.2f}")
    # Trade Details Table
    table = Table(title="Trade Records")
    table.add_column("Buy Date", justify="center")
    table.add_column("Sell Date", justify="center")
    table.add_column("Buy Price", justify="right")
    table.add_column("Sell Price", justify="right")
    table.add_column("Profit", justify="right")
    table.add_column("Position Size", justify="right")

    for trade in trades:
        table.add_row(
            trade['Buy Date'].strftime("%Y-%m-%d"),
            trade['Sell Date'].strftime("%Y-%m-%d"),
            f"{trade['Buy Price']:.2f}",
            f"{trade['Sell Price']:.2f}",
            f"{trade['Profit']:.2f}",
            f"{trade['Position Size']}",
        )

    console.print(table)

# Example usage
total_return, trades, avg_holding_time, max_holding_time, current_balance, dates, cumulative_returns = calculate_returns(buy_trades, sell_trades, df_data)
display_results(total_return, trades, avg_holding_time, max_holding_time)