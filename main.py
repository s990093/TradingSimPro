import numpy as np
import pandas as pd
import talib  # Technical Analysis library
import yfinance as yf  # For fetching stock data
from datetime import datetime
import matplotlib.pyplot as plt
import statistics

# Function to draw the stock closing price series
def drawStockSeries(df, target_stock):
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['close'], label='Close Price')
    plt.title(f'Stock Closing Price Series for {target_stock}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to draw the Moving Averages
def drawMA5_10_20Series(xpt, ypt1, ypt2, ypt3, target_stock):
    plt.figure(figsize=(14, 7))
    plt.plot(xpt, ypt1, label='MA5', color='blue')
    plt.plot(xpt, ypt2, label='MA10', color='orange')
    plt.plot(xpt, ypt3, label='MA20', color='green')
    plt.title(f'Moving Averages (5, 10, 20 Days) for {target_stock}')
    plt.xlabel('Date')
    plt.ylabel('Moving Average')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Parameters
    target_stock = "2498.TW"  # Example: '2330.TW' is TSMC
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2021, 7, 19)
    stopLoss = 0.1  # -10% stop loss

    # Fetch stock data
    df = yf.download(target_stock, start=start_date, end=end_date)
    
    if df.empty:
        print("No data fetched. Please check the stock ticker and date range.")
        return

    # Reindex and rename columns
    df = df.reindex(columns=['Open','High', 'Low', 'Close', 'Volume'])
    df.rename(columns={'Open':'open', 'High':'high', 'Low':'low', 'Close':'close',
                       'Volume':'volume' }, inplace=True)

    # Draw stock closing price series
    drawStockSeries(df, target_stock)

    # Calculate Moving Averages
    closePrices = df['close'].astype('float').values
    close_sma_5 = np.round(talib.SMA(closePrices, timeperiod=5), 2)
    close_sma_10 = np.round(talib.SMA(closePrices, timeperiod=10), 2)
    close_sma_20 = np.round(talib.SMA(closePrices, timeperiod=20), 2)

    # Add Moving Averages to DataFrame for plotting
    df['MA5'] = close_sma_5
    df['MA10'] = close_sma_10
    df['MA20'] = close_sma_20

    # Drop rows with NaN values due to MA calculation
    df.dropna(inplace=True)

    # Update closePrices and MAs after dropping NaNs
    closePrices = df['close'].astype('float').values
    close_sma_5 = df['MA5'].values
    close_sma_10 = df['MA10'].values
    close_sma_20 = df['MA20'].values
    indexDate = df.index
    xpt = indexDate
    ypt1 = close_sma_5
    ypt2 = close_sma_10
    ypt3 = close_sma_20

    # Draw Moving Averages
    drawMA5_10_20Series(xpt, ypt1, ypt2, ypt3, target_stock)

    # Initialize variables for backtesting
    flage = 0  # 0: not holding, 1: holding
    buyPrice = 0
    sellPrice = 0
    winTime = 0
    lossTime = 0
    culReturn = 0
    transList = []
    everyTranReturn = []
    tradingDetails = []
    tax = 0  # Transaction costs

    # Buy/Sell strategy backtesting
    for x in range(len(closePrices)):
        if flage == 0:
            # Check for Buy Signal: MA5 > MA10 > MA20
            if close_sma_5[x] > close_sma_10[x] and close_sma_10[x] > close_sma_20[x]:
                buyPrice = closePrices[x]
                buyDate = df.index[x].strftime('%Y-%m-%d')
                tradingDetails.append(("Buy Date", buyDate, "Buy Price", f"{buyPrice:.2f}"))
                tax += buyPrice * 0.001425  # Buy transaction fee
                flage = 1
        elif flage == 1:
            sellPrice = closePrices[x]
            # Check for Sell Signal: MA5 < MA10 < MA20
            if close_sma_5[x] < close_sma_10[x] and close_sma_10[x] < close_sma_20[x]:
                sellDate = df.index[x].strftime('%Y-%m-%d')
                tax += sellPrice * 0.001425 + sellPrice * 0.003  # Sell transaction fee
                profit = sellPrice - buyPrice - tax
                profit_percent = profit / buyPrice
                if profit > 0:
                    tradingDetails.append(("Sell Date", sellDate, "Sell Price", f"{sellPrice:.2f}",
                                           "Profit", f"{profit:.2f}", f"{profit_percent:.2%}"))
                    winTime += 1
                else:
                    tradingDetails.append(("Sell Date", sellDate, "Sell Price", f"{sellPrice:.2f}",
                                           "Loss", f"{profit:.2f}", f"{profit_percent:.2%}"))
                    lossTime += 1
                flage = 0
                everyTranReturn.append(profit)
                culReturn += profit
                transList.append(culReturn)
                tax = 0  # Reset tax after transaction

            # Check for Stop Loss
            elif (sellPrice - buyPrice - tax) / buyPrice < -stopLoss:
                sellDate = df.index[x].strftime('%Y-%m-%d')
                tax += sellPrice * 0.001425 + sellPrice * 0.003  # Sell transaction fee
                profit = sellPrice - buyPrice - tax
                profit_percent = profit / buyPrice
                tradingDetails.append(("Sell Date (Stop Loss)", sellDate, "Sell Price", f"{sellPrice:.2f}",
                                       "Loss", f"{profit:.2f}", f"{profit_percent:.2%}"))
                lossTime += 1
                flage = 0
                everyTranReturn.append(profit)
                culReturn += profit
                transList.append(culReturn)
                tax = 0  # Reset tax after transaction

    # Print trading details
    print("交易詳情 (Trading Details):")
    for detail in tradingDetails:
        print(detail)

    # Summary of backtesting
    total_trades = winTime + lossTime
    total_return = culReturn
    avg_return = statistics.mean(everyTranReturn) if everyTranReturn else 0
    max_return = max(everyTranReturn) if everyTranReturn else 0
    min_return = min(everyTranReturn) if everyTranReturn else 0
    win_rate = (winTime / total_trades) * 100 if total_trades > 0 else 0
    loss_rate = (lossTime / total_trades) * 100 if total_trades > 0 else 0

    print("\n回測結果 (Backtesting Results):")
    print(f"總交易次數 (Total Trades): {total_trades}")
    print(f"贏利次數 (Winning Trades): {winTime}")
    print(f"虧損次數 (Losing Trades): {lossTime}")
    print(f"贏率 (Win Rate): {win_rate:.2f}%")
    print(f"虧率 (Loss Rate): {loss_rate:.2f}%")
    print(f"總累計報酬 (Total Cumulative Return): {total_return:.2f}")
    print(f"平均每筆交易報酬 (Average Return per Trade): {avg_return:.2f}")
    print(f"最大單筆報酬 (Max Return): {max_return:.2f}")
    print(f"最小單筆報酬 (Min Return): {min_return:.2f}")

    # Plot Cumulative Returns
    plt.figure(figsize=(14, 7))
    # plt.plot(df.index[:len(transList)], transList, label='Cumulative Return')
    plt.title('Cumulative Return Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
