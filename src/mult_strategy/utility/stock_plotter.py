import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf


def plot_trades(algorithm, df, trades_df, total_return, initial_capital, market_df):
    
    plt.figure(figsize=(10, 7))

    # Forward fill to handle missing data
    market_df['Close'] = market_df['Close'].ffill()

    # Calculate S&P 500 Position Value
    market_df['Position Value'] = (market_df['Close'] / market_df['Close'].iloc[0]) * initial_capital

    # Plot close price and buy/sell signals
    plot_price_and_signals(df, trades_df, total_return)

    # Plot cumulative profit
    plot_cumulative_profit(trades_df, initial_capital, market_df)

    # Plot all trade actions with cumulative profit
    plot_trade_actions(trades_df, market_df)

    plt.tight_layout()
    plt.savefig(f"res/{algorithm}_trading_results_vs_market.png", format='png')
    plt.show()  

def plot_price_and_signals(df, trades_df, total_return):
    plt.subplot(3, 1, 1)
    plt.plot(df['Close'], label='Close Price', alpha=0.5)

    # Buy and Sell signals
    buy_signals = trades_df[trades_df['Action'] == 'Buy']
    sell_signals = trades_df[trades_df['Action'] == 'Sell']

    plt.scatter(buy_signals['Date'], buy_signals['Price'], marker='^', color='g', label='Buy Signal', s=20, zorder=5)
    plt.scatter(sell_signals['Date'], sell_signals['Price'], marker='v', color='r', label='Sell Signal', s=20, zorder=5)

    plt.title(f'Total Return: {total_return:.2f}', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.legend()
    plt.grid()

def plot_cumulative_profit(trades_df, initial_capital, market_df):
    # Calculate each trade's profit and cumulative profit
    trades_df['Profit'] = trades_df['Price'].diff().fillna(0)  # Profit from trades
    trades_df['Cumulative Profit'] = trades_df['Profit'].cumsum() + initial_capital  # Cumulative profit

    plt.subplot(3, 1, 2)
    plt.plot(market_df.index, market_df['Position Value'], label='S&P 500 Position Value', color='blue', linestyle='--')
    plt.plot(trades_df['Date'], trades_df['Cumulative Profit'], label='Strategy Cumulative Profit', color='orange')
    plt.axhline(initial_capital, color='black', lw=1, ls='--', label='Initial Capital')

    plt.title('Comparison with S&P 500', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.legend()
    plt.grid()
    
    
def plot_trade_actions(trades_df ,market_df):
    plt.subplot(3, 1, 3)
    trades_df['Profit'] = trades_df['Price'].diff().fillna(0) 
    trades_df['Cumulative Profit'] = trades_df['Profit'].cumsum() 

    # Create a list of dates and corresponding cumulative profit values
    trade_dates = trades_df['Date']
    cumulative_profit = trades_df['Cumulative Profit']

    # Create a new DataFrame for trade actions
    action_values = pd.Series(index=trade_dates, data=cumulative_profit)

    # Plot cumulative profit with lines connecting the points
    plt.plot(trades_df['Date'], trades_df['Cumulative Profit'], label='Strategy Cumulative Profit', color='blue')
    plt.plot(action_values.index, action_values, label='Cumulative Profit', color='orange', linewidth=2, alpha=0.7)

    # Set to track unique labels for the legend
    plotted_labels = set()

    # Mark buy, sell, and stop actions
    for _, trade in trades_df.iterrows():
        plt.scatter(trade['Date'], trade['Cumulative Profit'], 
                    marker='^' if trade['Action'] == 'Buy' else 'v' if trade['Action'] == 'Sell' else 'x', 
                    color='g' if trade['Action'] == 'Buy' else 'r' if trade['Action'] == 'Sell' else 'orange', 
                    s=13, zorder=5)

        # Track unique actions for legend
        plotted_labels.add(trade['Action'])

    # Add unique legend entries
    for label in plotted_labels:
        plt.scatter([], [], marker='^' if label == 'Buy' else 'v' if label == 'Sell' else 'x',
                    color='g' if label == 'Buy' else 'r' if label == 'Sell' else 'orange',
                    label=label)

    plt.title('Cumulative Profit with Trade Actions', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Cumulative Profit', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid()