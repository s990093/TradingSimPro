from matplotlib import pyplot as plt
import numpy as np
import numpy as np
import pandas as pd
from rich.progress import Progress
import threading

from TradingSimPro.config.ENV import Environment
from utility.func import adjust_weights

def calculate_returns(buy_signals, sell_signals, df, stop_loss_pct=0.02, take_profit_pct=0.05):
    total_return = 0.0
    holding = False
    buy_price = 0.0
    trades = []

    for i in range(len(df)):
        if buy_signals.iloc[i] == 1 and not holding:
            buy_price = df['Close'].iloc[i]
            holding = True
            trades.append({'Date': df.index[i], 'Action': 'Buy', 'Price': buy_price})

        elif holding:
            current_price = df['Close'].iloc[i]

            # Check for stop-loss
            if current_price <= buy_price * (1 - stop_loss_pct):
                # Trigger stop-loss sell
                sell_price = current_price
                total_return += (sell_price - buy_price)
                holding = False
                trades.append({
                    'Date': df.index[i],
                    'Action': 'Stop',
                    'Price': sell_price,
                    'Triggered Price': current_price,
                })

            # Check for take-profit or normal sell signal
            elif current_price >= buy_price * (1 + take_profit_pct) or sell_signals.iloc[i] == 1:
                # Trigger sell (normal or take profit)
                sell_price = current_price
                total_return += (sell_price - buy_price)
                holding = False
                trades.append({
                    'Date': df.index[i],
                    'Action': 'Sell',
                    'Price': sell_price,
                })

    # Uncomment the following block if you want to handle end-of-data sell
    # if holding:  
    #     sell_price = df['Close'].iloc[-1]
    #     total_return += (sell_price - buy_price)
    #     trades.append({'Date': df.index[-1], 'Action': 'Sell', 'Price': sell_price})

    # Create a DataFrame for the trades
    trades_df = pd.DataFrame(trades)

    return total_return, trades_df




def fitness(weights: np.ndarray, buy_threshold: float, sell_threshold: float, df_strategy: pd.DataFrame, df_data: pd.DataFrame, signal_columns: list):
    if len(weights) != len(signal_columns):
            raise ValueError("The number of weights must match the number of signals.")
    
    weighted_signals = np.array([
        weights[i] * df_strategy[signal] for i, signal in enumerate(signal_columns) if signal in df_strategy.columns
    ])
        
    df_strategy['combined_signal'] = np.sum(weighted_signals, axis=0) / np.sum(weights)

    df_strategy['combined_positions'] = df_strategy['combined_signal'].diff()
    buy_signals = (df_strategy['combined_positions'] > buy_threshold).astype(int)
    sell_signals = (df_strategy['combined_positions'] < -sell_threshold).astype(int)

    total_return, trades_df = calculate_returns(buy_signals, sell_signals, df_data)

    return total_return, trades_df






