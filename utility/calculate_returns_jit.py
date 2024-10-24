import numpy as np
import pandas as pd
from numba import njit
from typing import Tuple

# Numba optimized function for calculating returns
@njit
def calculate_returns(buy_signals: np.ndarray, sell_signals: np.ndarray, close_prices: np.ndarray, stop_loss_pct=0.02, take_profit_pct=0.05) -> Tuple[float, np.ndarray]:
    total_return = 0.0
    holding = False
    buy_price = 0.0
    trade_count = 0  # Keep track of trades

    # Allocate space for trades
    trades = np.zeros((len(close_prices), 4))  # Adjust size as needed

    for i in range(len(close_prices)):
        if buy_signals[i] == 1 and not holding:
            buy_price = close_prices[i]
            holding = True
            trades[trade_count, 0] = i  # Trade index
            trades[trade_count, 1] = 1  # Action (1 for Buy)
            trades[trade_count, 2] = buy_price  # Price
            trade_count += 1

        elif holding:
            current_price = close_prices[i]

            # Check for stop-loss
            if current_price <= buy_price * (1 - stop_loss_pct):
                sell_price = current_price
                total_return += (sell_price - buy_price)
                holding = False
                trades[trade_count, 0] = i  # Trade index
                trades[trade_count, 1] = 0  # Action (0 for Stop)
                trades[trade_count, 2] = sell_price  # Price
                trades[trade_count, 3] = current_price  # Triggered Price
                trade_count += 1

            # Check for take-profit or normal sell signal
            elif current_price >= buy_price * (1 + take_profit_pct) or sell_signals[i] == 1:
                sell_price = current_price
                total_return += (sell_price - buy_price)
                holding = False
                trades[trade_count, 0] = i  # Trade index
                trades[trade_count, 1] = 2  # Action (2 for Sell)
                trades[trade_count, 2] = sell_price  # Price
                trades[trade_count, 3] = -1  # Placeholder for triggered price
                trade_count += 1

    # Trim trades array to actual number of trades
    trades = trades[:trade_count]

    return total_return, trades

# Numba optimized function for calculating trading signals
@njit
def calculate_trading_signals(df_strategy_signals: np.ndarray, weights: np.ndarray, buy_threshold: float, sell_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    if len(weights) != df_strategy_signals.shape[1]:
        raise ValueError("The number of weights must match the number of signals.")

    # Create the weighted signals array dynamically
    weighted_signals = np.zeros(df_strategy_signals.shape[0])
    for i in range(len(weights)):
        weighted_signals += weights[i] * df_strategy_signals[:, i]

    # Manually compute the sum of weights
    weight_sum = np.sum(weights)

    # Handle division by zero case
    combined_signal = weighted_signals / weight_sum if weight_sum != 0 else np.zeros_like(weighted_signals)

    # Manual implementation of np.diff with prepend=0
    combined_positions = np.empty_like(combined_signal)
    combined_positions[0] = 0  # Equivalent to prepend=0
    for i in range(1, len(combined_signal)):
        combined_positions[i] = combined_signal[i] - combined_signal[i - 1]

    buy_signals = (combined_positions > buy_threshold).astype(np.int32)
    sell_signals = (combined_positions < -sell_threshold).astype(np.int32)

    return buy_signals, sell_signals

# Main fitness function that orchestrates data handling and computation
def fitness(weights: np.ndarray, buy_threshold: float, sell_threshold: float, df_strategy: pd.DataFrame, df_data: pd.DataFrame, signal_columns: list) -> Tuple[float, np.ndarray]:
    # Ensure inputs are in the correct format
    if not isinstance(weights, np.ndarray):
        weights = np.array(weights, dtype=np.float64)

    # Extract signals from DataFrame and convert to NumPy array
    df_strategy_signals = df_strategy[signal_columns].values.astype(np.float64)  # Convert to float64 for Numba
    close_prices = df_data['Close'].values.astype(np.float64)  # Ensure prices are float64

    # Call Numba optimized functions
    buy_signals, sell_signals = calculate_trading_signals(df_strategy_signals, weights, buy_threshold, sell_threshold)
    total_return, trades_np = calculate_returns(buy_signals, sell_signals, close_prices)

    return total_return, trades_np

# Example usage (replace with actual DataFrame and parameters)
# df_strategy = pd.DataFrame(...)  # Your strategy DataFrame
# df_data = pd.DataFrame(...)       # Your price DataFrame
# weights = np.array([...])         # Your weights
# buy_threshold = 0.5               # Your buy threshold
# sell_threshold = 0.5              # Your sell threshold
# total_return, trades_np = fitness(weights, buy_threshold, sell_threshold, df_strategy, df_data, signal_columns)
