import json
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Load the JSON data
with open('strategy_results.json', 'r') as file:
    data = json.load(file)

# Step 2: Convert df_data back to a DataFrame
df_data = pd.DataFrame(data['df_data'])

# Step 3: Extract necessary information
buy_signals = df_data[df_data['combined_positions'] > data['best_threshold']].index
sell_signals = df_data[df_data['combined_positions'] < -data['best_threshold']].index

# Step 4: Create the plot
plt.figure(figsize=(14, 7))
plt.plot(df_data['Close'], label='Close Price', alpha=0.5)
plt.plot(df_data['cumulative_returns'], label='Cumulative Returns', alpha=0.7)

# Step 5: Plot buy signals with smaller markers
plt.scatter(buy_signals, df_data.loc[buy_signals, 'cumulative_returns'], 
            label='Buy Signal', marker='^', color='green', s=50, alpha=1)

# Step 6: Plot sell signals with smaller markers
plt.scatter(sell_signals, df_data.loc[sell_signals, 'cumulative_returns'], 
            label='Sell Signal', marker='v', color='red', s=50, alpha=1)

# Step 7: Customize the plot
plt.title('Trading Signals and Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Price / Cumulative Returns')
plt.legend()
plt.grid()
plt.savefig('trading_signals_plot.png', dpi=300)  # Save the figure with higher quality
plt.show()
