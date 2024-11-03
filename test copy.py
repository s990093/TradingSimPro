# Standard library imports
import random
from random import choice

# Third-party imports
import numba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
from rich import print as rprint
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from numba import njit, prange

# Local imports
from utility.print import display_results
from utility.helper.tool import trades_to_dataframe
from utility.stock_plotter import plot_trades
from utility.calculate_returns_jit import fitness
from utility.helper.stock_data_cache import StockDataCache
from strategies import create_strategies
from ENV import Environment

# Download stock data for target stock and benchmark (e.g., S&P 500)
df_data = StockDataCache(Environment.target_stock, Environment.start_date, Environment.end_date).get_data()

benchmark_df = StockDataCache('^GSPC', Environment.start_date, Environment.end_date).get_data()



# Initialize parameters
initial_price = df_data.iloc[0]['Open']
strategy_manager = create_strategies()
df_strategy = strategy_manager.apply_all_strategies(df_data)
signal_columns = strategy_manager.get_signal_columns()
df_strategy_np = df_strategy[signal_columns].values.astype(np.float64)  # Extract only necessary columns
df_data_open = df_data['Close'].values.astype(np.float64)          # Example for 'Open' column

# df_strategy_signals = df_strategy[signal_columns].values.astype(np.float64)
# close_prices = df_data['Close'].values.astype(np.float64)  



# ABC algorithm parameters
D = len(signal_columns)       # Dimension of weights (signal columns)
N =  4000                        # Number of bees
max_iter = 70000              # Maximum iterations
# max_iter = 10              # Maximum iterations
limit = 400                    # Scout bee limit
weights_range = (-1, 1)       # Range for weights

buy_threshold, sell_threshold = 1.0, 1.0



# Initialize bee population and calculate initial fitness
fs = np.random.uniform(*weights_range, (N, D))
fitness_list = [fitness(fs[i], buy_threshold, sell_threshold, df_strategy_np, df_data_open, signal_columns)[0] for i in range(N)]
trial = np.zeros(N, dtype=int)



# Track best results
best_fitness = max(fitness_list)
best_solution = fs[np.argmax(fitness_list)].copy()
best_trades_np = None

@njit
def employed_bee_phase(N, D, fs, fitness_list, trial, buy_threshold, sell_threshold, df_strategy_np, df_data_open, signal_columns, weights_range):
    for i in range(N):
        fs_row = fs[i].copy()
        var_idx = np.random.randint(0, D - 1)
        
        # Select a random partner different from the current index
        partner_idx = np.random.choice(np.array([idx for idx in range(N) if idx != i]))

        # Update based on partner bee
        phi = np.random.uniform(-1, 1)
        fs_row[var_idx] += phi * (fs_row[var_idx] - fs[partner_idx, var_idx])
        fs_row[var_idx] = max(weights_range[0], min(fs_row[var_idx], weights_range[1]))

        new_fitness, _ = fitness(fs_row, buy_threshold, sell_threshold, df_strategy_np, df_data_open, signal_columns)

        if new_fitness > fitness_list[i]:
            fs[i] = fs_row
            fitness_list[i] = new_fitness
            trial[i] = 0
        else:
            trial[i] += 1

@njit
def onlooker_bee_phase(N, D, fs, fitness_list, trial, buy_threshold, sell_threshold, df_strategy_np, df_data_open, signal_columns, weights_range):
    fitness_array = np.array(fitness_list, dtype=np.float64)
    
    
    # 计算概率
    prob = fitness_array / np.sum(fitness_array)
    
    for i in range(N):
        if np.random.uniform(0, 1) < prob[i]:
            fs_row = fs[i].copy()
            var_idx = np.random.randint(0, D - 1)
            possible_partners = np.array([idx for idx in range(N) if idx != i])
            partner_idx = np.random.choice(possible_partners)
            
            phi = np.random.uniform(-1, 1)
            fs_row[var_idx] += phi * (fs_row[var_idx] - fs[partner_idx, var_idx])

            fs_row[var_idx] = np.clip(np.array(fs_row[var_idx]), *weights_range)


            new_fitness, _ = fitness(fs_row, buy_threshold, sell_threshold, df_strategy_np, df_data_open, signal_columns)
            if new_fitness > fitness_list[i]:
                fs[i] = fs_row
                fitness_list[i] = new_fitness
                trial[i] = 0
            else:
                trial[i] += 1


@njit
def scout_bee_phase(N, D, fs, fitness_list, trial, buy_threshold, sell_threshold, df_strategy_np, df_data_open, signal_columns, weights_range):
    for i in range(N):
        if trial[i] > limit:
            fs[i] = np.random.uniform(*weights_range, D)
            fitness_list[i] = fitness(fs[i], buy_threshold, sell_threshold, df_strategy_np, df_data_open, signal_columns)[0]
            trial[i] = 0
            
            
# Main optimization loop
def run_abc_optimization():
    global best_fitness, best_solution, best_trades_np

    best_fitness_history = []

    with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), TimeRemainingColumn()) as progress:
        task = progress.add_task("[cyan]Optimizing fitness...", total=max_iter)

        for it in range(max_iter):
            employed_bee_phase(N, D, fs, fitness_list, trial, buy_threshold, sell_threshold, df_strategy_np, df_data_open, signal_columns, weights_range)
            onlooker_bee_phase(N, D, fs, fitness_list, trial, buy_threshold, sell_threshold, df_strategy_np, df_data_open, signal_columns, weights_range)
            scout_bee_phase(N, D, fs, fitness_list, trial, buy_threshold, sell_threshold, df_strategy_np, df_data_open, signal_columns, weights_range)

            current_best_fitness = max(fitness_list)
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_solution = fs[np.argmax(fitness_list)].copy()
                _, best_trades_np = fitness(best_solution, buy_threshold, sell_threshold, df_strategy_np, df_data_open, signal_columns)

            best_fitness_history.append(best_fitness)
            if it % 100 == 0:  # Adjust output frequency for less overhead
                rprint(f"[yellow]Iteration {it + 1} - Best Fitness: {best_fitness:.4f}[/yellow]")
            progress.update(task, advance=1)


    plt.figure(figsize=(15, 8))
    plt.plot(best_fitness_history, linewidth=2, label="Best Fitness History", color='r')
    plt.xlabel("Iteration", fontsize=15)
    plt.ylabel("Fitness Value", fontsize=15)
    plt.legend()
    plt.show()

    # Output results
    rprint(f"[bold green]Best Fitness Value:[/bold green] {best_fitness}")
    rprint(f"[bold green]Best Weights:[/bold green] {best_solution}")
    # rprint("[bold green]Best Trades (交易記錄):[/bold green]")
    # rprint(best_trades_df)

    return best_fitness_history, best_solution, best_trades_np

# Call the function to run the optimization
best_fitness_history, best_solution, best_trades_np = run_abc_optimization()


# Create a table for algorithm parameters
table = Table(show_header=True, header_style="bold cyan", title="Algorithm Parameters", border_style="bold magenta")
table.add_column("Parameter", style="dim", width=20)
table.add_column("Value", justify="right")

# Add each parameter to the table
table.add_row("Bee Count (N)", str(N))
table.add_row("Max Iterations", str(max_iter))
table.add_row("Scout Bee Limit", str(limit))
table.add_row("Weights Range", f"{weights_range}")
table.add_row("Buy Threshold", str(buy_threshold))
table.add_row("Sell Threshold", str(sell_threshold))

# Display the table inside a panel
rprint(Panel(table, title="Initial Algorithm Parameters", border_style="green"))



best_trades_df = trades_to_dataframe(best_trades_np, df_data)


best_trades_df.to_pickle('best_trades.pkl')


plot_trades("algorithm", df_data, best_trades_df, best_fitness, 1000, benchmark_df)


