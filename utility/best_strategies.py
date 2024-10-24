import itertools
import os
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from concurrent.futures import ProcessPoolExecutor, as_completed

from ENV import Environment
from utility.calculate_returns import calculate_returns
max_workers = os.cpu_count()

# 设置批次大小
BATCH_SIZE = 100000  # 根据需要调整批次大小

def process_combination(combination, df_strategy, df_data):
    # Initialize combined positions to zeros
    combined_positions = np.zeros(len(df_strategy))

    # Combine signals by summing them (weights are assumed equal)
    for col in combination:
        combined_positions += df_strategy[col].values 

    # Add the combined positions back to the DataFrame
    df_strategy['combined_positions'] = combined_positions

    # Generate buy and sell signals based on combined positions
    buy_signals = (df_strategy['combined_positions'] > 0).astype(int)
    sell_signals = (df_strategy['combined_positions'] < 0).astype(int)

    # Calculate the performance of the current combination
    total_return, trades_df = calculate_returns(buy_signals, sell_signals, df_data)

    return combination, total_return


def select_best_strategies(df_strategy, df_data, signal_columns):
    console = Console()
    best_combination = None
    best_performance = float('-inf')

    # Get the number of strategies
    num_strategies = len(signal_columns)

    # Initialize progress bar
    total_combinations = sum(1 for r in range(4, num_strategies + 1) for _ in itertools.combinations(signal_columns, r))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.completed}/{task.total}"),
    ) as progress:
        
        task = progress.add_task("Processing combinations...", total=total_combinations)

        # Loop through combinations of 4 to the total number of strategies
        for r in range(4, num_strategies + 1):
            # Generate all combinations of the strategies in signal_columns for this value of r
            combinations = list(itertools.combinations(signal_columns, r))

            # Break combinations into batches
            for i in range(0, len(combinations), BATCH_SIZE):
                batch = combinations[i:i + BATCH_SIZE]

                # Create a ProcessPoolExecutor for each batch
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = []

                    # Submit batch combinations for processing
                    for combination in batch:
                        futures.append(executor.submit(process_combination, combination, df_strategy.copy(), df_data))

                    # Process completed futures
                    for future in as_completed(futures):
                        combination, total_return = future.result()

                        # Check if this combination gives the best performance so far
                        if total_return > best_performance:
                            best_performance = total_return
                            best_combination = combination

                            # Log the performance of the best combination found so far
                            console.print(Panel(f"New best combination found: {combination} with return: {total_return}", title="Best Combination", border_style="green"))

                        # Update the progress bar
                        progress.update(task, advance=1)

    return best_combination, best_performance
