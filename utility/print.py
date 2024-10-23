import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

console = Console()

def print_df_strategy(columns):
    column_count = len(columns)

    # Create a rich Table to display the columns neatly
    table = Table(title="DataFrame Strategy Columns", show_header=True, header_style="bold magenta")
    table.add_column("Index", justify="right", style="cyan")
    table.add_column("Column Name", style="green")

    # Add rows to the table
    for idx, col in enumerate(columns, start=1):
        table.add_row(str(idx), col)

    # Create a panel with the table and column count
    panel = Panel.fit(
        f"[bold]Total Columns: {column_count}[/bold]",
        title="DataFrame Strategy Information",
        border_style="bold blue"
    )

    # Print the table and the panel
    console.print(table)
    console.print(panel)

# Initialize console
console = Console()

def display_results(best_bee, best_fitness, time, profit_ratio, trades_df):
    # Formatting the best results
    # best_weights_str = ', '.join([f'{w:.2f}' for w in best_bee['weights']])
    best_weights_str = ', '.join([f'{w:.2f}' for w in best_bee])
    # best_threshold_str = f"Sell Threshold: {best_bee['sell_threshold']:.2f}, Buy Threshold: {best_bee['buy_threshold']:.2f}"
    best_return_str = f"{best_fitness:.0f} shares"

    # Calculate average and maximum holding times
    trades_df['Date'] = pd.to_datetime(trades_df['Date'])  # Ensure Date is in datetime format
    buy_dates = trades_df[trades_df['Action'] == 'Buy']['Date']
    sell_dates = trades_df[trades_df['Action'] == 'Sell']['Date']

    # Ensure equal number of buys and sells for accurate holding time calculation
    if len(buy_dates) > len(sell_dates):
        buy_dates = buy_dates[:len(sell_dates)]  # truncate buy_dates to match sell_dates
    elif len(sell_dates) > len(buy_dates):
        sell_dates = sell_dates[:len(buy_dates)]  # truncate sell_dates to match buy_dates

    holding_times = sell_dates.values - buy_dates.values

    # Convert holding times to seconds and then to days
    average_holding_time = holding_times.mean()  # This will be a numpy timedelta64
    max_holding_time = holding_times.max()  # This will be a numpy timedelta64

    # Convert timedelta to days
    average_holding_time_days = average_holding_time / pd.Timedelta(days=1) if holding_times.size > 0 else 0
    max_holding_time_days = max_holding_time / pd.Timedelta(days=1) if holding_times.size > 0 else 0

    # Ensure both are float
    average_holding_time_days = float(average_holding_time_days)
    max_holding_time_days = float(max_holding_time_days)


    # Calculate profits for each transaction
    profits = []
    for i in range(len(trades_df) - 1):
        if trades_df['Action'].iloc[i] == 'Buy' and trades_df['Action'].iloc[i + 1] == 'Sell':
            profits.append(trades_df['Price'].iloc[i + 1] - trades_df['Price'].iloc[i])

    profit_str = ', '.join([f"{p:.2f}" for p in profits]) if profits else "No Profits"
    

    # Create a table for better organization
    table = Table(title="ABC Algorithm Results")
    
    # Add columns to the table
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    # Add rows to the table
    table.add_row("Profit Ratio", f"[bold yellow]{profit_ratio:.2f}%[/bold yellow]")
    table.add_row("Best Weights", f"[bold green]{best_weights_str}[/bold green]")
    # table.add_row("Best Thresholds", f"[bold cyan]{best_threshold_str}[/bold cyan]")
    table.add_row("Best Return", f"[bold yellow]{best_return_str}[/bold yellow]")
    table.add_row("Execution Time", f"[bold yellow]{time} seconds[/bold yellow]")
    table.add_row("Avg Holding Time", f"[bold yellow]{average_holding_time_days:.2f} days[/bold yellow]")
    table.add_row("Max Holding Time", f"[bold yellow]{max_holding_time_days:.2f} days[/bold yellow]")
    table.add_row("Transaction Profits", f"[bold yellow]{profit_str}[/bold yellow]")

    # Display the table in a panel
    panel = Panel(table, title="Results Overview", border_style="bold blue")

    # Print the panel to console
    console.print(panel)
    
    
    
def display_best_strategies(best_strategies, best_performance, total_strategies, selected_strategies_count):
    console = Console()

    # Create a table for strategies
    table = Table(title="Best Strategies")

    # Add columns for the table
    table.add_column("Strategy Name", style="cyan", no_wrap=True)
    table.add_column("Details", style="magenta")

    # Add strategy details to the table
    for strategy in best_strategies:
        strategy_name = strategy.__class__.__name__
        strategy_details = str(strategy)  # You can customize how to display the details
        table.add_row(strategy_name, strategy_details)
    
    summary_message = (
        f"Best Performance: [bold yellow]{best_performance}[/bold yellow]\n"
        f"Total Strategies: [bold blue]{total_strategies}[/bold blue]\n"
        f"Selected Strategies Count: [bold blue]{selected_strategies_count}[/bold blue]"
    )
    
    
    console.print(Panel.fit(table, title="Selected Strategies", border_style="green"))

    # Combine the table and summary message into a single panel
    combined_content = f"{summary_message}"
    combined_panel = Panel(combined_content, title="Strategies Overview", border_style="blue")

    console.print(combined_panel)