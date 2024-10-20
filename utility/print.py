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

def display_results(best_bee, best_fitness, time, profit_ratio):
    # Formatting the best results
    best_weights_str = ', '.join([f'{w:.2f}' for w in best_bee['weights']])
    best_threshold_str = f"Sell Threshold: {best_bee['sell_threshold']:.2f}, Buy Threshold: {best_bee['buy_threshold']:.2f}"
    best_return_str = f"{best_fitness:.0f} shares"

    # Create a table for better organization
    table = Table(title="ABC Algorithm Results")
    
    # Add columns to the table
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    # Add rows to the table
    table.add_row("Profit Ratio", f"[bold yellow]{profit_ratio:.2f}% [/bold yellow]")
    table.add_row("Best Weights", f"[bold green]{best_weights_str}[/bold green]")
    table.add_row("Best Thresholds", f"[bold cyan]{best_threshold_str}[/bold cyan]")
    table.add_row("Best Return", f"[bold yellow]{best_return_str}[/bold yellow]")
    table.add_row("Execution Time", f"[bold yellow]{time} seconds[/bold yellow]")

    # Display the table in a panel
    panel = Panel(table, title="Results Overview", border_style="bold blue")

    # Print the panel to console
    console.print(panel)

