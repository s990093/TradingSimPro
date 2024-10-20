import numpy as np
import yfinance as yf
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

class Environment(object):
    MAX_THREAD_WORKERS = 30
    MAX_PROCESS_WORKERS = 30
    target_stock = "AAPL"
    start_date = datetime(2018, 4, 1)
    end_date = datetime(2022, 7, 19)
    
    population_size = 60
    max_iter = 8000
    weights_range = [0, 1, 2, 3, 4, 5]
    x_range = np.arange(0.05, 0.5, 0.05)
    
    
    signal_columns = [
        'ma_signal', 'rsi_signal', 'macd_signal', 'bb_signal',
        'momentum_signal', 'stochastic_signal', 'breakout_signal',
        'mean_reversion_signal', 'stop_loss_signal', 'trend_following_signal',
        'turtle_trading_signal', 'volume_price_signal'
    ]
    
    @classmethod
    def display_config(cls):
        """Display the configuration of the environment using rich."""
        console = Console()

        # Create a table to display the configuration
        table = Table(title="Environment Configuration")

        # Add columns to the table
        table.add_column("Property", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        # Populate the table with environment properties
        table.add_row("Target Stock", cls.target_stock)
        table.add_row("Start Date", cls.start_date.strftime('%Y-%m-%d'))
        table.add_row("End Date", cls.end_date.strftime('%Y-%m-%d'))
        table.add_row("Population Size", str(cls.population_size))
        table.add_row("Max Iterations", str(cls.max_iter))
        # table.add_row("Weights Range", str(cls.weights_range))
        # table.add_row("x Range", str(cls.x_range.tolist()))
        # table.add_row("Max Thread Workers", str(cls.MAX_THREAD_WORKERS))
        # table.add_row("Max Process Workers", str(cls.MAX_PROCESS_WORKERS))

        # Display the signal columns
        signal_columns_str = ", ".join(cls.signal_columns)
        table.add_row("Signal Columns", signal_columns_str)

        # Create a panel to contain the table
        panel = Panel(table, title="Configuration Details", border_style="bold blue")

        # Print the panel to the console
        console.print(panel)
