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
from multiprocessing import Pool, cpu_count

# Local imports
from utility.print import display_results
from utility.helper.tool import trades_to_dataframe
from utility.stock_plotter import plot_trades
from utility.calculate_returns_jit import fitness
from utility.helper.stock_data_cache import StockDataCache
from strategies import create_strategies
from ENV import Environment

            
            
@njit
def update_particle(positions, velocities, personal_best_positions, 
                    personal_best_fitness, global_best_position, w, c1, c2, 
                    weights_range):
    for i in range(len(positions)):
        # Update velocity
        r1, r2 = np.random.random(2)
        new_velocity = (w * velocities[i] +
                        c1 * r1 * (personal_best_positions[i] - positions[i]) +
                        c2 * r2 * (global_best_position - positions[i]))
        
        # Update position
        new_position = positions[i] + new_velocity
        
        # Clip position to bounds
        new_position = np.clip(new_position, weights_range[0], weights_range[1])
        
        # Update arrays
        velocities[i] = new_velocity
        positions[i] = new_position
        
    return positions, velocities

class PSOOptimizer:
    def __init__(self, df_strategy, df_data, signal_columns, n_particles=4000, max_iterations=70000,
                 weights_range=(-1, 1), buy_threshold=1.0, sell_threshold=1.0,
                 w=0.7, c1=2.0, c2=2.0):
        # PSO parameters
        self.n_particles = n_particles
        self.max_iter = max_iterations
        self.weights_range = weights_range
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        
        # PSO coefficients
        self.w = w      # Inertia weight
        self.c1 = c1    # Cognitive coefficient
        self.c2 = c2    # Social coefficient
        
        # Data
        self.df_strategy_np = df_strategy.to_numpy()
        self.df_data_open = df_data['Open'].to_numpy()
        self.signal_columns = signal_columns
        self.df_data = df_data
        
        # Best solutions tracking
        self.global_best_position = None
        self.global_best_fitness = float('-inf')
        self.global_best_trades = None


    def initialize_population(self, dimension):
        self.dimension = dimension
        
        # Initialize particle positions and velocities
        self.positions = np.random.uniform(*self.weights_range, (self.n_particles, self.dimension))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.n_particles, self.dimension))
        
        # Initialize personal best
        self.personal_best_positions = self.positions.copy()
        self.personal_best_fitness = np.array([
            fitness(self.positions[i], self.buy_threshold, self.sell_threshold,
                   self.df_strategy_np, self.df_data_open, self.signal_columns)[0]
            for i in range(self.n_particles)
        ])
        
        # Initialize global best
        best_idx = np.argmax(self.personal_best_fitness)
        self.global_best_position = self.positions[best_idx].copy()
        self.global_best_fitness = self.personal_best_fitness[best_idx]
        _, self.global_best_trades = fitness(
            self.global_best_position, self.buy_threshold, self.sell_threshold,
            self.df_strategy_np, self.df_data_open, self.signal_columns
        )

    def evaluate_particle(self, position):
        return fitness(position, self.buy_threshold, self.sell_threshold,
                      self.df_strategy_np, self.df_data_open, self.signal_columns)[0]

    def run_optimization(self):
        fitness_history = []
        n_cores = cpu_count()  # 獲取CPU核心數
        
        with Pool(processes=n_cores) as pool:  # 創建進程池
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeRemainingColumn()
            ) as progress:
                task = progress.add_task("[cyan]Running PSO optimization...", total=self.max_iter)
                
                for iteration in range(self.max_iter):
                    # Update particles
                    self.positions, self.velocities = update_particle(
                        self.positions, self.velocities,
                        self.personal_best_positions,
                        self.personal_best_fitness,
                        self.global_best_position,
                        self.w, self.c1, self.c2,
                        self.weights_range
                    )
                    
                    current_fitness = np.array(pool.map(self.evaluate_particle, self.positions))
                    
                    # Update personal bests
                    improved_mask = current_fitness > self.personal_best_fitness
                    self.personal_best_positions[improved_mask] = self.positions[improved_mask]
                    self.personal_best_fitness[improved_mask] = current_fitness[improved_mask]
                    
                    # Update global best
                    current_best_idx = np.argmax(current_fitness)
                    if current_fitness[current_best_idx] > self.global_best_fitness:
                        self.global_best_fitness = current_fitness[current_best_idx]
                        self.global_best_position = self.positions[current_best_idx].copy()
                        _, self.global_best_trades = fitness(
                            self.global_best_position, self.buy_threshold, self.sell_threshold,
                            self.df_strategy_np, self.df_data_open, self.signal_columns
                        )
                    
                    fitness_history.append(self.global_best_fitness)
                    
                    # Progress reporting
                    if iteration % 100 == 0:
                        rprint(f"[yellow]Iteration {iteration + 1} - Best Fitness: {self.global_best_fitness:.4f}[/yellow]")
                    progress.update(task, advance=1)
        
        return fitness_history

    def display_results(self, df_data, benchmark_df):
        # Create results table
        table = Table(show_header=True, header_style="bold cyan",
                     title="PSO Parameters", border_style="bold magenta")
        table.add_column("Parameter", style="dim", width=20)
        table.add_column("Value", justify="right")
        
        table.add_row("Particles Count", str(self.n_particles))
        table.add_row("Max Iterations", str(self.max_iter))
        table.add_row("Weights Range", f"{self.weights_range}")
        table.add_row("Inertia Weight (w)", f"{self.w}")
        table.add_row("Cognitive Coef (c1)", f"{self.c1}")
        table.add_row("Social Coef (c2)", f"{self.c2}")
        table.add_row("Buy Threshold", str(self.buy_threshold))
        table.add_row("Sell Threshold", str(self.sell_threshold))
        
        rprint(Panel(table, title="Algorithm Parameters", border_style="green"))
        
        # Plot results
        best_trades_df = trades_to_dataframe(self.global_best_trades, df_data)
        # plot_trades("PSO Algorithm", df_data, best_trades_df, 
        #            self.global_best_fitness, 1000, benchmark_df)
        return best_trades_df

def plot_portfolio_performance(best_trades_df, initial_capital=100000):
    """
    Plot the portfolio performance including cumulative returns and trade points
    
    Parameters:
    -----------
    best_trades_df : DataFrame
        DataFrame containing trade information with columns ['Action', 'Price', 'Shares']
    initial_capital : float, optional
        Initial investment amount (default: 100000)
    """
    # Calculate cumulative returns
    portfolio_value = initial_capital
    portfolio_values = []
    dates = []
    
    for idx, row in best_trades_df.iterrows():
        if row['Action'] == 'Buy':
            portfolio_value -= row['Price'] 
        elif row['Action'] == 'Sell':
            portfolio_value += row['Price'] 
        
        portfolio_values.append(portfolio_value)
        dates.append(idx)
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    plt.plot(dates, portfolio_values, label='Portfolio Value', color='blue', linewidth=2)
    
    # Add buy points
    buy_points = best_trades_df[best_trades_df['Action'] == 'Buy']
    plt.scatter(buy_points.index, [portfolio_values[dates.index(date)] for date in buy_points.index],
               marker='^', color='green', s=100, label='Buy')
    
    # Add sell points
    sell_points = best_trades_df[best_trades_df['Action'] == 'Sell']
    plt.scatter(sell_points.index, [portfolio_values[dates.index(date)] for date in sell_points.index],
               marker='v', color='red', s=100, label='Sell')
    
    # Customize the plot
    plt.title('Portfolio Value Over Time', fontsize=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add final return annotation
    final_return = portfolio_values[-1] - initial_capital
    final_return_pct = (final_return / initial_capital) * 100
    plt.annotate(f'Final Return: ${final_return:.2f} ({final_return_pct:.2f}%)',
                xy=(dates[-1], portfolio_values[-1]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(facecolor='white', alpha=0.7))
    
    plt.show()
    
    return final_return, final_return_pct

def main():
    # Data preparation
    df_data = StockDataCache(Environment.target_stock,
                            Environment.start_date,
                            Environment.end_date).get_data()
  
    benchmark_df = StockDataCache('^GSPC',
                                Environment.start_date,
                                Environment.end_date).get_data()
    # Strategy initialization
    strategy_manager = create_strategies()
    df_strategy = strategy_manager.apply_all_strategies(df_data)
    signal_columns = strategy_manager.get_signal_columns()
    
    # Initialize optimizer
    optimizer = PSOOptimizer(
        df_strategy=df_strategy,
        df_data=df_data,
        signal_columns=signal_columns,
        n_particles=500,
        max_iterations=8000,
    )
    
    optimizer.initialize_population(len(signal_columns))
    
    # Run optimization
    fitness_history = optimizer.run_optimization()
    
    # Display and save results
    best_trades_df = optimizer.display_results(df_data, benchmark_df)
    best_trades_df.to_pickle('best_trades.pkl')
    
    # Call the plotting function
    final_return, final_return_pct = plot_portfolio_performance(best_trades_df)
    print(f"Final Return: ${final_return:.2f} ({final_return_pct:.2f}%)")
    
    # Add visualization for best_trades_df
    plt.figure(figsize=(15, 8))
    
    # Plot stock price
    plt.plot(df_data.index, df_data['Close'], label='Stock Price', color='blue', alpha=0.6)
    
    # Plot buy signals
    buy_points = best_trades_df[best_trades_df['Action'] == 'BUY']
    plt.scatter(buy_points.index, buy_points['Price'], 
               marker='^', color='green', s=100, label='Buy Signal')
    
    # Plot sell signals
    sell_points = best_trades_df[best_trades_df['Action'] == 'SELL']
    plt.scatter(sell_points.index, sell_points['Price'], 
               marker='v', color='red', s=100, label='Sell Signal')
    
    plt.title('Trading Signals on Stock Price', fontsize=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Original fitness history plot
    plt.figure(figsize=(15, 8))
    plt.plot(fitness_history, linewidth=2, label="Best Fitness History", color='r')
    plt.xlabel("Iteration", fontsize=15)
    plt.ylabel("Fitness Value", fontsize=15)
    plt.legend()
    plt.show()
    



if __name__ == "__main__":
    main()