import pickle
import json
import matplotlib.pyplot as plt
from rich.console import Console

from utility.calculate_returns_jit import fitness

__all__ = ['fitness', 'AlgorithmManager']

# def fitness(weights, buy_threshold, sell_threshold, df_strategy, df_data, signal_columns):  
#     return calculate_trading_signals(df_strategy, weights, buy_threshold, sell_threshold, signal_columns, df_data)


class AlgorithmManager:
    def __init__(self, filename):
        self.filename = filename
        self.best_bee = None
        self.best_fitness = -float('inf')
        self.best_trades_df = None
        self.fitness_history = []
        self.console = Console() 

        
    # Method to store results in a pickle file
    def store_results_pickle(self):
        with open(f"data/{self.filename}.pkl", 'wb') as file:
            pickle.dump((self.best_bee, self.best_fitness, self.best_trades_df), file)
        print(f"Results saved to {self.filename}")

    # Method to load results from a pickle file
    def load_results_pickle(self):
        with open(f"data/{self.filename}.pkl", 'rb') as file:
            self.best_bee, self.best_fitness, self.best_trades_df = pickle.load(file)
        print(f"Results loaded from {self.filename}")
            
    def run_algorithm(self):
        pass
    
    def get_res(self):
        return self.best_bee, self.best_fitness, self.best_trades_df

    def plot_abc_algorithm_convergence(self, fitness_history, max_iter):
        plt.figure(figsize=(10, 6))
        plt.title('ABC Algorithm Convergence')
        plt.xlabel('Iterations')
        plt.ylabel('Fitness Value')

        # Plot the fitness history
        plt.plot(range(len(fitness_history)), fitness_history, label='Best Fitness', color='b')
        plt.legend()
        plt.grid(True)

        # Set the x and y axis limits
        plt.xlim(0, max_iter)
        plt.ylim(min(fitness_history) - 1, max(fitness_history) + 1)

        # Save the figure as a PNG file
        plt.tight_layout()
        plt.savefig(f"res/{self.filename}.png", dpi=300)

        # Close the plot to avoid displaying it
        plt.close()
        
