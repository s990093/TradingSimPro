import random
from rich.progress import Progress
from ENV import Environment
from utility.base.base_alg import AlgorithmManager, fitness
from multiprocessing import Pool
import multiprocessing as mp
from numba import njit

__all__ = ['AlgorithmManager']

def calculate_fitness(bee, df_strategy, df_data, signal_columns):
    """Calculate the fitness for a given bee."""
    return fitness(bee['weights'], bee['buy_threshold'], bee['sell_threshold'], df_strategy, df_data, signal_columns)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class ABCAlgorithmManager(AlgorithmManager):
    def __init__(self, df_strategy, df_data, CS, MCN, limit, weights_range, x_range, signal_columns,
                 restart_threshold=200, max_restarts=1000, num_processes=mp.cpu_count()):
        super().__init__(self.__class__.__name__)
        self.df_strategy = df_strategy
        self.df_data = df_data
        self.CS = CS
        self.MCN = MCN
        self.limit = limit
        self.weights_range = weights_range
        self.x_range = x_range
        self.signal_columns = signal_columns
        self.num_processes = num_processes
        
        self.restart_threshold = restart_threshold
        self.max_restarts = max_restarts

    
    def run_algorithm(self,):
        self.abc_algorithm(self.df_strategy, self.df_data, self.CS, self.MCN, self.limit, self.weights_range, self.x_range, self.signal_columns)
        
        
    @njit
    def calculate_new_thresholds(x_range):
        return (random.uniform(x_range[0], x_range[1]), random.uniform(x_range[0], x_range[1]))


    def abc_algorithm(self, df_strategy, df_data, CS, MCN, limit, weights_range, x_range, signal_columns):
        def initialize_bees():
            return [{
                'weights': [random.uniform(weights_range[0], weights_range[1]) for _ in range(len(signal_columns))],
                'buy_threshold': random.uniform(x_range[0], x_range[1]),
                'sell_threshold': random.uniform(x_range[0], x_range[1]),
                'trials': 0  # Initialize trial counts
            } for _ in range(CS)]

        restart_threshold = self.restart_threshold
        max_restarts = self.restart_threshold
        restarts = 0
        
        while restarts < max_restarts:
            # Initialize variables
            bees = initialize_bees()
            best_bee = None
            best_fitness = -float('inf')
            fitness_history = []  # To store the best fitness values over iterations
            best_trades_df = None
            cycle = 1
            first_three_best_fitness = []

            # Define chunk size based on CPU count and CS (colony size)
            chunk_size = max(1, CS // mp.cpu_count())

            with Progress() as progress:
                task = progress.add_task("Running ABC Algorithm...", total=MCN)

                while cycle <= MCN:
                    # Employed bee phase
                    with Pool(processes=mp.cpu_count()) as pool:
                        fitness_results = pool.starmap(
                            calculate_fitness,
                            [(bee, df_strategy, df_data, signal_columns) for batch in chunks(bees, chunk_size) for bee in batch]
                        )

                    # Process the results after the parallel computation
                    for s in range(CS):
                        new_weights = [random.uniform(weights_range[0], weights_range[1]) for _ in range(len(signal_columns))]
                        new_buy_threshold = random.uniform(x_range[0], x_range[1]) 
                        new_sell_threshold = random.uniform(x_range[0], x_range[1])

                        new_fitness, new_trades_df = fitness(new_weights, new_buy_threshold, new_sell_threshold, df_strategy, df_data, signal_columns)

                        if new_fitness > fitness_results[s][0]:
                            bees[s]['weights'] = new_weights
                            bees[s]['buy_threshold'] = new_buy_threshold
                            bees[s]['sell_threshold'] = new_sell_threshold
                            bees[s]['trials'] = 0
                        else:
                            bees[s]['trials'] += 1

                    # Onlooker bee phase
                    total_fitness = sum(result[0] for result in fitness_results)
                    probabilities = [result[0] / total_fitness for result in fitness_results]

                    for s in range(CS):
                        if random.random() < probabilities[s]:
                            new_weights = [random.uniform(weights_range[0], weights_range[1]) for _ in range(len(signal_columns))]
                            new_buy_threshold = random.uniform(x_range[0], x_range[1])
                            new_sell_threshold = random.uniform(x_range[0], x_range[1])

                            new_fitness, new_trades_df = fitness(new_weights, new_buy_threshold, new_sell_threshold, df_strategy, df_data, signal_columns)

                            if new_fitness > fitness_results[s][0]:
                                bees[s]['weights'] = new_weights
                                bees[s]['buy_threshold'] = new_buy_threshold
                                bees[s]['sell_threshold'] = new_sell_threshold
                                bees[s]['trials'] = 0
                            else:
                                bees[s]['trials'] += 1

                    # Scout bee phase
                    for s in range(CS):
                        if bees[s]['trials'] > limit:
                            bees[s] = {
                                'weights': [random.uniform(weights_range[0], weights_range[1]) for _ in range(len(signal_columns))],
                                'buy_threshold': random.uniform(x_range[0], x_range[1]),
                                'sell_threshold': random.uniform(x_range[0], x_range[1]),
                                'trials': 0
                            }

                    # Update best solution
                    for i, bee in enumerate(bees):
                        current_fitness, current_trades_df = fitness_results[i]
                        if current_fitness > best_fitness:
                            best_bee = bee
                            best_fitness = current_fitness
                            best_trades_df = current_trades_df
                            self.console.print(f"[bold green]Cycle {cycle}:[/bold green] Best Fitness: {best_fitness:.4f}")

                    fitness_history.append(best_fitness)

                    # Track best fitness in the first three cycles
                    if cycle <= 3:
                        first_three_best_fitness.append(best_fitness)

                    # Check if best fitness exceeded 150 within the first three cycles
                    if cycle == 3 and max(first_three_best_fitness) < restart_threshold:
                        self.console.print(f"[bold red]Restarting due to low fitness in first three cycles ({max(first_three_best_fitness)} < {restart_threshold})[/bold red]")
                        break 

                    # Update the progress bar
                    progress.update(task, advance=1)
                    cycle += 1

                    if cycle > MCN:
                        break

            # If fitness exceeded 150 in the first three cycles, return the result
            if max(first_three_best_fitness) >= restart_threshold:
                self.best_bee = best_bee
                self.best_fitness = best_fitness
                self.best_trades_df = best_trades_df
                self.plot_abc_algorithm_convergence(fitness_history, MCN)

            restarts += 1

        self.console.print("[bold red]Algorithm failed to find a solution after 1000 restarts.[/bold red]")
        return None, None, None      