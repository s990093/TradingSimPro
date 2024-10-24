import numpy as np
import random
from rich.progress import Progress
import multiprocessing as mp
from rich.console import Console

from utility.base.base_alg import AlgorithmManager, fitness

__all__ = ['ABCQlearningAlgorithmManager']

def evaluate_bee(bee, df_strategy, df_data, signal_columns):
    fitness_value, trades_df = fitness(
        bee['weights'],
        bee['buy_threshold'],
        bee['sell_threshold'],
        df_strategy,
        df_data,
        signal_columns
    )
    return fitness_value, trades_df

class ABCQlearningAlgorithmManager(AlgorithmManager):
    def __init__(self, df_strategy, df_data, CS, MCN, limit, weights_range, x_range, signal_columns, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(self.__class__.__name__)
        self.df_strategy = df_strategy
        self.df_data = df_data
        self.CS = CS
        self.MCN = MCN
        self.limit = limit
        self.weights_range = weights_range
        self.x_range = x_range
        self.signal_columns = signal_columns
        
        
        
        self.alpha = 0.02  # Learning rate
        self.gamma = 0.9   # Discount factor
        self.epsilon = 0.3  # Initial exploration rate

        # Epsilon decay parameters
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  # Decay factor
        


    def initialize_q_table(self, num_states, num_actions):
        return np.zeros((num_states, num_actions))

    def select_action(self, state, q_table, epsilon):
        if random.random() < epsilon:  # Explore
            return random.randint(0, q_table.shape[1] - 1)
        else:  # Exploit
            return np.argmax(q_table[state])

    def update_q_table(self, q_table, state, action, reward, next_state, alpha, gamma):
        best_next_action = np.argmax(q_table[next_state])
        q_table[state, action] += alpha * (reward + gamma * q_table[next_state, best_next_action] - q_table[state, action])

    def run_algorithm(self):
        self.abc_algorithm_qlearning(self.df_strategy, self.df_data, self.CS, self.MCN, self.limit, self.weights_range, self.x_range, self.signal_columns)

    def abc_algorithm_qlearning(self, df_strategy, df_data, CS, MCN, limit, weights_range, x_range, signal_columns):
        # Initialize bee population
        bees = [{
            'weights': [random.uniform(weights_range[0], weights_range[1]) for _ in range(len(signal_columns))],
            'buy_threshold': random.uniform(x_range[0], x_range[1]),
            'sell_threshold': random.uniform(x_range[0], x_range[1]),
            'trials': 0
        } for _ in range(CS)]

        # best_bee = None
        # best_fitness = -float('inf')
        # best_trades_df = None

        # Initialize Q-learning parameters
        q_table = self.initialize_q_table(num_states=CS, num_actions=len(signal_columns) * 3)
       

        with Progress() as progress:
            task = progress.add_task("Running ABC Algorithm with Q-learning...", total=MCN)

            cycle = 1
            while cycle <= MCN:
                with mp.Pool(processes=mp.cpu_count()) as pool:
                    # Evaluate fitness for each bee in parallel
                    fitness_results = pool.starmap(evaluate_bee, [(bee, df_strategy, df_data, signal_columns) for bee in bees])

                for s, (new_fitness, new_trades_df) in enumerate(fitness_results):
                    # Define state based on the current bee's weights and thresholds
                    state = s  # You can enhance this to be a more meaningful representation

                    # Select action using Q-learning
                    action = self.select_action(state, q_table, self.epsilon)

                    # Generate new solution based on action
                    new_weights = [random.uniform(weights_range[0], weights_range[1]) for _ in range(len(signal_columns))]
                    new_buy_threshold = random.uniform(x_range[0], x_range[1])
                    new_sell_threshold = random.uniform(x_range[0], x_range[1])

                    # Calculate fitness of current solution
                    fitness_of_current = fitness(bees[s]['weights'], bees[s]['buy_threshold'], bees[s]['sell_threshold'], df_strategy, df_data, signal_columns)[0]

                    # Determine reward (for example, fitness improvement)
                    reward = new_fitness - fitness_of_current

                    # Update Q-table
                    self.update_q_table(q_table, state, action, reward, state, self.alpha, self.gamma)

                    # Greedy selection logic
                    if new_fitness > fitness_of_current:
                        bees[s]['weights'] = new_weights
                        bees[s]['buy_threshold'] = new_buy_threshold
                        bees[s]['sell_threshold'] = new_sell_threshold
                        bees[s]['trials'] = 0
                    else:
                        bees[s]['trials'] += 1

                for s, (current_fitness, current_trades_df) in enumerate(fitness_results):
                    # Check if the current bee's fitness is better than the best found so far
                    if current_fitness > self.best_fitness:
                        self.best_bee = bees[s].copy()  # Store the current best bee's parameters
                        self.best_fitness = current_fitness
                        self.best_trades_df = current_trades_df
                        print(f"Cycle {cycle}: Best Fitness updated: {self.best_fitness:.4f}")

                # Update the progress bar
                progress.update(task, advance=1)

                # Update epsilon for exploration decay
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

                cycle += 1

        