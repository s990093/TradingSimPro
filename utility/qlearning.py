import numpy as np
import random
from rich.progress import Progress
import multiprocessing as mp

from utility.calculate_returns import calculate_trading_signals



def fitness(weights, buy_threshold, sell_threshold, df_strategy, df_data, signal_columns):  
    return calculate_trading_signals(df_strategy, weights, buy_threshold, sell_threshold, signal_columns, df_data)

def initialize_q_table(num_states, num_actions):
    return np.zeros((num_states, num_actions))

def select_action(state, q_table, epsilon):
    if random.random() < epsilon:  # Explore
        return random.randint(0, q_table.shape[1] - 1)
    else:  # Exploit
        return np.argmax(q_table[state])

def update_q_table(q_table, state, action, reward, next_state, alpha, gamma):
    best_next_action = np.argmax(q_table[next_state])
    q_table[state, action] += alpha * (reward + gamma * q_table[next_state, best_next_action] - q_table[state, action])

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

def abc_algorithm_qlearning(df_strategy, df_data, CS, MCN, limit, weights_range, x_range, signal_columns):
    # Initialize bee population
    bees = [{
        'weights': [random.uniform(weights_range[0], weights_range[1]) for _ in range(len(signal_columns))],
        'buy_threshold': random.uniform(x_range[0], x_range[1]),
        'sell_threshold': random.uniform(x_range[0], x_range[1]),
        'trials': 0
    } for _ in range(CS)]

    best_bee = None
    best_fitness = -float('inf')
    best_trades_df = None

    # Initialize Q-learning parameters
    q_table = initialize_q_table(num_states=CS, num_actions=len(signal_columns) * 3)
    alpha = 0.05  # Learning rate
    gamma = 0.9  # Discount factor
    epsilon = 0.1  # Exploration rate

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
                action = select_action(state, q_table, epsilon)

                # Generate new solution based on action
                new_weights = [random.uniform(weights_range[0], weights_range[1]) for _ in range(len(signal_columns))]
                new_buy_threshold = random.uniform(x_range[0], x_range[1])
                new_sell_threshold = random.uniform(x_range[0], x_range[1])

                # Calculate fitness of new solution
                fitness_of_current = fitness(bees[s]['weights'], bees[s]['buy_threshold'], bees[s]['sell_threshold'], df_strategy, df_data, signal_columns)[0]

                # Determine reward (for example, fitness improvement)
                reward = new_fitness - fitness_of_current

                # Update Q-table
                update_q_table(q_table, state, action, reward, state, alpha, gamma)

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
                if current_fitness > best_fitness:
                    best_bee = bees[s].copy()  # Store the current best bee's parameters
                    best_fitness = current_fitness
                    best_trades_df = current_trades_df
                    print(f"Cycle {cycle}: Best Fitness updated: {best_fitness:.4f}")

            # Update the progress bar
            progress.update(task, advance=1)
            cycle += 1

    # Plotting and returning results

    return best_bee, best_fitness, best_trades_df