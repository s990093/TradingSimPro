import numpy as np
import random
from rich.progress import Progress

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
    fitness_history = []
    best_trades_df = None

    # Initialize Q-learning parameters
    q_table = initialize_q_table(num_states=CS, num_actions=len(signal_columns) * 3)
    alpha = 0.1  # Learning rate
    gamma = 0.9  # Discount factor
    epsilon = 0.1  # Exploration rate

    with Progress() as progress:
        task = progress.add_task("Running ABC Algorithm with Q-learning...", total=MCN)

        cycle = 1
        while cycle <= MCN:
            for s in range(CS):
                # Define state based on the current bee's weights and thresholds
                state = s  # You can enhance this to be a more meaningful representation
                
                # Select action using Q-learning
                action = select_action(state, q_table, epsilon)
                
                # Generate new solution based on action
                new_weights = [random.uniform(weights_range[0], weights_range[1]) for _ in range(len(signal_columns))]
                new_buy_threshold = random.uniform(x_range[0], x_range[1])
                new_sell_threshold = random.uniform(x_range[0], x_range[1])

                # Calculate fitness of new solution
                new_fitness, new_trades_df = fitness(new_weights, new_buy_threshold, new_sell_threshold, df_strategy, df_data, signal_columns)

                # Determine reward (for example, fitness improvement)
                reward = new_fitness - fitness(bees[s]['weights'], bees[s]['buy_threshold'], bees[s]['sell_threshold'], df_strategy, df_data, signal_columns)[0]

                # Update Q-table
                update_q_table(q_table, state, action, reward, state, alpha, gamma)

                # Greedy selection logic remains the same
                if new_fitness > fitness(bees[s]['weights'], bees[s]['buy_threshold'], bees[s]['sell_threshold'], df_strategy, df_data, signal_columns)[0]:
                    bees[s]['weights'] = new_weights
                    bees[s]['buy_threshold'] = new_buy_threshold
                    bees[s]['sell_threshold'] = new_sell_threshold
                    bees[s]['trials'] = 0
                else:
                    bees[s]['trials'] += 1

            for bee in bees:
                current_fitness, current_trades_df = fitness(bee['weights'], bee['buy_threshold'], bee['sell_threshold'], df_strategy, df_data, signal_columns)
                
                # Check if the current bee's fitness is better than the best found so far
                if current_fitness > best_fitness:
                    best_bee = bee.copy()  # Store the current best bee's parameters
                    best_fitness = current_fitness
                    best_trades_df = current_trades_df
                    print(f"Cycle {cycle}: Best Fitness updated: {best_fitness:.4f}")

            # Update the progress bar
            progress.update(task, advance=1)
            cycle += 1

    # Plotting and returning results

    return best_bee, best_fitness, best_trades_df
