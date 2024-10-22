import random
from matplotlib import pyplot as plt
from rich.progress import Progress
from rich.console import Console
from ENV import Environment
from .calculate_returns import calculate_trading_signals


console = Console()  # Initialize rich console

# Define the fitness function
def fitness(weights, buy_threshold, sell_threshold, df_strategy, df_data, signal_columns):  
    return calculate_trading_signals(df_strategy, weights, buy_threshold, sell_threshold, signal_columns, df_data)


def plot_abc_algorithm_convergence(fitness_history, max_iter):
    plt.ion()  # Turn on interactive mode
    plt.figure(figsize=(10, 6))
    plt.title('ABC Algorithm Convergence')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness Value')

    # Initialize an empty line
    line, = plt.plot([], [], label='Best Fitness', color='b')  
    plt.legend()
    plt.grid(True)

    # Set the x and y axis limits
    plt.xlim(0, max_iter)
    plt.ylim(min(fitness_history) - 1, max(fitness_history) + 1)

    # Updating the plot
    line.set_xdata(range(len(fitness_history)))  # Update x data
    line.set_ydata(fitness_history)  # Update y data

    # Redraw the plot to show updates
    plt.draw()
    plt.pause(0.01)  # Pause to ensure the plot updates interactively

    # Save the figure as a PNG file
    plt.savefig('res/abc_algorithm_convergence.png', dpi=300)

    # Manually close the plot
    plt.ioff()  # Turn off interactive mode after plotting
    plt.show()  # Wait for manual plot closure

def abc_algorithm(df_strategy, df_data, CS, MCN, limit, weights_range, x_range, signal_columns):
    # Initialize bee population
    bees = [{
        'weights': [random.uniform(weights_range[0], weights_range[1]) for _ in range(len(signal_columns))],
        'buy_threshold': random.uniform(x_range[0], x_range[1]),
        'sell_threshold': random.uniform(x_range[0], x_range[1]),
        'trials': 0  # Initialize trial counts
    } for _ in range(CS)]

    best_bee = None
    best_fitness = -float('inf')
    fitness_history = []  # To store the best fitness values over iterations
    best_trades_df = None

    with Progress() as progress:
        task = progress.add_task("Running ABC Algorithm...", total=MCN)

        cycle = 1
        while cycle <= MCN:
            # Employed bee phase
            for s in range(CS):
                # Generate new solution
                new_weights = [random.uniform(weights_range[0], weights_range[1]) for _ in range(len(signal_columns))]
                new_buy_threshold = random.uniform(x_range[0], x_range[1])
                new_sell_threshold = random.uniform(x_range[0], x_range[1])
                
                # Calculate fitness of new solution
                new_fitness, new_trades_df = fitness(new_weights, new_buy_threshold, new_sell_threshold, df_strategy, df_data, signal_columns)

                # Greedy selection
                if new_fitness > fitness(bees[s]['weights'], bees[s]['buy_threshold'], bees[s]['sell_threshold'], df_strategy, df_data, signal_columns)[0]:
                    bees[s]['weights'] = new_weights
                    bees[s]['buy_threshold'] = new_buy_threshold
                    bees[s]['sell_threshold'] = new_sell_threshold
                    bees[s]['trials'] = 0  # Reset trial count
                else:
                    bees[s]['trials'] += 1  # Increase trial count

            # Onlooker bee phase
            fitness_values = [fitness(bee['weights'], bee['buy_threshold'], bee['sell_threshold'], df_strategy, df_data, signal_columns)[0] for bee in bees]
            total_fitness = sum(fitness_values)
            probabilities = [fitness_value / total_fitness for fitness_value in fitness_values]

            for s in range(CS):
                if random.random() < probabilities[s]:
                    # Generate new solution based on selected bee
                    new_weights = [random.uniform(weights_range[0], weights_range[1]) for _ in range(len(signal_columns))]
                    new_buy_threshold = random.uniform(x_range[0], x_range[1])
                    new_sell_threshold = random.uniform(x_range[0], x_range[1])
                    
                    # Calculate fitness of new solution
                    new_fitness, new_trades_df = fitness(new_weights, new_buy_threshold, new_sell_threshold, df_strategy, df_data, signal_columns)

                    # Greedy selection
                    if new_fitness > fitness(bees[s]['weights'], bees[s]['buy_threshold'], bees[s]['sell_threshold'], df_strategy, df_data, signal_columns)[0]:
                        bees[s]['weights'] = new_weights
                        bees[s]['buy_threshold'] = new_buy_threshold
                        bees[s]['sell_threshold'] = new_sell_threshold
                        bees[s]['trials'] = 0  # Reset trial count
                    else:
                        bees[s]['trials'] += 1  # Increase trial count

            # Scout bee phase
            for s in range(CS):
                if bees[s]['trials'] > limit:
                    # Reinitialize food source
                    bees[s] = {
                        'weights': [random.uniform(weights_range[0], weights_range[1]) for _ in range(len(signal_columns))],
                        'buy_threshold': random.uniform(x_range[0], x_range[1]),
                        'sell_threshold': random.uniform(x_range[0], x_range[1]),
                        'trials': 0
                    }

            # Update best solution found
            for bee in bees:
                current_fitness, current_trades_df = fitness(bee['weights'], bee['buy_threshold'], bee['sell_threshold'], df_strategy, df_data, signal_columns)
                if current_fitness > best_fitness:
                    best_bee = bee
                    best_fitness = current_fitness
                    best_trades_df = current_trades_df
                    console.print(f"[bold green]Cycle {cycle}:[/bold green] Best Fitness: {best_fitness:.4f}")
            
            # Store best fitness of this cycle
            fitness_history.append(best_fitness)

            # Update the progress bar
            progress.update(task, advance=1)
            cycle += 1

    plot_abc_algorithm_convergence(fitness_history, MCN)

    return best_bee, best_fitness, best_trades_df


def calculate_membership(fitness, average_fitness):
    if fitness >= 1.5 * average_fitness:
        return 1  # Very good solution
    elif fitness <= 0.5 * average_fitness:
        return 0  # Poor solution
    else:
        # Linear membership function between 0 and 1
        return (fitness - 0.5 * average_fitness) / (average_fitness)

def mutate_solution(bee, best_bee, weights_range):
    F = random.uniform(0, 1.5)  # Scaling factor for mutation
    mutated_weights = [
        bee['weights'][i] + F * (best_bee['weights'][i] - bee['weights'][i])
        for i in range(len(bee['weights']))
    ]
    # Ensure the new weights stay within bounds
    return [max(min(w, weights_range[1]), weights_range[0]) for w in mutated_weights]


def move_with_gbest_guidance(bee, gbest, weights_range):
    rand_factor = random.uniform(-1, 1)
    gbest_influence = random.uniform(0, 1.5)
    new_weights = [
        bee['weights'][i] + rand_factor * (gbest['weights'][i] - bee['weights'][i]) +
        gbest_influence * (gbest['weights'][i] - bee['weights'][i])
        for i in range(len(bee['weights']))
    ]
    return new_weights

def update_scout_bee(bee, population, best_bee, weights_range, x_range, mutation_factor=0.8, crossover_prob=0.9):
    """
    Enhanced scout bee phase combining random exploration and differential evolution.
    :param bee: The bee that needs to be reset
    :param population: List of all bees (for differential evolution)
    :param best_bee: The current best bee (for differential evolution guidance)
    :param weights_range: The range of values allowed for weights
    :param x_range: The range for buy/sell thresholds
    :param mutation_factor: Differential evolution mutation factor F
    :param crossover_prob: Probability of crossover
    :return: Updated bee
    """
    if random.random() < 0.5:
        # Differential Evolution-based exploration (using the best bee)
        # Select three random bees from the population
        candidates = random.sample(population, 3)
        
        # Perform mutation using the DE formula
        mutated_weights = [
            candidates[0]['weights'][i] + mutation_factor * (candidates[1]['weights'][i] - candidates[2]['weights'][i])
            for i in range(len(bee['weights']))
        ]
        
        # Apply crossover (with some probability, inherit from the mutated weights)
        new_weights = [
            mutated_weights[i] if random.random() < crossover_prob else bee['weights'][i]
            for i in range(len(bee['weights']))
        ]
        
        # Ensure the new weights are within bounds
        bee['weights'] = [max(min(w, weights_range[1]), weights_range[0]) for w in new_weights]
    else:
        # Random exploration (traditional scout bee behavior)
        bee['weights'] = [random.uniform(weights_range[0], weights_range[1]) for _ in bee['weights']]
    
    # Randomly reset buy/sell thresholds within the specified range
    bee['buy_threshold'] = random.uniform(x_range[0], x_range[1])
    bee['sell_threshold'] = random.uniform(x_range[0], x_range[1])
    
    return bee


# def abc_algorithm(df_strategy, df_data, population_size, max_iter, weights_range, x_range, signal_columns):
    bees = [{
        'weights': [random.uniform(weights_range[0], weights_range[1]) for _ in range(len(signal_columns))],
        'buy_threshold': random.uniform(x_range[0], x_range[1]),
        'sell_threshold': random.uniform(x_range[0], x_range[1])
    } for _ in range(population_size)]

    # Initialize best_bee to the first bee to avoid NoneType issue
    best_bee = bees[0]
    best_fitness = -float('inf')
    fitness_history = []
    
         # Enable interactive mode for matplotlib
    plt.ion()  # Turn on interactive mode
    plt.figure(figsize=(10, 6))
    plt.title('ABC Algorithm Convergence')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness Value')
    line, = plt.plot([], [], label='Best Fitness')  # Initialize an empty line
    plt.legend()
    plt.grid(True)


    # Create a progress bar using rich
    with Progress() as progress:
        task = progress.add_task("[green]Running ABC Algorithm...", total=max_iter)

        for iteration in range(max_iter):
            average_fitness = sum(fitness(bee['weights'], bee['buy_threshold'], bee['sell_threshold'], df_strategy, df_data, signal_columns)[0] for bee in bees) / len(bees)

            for bee in bees:
                bee_fitness, best_trades_df = fitness(bee['weights'], bee['buy_threshold'], bee['sell_threshold'], df_strategy, df_data, signal_columns)
                membership = calculate_membership(bee_fitness, average_fitness)
                
                if membership < 0.5:
                    bee['weights'] = mutate_solution(bee, best_bee, weights_range)
                else:
                    # Check if best_bee is valid
                    if best_bee:
                        bee['weights'] = move_with_gbest_guidance(bee, best_bee, weights_range)
                
                # Recalculate fitness
                bee_fitness, trades_df= fitness(bee['weights'], bee['buy_threshold'], bee['sell_threshold'], df_strategy, df_data, signal_columns)
                
                if bee_fitness > best_fitness:
                    best_bee = bee
                    best_trades_df = trades_df
                    best_fitness = bee_fitness
                    
            
            # Scout bee phase
            random_bee = random.choice(bees)
            update_scout_bee(random_bee, bees, best_bee, weights_range, x_range)
            
            # Track the fitness progress
            fitness_history.append(best_fitness)
            
            if len(fitness_history) > 0:
                            line.set_xdata(range(len(fitness_history)))  # Update x data
                            line.set_ydata(fitness_history)  # Update y data
                            plt.xlim(0, max_iter)  # Set x-axis limit
                            plt.ylim(min(fitness_history) - 1, max(fitness_history) + 1)  # Set y-axis limit dynamically
                            plt.draw()  # Redraw the plot
                            plt.pause(0.1)  # Pause to allow the plot to update
                            
            # Update the progress bar after each iteration
            progress.update(task, advance=1)
    # Save the figure as a PNG file
    plt.savefig('res/abc_algorithm_convergence.png', dpi=300)  
    
    return best_bee, best_fitness, best_trades_df