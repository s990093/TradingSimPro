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


    return best_bee, best_fitness, best_trades_df

