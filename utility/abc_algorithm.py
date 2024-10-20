import random
from matplotlib import pyplot as plt
from rich.progress import Progress
from rich.console import Console
from ENV import Environment
from .calculate_returns import calculate_trading_signals


console = Console()  # Initialize rich console

# Define the fitness function
def fitness(weights, buy_threshold, sell_threshold, df_strategy, df_data):  
    return calculate_trading_signals(df_strategy, weights, buy_threshold, sell_threshold, Environment.signal_columns, df_data)



def abc_algorithm(df_strategy, df_data, population_size, max_iter, weights_range, x_range):
    # Initialize bee population
    bees = [{
        'weights': [random.uniform(weights_range[0], weights_range[1]) for _ in range(len(Environment.signal_columns))],
        'buy_threshold': random.uniform(x_range[0], x_range[1]),  # Add buy_threshold
        'sell_threshold': random.uniform(x_range[0], x_range[1])  # Add sell_threshold
    } for _ in range(population_size)]

    best_bee = None
    best_trades_df = None
    best_fitness = -float('inf')
    fitness_history = []  # To store the best fitness values over iterations
    
     # Enable interactive mode for matplotlib
    plt.ion()  # Turn on interactive mode
    plt.figure(figsize=(10, 6))
    plt.title('ABC Algorithm Convergence')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness Value')
    line, = plt.plot([], [], label='Best Fitness')  # Initialize an empty line
    plt.legend()
    plt.grid(True)


    with Progress() as progress:
        task = progress.add_task("Running ABC Algorithm...", total=max_iter)

        for iteration in range(max_iter):
            # Employed bee phase
            for bee in bees:
                current_fitness, current_trades_df = fitness(bee['weights'], bee['buy_threshold'], bee["sell_threshold"], df_strategy, df_data)

                # Perform local search for the current bee
                new_weights = [random.uniform(weights_range[0], weights_range[1]) for _ in range(len(Environment.signal_columns))]
                new_x = random.uniform(x_range[0], x_range[1])
                new_fitness, newtrades_df =fitness(bee['weights'], bee['buy_threshold'], bee["sell_threshold"], df_strategy, df_data)
                
                # Update bee's position if new solution is better
                if new_fitness > current_fitness:
                    bee['weights'] = new_weights
                    bee['x'] = new_x
                    current_fitness = new_fitness

                # Update the best solution found
                if current_fitness > best_fitness:
                    best_bee = bee
                    best_trades_df = current_trades_df
                    best_fitness = current_fitness
                    console.print(f"[bold green]Iteration {iteration + 1}:[/bold green] Best Fitness: {best_fitness:.4f}")
                    

            # Scout bee phase: Randomly explore new areas
            random_bee = random.choice(bees)
            random_bee['weights'] = [random.uniform(weights_range[0], weights_range[1]) for _ in range(len(Environment.signal_columns))]
            random_bee['x'] = random.uniform(x_range[0], x_range[1])
               
            # Update the plot
            if len(fitness_history) > 0:
                line.set_xdata(range(len(fitness_history)))  # Update x data
                line.set_ydata(fitness_history)  # Update y data
                plt.xlim(0, max_iter)  # Set x-axis limit
                plt.ylim(min(fitness_history) - 1, max(fitness_history) + 1)  # Set y-axis limit dynamically
                plt.draw()  # Redraw the plot
                plt.pause(0.1)  # Pause to allow the plot to update


            # Store best fitness of this iteration
            fitness_history.append(best_fitness)

            # Update the progress bar
            progress.update(task, advance=1)
            # console.print(f"[bold green]Iteration {iteration + 1}:[/bold green] Best Fitness: {best_fitness:.4f}")



    # Plot fitness convergence
    # plt.figure(figsize=(10, 6))
    # plt.plot(fitness_history, label='Best Fitness')
    # plt.title('ABC Algorithm Convergence')
    # plt.xlabel('Iterations')
    # plt.ylabel('Fitness Value')
    # plt.legend()
    # plt.grid(True)

    # Save the figure as a PNG file
    plt.savefig('abc_algorithm_convergence.png', dpi=300)  
    
    
    # for bee in bees:
    #     current_fitness, current_trades_df = fitness(bee['weights'], bee['x'], df_strategy, df_data)
    #     print(current_fitness)
    #     if current_fitness > best_fitness:
    #         best_fitness = current_fitness
    #         best_trades_df = current_trades_df
    #         best_bee = bee
 

    return best_bee, best_fitness, best_trades_df
