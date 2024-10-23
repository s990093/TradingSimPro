from matplotlib import pyplot as plt


def plot_dqn_convergence(fitness_history, max_iter):
    plt.ion()  # Turn on interactive mode
    plt.figure(figsize=(10, 6))
    plt.title('DQN Algorithm Convergence')
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
    plt.savefig('res/dqn_algorithm_convergence.png', dpi=300)

    # Manually close the plot
    plt.ioff()  # Turn off interactive mode after plotting
    plt.show()  # Wait for manual plot closure


def plot_metrics(agent):
    # Plot loss
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.plot(agent.losses)
    plt.title('Loss Over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')

    # Plot rewards
    plt.subplot(1, 3, 2)
    plt.plot(agent.rewards)
    plt.title('Rewards Over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Average Reward')

    # Plot epsilon
    plt.subplot(1, 3, 3)
    plt.plot(agent.epsilon_history)
    plt.title('Epsilon Decay Over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Epsilon')

    plt.tight_layout()
    plt.show()