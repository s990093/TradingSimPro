import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from rich.progress import Progress
from rich.console import Console
from .calculate_returns import calculate_trading_signals

console = Console()  # Initialize rich console
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

console.print(device)

# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # Discount rate
        self.epsilon = 1.0   # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Initialize lists to track metrics
        self.losses = []
        self.rewards = []
        self.epsilon_history = []

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model(torch.FloatTensor(state))
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        total_loss = 0
        total_reward = 0
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(torch.FloatTensor(next_state))).item()
            target_f = self.model(torch.FloatTensor(state))
            target_f[action] = target
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, torch.FloatTensor(target_f))
            loss.backward()
            self.optimizer.step()
            
            
        
        # Record metrics
        self.losses.append(total_loss / batch_size)  # Average loss
        self.rewards.append(total_reward / batch_size)  # Average reward
        self.epsilon_history.append(self.epsilon)  # Track epsilon
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

# Define the fitness function as reward
def fitness(weights, buy_threshold, sell_threshold, df_strategy, df_data, signal_columns):  
    return calculate_trading_signals(df_strategy, weights, buy_threshold, sell_threshold, signal_columns, df_data)


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
    
def dqn_algorithm(df_strategy, df_data, episodes, weights_range, x_range, signal_columns):
    state_size = len(signal_columns) + 2  # Include weights, buy_threshold, and sell_threshold
    action_size = state_size  # Each component of the state can be an action
    agent = DQNAgent(state_size, action_size)
    batch_size = 128
    best_fitness = -float('inf')
    fitness_history = []
    best_trades_df = None
    best_bee = None
    max_steps = 3000  # Example maximum steps per episode
    best_reward = None
    step_count = 0  # Initialize step count


    with Progress() as progress:
        task = progress.add_task("Running DQN Algorithm...", total=episodes)

        for e in range(episodes):
            # Initialize state (e.g., random weights, buy_threshold, sell_threshold)
            state = np.concatenate(([random.uniform(weights_range[0], weights_range[1]) for _ in range(len(signal_columns))],
                                    [random.uniform(x_range[0], x_range[1]), random.uniform(x_range[0], x_range[1])]))
            done = False
            total_reward = 0
            step_count  = 0 
            while not done and step_count < max_steps:
                action = agent.act(state)

                # Take the action: Modify the weights, buy_threshold, sell_threshold
                new_state = state.copy()
                if action < len(signal_columns):
                    new_state[action] = random.uniform(weights_range[0], weights_range[1])
                elif action == len(signal_columns):
                    new_state[-2] = random.uniform(x_range[0], x_range[1])
                else:
                    new_state[-1] = random.uniform(x_range[0], x_range[1])

                # Calculate the reward (fitness function)
                reward, current_trades_df = fitness(new_state[:-2], new_state[-2], new_state[-1], df_strategy, df_data, signal_columns)
                total_reward += reward

                # Check if the episode is done (e.g., based on fitness threshold or max steps)
                done = reward > best_fitness

                # Remember the experience
                agent.remember(state, action, reward, new_state, done)
                state = new_state

                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)
                
                step_count += 1  # Increment step count
                
            if step_count >= max_steps:
                print(f"[Warning] Episode {e + 1} exceeded maximum steps: {max_steps}. Total Reward: {total_reward:.4f}")


            # Log the best fitness
            if total_reward > best_fitness:
                best_fitness = total_reward
                best_reward = reward
                best_bee = state
                best_trades_df = current_trades_df
                console.print(f"[bold green]Episode {e + 1}:[/bold green] Best Fitness: {best_reward:.4f}")
                fitness_history.append(best_fitness)

            progress.update(task, advance=1)
            
        
    
    plot_dqn_convergence(fitness_history, episodes)
    
    # ???
    # plot_metrics(agent)

    return best_bee, best_fitness, best_trades_df