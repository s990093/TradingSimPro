import os
import random
import numpy as np
from rich.progress import Progress
from rich.console import Console
import multiprocessing as mp

from utility.tool import log_gpu_usage, get_local_device
from utility.DQNAgent import DQNAgent
from utility.plot_dqn_convergence import plot_dqn_convergence
from .calculate_returns import calculate_trading_signals
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

import platform
console = Console()  # Initialize rich console




# Define the fitness function as reward
def fitness(weights, buy_threshold, sell_threshold, df_strategy, df_data, signal_columns):  
    return calculate_trading_signals(df_strategy, weights, buy_threshold, sell_threshold, signal_columns, df_data)


def dqn_algorithm(df_strategy, df_data, episodes, weights_range, x_range, signal_columns):
    state_size = len(signal_columns) + 2  # Include weights, buy_threshold, and sell_threshold
    device = get_local_device()
    action_size = state_size  # Each component of the state can be an action
    agent = DQNAgent(device, state_size, action_size)
    batch_size = 128 * 2
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
                    loass = agent.replay(batch_size)
                    writer.add_scalar('Loss/episode', loass , step_count)  # Log the loss in TensorBoard
                    
                step_count += 1 
                
            if step_count >= max_steps:
                print(f"[Warning] Episode {e + 1} exceeded maximum steps: {max_steps}. Total Reward: {total_reward:.4f}")
                
            writer.add_scalar('Reward/episode', total_reward, e)


            # Log the best fitness
            if total_reward > best_fitness:
                best_fitness = total_reward
                best_reward = reward
                best_bee = state
                best_trades_df = current_trades_df
                console.print(f"[bold green]Episode {e + 1}:[/bold green] Best Fitness: {best_reward:.4f}")
                fitness_history.append(best_fitness)

            progress.update(task, advance=1)
            
        
    
    writer.close() 
    plot_dqn_convergence(fitness_history, episodes)
  
    

    return best_bee, best_fitness, best_trades_df