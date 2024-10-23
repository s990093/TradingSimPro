from utility.QNetwork import QNetwork


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


import random
from collections import deque


class DQNAgent:
    def __init__(self, device, state_size, action_size):
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        # Initialize lists to track metrics
        self.losses = []
        self.rewards = []
        self.epsilon_history = []

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Ensure state is on the GPU
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():  # Avoid computing gradients for action selection
            act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        total_loss = 0
        total_reward = 0

        # Sample a random batch of experiences
        minibatch = random.sample(self.memory, batch_size)

        # Prepare the batch
        states = torch.FloatTensor([sample[0] for sample in minibatch]).to(self.device)
        actions = torch.LongTensor([sample[1] for sample in minibatch]).to(self.device)
        rewards = torch.FloatTensor([sample[2] for sample in minibatch]).to(self.device)
        next_states = torch.FloatTensor([sample[3] for sample in minibatch]).to(self.device)
        dones = torch.FloatTensor([sample[4] for sample in minibatch]).to(self.device)

        # Predict Q-values for the current states
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Predict Q-values for the next states
        next_q_values = self.model(next_states).max(1)[0]

        # Compute target Q-values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = self.criterion(current_q_values, target_q_values.detach())

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Record metrics
        total_loss += loss.item()
        total_reward += rewards.mean().item()
        self.losses.append(total_loss / batch_size)
        self.rewards.append(total_reward)
        self.epsilon_history.append(self.epsilon)

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return total_loss