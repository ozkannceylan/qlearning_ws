import numpy as np
import random
import os

from collections import deque
import time
from tqdm import tqdm

# Training settings
REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 1000
MODEL_NAME = "256x2"
MINIBATCH_SIZE = 4
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5
MIN_REWARD = -200

# Environment settings
EPISODES = 20000

# Exploration settings
epsilon = 1
EPSILON_DECAY = 0.99975
MIN_EPSILON = 1e-3

# Stats settings
STATS_EVERY = 50 # episodes
SHOW_PREVIEW = False

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim

class PyTorchModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, activation_type):
        super(PyTorchModel, self).__init__()
        layers = []

        # Input layer
        if len(hidden_layers) == 0:
            layers.append(nn.Linear(input_size, output_size))
        else:
            layers.append(nn.Linear(input_size, hidden_layers[0]))
            layers.append(self.get_activation(activation_type))

            # Hidden layers
            for index in range(1, len(hidden_layers)):
                layers.append(nn.Linear(hidden_layers[index - 1], hidden_layers[index]))
                layers.append(self.get_activation(activation_type))

            # Output layer
            layers.append(nn.Linear(hidden_layers[-1], output_size))

        # ModuleList is a holder for Modules that can be indexed like a regular Python list
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_activation(self, activation_type):
        if activation_type == "LeakyReLU":
            return nn.LeakyReLU(0.01)
        elif activation_type == "ReLU":
            return nn.ReLU()
        else:
            # Default to linear activation if unknown type
            return nn.Identity()

# def create_pytorch_model(input_size, output_size, hidden_layers, activation_type, learning_rate):
#     model = PyTorchModel(input_size, output_size, hidden_layers, activation_type)
#     optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9, eps=1e-06)
#     return model






class DQNAgent:
    def __init__(self, state_size=4, action_size=3, replay_memory_size=50000, min_replay_memory_size=1000, minibatch_size=64, discount_factor=0.99, update_target_every=5):
        self.state_size = state_size
        self.action_size = action_size

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.load_state_dict(self.model.state_dict())

        self.replay_memory = deque(maxlen=replay_memory_size)
        self.min_replay_memory_size = min_replay_memory_size
        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor
        self.update_target_every = update_target_every
        self.target_update_counter = 0
        

    def create_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.action_size)
        )
        self.optimizer = optim.RMSprop(model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        return model


    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        with torch.no_grad():
            return self.model(torch.Tensor(state).unsqueeze(0)).detach().numpy()[0]

    def train(self, terminal_state, step):
        if len(self.replay_memory) < self.min_replay_memory_size:
            return

        minibatch = random.sample(self.replay_memory, self.minibatch_size)

        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model(torch.Tensor(current_states))

        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model(torch.Tensor(new_current_states))

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = torch.max(future_qs_list[index])
                new_q = reward + self.discount_factor * max_future_q.item()
            else:
                new_q = reward

            current_qs = current_qs_list[index].clone()
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs.tolist())

        self.optimizer.zero_grad()
        output = self.model(torch.Tensor(np.array(X)))
        loss = self.criterion(output, torch.Tensor(np.array(y)))
        loss.backward()
        self.optimizer.step()

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > self.update_target_every:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0
