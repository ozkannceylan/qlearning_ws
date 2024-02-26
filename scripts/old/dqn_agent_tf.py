
# import modified_tensorboard
from tools.modified_tensorboard import ModifiedTensorBoard

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
import numpy as np
import random
import os

from collections import deque
import time
from tqdm import tqdm

# Training settings
REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 100
MODEL_NAME = "256x2"
MINIBATCH_SIZE = 64
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

class DQNAgent:
	def __init__(self):

		# Main network for fitting / training
		self.model = self.create_model()

		# Target network for predicting
		self.target_model = self.create_model()
		self.target_model.set_weights(self.model.get_weights())

		self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
		self.tensorboard = ModifiedTensorBoard(
			log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
		self.target_update_counter = 0


	def create_model(self):
		model = Sequential()
		hidden_units=[64, 64]
		model.add(Dense(256, input_dim=4, activation="relu"))
		# model.add(Dropout(0.2))
		model.add(Dense(256, activation="relu"))
		# model.add(Dropout(0.2))
		
		
		model.add(Dense(3, activation="linear"))
		
		model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse", metrics=["accuracy"])
		return model

	# Adds step's data to a memory replay array
	def update_replay_memory(self, transition):
		self.transition = transition
		self.replay_memory.append(transition)

	# Queries main network for Q values given current observation space (environment state)
	def get_qs(self, state):
		state= np.array(state).reshape(1, len(state))
		qs= self.model.predict(state)[0]
		return qs

	# Trains main network every step during episode
	def train(self, terminal_state, step):
		# print("aaaaaaa")
		# print(self.replay_memory)
		# Start training only if certain number of samples is already saved
		if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
			return

		# Get a minibatch of random samples from memory replay array
		minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
		# print(MINIBATCH_SIZE)
		# print(minibatch)

		# Get current states from minibatch, then query NN model for Q values
		current_states = np.array([self.transition[0] for self.transition in minibatch])
		# print(current_states)
		current_qs_list = self.model.predict(current_states)
		# print("current_qs_list: ", current_qs_list)

		# Get future states from minibatch, then query NN model for Q values
		new_current_states = np.array([self.transition[3] for self.transition in minibatch])
		future_qs_list = self.target_model.predict(new_current_states)

		X = []
		y = []

		# Enumerate our batches
		for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
			
			# If not a terminal state, get new q from future states, otherwise set it to 0
			if not done:
				max_future_q = np.max(future_qs_list[index])
				new_q = reward + DISCOUNT * max_future_q
			else:
				new_q = reward

			# Update Q value for given state
			current_qs = current_qs_list[index]
			current_qs[action] = new_q

		    # Append to our training data
			X.append(current_state)
			y.append(current_qs)

		# Fit on all samples as one batch, log only on terminal state
		self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE,
			verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

		# Update target network counter every episode
		if terminal_state:
			self.target_update_counter += 1

		# If counter reaches set value, update target network with weights of main network
		if self.target_update_counter > UPDATE_TARGET_EVERY:
			self.target_model.set_weights(self.model.get_weights())
			self.target_update_counter = 0

# env = BlobEnv()

# ep_rewards = [-200]

# if not os.path.isdir('models'):
#     os.makedirs('models')

# agent = DQNAgent()

# for episode in tqdm(range(1, EPISODES + 1), unit= "episode"):
# 	agent.tensorboard.step = episode

# 	episode_reward = 0
# 	step = 1
# 	current_state = env.reset()

# 	done = False

# 	while not done:
# 		if np.random.random() > epsilon:
# 			action = np.argmax(agent.get_qs(current_state))
# 		else:
# 			action = np.random.randint(0, env.ACTION_SPACE_SIZE)

# 		new_state, reward, done = env.step(action)

# 		episode_reward += reward

# 		if SHOW_PREVIEW and episode % STATS_EVERY == 0:
# 			env.render()

# 		agent.update_replay_memory((current_state, action, reward, new_state, done))
# 		agent.train(done, step)

# 		current_state = new_state
# 		step += 1

# 	ep_rewards.append(episode_reward)
# 	if episode % STATS_EVERY == 0 or episode == 1:
# 		recent_eps = ep_rewards[-STATS_EVERY:]
# 		avg_reward = sum(recent_eps) / len(recent_eps)
# 		min_reward = min(recent_eps)
# 		max_reward = max(recent_eps)
# 		agent.tensorboard.update_stats(reward_avg=avg_reward, reward_min=min_reward, reward_max=max_reward)

# 		if avg_reward >= MIN_REWARD:
# 			agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{avg_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

# 	if epsilon > MIN_EPSILON:
# 		epsilon *= EPSILON_DECAY
# 		epsilon = max(MIN_EPSILON, epsilon)