import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses INFO and WARNING messages from TensorFlow

import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from collections import deque

# Set environment variable to turn off oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Define the environment
class MontyHallEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.car_door = random.randint(0, 2)
        self.player_pick = random.randint(0, 2)
        self.revealed_door = self._reveal_door()
        return self._get_state()

    def _reveal_door(self):
        doors = [0, 1, 2]
        doors.remove(self.car_door)
        if self.player_pick in doors:
            doors.remove(self.player_pick)
        return random.choice(doors)

    def _get_state(self):
        return (self.player_pick, self.revealed_door)

    def step(self, action):
        # action: 0 = stick, 1 = switch
        if action == 1:  # switch
            self.player_pick = 3 - self.player_pick - self.revealed_door
        reward = 1 if self.player_pick == self.car_door else 0
        return self._get_state(), reward

# Neural Network model for Q-learning
def create_model():
    model = Sequential()
    model.add(Input(shape=(2,)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(2, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Parameters
num_episodes_input = input("Set episode number? (y/n): ")
if num_episodes_input == 'n':
    num_episodes = 1000
elif num_episodes_input == 'y':
    num_episodes = int(input("Enter the episode number: "))

gamma_input = input("Set gamma (discount factor)? (y/n): ")
if gamma_input == 'n':
    gamma = 0.95
elif gamma_input == 'y':
    gamma = float(input("Enter the gamma rate: "))

epsilon_input = input("Set epsilon (exploration rate)? (y/n): ")
if epsilon_input == 'n':
    epsilon = 1.0
elif epsilon_input == 'y':
    epsilon = float(input("Enter the epsilon rate: "))

epsilon_min = 0.01
epsilon_decay = 0.995

batch_size_input = input("Set batch size? (y/n): ")
if batch_size_input == 'n':
    batch_size = 32  # Default value for batch size if 'n'
elif batch_size_input == 'y':
    batch_size = int(input("Enter the batch size: "))

memory = deque(maxlen=2000)

# Initialize the environment and model
env = MontyHallEnv()
model = create_model()

# Experience replay buffer
for episode in range(num_episodes):
    state = env.reset()
    state = np.array(state).reshape(1, -1)
    done = False

    while not done:
        if np.random.rand() <= epsilon:
            action = random.randint(0, 1)  # Explore: select a random action
        else:
            q_values = model.predict(state)
            action = np.argmax(q_values[0])  # Exploit: select the action with max Q-value

        next_state, reward = env.step(action)
        next_state = np.array(next_state).reshape(1, -1)
        memory.append((state, action, reward, next_state))

        state = next_state
        done = True  # In Monty Hall, the game ends after one action

        # Experience replay
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
            for s, a, r, ns in minibatch:
                target = r
                if not done:
                    target = r + gamma * np.amax(model.predict(ns)[0])
                target_f = model.predict(s)
                target_f[0][a] = target
                model.fit(s, target_f, epochs=1, verbose=0)

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

# Evaluation
stick_wins = 0
switch_wins = 0
num_evaluations = 1000

for _ in range(num_evaluations):
    state = env.reset()
    state = np.array(state).reshape(1, -1)
    q_values = model.predict(state)
    action = np.argmax(q_values[0])  # Choose the best action based on learned Q-values
    _, reward = env.step(action)
    if action == 0:
        stick_wins += reward
    else:
        switch_wins += reward

print(f"Wins by sticking: {stick_wins} ({stick_wins / num_evaluations * 100}%)")
print(f"Wins by switching: {switch_wins} ({switch_wins / num_evaluations * 100}%)")

