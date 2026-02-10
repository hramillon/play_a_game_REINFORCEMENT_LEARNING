import gym
import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

env = gym.make("FrozenLake-v1", is_slippery=True)

n_states = env.observation_space.n
n_actions = env.action_space.n

gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

# Réseau Q
model = models.Sequential([
    layers.Input(shape=(n_states,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(n_actions)
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss="mse"
)

def one_hot(s):
    x = np.zeros(n_states)
    x[s] = 1
    return x

episodes = 2000
success = 0

for ep in range(episodes):
    s = env.reset()[0]
    done = False

    while not done:
        state = one_hot(s)

        # ε-greedy
        if np.random.rand() < epsilon:
            a = env.action_space.sample()
        else:
            q_vals = model.predict(state.reshape(1, -1), verbose=0)
            a = np.argmax(q_vals)

        s1, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated


        target = model.predict(state.reshape(1, -1), verbose=0)
        next_q = model.predict(one_hot(s1).reshape(1, -1), verbose=0)

        target[0, a] = r + gamma * np.max(next_q) * (1 - done)

        model.train_on_batch(state.reshape(1, -1), target)

        s = s1

    if r == 1:
        success += 1

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

print("Taux de réussite :", success / episodes)
