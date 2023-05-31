import statistics

import pandas as pd
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
# from tmu.models.classification.vanilla_classifier import TMClassifier
from time import time
import gym
from pyTsetlinMachine.tools import Binarizer
from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO, DQN, A2C
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

clauses = 500
s = 2.7
thresh = 0.3
steps = 10000
episodes = 100
splitratio = 0.7

ppo_demo_scores = []
ppo_scores = []
math_scores = []
dqn_scores = []
a2c_scores = []
ppo_norm_scores = []

print_list = [ppo_demo_scores,
              ppo_scores,
              math_scores,
              dqn_scores,
              a2c_scores,
              ppo_norm_scores]

name_list = ["ppo_demo_tsetlin_scores",
             "ppo_tsetlin_scores",
             "math_tsetlin_scores",
             "dqn_scores",
             "a2c_scores",
             "ppo_scores"]

# PPO Demo Tsetlin Agent
# --------------------------------------------------------------------------
checkpoint = load_from_hub(
    repo_id="sb3/demo-hf-CartPole-v1",
    filename="ppo-CartPole-v1.zip",
)
model = PPO.load(checkpoint)
X = np.load("states/statesCartPolePPO.npy")
Y = np.loadtxt("actions/actionsCartPolePPO.txt", dtype=int)
b = Binarizer(max_bits_per_feature=4)
b.fit(X)
X_transformed = b.transform(X)
X_train = X_transformed[:int(len(X) * splitratio)]
X_test = X_transformed[int(len(X) * splitratio):]
Y_train = Y[:int(len(Y) * splitratio)]
Y_test = Y[int(len(Y) * splitratio):]
eval_env = gym.make("CartPole-v1")
score = 0
obs = eval_env.reset()
same = 0
tm = MultiClassTsetlinMachine(clauses, clauses * thresh, s, number_of_state_bits=7)
result = 0
obsTemp = np.array([obs])
state = b.transform(obsTemp)
obsInit = np.array([state[0], state[0]])
predInit = np.array([1, 0])
tm.fit(obsInit, predInit, epochs=0)
for i in range(1):
    start = time()
    for j in range(len(X_train)):
        tm.fit(np.array([X_train[j]]), np.array([Y_train[j]]), epochs=1, incremental=True)
    stop = time()
    result = 100 * (tm.predict(X_test) == Y_test).mean()

for j in range(episodes):  # episodes
    obs = eval_env.reset()
    step = 0
    done = False
    total_rewards_ep = 0
    start_training = time()
    i = 0
    while True:
        obsTemp = np.array([obs])
        a = b.transform(obsTemp)
        action3 = model.predict(obs)
        action = tm.predict(a)
        if action[0] == action3[0]:
            same += 1
        obs, reward, done, info = eval_env.step(action[0])
        if not reward == 1.0:
            print(reward)
        score += 1
        if done:
            obs = eval_env.reset()
            ppo_demo_scores.append(score)
            score = 0
            break

# PPO Tsetlin Agent
# --------------------------------------------------------------------------
checkpoint = load_from_hub(
    repo_id="sb3/ppo-CartPole-v1",
    filename="ppo-CartPole-v1.zip",
)
model = PPO.load(checkpoint)
X = np.load("states/statesCartPolePPOv1.npy")
Y = np.loadtxt("actions/actionsCartPolePPOv1.txt", dtype=int)
b = Binarizer(max_bits_per_feature=4)
b.fit(X)
X_transformed = b.transform(X)
X_train = X_transformed[:int(len(X) * splitratio)]
X_test = X_transformed[int(len(X) * splitratio):]
Y_train = Y[:int(len(Y) * splitratio)]
Y_test = Y[int(len(Y) * splitratio):]
eval_env = gym.make("CartPole-v1")
score = 0
obs = eval_env.reset()
same = 0
tm = MultiClassTsetlinMachine(clauses, clauses * thresh, s, number_of_state_bits=7)
result = 0
obsTemp = np.array([obs])
state = b.transform(obsTemp)
obsInit = np.array([state[0], state[0]])
predInit = np.array([1, 0])
tm.fit(obsInit, predInit, epochs=0)
for i in range(1):
    start = time()
    for j in range(len(X_train)):
        tm.fit(np.array([X_train[j]]), np.array([Y_train[j]]), epochs=1, incremental=True)
    stop = time()
    result = 100 * (tm.predict(X_test) == Y_test).mean()

for j in range(episodes):  # episodes
    obs = eval_env.reset()
    step = 0
    done = False
    total_rewards_ep = 0
    start_training = time()
    i = 0
    while True:
        obsTemp = np.array([obs])
        a = b.transform(obsTemp)
        action3 = model.predict(obs)
        action = tm.predict(a)
        if action[0] == action3[0]:
            same += 1
        obs, reward, done, info = eval_env.step(action[0])
        if not reward == 1.0:
            print(reward)
        score += 1
        if done:
            obs = eval_env.reset()
            ppo_scores.append(score)
            score = 0
            break


# Math Tsetlin Agent
# ------------------------------------------------------------------------------


def theta_omega_policy(obs):
    theta, w = obs[2:4]
    if abs(theta) < 0.03:
        return 0 if w < 0 else 1
    else:
        return 0 if theta < 0 else 1


X = np.load("states/statesCartPoleMath.npy")
Y = np.loadtxt("actions/actionsCartPoleMath.txt", dtype=int)
b = Binarizer(max_bits_per_feature=4)
b.fit(X)
X_transformed = b.transform(X)
X_train = X_transformed[:int(len(X) * splitratio)]
X_test = X_transformed[int(len(X) * splitratio):]
Y_train = Y[:int(len(Y) * splitratio)]
Y_test = Y[int(len(Y) * splitratio):]
score = 0
obs = eval_env.reset()
same = 0
tm = MultiClassTsetlinMachine(clauses, clauses * thresh, s, number_of_state_bits=7)
result = 0
for i in range(1):
    obsTemp = np.array([obs])
    state = b.transform(obsTemp)
    obsInit = np.array([state[0], state[0]])
    predInit = np.array([1, 0])
    tm.fit(obsInit, predInit, epochs=0)
    start = time()
    for j in range(len(X_train)):
        tm.fit(np.array([X_train[j]]), np.array([Y_train[j]]), epochs=1, incremental=True)
    stop = time()
    result = 100 * (tm.predict(X_test) == Y_test).mean()

for j in range(episodes):  # episodes
    obs = eval_env.reset()
    step = 0
    done = False
    total_rewards_ep = 0
    start_training = time()
    i = 0
    while True:
        obsTemp = np.array([obs])
        a = b.transform(obsTemp)
        action2 = theta_omega_policy(obs)
        action = tm.predict(a)
        if action[0] == action2:
            same += 1
        obs, reward, done, info = eval_env.step(action[0])
        if not reward == 1.0:
            print(reward)
        score += 1
        if done:
            obs = eval_env.reset()
            math_scores.append(score)
            score = 0
            break

# DQN
# ------------------------------------------------------------------------------

checkpoint = load_from_hub(
    repo_id="sb3/dqn-CartPole-v1",
    filename="dqn-CartPole-v1.zip",
)
model = DQN.load(checkpoint)
actions = []
states = []
obs = eval_env.reset()
score = 0
same = 0

for j in range(episodes):  # episodes
    obs = eval_env.reset()
    step = 0
    done = False
    total_rewards_ep = 0
    start_training = time()
    i = 0
    while True:
        s = obs.tolist()
        action, _state = model.predict(obs, deterministic=True)
        actions.append(action)
        states.append(s)
        obs, reward, done, info = eval_env.step(action)
        score += 1
        if done:
            obs = eval_env.reset()
            dqn_scores.append(score)
            score = 0
            break

# A2C
# ------------------------------------------------------------------------------

checkpoint = load_from_hub(
    repo_id="sb3/a2c-CartPole-v1",
    filename="a2c-CartPole-v1.zip",
)
model = A2C.load(checkpoint)
actions = []
states = []
obs = eval_env.reset()
score = 0
same = 0

for j in range(episodes):  # episodes
    obs = eval_env.reset()
    step = 0
    done = False
    total_rewards_ep = 0
    start_training = time()
    i = 0
    while True:
        s = obs.tolist()
        action, _state = model.predict(obs, deterministic=True)
        actions.append(action)
        states.append(s)
        obs, reward, done, info = eval_env.step(action)
        score += 1
        if done:
            obs = eval_env.reset()
            a2c_scores.append(score)
            score = 0
            break

# PPO - normal
# ------------------------------------------------------------------------------
checkpoint = load_from_hub(
    repo_id="sb3/ppo-CartPole-v1",
    filename="ppo-CartPole-v1.zip",
)
model = PPO.load(checkpoint)
actions = []
states = []
obs = eval_env.reset()
score = 0
same = 0

for j in range(episodes):  # episodes
    obs = eval_env.reset()
    step = 0
    done = False
    total_rewards_ep = 0
    start_training = time()
    i = 0
    while True:
        s = obs.tolist()
        action, _state = model.predict(obs, deterministic=True)
        actions.append(action)
        states.append(s)
        obs, reward, done, info = eval_env.step(action)
        score += 1
        if done:
            obs = eval_env.reset()
            ppo_norm_scores.append(score)
            score = 0
            break

# Print
# --------------------------------------------------------------------------

for i in range(len(print_list)):
    mean = statistics.mean(print_list[i])
    std = statistics.stdev(print_list[i])
    print(name_list[i], mean, std)
    plt.plot((np.arange(0, episodes, 1)), print_list[i], color='blue')
    plt.title(name_list[i])
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.ylim(0, 510)
    plt.show()
