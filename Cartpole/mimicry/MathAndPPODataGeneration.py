import statistics
from time import time
import matplotlib.pyplot as plt
import numpy as np
import gym
#import pyglet
from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO, DQN


# Retrieve the model from the hub
## repo_id = id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name})
## filename = name of the model zip file from the repository
def theta_omega_policy(obs):
    theta, w = obs[2:4]
    if abs(theta) < 0.03:
        return 0 if w < 0 else 1
    else:
        return 0 if theta < 0 else 1

# Evaluate the agent and watch it
eval_env = gym.make("CartPole-v1")
checkpoint = load_from_hub(
    repo_id="sb3/ppo-CartPole-v1",
    filename="ppo-CartPole-v1.zip",
)
model = PPO.load(checkpoint)

actions = []
states = []
obs = eval_env.reset()
score = 0
math_scores = []
ppo_scores = []
episodes = 100

for j in range(episodes):  # episodes
    obs = eval_env.reset()
    step = 0
    done = False
    total_rewards_ep = 0
    start_training = time()
    i = 0
    while True:
        s = obs.tolist()
        #action = theta_omega_policy(obs)
        action, _state = model.predict(obs, deterministic=True)
        #b = []
        actions.append(action)
        states.append(s)
        #plusMinus.append(b)
        obs, reward, done, info = eval_env.step(action)
        score += 1
        if done:
            obs = eval_env.reset()
            ppo_scores.append(score)
            score = 0
            #eval_env.render()
            break

ppo_std = statistics.stdev(ppo_scores)
ppo_mean = statistics.mean(ppo_scores)
score = 0
'''
for j in range(episodes):  # episodes
    obs = eval_env.reset()
    step = 0
    done = False
    total_rewards_ep = 0
    start_training = time()
    i = 0
    while True:
        s = obs.tolist()
        action = theta_omega_policy(obs)
        #action, _state = model.predict(obs, deterministic=True)
        #b = []
        actions.append(action)
        states.append(s)
        #plusMinus.append(b)
        obs, reward, done, info = eval_env.step(action)
        score += 1
        if done:
            obs = eval_env.reset()
            math_scores.append(score)
            score = 0
            #eval_env.render()
            break

math_std = statistics.stdev(math_scores)
math_mean = statistics.mean(math_scores)

for j in range(episodes):  # episodes
    obs = eval_env.reset()
    step = 0
    done = False
    total_rewards_ep = 0
    start_training = time()
    i = 0
    while True:
        s = obs.tolist()
        #action = theta_omega_policy(obs)
        action, _state = model.predict(obs, deterministic=True)
        #b = []
        actions.append(action)
        states.append(s)
        #plusMinus.append(b)
        obs, reward, done, info = eval_env.step(action)
        score += 1
        if done:
            obs = eval_env.reset()
            scores.append(score)
            score = 0
            #eval_env.render()
            break
'''

#plt.plot((np.arange(0, episodes, 1)), ppo_scores, color='r', label='ppo')
#plt.plot((np.arange(0, episodes, 1)), math_scores, color='b', label='math')
#plt.show()

print(ppo_mean, ppo_std)
#print(math_mean, math_std)

newArray = np.array(states)

with open('states/statesCartPolePPOv1.npy', 'wb') as f:
    np.save(f, newArray)

with open('actions/actionsCartPolePPOv1.txt', 'w') as f:
    for line in actions:
        f.write(f"{line}\n")
