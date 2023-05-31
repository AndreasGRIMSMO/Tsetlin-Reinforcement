import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import gym
#import gym_pygame
from huggingface_sb3 import load_from_hub
# from stable_baselines3 import Reinforce

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

device = torch.device("cpu")

class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size * 2)
        self.fc3 = nn.Linear(h_size * 2, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


env_id = "Pong-PLE-v0"
env = gym.make(env_id)
eval_env = gym.make(env_id)
s_size = env.observation_space.shape[0]
a_size = env.action_space.n


model = Policy(s_size, a_size, 64)
model.load_state_dict(torch.load("PongModel/model.pt"))


env_id = "Pong-PLE-v0"
env = gym.make(env_id)
eval_env = gym.make(env_id)
s_size = env.observation_space.shape[0]
a_size = env.action_space.n

episode_rewards = []
for episode in range(10000):
    state = env.reset()
    step = 0
    done = False
    total_rewards_ep = 0

    for step in range(1000):
        action, _ = model.act(state)
        new_state, reward, done, info = env.step(action)
        total_rewards_ep += reward

        if done:
            break
        state = new_state
    episode_rewards.append(total_rewards_ep)
mean_reward = np.mean(episode_rewards)
std_reward = np.std(episode_rewards)


print(mean_reward)
print(std_reward)