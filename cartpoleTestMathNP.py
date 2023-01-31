import statistics
import numpy as np
import gym
#import pyglet
from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
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
    repo_id="sb3/demo-hf-CartPole-v1",
    filename="ppo-CartPole-v1.zip",
)
model = PPO.load(checkpoint)

actions = []
states = []
obs = eval_env.reset()
score = 0
scores = []
for i in range(100000):
    if i%200 == 0:
        print(i)
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

std = statistics.stdev(scores)
mean = statistics.mean(scores)

for i in range(100000):
    if i%200 == 0:
        print(i)
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
        scores.append(score)
        score = 0
        #eval_env.render()

print(mean)
print(std)


newArray = np.array(states)

with open('statesCartPoleV5.npy', 'wb') as f:
    np.save(f, newArray)

with open('actionsCartPoleV5.txt', 'w') as f:
    for line in actions:
        f.write(f"{line}\n")
