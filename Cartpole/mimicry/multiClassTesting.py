import pandas as pd
from pyTsetlinMachine.tm import MultiClassTsetlinMachine, RegressionTsetlinMachine
#from tmu.models.classification.vanilla_classifier import TMClassifier
from time import time
import gym
import statistics
from pyTsetlinMachine.tools import Binarizer
from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import numpy as np
# Import seaborn
import seaborn as sns

# Retrieve the model from the hub
## repo_id = id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name})
## filename = name of the model zip file from the repository
checkpoint = load_from_hub(
    repo_id="sb3/demo-hf-CartPole-v1",
    filename="ppo-CartPole-v1.zip",
)
model = PPO.load(checkpoint)

X = np.load("states/statesCartPolePPO.npy")
Y = np.loadtxt("actions/actionsCartPolePPO.txt", dtype=int)
splitratio = 0.7
b = Binarizer(max_bits_per_feature=4)
b.fit(X)
X_transformed = b.transform(X)
X_train = X_transformed[:int(len(X) * splitratio)]
X_test = X_transformed[int(len(X) * splitratio):]
Y_train = Y[:int(len(Y) * splitratio)]
Y_test = Y[int(len(Y) * splitratio):]
clauses = 500
s = 2.7
thresh = 0.3


def theta_omega_policy(obs):
    theta, w = obs[2:4]
    if abs(theta) < 0.03:
        return 0 if w < 0 else 1
    else:
        return 0 if theta < 0 else 1


eval_env = gym.make("CartPole-v1")

ppo_scores = []
math_scores = []
dqn_scores = []
score = 0
obs = eval_env.reset()
same = 0
steps = 10000
episodes = 100
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
    # plotArray = np.append(plotArray, result)
    print("#%d Accuracy: %.2f%% (%.2fs)" % (1, result, stop - start))
    # with open('tsetlinAnimals1', 'wb') as tsetlin_file:
    #        pickle.dump(tm, tsetlin_file)


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
        #action2 = theta_omega_policy(obs)
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

mean = statistics.mean(ppo_scores)
std = statistics.stdev(ppo_scores)
print(mean, std)


'''
X = np.load("states/statesCartPoleMath.npy")
Y = np.loadtxt("actions/actionsCartPoleMath.txt", dtype=int)
splitratio = 0.7
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
        tm.fit(np.array([X_train[j]]), np.array([Y_train[j]]), epochs=1,incremental=True)
    stop = time()
    result = 100 * (tm.predict(X_test) == Y_test).mean()
    # plotArray = np.append(plotArray, result)
    print("#%d Accuracy: %.2f%% (%.2fs)" % (1, result, stop - start))
    # with open('tsetlinAnimals1', 'wb') as tsetlin_file:
    #        pickle.dump(tm, tsetlin_file)

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
        #action3 = model.predict(obs)
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
'''
'''
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


plot = pd.DataFrame({"Tsetlin-PPO": ppo_scores, "Tsetlin-Math": math_scores, "DQN": dqn_scores})
graph = sns.lineplot(data=plot, errorbar='sd')
graph.legend(labels=["Tsetlin-PPO", "Tsetlin-Math", "DQN"])
graph.legend(title=['Models'])
graph.legend(loc='upper left')
graph.set(xlabel='Episode', ylabel='Reward')
graph.margins(y=0.40)
plt.show()
'''