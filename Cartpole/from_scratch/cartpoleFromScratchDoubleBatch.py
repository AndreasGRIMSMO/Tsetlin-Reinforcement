import statistics

import numpy as np
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
import random
from pyTsetlinMachine.tools import Binarizer
import gym
from tmu.models.classification.vanilla_classifier import TMClassifier

X = np.load("states/statesCartPoleRandom.npy")
Y = np.loadtxt("actions/actionsCartPoleRandom.txt", dtype=int)
print(type(Y))
b = Binarizer(max_bits_per_feature=10)
b.fit(X)
del (X)
del (Y)
clauses = 500
s = 3.9
thresh = clauses * 0.3
results = []
# tm = MultiClassTsetlinMachine(clauses, thresh, s)
tmA = TMClassifier(clauses, thresh, s, incremental=True)
tmB = TMClassifier(clauses, thresh, s, incremental=True)

eval_env = gym.make("CartPole-v1")
scores = []
score = 0
obs = eval_env.reset()
steps = 100000  # 10000000 is too much
explorationRate = 1.0 # exploration rate in start, 1.0 = 100%
rateChange = 0.01 #how much it is changed
stepChange = steps * rateChange
minExp = 0.1 #Minimum required exploration
batchSize = 10 #Size of batches it is trained on
batchChange = 20
EpsilonGreedy = True #Wherter or not to use the epslion greedy for exploration and exploitation
# initialize tsetlin
obsTemp = np.array([obs])
state = b.transform(obsTemp)
obsInit = np.array([state[0], state[0]])
predInit = np.array([1, 0])
print("First fit")
tmA.fit(obsInit, predInit, epochs=0)
tmB.fit(obsInit, predInit, epochs=0)
del (obsInit)
del (predInit)
states = []
actions = []
print("Starting loop")
switch = 100
currentBatchNum = 0
for i in range(steps):
    if i % switch == 0:
        tmA = tmB

    if i == steps/2 or i == steps*3/4 or i == steps/4:
        batchSize += batchChange
    # if i % 100 == 0:
    #    print(i)
    # Exploration and exploitation part
    obsTemp = np.array([obs])
    firstAngle = abs(obs[2])
    state = b.transform(obsTemp)
    if random.uniform(0, 1) <= explorationRate:
        action1 = random.randint(0, 1)
        action = np.array([action1])
        obs, reward, done, info = eval_env.step(action[0])
        actions.append(action[0])
    else:
        action = tmA.predict(state)
        actions.append(action[0])
        obs, reward, done, info = eval_env.step(action[0])
    if i % stepChange == 0 and i != 0 and EpsilonGreedy:
        explorationRate -= rateChange
        if explorationRate < minExp:
            explorationRate = minExp
    states.append(state)
    # Learning
    secondAngle = abs(obs[2])

    if currentBatchNum == batchSize:
        tmB.fit(np.array(states), np.array(actions), epochs=1)
        currentBatchNum = 0
        actions = []
        states = []

    if done:
        obs = eval_env.reset()
        scores.append(score)
        if score >= 499:
            tmB.fit(np.array(states), np.array(actions), epochs=1)
        else:
            currentBatchNum = 0
            actions = []
            states = []
        # print(score)
        score = 0
        # print("DONE")
        # eval_env.render()

    else:
        score += 1
        currentBatchNum += 1
        # tmB.fit(state, np.array([action[0]]), epochs=1)

print("Done with training")
print("Largest reward: ", max(scores))
print("Starting Evaluation")
scores = []
score = 0
obs = eval_env.reset()
tmA = tmB
for i in range(10000):
    score += 1
    obsTemp = np.array([obs])
    state = b.transform(obsTemp)
    action = tmA.predict(state)
    obs, reward, done, info = eval_env.step(action[0])
    if done:
        obs = eval_env.reset()
        scores.append(score)
        # print(score)
        score = 0
        # print("DONE")
        # eval_env.render()
    #else:


std = statistics.stdev(scores)
mean = statistics.mean(scores)

print("mean reward:", mean)
print("std:", std)
print("Largest reward: ", max(scores))
print("Smallest reward: ", min(scores))
