import statistics

import numpy as np
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
import random
from pyTsetlinMachine.tools import Binarizer
import gym

X = np.load("statesCartPoleV5.npy")
Y = np.loadtxt("actionsCartPoleV5.txt", dtype=int)
print(type(Y))
b = Binarizer(max_bits_per_feature=6)
b.fit(X)
clauses = 3000
s = 3.9
thresh = 0.4
tm = MultiClassTsetlinMachine(clauses, clauses * thresh, s)

eval_env = gym.make("CartPole-v1")
scores = []
score = 0
obs = eval_env.reset()
steps = 10000
explorationRate = 1.0

for i in range(steps):
    expCheck = False
    obsTemp = np.array([obs])
    state = b.transform(obsTemp)
    observation = 0
    if random.uniform(0, 1) <= explorationRate:
        action = random.randint(0, 1)
        obs, reward, done, info = eval_env.step(action)
        expCheck = True
    else:
        action = tm.predict(state)
        obs, reward, done, info = eval_env.step(action[0])
        expCheck = False
    if i % 100 == 0 and i != 0 and explorationRate != 0:
        explorationRate -= 0.01
    elif i == 0:
        obsInit = np.array([obs, obs])
        predInit = np.array([1, 0])
        tm.fit(obsInit, predInit, epochs=1)

    if done:
        if action == 0:
            tm.fit(state, np.array([1]), epochs=1)
        else:
            tm.fit(state, np.array([0]), epochs=1)
        obs = eval_env.reset()
        scores.append(score)
        # print(score)
        score = 0
        # print("DONE")
        # eval_env.render()
    else:
        score += 1
        if expCheck:
            tm.fit(state, np.array([action]), epochs=1)
        else:
            tm.fit(state, np.array([action[0]]), epochs=1)

print("Done with training")
print("Largest reward: ", max(scores))
print("Starting Evaluation")
scores = []
score = 0
for i in range(steps):
    obsTemp = np.array([obs])
    state = b.transform(obsTemp)
    observation = 0
    action = tm.predict(state)
    obs, reward, done, info = eval_env.step(action[0])
    if done:
        obs = eval_env.reset()
        scores.append(score)
        #print(score)
        score = 0
        # print("DONE")
        # eval_env.render()
    else:
        score += 1

std = statistics.stdev(scores)
mean = statistics.mean(scores)

print("mean reward:", mean)
print("std:", std)
print("Largest reward: ", max(scores))
print("Smallest reward: ", min(scores))
