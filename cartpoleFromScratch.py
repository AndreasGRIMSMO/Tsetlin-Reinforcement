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
del(X)
del(Y)
clauses = 500
s = 3.9
thresh = 0.4
tm = MultiClassTsetlinMachine(clauses, clauses * thresh, s)

eval_env = gym.make("CartPole-v1")
scores = []
score = 0
obs = eval_env.reset()
steps = 10000 #10000000 is too much
explorationRate = 1.0
rateChange = 0.01
stepChange = steps * rateChange

# initialize tsetlin
obsTemp = np.array([obs])
state = b.transform(obsTemp)
obsInit = np.array([state[0], state[0]])
predInit = np.array([1, 0])
print("First fit")
tm.fit(obsInit, predInit, epochs=0)
del(obsInit)
del(predInit)
states = []
actions = []
print("Starting loop")
for i in range(steps):
    #if i % 100 == 0:
    #    print(i)
    # Exploration and exploitation part
    obsTemp = np.array([obs])
    firstAngle = abs(obs[2])
    state = b.transform(obsTemp)
    if random.uniform(0, 1) <= explorationRate:
        action1 = random.randint(0, 1)
        action = np.array([action1])
        obs, reward, done, info = eval_env.step(action[0])
    else:
        action = tm.predict(state)
        obs, reward, done, info = eval_env.step(action[0])
    if i % stepChange == 0 and i != 0 and explorationRate <= 0.1:
        explorationRate -= rateChange
    # elif i == 0:
    #    obsInit = np.array([obs, obs])
    #    predInit = np.array([1, 0])
    #    tm.fit(obsInit, predInit, epochs=1)
    # Learning
    secondAngle = abs(obs[2])
    # print(obs[2])
    #if secondAngle >= 0.05 or firstAngle - secondAngle > 0:
    states.append(state)
    if firstAngle - secondAngle > 0:
        tm.fit(state, np.array([action[0]]), epochs=1,incremental=True)
        #actions.append(action[0])
    else:
        if action[0] == 0:
            tm.fit(state, np.array([1]), epochs=1,incremental=True)
            #actions.append(1)
        else:
            tm.fit(state, np.array([0]), epochs=1,incremental=True)
            #actions.append(0)

    if done:
        #tm.fit(np.array(states), np.array(actions), epochs=1)
        # if action[0] == 0:
        #    tm.fit(state, np.array([1]), epochs=1)
        # else:
        #    tm.fit(state, np.array([0]), epochs=1)
        obs = eval_env.reset()
        scores.append(score)
        # print(score)
        score = 0
        # print("DONE")
        # eval_env.render()
    else:
        score += 1
        # tm.fit(state, np.array([action[0]]), epochs=1)

print("Done with training")
print("Largest reward: ", max(scores))
print("Starting Evaluation")
scores = []
score = 0
obs = eval_env.reset()
for i in range(10000):
    obsTemp = np.array([obs])
    state = b.transform(obsTemp)
    action = tm.predict(state)
    obs, reward, done, info = eval_env.step(action[0])
    if done:
        obs = eval_env.reset()
        scores.append(score)
        # print(score)
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
