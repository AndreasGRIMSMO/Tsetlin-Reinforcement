import statistics
import typing
import numpy as np
#from pyTsetlinMachine.tm import MultiClassTsetlinMachine
import random
from pyTsetlinMachine.tools import Binarizer
import gym
from tmu.models.classification.vanilla_classifier import TMClassifier
from time import time
import matplotlib.pyplot as plt
X = np.load("states/statesCartPoleV7.npy")
Y = np.loadtxt("actions/actionsCartPoleV7.txt", dtype=int)

b = Binarizer(max_bits_per_feature=6)
b.fit(X)
del (X)
del (Y)
clauses = 3000
s = 3.9
thresh = clauses * 0.3
# tm = MultiClassTsetlinMachine(clauses, thresh, s)
results = []
stds = []

tm = TMClassifier(clauses, thresh, s, incremental=True)


def validation(x1, mytm, myenv):
    scores1 = []
    score1 = 0
    obs = myenv.reset()
    print("starting eval")
    for i in range(100):
        while True:
            score1 += 1
            obsTemp = np.array([obs])
            state = b.transform(obsTemp)
            # print(state)
            # print(type(state))
            action = mytm.predict(state)
            # print(action)
            # print(action[0])
            obs, reward, done, info = myenv.step(action[0])
            if done:
                obs = myenv.reset()
                scores1.append(score1)
                # print(score)
                score1 = 0
                break
                # print("DONE")
                # eval_env.render()
    print("finished eval")

    std = statistics.stdev(scores1)
    mean = statistics.mean(scores1)
    results.append(mean)
    print("Episode number:", x1)
    print("mean reward:", mean)
    print("std:", std)
    # print("Largest reward: ", max(scores))
    # print("Smallest reward: ", min(scores))
eval_env = gym.make("CartPole-v1")
scores = []
score = 0
obs = eval_env.reset()
steps = 100000  # 10000000 is too much
explorationRate = 1.0 # exploration rate in start, 1.0 = 100%
rateChange = 0.01 #how much it is changed
stepChange = steps * rateChange
minExp = 0.1 #Minimum required exploration
batchSize = 20 #Size of batches it is trained on
batchChange = 10
EpsilonGreedy = False #Wheter or not to use the epslion greedy for exploration and exploitation
# initialize tsetlin
obsTemp = np.array([obs])
state = b.transform(obsTemp)
print(state[0])
obsInit = np.array([state[0], state[0]])
print(obsInit)
predInit = np.array([1, 0])
#print("First fit")
tm.fit(obsInit, predInit, epochs=0)
del (obsInit)
del (predInit)
states = []
actions = []
#print("Starting loop")
currentBatchNum = 0
for x in range(1000): #1000 episodes
    obs = eval_env.reset()
    done = False
    if x == 70 or x == 150 or x == 220:
        batchSize += batchChange
    while True:

        # if i % 100 == 0:
        #    print(i)
        # Exploration and exploitation part
        obsTemp = np.array([obs])
        state = b.transform(obsTemp)
        action1 = random.randint(0, 1)
        action = np.array([action1])
        obs, reward, done, info = eval_env.step(action[0])
        actions.append(action[0])
        states.append(state)
        # Learning
        if currentBatchNum >= batchSize:
            tm.fit(np.array(states), np.array(actions), epochs=1)
            currentBatchNum = 0
            actions = []
            states = []

        if done:
            obs = eval_env.reset()
            scores.append(score)
            if score >= 499:
                tm.fit(np.array(states), np.array(actions), epochs=1)
                currentBatchNum = 0
                actions = []
                states = []
            else:
                currentBatchNum = 0
                actions = []
                states = []
            score = 0
            validation(x, tm, eval_env)
            break

        else:
            score += 1
            currentBatchNum += 1
            # tm.fit(state, np.array([action[0]]), epochs=1)

#print("Done with training")
#print("Largest reward: ", max(scores))
#print("Starting Evaluation")

plt.plot(results)
plt.ylabel('reward')
plt.xlabel("reward after training x episodes")
plt.show()

