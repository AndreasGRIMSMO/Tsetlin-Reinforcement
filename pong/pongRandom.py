import statistics
import typing
import numpy as np
# from pyTsetlinMachine.tm import MultiClassTsetlinMachine
import random
import gymnasium as gym
from tmu.models.classification.vanilla_classifier import TMClassifier
from time import time


def ramToBin(obs):
    myList = []
    for i in obs:
        myList.append(np.binary_repr(i, 8))
    myNpArray = np.array(list("".join(myList)), dtype=int)
    return myNpArray


clauses = 3000
s = 3.9
thresh = clauses * 0.3
# tm = MultiClassTsetlinMachine(clauses, thresh, s)
results = []
stds = []
start_training = time()
stop_training = time()
for xyz in range(1):
    tm = TMClassifier(clauses, thresh, s, incremental=True)

    env = gym.make("Pong-ram-v4")
    scores = []
    score = 0
    obs, info = env.reset()
    steps = 10000  # 10000000 is too much
    episodes = 500
    explorationRate = 1.0  # exploration rate in start, 1.0 = 100%
    rateChange = 0.01  # how much it is changed
    stepChange = steps * rateChange
    minExp = 0.1  # Minimum required exploration

    EpsilonGreedy = True  # Whether to use the epslion greedy for exploration and exploitation
    # initialize Tsetlin
    obsTemp = ramToBin(obs)
    print(obsTemp)
    obsInit = np.array([obsTemp, obsTemp])
    print(obsInit)
    predInit = np.array([2, 3])
    # print("First fit")
    tm.fit(obsInit, predInit, epochs=0)
    del obsInit
    del predInit
    statesZero = []
    actionsZero = []
    batchSizeZero = 30  # Size of batches it is trained on
    batchChangeZero = 0
    statesOne = []
    actionsOne = []
    batchSizeOne = 20  # Size of batches it is trained on
    batchChangeOne = 0
    # print("Starting loop")
    currentBatchNum = 0
    for j in range(episodes):  # episodes
        if j % 100 == 0:
            print(j)
        obs, info = env.reset()
        step = 0
        done = False
        total_rewards_ep = 0
        start_training = time()
        i = 0
        while True:
            if i == steps / 2 or i == steps * 3 / 4 or i == steps / 4:
                batchSizeZero += batchChangeZero
            i += 1
            # if i % 100 == 0:
            #    print(i)
            # Exploration and exploitation part
            obsTemp = ramToBin(obs)
            if True:
                action = random.randint(2, 3)
                obs, reward, done, truncated, info = env.step(action)
                actionsZero.append(action)
                actionsOne.append(action)
            if i % stepChange == 0 and i != 0 and EpsilonGreedy:
                explorationRate -= rateChange
                if explorationRate < minExp:
                    explorationRate = minExp
            statesZero.append(obsTemp)
            statesOne.append(obsTemp)
            # print(reward)
            # Learning
            if float(reward) < 0.0:
                actionsZero = []
                statesZero = []
                actionsOne = []
                statesOne = []
            if currentBatchNum == batchSizeZero:
                tm.fit(np.array(statesZero), np.array(actionsZero), epochs=1)
                currentBatchNum = 0
                actionsZero = []
                statesZero = []
            if float(reward) > 0.0:
                tm.fit(np.array(statesOne), np.array(actionsOne), epochs=2)
                actionsOne = []
                statesOne = []
            if done or truncated:
                obs, info = env.reset()
                scores.append(score)
                currentBatchNum = 0
                actionsZero = []
                statesZero = []
                actionsOne = []
                statesOne = []
                stop_training = time()
                # print(score)
                # print("DONE")
                # print(stop_training - start_training)
                break

            else:
                currentBatchNum += 1
                # tm.fit(state, np.array([action[0]]), epochs=1)

    # print("Done with training")
    # print("Largest reward: ", max(scores))
    # print("Starting Evaluation")
    scores = []
    score = 0
    obs, info = env.reset()
    print("starting validation")
    for i in range(10):
        start_training = time()
        while True:
            action = tm.predict(np.array([ramToBin(obs)]))
            obs, reward, done, truncated, info = env.step(action[0])
            score += reward
            if done or truncated:
                obs, info = env.reset()
                scores.append(score)
                # print(score)
                score = 0
                print(time() - start_training)
                break
                # print("DONE")
                # eval_env.render()
        # else:
    print(scores)
    #std = statistics.stdev(scores)
    #mean = statistics.mean(scores)
    #results.append(mean)
    # print("mean reward:", mean)
    # print("std:", std)
    # print("Largest reward: ", max(scores))
    # print("Smallest reward: ", min(scores))

#std = statistics.stdev(results)
#mean = statistics.mean(results)
#print("mean reward:", mean)
#print("std:", std)
#print("Largest reward: ", max(results))
#print("Smallest reward: ", min(results))
# print(stop_training - start_training)
