import statistics
import typing
import numpy as np
# from pyTsetlinMachine.tm import MultiClassTsetlinMachine
import random
from pyTsetlinMachine.tools import Binarizer
import gymnasium as gym
from tmu.models.classification.vanilla_classifier import TMClassifier
from time import time
import matplotlib.pyplot as plt

clauses = 3000
s = 3.9
thresh = clauses * 0.3

# tm = MultiClassTsetlinMachine(clauses, thresh, s)
results = []
stds = []
highestScores = []


def ramToBin(obs):
    myList = []
    for i in obs:
        myList.append(np.binary_repr(i, 8))
    myNpArray = np.array(list("".join(myList)), dtype=int)
    return myNpArray


tm = TMClassifier(clauses, thresh, s, incremental=True)
counting = 0
video_env = gym.make("Pong-ram-v4", render_mode="rgb_array")
video_env = gym.wrappers.RecordVideo(video_env, 'video2', episode_trigger=lambda episode_id: True)
video_number = 0


def validation(x1, mytm, myenv):
    global video_number
    scores1 = []
    score1 = 0
    obs, info = video_env.reset()
    print("starting eval")
    for i in range(10):
        video_number += 1
        while True:

            obsTemp = ramToBin(obs)
            # print(state)
            # print(type(state))
            action = mytm.predict(np.array([obsTemp]))
            # print(action)
            # print(action[0])
            obs, reward, done, truncated, info = video_env.step(action[0])
            score1 += reward
            if done or truncated:
                print("Episode number: ", video_number)
                print(score1)
                # video_env.close()
                obs, info = video_env.reset()
                scores1.append(score1)
                # print(score)
                score1 = 0
                break
                # print("DONE")
                # eval_env.render()
    print("finished eval")

    std = statistics.stdev(scores1)
    mean = statistics.mean(scores1)
    highestScores.append(max(scores1))
    #print("Highest reward: ")
    results.append(mean)
    print("Total episode number:", x1)

    #print("mean reward:", mean)
    #print("std:", std)
    #print("Highest reward: ", max(scores1))
    # print("Largest reward: ", max(scores))
    # print("Smallest reward: ", min(scores))


eval_env = gym.make("Pong-ram-v4")

scores = []
score = 0
obs, info = eval_env.reset()
steps = 100000  # 10000000 is too much
explorationRate = 1.0  # exploration rate in start, 1.0 = 100%
rateChange = 0.01  # how much it is changed
stepChange = steps * rateChange
minExp = 0.1  # Minimum required exploration
batchSize = 20  # Size of batches it is trained on
batchChange = 0
EpsilonGreedy = False  # Wheter or not to use the epslion greedy for exploration and exploitation
# initialize tsetlin
obsTemp = ramToBin(obs)
print(obsTemp)
obsInit = np.array([obsTemp, obsTemp])
print(obsInit)
predInit = np.array([2, 3])
# print("First fit")
tm.fit(obsInit, predInit, epochs=0)
del (obsInit)
del (predInit)
states = []
actions = []
# print("Starting loop")
currentBatchNum = 0
epsiodes = 3000
statesZero = []
actionsZero = []
batchSizeZero = 30  # Size of batches it is trained on
batchChangeZero = 0
statesOne = []
actionsOne = []
batchSizeOne = 20  # Size of batches it is trained on
batchChangeOne = 0
for x in range(epsiodes):  # 1000 episodes
    obs, info = eval_env.reset()
    step = 0
    done = False
    total_rewards_ep = 0
    start_training = time()
    i = 0
    while True:
        i += 1
        # if i % 100 == 0:
        #    print(i)
        # Exploration and exploitation part
        obsTemp = ramToBin(obs)
        if True:
            action = random.randint(2, 3)
            obs, reward, done, truncated, info = eval_env.step(action)
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
            tm.fit(np.array(statesOne), np.array(actionsOne), epochs=1)
            actionsOne = []
            statesOne = []
        if done or truncated:
            obs, info = eval_env.reset()
            validation(x, tm, eval_env)
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

std = statistics.stdev(results)
mean = statistics.mean(results)
print("mean reward final:", mean)
print("std final:", std)
# print("Done with training")
# print("Largest reward: ", max(scores))
# print("Starting Evaluation")
print(counting)
plt.plot(results, color='r', label='Average reward per learning episode')
plt.plot(highestScores, color='g', label='Highest reward per  learning episode')
plt.ylabel('reward')
plt.xlabel("episodes")
plt.title("Rewards from the TM learning pong from scratch")
plt.legend()
plt.show()
print(counting)
