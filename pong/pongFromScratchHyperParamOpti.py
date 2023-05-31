import statistics
import typing
import numpy as np
# from pyTsetlinMachine.tm import MultiClassTsetlinMachine
import random
import gymnasium as gym
from tmu.models.classification.vanilla_classifier import TMClassifier
from time import time
from ray import tune


def ramToBin(obs):
    myList = []
    for i in obs:
        myList.append(np.binary_repr(i, 8))
    myNpArray = np.array(list("".join(myList)), dtype=int)
    return myNpArray


def objective(config):
    listOfScores = []
    for h in range(10):
        tm = TMClassifier(config["clauses"], config["clauses"] * config["thresh"], config["s"], incremental=True)
        env = gym.make("Pong-ram-v4")
        scores = []
        score = 0
        obs, info = env.reset()
        episodes = 1000
        explorationRate = 1.0  # exploration rate in start, 1.0 = 100%
        rateChange = 0.01  # how much it is changed
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
        batchSizeZero = config["batch_size"]  # Size of batches it is trained on
        batchChangeZero = 0
        statesOne = []
        actionsOne = []
        batchSizeOne = 20  # Size of batches it is trained on
        batchChangeOne = 0
        # print("Starting loop")
        currentBatchNum = 0
        for j in range(episodes):  # episodes
            obs, info = env.reset()
            i = 0
            while True:
                i += 1
                # if i % 100 == 0:
                #    print(i)
                # Exploration part
                obsTemp = ramToBin(obs)
                if True:
                    action = random.randint(2, 3)
                    obs, reward, done, truncated, info = env.step(action)
                    actionsZero.append(action)
                    actionsOne.append(action)
                statesZero.append(obsTemp)
                statesOne.append(obsTemp)
                # print(reward)
                # Learning
                if float(reward) < 0.0:
                    currentBatchNum = 0
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
        for i in range(1):
            while True:
                action = tm.predict(np.array([ramToBin(obs)]))
                obs, reward, done, truncated, info = env.step(action[0])
                score += reward
                if done or truncated:
                    obs, info = env.reset()
                    listOfScores.append(score)
                    # print(score)
                    score = 0
                    break
                    # print("DONE")
                    # eval_env.render()
            # else:
        listOfScores.append(scores)
    return {"score": statistics.mean(listOfScores)}


search_space = {
    "clauses": tune.grid_search([3000, 5000, 10000]),
    "thresh": tune.grid_search([0.3, 0.4]),
    "s": tune.grid_search([3.7]),
    "batch_size": tune.grid_search([50, 70, 100, 120, 150]),
}

tuner = tune.Tuner(objective, param_space=search_space)
results = tuner.fit()
print(results.get_best_result(metric="score", mode="max").config)
print(results.get_best_result(metric="score", mode="max").metrics)

