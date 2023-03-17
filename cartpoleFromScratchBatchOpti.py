import numpy as np
# from pyTsetlinMachine.tm import MultiClassTsetlinMachine
from pyTsetlinMachine.tools import Binarizer
from ray import tune
import gym
import random
import statistics
from tmu.models.classification.vanilla_classifier import TMClassifier

X = np.load("statesCartPoleV7.npy")
Y = np.loadtxt("actionsCartPoleV7.txt", dtype=int)
eval_env = gym.make("CartPole-v1")


def objective(config):
    means = []
    b = Binarizer(max_bits_per_feature=config["bits_per_feature"])
    b.fit(X)
    for h in range(10):
        tm = TMClassifier(config["clauses"], config["clauses"] * config["thresh"], config["s"], incremental=True,
                          number_of_state_bits_ta=config["number_of_bits"])

        eval_env = gym.make("CartPole-v1")
        scores = []
        score = 0
        obs = eval_env.reset()
        steps = 100000  # 10000000 is too much
        explorationRate = 1.0  # exploration rate in start, 1.0 = 100%
        rateChange = 0.01  # how much it is changed
        stepChange = steps * rateChange
        minExp = 0.1  # Minimum required exploration
        batchSize = 10  # Size of batches it is trained on
        batchChange = config["batch_change"]
        EpsilonGreedy = False  # Wherter or not to use the epslion greedy for exploration and exploitation
        # initialize tsetlin
        obsTemp = np.array([obs])
        state = b.transform(obsTemp)
        obsInit = np.array([state[0], state[0]])
        predInit = np.array([1, 0])
        # print("First fit")
        tm.fit(obsInit, predInit, epochs=0)
        del (obsInit)
        del (predInit)
        states = []
        actions = []
        # print("Starting loop")
        currentBatchNum = 0
        for i in range(steps):
            if i == steps / 2 or i == steps * 3 / 4 or i == steps / 4:
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
                action = tm.predict(state)
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
                tm.fit(np.array(states), np.array(actions), epochs=1)
                currentBatchNum = 0
                actions = []
                states = []

            if done:
                obs = eval_env.reset()
                scores.append(score)
                if score >= 499:
                    tm.fit(np.array(states), np.array(actions), epochs=1)
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
                # tm.fit(state, np.array([action[0]]), epochs=1)

        # print("Done with training")
        # print("Largest reward: ", max(scores))
        # print("Starting Evaluation")
        scores = []
        score = 0
        obs = eval_env.reset()
        for i in range(10000):
            score += 1
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
            # else:

        mean = statistics.mean(scores)
        means.append(mean)
    return statistics.mean(means)


search_space = {
    "clauses": tune.grid_search([500, 1000, 3000]),
    "thresh": tune.grid_search([0.4]),
    "s": tune.grid_search([1.5, 3.7, 7]),
    "bits_per_feature": tune.grid_search([6, 8, 10]),
    "number_of_bits": tune.grid_search([7, 8, 9]),
    "batch_change": tune.grid_search([0, 10, 20]),
}

tuner = tune.Tuner(objective, param_space=search_space)
results = tuner.fit()
print(results.get_best_result(metric="score", mode="max").config)
print(results.get_best_result(metric="score", mode="max").metrics)