import numpy as np
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
from pyTsetlinMachine.tools import Binarizer
from ray import tune
import gym
import random
import statistics


X = np.load("statesCartPoleV5.npy")
Y = np.loadtxt("actionsCartPoleV5.txt", dtype=int)
eval_env = gym.make("CartPole-v1")


def objective(config):
    means = []
    b = Binarizer(max_bits_per_feature=config["bits_per_feature"])
    b.fit(X)
    for h in range(10):
        tm = MultiClassTsetlinMachine(config["clauses"], config["clauses"] * config["thresh"], config["s"])
        scores = []
        score = 0
        obs = eval_env.reset()
        steps = 100000
        explorationRate = 1.0
        rateChange = 0.01
        stepChange = steps * rateChange
        minExp = 0.1
        obsTemp = np.array([obs])
        state = b.transform(obsTemp)
        obsInit = np.array([state[0], state[0]])
        predInit = np.array([1, 0])
        tm.fit(obsInit, predInit, epochs=0)
        del obsInit
        del predInit
        states = []
        for i in range(steps):
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
                if explorationRate < minExp:
                    explorationRate = minExp
            secondAngle = abs(obs[2])
            states.append(state)
            if firstAngle - secondAngle > 0:
                tm.fit(state, np.array([action[0]]), epochs=1, incremental=True)
            else:
                if action[0] == 0:
                    tm.fit(state, np.array([1]), epochs=1, incremental=True)
                else:
                    tm.fit(state, np.array([0]), epochs=1, incremental=True)

            if done:
                obs = eval_env.reset()
                scores.append(score)
                score = 0
            else:
                score += 1

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
                score = 0
            else:
                score += 1
        mean = statistics.mean(scores)
        means.append(mean)
    return statistics.mean(means)


search_space = {
    "clauses": tune.grid_search([500, 1000, 3000, 5000]),
    "thresh": tune.grid_search([0.1, 0.3, 0.4]),
    "s": tune.grid_search([1.5, 2.7, 3.9, 10, 25]),
    "bits_per_feature": tune.grid_search([4, 6, 8, 10, 12]),
    "number_of_bits": tune.grid_search([4, 5, 6, 7, 8, 9, 10])
}

tuner = tune.Tuner(objective, param_space=search_space)
results = tuner.fit()
print(results.get_best_result(metric="score", mode="max").config)
