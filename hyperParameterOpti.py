import numpy as np
from pyTsetlinMachine.tm import MultiClassTsetlinMachine

from ray import tune

X = np.loadtxt("statesCartPoleV3.txt")
Y = np.loadtxt("actionsCartPoleV3.txt")
splitratio = 0.7
X_train = X[:int(len(X)*splitratio)]
X_test = X[int(len(X)*splitratio):]
Y_train = Y[:int(len(Y)*splitratio)]
Y_test = Y[int(len(Y)*splitratio):]

def objective(config):
    tm = MultiClassTsetlinMachine(config["clauses"], config["clauses"] * config["thresh"], config["s"])
    result = 0
    for i in range(200):
        tm.fit(X_train, Y_train, epochs=1)
        result = 100*(tm.predict(X_test) == Y_test).mean()
    return {"score": result}

search_space = {
    "clauses": tune.grid_search([500, 1000, 3000, 5000]),
    "thresh": tune.grid_search([0.3, 0.4, 0.5]),
    "s": tune.grid_search([3.9, 10, 50, 100, 200])
}

tuner = tune.Tuner(objective, param_space=search_space)
results = tuner.fit()
print(results.get_best_result(metric="score", mode="max").config)
