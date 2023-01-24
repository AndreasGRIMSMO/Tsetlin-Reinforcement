import numpy as np
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
from time import time
import matplotlib.pyplot as plt
import gym
import statistics

from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Retrieve the model from the hub
## repo_id = id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name})
## filename = name of the model zip file from the repository
checkpoint = load_from_hub(
    repo_id="sb3/demo-hf-CartPole-v1",
    filename="ppo-CartPole-v1.zip",
)
model = PPO.load(checkpoint)

X = np.loadtxt("statesCartPoleV2.txt", dtype=int)
Y = np.loadtxt("actionsCartPoleV2.txt", dtype=int)
splitratio = 0.7
X_train = X[:int(len(X)*splitratio)]
X_test = X[int(len(X)*splitratio):]
Y_train = Y[:int(len(Y)*splitratio)]
Y_test = Y[int(len(Y)*splitratio):]
print(X[0])
print(len(X[0]))
print(Y[0])
clauses = 3000
s = 3.9
thresh = 0.4
#tm = MultiClassTsetlinMachine(clauses, clauses*0.4, 3.9)


def theta_omega_policy(obs):
    theta, w = obs[2:4]
    if abs(theta) < 0.03:
        return 0 if w < 0 else 1
    else:
        return 0 if theta < 0 else 1

tm = MultiClassTsetlinMachine(clauses, clauses * thresh, s)
result = 0
for i in range(1):
    start = time()
    tm.fit(X_train, Y_train, epochs=1)
    stop = time()
    result = 100*(tm.predict(X_test) == Y_test).mean()
    #plotArray = np.append(plotArray, result)
    print("#%d Accuracy: %.2f%% (%.2fs)" % (i+1, result, stop-start))
    #with open('tsetlinAnimals1', 'wb') as tsetlin_file:
    #        pickle.dump(tm, tsetlin_file)


eval_env = gym.make("CartPole-v1")
#maximum = [18, 17, 13, 17]
maximum = [18, 18, 15, 16]
scores = []
score = 0
obs = eval_env.reset()
same = 0
steps = 10000
for i in range(steps):
    if i%20 == 0:
        print(i)
    obsTemp = obs*100000

    a = [bin(int(obsTemp[0].item())), bin(int(obsTemp[1].item())), bin(int(obsTemp[2].item())), bin(int(obsTemp[3].item()))]
    b = []
    states = [0, 0, 0, 0]
    action2, _state = model.predict(obs, deterministic=True)
    for j in range(len(a)):
        if a[j][0] == "-":
            b.append("0")
            a[j] = a[j][3:]
        else:
            b.append("1")
            a[j] = a[j][2:]

    for j in range(4):
        if len(a[j]) <= maximum[j]:
            a1 = maximum[j] - len(a[j])
            padding = "0" * a1
            PM = str(b[j])
            st = str(a[j])
            states[j] = PM + padding + st
            #print("short")
        else:
            st = a[j][len(a[j]) - maximum[j]:]
            PM = str(b[j])
            states[j] = PM + st
            #print("long")


    states = "".join(states)
    #print(type(states))
    #states = " ".join(states)
    #print(states)
    sta = np.array(list(states), dtype=int)
    #if (len(states) != 69):
    #    print("Len is: " + str(len(states)))
    #sta = np.fromstring(states, sep=' ')
    #print(sta)
    #print(len(sta))
    #print(type(sta))
    action = tm.predict(sta)
    if action[0] == action2:
        same +=1
    #print(action)
    obs, reward, done, info = eval_env.step(action[0])
    score += 1
    if done:
        obs = eval_env.reset()
        scores.append(score)
        score = 0
        #eval_env.render()

std = statistics.stdev(scores)
mean = statistics.mean(scores)

print(mean)
print(std)
print(same/steps)
