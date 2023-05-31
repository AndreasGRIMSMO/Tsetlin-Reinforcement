import statistics
import numpy as np
import gym
import random



# Evaluate the agent and watch it
eval_env = gym.make("CartPole-v1")


actions = []
states = []
obs = eval_env.reset()
score = 0
scores = []

for i in range(1000000):
    #if i%200 == 0:
    #    print(i)
    s = obs.tolist()
    #action = theta_omega_policy(obs)
    action = random.randint(0, 1)
    #b = []
    actions.append(action)
    states.append(s)
    #plusMinus.append(b)
    obs, reward, done, info = eval_env.step(action)
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


newArray = np.array(states)

with open('statesCartPoleRandom.npy', 'wb') as f:
    np.save(f, newArray)

with open('actionsCartPoleRandom.txt', 'w') as f:
    for line in actions:
        f.write(f"{line}\n")
