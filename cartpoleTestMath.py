import statistics

import gym
#import pyglet

# Retrieve the model from the hub
## repo_id = id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name})
## filename = name of the model zip file from the repository
def theta_omega_policy(obs):
    theta, w = obs[2:4]
    if abs(theta) < 0.03:
        return 0 if w < 0 else 1
    else:
        return 0 if theta < 0 else 1

# Evaluate the agent and watch it
eval_env = gym.make("CartPole-v1")
#mean_reward, std_reward = evaluate_policy(
#    model, eval_env, render=False, n_eval_episodes=5, deterministic=True, warn=False
#)
#print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

max = [0, 0, 0, 0]
actions = []
states = []
plusMinus = []
obs = eval_env.reset()
score = 0
scores = []
for i in range(100000):
    if i%200 == 0:
        print(i)
    obsTemp = obs*100000
    action = theta_omega_policy(obs)
    b = []
    a = [bin(int(obsTemp[0].item())), bin(int(obsTemp[1].item())), bin(int(obsTemp[2].item())), bin(int(obsTemp[3].item()))]
    for j in range(len(a)):
        if a[j][0] == "-":
            b.append("0")
            a[j] = a[j][3:]
        else:
            b.append("1")
            a[j] = a[j][2:]
        if len(a[j]) > max[j]:
            max[j] = len(a[j])
    actions.append(action)
    states.append(a)
    plusMinus.append(b)
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


print(max)
for i in range(len(states)):
    for j in range(4):
        a = max[j] - len(states[i][j])
        padding = "0" * a
        PM = str(plusMinus[i][j])
        st = str(states[i][j])
        states[i][j] = PM + padding + st

for i in range(len(states)):
    states[i] = "".join(states[i])

with open('statesCartPoleV3.txt', 'w') as f:
    for line in states:
        l = " ".join(line)
        f.write(f"{l}\n")

with open('actionsCartPoleV3.txt', 'w') as f:
    for line in actions:
        f.write(f"{line}\n")
