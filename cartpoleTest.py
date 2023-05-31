import gym
#import pyglet
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
for i in range(100000):
    #if i%20 == 0:
    #    print(i)
    obsTemp = obs*100000
    action, _state = model.predict(obs, deterministic=True)
    #print(action)
    a = [bin(int(obsTemp[0].item())), bin(int(obsTemp[1].item())), bin(int(obsTemp[2].item())), bin(int(obsTemp[3].item()))]
    b = []
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
    if done:
        print(reward)
        obs = eval_env.reset()
        #eval_env.render()

for i in range(len(states)):
    for j in range(4):
        a = max[j] - len(states[i][j])
        padding = "0" * a
        PM = str(plusMinus[i][j])
        st = str(states[i][j])
        states[i][j] = PM + padding + st

for i in range(len(states)):
    states[i] = "".join(states[i])

with open('states/statesCartPolePPO.txt', 'w') as f:
    for line in states:
        l = " ".join(line)
        f.write(f"{l}\n")

with open('actions/actionsCartPolePPO.txt', 'w') as f:
    for line in actions:
        f.write(f"{line}\n")


print(max)