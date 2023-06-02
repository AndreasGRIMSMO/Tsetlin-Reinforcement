import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statistics

mean_ppo_demo_scores = []
mean_ppo_scores = []
mean_math_scores = []
ppo_demo_scores = np.genfromtxt('ppo_demo_scores_100.csv', delimiter=',')
ppo_scores = np.genfromtxt('ppo_scores_100.csv', delimiter=',')
math_scores = np.genfromtxt('math_scores.csv', delimiter=',')


for j in range(len(ppo_demo_scores[0])):
    tempResults = []
    for li in ppo_demo_scores:
        tempResults.append(li[j])
    meanVal = statistics.mean(tempResults)
    mean_ppo_demo_scores.append(meanVal)
ax = sns.lineplot(x=np.arange(0, len(mean_ppo_demo_scores), 1), y=np.asarray(mean_ppo_demo_scores), label='ppo demo tsetlin')

for j in range(len(ppo_scores[0])):
    tempResults = []
    for li in ppo_scores:
        tempResults.append(li[j])
    meanVal = statistics.mean(tempResults)
    mean_ppo_scores.append(meanVal)
ax = sns.lineplot(x=np.arange(0, len(mean_ppo_scores), 1), y=np.asarray(mean_ppo_scores), label='ppo tsetlin')


for j in range(len(math_scores[0])):
    tempResults = []
    for li in math_scores:
        tempResults.append(li[j])
    meanVal = statistics.mean(tempResults)
    mean_math_scores.append(meanVal)
ax = sns.lineplot(x=np.arange(0, len(mean_math_scores), 1), y=np.asarray(mean_math_scores), label='math tsetlin')

ax = sns.lineplot(x=(0, 100), y=500, label='dqn')
ax = sns.lineplot(x=(0, 100), y=500, label='a2c')
ax = sns.lineplot(x=(0, 100), y=500, label='ppo')

ax.set(xlabel='Episode', ylabel='Score')
plt.ylim((0, 510))
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title('Compare models: PPO Demo vs PPO vs MATH')
fig = ax.get_figure()
fig.savefig("out.png", bbox_inches='tight')