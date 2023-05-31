import re
import matplotlib.pyplot as plt


lines_to_read = []
with open('pong_results.txt', 'r') as file:
    for i, line in enumerate(file):
        if i >= 6 and (i - 6) % 7 == 0:
            lines_to_read.append(line.strip())

print(lines_to_read)

numbers_list = []
for line in lines_to_read:
    match = re.search(r'[-+]?\d*\.?\d+', line)
    if match:
        number = float(match.group())
        numbers_list.append(number)

print(numbers_list)
plt.plot(numbers_list,color='g')
plt.ylabel('reward')
plt.xlabel("episodes")
plt.title("Highest rewards from the TM learning pong from scratch")
plt.legend()
plt.show()
