import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from cycler import cycler

import os


def calc(dir):
    if not os.path.exists(dir):
        return None

    csvs = [os.path.join(dir, x) for x in os.listdir(dir)]
    if len(csvs) == 0:
        return None
    all_rewards = np.zeros((len(csvs), 100))

    for i, csv in enumerate(csvs):
        csv_content = np.genfromtxt(csv, delimiter=",", dtype=np.int32, skip_header=1)  # [num_episodes, reward]
        all_rewards[i, :] = csv_content[:, 1]

    avg_rewards = np.mean(all_rewards, axis=1)
    std_err = stats.sem(all_rewards, axis=1)
    std = np.std(all_rewards, axis=1)
    # mean = np.mean(avg_rewards)
    # std = np.std(avg_rewards, ddof=1)
    return avg_rewards, std_err


models = ["05", "10", "50", "MF"]
# models = ["10"]
games = ["M1D0", "M1D1", "M4D0"]
# games = ["M1D1"]

means = {}
stds = {}

for game in games:
    # print("{}:\t\t#####".format(game))
    for model in models:
        ret = calc("{}/{}".format(model, game))
        if ret is not None:
            # print("{}-{}\t{}\t{}".format(model, game, ret[0], ret[1]))
            means[model, game] = ret[0]
            stds[model, game] = ret[1]


plt.rc('axes', prop_cycle=(cycler('color', ['r', 'c', 'b', 'm'])))
# width = 0.2
for j, game in enumerate(games):
    for i, model in enumerate(models):
        mean, std = means[model, game], stds[model, game]
        count = len(mean)
        x = np.arange(count)/15 + 9*j + 1.8*i
        if game == games[0]:
            plt.errorbar(x, mean, yerr=std, alpha=0.4, fmt='.', label=model)
        else:
            plt.errorbar(x, mean, yerr=std, alpha=0.4, fmt='.')
for j, game in enumerate(games):
    for i, model in enumerate(models):
        mean, std = means[model, game], stds[model, game]
        mm = np.mean(mean)
        ss = stats.sem(mean)
        print("{} - {} : {}".format(model, game, mm))
        plt.errorbar(9*j+1.8*i - 0.05, mm, yerr=ss, alpha=0.3, fmt="o")
    print()

#     plt.bar(np.arange(len(games))+i*width, means[model], width, yerr=stds[model], label=model)
plt.xticks([3, 12, 21], games)
# axes = plt.gca()
plt.legend(loc='best')
plt.show()
