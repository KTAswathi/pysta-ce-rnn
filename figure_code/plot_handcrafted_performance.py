"""Code for plotting the performance of different handcrafted models"""

#%%

import pysta
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl
from pysta import basedir
import copy
ext = ".pdf"
np.random.seed(0)

#%% set font with arial .ttf file
import matplotlib as mpl
import matplotlib.font_manager as fm
font_path = f"{basedir}/data/arial.ttf"
fm.fontManager.addfont(font_path)
mpl.rcParams['font.family'] = "Arial"
mpl.rcParams['font.size'] = 8

#%% Plot performance comparison

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

data = pickle.load(open(f"{basedir}/data/comparisons/handcrafted_agent_performances.pickle", "rb"))
env_labels = ["static", "changing", "moving", "landscape"]
env_titles = ["constant goal", "changing goal", "moving goal", "reward landscape"]

agent_labels = ["TD", "SR", "STA"]
agent_colors = [plt.get_cmap("tab10")(i) for i in [1,2,0]]

perfs = data["perfs"]

for ienv, env in enumerate(env_labels):

    fig = plt.figure(figsize = (1.76,1.35))
    ax = plt.gca()

    data = perfs[:, ienv, :-1] # performance of the agents
    baseline = perfs[:, ienv, -1].mean() # performance of random baseline
        
    xs, m, s = np.arange(data.shape[1]), np.mean(data, axis = 0), np.std(data, axis = 0)
    jitters = np.random.normal(0, 0.1, len(data)) # jitter for plotting
    ax.bar(xs, m, yerr = s, color = agent_colors, capsize = 3, error_kw={'elinewidth': 2, "markeredgewidth": 2})
    
    for idata, datapoints in enumerate(data.T):
        plt.scatter(jitters+xs[idata], datapoints, marker = ".", color = "k", alpha = 0.5, linewidth = 0.0)
    
    ax.set_xticks(xs)
    ax.set_xticklabels(agent_labels)

    ax.set_ylim(0, 1.02)
    ax.set_ylabel("% correct", labelpad = -5)

    ax.set_yticks([0,1])
    ax.axhline(baseline, color = np.ones(3)*0.5)

    plt.savefig(f"{basedir}/figures/handcrafted_performance/{env}_performance{ext}", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()

plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.top'] = True

# %%
