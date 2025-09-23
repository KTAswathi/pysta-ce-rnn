"""Code for plotting the performance and metabolic cost of different RNNs in different tasks"""

#%%

import pysta
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy
import matplotlib as mpl
from pysta import basedir
ext = ".pdf"
np.random.seed(0)

#%% set font with arial .ttf file
import matplotlib as mpl
import matplotlib.font_manager as fm
font_path = f"{basedir}/data/arial.ttf"
fm.fontManager.addfont(font_path)
mpl.rcParams['font.family'] = "Arial"
mpl.rcParams['font.size'] = 8

#%% plot performance comparison

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

gen_data = pickle.load(open(f"{basedir}/data/comparisons/rnn_generalisation.pickle", "rb"))

short_labels = ["landscape", "moving", "static"]
agent_colors = [plt.get_cmap("tab10")(i) for i in [1,2,0]]

order = np.array([2,1,0]) # plotting order is static -> moving -> landscape


loss_labels = ["acc", "rate", "param"]
ylabels = ["% correct", r"$|r|^2$", r"$|\theta|^2$"]

for iloss, loss_label in enumerate(loss_labels):

    perfs = gen_data["perfs"][..., iloss].mean(1)
    
    if iloss != 0:
        perfs = perfs / perfs.mean(0)[:, :3].max(-1)[None, :, None] # normalize to 1 for rate and param losses

    for ienv, ind in enumerate(order):
        env = short_labels[ind]

        fig = plt.figure(figsize = (1.65,1.15))
        ax = plt.gca()

        data = perfs[:, ind, :3][..., order] # performance of the agents
        baseline = perfs[:, ind, -1].mean(0) # performance of random baseline
        
        xs, mean, std = np.arange(data.shape[1]), np.mean(data, axis = 0), np.std(data, axis = 0)
        jitters = np.random.normal(0, 0.1, len(data)) # jitter for plotting
        ax.bar(xs, mean, yerr = std, color = agent_colors, capsize = 3, error_kw={'elinewidth': 2, "markeredgewidth": 2})
        
        for idata, datapoints in enumerate(data.T):
            ax.scatter(jitters+xs[idata], datapoints, marker = ".", color = "k", alpha = 0.5, linewidth = 0.0, s = 80)


        ax.set_xticks(xs)
        ax.set_xticklabels([short_labels[i] for i in order], rotation = 35, ha = "right", rotation_mode="anchor")
        ax.tick_params(axis='x', which='major', pad=2)

        if ienv == 0 or iloss == 2:
            ax.set_ylabel(ylabels[iloss], labelpad = -7)
            ax.set_yticks([0,1])
        else:
            ax.set_yticks([])
        
        ax.set_ylim(0, 1.06)
        if iloss == 0:
            ax.axhline(baseline, color = np.ones(3)*0.5)
        
        if (iloss != 2) or (ienv == 2): # only plot one weight loss
            plt.tight_layout = False
            plt.savefig(f"{basedir}/figures/rnn_generalisation/{env}_rnn_{loss_label}{ext}", bbox_inches = "tight", transparent = True)
            plt.show()
            plt.close()

plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.top'] = True




# %%
