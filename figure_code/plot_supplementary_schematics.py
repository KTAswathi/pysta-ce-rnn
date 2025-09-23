"""Code for plotting schematics used in supplementary figures"""

#%% load libraries
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

#%% generate little schematic of relative vs absolute

locs = np.arange(6)
neurals = np.arange(4)
sequence_colors = [plt.get_cmap("viridis")(iind / (len(neurals)-0.7)) for iind in range(len(neurals))][::-1]
for i in range(2):
    plt.figure(figsize = (1.0, 0.75))
    for t in neurals:
        perf = np.zeros(len(locs))
        if i == 0:
            perf[locs == t+2] = 1.0
        else:
            perf[locs == 3] = 0.9**t
        perf = perf + (1-perf.sum())/len(perf)
        plt.plot(locs, perf, lw = 2, color = sequence_colors[int(t)])
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"{pysta.basedir}/figures/schematics/decoding_support_schematic{i}{ext}", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()

# %%
