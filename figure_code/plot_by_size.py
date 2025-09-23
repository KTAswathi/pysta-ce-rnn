"""Code for plotting supplementary panels on models with different sizes"""

#%%
 
import pysta
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy
import torch
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

#%% load data

data = pickle.load(open(f"{basedir}/data/comparisons/performance_and_decoding_by_size.pickle", "rb"))
sizes, perfs, decoding = data["sizes"], np.array(data["perfs"]), np.array(data["decoding"])
cols = [plt.get_cmap("viridis")(s/np.amax(sizes)*0.92) for s in sizes]
#%%

sizeticks = np.arange(0, 801, 200)

plt.figure(figsize = (2.1, 1.7))
xs = np.arange(decoding.shape[-1])+1
for isize, size in enumerate(sizes):
    plt.plot(decoding[isize, 0, :], label = f"N={size}", color = cols[isize])
plt.xlabel("time in future", labelpad = 3.5)
plt.ylabel("decoding accuracy", labelpad = 2.5)
plt.ylim(0.5, 0.91)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.savefig(f"{pysta.basedir}/figures/size/acc_vs_time{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()

figsize = (1.7, 1.7)
plt.figure(figsize = figsize)
plt.scatter(sizes, decoding[:, 0, :].mean(-1), c = cols)
plt.xlabel("network size", labelpad = 3.5)
plt.ylabel("mean decoding", labelpad = 2.5)
plt.xlim(0, 840)
plt.ylim(0.5, 0.91)
plt.xticks(sizeticks)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.savefig(f"{pysta.basedir}/figures/size/acc_vs_size{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()

plt.figure(figsize = figsize)
plt.scatter(sizes, perfs.mean(-1), c = cols)
plt.xlabel("network size", labelpad = 3.5)
plt.ylabel("performance", labelpad = 2.5)
plt.ylim(0.5, 0.91)
plt.xticks(sizeticks)
plt.xlim(0, 840)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.savefig(f"{pysta.basedir}/figures/size/perf_vs_size{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()

plt.figure(figsize = figsize)
plt.scatter(decoding[:, 0, :].mean(-1), perfs.mean(-1), c = cols)
plt.xlabel("mean decoding", labelpad = 3.5)
plt.ylabel("performance", labelpad = 2.5)
plt.ylim(0.50, 0.91)
plt.xlim(0.5, 0.91)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.savefig(f"{pysta.basedir}/figures/size/acc_vs_perf{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()

#%%

