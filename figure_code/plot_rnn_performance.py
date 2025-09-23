"""Code for plotting the learning curves and behaviour of RNNs against reward and value differences"""

#%%

import pysta
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy
import torch
from pysta import basedir
from scipy.stats import binned_statistic
ext = ".pdf"
basefigdir = f"{basedir}/figures/rnn_behaviour/"
np.random.seed(0)

#%% set font with arial .ttf file
import matplotlib as mpl
import matplotlib.font_manager as fm
font_path = f"{basedir}/data/arial.ttf"
fm.fontManager.addfont(font_path)
mpl.rcParams['font.family'] = "Arial"
mpl.rcParams['font.size'] = 8

#%% load some data
seeds = [31,32,33,34,35]
model_names = [f"MazeEnv_L4_max6/landscape_changing-rew_dynamic-rew_constant-maze/allo_planrew_plan5-6-7/VanillaRNN/iter10_tau5.0_opt/N800_linout/model{seed}" for seed in seeds]
datadirs = [f"{basedir}/data/rnn_analyses/" + "_".join(model_name.split("/")) + "_" for model_name in model_names]
figsize = (1.85, 1.6)

#%% first plot learning curve

epochs, accs = [], []

for i in range(len(model_names)):
    training_result = pickle.load(open(f"{basedir}/models/{model_names[i]}.p", "rb"))
    accs.append(training_result["accs"])
    epochs.append(np.linspace(0, training_result["epoch"], len(accs[-1])))

epochs, accs = [np.array(arr) for arr in [epochs, accs]]

plt.figure(figsize = figsize)
for idata in range(len(epochs)):
    xs, ys = epochs[idata], accs[idata]
    plt.plot(xs[::10], ys[::10])

plt.gca().spines[['right', 'top']].set_visible(False)
plt.xlabel("epoch", labelpad = 3.5)
plt.ylabel("performance", labelpad = 2.5)
plt.xlim(-xs[-1]*0.01, xs[-1])
plt.savefig(f"{basefigdir}training_curves{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()

#%% then plot performance vs. action number

results = pickle.load(open(f"{basedir}/data/rnn_analyses/analyse_performance.p", "rb"))

plt.figure(figsize = figsize)
for model_name in model_names:
    ys, ctrl = results[model_name]["frac_optimal"], results[model_name]["rand_optimal"]
    xs = np.arange(len(ys))
    plt.plot(xs, ys)
    
plt.plot(xs, ctrl, color = np.ones(3)*0.6)

plt.gca().spines[['right', 'top']].set_visible(False)
plt.xlabel("action number", labelpad = 3.5)
plt.ylabel("performance", labelpad = 2.5)
plt.xlim(xs[0], xs[-1])
plt.ylim(0.91, 0.94)
plt.savefig(f"{basefigdir}perf_by_time{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()

#%% then plot p(optimal) vs. difference in value or reward
# here, 'optimality' is defined w.r.t. value and reward respectively

num_bins = 41
labels = ["reward", "value"]
for imetric, metric_str in enumerate(["rew0", "val0"]):
    
    all_deltas, all_opts = [], []
    for model_name in model_names:
        loc0, act0 = results[model_name]["loc0"], results[model_name]["act0"]
        adjacent_locs = results[model_name]["rnn"].env.adjacency[0, ...][loc0, :].detach().numpy()
        
        metric = results[model_name][metric_str] + (adjacent_locs-1.0)*1e10 # what's best to do of possible things?
        all_opts.append(np.argmax(metric, axis = -1) == act0.astype(int)) # is the action optimal?
        sorted_metric = np.sort(metric, axis = -1)
        all_deltas.append(sorted_metric[:, -1] - sorted_metric[:, -2]) # difference between best and second best reward
    
    cat_deltas = np.concatenate(all_deltas, axis = 0)
    min_val, max_val = np.amin(cat_deltas), np.quantile(cat_deltas, 0.95)
    bins = np.linspace(min_val, max_val, num_bins)
    xs = (bins[1:] + bins[:-1]) / 2
    
    plt.figure(figsize = figsize)
    for imodel in range(len(model_names)):
        ys, _, _ = binned_statistic(all_deltas[imodel], all_opts[imodel], statistic = "mean", bins = bins)
        plt.plot(xs, ys)
        
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.xlabel(f"{labels[imetric]} difference", labelpad = 3.5)
    plt.ylabel("accuracy", labelpad = 2.5)
    plt.xlim(-xs[-1]*0.01, xs[-1])
    plt.ylim(0.4, 1.01)
    plt.axhline(0.5, color = np.ones(3)*0.6, linestyle = "-")
    plt.savefig(f"{basefigdir}perf_by_diff{imetric}{ext}", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()

# %%
