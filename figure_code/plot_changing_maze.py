"""Code for plotting all panels related to RNNs trained in changing environments"""

#%% load libraries

import pysta
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy
import torch
import matplotlib as mpl
from scipy.stats import pearsonr
from pysta import basedir

ext = ".pdf"
basefigdir = f"{basedir}/figures/changing_maze_rnn/"
np.random.seed(0)

#%% set font with arial .ttf file
import matplotlib as mpl
import matplotlib.font_manager as fm
font_path = f"{basedir}/data/arial.ttf"
fm.fontManager.addfont(font_path)
mpl.rcParams['font.family'] = "Arial"
mpl.rcParams['font.size'] = 8

#%% load data
seeds = [31,32,33,34,35]
model_names = [f"MazeEnv_L4_max6/landscape_changing-rew_dynamic-rew_changing-maze/allo_planrew_plan5-6-7/VanillaRNN/iter10_tau5.0_opt/N800_linout/model{seed}" for seed in seeds]
model_names_ref = [name.replace("changing-maze", "constant-maze") for name in model_names]
datadirs = [f"{basedir}/data/rnn_analyses/" + "_".join(model_name.split("/")) + "_" for model_name in model_names]
datadirs_ref = [f"{basedir}/data/rnn_analyses/" + "_".join(model_name.split("/")) + "_" for model_name in model_names_ref]


#%% Plot performance as a bar plot

perfs = pickle.load(open(f"{basedir}/data/comparisons/rnn_generalisation.pickle", "rb"))["perfs"]

for ienv, env in enumerate([0, 3]):
    data =  perfs[..., env, np.array([0,3]), 0].mean(1) # performance of static/changing models in static/changing tasks across seeds and repeats, then avg across repeats
    plt.figure(figsize = (1.4,1.1))

    xs, ms, ss = np.arange(data.shape[1]), np.mean(data, axis = 0), np.std(data, axis = 0)
    jitters = np.random.normal(0, 0.1, len(data)) # jitter for plotting
    plt.bar(xs, ms, yerr = ss, capsize = 3, error_kw={'elinewidth': 2, "markeredgewidth": 2})
    for idata, datapoints in enumerate(data.T):
        plt.scatter(jitters+xs[idata], datapoints, marker = ".", color = "k", alpha = 0.5, linewidth = 0.0, s = 80)

    if ienv == 1:
        plt.xticks(xs, ["trained in\nfixed", "trained in\nchanging"])#, rotation = 45, ha = "right")
        plt.ylabel("performance in\nchanging maze", labelpad = 0)
        plt.gca().tick_params(axis='x', which='major', pad=2)
    else:
        plt.xticks([])
        plt.ylabel("performance in\nfixed maze", labelpad = 0)
        
    plt.axhline(0.2, color = np.ones(3)*0.6)
    plt.yticks([0, 1])
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.savefig(f"{basefigdir}performance{ienv}{ext}", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()

#%% load data on effective connectivity

changing_maze_results = pickle.load(open(f"{datadirs[2]}correlation_results.pickle", "rb")) # load data for example agent
walls = changing_maze_results["wall_configs"] # the walls used in the trials

#%% plot two examples mazes

all_adjs = np.array(changing_maze_results["adjs"]).reshape((-1, 16*16))
all_adjs = 2*all_adjs - 1

# pick maximally differente xamples
adj_cors = (all_adjs[None, ...] * all_adjs[:, None, :]).mean(-1) # (num_walls, num_walls, 16*16)
ind1, ind2 = [arr[0] for arr in np.where(adj_cors == np.amin(adj_cors))]
adj1, adj2 = all_adjs[ind1], all_adjs[ind2]

adj_stds = np.std(all_adjs, axis = 0)
always_same = np.zeros((16, 16))
always_same[(adj_stds == 0).reshape(16, 16)] = np.nan

figsize = (1.2, 1.2)
for iind, ind in enumerate([ind1, ind2]): # for each example
    # plot the maze structure
    pysta.plot_utils.plot_flat_frame(filename = f"{basefigdir}ex_maze{iind}{ext}", figsize = figsize, vmap = np.zeros((4,4))+0.15, vmin = 0, vmax = 1, cmap = "Greys", walls = walls[ind], show = True)
    
    # plot the true adjacency matrix
    plt.figure(figsize = figsize)
    plt.imshow(changing_maze_results["adjs"][ind] + always_same, cmap = "coolwarm")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"{basefigdir}ex_adj{iind}{ext}", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()
    
    # plot the inferred adjacency matrix
    plt.figure(figsize = figsize)
    Weff = changing_maze_results["Weffs"][ind]
    vmin, vmax = np.quantile(Weff, 0.15), np.quantile(Weff, 0.9)
    plt.imshow(Weff + always_same, cmap = "coolwarm", vmin = vmin, vmax = vmax)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"{basefigdir}ex_W{iind}{ext}", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()


#%% plot avg wall correlation across mazes

cols = [np.array(plt.get_cmap("tab10")(0)) for _ in range(2)]
cols[1] = np.zeros(3)+0.6

all_maze_results = [pickle.load(open(f"{datadir}correlation_results.pickle", "rb")) for datadir in datadirs] # save the trial data
all_true, all_false = [], []
for maze_results in all_maze_results:
    all_true.append(np.mean(maze_results["true_cors"]))
    all_false.append(np.mean(maze_results["false_cors"]))
    
data = np.array([all_true, all_false]).T # (2, num_mazes, num_pairs)

plt.figure(figsize = (1.4, 1.6))

xs, ms, ss = np.arange(data.shape[1]), np.mean(data, axis = 0), np.std(data, axis = 0)
jitters = np.random.normal(0, 0.1, len(data)) # jitter for plotting
plt.bar(xs, ms, yerr = ss, capsize = 3, error_kw={'elinewidth': 2, "markeredgewidth": 2}, color = cols)
for idata, datapoints in enumerate(data.T):
    plt.scatter(jitters+xs[idata], datapoints, marker = ".", color = "k", alpha = 0.5, linewidth = 0.0, s = 80)

plt.xticks(xs, ["same\nmaze", "different\nmaze"])#, rotation = 45, ha = "right")
plt.ylabel("correlation\nbetween A and W", labelpad = -5)
plt.gca().tick_params(axis='x', which='major', pad=2)

plt.axhline(0.0, color = "k")
plt.gca().spines[['right', 'top']].set_visible(False)
plt.ylim(0.0, 1.0)
plt.yticks([0, 1])
plt.savefig(f"{basefigdir}all_adjacency_matrix_correlations{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()



#%% Plot within-maze and across-maze slot similarity

cols = [np.array(plt.get_cmap("tab10")(0)) for _ in range(2)]
cols[1] = np.zeros(3)+0.6

sames, diffs = [], []

for datadir in datadirs:

    sub_data = pickle.load(open(f"{datadir}sub_data_split.pickle", "rb")) # save the trial data
    Csubs = sub_data["Csubs"] # the Csub matrices

    for i1 in range(len(Csubs)):
        for i2 in range(len(Csubs)):
            sim = (Csubs[i1][0] * Csubs[i2][1]).sum(-1)[:-1].mean(-1)
            if i1 == i2:
                sames.append(sim)
            else:
                diffs.append(sim)
                

sames, diffs = np.array(sames), np.array(diffs)

true_cors, false_cors = sames.mean(-1), diffs.mean(-1)

mtrue, strue = np.mean(true_cors), np.std(true_cors)
bins_false = np.linspace(np.amin(false_cors)-1e-5, np.amax(false_cors)+1e-5, 12)
bins_true = np.linspace(np.amin(true_cors)-1e-5, np.amax(true_cors)+1e-5, 6)

plt.figure(figsize = (2.3,1.15))
countst, _, _ = plt.hist(true_cors, bins = bins_true, label = "same maze", density = True, color = cols[0])
countsf, _, _ = plt.hist(false_cors, bins = bins_false, label = "different maze", density = True, color = cols[1])

ymin, ymax = 0, np.ceil(max(np.amax(countsf), np.amax(countst))/5)*5
plt.ylim(ymin, ymax)
plt.xlabel(r"subspace correlation", labelpad = 3.5)
plt.ylabel("frequency", labelpad = 2.5)
plt.yticks([])
plt.legend(frameon = False)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.savefig(f"{basefigdir}slot_overlaps{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()


# %% plot wall input to locations or non-locations in the subspace

analysis_results = [pickle.load(open(f"{datadir}wall_to_transition_result.pickle","rb")) for datadir in datadirs]     

cols = {"stim": (0.35, 0.65, 0.2),
          "strong_ex": plt.get_cmap("coolwarm")(0.95), 
          "weak_ex": plt.get_cmap("coolwarm")(0.55),
          "strong_inh": plt.get_cmap("coolwarm")(0.05),
          "weak_inh": plt.get_cmap("coolwarm")(0.35),
          "neutral": plt.get_cmap("coolwarm")(0.44),}



#%% first plot effective connectivity

endpoints, adjacents, others = [np.array([result[key] for result in analysis_results]) for key in ["connection_to_endpoint", "connection_to_adjacent", "connection_to_other"]]

data = np.array([endpoints, adjacents, others]).mean(-1).T
plotcols = [cols[key] for key in ["strong_ex", "weak_inh"]] + [np.zeros(3)+0.6]
xticks = ["consistent", "adjacent", "other"]

figsize = (2.1,1.6)
plt.figure(figsize = figsize)

xs, ms, ss = np.arange(data.shape[1]), np.mean(data, axis = 0), np.std(data, axis = 0)
jitters = np.random.normal(0, 0.1, len(data)) # jitter for plotting
plt.bar(xs, ms, yerr = ss, color = plotcols, capsize = 3, error_kw={'elinewidth': 2, "markeredgewidth": 2})
for idata, datapoints in enumerate(data.T):
    plt.scatter(jitters+xs[idata], datapoints, marker = ".", color = "k", alpha = 0.5, linewidth = 0.0, s = 80)
    
plt.xticks(xs, xticks)#, rotation = 45, ha = "right")
plt.ylabel("effective connectivity")
plt.yticks([0,1,2])
plt.axhline(0.0, color = "k", lw = 1)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.savefig(f"{basefigdir}effective_transition_connectivity{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()

#%% then wall input to different transitions within subspace

walls, adjacents, others = [np.array([result[key] for result in analysis_results]).mean(-1) for key in ["input_to_wall", "input_to_adjacent", "input_to_other"]]

plotcols2 = plotcols
plotcols2[0] = cols["strong_inh"]
data = np.array([walls, adjacents, others]).T

plt.figure(figsize = figsize)

xs, ms, ss = np.arange(data.shape[1]), np.mean(data, axis = 0), np.std(data, axis = 0)
jitters = np.random.normal(0, 0.1, len(data)) # jitter for plotting
plt.bar(xs, ms, yerr = ss, color = plotcols2, capsize = 3, error_kw={'elinewidth': 2, "markeredgewidth": 2})
for idata, datapoints in enumerate(data.T):
    plt.scatter(jitters+xs[idata], datapoints, marker = ".", color = "k", alpha = 0.5, linewidth = 0.0, s = 80)
    
plt.ylabel("wall input projection")
plt.xticks(xs, ["through wall   ", "   adjacent", "other"])
plt.axhline(0.0, color = "k", lw = 1)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.savefig(f"{basefigdir}wall_to_transition_input{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()


#%% plot learning curves

epochs, epochs_ref, accs, accs_ref = [], [], [], []

for i in range(len(model_names)):
    training_result = pickle.load(open(f"{basedir}/models/{model_names[i]}.p", "rb"))
    training_result_ref = pickle.load(open(f"{basedir}/models/{model_names_ref[i]}.p", "rb"))
    accs.append(training_result["accs"])
    epochs.append(np.linspace(0, training_result["epoch"], len(accs[-1])))
    accs_ref.append(training_result_ref["accs"])
    epochs_ref.append(np.linspace(0, training_result_ref["epoch"], len(accs_ref[-1])))

maxlen = np.amin([len(epoch) for epoch in epochs])
epochs, accs = [np.array([arr[:maxlen] for arr in data]) for data in [epochs, accs]]
maxlen = np.amin([len(epoch) for epoch in epochs_ref])
epochs_ref, accs_ref = [np.array([arr[:maxlen] for arr in data]) for data in [epochs_ref, accs_ref]]

plt.figure(figsize = (2.1,1.5))
labels = ["changing maze", "fixed maze"]
for idata, data in enumerate([(epochs, accs), (epochs_ref, accs_ref)]):
    xs = np.mean(data[0], 0)
    m, s = np.mean(data[1], 0), np.std(data[1], 0)
    plt.plot(xs, m, label = labels[idata])
    plt.fill_between(xs, m-s, m+s, alpha = 0.2, linewidth = 0)

plt.legend(frameon = False)
plt.gca().spines[['right', 'top']].set_visible(False)
xmax = int(max(epochs[:, -1].max(), epochs_ref[:, -1].max()))
plt.xlim(-5000, xmax)
plt.xticks(range(0, xmax+1, 100000))
plt.xlabel("epoch", labelpad = 3.5)
plt.ylabel("performance", labelpad = 2.5)
plt.savefig(f"{basefigdir}training_curves{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()


#%% plot model support

model_results = [pickle.load(open(f"{datadir}decoder_generalization_performance.pickle", "rb")) for datadir in datadirs]
    
all_res = np.array([pysta.utils.compute_model_support(result) for result in model_results])
data = all_res[..., 1] - all_res[..., 0] # performance when satisfing cond minus performance when not


xs, mean, std = np.arange(data.shape[1]), np.mean(data, axis = 0), np.std(data, axis = 0)
jitters = np.random.normal(0, 0.1, len(data)) # jitter for plotting

plt.figure(figsize = (1.3, 1.5))
plt.bar(xs, mean, yerr = std, capsize = 3, error_kw={'elinewidth': 2, "markeredgewidth": 2})
for idata, datapoints in enumerate(data.T):
    plt.scatter(jitters+xs[idata], datapoints, marker = ".", color = "k", alpha = 0.5, linewidth = 0.0, s = 80)

plt.axhline(0.0, color = "k")
plt.gca().spines[['right', 'top']].set_visible(False)

plt.yticks(np.arange(0, 0.81, 0.2))
plt.ylim(-0.08, 0.8)
plt.ylabel("pattern overlap")

plt.xticks(xs, ["relative", "absolute"], rotation = 45, ha ="right", rotation_mode="anchor")
plt.gca().tick_params(axis='x', which='major', pad=2)
plt.ylabel("pattern overlap")

plt.savefig(f"{basefigdir}decoding_model_support{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()


#%% look at future position coding

labels = ["states", "transitions"]
for idata, data_type in enumerate(["", "transition_"]):

    all_data_true = [pickle.load(open(f"{datadir}decoder_{data_type}generalization_performance.pickle", "rb")) for datadir in datadirs]
    all_data_ref = [pickle.load(open(f"{datadir}decoder_{data_type}generalization_performance.pickle", "rb")) for datadir in datadirs_ref]
    ylabel = "% correctly\npredicted location" if idata == 0 else "% correctly\npredicted transition"

    plan_perfs, plan_xs = [], []
    ex_perfs, ex_xs = [], []
    test_neural = -1
    for all_data in [all_data_true, all_data_ref]:

        neural_ts, loc_ts = [list(np.array([data[key] for data in all_data]).mean(0).astype(int)) for key in ["neural_times", "loc_times"]]

        perfs = np.array([data["nongen_scores"] for data in all_data])

        plan_perfs.append(perfs[:, neural_ts.index(test_neural), 1:])
        plan_xs.append(loc_ts[1:])

        deltas = np.arange(-4, 5)
        delta_perfs = [[] for _ in range(len(deltas))]
        for idelta, delta in enumerate(deltas):
            for i1, t1 in enumerate(neural_ts):
                for i2, t2 in enumerate(loc_ts):
                    if t1 >= 0 and (t2 - t1 == delta):
                        delta_perfs[idelta].append(perfs[:, i1, i2])

        ex_perfs.append(np.array([np.mean(np.array(perf), 0) for perf in delta_perfs]).T)
        ex_xs.append(deltas)


    plt.figure(figsize = (2.1, 1.5))
    for i in range(2):
        xs, ms, ss = plan_xs[i], np.mean(plan_perfs[i], 0), np.std(plan_perfs[i], 0)
        plt.plot(xs, ms)
        plt.fill_between(xs, ms-ss, ms+ss, alpha = 0.2)
    plt.xlabel("time in future", labelpad = 3.5)
    plt.ylabel(ylabel, labelpad = -4)
    plt.yticks([0, 1])
    plt.ylim(0, 1)
    plt.xlim(xs[0], xs[-1])
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.savefig(f"{basefigdir}future_{labels[idata]}_planning{ext}", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()

    plt.figure(figsize = (2.1, 1.5))
    for i in range(2):
        xs, ms, ss = ex_xs[i], np.mean(ex_perfs[i], 0), np.std(ex_perfs[i], 0)
        ms[xs == 0], ss[xs == 0] = np.nan, np.nan
        plt.plot(xs, ms)
        plt.fill_between(xs, ms-ss, ms+ss, alpha = 0.2)
    plt.fill_between([-1, +1], [0, 0], [1,1], color = "k", alpha = 0.07, zorder = -10, linewidth = 0)

    plt.yticks([0, 1])
    plt.ylim(0, 1)
    plt.xticks(range(xs[0], xs[-1]+1, 2))
    plt.xlabel("time from now", labelpad = 3.5)
    plt.ylabel(ylabel, labelpad = -4)
    plt.xlim(xs[0], xs[-1])
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.savefig(f"{basefigdir}future_{labels[idata]}_execution{ext}", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()

# %%
