"""Code for plotting decoding analyses on RNNs trained in the static and moving goal tasks"""

#%%

import pysta
from pysta import basedir
import pickle
import matplotlib.pyplot as plt
import numpy as np
ext = ".pdf"
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
basetasks = ["static_relrew", "static_planrew", "moving_relrew", "moving_planrew"]
labels = ["RNN", "STA"]
basenames = {"RNN": {"static_relrew": "MazeEnv_L4_max6_goal_changing-rew_static-rew_constant-maze_allo_relrew_plan5-6-7_VanillaRNN_iter9-10-11_tau5.0_opt_N800_linout_model",
                "static_planrew": "MazeEnv_L4_max6_goal_changing-rew_static-rew_constant-maze_allo_planrew_plan5-6-7_VanillaRNN_iter10_tau5.0_opt_N800_linout_model",
                "moving_relrew": "MazeEnv_L4_max6_goal_changing-rew_dynamic-rew_constant-maze_allo_relrew_plan5-6-7_VanillaRNN_iter9-10-11_tau5.0_opt_N800_linout_model",
                "moving_planrew": "MazeEnv_L4_max6_goal_changing-rew_dynamic-rew_constant-maze_allo_planrew_plan5-6-7_VanillaRNN_iter10_tau5.0_opt_N800_linout_model",}
            }

basenames["STA"] = {key: "sta_"+value for (key, value) in basenames["RNN"].items()}
min_dist, max_dist = 3, 6
cols = {"static_relrew": plt.get_cmap("tab10")(1), "static_planrew": plt.get_cmap("tab10")(1), "moving_relrew": plt.get_cmap("tab10")(2), "moving_planrew": plt.get_cmap("tab10")(2)}

#%% plot full future decoding

for basetask in basetasks:
    results = []
    for label in labels:
        datadirs = [f"{basedir}/data/rnn_analyses/" + "_".join(basenames[label][basetask].split("/")) + str(seed) + "_" for seed in seeds]
        results.append(np.array([pickle.load(open(f"{datadir}simple_{basetask}_decoder_generalization_performance_minmax{min_dist}-{max_dist}.pickle", "rb"))["nongen_scores"] for datadir in datadirs]))
    results = np.array(results)

    xs = np.arange(results.shape[-1])[1:]
    ineural = 0 # from end of planning
    plt.figure(figsize = (2.0,1.6))
    for imodel in range(results.shape[0]):
        col = cols[basetask] if imodel == 0 else plt.get_cmap("tab10")(0)
        data = results[imodel, :, ineural, 1:]
        ms, ss = data.mean(0), data.std(0)
        plt.plot(xs, ms, label = labels[imodel], color = col)
        plt.fill_between(xs, ms-ss, ms+ss, alpha = 0.2, color = col)
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.legend(frameon = False)
    plt.ylim(0,1)
    plt.xticks([1,2,3])
    plt.xlabel("time in trial", labelpad = 3.5)
    
    if basetask == "static_relrew":
        plt.ylabel("prediction accuracy", labelpad = 2.5)
    else:
        plt.yticks([])
    plt.xlim(1,3)
    plt.savefig(f"{pysta.basedir}/figures/simple_tasks/state_at_time_decoding_{basetask}{ext}", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()
        
#%% plot decoding for being at some location at any time in the future

for basetask in basetasks:
    results = []
    for label in labels:
        datadirs = [f"{basedir}/data/rnn_analyses/" + "_".join(basenames[label][basetask].split("/")) + str(seed) + "_" for seed in seeds]
        results.append(np.array([pickle.load(open(f"{datadir}simple_{basetask}_decode_from_planning_minmax{min_dist}-{max_dist}.pickle", "rb")) for datadir in datadirs]))
    results = np.array(results)
    
    data = np.nanmean(results, axis = (-1,-2)).T
    
    plt.figure(figsize = (1.5,1.3))
    xs, ms, ss = np.arange(data.shape[1]), np.mean(data, axis = 0), np.std(data, axis = 0)
    jitters = np.random.normal(0, 0.1, len(data)) # jitter for plotting
    plt.bar(xs, ms, yerr = ss, capsize = 3, error_kw={'elinewidth': 2, "markeredgewidth": 2}, color = [cols[basetask], plt.get_cmap("tab10")(0)])
    
    for idata, datapoints in enumerate(data.T):
        plt.scatter(jitters+xs[idata], datapoints, marker = ".", color = "k", alpha = 0.5, linewidth = 0.0, s = 80)
    
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.ylim(0,1)
    if basetask == "static_relrew":
        plt.ylabel("prediction accuracy", labelpad = 2.5)
    else:
        plt.yticks([])
        
    plt.xticks(xs, labels)
    plt.savefig(f"{pysta.basedir}/figures/simple_tasks/state_any_time_decoding_{basetask}{ext}", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()



# %%
