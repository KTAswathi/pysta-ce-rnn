"""Code for plotting all decoding analyses for the RNNs trained in single mazes"""

#%% load libraries
import pysta
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
from pysta.utils import compute_model_support
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

#%% set some parameters
seeds = [31,32,33,34,35]

basenames = {"WM": "MazeEnv_L4_max6_landscape_changing-rew_dynamic-rew_constant-maze_allo_planrew_plan5-6-7_VanillaRNN_iter10_tau5.0_opt_N800_linout_model",
             "STA":
            "sta_MazeEnv_L4_max6_landscape_changing-rew_dynamic-rew_constant-maze_allo_planrew_plan5-6-7_VanillaRNN_iter10_tau5.0_opt_N800_linout_model",
             "DP": "dp_MazeEnv_L4_max6_landscape_changing-rew_dynamic-rew_constant-maze_allo_planrew_plan5-6-7_VanillaRNN_iter10_tau5.0_opt_N800_linout_model",
             "relrew": "MazeEnv_L4_max6_landscape_changing-rew_dynamic-rew_constant-maze_allo_relrew_plan5-6-7_VanillaRNN_iter9-10-11_tau5.0_opt_N800_linout_model",
            "egocentric": "MazeEnv_L4_max6_landscape_changing-rew_dynamic-rew_constant-maze_ego_planrew_plan5-6-7_VanillaRNN_iter10_tau5.0_opt_N800_linout_model",}

all_planning_decoding = {}
all_execution_decoding = {}
all_scores = {}
all_time = {}

figsize = (2.0, 1.25)
models = ["WM", "STA", "DP", "relrew", "egocentric"]

#%% plot figures

for imodel, model in enumerate(models):

    #%%

    # load data
    model_results = [pickle.load(open(f"{basedir}data/rnn_analyses/{basenames[model]}{seed}_decoder_generalization_performance.pickle", "rb")) for seed in seeds]

    neural_times, loc_times, scores, nongen_scores = [np.array([result[key] for result in model_results]) for key in ["neural_times", "loc_times", "scores", "nongen_scores"]]

    mean_neural, mean_loc = neural_times.mean(0), loc_times.mean(0)
    mean_nongen_scores, std_nongen_scores = nongen_scores.mean(0), nongen_scores.std(0)
    all_scores[f"{model}_scores"], all_scores[f"{model}_neural_ts"], all_scores[f"{model}_loc_ts"] = scores, mean_neural, mean_loc

    #%%
    
    # plot result
    plot_inds = np.arange(len(mean_neural))[2:-1]
    labels = [f"t={int(i)}" for i in mean_neural[plot_inds]] if model == "DP" else None
    sequence_colors = [plt.get_cmap("viridis")(iind / (len(plot_inds)-0.7)) for iind in range(len(plot_inds))][::-1]
    pysta.plot_utils.plot_prediction_result(mean_nongen_scores[plot_inds, :], mean_neural[plot_inds], mean_loc, cols = sequence_colors, figsize = figsize, ymax = 1.07,
                                            error = std_nongen_scores[plot_inds, :], show = False, ts_train = None, xticks = range(6), yticks = [0, 1], labelpad = -5, labels = labels)
    
    if model in ["DP", "egocentric", "relrew"]:
        xticks, xlabel = mean_loc.astype(int), "time in future"
    else:
        xticks, xlabel = [], None
        
    if model == "DP":
        plt.legend(loc = "upper center", bbox_to_anchor = (0.5, 1.0), ncol = 3, frameon = False, columnspacing = 0.9, handlelength = 1.1, handletextpad = 0.5)
    plt.xticks(xticks)
    plt.xlabel(xlabel, labelpad = 3.5)
    plt.ylabel("% correctly\npredicted location", labelpad = 1)
    plt.savefig(f"{pysta.basedir}/figures/rnn_decoding/{model}_decoding_by_time{ext}", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()
    
    #%% plot averaged over time differences
    
    max_delta = int(mean_neural[-1] - mean_loc[0])-1
    deltas = list(range(-max_delta, max_delta+1))
    accs = np.zeros((nongen_scores.shape[0], len(deltas)))
    counts = np.zeros(len(deltas))
    for i1, t1 in enumerate(mean_neural):
        if t1 >= 0:
            for i2, t2 in enumerate(mean_loc):
                delta = t2 - t1
                if delta in deltas:
                    ind = deltas.index(int(delta))
                    accs[:, ind] += nongen_scores[:, i1, i2]
                    counts[ind] += 1
                
    accs /= counts
    accs[:, deltas.index(0)] = np.nan
    mean_accs, std_accs = accs.mean(0), accs.std(0)
    all_execution_decoding[model+"_accs"], all_execution_decoding[model+"_xvals"] = accs, deltas
    
    pysta.plot_utils.plot_prediction_result(mean_accs[None, :], [0], deltas, cols = [plt.get_cmap("tab10")(0)], figsize = figsize, ymax = 1.0,
                                        error = std_accs[None, :], show = False, ts_train = None, xticks = None, yticks = [0, 1], labelpad = -5)
    
    if model in ["DP", "egocentric", "relrew"]:
        xticks, xlabel = deltas, "delay"
    else:
        xticks, xlabel = [], ""
        
    plt.xticks(xticks)
    plt.xlabel(xlabel)
    plt.savefig(f"{pysta.basedir}/figures/rnn_decoding/{model}_avg_execution_decoding{ext}", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()
    
    #%% now plot time decoding
    
    time_datas = [pickle.load(open(f"{basedir}data/rnn_analyses/{basenames[model]}{seed}_decode_time_of_loc.pickle", "rb")) for seed in seeds]
    
    all_accs_by_target = []
    all_time_steps = []
    for time_data in time_datas:
        time_scores, time_preds, time_targets, time_steps = [time_data[key] for key in ["all_scores", "all_preds", "all_targets", "all_steps"]]
        cat_targets = np.concatenate([np.concatenate(target) for target in time_targets if len(target) > 0])
        unique_targets = np.unique(cat_targets)
        time_steps = np.array(time_steps).mean(0).astype(int)
        
        accs_by_target = []
        for target in unique_targets:
            
            accs_by_target.append([np.mean(np.concatenate(time_preds[ind])[np.concatenate(time_targets[ind]) == target] == target) for ind in range(len(time_preds)) if len(time_preds[ind]) > 0])
            
        accs_by_target = np.array(accs_by_target)
        all_time_steps.append(time_steps)
        all_accs_by_target.append(accs_by_target.mean(-1))
    all_time_steps = np.array(all_time_steps).mean(0)
    all_accs_by_target = np.array(all_accs_by_target)

    plt.figure(figsize = figsize)
    plt.plot(all_time_steps, all_accs_by_target.mean(0))
    plt.xlim(all_time_steps[0], all_time_steps[-1])
    plt.ylim(0, 1)
    plt.yticks([0, 1])
    plt.xticks(all_time_steps.astype(int))
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.xlabel("time in future")
    plt.ylabel("accuracy", labelpad = -8)
    plt.savefig(f"{pysta.basedir}/figures/rnn_decoding/{model}_decode_time_of_loc{ext}", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()

    all_time[f"{model}_steps"], all_time[f"{model}_accs"] = all_time_steps, all_accs_by_target
    
    #%% plot for the end of planning
    
    t_neural = -1
    ind = list(mean_neural).index(t_neural) # index in the data array
    
    pysta.plot_utils.plot_prediction_result(mean_nongen_scores[ind:ind+1, 1:], [t_neural], mean_loc[1:], cols = [plt.get_cmap("tab10")(0)], figsize = figsize, ymax = 1.0,
                                        error = std_nongen_scores[ind:ind+1, 1:], show = False, xticks = mean_loc[1:].astype(int), yticks = [0, 1], labelpad = -5)
    
    
    all_planning_decoding[model+"_accs"], all_planning_decoding[model+"_xvals"], all_planning_decoding[model+"_neural"] = nongen_scores, mean_loc, mean_neural
    
    if model in ["DP", "egocentric", "relrew"]:
        xticks, xlabel = range(1,6), "time in future"
    else:
        xticks, xlabel = [], ""
        
    plt.xticks(xticks)
    plt.xlabel(xlabel)
    plt.savefig(f"{pysta.basedir}/figures/rnn_decoding/{model}_decoding_from_planning{ext}", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()


    #%% now an example generalisation

    neural_time, loc_time = (1,3)
    train_neural_ind = np.argmin((neural_times - neural_time)**2)
    train_loc_ind = np.argmin((loc_times - loc_time)**2)

    # performance when taking the decoder trained to predict loc_time from neural_time, and applying it to all other combinations
    gen_perfs = scores[:, train_neural_ind, train_loc_ind, ...] # (models, neural_times, loc_times)
    mean_scores, std_scores = gen_perfs.mean(0), gen_perfs.std(0) # mean and std across models
    
    # plot the result
    ts_train = (neural_time, loc_time)
    pysta.plot_utils.plot_prediction_result(mean_scores[plot_inds, :], mean_neural[plot_inds], mean_loc, xticks = xticks, xlabel = xlabel, cols = sequence_colors, figsize = figsize, yticks = [0, 1],
                                            ymax = 1.07, error = std_scores[plot_inds, :], ts_train = ts_train, show = False, labelpad = -5, filename = None)
        
    plt.xticks(xticks)
    plt.xlabel(xlabel, labelpad = 3.5)
    plt.ylabel("% correctly\npredicted location", labelpad = 1)
    plt.savefig(f"{basedir}/figures/rnn_decoding/{model}_decoding_gen_{neural_time}_{loc_time}{ext}", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()

    #%% quantify 'goodness' of the 'relative' and 'absolute' coding models

    pysta.reload()

    all_res = np.array([compute_model_support(result) for result in model_results])
    data = all_res[..., 1] - all_res[..., 0] # performance when satisfing cond minus performance when not


    xs, mean, std = np.arange(data.shape[1]), np.mean(data, axis = 0), np.std(data, axis = 0)
    jitters = np.random.normal(0, 0.1, len(data)) # jitter for plotting
    
    plt.figure(figsize = (1.3, figsize[1]))
    plt.bar(xs, mean, yerr = std, capsize = 3, error_kw={'elinewidth': 2, "markeredgewidth": 2})
    for idata, datapoints in enumerate(data.T):
        plt.scatter(jitters+xs[idata], datapoints, marker = ".", color = "k", alpha = 0.5, linewidth = 0.0, s = 80)
            
    plt.axhline(0.0, color = "k")
    plt.gca().spines[['right', 'top']].set_visible(False)

    plt.yticks(np.arange(0, 0.81, 0.2))
    plt.ylim(-0.08, 0.8)
    plt.ylabel("pattern overlap")
        
    if model in ["DP", "egocentric", "relrew"]:
        plt.xticks(xs, ["relative", "absolute"], rotation = 45, ha ="right", rotation_mode="anchor")
        plt.gca().tick_params(axis='x', which='major', pad=2)
        plt.ylabel("pattern overlap")
    else:
        plt.xticks([])
        
    fname = f"{basedir}/figures/rnn_decoding/{model}_decoding_model_support"
    plt.savefig(f"{fname}{ext}", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()


#%% plot mean decoding

plot_names = ["STA", "WM", "DP"]
plot_labels = ["STA", "RNN", "value"]
rnn_col = np.array(plt.get_cmap("tab10")(0))
sta_col = np.minimum(1.0, rnn_col*1.5)
sta_col = "cyan"
sta_col = np.array([49, 204, 199])/255
cols = [sta_col, rnn_col, np.ones(3)*0.6]
for idata, decoding_data in enumerate([all_planning_decoding, all_execution_decoding]):
    mean_accs = np.array([decoding_data[name+"_accs"].mean(0) for name in plot_names])
    std_accs = np.array([decoding_data[name+"_accs"].std(0) for name in plot_names])
    xvals = np.array([decoding_data[name+"_xvals"] for name in plot_names])

    if idata == 0:
        inds = [list(decoding_data[name+"_neural"].astype(int)).index(-1) for name in plot_names]
        assert len(set(inds)) == 1
        ind = inds[0]
        xvals, mean_accs, std_accs = xvals[:, 1:], mean_accs[:, ind, 1:], std_accs[:, ind, 1:]
        
        
    plt.figure(figsize = (1.6, 1.5))
    for i in range(len(plot_names)):
        x, m, s = xvals[i], mean_accs[i], std_accs[i]
        plt.plot(x, m, label = plot_labels[i], color = cols[i], zorder = -i)
        plt.fill_between(x, m-s, m+s, alpha = 0.2, color = cols[i], edgecolor = [1,1,1,0], zorder = -i)
    
    if idata == 0:
        xlabel = "time in future"
    else:
        xlabel = "time from now"

    plt.xticks(x)
    plt.xlabel(xlabel, labelpad = 3.5)
    plt.ylabel("% correctly\npredicted location", labelpad = -7)
    plt.ylim(0, 1)
    plt.xlim(xvals.min(), xvals.max())
    plt.yticks([0, 1])
    
    if idata == 1:
        plt.xticks([-4, -2, 0, 2, 4])
        plt.fill_between([-1, +1], [0, 0], [1,1], color = "k", alpha = 0.07, zorder = -10, linewidth = 0)
    
    plt.gca().spines[['right', 'top']].set_visible(False)
    
    plt.savefig(f"{pysta.basedir}/figures/rnn_decoding/compare_decoding{idata}{ext}", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()
    


#%% plot neural vs behavioural time

ex = (1,3)
plt.figure(figsize = (1.3, 1.5))
for i, name in enumerate(plot_names):
    scores, neural_ts, loc_ts = [all_scores[name] for name in [f"{plot_names[i]}_scores", f"{plot_names[i]}_neural_ts", f"{plot_names[i]}_loc_ts"]]
    neural_ts, loc_ts = [[int(i) for i in arr] for arr in [neural_ts, loc_ts]]
    ex_scores = scores[:, neural_ts.index(ex[0]), loc_ts.index(ex[1]), ...].mean(0)
    pred_ts = np.argmax(ex_scores, axis = -1)
    pred_conf = np.amax(ex_scores, axis = -1)
    not_conf = (pred_conf.mean(0) < 2/9)
    mean_pred = pred_ts + 0.07*((i==0) - (i==1))
    mean_pred[not_conf] = np.nan
    
    plot_ts = [0,1,2,3]
    plot_inds = np.array([list(neural_ts).index(t) for t in plot_ts])
    plt.plot(np.array(neural_ts)[plot_inds], mean_pred[plot_inds], label = plot_labels[i], color = cols[i], zorder = -i)

plt.scatter([ex[0]], [ex[1]], marker = "o", color = [1,1,1,0], s = 100, zorder = 10, edgecolor = "k", lw = 2.0)

xmax, ymax, ymin = 3, 5, 0
plt.ylim(ymin, ymax)
plt.xlim(0, xmax)
plt.yticks(range(ymin, ymax+1))
plt.xticks(range(xmax+1))
plt.gca().spines[['right', 'top']].set_visible(False)
plt.xlabel("time of activity", labelpad = 3.5)
plt.ylabel("time of\npredicted location", labelpad = 3.5)
plt.savefig(f"{pysta.basedir}/figures/rnn_decoding/decoder_generalization{ext}", bbox_inches = "tight", transparent = True)
plt.show()


#%% plot time decoding

plt.figure(figsize = (1.6, 1.5))
for i, name in enumerate(plot_names):
    steps, accs = all_time[f"{name}_steps"], all_time[f"{name}_accs"]
    m, s = accs.mean(0), accs.std(0)
    plt.plot(steps, m, color = cols[i], zorder = -i, label = plot_labels[i])
    plt.fill_between(steps, m-s, m+s, color = cols[i], alpha = 0.2, linewidth = 0)
plt.xlim(steps[0], steps[-1])
plt.ylim(0, 1)
plt.yticks([0, 1])
plt.xticks(steps)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.xlabel("true time in future", labelpad = 3.5)
plt.ylabel("% correctly\npredicted time", labelpad = -7)
plt.legend(loc = "upper center", bbox_to_anchor = (0.6, 0.72), ncol = 1, frameon = False, columnspacing = 0.4, handlelength = 1.2, handletextpad = 0.5)
plt.savefig(f"{pysta.basedir}/figures/rnn_decoding/decode_time_of_loc{ext}", bbox_inches = "tight", transparent = True)
plt.show()


#%%
















