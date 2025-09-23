"""Code for plotting all RNN connectivity panels"""

#%% load libraries
import pysta
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy
import torch
import os
import matplotlib as mpl
from pysta import basedir
ext = ".pdf"
basefigdir = f"{basedir}/figures/rnn_connectivity/"
np.random.seed(0)

#%% set font with arial .ttf file
import matplotlib as mpl
import matplotlib.font_manager as fm
font_path = f"{basedir}/data/arial.ttf"
fm.fontManager.addfont(font_path)
mpl.rcParams['font.family'] = "Arial"
mpl.rcParams['font.size'] = 8

#%% set some parameters
ext = ".pdf"
seeds = [31,32,33,34,35]

basenames = {"WM": "MazeEnv_L4_max6_landscape_changing-rew_dynamic-rew_constant-maze_allo_planrew_plan5-6-7_VanillaRNN_iter10_tau5.0_opt_N800_linout_model",
            "STA":
            "sta_MazeEnv_L4_max6_landscape_changing-rew_dynamic-rew_constant-maze_allo_planrew_plan5-6-7_VanillaRNN_iter10_tau5.0_opt_N800_linout_model",
            "relrew": "MazeEnv_L4_max6_landscape_changing-rew_dynamic-rew_constant-maze_allo_relrew_plan5-6-7_VanillaRNN_iter9-10-11_tau5.0_opt_N800_linout_model",
            "egocentric": "MazeEnv_L4_max6_landscape_changing-rew_dynamic-rew_constant-maze_ego_planrew_plan5-6-7_VanillaRNN_iter10_tau5.0_opt_N800_linout_model",}


#%% some functions to load things

def construct_true_Csubs(Csubs, rnn):
    Csubs_true = np.zeros(Csubs.shape)
    baserange = np.arange(rnn.env.num_locs)
    for i in range(len(Csubs)):
        inds = baserange + i * rnn.env.num_locs
        Csubs_true[i, baserange, inds] = 1
    return Csubs_true

def get_model_and_subs(model, seed, subspace_type):
    model_name = basenames[model.removesuffix("_true") ]+ str(seed)
    if subspace_type == "planning":
        connectivity_data = pickle.load(open(f"{basedir}/data/rnn_analyses/{model_name}_planning_subspaces.pickle", "rb"))
    else:
        connectivity_data = pickle.load(open(f"{basedir}/data/rnn_analyses/{model_name}_connectivity_data.pickle", "rb"))
        

    Csubs = connectivity_data["Csubs"]
    if "true" in model:
        Csubs = construct_true_Csubs(Csubs, rnn)
    
    return connectivity_data, Csubs

#%%

models = ["WM", "STA", "STA_true", "relrew", "egocentric"]
ex_seed = 0
for imodel, model in enumerate(models):
    os.makedirs(f"{basefigdir}{model}/", exist_ok = True)
    os.makedirs(f"{basefigdir}{model}/input_panels", exist_ok = True)
    
    for subspace_type in ["planning", "rel"]:
        
        substr = "_plan" if subspace_type == "planning" else ""
        
        #%%
        
        # load data
        connectivity_data, Csubs = get_model_and_subs(model, seeds[ex_seed], subspace_type)
        Csub_flat = np.concatenate(Csubs, axis = 0)
        
        rnn = connectivity_data["rnn"]
        Wrec, Win, Wout = [W.detach().numpy() for W in [rnn.Wrec, rnn.Win, rnn.Wout]]
        if len(Wrec.shape) == 3: # the handcrafted STA has a whole set of recurrent weights across a batch
            Wrec = Wrec[0]
        # we only care about the inputs corresponding to current location and the reward function
        obs_inds = rnn.env.obs_inds() # different parts of the model input
        Win = Win[:, np.concatenate([obs_inds["loc"], obs_inds["goal"]])]
        adj = rnn.env.adjacency[0].detach().numpy()
        
        A2, A3 = adj @ adj, adj @ adj @ adj
        num_locs = adj.shape[0]

        #%% plot all input weights
        figsize = (3.4, 3.4)
        num_slots = 5
        Win_eff = Csub_flat @ Win # (slots, inputs)
        keep_input_inds = np.concatenate([np.arange(16), np.arange(32, 32+(num_slots-1)*16)])
        keep_slot_inds = np.arange(num_slots*16) # keep only the first 5 slots
        Win_eff_plot = Win_eff[:, keep_input_inds][keep_slot_inds, :]
        xticks = ["location"]+[r"$\delta$"+f"={i} reward" for i in range(1, num_slots)]
            
        pysta.plot_utils.plot_slot_connectivity(Win_eff_plot, num_locs, filename = f"{basefigdir}{model}/input_weights{substr}{ext}", show = True,
                                                xticks = xticks, figsize = figsize,
                                                yticks = range(0, num_slots), ylabel = "subspace number", transparent = True)

        #%% plot all output weights
        Wout_eff = Wout @ Csub_flat.T # (output, slots)
        pysta.plot_utils.plot_slot_connectivity(Wout_eff[:, keep_slot_inds].T, num_locs, xtickrot=0, xticks = [], xlabel = "output", yticks = range(0, num_slots),
                                                filename = f"{basefigdir}{model}/output_weights{substr}{ext}", show = True, figsize = figsize, ylabel = "subspace number", transparent = True)
    
    
        #%% all recurrent weights

        Wrec_eff = Csub_flat @ Wrec @ Csub_flat.T # (slots, slots)
        Wrec_eff_plot = Wrec_eff[keep_slot_inds, :][:, keep_slot_inds] # (slots, slots)
        pysta.plot_utils.plot_slot_connectivity(Wrec_eff_plot, num_locs, filename = f"{basefigdir}{model}/recurrent_weights{substr}{ext}", show = True, figsize = figsize, vmin = 0.2, vmax = 0.97, transparent = True,
                                                yticks = [str(i) for i in range(0, num_slots)], xticks = [str(i) for i in range(0, num_slots)], xlabel = "input subspace", ylabel = "output subspace", xtickrot = 0)

        #%% also plot 1 input row at a time
        num_slots = 3
        keep_input_inds = np.concatenate([np.arange(16), np.arange(32, 32+(num_slots-1)*16)])
        keep_slot_inds = np.arange(num_slots*16) # keep only the first 3 slots
        Win_eff_plot = Win_eff[:, keep_input_inds][keep_slot_inds, :]
        
        if model in ["WM", "relrew", "egocentric"]:
            xlabel = "input"
            xticks = xticks[:num_slots]
        else:
            xticks = []
            xlabel = None

        figsize = (3.5, 2) if model in ["relrew", "egocentric"] else (2.65,1.33)
        for irow, Win_row in enumerate(Win_eff_plot.reshape(num_slots,num_locs,num_slots*num_locs)):
            ax = pysta.plot_utils.plot_slot_connectivity(Win_row, num_locs, xticks = xticks, xtickrot=0, yticks = [],
                                                        show = False, figsize = figsize, transparent = True)
            plt.xlabel(xlabel, labelpad = +1)
            plt.ylabel(f"subspace {irow}", labelpad = 2.5)
            ax.tick_params(axis='x', which='major', pad=2, length = 0)
            plt.savefig(f"{basefigdir}{model}/input_row{irow}{substr}{ext}", transparent = True)
            plt.show()
            plt.close()
            
            # annoyingly we also have to plot 1 _Panel_ at a time to avoid rasterization
            for ipanel, panel in enumerate(Win_row.reshape(num_locs, -1, num_locs).transpose(1,0,2)):
                plt.figure(figsize = (np.amin(figsize), np.amin(figsize)))
                plt.imshow(panel, cmap = "coolwarm", vmin = np.amin(Win_row), vmax = np.amax(Win_row))
                plt.gca().spines[['right', 'top', 'bottom', 'left']].set_visible(False)
                plt.yticks([])
                plt.xticks([])
                plt.savefig(f"{basefigdir}{model}/input_panels/{irow}_{ipanel}{substr}{ext}", transparent = True)
                plt.show()
                plt.close()
            

        #%% plot average n+1 connectivity together with adjacency matrices

        Ws = []
        for i in range(1, len(Csubs)-1): # don't include delta=0 <-> delta=1 connectivity
            Ws.append(Csubs[i] @ Wrec @ Csubs[i+1].T)
            Ws.append(Csubs[i+1] @ Wrec @ Csubs[i].T)
        W12 = np.mean(np.array(Ws), axis = 0)

        for imat, mat in enumerate([W12, np.eye(adj.shape[0]), adj, np.sign(A2), np.sign(A3)]):
            plt.figure(figsize = (1.0,1.0))
            plt.imshow(mat, cmap = "coolwarm", vmin = np.quantile(mat, 0.25), vmax = np.quantile(mat, 0.94))
            plt.xticks([])
            plt.yticks([])
            plt.savefig(basefigdir + model+"/"+[f"W12{substr}", "A0", "A1", "A2", "A3"][imat] + ext, bbox_inches = "tight", transparent = True)
            plt.show()
            plt.close()

        # %% now plot similarity of different connectivity matrices to different powers of the adjacency matrix

        abs_deltas = np.arange(0, 4) # subspace differences to look at
        deltas = np.arange(-abs_deltas.max(), abs_deltas.max() + 1) # both positive and negative deltas
        all_Wavgs, all_scales, all_scores, all_Arefs, all_scores_abs, all_Wavgs_abs = [], [], [], [], [], []
        for seed in seeds:

            # load data
            connectivity_data_m, Csubs_m = get_model_and_subs(model, seed, subspace_type)
            
            Wrec_m = connectivity_data_m["rnn"].Wrec.detach().numpy()
            if len(Wrec_m.shape) == 3: # the handcrafted STA has a whole set of recurrent weights across a batch
                Wrec_m = Wrec_m[0]
            adj_m = connectivity_data_m["rnn"].env.adjacency[0].detach().numpy()
            # compute powers of the adjacency matrix for comparison. Open question whether to only use the sign. Should investigate empirically what the RNN learns.
            Arefs = [torch.matrix_power(torch.tensor(adj_m), i).numpy() for i in range(4)]
            Wavgs, Wavgs_abs = [], []

            abs_scores = np.zeros((len(abs_deltas), len(Arefs))) # effective weight matrix match to each adjacency matrix power
            scales = np.zeros((len(abs_deltas))) # also look at the 'scale' (std) of the connectivity for each delta
            scores = np.zeros((2, len(abs_deltas), len(Arefs)))
            for idelta, abs_delta in enumerate(abs_deltas): # for each subspace difference
                Weffs = [] # initialize effective weight matrix
                for isign, delta in enumerate([-abs_delta, +abs_delta]): # consider positive and negative together
                    Weffs.append([])
                    for mod0 in range(Csubs_m.shape[0]): # go through subspaces
                        mod1 = mod0 + delta # for each paired subspace
                        if min(mod0, mod1) >= 0 and max(mod0, mod1) < Csubs_m.shape[0]: # only consider pairs that are in range
                            if not ((min(mod0, mod1) == 0 and max(mod0, mod1) == 1) or (mod0 == 0 and mod1 == 0)): # don't consider immediate connections
                                # the effective connectivity to our list
                                Weffs[-1].append(Csubs_m[mod1] @ Wrec_m @ Csubs_m[mod0].T)
                    
                Weffs = np.array(Weffs)
                Weff = np.mean(Weffs, axis = 1) # average across pairs of subspaces
                Weff_abs = Weff.mean(0) # also average positive and negative
                
                Wavgs.append(Weff)
                Wavgs_abs.append(Weff_abs)

                # compute the point-biserial correlation
                abs_scores[idelta, :] = np.array([pysta.utils.point_biserial_correlation(Weff_abs, A) for A in Arefs])
                for isign in range(2):
                    scores[isign, idelta, :] = np.array([pysta.utils.point_biserial_correlation(Weff[isign, ...], A) for A in Arefs])
                
                # compute std
                scales[idelta] = Weffs.std((-1,-2)).mean() # average across pairs of subspaces and sign
                
            all_scales.append(scales)
            all_scores.append(scores)
            all_scores_abs.append(abs_scores)
            all_Wavgs.append(Wavgs)
            all_Wavgs_abs.append(Wavgs_abs)
            all_Arefs.append(Arefs)
            
        Wavgs_abs = np.array(all_Wavgs_abs[ex_seed])
        Wavgs = np.array(all_Wavgs[ex_seed])
        Arefs = all_Arefs[ex_seed]
        
        #%% compute within vs. across maze similarity
        same, diff = [], []
        for i1 in range(len(all_Wavgs_abs)):
            W1 = all_Wavgs_abs[i1][1]
            for i2 in range(len(all_Wavgs_abs)):
                A2 = all_Arefs[i2][1]
                sim = pysta.utils.point_biserial_correlation(W1, A2)
                if i1 == i2:
                    same.append(sim)
                else:
                    diff.append(sim)
                    
        if (model == "WM") and (subspace_type == "planning"):
            print("Within maze correlation:", np.mean(same), np.std(same))
            print("Across maze correlation:", np.mean(diff), np.std(diff))

        #%% plot the average weight matrix for different distances

        vmin, vmax = np.quantile(Wavgs_abs, [0.10, 0.95])
        for imat, mat in enumerate(Wavgs_abs):
            plt.figure(figsize = (3,3))
            plt.imshow(mat, cmap = "coolwarm", vmin = vmin, vmax = vmax)
            plt.xticks([])
            plt.yticks([])
            plt.savefig(f"{basefigdir}{model}/avg_connectivity_{abs_deltas[imat]}{substr}{ext}", bbox_inches = "tight", transparent = True)
            plt.title(f"{abs_deltas[imat]}")
            plt.show()
            plt.close()


        vmin, vmax = np.quantile(Wavgs, [0.10, 0.95])
        for imat, mat in enumerate(Wavgs):
            for isign in range(2):
                plt.figure(figsize = (3,3))
                plt.imshow(mat[isign], cmap = "coolwarm", vmin = vmin, vmax = vmax)
                plt.xticks([])
                plt.yticks([])
                plt.savefig(f"{basefigdir}{model}/avg_connectivity_{abs_deltas[imat]}-{isign}{substr}{ext}", bbox_inches = "tight", transparent = True)
                plt.title(f"{abs_deltas[imat]}-{isign}")
                plt.show()
                plt.close()


        #%% plot average connectivity as a projection
        
        ind = 6
        walls = rnn.env.walls[0].numpy()
        weights = Wavgs_abs[:, ind, :] + Wavgs_abs[:, :, ind]
        weights = (weights - weights.mean()) / weights.std()
        
        sequence_colors = [plt.get_cmap("viridis")(iind / 5 + 0.35) for iind in range(4)][::-1]
        edgecolors = [[1,1,1,0] for _ in range(weights.size)]
        for i in range(len(weights)):
            adjs = np.where(Arefs[i][ind, :] > 0)[0]
            for adj in adjs:
                edgecolors[16*i + adj] = sequence_colors[1]
        
        pysta.plot_utils.plot_perspective_attractor(walls, weights, vmin = -0.6, vmax = 1.9, cmap = "coolwarm", filename = f"{basefigdir}{model}/avg_proj{substr}{ext}",
                                                    lw = 4, plot_proj = False, figsize = (7,4), aspect = (1,1,3.6), view_init = (-38,-10,-90), show = True, edgecolors = edgecolors,
                                                    bbox_inches = mpl.transforms.Bbox([[2.0,1.35], [5.1,2.65]]), transparent = True)
                    

        #%% now plot the similarity plots
        figsize = (1.9, 1.25)
        for isign in range(3):
            plot_deltas = abs_deltas
            if isign == 0:
                plot_scores = np.array(all_scores_abs)
            elif isign == 1:
                plot_scores = np.array(all_scores)[:, 0, ...]
            else:
                plot_scores = np.array(all_scores)[:, 1, ...]

            # plot the similarity of each effective weight matrix to each reference power of the adjacency matrix
            plt.figure(figsize = figsize)
            for idelta, score in enumerate(np.transpose(plot_scores, (1,0,2))):
                m, s = score.mean(axis = 0), score.std(axis = 0)
                plt.plot(plot_deltas, m, label = f"{idelta}")
                plt.fill_between(plot_deltas, m-s, m+s, alpha = 0.2, linewidth = 0)
            plt.xlabel("order of adjacency matrix", labelpad = 3.5)
            plt.ylabel("similarity", labelpad = 2.5)
            plt.gca().spines[['right', 'top']].set_visible(False)
            plt.legend(loc = "upper center", bbox_to_anchor = (0.5, 1.15), ncol = 4,
                    handlelength = 1.2, handletextpad = 0.5, columnspacing = 0.8, frameon = False)
            plt.xticks(plot_deltas)
            plt.xlim(0,3)
            plt.savefig(f"{basefigdir}{model}/similarity{substr}{isign}{ext}", bbox_inches = "tight", transparent = True)
            plt.show()
            plt.close()


        # plot the parameter scale
        plt.figure(figsize = figsize)
        m, s = np.array(all_scales).mean(0), np.array(all_scales).std(0)
        plt.plot(abs_deltas, m, color = "k")
        plt.fill_between(abs_deltas, m-s, m+s, alpha = 0.2, color = "k", linewidth = 0)
        plt.xlabel("subspace difference")
        plt.ylabel("average strength")
        plt.gca().spines[['right', 'top']].set_visible(False)
        plt.xticks(abs_deltas)
        plt.savefig(f"{basefigdir}{model}/scale{substr}{ext}", bbox_inches = "tight", transparent = True)
        plt.show()
        plt.close()


#%% plot planning >< execution overlap

all_overlaps = []
for imodel, model in enumerate(["WM", "relrew"]):
    overlaps = []
    for seed in seeds:
        new_Cs = []
        for subspace_type in ["planning", "rel"]:
        
            # load data
            connectivity_data_m, Csubs_m = get_model_and_subs(model, seed, subspace_type)
            new_Cs.append(Csubs_m)
        
        overlaps.append((new_Cs[0][0:-1, ...] * new_Cs[1][0:, ...]).sum(-1))
    all_overlaps.append(overlaps)  

all_overlaps = np.array(all_overlaps)

data = all_overlaps.mean((-1, -2)).T

xs, ms, ss = np.arange(data.shape[1]), np.mean(data, axis = 0), np.std(data, axis = 0)
jitters = np.random.normal(0, 0.1, len(data)) # jitter for plotting

plt.figure(figsize = (1.5,1.8))

plt.bar(xs, ms, yerr = ss, capsize = 3, error_kw={'elinewidth': 2, "markeredgewidth": 2})
for idata, datapoints in enumerate(data.T):
    plt.scatter(jitters+xs[idata], datapoints, marker = ".", color = "k", alpha = 0.5, linewidth = 0.0, s = 80)
    
plt.ylabel("overlap")
plt.ylim(0, 1)
plt.xticks(xs, ["WM", "continual"])#, rotation = 45, ha = "right")
plt.gca().spines[['right', 'top']].set_visible(False)
plt.savefig(f"{basedir}/figures/rnn_connectivity/subspace_overlap{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()

#%%

