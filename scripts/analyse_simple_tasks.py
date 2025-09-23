"""Here we analyse STA and RNN representations in the static and moving goal tasks"""

#%% load packages
import pysta
from pysta import basedir
import torch
import pickle
import copy
import numpy as np              
import matplotlib.pyplot as plt
import sys
from sklearn.linear_model import LogisticRegression
import os
np.random.seed(0)

#%% define some helper functions

def reset_env_by_dist_static(rnn, min_dist, max_dist, val_pos, val_neg):
    dists = [pysta.maze_utils.compute_shortest_dists(adj.numpy()) for adj in rnn.env.adjacency]
    
    possible_starts = [np.where(np.amax(dists[i], 0) >= min_dist)[0] for i in range(len(dists))]
    
    rnn.env.goal = torch.tensor([np.random.choice(possible_starts[i]) for i in range(len(possible_starts))])
    rnn.env.goal = rnn.env.goal[:, None] + torch.zeros(rnn.env.batch, rnn.env.max_steps+1, dtype = int)
    rnn.env.rews = torch.zeros(rnn.env.batch, rnn.env.max_steps+1, rnn.env.num_locs) + val_neg # 0.1 #default small negative reward
    for t in range(rnn.env.max_steps+1):
        rnn.env.rews[rnn.env.batch_inds, t, rnn.env.goal[:, t]] = val_pos #0.8 # positive reward at the goal location

    rnn.env.compute_value_function()

    goal = rnn.env.goal.numpy()[:,0]
    
    # make sure always exactly between min_dist and max_dist away
    new_loc = np.array([np.random.choice(np.where( (dists[i][:, goal[i]] >= min_dist) & (dists[i][:, goal[i]] <= max_dist) )[0]) for i in range(len(goal))]) # set the location to a random one that is 6 away from the goal
 
    rnn.env.loc = torch.tensor(new_loc)
    
    return


def reset_env_by_dist_dynamic(rnn, min_dist, max_dist, val_pos, val_neg):
    
    vmin, vmax = [dist*val_neg + (rnn.env.max_steps-dist+1)*val_pos for dist in [max_dist, min_dist]] # compute the minimum and maximum value of the reward function
    vmin_t, vmax_t = vmin - 0.5*np.abs(val_neg), vmax + 0.5*np.abs(val_neg) # add a little bit of margin to the value function

    rnn.reset() # reset agent
    invalid_inds = np.arange(rnn.env.batch)
    
    while len(invalid_inds) > 0:
        
        new_path = rnn.env.sample_path(rnn.env.max_steps+1) #(batch, max_steps)
        rnn.env.goal[invalid_inds, :] = new_path[invalid_inds, :] 
        
        rnn.env.rews = torch.zeros(rnn.env.batch, rnn.env.max_steps+1, rnn.env.num_locs) + val_neg # 0.1 #default small negative reward
        for t in range(rnn.env.max_steps+1):
            rnn.env.rews[rnn.env.batch_inds, t, rnn.env.goal[:, t]] = val_pos #0.8 # positive reward at the goal location

        rnn.env.compute_value_function()
        
        new_invalid_inds = []
        for b in invalid_inds:
            v0s = rnn.env.vs[b, 0, :].detach().numpy()
            possible_starts = np.where((v0s > vmin_t) & (v0s < vmax_t))[0]
            if len(possible_starts) > 0:
                rnn.env.loc[b] = np.random.choice(possible_starts) # set the location to a random one that is between min_dist and max_dist away from the goal
            else:
                new_invalid_inds.append(b)
        invalid_inds = np.array(new_invalid_inds)
        
    return


def run_func(rnn, min_dist = 5, max_dist = 6):
    
    val_neg, val_pos = -0.6, 0.6

    rnn.reset() # reset agent
    if rnn.env.dynamic_rew:
        reset_env_by_dist_dynamic(rnn, min_dist, max_dist, val_pos, val_neg)
    else:
        reset_env_by_dist_static(rnn, min_dist, max_dist, val_pos, val_neg)

    while not torch.all(rnn.env.finished): # as long as there are some trials left to act in
        rnn.update_optimal_actions() # cache the optimal actions at the current location
        
        x = rnn.env.observation().to(rnn.z0.device) # observation at this point in time
        rnn.step(x).to(x.device) # update RNN state, compute policy, and sample an action
        
        # now update environment and optionally store env+agent state (don't propagate gradients through this)
        with torch.no_grad():
            rnn.update_store()
            rnn.env.step(rnn.env_action) # action passed to the environment (optionally restricted to be optimal)


#%% now define the functions for collecting data

def get_model(base_model_name, model_type):
    
    assert model_type in ["base_rnn", "sta"]

    base_static = "static-rew" in base_model_name
    base_wm = "planrew" in base_model_name
    basetask = ["moving", "static"][base_static]+"_"+["relrew", "planrew"][int(base_wm)]
    rnn, _, datadir = pysta.utils.load_model(base_model_name, create_sta = model_type == "sta") # load the model

    return rnn, datadir, basetask

def collect_trial_data(base_model_name, model_type, keep_only_good_trials = True, min_dist = 3, max_dist = 6, num_trials = 20000, save = True):

    rnn, datadir, basetask = get_model(base_model_name, model_type)
    sys.stdout.flush()

    def dist_run_func(rnn):
        """wrapper for collecting data"""
        return run_func(rnn, min_dist = min_dist, max_dist = max_dist) 

    trial_data = pysta.analysis_utils.collect_data(rnn, num_trials = num_trials, run_func = dist_run_func) # this just simulates enough batches to get num_trials data and puts it all into a dict

    print(trial_data["sample_rews"].mean(0))
    print(np.mean(trial_data["sample_rews"].max(-1) >= 0.3))
    sys.stdout.flush()

    if keep_only_good_trials:
        good_trials = np.where(trial_data["sample_rews"].max(-1) >= 0.3)[0]
        for key, value in trial_data.items():
            trial_data[key] = value[good_trials, ...]
            
    assert np.nanstd(trial_data["step_nums"][..., 0], 0).sum() == 0
    step_nums = np.nanmean(trial_data["step_nums"][..., 0], 0).astype(int)
    trial_data["step_nums"][..., 0] = step_nums[None, :]
            
    # store this data for downstream analyses
    if save:
        pickle.dump(trial_data, open(f"{datadir}simple_{basetask}_trial_data_minmax{min_dist}-{max_dist}.pickle", "wb")) # save the trial data

    return trial_data

#%% functions for running analyses

def run_rnn_decoding(base_model_name, model_type, min_dist = 3, max_dist = 6, save = True):
    
    rnn, datadir, basetask = get_model(base_model_name, model_type)
    trial_data = pickle.load(open(f"{datadir}simple_{basetask}_trial_data_minmax{min_dist}-{max_dist}.pickle", "rb")) # load the trial data

    # run the decoding (using a reasonable inverse regularization strength based on the previous decoder)
    neural_times, loc_times = np.arange(-1, min_dist), np.arange(min_dist+1)
    cv_result = pysta.analysis_utils.predict_locations_from_neurons(trial_data, crossvalidate_loc = True, neural_times = neural_times, loc_times = loc_times)

    print(np.round(cv_result["nongen_scores"], 2)) # print avg performance for comparison

    # save the result
    if save:
        pickle.dump(cv_result, open(f"{datadir}simple_{basetask}_decoder_generalization_performance_minmax{min_dist}-{max_dist}.pickle", "wb"))

    return cv_result


def decode_from_planning(base_model_name, model_type, min_dist = 3, max_dist = 6, save = True, num_trials = 10000, keep_only_good_trials = True):
    
    # try to decode from planning period whether a location will be visited _at some point_ (not start or goal)
    
    rnn, datadir, basetask = get_model(base_model_name, model_type)
    trial_data = pickle.load(open(f"{datadir}simple_{basetask}_trial_data_minmax{min_dist}-{max_dist}.pickle", "rb")) # load the trial data
    assert np.nanstd(trial_data["step_nums"][..., 0], 0).sum() == 0
    step_nums = np.nanmean(trial_data["step_nums"][..., 0], 0).astype(int)

    alt_trial_data = copy.deepcopy(trial_data) # copy the trial data so we can normalise it for the decoder
    rs = alt_trial_data["rs"]
    alt_trial_data["rs"] = (rs - np.nanmean(rs, 0)[None, ...]) / (1e-10+np.nanstd(rs, 0)[None, ...])

    #%% try to decode location at _any_ time
    neural_time = -1
    neural_ind = list(step_nums).index(neural_time)
    curr_locs = alt_trial_data["locs"][:, neural_ind, 0]
    target_locs = alt_trial_data["locs"][:, list(step_nums).index(1):, 0]
    anytime_scores = np.zeros((rnn.env.num_locs, rnn.env.num_locs)) + np.nan
    for curr_loc in np.arange(rnn.env.num_locs):
        test_inds = np.where(curr_locs == curr_loc)[0]
        train_inds = np.where(curr_locs != curr_loc)[0]
        train_rs = alt_trial_data["rs"][train_inds, neural_ind, :]
        test_rs = alt_trial_data["rs"][test_inds, neural_ind, :]
        for target_loc in np.arange(rnn.env.num_locs):
            if curr_loc != target_loc:
                ys = np.array([target_loc in locs for locs in target_locs])
                train_ys = ys[train_inds]
                test_ys = ys[test_inds]
                if train_ys.sum() > 0: # if we sometimes pass through

                        # fit model
                        logistic_C = 1e-2
                        clf = LogisticRegression(C = logistic_C, max_iter = 100, class_weight = "balanced", tol = 1e-3).fit(train_rs, train_ys)
                        
                        score  = clf.score(test_rs, test_ys)
                        
                        # store result (held out loc, neural train time, loc train time, neural test time, loc test time)
                        anytime_scores[curr_loc, target_loc] = score

        print(curr_loc, np.nanmean(anytime_scores[curr_loc, :]))
        sys.stdout.flush()
                        
    print(np.nanmean(anytime_scores, axis = 1))
    sys.stdout.flush()
    
    #%%
    if save:
        pickle.dump(anytime_scores, open(f"{datadir}simple_{basetask}_decode_from_planning_minmax{min_dist}-{max_dist}.pickle", "wb"))

    return


#%%

if __name__ == "__main__":
    
    if len(sys.argv) >= 2:
        base_model_name = sys.argv[1]
        model_type = sys.argv[2]
        collect = ("collect" in sys.argv)
        decoding = ("decoding" in sys.argv)
        from_planning = ("planning" in sys.argv)
    else:
        collect, decoding, from_planning = True, True, True
        model_type = "base_rnn"
        base_model_name = "MazeEnv_L4_max6/goal_changing-rew_static-rew_constant-maze/allo_planrew_plan5-6-7/VanillaRNN/iter10_tau5.0_opt/N800_linout/model31"

    min_dist, max_dist = 3, 6
    seed = int(base_model_name.split("/model")[-1])
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Running analyses for model: {base_model_name}.\nModel type: {model_type}, Collect: {collect}, Decoding: {decoding}")

    if collect:
        print("\ncollecting data")
        collect_trial_data(base_model_name, model_type, min_dist = min_dist, max_dist = max_dist, save = True)
    
    if decoding:
        print("\nrunning decoding during execution")
        run_rnn_decoding(base_model_name, model_type, min_dist = min_dist, max_dist = max_dist, save = True)

    if from_planning:
        print("\nrunning decoding from planning")
        decode_from_planning(base_model_name, model_type, min_dist = min_dist, max_dist = max_dist, save = True)

    print("\nFinished")

