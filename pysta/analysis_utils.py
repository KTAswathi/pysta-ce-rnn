
"""
here we write some functions that might be useful for analysing trained RNNs
"""


import pysta
import numpy as np
import time
import sys
from sklearn.linear_model import LogisticRegression

def collect_data(agent, num_trials = 10000, run_func = None):
    """function for collecting data from an agent running a bunch of trials"""

    # we enforce a constant amount of thinking time for simplicity
    planning_steps_ = agent.env.planning_steps
    agent.env.planning_steps = planning_steps_ if type(planning_steps_) == int else int(np.ceil(np.median(planning_steps_)))

    batch_size = agent.env.batch
    num_batches = int(np.ceil(num_trials / batch_size))
    
    stores = []
    
    max_num_steps = agent.env.max_steps + agent.env.planning_steps # maximum number of steps in each episode
    all_rs, all_step_nums, all_locs, all_inps, all_optimal, all_actions = [np.zeros((num_batches, max_num_steps, batch_size, dim))+np.nan for dim in [agent.Nrec, 1, 1, agent.env.obs_dim, agent.env.output_dim, 1]]
    all_finished = np.ones((num_batches, max_num_steps, batch_size)).astype(bool)
    all_rews, all_vals = [np.zeros((num_batches, batch_size, agent.env.max_steps+1, agent.env.num_locs)) for _ in range(2)]
    all_sample_rews = np.zeros((num_batches, batch_size, agent.env.max_steps+1))

    for batch in range(num_batches):
        
        if run_func is None:
            agent.forward(store = True)
        else:
            run_func(agent)
        
        num_steps = len(agent.store) # number of steps in this batch
        
        # flatten along final dimension. For the handcrafted STA, this gives a vector where consecutive blocks of num_states elements correspond to different slots
        rs = np.array([s["rs"].numpy().reshape(batch_size, -1) for s in agent.store])
        all_rs[batch, :num_steps, ...] = rs
        all_step_nums[batch, :num_steps, ...] = np.array([s["step_num"] for s in agent.store])[..., None, None] #+ np.zeros((1, batch_size, 1))
        all_locs[batch, :num_steps, ...] = np.array([s["loc"].numpy() for s in agent.store])[..., None]
        all_actions[batch, :num_steps, ...] = np.array([s["action"].numpy() for s in agent.store])[..., None]
        all_finished[batch, :num_steps, ...] = np.array([s["finished"].numpy() for s in agent.store])
        all_rews[batch, ...] = agent.env.rews.numpy()
        all_vals[batch, ...] = agent.env.vs.numpy()
        all_inps[batch, :num_steps, ...] = np.array([s["xs"].numpy() for s in agent.store])
        all_optimal[batch, :num_steps, ...] = np.array([s["optimal_actions"].numpy() for s in agent.store])
        all_sample_rews[batch, ...] = agent.env.sample_rews.numpy() # (num_batches, batch_size, max_steps+1)
        all_actions[batch, ]
        
        if int(np.round(100 * batch / num_batches)) % 10 == 0:
            print(f"batch {batch} of {num_batches}")
        
    all_rs, all_step_nums, all_locs, all_finished, all_inps, all_optimal, all_actions = [np.concatenate(arr.swapaxes(1,2), 0) for arr in [all_rs, all_step_nums, all_locs, all_finished, all_inps, all_optimal, all_actions]] # (trials, steps, dim, )
    all_rews, all_vals, all_sample_rews = [np.concatenate(arr, 0) for arr in [all_rews, all_vals, all_sample_rews]] # (trials, steps, num_locs)
    
    num_steps_per_trial = np.sum(~all_finished, axis = -1)
    for arr in [all_rs, all_step_nums, all_locs]:
        for trial in np.arange(arr.shape[0]):
            arr[trial, num_steps_per_trial[trial]+1:, ...] = np.nan # no data when trial is finished
    
    trial_data = {
        "rs": all_rs,
        "step_nums": all_step_nums,
        "locs": all_locs,
        "finished": all_finished,
        "num_steps": num_steps_per_trial,
        "rews": all_rews,
        "vals": all_vals,
        "xs": all_inps,
        "optimal": all_optimal,
        "sample_rews": all_sample_rews,
        "actions": all_actions,
    }
    
    agent.env.planning_steps = planning_steps_

    return trial_data


def predict_locations_from_neurons(trial_data, test_trial_data = None, crossvalidate_loc = True, neural_times = None, loc_times = None, logistic_C = 1e0, normalise = True):
    """
    function for prediction curret, past, and future locations.
    consider restricting to trials that have all combinations of train and test.
    
    Parameters
    --------------
    trial_data : dict
        dictionary of trial data
    crossvalidate_loc : bool
        If True, train and test decoders with different _current_ locations (i.e. neural activity from different locations)
    neural_times : list/array of inds
        timesteps within a trial to use for training
    loc_times : list/array of inds
        timesteps within a trial to predict locations for
    logistic_C : float
        inverse strength of logistic regression regularization
    """
    
    step_nums, rs, locs, finished, num_steps = [trial_data[key] for key in ["step_nums", "rs", "locs", "finished", "num_steps"]] # (trials, steps, dim, )
    
    assert np.nanstd(step_nums[..., 0], 0).sum() == 0 # for now assume that our data array is aligned in time-within-trial (but might end at different times)
    
    if normalise: # normalise neural activity
        rs = (rs - np.nanmean(rs, 0)[None, ...]) / (1e-10+np.nanstd(rs, 0)[None, ...])
    
    step_nums = np.nanmean(step_nums[..., 0], 0).astype(int)
    t0 = step_nums[0] # initial time
    
    # by default, just test all combinations of neural and behavioural data
    if neural_times is None:
        neural_times = np.arange(np.nanmin(step_nums), np.nanmax(step_nums))
        
    if loc_times is None:
        loc_times = np.arange(np.nanmin(step_nums), np.nanmax(step_nums))
    neural_inds, loc_inds = np.arange(len(neural_times)), np.arange(len(loc_times)) # precompute indices
    
    unique_locs = np.unique(locs[~np.isnan(locs)])
    
    if crossvalidate_loc:
        held_out_locs = unique_locs # test each location separately
    else:
        held_out_locs = [None] # no held-out test-locations
    
    initial_time = time.time()
    scores = np.zeros((len(held_out_locs), len(neural_times), len(loc_times), len(neural_times), len(loc_times))) + np.nan
    for iheld, held_out_loc in enumerate(held_out_locs):
        for ineural, neural_time in enumerate(neural_times):
            neural_ind = neural_time-t0
            for iloc, loc_time in enumerate(loc_times):
                loc_ind = loc_time-t0
                possible_inds = np.where(num_steps >= max(neural_ind, loc_ind))[0] # can only run decoding on trials where we have data from the required time points
                
                # extract some data at different time-within-trial for our trials of interest
                rs_at_neural_time, locs_at_neural_time, locs_at_loc_time = rs[possible_inds, neural_ind, ...], locs[possible_inds, neural_ind, ..., 0].astype(int), locs[possible_inds, loc_ind, ..., 0].astype(int)
        
                if held_out_loc is None:
                    # by default just train and test on alternating trials
                    train_trials = np.arange(0, len(rs_at_neural_time), 2)
                else:
                    train_trials = np.where(locs_at_neural_time != held_out_loc)[0] # train on all locations
                
                have_data = False
                if len(train_trials) >= 1:
                    # test and train activity and locations
                    train_rs, train_locs = rs_at_neural_time[train_trials, ...], locs_at_loc_time[train_trials, ...]
                    
                    # fit model
                    clf = LogisticRegression(C = logistic_C, max_iter = 100, class_weight = "balanced", tol = 1e-3).fit(train_rs, train_locs)
                    # test every model on every combination of neural and loc times
                    for itest_neural, neural_test_time in enumerate(neural_times):
                        for itest_loc, loc_test_time in enumerate(loc_times):
                            
                            neural_test_ind = neural_test_time-t0
                            loc_test_ind = loc_test_time-t0
                            possible_test_inds = np.where(num_steps >= max(neural_test_ind, loc_test_ind))[0] # can only run decoding on trials where we have data from the required time points

                            rs_at_neural_test_time, locs_at_neural_test_time, locs_at_loc_test_time = rs[possible_test_inds, neural_test_ind, ...], locs[possible_test_inds, neural_test_ind, ..., 0].astype(int), locs[possible_test_inds, loc_test_ind, ..., 0].astype(int)
        
                            if held_out_loc is None:
                                test_trials = np.arange(1, len(rs_at_neural_test_time), 2) # every other trial
                            else:
                                test_trials = np.where(locs_at_neural_test_time == held_out_loc)[0] # test on this location
                            
                            if len(test_trials) >= 1:
                                have_data = True
                                # evaluate
                                test_rs, test_locs = rs_at_neural_test_time[test_trials, ...], locs_at_loc_test_time[test_trials, ...]
                                score  = clf.score(test_rs, test_locs)
                                
                                # store result (held out loc, neural train time, loc train time, neural test time, loc test time)
                                scores[iheld, ineural, iloc, itest_neural, itest_loc] = score
                if not have_data:
                    print(held_out_loc, neural_time, loc_time, "NO DATA", len(train_trials), len(test_trials))
                
        nongen_scores = scores[iheld][neural_inds, :, neural_inds, :][..., loc_inds, loc_inds] # score on train/test locs
        print(held_out_loc, ":", iheld, "of", len(held_out_locs), "t =", time.time()-initial_time, "\n", np.round(nongen_scores, 2))
        sys.stdout.flush()

    # avg over held-out locs
    mean_scores = np.nanmean(scores, 0)
    # scores at train/test times
    nongen_scores = mean_scores[neural_inds, :, neural_inds, :][..., loc_inds, loc_inds]
    result = {"scores": mean_scores, "nongen_scores": nongen_scores, "neural_times": neural_times, "loc_times": loc_times, "held_out_locs": held_out_locs}
    return result
    



