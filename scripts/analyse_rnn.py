
"""
Here we run a suite of analyses on a the pre-trained RNNs to query the learned representations and connectivity
"""

#%% load packages
import pysta
import torch
import pickle
import numpy as np
import sys        
from sklearn.linear_model import LogisticRegression
from pysta import basedir

#%% load a model and collect data

def collect_trial_data(model_name, save = True, num_trials = 30000, sta = False, dp = False):

    rnn, figdir, datadir = pysta.utils.load_model(model_name, create_sta = sta, create_dp = dp) # load the model
    rnn.env.planning_steps = int(np.amax(rnn.env.planning_steps)) # match planning steps for simplicity
    trial_data = pysta.analysis_utils.collect_data(rnn, num_trials = num_trials) # this just simulates enough batches to get num_trials data and puts it all into a dict

    # store this data for downstream analyses
    if save:
        pickle.dump(trial_data, open(f"{datadir}trial_data.pickle", "wb")) # save the trial data

    return trial_data

def decode_time_of_location(model_name, save = True, sta = False, dp = False):
    
    rnn, figdir, datadir = pysta.utils.load_model(model_name, create_sta = sta, create_dp = dp) # load the model
    trial_data = pickle.load(open(f"{datadir}trial_data.pickle", "rb")) # load the trial data
    
    #%%
    all_scores = []
    all_preds = []
    all_targets = []
    all_steps = []
    
    for test_loc in range(rnn.env.num_locs):
        step_nums, rs, locs = [trial_data[key] for key in ["step_nums", "rs", "locs"]] # (trials, steps, dim, )
        assert np.nanstd(step_nums[..., 0], 0).sum() == 0 # enforce that our data array is aligned in time-within-trial
        step_nums = np.nanmean(step_nums[..., 0], 0).astype(int)
        
        future_inds = np.where(step_nums >= 0.5)[0] # only in the future
        current_ind = np.where(step_nums == -1)[0][0] # location at the end of planning
        
        future_locs = locs[:, future_inds, 0]
        current_locs = locs[:, current_ind, 0]
        all_steps.append(step_nums[future_inds])
        
        # require the agent to pass the test location exactly once in this trial
        pass_test_loc_once = np.where((future_locs == test_loc).sum(-1) == 1)[0]
        
        new_rs, new_future_locs, new_current_locs = rs[pass_test_loc_once], future_locs[pass_test_loc_once], current_locs[pass_test_loc_once]
        time_at_test = np.where(new_future_locs == test_loc)[1] # time at which the test location is passed
        rs_at_plan = new_rs[:, current_ind, :]
        rs_at_plan = (rs_at_plan -  rs_at_plan.mean(0, keepdims = True)) / (1e-10 + rs_at_plan.std(0, keepdims = True)) # center the data at the first step
        
        
        unique_curr_locs = np.unique(new_current_locs) # unique current locations
        scores = np.zeros(len(unique_curr_locs)) # scores for each current location
        preds = []
        targets = []
        for icurr, curr_loc in enumerate(unique_curr_locs): # hold out each location
            train_inds, test_inds = np.where(new_current_locs != curr_loc)[0], np.where(new_current_locs == curr_loc)[0]
            train_rs, train_ts = rs_at_plan[train_inds], time_at_test[train_inds]

            if len(np.unique(time_at_test)) > 1:
                clf = LogisticRegression(C = 1e-1, max_iter = 100, class_weight = "balanced", tol = 1e-3).fit(train_rs, train_ts)
                scores[icurr] = clf.score(rs_at_plan[test_inds], time_at_test[test_inds]) # print the performance on the test set
            
                preds.append(clf.predict(rs_at_plan[test_inds]))
                targets.append(time_at_test[test_inds])
            
        
        print(test_loc, scores.mean(), len(scores))
        all_scores.append(scores)
        all_preds.append(preds)
        all_targets.append(targets)
            
    # save the result
    if save:
        result = {"all_scores": all_scores, "all_preds": all_preds, "all_targets": all_targets, "all_steps": all_steps}
        pickle.dump(result, open(f"{datadir}decode_time_of_loc.pickle", "wb"))

    return result

def run_rnn_decoding(model_name, save = True, neural_times = [-2,-1,0,1,2,3,4,5], loc_times = [0,1,2,3,4,5], sta = False, dp = False):

    # %% now we run some location decoding analyses

    rnn, figdir, datadir = pysta.utils.load_model(model_name, create_sta = sta, create_dp = dp) # load the model
    trial_data = pickle.load(open(f"{datadir}trial_data.pickle", "rb")) # load the trial data

    #%% tun decoding while crossvalidating across current locations.
    cv_result = pysta.analysis_utils.predict_locations_from_neurons(trial_data, crossvalidate_loc = True, neural_times = neural_times, loc_times = loc_times)
    print(np.round(cv_result["nongen_scores"], 2)) # print avg performance

    # save the result
    if save:
        pickle.dump(cv_result, open(f"{datadir}decoder_generalization_performance.pickle", "wb"))

    return cv_result

def find_rnn_subspaces(model_name, save = True, sta = False, dp = False):

    # %% identify slots/subspaces using decoding, with additional overlap regularization to encourage orthogonality
    rnn, figdir, datadir = pysta.utils.load_model(model_name, create_sta = sta, create_dp = dp) # load the model
    trial_data = pickle.load(open(f"{datadir}trial_data.pickle", "rb")) # load the trial data

    Csubs, Csubs_raw, biases = pysta.subspace_utils.find_orthogonal_subspaces(trial_data, slot_type = "relative", return_all = True) # this also works very well
    Csub_flat = np.concatenate(Csubs, axis = 0) # combine across subpaces

    # compute avg subspace overlap (abs angles between pairs of slot-locations, avg across first location, sum across second location)
    overlap = np.abs(pysta.subspace_utils.calc_overlap(Csubs)).sum((-1, -2)) / rnn.env.num_locs
    print("overlap:\n", np.round(overlap, 2))

    # %% compute effective weight matrices
    try:
        Wrec, Win, Wout = [W.detach().numpy() for W in [rnn.Wrec, rnn.Win, rnn.Wout]]
        if len(Wrec.shape) == 3: # the handcrafted STA has a whole set of recurrent weights across a batch
            Wrec = Wrec[0]

        # we only care about the inputs corresponding to current location and the reward function
        obs_inds = rnn.env.obs_inds() # different parts of the model input
        Win = Win[:, np.concatenate([obs_inds["loc"], obs_inds["goal"]])]

        Wrec_eff = Csub_flat @ Wrec @ Csub_flat.T # (slots, slots)
        Win_eff = Csub_flat @ Win # (slots, inputs)
        Wout_eff = Wout @ Csub_flat.T # (output, slots)


    except AttributeError: # the value-based agent doesn't have any weight matrices
        Wrec, Win, Wout, Wrec_eff, Win_eff, Wout_eff = [None for _ in range(6)]

    # store the results
    connectivity_data = {
        "Csubs": Csubs,
        "Csub_flat": Csub_flat,
        "Csubs_raw": Csubs_raw,
        "biases": biases,
        "Wrec": Wrec,
        "Win": Win,
        "Wout": Wout,
        "Wrec_eff": Wrec_eff,
        "Win_eff": Win_eff,
        "Wout_eff": Wout_eff,
        "rnn": rnn,
        "adjacency": rnn.env.adjacency[0].detach().numpy()
    }

    if save:
        pickle.dump(connectivity_data, open(f"{datadir}connectivity_data.pickle", "wb"))

    # %% finally identify the planning slots as well

    Csubs_p, Csubs_raw_p, biases_p = pysta.subspace_utils.find_orthogonal_subspaces(trial_data, slot_type = "planning", return_all = True)

    if save:
        pickle.dump({"Csubs": Csubs_p, "Csubs_raw": Csubs_raw_p, "biases": biases_p, "rnn": rnn}, open(f"{datadir}planning_subspaces.pickle", "wb"))

    return Csubs, Csubs_p

#%%

if __name__ == "__main__":
    
    if len(sys.argv) >= 2:
        model_name = sys.argv[1]
        collect = ("collect" in sys.argv)
        decoding = ("decoding" in sys.argv)
        subspaces = ("subspaces" in sys.argv)
        time_of_loc = ("time" in sys.argv)
        sta = ("sta" in sys.argv)
        dp = ("dp" in sys.argv)
    else:
        collect, decoding, subspaces, time_of_loc, sta, dp = True, True, True, True, False, False
        model_name = "MazeEnv_L4_max6/landscape_changing-rew_dynamic-rew_constant-maze/allo_planrew_plan5-6-7/VanillaRNN/iter10_tau5.0_opt/N800_linout/model31"

    seed = int(model_name.split("model")[-1])
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Running analyses for model: {model_name}.\nCollect: {collect}, Time: {time_of_loc}, Decoding: {decoding}, Subspaces: {subspaces}, STA: {sta}, DP: {dp}")

    if collect:
        print("\ncollecting data")
        sys.stdout.flush()
        collect_trial_data(model_name, sta = sta, dp = dp)
        
    if time_of_loc:
        print("\ndecoding time of location")
        sys.stdout.flush()
        decode_time_of_location(model_name, sta = sta, dp = dp)
    
    if decoding:
        print("\nrunning decoding")
        sys.stdout.flush()
        run_rnn_decoding(model_name, sta = sta, dp = dp)
    
    if subspaces:
        print("\nfinding subspaces")
        sys.stdout.flush()
        find_rnn_subspaces(model_name, sta = sta, dp = dp)

    print("\nFinished")
    sys.stdout.flush()
