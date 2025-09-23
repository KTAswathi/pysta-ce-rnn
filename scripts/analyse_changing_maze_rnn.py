"""Code for analysing the behaviour, representations, and connectivity of RNNs trained across changing mazes"""

#%% first load some libraries
import numpy as np
import pysta
import pickle
import time
import torch
import copy
import sys
from pysta import basedir

#%% collect data for many different wall configurations

def collect_trial_by_wall_data(model_name, save = True, num_trials_per_wall = 5000, num_walls = 30):
    
    rnn, figdir, datadir = pysta.utils.load_model(model_name, store_all_activity = False) # load the model

    wall_trial_data = []
    wall_configs = []
    adjs = []
    perfs = []

    t0 = time.time()
    rnn.env.changing_trial_maze = False # now we don't automatically resample walls
    for wall_num in range(num_walls): # for each wall configuration
        rnn.env.reset(hard = True)
        trial_data = pysta.analysis_utils.collect_data(rnn, num_trials = num_trials_per_wall) # this just simulates enough batches to get num_trials data and puts it all into a dict
        del trial_data["rews"]
        del trial_data["xs"] # these cost memory for no reason
        
        wall_trial_data.append(trial_data)
        wall_configs.append(rnn.env.walls[0].detach().numpy())
        adjs.append(rnn.env.adjacency[0].detach().numpy())
        perf = rnn.eval(num_eval = 5)
        perfs.append(perf)
        
        print(wall_num, perf, time.time() - t0)
        sys.stdout.flush() # flush the output to make sure we see it in real time

    all_wall_data = {
        "trial_data": wall_trial_data,
        "wall_configs": wall_configs,
        "adjs": adjs,
        "Wrec": rnn.Wrec.detach().numpy(),
        "perfs": perfs,
        "rnn": rnn,
    }

    if save:
        pickle.dump(all_wall_data, open(f"{datadir}trial_data_by_wall.pickle", "wb")) # save the trial data

    return all_wall_data

#%% now identify subspaces

def get_state_subspaces_by_wall(model_name, save = True, split = False):
    """
    split: Bool
        If true, estimate subspaces separately from first and second half of data
    """

    rnn, figdir, datadir = pysta.utils.load_model(model_name, store_all_activity = False) # load the model

    all_wall_data = pickle.load(open(f"{datadir}trial_data_by_wall.pickle", "rb")) # load the trial data

    slot_type = "planning" if rnn.env.working_memory else "relative" # planning subspaces if WM model
    wall_Csubs = []
    t0 = time.time()
    for n, trial_data in enumerate(all_wall_data["trial_data"]):
        print(n, time.time()-t0)
        
        if not split:
            Csubs = pysta.subspace_utils.find_orthogonal_subspaces(trial_data, slot_type = slot_type)
            wall_Csubs.append(Csubs)
        else:
            tot_data = trial_data["rs"].shape[0]
            inds1, inds2 = np.arange(0, tot_data//2), np.arange(tot_data//2, tot_data) # split the data into two halves
            trial_data1, trial_data2 = [{key: value[inds, ...] for key, value in trial_data.items()} for inds in [inds1, inds2]] # split the data into two halves
            Csubs = [pysta.subspace_utils.find_orthogonal_subspaces(trial_data, slot_type = slot_type) for trial_data in [trial_data1, trial_data2]]
            wall_Csubs.append(Csubs)

    sub_data = {
        "Csubs": wall_Csubs,
        "adjs": all_wall_data["adjs"],
        "Wrec": all_wall_data["Wrec"],
        "perfs": all_wall_data["perfs"],
        "wall_configs": all_wall_data["wall_configs"],
    }

    splitstr = "_split" if split else ""
    if save:
        pickle.dump(sub_data, open(f"{datadir}sub_data{splitstr}.pickle", "wb")) # save the trial data

    return sub_data
    
#%% now compute recurrent weight matrices

def compare_weights_to_adjacency(model_name, save = True):
    rnn, figdir, datadir = pysta.utils.load_model(model_name, store_all_activity = False) # load the model

    sub_data = pickle.load(open(f"{datadir}sub_data.pickle", "rb")) # save the trial data
    wall_Csubs, Wrec, adjs = sub_data["Csubs"], sub_data["Wrec"], sub_data["adjs"]
    wall_Weffs = []

    for wall_num, Csubs in enumerate(wall_Csubs):
        Weffs = []
        for idelta, delta in enumerate([-1, 1]): # for each subspace difference
            for mod0 in range(Csubs.shape[0]): # go through subspaces
                mod1 = mod0 + delta # for each paired subspace
                if min(mod0, mod1) >= 0 and max(mod0, mod1) < Csubs.shape[0]: # only consider pairs that are in range
                    if not (min(mod0, mod1) == 0 and max(mod0, mod1) == 1): # don't consider immediate connections
                        # add the effective connectivity to our list
                        Weffs.append(Csubs[mod1] @ Wrec @ Csubs[mod0].T)
        wall_Weffs.append(np.mean(np.array(Weffs), axis = 0)) # average over all pairs


    #%% now compare the effective connectivity to the adjacency matrix
    same, diff = [], []
    same_adj, diff_adj = [], []
    for wall1 in range(len(adjs)):
        A1, Weff1 = adjs[wall1], wall_Weffs[wall1]
        for wall2 in range(len(adjs)):
            A2, Weff2 = adjs[wall2], wall_Weffs[wall2]
            
            sim = pysta.utils.point_biserial_correlation(Weff1, A2)
            sim_adj = pysta.utils.point_biserial_correlation(A1, A2)
            if wall1 == wall2:
                same.append(sim)
                same_adj.append(sim_adj)
            else:
                diff.append(sim)
                diff_adj.append(sim_adj)

    print(np.mean(same), np.mean(diff))
    print(np.mean(same_adj), np.mean(diff_adj))
            
    changing_maze_results = {
        "true_cors": same,
        "false_cors": diff,
        "true_adj_cors": same_adj,
        "false_adj_cors": diff_adj,
        "Weffs": wall_Weffs,
        "adjs": adjs,
        "wall_configs": sub_data["wall_configs"],
        "perfs": sub_data["perfs"],
    }

    if save:
        pickle.dump(changing_maze_results, open(f"{datadir}correlation_results.pickle", "wb")) # save the trial data

    return changing_maze_results

#%% now try to understand what the wall input is like

def get_loc_from_wall_input_ind(ind, rnn):
    num_locs, side_length = rnn.env.num_locs, rnn.env.side_length
    if ind < num_locs: # horizontal
        loc1 = ind
        loc2 = pysta.maze_utils.ind_a_ind(ind, 0, side_length) # the one to the right
    else: # vertical
        loc1 = ind - num_locs # loop back around
        loc2 = pysta.maze_utils.ind_a_ind(loc1, 2, side_length) # the one below
    return loc1, int(loc2)


#%% now we to find slots for 'transitions'
def analyse_wall_to_transition_input(model_name, save = True, num_trials = 30000):

    rnn, figdir, datadir = pysta.utils.load_model(model_name, store_all_activity = False) # load the model
    num_locs = rnn.env.num_locs

    rnn.env.changing_trial_maze = True # make sure to automatically resample walls
    rnn.env.reset(hard = True)
    
    Win = rnn.Win.detach().numpy() # all input weights
    wall_inds = rnn.env.obs_inds()["walls"] # which of these correspond to walls?
    wall_Win_raw = Win[:, wall_inds] # extract them
    Wrec = rnn.Wrec.detach().numpy()

    # first collect data
    trial_data = pysta.analysis_utils.collect_data(rnn, num_trials = num_trials)

    # first define the set of possible transitions
    transitions_self = [(i, i) for i in range(num_locs)] # can transition to self
    transitions_other = [get_loc_from_wall_input_ind(i, rnn) for i in range(2*num_locs)] # can transition to neighbor
    transitions_other = [pair for pair in transitions_other if pair[0] != pair[1]] # we've already done the self transitions
    transitions_other_rev = [(pair[1], pair[0]) for pair in transitions_other] # reverse transitions
    transitions = transitions_self + transitions_other + transitions_other_rev
    assert len(transitions) == len(set(transitions))

    # construct a dict to query them
    trans_to_ind = {trans: i for (i, trans) in enumerate(transitions)}

    # then construct an array with the sequence of transitions in each trial
    locs = trial_data["locs"]
    trial_transitions = np.zeros(locs.shape, dtype = int)[:, :-1, :] # transitions from each loc
    for trial, trial_locs in enumerate(locs[..., 0]):
        for step in range(len(trial_locs)-1):
            trial_transitions[trial, step] = trans_to_ind[(trial_locs[step], trial_locs[step+1])]


    # finally identify transition 'slots'
    trial_data_transitions = copy.deepcopy(trial_data)
    trial_data_transitions["locs"] = trial_transitions # just create a new dict where the 'states' are transitions, then run our normal subspace identification
    slot_type = "planning" if rnn.env.working_memory else "relative" # planning subspaces if WM model
    Csubs_trans = pysta.subspace_utils.find_orthogonal_subspaces(trial_data_transitions, tmax_offset = -1, slot_type = slot_type)

    # store this data
    if save:
        transition_data = {"trial_data": trial_data, "trial_transitions": trial_transitions, "Csubs_transitions": Csubs_trans, "rnn": rnn, "transitions": transitions, "trans_to_ind": trans_to_ind}
        pickle.dump(transition_data, open(f"{datadir}transition_data.pickle", "wb")) # save the trial data

    #%% look at the effective recurrent connectivity between these slots
    # compare connection strength of transition -> endpoint transitions, transition -> other adjacent transitions, and transitions -> random reference points

    transition_data = pickle.load(open(f"{datadir}transition_data.pickle", "rb"))
    trial_transitions, Csubs_trans, rnn, transitions, trans_to_ind = [transition_data[key] for key in ["trial_transitions", "Csubs_transitions", "rnn", "transitions", "trans_to_ind"]]

    Wrec_eff_trans = np.zeros((len(transitions), len(transitions))) # only consider adjacent transitions
    for ind in range(Csubs_trans.shape[0] - 1):
        Wrec_eff_trans += Csubs_trans[ind+1] @ Wrec @ Csubs_trans[ind].T # forward transitions

    # normalize the effective connectivity matrix
    Wrec_eff_trans = (Wrec_eff_trans - np.mean(Wrec_eff_trans))/np.std(Wrec_eff_trans)

    endpoint, adjacent, other = [], [], []
    for itrans, trans in enumerate(transitions): # for each transition
        loc1, loc2 = trans[0], trans[1] # start and end point
        
        new_end, new_adj, new_oth = [], [], []
        for itrans2, trans2 in enumerate(transitions):
            strength = Wrec_eff_trans[itrans2, itrans]
            if trans2[0] == loc2: # these are 'consistent'
                new_end.append(strength)
            elif trans2[0] in rnn.env.nowall_neighbors[loc1]: # another neighboring point but not connected ('adjacent')
                new_adj.append(strength)
            else:
                new_oth.append(strength) # other pairs
                
        if len(new_adj) > 0: # only include transitions with appropriate adjacent controls
            endpoint.append(np.mean(new_end))
            adjacent.append(np.mean(new_adj))
            other.append(np.mean(new_oth))

    print(np.mean(endpoint), np.mean(adjacent), np.mean(other))

    # now look at projection of wall input onto different subspaces
    proj_walls, proj_adjs, proj_others = [], [], []
    # also just store everything
    all_proj = []

    for ex_wall_ind in range(rnn.env.num_locs*2):
        corresponding_locs = np.array(get_loc_from_wall_input_ind(ex_wall_ind, rnn)) # the locations corresponding to the example wall input
        loc1, loc2 = corresponding_locs

        if loc1 != loc2: # should not be the edge of the arena
            
            ex_Win_raw = wall_Win_raw[:, ex_wall_ind] # subspace by location
            
            sub_proj = (Csubs_trans * ex_Win_raw[None, None, :]).sum(-1) # project onto subspaces
            all_proj.append((sub_proj, transitions, corresponding_locs)) # store the projection for later
            
            # normalize within subspace and see if I target mostly transitions
            sub_proj = sub_proj / np.sqrt(np.sum(sub_proj**2, axis = -1, keepdims = True)) # normalize projection
            
            wall_trans, adj_trans, other_trans = [], [], []
            for ind, trans in enumerate(transitions):
                if (trans[0] == loc1 and trans[1] == loc2) or (trans[0] == loc2 and trans[1] == loc1):
                    # transitioning specifically between these two locations in either direction
                    wall_trans.append(ind)
                elif trans[0] in [loc1, loc2]:
                    # other transitions from either location in different directions
                    adj_trans.append(ind)
                else:
                    #completely different
                    other_trans.append(ind)
            
            # extract projection onto these transitions
            wall_proj, adj_proj, other_proj = [sub_proj[:, trans] for trans in [wall_trans, adj_trans, other_trans]]
            # store mean for each class
            proj_walls.append(wall_proj.mean())
            proj_adjs.append(adj_proj.mean())
            proj_others.append(other_proj.mean())


    # print and save results
    print(np.mean(proj_walls), np.mean(proj_adjs), np.mean(proj_others))

    wall_to_transition_input = {"connection_to_endpoint": endpoint, "connection_to_adjacent": adjacent, "connection_to_other": other,
                                "input_to_wall": proj_walls, "input_to_adjacent": proj_adjs, "input_to_other": proj_others,
                                "all_proj": all_proj}

    if save:
        pickle.dump(wall_to_transition_input, open(f"{datadir}wall_to_transition_result.pickle", "wb")) # save the trial data

    return wall_to_transition_input

#%% run a simple state decoding analysis on just one wall configuration

def run_generalized_decoding(model_name, save = True, num_trials = 30000, neural_times = [-2,-1,0,1,2,3,4,5], loc_times = [0,1,2,3,4,5]):

    print("decoding locations")
    sys.stdout.flush()
    rnn, figdir, datadir = pysta.utils.load_model(model_name, store_all_activity = False) # load the model

    # first collect data
    if rnn.env.changing_trial_maze:
        rnn.env.changing_trial_maze = False # now we don't automatically resample walls
        rnn.env.reset(hard = True)
    trial_data = pysta.analysis_utils.collect_data(rnn, num_trials = num_trials) # this just simulates enough batches to get num_trials data and puts it all into a dict

    if save:
        pickle.dump(trial_data, open(f"{datadir}single_wall_trial_data.pickle", "wb")) # save the trial data

    # then run decoding
    cv_result = pysta.analysis_utils.predict_locations_from_neurons(trial_data, crossvalidate_loc = True, neural_times = neural_times, loc_times = loc_times)
    print(np.round(cv_result["nongen_scores"], 2)) # print avg performance

    # save the result
    if save:
        pickle.dump(cv_result, open(f"{datadir}decoder_generalization_performance.pickle", "wb"))

    return cv_result

#%% run transition decoding analyses across wall configurations

def future_transition_decoding(model_name, save = True, neural_times = [-2,-1,0,1,2,3,4], loc_times = [0,1,2,3,4]):
    """
    Run generalized decoding of future transitions
    """
    
    print("decoding transitions")
    sys.stdout.flush()
    rnn, figdir, datadir = pysta.utils.load_model(model_name, store_all_activity = False) # load the model
    
    transition_data = pickle.load(open(f"{datadir}transition_data.pickle", "rb"))
    # overwrite 'location data' to be 'transition data' instead
    trial_data, trial_transitions = [transition_data[key] for key in ["trial_data", "trial_transitions",]]
    trial_data_transitions = copy.deepcopy(trial_data)
    trial_data_transitions["locs"] = trial_transitions # just create a new dict where the 'states' are transitions, then run our normal subspace identification

    # run the decoding (using a reasonable inverse regularization strength based on the previous decoder)
    cv_result = pysta.analysis_utils.predict_locations_from_neurons(trial_data_transitions, crossvalidate_loc = True, neural_times = neural_times, loc_times = loc_times)

    print(np.round(cv_result["nongen_scores"], 2)) # print avg performance for comparison

    # save the result
    if save:
        pickle.dump(cv_result, open(f"{datadir}decoder_transition_generalization_performance.pickle", "wb"))

    return cv_result

if __name__ == "__main__":

    if len(sys.argv) >= 2:
        model_name = sys.argv[1]
        effective_connectivity = ("connectivity" in sys.argv)
        transitions = ("transition" in sys.argv)
        decoding = ("decoding" in sys.argv)
    else:
        effective_connectivity, transitions, decoding = True, True, True
        model_name = "MazeEnv_L4_max6/landscape_changing-rew_dynamic-rew_changing-maze/allo_planrew_plan5-6-7/VanillaRNN/iter10_tau5.0_opt/N800_linout/model31"

    print(f"Running analyses for model: {model_name}.\nEffective connectivity: {effective_connectivity}, Transitions: {transitions}, Decoding: {decoding}")

    seed = int(model_name.split("model")[-1])
    np.random.seed(seed)
    torch.manual_seed(seed)
        
    if effective_connectivity:
        print("\nanalysing effective connectivity")
        collect_trial_by_wall_data(model_name)
        get_state_subspaces_by_wall(model_name)
        get_state_subspaces_by_wall(model_name, split = True)
        compare_weights_to_adjacency(model_name)
        sys.stdout.flush()
        
    if transitions:
        print("\nanalysing wall to transition input")
        analyse_wall_to_transition_input(model_name)
        future_transition_decoding(model_name)
        sys.stdout.flush()
        
    if decoding:
        print("\nrunning decoding analysis")
        run_generalized_decoding(model_name)
        sys.stdout.flush()

    print("\nFinished")
    sys.stdout.flush()

# %%
