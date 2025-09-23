"""Code for analysing how well reward and value predict RNN choices"""

#%%

import pysta
import numpy as np
import pickle
import torch
import sys
import copy
from pysta import basedir

def analyse_rnn_behaviour(base_model_name, seeds, save = True):

    base_model_name = base_model_name.split("model")[0]+"model"
    model_names = [f"{base_model_name}{seed}" for seed in seeds]

    results = {}

    for model_name in model_names:
        
        #%% collect some data
        
        rnn, figdir, datadir = pysta.utils.load_model(model_names[0], create_sta = False, create_dp = False) # load the model
        rnn.env.planning_steps = int(np.amax(rnn.env.planning_steps)) # match planning steps for simplicity
        data = pysta.analysis_utils.collect_data(rnn, num_trials = 30000) # this just simulates enough batches to get num_trials data and puts it all into a dict
        results[model_name] = {"rnn": rnn}

        #%% compute %optimal against action number
        actions, optimal_actions = copy.deepcopy(data["actions"])[..., 0].astype(int), copy.deepcopy(data["optimal"])
        
        step_nums = np.mean(data["step_nums"], axis = 0)[:, 0]
        frac_optimal = []
        rand_optimal = []

        for ind in np.where(step_nums >= 0)[0]:
            
            opt_bools = optimal_actions[np.arange(len(optimal_actions)), ind, actions[:, ind]].astype(bool)
            frac_optimal.append(opt_bools.mean())
            rand_optimal.append(optimal_actions[:, ind, :].mean())

            actions, optimal_actions = actions[opt_bools, ...], optimal_actions[opt_bools, ...]

        results[model_name]["frac_optimal"] = np.array(frac_optimal)
        results[model_name]["rand_optimal"] = np.array(rand_optimal)
        
        #%% store info on rewards, values, locations, and actions
        
        ind = np.where(step_nums == 0)[0][0]
        loc0 = data["locs"][:, ind, 0] # initial location
        val0 = copy.deepcopy(data["vals"][:, 1, :]) # relevant value is at the _next_ time step
        rew0 = copy.deepcopy(data["rews"][:, 1, :]) # relevant reward is at the _next_ time step
        act0 = data["actions"][:, ind, 0]
        
        results[model_name]["loc0"] = loc0
        results[model_name]["rew0"] = rew0
        results[model_name]["val0"] = val0
        results[model_name]["act0"] = act0

    #%% save results
    if save:
        pickle.dump(results, open(f"{basedir}/data/rnn_analyses/analyse_performance.p", "wb"))


# %%

if __name__ == "__main__":
    
    if len(sys.argv) >= 2:
        base_model_name = sys.argv[1]
        seeds = [int(seed) for seed in sys.argv[2:]]
    else:
        base_model_name = "MazeEnv_L4_max6/landscape_changing-rew_dynamic-rew_constant-maze/allo_planrew_plan5-6-7/VanillaRNN/iter10_tau5.0_opt/N800_linout/model31"
        seeds = [31,32,33,34,35]

    seed = seeds[0]
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Running RNN behavioural analysis: Base name: {base_model_name}.\nSeeds: {seeds}")
    sys.stdout.flush()
    
    analyse_rnn_behaviour(base_model_name, seeds)

    print("\nFinished")
    sys.stdout.flush()
    
    