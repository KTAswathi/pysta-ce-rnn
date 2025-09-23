"""
consider RNNs trained on different tasks, and quantify their performance + regularization costs across tasks.
"""

#%%

import numpy
import torch
import pysta
import pickle
import sys
import numpy as np
from pysta import basedir

#%%
def analyse_rnn_generalisation(base_model_name, seeds, save = True, reps = 30):

    base_model_name = base_model_name.split("model")[0]+"model"
    all_model_names = [[f"{base_model_name}{seed}",  base_model_name.replace("landscape", "goal")+str(seed),
                        base_model_name.replace("landscape", "goal").replace("dynamic-rew", "static-rew")+str(seed),
                        base_model_name.replace("constant-maze", "changing-maze")+str(seed)]
                       for seed in seeds]

    num_envs = len(all_model_names[0])
    results = np.zeros((len(seeds), reps, num_envs, num_envs+1, 3)) # seeds, batches, envs, models, (acc, rate, weight)
        
    for iseed, model_names in enumerate(all_model_names):

        rnns = [pysta.utils.load_model(model_name, store_all_activity = False, greedy = True)[0] for model_name in model_names]
        
        for rnn in rnns:
            rnn.store_all_activity = False

        envs = [rnn.env for rnn in rnns] # test on all three environments

        for rep in range(reps):
            if rep % 5 == 0: print(rep)
            for ienv, env in enumerate(envs):
                for irnn, rnn in enumerate(rnns):
                    
                    rnn.env = env # test in this env
                    rnn.reset()
                    
                    # random baseline is just the fraction of initial actions that are optimal
                    opt_acts = env.optimal_actions()
                    if env.output_format == "allocentric":
                        opt_acts = np.sign(rnn.allo_to_ego_pi(opt_acts))
                    
                    results[iseed, rep, ienv, len(all_model_names[0]), 0] += opt_acts[opt_acts.mean(-1) < 1.0, :].mean() / len(rnns)
                    
                    # now evaluate actual accruacy
                    rnn.forward(store = True)

                    step_nums = np.array([s["step_num"] for s in rnn.store])
                    accs = np.array([s["corrects"] for s in rnn.store])

                    acc = accs[step_nums == 0, :].mean() # just consider initial accuracy

                    # also parameter and rate loss
                    rate = rnn.rate_loss.item() / rnn.r_reg / rnn.env.batch
                    weight = rnn.weight_loss.item() / rnn.W_reg / rnn.env.batch

                    if rep == 0: # occasionally print progress
                        if irnn == 0: print()
                        print(rnn.acc_loss.item(), rnn.rate_loss.item(), rnn.weight_loss.item())
                        print(acc, rate, weight)
                    
                    results[iseed, rep, ienv, irnn, :] = np.array([acc, rate, weight])

        for i in range(3):
            print(results[iseed, ..., i].mean(0))
            
    #%% save the data
    if save:
        data = {"models": all_model_names, "perfs": results}
        pickle.dump(data, open(f"{basedir}/data/comparisons/rnn_generalisation.pickle", "wb"))


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

    print(f"Running RNN generalisation analysis: Base name: {base_model_name}.\nSeeds: {seeds}")
    sys.stdout.flush()
    
    analyse_rnn_generalisation(base_model_name, seeds)

    print("\nFinished")
    sys.stdout.flush()
    
    
