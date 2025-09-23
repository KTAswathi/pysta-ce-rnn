"""Code for analysing performance and representations for RNNs of different sizes"""

#%%
import torch
import pysta
import pickle
import numpy as np
import sys
from pysta import basedir

#%%

def analyse_by_network_size(base_model_name, sizes, save = True):

    size_accs = []
    size_decoding = []
    base_rnn = pickle.load(open(f"{basedir}/models/{base_model_name}.p", "rb"))["rnn"]
    base_Nrec = base_rnn.Nrec

    for size in sizes:
        size_accs.append([])
        model_name = base_model_name.replace(f"N{base_Nrec}", f"N{size}")
        rnn = pysta.utils.load_model(model_name, store_all_activity = False)[0]
        
        # compute accuracy
        for _ in range(10):
            rnn.reset()

            rnn.forward(store = True)
            step_nums = np.array([s["step_num"] for s in rnn.store])
            batch_accs = np.array([s["corrects"] for s in rnn.store])

            acc = batch_accs[step_nums == 0, :].mean() # just consider initial accuracy
            size_accs[-1].append(acc)
            
        # also assess future decoding
        rnn.env.planning_steps = int(np.amax(rnn.env.planning_steps)) # match planning steps for simplicity
        trial_data = pysta.analysis_utils.collect_data(rnn, num_trials = 10000) # this just simulates enough batches to get num_trials data and puts it all into a dict
        cv_result = pysta.analysis_utils.predict_locations_from_neurons(trial_data, crossvalidate_loc = True, neural_times = [-1, 0], loc_times = [1,2,3,4,5])
        size_decoding.append(cv_result["nongen_scores"])
        
        print(size, np.mean(size_accs[-1]), size_decoding[-1])
        sys.stdout.flush()

    if save:
        data = {"sizes": sizes, "perfs": size_accs, "decoding": size_decoding}
        pickle.dump(data, open(f"{basedir}/data/comparisons/performance_and_decoding_by_size.pickle", "wb"))


if __name__ == "__main__":
    
    if len(sys.argv) >= 2:
        base_model_name = sys.argv[1]
        sizes = [int(size) for size in sys.argv[2:]]
    else:
        base_model_name = "MazeEnv_L4_max6/landscape_changing-rew_dynamic-rew_constant-maze/allo_planrew_plan5-6-7/VanillaRNN/iter10_tau5_opt/N100_linout/model10"
        sizes = [50,100,150,250,350,500,600,800]

    seed = int(base_model_name.split("model")[-1])
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Running analyses by network size: Base name: {base_model_name}.\nSizes: {sizes}")
    sys.stdout.flush()
    
    analyse_by_network_size(base_model_name, sizes)

    print("\nFinished")
    sys.stdout.flush()
    
    