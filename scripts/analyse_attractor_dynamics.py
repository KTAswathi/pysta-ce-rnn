
"""Code for analysing the robustness and dynamics of RNN and STA representations in response to perturbations"""

#%%
import numpy as np
import torch
import matplotlib.pyplot as plt
import pysta
import sys
import pickle
from pysta import basedir
import torch
np.random.seed(0)
torch.manual_seed(0)


#%%

def run_attractor_analyses(model_name, sta = False, ctrl = False,
                           paths = [[0,4,8,9,10,10,10], [0,1,2,6,10,10,10]],
                           stim_t = 2):


    #%%
    
    stim_loc = paths[0][stim_t]
    rnn, figdir, datadir = pysta.utils.load_model(model_name, create_sta = sta) # load the model
    rnn.env.planning_steps = int(np.amax(rnn.env.planning_steps)) # match planning steps for simplicity
    
    rnn.reset()
    rnn.env.rews[...] = -1.0
    for ipath, path in enumerate(paths):
        for i in range(len(path)):
            rnn.env.rews[:, i, path[i]] = 0.7 + 0.3*ipath
    rnn.env.compute_value_function()
    rnn.env.loc[...] = paths[0][0]

    if sta:
        stim_dir = torch.zeros(rnn.Nrec, 1)
        stim_dir[stim_t*rnn.env.num_locs+stim_loc, 0] = 1
        strengths = np.linspace(0, 500, 19)
        num_initial_steps = 1
        num_p1, num_p2, num_p3 = 2,2,2
        
        def decode(r):
            return r
        
    else:

        connectivity_data  = pickle.load(open(f"{datadir}planning_subspaces.pickle", "rb"))
        Csubs, Csubs_raw, biases = [connectivity_data[k] for k in ["Csubs", "Csubs_raw", "biases"]]
        
        def decode(r):
            if len(r.shape) == 3:
                new_r = (Csubs_raw[None, ...] @ r[:, None, None, ...] + biases[None, None, ...])[:, 0, ...]
            else:
                new_r = Csubs_raw @ r + biases
            new_r = np.exp(new_r)[..., 0]
            new_r = new_r / new_r.sum(-1, keepdims = True)
            return new_r
        
        strengths = np.concatenate([np.linspace(0, 2.0, 21), np.ones(1)*3.0])
        strengths = np.linspace(0, 10.0, 31)
        stim_dir = torch.tensor(Csubs[stim_t, stim_loc])[:, None]

        num_initial_steps = np.amax(rnn.env.planning_steps)
        num_p1, num_p2, num_p3 = 10, 10, 10

    if ctrl:
        stim_dir = torch.randn(stim_dir.shape)
    stim_dir = stim_dir / torch.sqrt(torch.sum(stim_dir**2))

    x = rnn.env.observation().to(rnn.z0.device) # observation at this point in time
    for _ in range(num_initial_steps):
        rnn.step(x) # update RNN state, compute policy, and sample an action
    old_r = rnn.r.detach().clone()
    old_z = rnn.z.detach().clone()
    old_bias = rnn.brec.detach().clone()

    new_rs = []
    relax_rs = []
    deltas = []
    deltas_raw = []
    relax_deltas = []
    all_all_acts = []
    for strength in strengths:
        rnn.r = old_r
        rnn.z = old_z
        rnn.all_acts = [[], [], []]

        rnn.brec = torch.nn.Parameter(old_bias) # first just show it's an attractor state
        for _ in range(num_p1):
            rnn.step(x) # update RNN state, compute policy, and sample an action 
        
        # then perturb
        rnn.brec = torch.nn.Parameter(old_bias + strength*stim_dir)
        for _ in range(num_p2):
            rnn.step(x) # update RNN state, compute policy, and sample an action 
        new_r = rnn.r.detach().clone()

        # now relax
        rnn.brec = torch.nn.Parameter(old_bias)
        for _ in range(num_p3):
            rnn.step(x)
        relax_r = rnn.r.detach().clone()

        delta = np.abs(decode(new_r.numpy()) - decode(old_r.numpy())).sum((-1,-2))
        relax_delta = np.abs(decode(relax_r.numpy()) - decode(old_r.numpy())).sum((-1,-2))
        deltas.append(delta)
        relax_deltas.append(relax_delta)
        new_rs.append(new_r)
        relax_rs.append(relax_r)
        deltas_raw.append(((new_r.numpy() - old_r.numpy())**2).sum((-1,-2)))

        print(strength, delta.mean(), relax_delta.mean())
        sys.stdout.flush()
        all_all_acts.append(np.array(rnn.all_acts[0])[:, :5, ...])

    all_all_acts = np.array(all_all_acts)

    deltas, relax_deltas, deltas_raw = [np.array(arr) for arr in [deltas, relax_deltas, deltas_raw]]
    proj_old_r = decode(old_r.numpy())
    proj_new_rs, proj_relax_rs = [[decode(r.numpy()) for r in rs] for rs in [new_rs, relax_rs]]
    proj_all_acts = np.array([[decode(all_all_acts[i, j, ...]) for j in range(all_all_acts.shape[1])] for i in range(all_all_acts.shape[0])])
    walls = rnn.env.walls[0].numpy()
    num_ps = [num_p1, num_p2, num_p3]
    rnn.all_acts = [[], [], []]

    #%%
    
    result = {"all_all_acts": all_all_acts, "old_r": old_r, "new_rs": new_rs, "relax_rs": relax_rs, "deltas": deltas,
            "relax_deltas": relax_deltas, "walls": walls, "proj_old_r": proj_old_r, "proj_new_rs": proj_new_rs,
            "proj_relax_rs": proj_relax_rs, "proj_all_acts": proj_all_acts, "strengths": strengths, "num_ps": num_ps, "deltas_raw": deltas_raw,
            "paths": paths, "rews": rnn.env.rews.numpy(), "rnn": rnn}

    ctrlstr = "_ctrl" if ctrl else ""
    pickle.dump(result, open(f"{datadir}fixed_point_analyses{ctrlstr}.p", "wb"))
    
    return


# %%

if __name__ == "__main__":
    
    model_name = "MazeEnv_L4_max6/landscape_changing-rew_dynamic-rew_constant-maze/allo_planrew_plan5-6-7/VanillaRNN/iter10_tau5.0_opt/N800_linout/model35"
    paths = [[0,4,8,9,10,10,10], [0,1,2,6,10,10,10]]
    seed = int(model_name.split("model")[-1])
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Running RNN attractor analysis for {model_name}.")
    sys.stdout.flush()
    torch.set_grad_enabled(False)
    
    # run actual analysis
    run_attractor_analyses(model_name, sta = False, ctrl = False, paths = paths)
    # run for a control stimulus direction
    run_attractor_analyses(model_name, sta = False, ctrl = True, paths = paths)
    # run on the handcrafted STA
    run_attractor_analyses(model_name, sta = True, ctrl = False, paths = paths)

    print("\nFinished")
    sys.stdout.flush()
    