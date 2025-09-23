import importlib
import pysta
import torch
import numpy as np
import os
import pickle
from scipy.stats import pearsonr

basedir = "/ceph/behrens/kris/research/sta_code/"


def reload():
    importlib.reload(pysta)
    importlib.reload(pysta.envs)
    importlib.reload(pysta.agents)
    importlib.reload(pysta.maze_utils)
    importlib.reload(pysta.utils)
    importlib.reload(pysta.train_rnn)
    importlib.reload(pysta.plot_utils)
    importlib.reload(pysta.analysis_utils)
    importlib.reload(pysta.subspace_utils)
    

def optimal_initial_action_freq(agent, reset_after = True, ignore_futile = True):
    """"
    Parameters:
    reset_after : bool
        This function performs a 'step' through the environment. If true, we reset the environment again at the end.
    ignore_futile : bool
        If true, only average over trials where there is an optimal action
    """
    agent.reset()
    emp_acts = agent.step(agent.env.observation()) # which actions does the agent take?
    opt_acts = agent.env.optimal_actions() # what are the optimal actions?
    acts_are_opt = opt_acts[torch.arange(len(emp_acts)), emp_acts] # which actions were optimal?
    
    if ignore_futile: # only average where there is an optimal policy
        not_futile = torch.where(opt_acts.mean(-1) < 1.0)[0]
        frac_opt = acts_are_opt[not_futile].mean() # what fraction of actions are optimal?
    else: # just avg over everything
        frac_opt = acts_are_opt.mean()

    if reset_after:
        agent.reset()
    return frac_opt
    
    
def correlate_offdiagonal(M1, M2):
    """this function correlates the non-diagonal elements of two matrices"""
    inds = np.concatenate([np.ravel_multi_index(ind, M1.shape) for ind in [np.triu_indices_from(M1, k=1), np.triu_indices_from(M1, k=1)]])
    return pearsonr(M1.flatten()[inds], M2.flatten()[inds]).statistic


def point_biserial_correlation(W, A):
    """
    compute the Point-biserial correlation coefficient
    (equivalent to the pearson correlation when one variable is binary)
    W is continuous, A is binar
    """
    
    Wstd = W.std()
    if Wstd == 0:
        return np.nan
    
    Aposs, Aimp = np.sign(A), (1-np.sign(A)) # possible and impossible transitions
    nposs, nimp, ntot = Aposs.sum(), Aimp.sum(), A.size # number of transitions
    mposs, mimp = (Aposs*W).sum() / nposs, (Aimp*W).sum() / nimp # mean W for possible/impossible transitions
    corr = (mposs - mimp)/Wstd * np.sqrt(nposs*nimp/ntot**2) # point-biserial correlation
    
    return  corr


schematic_walls = np.zeros((9, 4))
schematic_barriers = [(0,0),(3,1),(2,0),(5,1),(6,2),(7,3),(4,2),(5,3)]
for barrier in schematic_barriers: schematic_walls[barrier[0], barrier[1]] = 1.0

def get_rnn_name(kwargs):
    
    # instantiate env and agent to generate name
    env = pysta.envs.MazeEnv(**kwargs)
    rnn = pysta.agents.VanillaRNN(env, **kwargs)

    # put these two parts together
    full_name = f"{env.name}/{rnn.name}"
    
    return full_name

def str_to_val(s):
    """convert a command line argument string to a value"""
    try:
        return int(s) # if it's a simple int, return that (including bools)
    except ValueError: # otherwise
        try:
            return float(s) # if it can be interpreted as a float, return that
        except ValueError:
            return str(s) # otherwise just return the string

def command_to_kwargs(command):
    new_kwargs = {}
    command_kwargs = command.split(".py --")[-1].split("--")
    for kwarg in command_kwargs:
        kwarg = kwarg.split()
        key = kwarg[0]
        if len(kwarg) == 2:
            val = str_to_val(kwarg[1])
        else:
            val = [str_to_val(x) for x in kwarg[1:]]
        new_kwargs[key] = val  # Handle flags without values
    return new_kwargs

def command_to_model_name(command):
    new_kwargs = command_to_kwargs(command)
    all_kwargs = pysta.argparser.parse_args(**new_kwargs)
    rnn_name = pysta.utils.get_rnn_name(all_kwargs)
    model_name = f"{rnn_name}/model{new_kwargs['seed']}"
    return model_name


def load_model(model_name, store_all_activity = True, greedy = True, force_optimal = False, create_sta = False, create_dp = False):
    print(model_name)

    assert create_sta + create_dp <= 1 # mutually exclusive

    training_result = pickle.load(open(f"{basedir}/models/{model_name}.p", "rb"))
    rnn = training_result["rnn"]
    
    if create_sta:
        addstr = "sta_"
        rnn = pysta.agents.SpaceTimeAttractor(rnn.env)
        rnn.env.working_memory = False # STA not implemented with working memory
    elif create_dp:
        addstr = "dp_"
        rnn = pysta.agents.DPAgent(rnn.env)
    else:
        addstr = ""
            
    figdir = f"{basedir}/figures/by_model/{addstr}{model_name}/"
    os.makedirs(figdir, exist_ok = True) # make the directory if it doesn't exist
    datadir = f"{basedir}/data/rnn_analyses/{addstr}" + "_".join(model_name.split("/")) + "_"
    
    rnn.store_all_activity = store_all_activity
    rnn.greedy = greedy
    rnn.force_optimal = force_optimal # consider whether to do this or not
    
    return rnn, figdir, datadir

def cond_rel(tas, tbs):
    """for two sets of time points, tas = (ta_neural, ta_loc), tbs = (tb_neural, tb_loc),
    return whether tas and tbs are equivalent in a relative coding scheme"""
    return (tas[1] - tas[0]) == (tbs[1] - tbs[0])

def cond_abs(tas, tbs):
    """for two sets of time points, tas = (ta_neural, ta_loc), tbs = (tb_neural, tb_loc),
    return whether tas and tbs are the same in an absolute coding scheme"""
    return tas[1] == tbs[1]

def compute_model_support(results):
    neural_times, loc_times = results["neural_times"], results["loc_times"]
    all_res = []
    for icond, cond in enumerate([cond_rel, cond_abs]): # for each model to consider
        # go through all combinations of train neural and loc times + test neural and loc times
        # store performnace separately for pairs that should code similarly, and pairs that should not
        res = [[], []]    
        for ia1, ta1 in enumerate(neural_times):
            for ia2, ta2 in enumerate(loc_times):
                for ib1, tb1 in enumerate(neural_times):
                    for ib2, tb2 in enumerate(loc_times): 
                        if (ta1 != tb1) and (ta2 > ta1) and (min(ta1, tb1) >= 0): # don't compare same neural time points, and don't compare 'current' decoder
                            perf = results["scores"][ia1, ia2, ib1, ib2]
                            res[int(cond((ta1, ta2), (tb1, tb2)))].append(perf)
        all_res.append([np.mean(res[i]) for i in range(2)])
    return all_res
    
    
    