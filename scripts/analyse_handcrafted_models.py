"""Code for comparing the handcrafted STA, SR, and TD models"""

#%%
import pysta
import torch
import numpy as np
import pickle
import sys
from pysta import basedir


def compare_handcrafted_models(seed = 0, save = True):

    np.random.seed(seed)
    torch.manual_seed(seed)

    #%% run a quantitative comparisons of every agent in every environment

    kwargs = {"batch_size": 201, "output_format": "egocentric", "planning_steps": 0,}
    envs = [
        pysta.envs.MazeEnv(rew_landscape = False, dynamic_rew = False, changing_trial_rew = False, **kwargs), # static rew that is constant across trials
        pysta.envs.MazeEnv(rew_landscape = False, dynamic_rew = False, changing_trial_rew = True, **kwargs), # static within trial, changes across trials
        pysta.envs.MazeEnv(rew_landscape = False, dynamic_rew = True, changing_trial_rew = True, **kwargs), # moving within trial, changes across trial
        pysta.envs.MazeEnv(rew_landscape = True, dynamic_rew = True, changing_trial_rew = True, **kwargs) # general space-time reward function
    ]

    num_agents = 3
    nreps = 20
    results = np.zeros((nreps, len(envs), num_agents+1))

    for ienv, env in enumerate(envs):
        
        print("\n", env.name)
        
        for irep in range(nreps): # run a bunch of repeats
            
            env.reset(hard = True) # sample a completely new environment of the same type
            
            agents = [
                pysta.agents.TDLearner(env, greedy = True, force_optimal = False),
                pysta.agents.SRLearner(env, greedy = True, force_optimal = False),
                pysta.agents.SpaceTimeAttractor(env, greedy = True, force_optimal = False)
            ]
            
            for iagent, agent in enumerate(agents):
                
                if agent.classname == "TDLearner":
                    agent.greedy = False
                    [agent.forward() for _ in range(200)] # learn value function for some episodes
                    agent.greedy = True
                
                frac_opt = pysta.utils.optimal_initial_action_freq(agent)
                results[irep, ienv, iagent] = frac_opt.item()
                
            # random baseline is just the fraction of actions that are optimal
            opt_acts = env.optimal_actions()
            results[irep, ienv, -1] = opt_acts[opt_acts.mean(-1) < 1.0, :].mean()

            print(irep, results[irep, ienv, :])
            sys.stdout.flush()
            
    #%% save the data
    if save:
        data = {"envs": [env.name for env in envs],
                "agents": [agent.classname for agent in agents],
                "perfs": results}
        
        pickle.dump(data, open(f"{basedir}/data/comparisons/handcrafted_agent_performances.pickle", "wb"))


# %%

if __name__ == "__main__":

    compare_handcrafted_models()
    
    print("\nFinished")
    