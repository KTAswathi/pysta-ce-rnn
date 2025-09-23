 """Here we generate some example value functions and representations for different handcrafted agents"""
 
 #%% load libraries
 
import pysta
import torch
import numpy as np
import sys
import pickle
from pysta import basedir

save = True

#%% examples of each task

example_task_data = {}

# moving goal (static goal will just be first location)
seed = 18
torch.manual_seed(seed)
np.random.seed(seed)
max_steps = 5
env = pysta.envs.MazeEnv(rew_landscape = False, output_format = "egocentric", batch_size = 1, dynamic_rew = True, planning_steps = 0, max_steps = max_steps, rew_goal = 1.0, rew_nogoal = 0.0) # static within trial, changes across trials
env.loc[0] = 5

loc0 = env.loc.clone()
goal0 = env.goal.clone()
walls0 = env.walls.clone()
rews0 = env.rews.clone()
env.plot(filename = None, values = False)
example_task_data["moving_goal"] = {
    "walls": env.walls[0],
    "goal": env.goal[0].clone(),
    "loc": env.loc[0].clone(),
    "optimal_actions": env.optimal_actions()[0, :],
    "rew": env.rews[0].clone(),
}

env.step(torch.multinomial(env.optimal_actions(), 1)[:, 0])

env.plot(filename = None, values = False)
example_task_data["moving_goal"]["optimal_actions_t2"] = env.optimal_actions()[0, :] # everything else is the same
example_task_data["moving_goal"]["loc_t2"] = env.loc[0] # everything else is the same

# static maze
env.loc = loc0
env.rews[:, 1:, :] = env.rews[:, :1, :]
env.goal[:, 1:] = env.goal[:, :1]
env.compute_value_function()
example_task_data["static_goal"] = {
    "walls": env.walls[0],
    "goal": env.goal[0, 0].item(),
    "loc": env.loc[0],
    "optimal_actions": env.optimal_actions()[0, :],
    "rew": env.rews[0],
}

# reward landscape
env.rew_landscape = True
[env.reset() for _ in range(1)]
rew_landscape = env.rews.clone()
env.loc = loc0
env.plot(filename = None, values = False)
env.plot(filename = None, values = True)

example_task_data["rew_landscape"] = {
    "walls": env.walls[0],
    "loc": env.loc[0],
    "optimal_actions": env.optimal_actions()[0, :],
    "rews": env.rews[0]
}
all_locs = [env.loc[0].item()]
for _ in range(env.max_steps):
    env.step(torch.multinomial(env.optimal_actions(), 1)[:, 0])
    all_locs.append(env.loc[0].item())
example_task_data["rew_landscape"]["all_locs"] = all_locs

if save:
    pickle.dump(example_task_data, open(f"{basedir}/data/examples/example_task_data.pickle", "wb"))



#%% now look at different agent representations

env_kwargs = {"output_format": "egocentric", "planning_steps": 0, "batch_size": 101, "max_steps": max_steps, "rew_goal": 1.0, "rew_nogoal": 0.0}
envs = [
    pysta.envs.MazeEnv(rew_landscape = False, dynamic_rew = False, changing_trial_rew = False, **env_kwargs), # static rew that is constant across trials
    pysta.envs.MazeEnv(rew_landscape = False, dynamic_rew = False, changing_trial_rew = True, **env_kwargs), # static within trial, changes across trials
    pysta.envs.MazeEnv(rew_landscape = False, dynamic_rew = True, changing_trial_rew = True, **env_kwargs), # moving within trial, changes across trial
    pysta.envs.MazeEnv(rew_landscape = True, dynamic_rew = True, changing_trial_rew = True, **env_kwargs) # general space-time reward function
]

env_labels = ["static", "changing", "moving", "landscape"]
example_rep_data = {}

for ienv, env in enumerate(envs):
    example_rep_data[env_labels[ienv]] = {}
    agents = [pysta.agents.TDLearner(env, greedy = False),
            pysta.agents.SRLearner(env, greedy = False),
            pysta.agents.SpaceTimeAttractor(env, greedy = False)]
            
    for agent in agents:

        env.reset()
        env.sample_maze(walls = walls0+torch.zeros(env.batch, 1, 1))
        agent.reset() # recompute transition statistics
        env.loc[...] = loc0
        if ienv == 0: # set a static goal to what it was before
            env.goal[...] = goal0[:, :1]
            env.rews[...] = rews0[:, :1, :]
            env.compute_value_function()
            test_goal = goal0[0,0].item()
            rmin, rmax = env.rews.min(), env.rews.max()
        elif ienv == 1:
            test_goal = 1
            env.rews[...] = rmin
            env.rews[..., test_goal] = rmax
            env.compute_value_function()
        elif ienv == 2:
            env.goal[...] = goal0[...].clone()
            env.rews[...] = rews0[...].clone()
            test_goal = goal0[0].clone()
        elif ienv == 3:
            env.rews[...] = rew_landscape
            env.compute_value_function()
            test_goal = None
        
        if agent.label == "td": # td learner should train for a bit
            [agent.forward() for _ in range(200)] # learn value function for some episodes
        else:
            agent.step(env.observation()) # compute response to observation
        
        if agent.label == "sta":
            kwargs = {"walls": walls0[0],
                      "vmap": agent.r[0],
                      "loc": loc0,
                      "goal": test_goal}
            pysta.plot_utils.plot_perspective_attractor(filename = None, **kwargs, cmap = "YlOrRd", show = True)
            
            
        else:
            kwargs = {"walls": walls0[0],
                    "vmap": (agent.values[0] if agent.label == "sr" else agent.values).clone(),
                    "loc": loc0,
                    "goal": test_goal}

            pysta.plot_utils.plot_flat_frame(filename = None, **kwargs, cmap = "YlOrRd", vmin = None, vmax = None, show = True)
            
        example_rep_data[env_labels[ienv]][agent.label] = kwargs

if save:
    pickle.dump(example_rep_data, open(f"{basedir}/data/examples/example_rep_data.pickle", "wb"))


print("\nFinished")
sys.stdout.flush()
    
# %%
