
"""
This file provides a minimal example of using the 'pysta' package for simulating and training spacetime attractors.
"""

#%% load some packages

import pysta
from pysta import basedir
import numpy as np
import torch
import os
np.random.seed(1)
torch.manual_seed(1)
os.makedirs(f"{basedir}/figures/examples", exist_ok = True)

#%% run an example handcrafted spacetime attractor model

# initialize an environment with a stationary goal (change 'dynamic_rew' to 'True' for a moving goal)
print("\nInstantiating environment")
env = pysta.envs.MazeEnv(rew_landscape = False, batch_size = 1, dynamic_rew = False, max_steps = 8, side_length = 5)

# plot the environment
print("\nPlotting environment")
env.plot(filename = f"{basedir}/figures/examples/environment.pdf")

#%% initialize a model
print("\nInstantiating Spacetime Attractor model")
sta = pysta.agents.SpaceTimeAttractor(env, iters_per_action = 200)

# compute a representation and policy
print("\nRunning dynamics to compute STA representation and policy")
action = sta.step(env.observation())

#%% plot the representation
print("\nPlotting STA representation")
plot_kwargs = {"walls": env.walls[0], "vmap": sta.r[0], "loc": env.loc[0], "goal": env.goal[0], "aspect": (1,1,6)}
pysta.plot_utils.plot_perspective_attractor(filename = f"{basedir}/figures/examples/sta_representation.pdf", **plot_kwargs, show = True, vmin = -0.1, vmax = 1.2, figsize = (8,5))

#%% run a full sta trial and plot a gif
print("\nRunning a full STA trial")
sta.iters_per_action = 100 # reduce number of iterations per action to speed things up a bit. This means the STA may not fully converge between actions
step_num = 0
sta.store_all_activity = True
np.random.seed(0)
torch.manual_seed(0)
while step_num < 6:
        sta.forward()
        step_num = env.step_num
        print(step_num)

#%% plot a gif
print("\nPlotting a lil' gif")
activity, locs = [np.array(sta.all_acts[i])[::8, 0, ...] for i in [0,1]] # extract activity and position at every network iteration
pysta.plot_utils.plot_perspective_gif(env.walls[0], activity, locs = locs, goal = env.goal[0], smooth = 2.5, tempdir = f"{basedir}/figures/examples/temp/", filename = f"{basedir}/figures/examples/sta.gif", delay = 20, vmin = -0.1, vmax = 1.2, print_freq = 25)

#%% quantify performance on static goal and rew landscape tasks (use a smaller environment with larger max_steps to avoid impossible trials)
print("\nComparing performance of STA and SR on static goal and reward landscape tasks")
envs = [pysta.envs.MazeEnv(rew_landscape = False, batch_size = 100, dynamic_rew = False, max_steps = 10, side_length = 5),
        pysta.envs.MazeEnv(rew_landscape = True, batch_size = 100, dynamic_rew = True, max_steps = 10, side_length = 5)]
for ienv, env in enumerate(envs):
    sta = pysta.agents.SpaceTimeAttractor(env, greedy = True)
    sr = pysta.agents.SRLearner(env, greedy = True)
    for agent in [sta, sr]:
        agent.step(env.observation())
        frac_opt = pysta.utils.optimal_initial_action_freq(agent)
        print(f"{['static goal', 'reward landscape'][ienv]}, {agent.classname}: {frac_opt.mean()}")


#%% run some example RNN training

np.random.seed(0)
torch.manual_seed(0)

# train an RNN for a few steps on the reward landscape task
print("\nTraining an RNN on the reward landscape task")
kwargs = pysta.argparser.parse_args(working_memory = False, eval_freq = 50, seed = 0, planning_steps = [3,4], max_steps = 5, num_epochs = 1000, lrate = 3e-3, side_length = 4, overwrite = True) # training parameters
rnn = pysta.train_rnn.main_train(kwargs)

# collect some data post training (this is too little data, which speeds things up)
print("\nCollecting empirical data from trained RNN")
trial_data = pysta.analysis_utils.collect_data(rnn, num_trials = 6000)

# deocding analysis. for simplicity, don't perform crossvalidate across positions. This is also too strongly regularized, which speeds things up.
print("\nPerforming a simple decoding analysis")
neural_times, loc_times = [0,1,2,3], [0,1,2,3,4] # what times to take neural activity from and predict location at
pred_result = pysta.analysis_utils.predict_locations_from_neurons(trial_data, crossvalidate_loc = False, neural_times = neural_times, loc_times = loc_times, logistic_C = 1e-1)

# plot the result of this decoding
print("\nPlotting decoding result")

# first plot whether the information exists
pysta.plot_utils.plot_prediction_result(pred_result["nongen_scores"], neural_times, loc_times, filename = f"{basedir}/figures/examples/rnn_decoding.pdf", show = True)

# then plot generalisation of the (1,3) decoder
pysta.plot_utils.plot_prediction_result(pred_result["scores"][1,3], neural_times, loc_times, filename = f"{basedir}/figures/examples/rnn_decoding_gen.pdf", show = True)

# %%
