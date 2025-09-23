"""For for submitting slurm jobs that train all RNNs used in this work"""

#%%
import pickle
import pysta
import os
os.chdir(f"{pysta.basedir}/scripts")
from submit_slurm import submit_slurm

seeds = [31,32,33,34,35]
base_command = f"python {pysta.basedir}/pysta/train_rnn.py --overwrite 1"
commands = {}

#%% train WM RNNs
commands["WM"] = []
for seed in seeds:
    command = f"{base_command} --seed {seed}"
    print(submit_slurm(command, f"train_RNN_WM_{seed}"))
    commands["WM"].append(command)
    
#%% train WM changing maze
commands["changing_maze"] = []
for seed in seeds:
    command = f"{base_command} --changing_trial_maze 1 --num_epochs 250000 --seed {seed}"
    print(submit_slurm(command, f"train_RNN_changing_maze_{seed}", time = "60:00:00"))
    commands["changing_maze"].append(command)
    
#%% train RNNs with continual reward input
commands["relrew"] = []
for seed in seeds:
    command = f"{base_command} --iters_per_action 9 10 11 --working_memory 0 --seed {seed}"
    print(submit_slurm(command, f"train_RNN_relrew_{seed}"))
    commands["relrew"].append(command)

#%% train egocentric
commands["egocentric"] = []
for seed in seeds:
    command = f"{base_command} --output_format egocentric --seed {seed}"
    print(submit_slurm(command, f"train_RNN_egocentric_{seed}"))
    commands["egocentric"].append(command)

#%% then train on simpler tasks
commands["moving_goal"] = []
for seed in seeds:
    command = f"{base_command} --rew_landscape 0 --seed {seed}"
    print(submit_slurm(command, f"train_RNN_moving_goal_{seed}"))
    commands["moving_goal"].append(command)
    
commands["static_goal"] = []
for seed in seeds:
    command = f"{base_command} --rew_landscape 0 --dynamic_rew 0 --seed {seed}"
    print(submit_slurm(command, f"train_RNN_static_goal_{seed}"))
    commands["static_goal"].append(command)
    
#%% then train on simpler tasks with relative reward
commands["moving_goal_rel"] = []
for seed in seeds:
    command = f"{base_command} --rew_landscape 0 --iters_per_action 9 10 11 --working_memory 0 --seed {seed}"
    print(submit_slurm(command, f"train_RNN_moving_goal_rel_{seed}"))
    commands["moving_goal_rel"].append(command)
    
commands["static_goal_rel"] = []
for seed in seeds:
    command = f"{base_command} --rew_landscape 0 --dynamic_rew 0 --iters_per_action 9 10 11 --working_memory 0 --seed {seed}"
    print(submit_slurm(command, f"train_RNN_static_goal_rel_{seed}"))
    commands["static_goal_rel"].append(command)

#%% train networks of different sizes 50-800
sizes = [50,100,150,250,350,500,600,800]
commands["sizes"] = []
seed = 10
for size in sizes:
    command = f"{base_command} --Nrec {size} --seed {seed}"
    print(submit_slurm(command, f"train_RNN_size{size}_{seed}", time = "48:00:00"))
    commands["sizes"].append(command)    

#%% save commands
pickle.dump(commands, open(f"{pysta.basedir}/slurm/commands/train_all_rnns.p", "wb"))


