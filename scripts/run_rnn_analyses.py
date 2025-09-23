"""Code that submits slurm jobs for running analyses of pretrained RNNs"""

#%%
import pickle
import pysta
import os
os.chdir(f"{pysta.basedir}/scripts")
from submit_slurm import submit_slurm

train_commands = pickle.load(open(f"{pysta.basedir}/slurm/commands/train_all_rnns.p", "rb"))

#%% Run standard analyses of the RNNs in fixed mazes

base_analysis_command = f"python {pysta.basedir}/scripts/analyse_rnn.py "
for command_type in ["WM", "relrew", "egocentric"]:
    for command in train_commands[command_type]:
        model_name = pysta.utils.command_to_model_name(command)
        seed = model_name.split("model")[-1]
        
        command_analysis = f"{base_analysis_command} {model_name} collect time decoding subspaces"
        print(submit_slurm(command_analysis, f"analyse_RNN_{command_type}_{seed}", time = "5:00:00", partition = "gpu", extra_commands = "#SBATCH --gres=gpu:1"))

        # for the WM model, also run (i) the same analyses on the handcrafted sta and value-based model, and (ii) baseline analyses for transition decoding
        if command_type == "WM":
            print(submit_slurm(command_analysis+" sta", f"analyse_STA_{command_type}_{seed}", time = "5:00:00", partition = "gpu", extra_commands = "#SBATCH --gres=gpu:1"))
            
            print(submit_slurm(command_analysis+" dp", f"analyse_DP_{command_type}_{seed}", time = "5:00:00", partition = "gpu", extra_commands = "#SBATCH --gres=gpu:1"))

            command_changing_maze = f"python {pysta.basedir}/scripts/analyse_changing_maze_rnn.py {model_name} connectivity transition"
            print(submit_slurm(command_changing_maze, f"analyse_{command_type}_ctrl_changing_maze_{seed}", time = "32:00:00", partition = "gpu", extra_commands = "#SBATCH --gres=gpu:1", mem = "64G"))

#%% Analyse the changing maze RNNs

for command in train_commands["changing_maze"]:
    model_name = pysta.utils.command_to_model_name(command)
    seed = model_name.split("model")[-1]

    command_changing_maze = f"python {pysta.basedir}/scripts/analyse_changing_maze_rnn.py {model_name} connectivity transition decoding"
    print(submit_slurm(command_changing_maze, f"analyse_changing_maze_{seed}", time = "32:00:00", partition = "gpu", extra_commands = "#SBATCH --gres=gpu:1", mem = "64G"))

#%% Analyse the simple tasks

for task in ["static_goal", "moving_goal"]:
    for relstr in ["", "_rel"]:

        for command in train_commands[task+relstr]:
            model_name = pysta.utils.command_to_model_name(command)
            seed = model_name.split("model")[-1]
            
            for model_type in ["base_rnn", "sta"]:
                command_analysis = f"python {pysta.basedir}/scripts/analyse_simple_tasks.py {model_name} {model_type} collect decoding planning"
                print(submit_slurm(command_analysis, f"analyse_{task}_task_{seed}_{model_type}{relstr}", time = "4:00:00"))
                        
#%% analyse the networks with different sizes

size_commands = train_commands["sizes"]
base_model_name = pysta.utils.command_to_model_name(size_commands[0])
sizes = [command.split("--Nrec ")[1].split()[0] for command in size_commands]
command_size = f"python {pysta.basedir}/scripts/analyse_by_size.py {base_model_name} "+" ".join(sizes)

print(submit_slurm(command_size, f"analyse_by_size", time = "16:00:00"))

#%% analyse network generalisation across tasks

base_model_name = pysta.utils.command_to_model_name(train_commands["WM"][0])
seeds = [command.split("--seed ")[1].split()[0] for command in train_commands["WM"]]
command_gen = f"python {pysta.basedir}/scripts/analyse_rnn_generalisation.py {base_model_name} "+" ".join(seeds)

print(submit_slurm(command_gen, f"analyse_rnn_generalisation", time = "2:00:00"))

#%% run analyses of RNN behaviour

base_model_name = pysta.utils.command_to_model_name(train_commands["WM"][0])
seeds = [command.split("--seed ")[1].split()[0] for command in train_commands["WM"]]
command_beh = f"python {pysta.basedir}/scripts/analyse_rnn_behaviour.py {base_model_name} "+" ".join(seeds)

print(submit_slurm(command_beh, f"analyse_rnn_behaviour", time = "2:00:00"))



