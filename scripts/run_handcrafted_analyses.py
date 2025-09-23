"""Code that submit slurm jobs for the analyses of handcrafted models"""

import pysta
import os
os.chdir(f"{pysta.basedir}/scripts")
from submit_slurm import submit_slurm

#%% generate some examples
command_compare = f"python {pysta.basedir}/scripts/generate_handcrafted_examples.py "
print(submit_slurm(command_compare, f"analyse_handcrafted_examples", time = "2:00:00"))

#%% run quantitative comparison
command_compare = f"python {pysta.basedir}/scripts/analyse_handcrafted_models.py "
print(submit_slurm(command_compare, f"analyse_handcrafted_perf", time = "2:00:00"))


