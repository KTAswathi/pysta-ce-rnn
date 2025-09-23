"""Some code for instantiating various directories that models, data, and figure will be saved to"""

#%%
import pysta
import os
from pysta import basedir

# Models
os.makedirs(f"{basedir}/models", exist_ok = True)

# Slurm submission
os.makedirs(f"{basedir}/slurm", exist_ok = True)
os.makedirs(f"{basedir}/slurm/commands", exist_ok = True)
os.makedirs(f"{basedir}/slurm/error", exist_ok = True)
os.makedirs(f"{basedir}/slurm/out", exist_ok = True)
os.makedirs(f"{basedir}/slurm/submission", exist_ok = True)

# Data
os.makedirs(f"{basedir}/data", exist_ok = True)
os.makedirs(f"{basedir}/data/rnn_analyses", exist_ok = True)
os.makedirs(f"{basedir}/data/comparisons", exist_ok = True)
os.makedirs(f"{basedir}/data/examples", exist_ok = True)

# Figures
os.makedirs(f"{basedir}/figures", exist_ok = True)
os.makedirs(f"{basedir}/figures/rnn_decoding", exist_ok = True)
os.makedirs(f"{basedir}/figures/rnn_connectivity", exist_ok = True)
os.makedirs(f"{basedir}/figures/schematics", exist_ok = True)
os.makedirs(f"{basedir}/figures/rnn_generalisation", exist_ok = True)
os.makedirs(f"{basedir}/figures/rnn_behaviour", exist_ok = True)
os.makedirs(f"{basedir}/figures/simple_tasks", exist_ok = True)
os.makedirs(f"{basedir}/figures/size", exist_ok = True)
os.makedirs(f"{basedir}/figures/changing_maze_rnn", exist_ok = True)
os.makedirs(f"{basedir}/figures/attractor", exist_ok = True)
os.makedirs(f"{basedir}/figures/tests", exist_ok = True)

#%%
