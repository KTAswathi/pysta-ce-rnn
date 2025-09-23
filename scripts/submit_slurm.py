"""Here we define a function that writes and submits a simple slurm job. This is useful for automating training and analyses."""
#%%
import subprocess
import pysta

def slurm_submission_script(command, jobname, time = "48:00:00", mem = "32G", nodes = 1, cpus_per_task = 8, partition = "cpu", extra_commands = ""):

    script = f"""#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH --output={pysta.utils.basedir}/slurm/out/{jobname}.out
#SBATCH --error={pysta.utils.basedir}/slurm/error/{jobname}.err
#SBATCH --nodes={nodes}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --partition={partition}
#SBATCH --ntasks-per-node=1
{extra_commands}

source ~/.bashrc
conda activate pysta

conda info

{command}
"""
    
    return script
    

def submit_slurm(command, jobname, **kwargs):
    
    slurm_fname = f"{pysta.basedir}/slurm/submission/{jobname}.sh"
    
    with open(slurm_fname, "w") as f:
        f.write(slurm_submission_script(command, jobname, **kwargs))
    
    return subprocess.run(["sbatch", slurm_fname], check=True)



# %%
