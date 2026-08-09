# pysta-cernn

(Ongoing research) code for training and analysing cortically embedded recurrent neural networks for planning representations.

This repository is adapted from Kris Jensen’s spacetime attractor/RNN codebase and is currently being extended to test whether planning representations in RNNs can develop spatial gradients when the recurrent units are embedded on an mPFC cortical surface.

The project is under active development

## Human 7T-fMRI ABCD task

The human ABCD instruction/navigation task is available alongside the original
Jensen maze as `--task abcd_fmri`. One episode is one complete recurrent block:
four locations are shown sequentially (twice by default), then the agent
navigates continuously through five circular four-goal loops with one explicit
reward-dwell timestep after every correct goal. The environment is implemented
in `pysta/abcd_env.py`; the Jensen `MazeEnv` remains the default task.

The ABCD observation is exactly 24 channels:

| Group | Slice | Routing |
|---|---:|---|
| `current_location` | `0:9` | local cortical anchor zone |
| `instruction_location` | `9:18` | local cortical anchor zone |
| `execution_rule` (`SAME`, `REVERSE`) | `18:20` | global |
| `phase` (`INSTRUCTION`, `NAVIGATION`, `REWARD`) | `20:23` | global |
| `reward_event` | `23:24` | global |

Only the phase-appropriate groups are active. In particular, configuration,
rule, target identity/location, distance, and future path are absent during
navigation. The ABCD readout is four logits ordered `up, down, left, right`;
boundary attempts remain possible model errors.

Task-conditioned defaults use the preserved human N480
`mpfc_projected_mask_linear0p1` cortical embedding and a global readout. For
full-block BPTT the ABCD batch default is 8 (the Maze default remains 200).
For example (this starts real N480 training, so choose resources deliberately):

```bash
python -m pysta.train_rnn --task abcd_fmri \
  --n_loops 5 --instruction_repeats 2 \
  --num_train_configurations 48 --num_eval_configurations 12
```

Explicit ordered `A,B,C,D` banks can be supplied as semicolon-separated rows,
for example `--train_configurations '0,2,8,6;8,6,0,2'`. Generated banks are
seeded, enforce the goal-distance constraint, and held-out banks exclude both
training tuples and their direct reversals. Exact experimental coordinates are
not encoded.

Two lightweight verification commands do not train N480:

```bash
python scripts/trace_abcd_block.py
python scripts/smoke_abcd_overfit.py
```

Calling `agent.forward(store=True)` records exact inputs, activity/potentials,
policies/actions, phase and task state, transition endpoints, reward/correctness,
and padding masks. `pysta.abcd_analysis_utils.save_agent_store` exports these
records, while `extract_navigation_trajectories` constructs arbitrary future
physical-location lags using navigation actions only (instruction and reward
dwell timesteps are ignored). It intentionally does not compute or reinterpret
the existing planning/Csubs analyses.

## Installation

Create and activate a conda environment:

```bash
conda create -n pysta python=3.12 pip
conda activate pysta
```

Install the package requirements:

```bash
pip install -r requirements.txt
pip install -e .
```

Current `requirements.txt`:

```text
numpy<2
scikit-learn
scipy

matplotlib
svgpathtools
svgpath2mpl

torch==2.2.2

gdist
nibabel
nilearn
```
