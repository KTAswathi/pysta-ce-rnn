# ABCD reference analysis entry points

## Run the complete managed analysis

Pass only the exact directory basename below `models/abcd_fmri/`:

```bash
python scripts/ABCD_task/run_abcd_reference_analysis.py \
  mech_baseline_seed1_a1b2c3d4e5
```

The wrapper resolves
`models/abcd_fmri/mech_baseline_seed1_a1b2c3d4e5/checkpoints/best.pt`
and never falls back to `latest.pt`. It then runs the frozen pipeline in
fail-fast order: collection once, raw activity, Csubs, and local RSA. All three
downstream commands consume that one shared collection; they never evaluate
the model independently. Outputs remain:

```text
data/abcd_task_analyses/mech_baseline_seed1_a1b2c3d4e5/
├── trial_collection/
├── raw_activity/
├── csubs/
└── local_rsa/
```

By default the wrapper refuses to overwrite any existing stage output. Use
`--skip-existing` to skip complete stages and retry incomplete/missing ones, or
`--force` to rerun the selected stages. These flags are mutually exclusive.
Run exactly one stage with `--only collect`, `--only raw`, `--only csubs`, or
`--only local_rsa`; analysis-only choices require an already complete matching
trial collection. Existing collections are reused only when their manifest and
QC hashes match the requested `best.pt` and `config.yaml`. A compact
`pipeline_status.json` additionally binds every completed stage to the exact
shared collection, fixed command/settings, scientific source files, and output
hashes. Corrupt, partial, copied, or stale output is never skipped; recollecting
trials invalidates all downstream completion records. Ambiguous or mismatched
output is refused even under `--force`.

The wrapper exposes no scientific-analysis settings. It fixes the existing
pipeline values at 1,000 Csubs label permutations (seed 881), 2,000 maximum
Csubs iterations with automatic device selection, and a 6 mm local-RSA
searchlight. It never trains/resumes a model or modifies checkpoints. Direct
legacy-checkpoint analysis remains available through the four original entry
points below; the basename-only restriction applies only to this managed-run
wrapper. Native managed training must have reached its configured update count
before analysis, preventing a changing `best.pt` from contaminating collection;
explicitly frozen, non-resumable legacy imports remain accepted.

The replacement pipeline has four plainly named steps and one shared module:

1. `collect_abcd_reference_trials.py` — two autonomous, independently-noised
   frozen-model repeats plus provenance and normalized-progress data.
2. `analyse_abcd_normalized_raw.py` — nuisance-controlled raw representation.
3. `analyse_abcd_normalized_csubs.py` — cross-fitted original-filter Csubs,
   population decoding, and the two pre-specified sensitivities.
4. `analyse_abcd_local_rsa.py` — cross-repeat local RSA and reference-code COM.

For managed runs, all outputs live below
`data/abcd_task_analyses/<run-name>_<config-hash>/` in the plainly named
`trial_collection/`, `raw_activity/`, `csubs/`, and `local_rsa/` subdirectories.
The collector accepts either a managed run directory or its
`checkpoints/best.pt`; legacy `*_best.pt` plus portable-artifact pairs remain
supported. Each analysis writes only `results.npz`, one concise table/metadata
file, and one multi-panel `summary.png` (Csubs additionally retains a text
optimizer log).

`abcd_analysis_common.py` owns checkpoint reconstruction, hashes, task-normalized
indexing, collection schemas, and cortical geometry. 
The exact historical joint-decoder implementation needed by the active Csubs
analysis lives independently in `abcd_csubs_decoder.py`.
