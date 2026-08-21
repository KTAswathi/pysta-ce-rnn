# ABCD reference analysis entry points

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
