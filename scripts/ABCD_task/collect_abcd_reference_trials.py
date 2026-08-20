"""Collect two autonomous frozen-model ABCD repeats for reference analyses.

This is the sole collection entry point for the normalized raw, Csubs, and
local-RSA analyses.  It never changes model parameters and always evaluates
the exact deterministic familiar 5 x 2 x 2 schedule with the recurrent-noise
amplitude stored in the trained checkpoint configuration.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pysta
from pysta.abcd_analysis_utils import (
    extract_navigation_trajectories,
    save_agent_store,
    stack_store_records,
)

from scripts.ABCD_task.abcd_analysis_common import (
    build_analysis_manifest,
    build_normalized_navigation,
    concatenate_normalized_blocks,
    derive_repeat_seeds,
    file_sha256,
    infer_portable_files,
    reconstruct_trained_model,
    resolve_analysis_root,
    save_normalized_navigation,
    state_dict_sha256,
    write_json,
)


def _cpu_array(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def _environment_scalar(metrics: dict[str, Any], key: str) -> int | bool:
    value = _cpu_array(metrics[key]).reshape(-1)
    if len(value) != 1:
        raise ValueError(f"Expected scalar batch-one environment metric {key!r}.")
    item = value[0]
    return bool(item) if np.issubdtype(value.dtype, np.bool_) else int(item)


def _block_summary(
    cell: dict[str, Any],
    trajectory: dict[str, Any],
) -> dict[str, Any]:
    environment = cell["environment"]
    goals = _environment_scalar(environment, "successful_goal_count")
    expected = _environment_scalar(environment, "required_goal_count")
    completed_metric = _environment_scalar(environment, "completed")
    truncated = _environment_scalar(environment, "truncated")
    return {
        "factorial_index": int(cell["factorial_index"]),
        "base_configuration_index": int(cell["base_configuration_index"]),
        "base_configuration": [int(x) for x in cell["base_configuration"]],
        "instruction_direction": int(cell["instruction_direction"]),
        "execution_relation": int(cell["execution_relation"]),
        "presented_abstract_sequence": [
            int(x) for x in cell["presented_abstract_sequence"]
        ],
        "effective_execution_abstract_sequence": [
            int(x) for x in cell["effective_execution_abstract_sequence"]
        ],
        "task_seed": int(cell["task_seed"]),
        "start_location": int(cell["start_location"]),
        "loss": float(cell["loss"]),
        "accuracy": float(cell["accuracy"]),
        "navigation_steps": int(trajectory["num_navigation_steps"]),
        "successful_goals": int(goals),
        "expected_goals": int(expected),
        "finished": bool(completed_metric),
        "truncated": bool(truncated),
        "completed": bool(completed_metric and not truncated and goals == expected),
        "route_consistency": cell["route_consistency"],
    }


def collect_repeat(
    model: torch.nn.Module,
    schedule: tuple[Any, ...],
    *,
    repeat_index: int,
    evaluation_seed: int,
    repeat_dir: Path,
    expected_loops: int,
    num_locations: int,
) -> dict[str, Any]:
    """Run and save one complete independently-noised frozen evaluation."""

    before_hash = state_dict_sha256(model)
    _, _, metrics = pysta.train_rnn.evaluate_abcd_fmri_factorial(
        model, schedule, evaluation_seed=evaluation_seed
    )
    after_hash = state_dict_sha256(model)
    if before_hash != after_hash:
        raise RuntimeError("A trained parameter changed during trial collection.")
    if not metrics.get("autonomous") or metrics.get("force_optimal"):
        raise RuntimeError("Reference collection was not autonomous.")
    if not metrics.get("greedy") or not metrics.get("weights_frozen"):
        raise RuntimeError("Reference collection must be greedy with frozen weights.")

    stores = metrics.pop("trajectory_stores")
    cells = metrics["factorial_cells"]
    if len(stores) != len(schedule) or len(cells) != len(schedule):
        raise RuntimeError("Factorial metadata and trajectory-store counts differ.")

    blocks_dir = repeat_dir / "blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)
    normalized_blocks = []
    block_summaries = []
    for schedule_cell, cell, store in zip(schedule, cells, stores):
        stacked = stack_store_records(store)
        block_index = int(cell["factorial_index"])
        time_steps, batch_size = np.asarray(stacked["valid_timestep"]).shape
        if batch_size != 1:
            raise RuntimeError("Reference factorial stores must be batch one.")
        stacked["factorial_index"] = np.full(
            (time_steps, batch_size), block_index, dtype=np.int16
        )
        stacked["base_configuration_index"] = np.full(
            (time_steps, batch_size),
            int(cell["base_configuration_index"]),
            dtype=np.int16,
        )
        stacked["evaluation_repeat_index"] = np.full(
            (time_steps, batch_size), int(repeat_index), dtype=np.int8
        )
        save_agent_store(
            blocks_dir / f"block_{block_index:03d}.npz",
            stacked,
            overwrite=True,
        )
        trajectories = extract_navigation_trajectories(stacked, future_lags=(0,))
        if len(trajectories) != 1:
            raise RuntimeError("Factorial cells must contain one batch-one trajectory.")
        trajectory = trajectories[0]
        metadata = {
            **cell,
            "factorial_index": int(schedule_cell.factorial_index),
            "base_configuration_index": int(
                schedule_cell.base_configuration_index
            ),
        }
        normalized_blocks.append(
            build_normalized_navigation(
                trajectory,
                metadata,
                repeat_index=repeat_index,
                expected_loops=expected_loops,
            )
        )
        block_summaries.append(_block_summary(cell, trajectory))

    normalized = concatenate_normalized_blocks(
        normalized_blocks, num_locations=num_locations
    )
    normalized_path = save_normalized_navigation(
        repeat_dir / "normalized_navigation.npz", normalized
    )
    completed = sum(int(block["completed"]) for block in block_summaries)
    summary = {
        "repeat_index": int(repeat_index),
        "evaluation_seed": int(evaluation_seed),
        "recurrent_noise": float(metrics["evaluation_recurrent_noise"]),
        "autonomous": True,
        "greedy": True,
        "force_optimal": False,
        "weights_frozen": True,
        "num_blocks": len(block_summaries),
        "completed_blocks": completed,
        "mean_accuracy": float(metrics["accuracy"]),
        "normalized_rows": int(len(normalized["rs"])),
        "normalized_rows_per_block": int(
            len(normalized["rs"]) // len(block_summaries)
        ),
        "normalized_navigation": str(normalized_path),
        "blocks": block_summaries,
    }
    write_json(repeat_dir / "behaviour_summary.json", summary)
    return summary


def collect_reference_trials(
    checkpoint: Path,
    *,
    output_dir: Path | None = None,
) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    portable_state, portable_kwargs = infer_portable_files(checkpoint)
    model, kwargs = reconstruct_trained_model(portable_state, portable_kwargs)
    if int(model.Nout) != 4 or int(model.Nin) != 24:
        raise ValueError(
            f"Expected the ABCD 24-input/4-output interface, got "
            f"Nin={model.Nin}, Nout={model.Nout}."
        )

    schedule = tuple(pysta.tasks.make_fmri_evaluation_schedule(kwargs))
    # The evaluator itself verifies exact base-major factorial balance.  Keep
    # this collection dynamic with respect to unit count and geometry.
    repeat_seeds = derive_repeat_seeds(int(kwargs["seed"]))
    analysis_root = resolve_analysis_root(checkpoint, output_dir)
    collection_dir = analysis_root / "trial_collection"
    collection_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_analysis_manifest(
        checkpoint=checkpoint,
        portable_state=portable_state,
        portable_kwargs=portable_kwargs,
        model=model,
        kwargs=kwargs,
        schedule=schedule,
        repeat_seeds=repeat_seeds,
        analysis_root=analysis_root,
    )
    write_json(analysis_root / "analysis_manifest.json", manifest)

    trained_noise = float(model.rec_noise)
    if trained_noise <= 0:
        raise ValueError(
            "Two independent same-regime repeats require positive trained "
            "recurrent noise."
        )
    parameter_snapshot = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    repeat_summaries = []
    for repeat_index, seed in enumerate(repeat_seeds, start=1):
        if float(model.rec_noise) != trained_noise:
            raise RuntimeError("Recurrent noise changed between repeats.")
        repeat_summaries.append(
            collect_repeat(
                model,
                schedule,
                repeat_index=repeat_index,
                evaluation_seed=seed,
                repeat_dir=collection_dir / f"repeat_{repeat_index:02d}",
                expected_loops=int(manifest["task"]["num_loops_per_full_block"]),
                num_locations=int(manifest["task"]["num_locations"]),
            )
        )
    for name, before in parameter_snapshot.items():
        if not torch.equal(before, model.state_dict()[name].detach().cpu()):
            raise RuntimeError(f"Frozen parameter {name!r} changed during collection.")

    starts = [
        [block["start_location"] for block in repeat["blocks"]]
        for repeat in repeat_summaries
    ]
    conditions = [
        [
            (
                block["base_configuration_index"],
                block["instruction_direction"],
                block["execution_relation"],
            )
            for block in repeat["blocks"]
        ]
        for repeat in repeat_summaries
    ]
    if conditions[0] != conditions[1]:
        raise RuntimeError("The two repeats do not contain the same factorial cells.")
    if starts[0] != starts[1]:
        raise RuntimeError("The two repeats do not share the same task start states.")
    qc = {
        "schema": "abcd_reference_trial_collection/v1",
        "checkpoint_sha256": file_sha256(checkpoint),
        "portable_state_sha256": file_sha256(portable_state),
        "loaded_state_dict_sha256": state_dict_sha256(model),
        "weights_unchanged": True,
        "same_trained_recurrent_noise": True,
        "distinct_recurrent_noise_seeds": len(set(repeat_seeds)) == 2,
        "matched_factorial_cells": True,
        "matched_start_locations": True,
        "repeats": repeat_summaries,
    }
    write_json(collection_dir / "collection_qc.json", qc)
    return analysis_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional model-specific analysis root.",
    )
    args = parser.parse_args()
    root = collect_reference_trials(args.checkpoint, output_dir=args.output_dir)
    print(f"Reference trial collection: {root / 'trial_collection'}")
    print(f"Analysis manifest: {root / 'analysis_manifest.json'}")


if __name__ == "__main__":
    main()
