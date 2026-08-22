
import numpy as np
import time
import pickle
import copy
import json
import os
import sys
import torch
import pysta
from pysta import run_manager


def _make_rnn(env, kwargs):
    """Construct the requested recurrent model around an environment."""
    if kwargs["model_type"] == "vanilla":
        return pysta.agents.VanillaRNN(env, **kwargs)
    if kwargs["model_type"] == "lineembedded":
        return pysta.agents.LineEmbeddedRNN(env, **kwargs)
    if kwargs["model_type"] == "corticallyembedded":
        return pysta.agents.CorticallyEmbeddedRNN(env, **kwargs)
    raise ValueError(f"Unknown model_type: {kwargs['model_type']}")


def _json_safe_training_value(value):
    """Convert resolved training arguments to lossless JSON-compatible values."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return _json_safe_training_value(value.item())
    if isinstance(value, np.ndarray):
        return [_json_safe_training_value(item) for item in value.tolist()]
    if torch.is_tensor(value):
        return _json_safe_training_value(value.detach().cpu().tolist())
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, dict):
        return {
            str(key): _json_safe_training_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_training_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_json_safe_training_value(item) for item in value),
            key=repr,
        )
    raise TypeError(
        "ABCD portable constructor arguments must be JSON-compatible; "
        f"unsupported value {value!r} of type {type(value).__name__}."
    )


def _write_abcd_portable_kwargs(savename, kwargs):
    """Save resolved ABCD constructor arguments using the analysis filename contract."""
    destination = f"{savename}_portable_kwargs.json"
    payload = _json_safe_training_value(kwargs)
    with open(destination, "w", encoding="utf8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return destination


def _save_best_checkpoint(rnn, savename, *, task):
    """Save the established checkpoint plus an exact portable ABCD state dict."""
    checkpoint_path = f"{savename}_best.pt"
    torch.save(rnn, checkpoint_path)
    portable_state_path = None
    if task == "abcd_fmri":
        portable_state_path = f"{savename}_best_portable_state_dict.pt"
        # CPU clones make this artifact device-independent without mutating the
        # live model or changing the established whole-object checkpoint.
        portable_state = {
            name: value.detach().cpu().clone()
            for name, value in rnn.state_dict().items()
        }
        torch.save(portable_state, portable_state_path)
    return checkpoint_path, portable_state_path


def _stored_execution_accuracy(store):
    """Calculate accuracy only where the environment requests policy loss."""
    corrects = []
    masks = []
    for state in store:
        mask = state.get("policy_loss_mask", state.get("loss_mask"))
        if mask is None:
            # Backward-compatible fallback for stores made by the original agent.
            if state.get("step_num", -1) < 0:
                continue
            mask = ~state["finished"]
        corrects.append(state["corrects"])
        masks.append(mask.to(dtype=torch.bool))
    if not corrects:
        return np.nan
    corrects = torch.stack(corrects)
    masks = torch.stack(masks).to(device=corrects.device)
    corrects = corrects.clone()
    corrects[~masks] = torch.nan
    return float(torch.nanmean(corrects, axis=0).mean().detach().cpu())


def _metric_to_cpu(value):
    """Convert environment metrics to pickle-friendly CPU values."""
    if torch.is_tensor(value):
        value = value.detach().cpu()
        return value.item() if value.ndim == 0 else value.numpy()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _aggregate_environment_metrics(block_metrics):
    """Combine scalar and per-trial metrics across evaluation blocks."""
    if not block_metrics:
        return {}

    combined = {}
    keys = set().union(*(metrics.keys() for metrics in block_metrics))
    for key in sorted(keys):
        values = [metrics[key] for metrics in block_metrics if key in metrics]
        arrays = [np.asarray(value) for value in values]
        if all(array.ndim == 0 for array in arrays):
            try:
                combined[key] = float(np.mean(arrays))
            except (TypeError, ValueError):
                combined[key] = values
        else:
            try:
                combined[key] = np.concatenate(
                    [array.reshape(1) if array.ndim == 0 else array for array in arrays],
                    axis=0,
                )
            except (TypeError, ValueError):
                combined[key] = values
    return combined


def _aggregate_route_consistency(block_summaries):
    """Pool count-based route metrics without comparing routes across trials."""
    if not block_summaries:
        return {}
    overalls = [summary["overall"] for summary in block_summaries]

    def summed(name):
        return sum(float(overall[name]) for overall in overalls)

    action_pairs = int(summed("action_pair_count"))
    action_matches = int(summed("exact_action_pair_matches"))
    location_pairs = int(summed("location_pair_count"))
    location_matches = int(summed("exact_location_pair_matches"))
    completed_routes = int(summed("completed_route_count"))
    shortest_routes = int(summed("shortest_route_count"))
    navigation_actions = int(summed("navigation_action_count"))
    invalid_actions = int(summed("invalid_boundary_action_count"))
    consistency_routes = int(summed("consistency_route_count"))
    action_modal_routes = sum(
        (overall["action_modal_fraction"] or 0.0)
        * overall["consistency_route_count"]
        for overall in overalls
    )
    location_modal_routes = sum(
        (overall["location_modal_fraction"] or 0.0)
        * overall["consistency_route_count"]
        for overall in overalls
    )
    return {
        "action_source": "env_action",
        "initial_block_start_leg_excluded_from_repetition": True,
        "num_blocks": len(block_summaries),
        "num_trials": sum(summary["num_trials"] for summary in block_summaries),
        "complete_trial_count": sum(
            trial["status"] == "complete"
            for summary in block_summaries
            for trial in summary["trials"]
        ),
        "truncated_trial_count": sum(
            trial["status"] == "truncated"
            for summary in block_summaries
            for trial in summary["trials"]
        ),
        "consistency_route_count": consistency_routes,
        "action_pair_count": action_pairs,
        "exact_action_pair_matches": action_matches,
        "exact_action_pairwise_rate": (
            action_matches / action_pairs if action_pairs else None
        ),
        "location_pair_count": location_pairs,
        "exact_location_pair_matches": location_matches,
        "exact_location_pairwise_rate": (
            location_matches / location_pairs if location_pairs else None
        ),
        "action_modal_fraction": (
            action_modal_routes / consistency_routes
            if consistency_routes
            else None
        ),
        "location_modal_fraction": (
            location_modal_routes / consistency_routes
            if consistency_routes
            else None
        ),
        "completed_route_count": completed_routes,
        "shortest_route_count": shortest_routes,
        "shortest_route_fraction": (
            shortest_routes / completed_routes if completed_routes else None
        ),
        "total_excess_actions": int(summed("total_excess_actions")),
        "mean_excess_actions": (
            summed("total_excess_actions") / completed_routes
            if completed_routes
            else None
        ),
        "navigation_action_count": navigation_actions,
        "invalid_boundary_action_count": invalid_actions,
        "invalid_boundary_action_fraction": (
            invalid_actions / navigation_actions if navigation_actions else None
        ),
    }


def _cpu_snapshot(value):
    """Recursively detach factorial-evaluation trajectories onto CPU."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    return value


def _capture_torch_rng_states(device):
    """Capture RNG state so deterministic evaluation does not perturb training."""
    states = {"cpu": torch.random.get_rng_state()}
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    if device.type == "mps" and hasattr(torch, "mps"):
        states["mps"] = torch.mps.get_rng_state()
    return states


def _seed_torch_rng(seed, device):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.manual_seed(int(seed))


def _restore_torch_rng_states(states):
    torch.random.set_rng_state(states["cpu"])
    if "cuda" in states:
        torch.cuda.set_rng_state_all(states["cuda"])
    if "mps" in states:
        torch.mps.set_rng_state(states["mps"])


_AGENT_TRANSIENT_STATE_NAMES = (
    "z",
    "r",
    "logpi",
    "pi",
    "action",
    "env_action",
    "optimal_actions",
    "acc_loss",
    "weight_loss",
    "rate_loss",
    "ent_loss",
    "store",
    "all_acts",
    "all_acts_time_name",
    "all_acts_metadata",
)


_ABCD_ENV_TRANSIENT_STATE_NAMES = (
    "configuration",
    "configuration_index",
    "instruction_direction",
    "execution_relation",
    "presented_sequence",
    "effective_execution_sequence",
    "phase",
    "instruction_presentation_index",
    "sequence_position",
    "loop_index",
    "successful_goal_count",
    "navigation_step_count",
    "block_timestep",
    "reward_event",
    "latest_rew",
    "finished",
    "truncated",
    "_truncate_after_reward",
    "loc",
    "start_location",
    "started_on_first_goal",
    "_post_step",
)


def _capture_agent_transient_state(rnn):
    """Retain exact pre-evaluation state objects for transparent restoration."""
    present = {
        name: getattr(rnn, name)
        for name in _AGENT_TRANSIENT_STATE_NAMES
        if hasattr(rnn, name)
    }
    missing = set(_AGENT_TRANSIENT_STATE_NAMES).difference(present)
    return present, missing


def _restore_agent_transient_state(rnn, present, missing):
    """Undo state/cache changes made by factorial ``forward`` calls."""
    for name, value in present.items():
        setattr(rnn, name, value)
    for name in missing:
        if hasattr(rnn, name):
            delattr(rnn, name)


def _capture_abcd_environment_state(env):
    """Capture exact evaluation-only objects changed by an ABCD rollout."""
    present = {
        name: getattr(env, name)
        for name in _ABCD_ENV_TRANSIENT_STATE_NAMES
        if hasattr(env, name)
    }
    missing = set(_ABCD_ENV_TRANSIENT_STATE_NAMES).difference(present)
    rng_state = copy.deepcopy(env.rng.bit_generator.state)
    return present, missing, env.rng, rng_state, int(env._block_counter)


def _restore_abcd_environment_state(env, state):
    """Restore the evaluation environment to its exact pre-validation state."""
    present, missing, rng, rng_state, block_counter = state
    for name, value in present.items():
        setattr(env, name, value)
    for name in missing:
        if hasattr(env, name):
            delattr(env, name)
    env.rng = rng
    env.rng.bit_generator.state = copy.deepcopy(rng_state)
    env._block_counter = int(block_counter)


@torch.no_grad()
def _evaluate_abcd(
    rnn,
    eval_env,
    num_eval,
    evaluation_mode,
    validation_seed=None,
):
    """Evaluate complete ABCD blocks on the selected configuration set.

    Each call to ``forward`` resets the evaluation environment and runs one
    complete block. The fixed validation seed independently replays both the
    evaluation-environment draws and recurrent-noise draws at every call.
    Global RNG streams, model caches, the training environment, and the
    evaluation environment are restored even if an evaluation block fails, so
    inserting validation cannot change the next training update.
    """
    train_env = rnn.env
    train_greedy = rnn.greedy
    train_force_optimal = rnn.force_optimal
    device = next(rnn.parameters()).device
    rng_states = _capture_torch_rng_states(device)
    numpy_rng_state = np.random.get_state()
    transient_state, transient_state_missing = _capture_agent_transient_state(rnn)
    eval_environment_state = _capture_abcd_environment_state(eval_env)
    if validation_seed is None:
        # The split-specific evaluation task seed is independent of the model/
        # training seed and is already part of the immutable run configuration.
        validation_seed = int(eval_env.seed)

    if eval_env.obs_dim != train_env.obs_dim:
        raise ValueError(
            "Training and evaluation environments must have the same "
            f"observation dimension ({train_env.obs_dim} != {eval_env.obs_dim})."
        )
    if eval_env.output_dim != train_env.output_dim:
        raise ValueError(
            "Training and evaluation environments must have the same "
            f"output dimension ({train_env.output_dim} != {eval_env.output_dim})."
        )

    block_losses = []
    block_accuracies = []
    environment_blocks = []
    route_consistency_blocks = []
    try:
        _seed_torch_rng(validation_seed, device)
        np.random.seed(int(validation_seed))
        eval_env.rng = np.random.default_rng(int(eval_env.seed))
        eval_env._block_counter = -1
        rnn.env = eval_env
        rnn.greedy = True
        rnn.force_optimal = False
        for _ in range(num_eval):
            # BaseAgent.forward calls env.reset() once and retains the full
            # block state sequence. This helper is no-grad, while training
            # remains full-block BPTT.
            block_loss = rnn.forward(store=True)
            block_losses.append(float(block_loss.detach().cpu()))
            block_accuracies.append(_stored_execution_accuracy(rnn.store))
            route_consistency_blocks.append(
                pysta.abcd_analysis_utils.summarize_route_consistency(rnn)
            )
            if hasattr(eval_env, "evaluation_metrics"):
                environment_blocks.append({
                    key: _metric_to_cpu(value)
                    for key, value in eval_env.evaluation_metrics().items()
                })
    finally:
        rnn.env = train_env
        rnn.greedy = train_greedy
        rnn.force_optimal = train_force_optimal
        _restore_torch_rng_states(rng_states)
        np.random.set_state(numpy_rng_state)
        _restore_agent_transient_state(
            rnn, transient_state, transient_state_missing
        )
        _restore_abcd_environment_state(eval_env, eval_environment_state)

    metrics = {
        "evaluation_mode": str(evaluation_mode),
        "configuration_bank_name": eval_env.bank_name,
        "configuration_bank": tuple(eval_env.configuration_bank),
        "configuration_bank_statistics": (
            pysta.abcd_env.configuration_bank_statistics(
                eval_env.configuration_bank
            )
        ),
        "evaluation_task_seed": int(eval_env.seed),
        "validation_recurrent_noise_seed": int(validation_seed),
        "validation_rng_replayed": True,
        "loss": float(np.mean(block_losses)),
        "accuracy": float(np.nanmean(block_accuracies)),
        "block_losses": block_losses,
        "block_accuracies": block_accuracies,
        "num_blocks": int(num_eval),
        "environment": _aggregate_environment_metrics(environment_blocks),
        "environment_blocks": environment_blocks,
        "route_consistency": _aggregate_route_consistency(
            route_consistency_blocks
        ),
        "route_consistency_blocks": route_consistency_blocks,
        "evaluation_recurrent_noise": float(rnn.rec_noise),
    }
    return metrics["loss"], metrics["accuracy"], metrics


def _evaluate_heldout(rnn, eval_env, num_eval):
    """Backward-compatible wrapper for the former held-out-only helper."""
    return _evaluate_abcd(
        rnn,
        eval_env,
        num_eval,
        evaluation_mode="heldout",
    )


def _validation_checkpoint_is_better(
    accuracy,
    loss,
    *,
    best_accuracy,
    best_loss,
):
    """Rank ABCD checkpoints by accuracy, then total validation loss."""

    candidate_accuracy = float(accuracy)
    incumbent_accuracy = float(best_accuracy)
    if np.isnan(candidate_accuracy):
        candidate_accuracy = -np.inf
    if np.isnan(incumbent_accuracy):
        incumbent_accuracy = -np.inf
    return bool(
        candidate_accuracy > incumbent_accuracy
        or (
            candidate_accuracy == incumbent_accuracy
            and float(loss) < float(best_loss)
        )
    )


def _best_accuracy_from_history(validation_history, best_update):
    """Recover the selected checkpoint accuracy from resumable history."""

    if best_update is None:
        return -np.inf
    matches = [
        row
        for row in validation_history
        if int(row["update"]) == int(best_update)
    ]
    if not matches:
        raise ValueError(
            f"best.pt update {best_update} has no validation-history row."
        )
    accuracy = matches[-1].get("accuracy")
    if accuracy is None or np.isnan(float(accuracy)):
        return -np.inf
    return float(accuracy)


@torch.no_grad()
def evaluate_abcd_fmri_factorial(rnn, schedule, evaluation_seed=None):
    """Run the frozen model over the exact final 5 x 2 x 2 design.

    ``schedule`` must be produced by
    :func:`pysta.tasks.make_fmri_evaluation_schedule`. It contains one
    independent batch-one environment for each base-major factorial cell, so
    every block resets recurrent state and no condition is randomly sampled.
    Model actions are autonomous and greedy; invalid boundary actions remain
    genuine errors. Complete CPU trajectory stores are returned, aligned with
    explicit factorial-cell metadata for later behavioural/fMRI analyses.
    """
    from pysta.abcd_env import (
        EXECUTION_RELATION_NAMES,
        INSTRUCTION_DIRECTION_NAMES,
    )

    schedule = tuple(schedule)
    if len(schedule) != 20:
        raise ValueError(
            "Final factorial fMRI evaluation requires exactly 20 cells "
            "(five bases x two instruction directions x two execution relations)."
        )

    observed_order = [
        (
            int(cell.factorial_index),
            int(cell.base_configuration_index),
            int(cell.instruction_direction),
            int(cell.execution_relation),
        )
        for cell in schedule
    ]
    expected_order = [
        (factorial_index, base_index, instruction_direction, execution_relation)
        for factorial_index, (base_index, instruction_direction, execution_relation) in enumerate(
            (
                (base_index, instruction_direction, execution_relation)
                for base_index in range(5)
                for instruction_direction in range(2)
                for execution_relation in range(2)
            )
        )
    ]
    if observed_order != expected_order:
        raise ValueError(
            "The factorial schedule must be in exact base-major order and "
            "enumerate each base x instruction direction x execution relation "
            "exactly once."
        )

    train_env = rnn.env
    train_greedy = rnn.greedy
    train_force_optimal = rnn.force_optimal
    device = next(rnn.parameters()).device
    rng_states = _capture_torch_rng_states(device)
    numpy_rng_state = np.random.get_state()
    transient_state, transient_state_missing = _capture_agent_transient_state(rnn)
    if evaluation_seed is None:
        evaluation_seed = int(schedule[0].seed)

    block_losses = []
    block_accuracies = []
    environment_blocks = []
    route_consistency_blocks = []
    factorial_cells = []
    trajectory_stores = []
    try:
        _seed_torch_rng(evaluation_seed, device)
        np.random.seed(int(evaluation_seed))
        rnn.greedy = True
        rnn.force_optimal = False
        for cell in schedule:
            eval_env = cell.environment
            if eval_env.batch != 1:
                raise ValueError("Each factorial evaluation environment must have batch=1.")
            if eval_env.obs_dim != train_env.obs_dim:
                raise ValueError(
                    "Training and factorial evaluation environments must have "
                    f"the same observation dimension ({train_env.obs_dim} != "
                    f"{eval_env.obs_dim})."
                )
            if eval_env.output_dim != train_env.output_dim:
                raise ValueError(
                    "Training and factorial evaluation environments must have "
                    f"the same output dimension ({train_env.output_dim} != "
                    f"{eval_env.output_dim})."
                )

            # Rewind the environment-owned RNG as well as recurrent noise so a
            # repeated call with the same schedule is bitwise reproducible.
            eval_env.rng = np.random.default_rng(int(cell.seed))
            eval_env._block_counter = -1
            rnn.env = eval_env
            block_loss = rnn.forward(store=True)
            loss = float(block_loss.detach().cpu())
            accuracy = _stored_execution_accuracy(rnn.store)
            route_summary = pysta.abcd_analysis_utils.summarize_route_consistency(rnn)
            environment_metrics = {
                key: _metric_to_cpu(value)
                for key, value in eval_env.evaluation_metrics().items()
            }

            actual_configuration = tuple(int(value) for value in eval_env.configuration[0])
            actual_instruction = int(eval_env.instruction_direction[0])
            actual_relation = int(eval_env.execution_relation[0])
            if (
                actual_configuration != tuple(cell.configuration)
                or actual_instruction != int(cell.instruction_direction)
                or actual_relation != int(cell.execution_relation)
            ):
                raise RuntimeError(
                    "A factorial environment resampled its fixed configuration "
                    "or condition; the evaluation would not be balanced."
                )

            cell_metadata = {
                "factorial_index": int(cell.factorial_index),
                "base_configuration_index": int(cell.base_configuration_index),
                "base_configuration": tuple(cell.configuration),
                "instruction_direction": actual_instruction,
                "instruction_direction_name": INSTRUCTION_DIRECTION_NAMES[
                    actual_instruction
                ],
                "execution_relation": actual_relation,
                "execution_relation_name": EXECUTION_RELATION_NAMES[actual_relation],
                "presented_abstract_sequence": tuple(
                    int(value) for value in eval_env.presented_sequence[0]
                ),
                "effective_execution_abstract_sequence": tuple(
                    int(value) for value in eval_env.effective_execution_sequence[0]
                ),
                "task_seed": int(cell.seed),
                "start_location": int(eval_env.start_location[0]),
                "loss": loss,
                "accuracy": accuracy,
                "environment": environment_metrics,
                "route_consistency": route_summary,
            }
            block_losses.append(loss)
            block_accuracies.append(accuracy)
            environment_blocks.append(environment_metrics)
            route_consistency_blocks.append(route_summary)
            factorial_cells.append(cell_metadata)
            trajectory_stores.append(_cpu_snapshot(rnn.store))
    finally:
        rnn.env = train_env
        rnn.greedy = train_greedy
        rnn.force_optimal = train_force_optimal
        _restore_torch_rng_states(rng_states)
        np.random.set_state(numpy_rng_state)
        _restore_agent_transient_state(
            rnn, transient_state, transient_state_missing
        )

    metrics = {
        "evaluation_design": "five_base_x_instruction_direction_x_execution_relation",
        "factorial_order": (
            "base_configuration_index, then FORWARD/BACKWARD "
            "instruction_direction, then SAME/REVERSE execution_relation"
        ),
        "weights_frozen": True,
        "autonomous": True,
        "greedy": True,
        "force_optimal": False,
        "num_base_configurations": 5,
        "num_blocks": 20,
        "base_configurations": tuple(
            tuple(schedule[base_index * 4].configuration)
            for base_index in range(5)
        ),
        "base_configuration_statistics": (
            pysta.abcd_env.configuration_bank_statistics(
                tuple(
                    schedule[base_index * 4].configuration
                    for base_index in range(5)
                )
            )
        ),
        "evaluation_seed": int(evaluation_seed),
        "evaluation_recurrent_noise": float(rnn.rec_noise),
        "loss": float(np.mean(block_losses)),
        "accuracy": float(np.nanmean(block_accuracies)),
        "block_losses": block_losses,
        "block_accuracies": block_accuracies,
        "factorial_cells": factorial_cells,
        "environment": _aggregate_environment_metrics(environment_blocks),
        "route_consistency": _aggregate_route_consistency(
            route_consistency_blocks
        ),
        "trajectory_stores": trajectory_stores,
    }
    return metrics["loss"], metrics["accuracy"], metrics


def main_train(kwargs):
    
    # print arguments
    print("\n\nSetting up model for training:")
    print(kwargs)

    # set some parameters
    np.random.seed(kwargs["seed"])
    torch.manual_seed(kwargs["seed"])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # instantiate model
    task = kwargs.get("task", "maze")
    evaluation_mode = (
        pysta.tasks._abcd_evaluation_mode(kwargs)
        if task == "abcd_fmri"
        else None
    )
    env = pysta.tasks.make_environment(kwargs, split="train")
    eval_env = (
        pysta.tasks.make_environment(kwargs, split="eval")
        if task == "abcd_fmri"
        else None
    )
    fmri_factorial_schedule = None
    if kwargs.get("run_final_fmri_evaluation", False):
        # Validate before optimization starts, rather than discovering a
        # missing/malformed five-base bank after a long training run.
        fmri_factorial_schedule = pysta.tasks.make_fmri_evaluation_schedule(kwargs)

    # choose between baseline, line-embedded, and cortical-embedded RNN
    rnn = _make_rnn(env, kwargs).to(device)

    # create some filenames and directories
    managed_run = kwargs.get("run_name") is not None
    resume_managed = bool(kwargs.get("resume", False))
    run_dir = None
    run_document = None
    managed_resume_provenance_hash = None
    if managed_run:
        if task != "abcd_fmri":
            raise ValueError("--run_name is currently supported only for --task abcd_fmri.")
        if not kwargs.get("save_results", True):
            raise ValueError("--run_name requires --save_results 1.")
        if kwargs.get("prefix"):
            raise ValueError("--prefix is a legacy-layout option; use --run_name alone.")
        if kwargs.get("overwrite"):
            raise ValueError(
                "Managed run directories never overwrite; use --resume 1 for an "
                "exact interrupted run or choose another --run_name/configuration."
            )
        if int(kwargs["num_epochs"]) < 1:
            raise ValueError("Managed training requires --num_epochs >= 1.")
        training_config = run_manager.build_training_config(
            kwargs,
            env,
            rnn,
            repo_root=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            eval_env=eval_env,
            fmri_factorial_schedule=fmri_factorial_schedule,
        )
        launch_controls = {
            key: kwargs.get(key)
            for key in sorted(run_manager.NON_CONFIG_ARGUMENTS)
        }
        launch_controls["training_device"] = str(device)
        run_dir, run_document = run_manager.prepare_run_directory(
            basedir=pysta.utils.basedir,
            run_name=kwargs["run_name"],
            training_config=training_config,
            launch_controls=launch_controls,
            resume=resume_managed,
            repo_root=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        )
        managed_resume_provenance_hash = run_manager.resume_provenance_hash(
            run_document
        )
        dirname = str(run_dir)
        savename = None
    else:
        if resume_managed:
            raise ValueError("--resume requires --run_name.")
        model_namespace = f"{env.name}/{rnn.name}"
        if task == "abcd_fmri":
            # Best-checkpoint selection depends on the evaluation configuration
            # set, so keep familiar- and held-out-selected runs unambiguous on disk.
            model_namespace = f"{model_namespace}/evaluation_{evaluation_mode}"
        dirname = f"{pysta.utils.basedir}/models/{model_namespace}"
        savename = f"{dirname}/{kwargs['prefix']}model{kwargs['seed']}"
    print("Saving to:")
    print(run_dir if managed_run else savename)

    if not managed_run and os.path.isfile(f"{savename}.p"): # if this model already exists, check whether we can overwrite
        if bool(kwargs["overwrite"]):
            print(f"{savename} aleady exists, overwriting!")
        else:
            print(f"{savename} already exists, exiting!")
            raise FileExistsError
        
    if kwargs["save_results"]:
        os.makedirs(f"{dirname}", exist_ok = True)
        if task == "abcd_fmri" and not managed_run:
            _write_abcd_portable_kwargs(savename, kwargs)

    # instantiate optimizer and some variables to keep track of
    optim = torch.optim.Adam(rnn.parameters(), lr=kwargs["lrate"])
    all_losses, all_accs, eval_metrics = [], [], []
    validation_history = []
    managed_validation_metrics = []
    fmri_factorial_metrics = None
    best_loss = np.inf
    best_accuracy = -np.inf
    best_update = None
    start_update = 0
    epoch = -1  # permits a setup/save-only run with num_epochs=0
    if managed_run and not resume_managed:
        # Establish an exact update-zero recovery boundary before the first
        # validation call advances the evaluation environment RNG.
        run_manager.save_checkpoint(
            run_dir,
            run_manager.checkpoint_payload(
                kind="latest",
                config_hash=run_document["config_hash"]["full"],
                resume_provenance_hash=managed_resume_provenance_hash,
                completed_updates=0,
                model=rnn,
                optimizer=optim,
                best_validation_loss=float(best_loss),
                best_update=best_update,
                validation_history=validation_history,
                env=env,
                eval_env=eval_env,
            ),
        )
    if managed_run and resume_managed:
        (
            start_update,
            best_loss,
            best_update,
            validation_history,
        ) = run_manager.load_latest_checkpoint(
            run_dir,
            expected_config_hash=run_document["config_hash"]["full"],
            expected_resume_provenance_hash=managed_resume_provenance_hash,
            model=rnn,
            optimizer=optim,
            env=env,
            eval_env=eval_env,
            device=device,
        )
        all_losses = [float(row["loss"]) for row in validation_history]
        all_accs = [
            np.nan if row["accuracy"] is None else float(row["accuracy"])
            for row in validation_history
        ]
        best_accuracy = _best_accuracy_from_history(
            validation_history, best_update
        )
        checkpointed_validation_updates = {
            int(row["update"]) for row in validation_history
        }
        managed_validation_metrics = [
            row
            for row in run_manager.load_validation_metrics(run_dir)
            if int(row["update"]) in checkpointed_validation_updates
        ]
        run_manager.save_validation_metrics(run_dir, managed_validation_metrics)
        if start_update > int(kwargs["num_epochs"]):
            raise ValueError(
                f"latest.pt has {start_update} completed updates, exceeding "
                f"configured num_epochs={kwargs['num_epochs']}."
            )
        print(f"Resuming managed run after {start_update} completed updates.")

    # print training message
    time.sleep(5e-2)
    print(f"Training {kwargs['num_epochs']} batches of size {rnn.env.batch} on {device}")

    # now run actual training loop
    t0 = time.time()
    elapsed_offset_minutes = (
        float(validation_history[-1]["elapsed_minutes"])
        if validation_history
        else 0.0
    )
    for epoch in range(start_update, kwargs["num_epochs"]):
        
        if epoch % kwargs["eval_freq"] == 0:
            with torch.no_grad():
                if task == "abcd_fmri":
                    loss, acc, metrics = _evaluate_abcd(
                        rnn,
                        eval_env,
                        kwargs["num_eval"],
                        evaluation_mode=evaluation_mode,
                    )
                    metrics["epoch"] = epoch
                    metrics["update"] = epoch
                    eval_metrics.append(metrics)
                else:
                    # Preserve the original same-environment evaluation.
                    loss, acc = rnn.eval(num_eval = kwargs["num_eval"])
                all_losses.append(loss)
                all_accs.append(acc)
                if task == "abcd_fmri":
                    checkpoint_is_better = _validation_checkpoint_is_better(
                        acc,
                        loss,
                        best_accuracy=best_accuracy,
                        best_loss=best_loss,
                    )
                else:
                    # Preserve the original-task checkpoint rule unchanged.
                    checkpoint_is_better = loss <= best_loss
                if checkpoint_is_better:
                    best_loss = loss
                    if task == "abcd_fmri":
                        best_accuracy = (
                            -np.inf if np.isnan(float(acc)) else float(acc)
                        )
                    best_update = epoch
                    if kwargs["save_results"]:
                        if managed_run:
                            run_manager.save_checkpoint(
                                run_dir,
                                run_manager.checkpoint_payload(
                                    kind="best",
                                    config_hash=run_document["config_hash"]["full"],
                                    resume_provenance_hash=managed_resume_provenance_hash,
                                    completed_updates=epoch,
                                    model=rnn,
                                    best_validation_loss=float(best_loss),
                                    best_update=best_update,
                                ),
                            )
                        else:
                            _save_best_checkpoint(rnn, savename, task=task)

                if managed_run:
                    elapsed_minutes = elapsed_offset_minutes + (time.time() - t0) / 60
                    validation_history.append(
                        {
                            "update": int(epoch),
                            "loss": float(loss),
                            "accuracy": float(acc),
                            "best_loss": float(best_loss),
                            "elapsed_minutes": float(elapsed_minutes),
                        }
                    )
                    managed_validation_metrics.append(dict(metrics))
                    run_manager.save_validation_metrics(
                        run_dir, managed_validation_metrics
                    )
                
                losses = [np.round(l.item(), 4) for l in [rnn.acc_loss, rnn.ent_loss, rnn.weight_loss, rnn.rate_loss]]
                print(epoch, loss, acc, np.round((time.time() - t0)/60, 2), best_loss, losses)
                sys.stdout.flush()
                
                if kwargs["save_results"]:
                    if not managed_run:
                        pickle.dump({"epoch": epoch, "loss": all_losses, "accs": all_accs, "eval_metrics": eval_metrics, "evaluation_mode": evaluation_mode if task == "abcd_fmri" else None, "fmri_factorial_metrics": fmri_factorial_metrics, "rnn": rnn, "best_loss": best_loss, "kwargs": kwargs, "optim": optim}, open(f"{savename}.p", "wb"))
                
        optim.zero_grad() # reset gradient accumulator
        loss = rnn.forward() # compute loss
        loss.backward() # compute gradients
        optim.step() # update parameters

        completed_updates = epoch + 1
        if managed_run and (
            epoch % kwargs["eval_freq"] == 0
            or completed_updates == kwargs["num_epochs"]
        ):
            run_manager.save_checkpoint(
                run_dir,
                run_manager.checkpoint_payload(
                    kind="latest",
                    config_hash=run_document["config_hash"]["full"],
                    resume_provenance_hash=managed_resume_provenance_hash,
                    completed_updates=completed_updates,
                    model=rnn,
                    optimizer=optim,
                    best_validation_loss=float(best_loss),
                    best_update=best_update,
                    validation_history=validation_history,
                    env=env,
                    eval_env=eval_env,
                ),
            )
            if validation_history:
                run_manager.save_validation_curve(
                    run_dir,
                    validation_history,
                    best_update=best_update,
                    latest_checkpoint_update=completed_updates,
                )

    if fmri_factorial_schedule is not None:
        _, _, fmri_factorial_metrics = evaluate_abcd_fmri_factorial(
            rnn,
            fmri_factorial_schedule,
            evaluation_seed=int(fmri_factorial_schedule[0].seed),
        )
        print(
            "Final fMRI factorial evaluation:",
            fmri_factorial_metrics["loss"],
            fmri_factorial_metrics["accuracy"],
            fmri_factorial_metrics["num_blocks"],
        )
        if managed_run:
            run_manager.save_final_evaluation(run_dir, fmri_factorial_metrics)

    if kwargs["save_results"]:
        if not managed_run:
            pickle.dump({"epoch": epoch, "loss": all_losses, "accs": all_accs, "eval_metrics": eval_metrics, "evaluation_mode": evaluation_mode if task == "abcd_fmri" else None, "fmri_factorial_metrics": fmri_factorial_metrics, "rnn": rnn, "best_loss": best_loss, "kwargs": kwargs, "optim": optim}, open(f"{savename}.p", "wb"))
            torch.save(rnn, f"{savename}_final.pt")

    return rnn

if __name__ == "__main__": 
    kwargs = pysta.argparser.parse_args()
    main_train(kwargs)
