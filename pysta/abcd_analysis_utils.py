"""Store export and realised-trajectory helpers for the ABCD task.

The generic RNN agent stores one mapping per environment timestep.  Each
batched value has shape ``(batch, ...)`` and ABCD trials can finish at
different times.  This module stacks those records without imposing Jensen's
signed-time or fixed-horizon conventions.

Stacked arrays are **time-major**: ``(time, batch, ...)``, matching the order
of ``agent.store``.  ``valid_timestep[time, batch]`` identifies real task
timesteps; values in rows after a trial has finished are padding and must not
be analysed.  No cortical-gradient or Csub quantity is calculated here.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .abcd_env import manhattan_distance


def _batch_aligned_value(value: Any, batch_size: int, field: str, timestep: int):
    """Copy one stored value to CPU and ensure its leading axis is the batch."""
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.ndim == 0:
            value = value.expand(batch_size).clone()
        elif value.shape[0] != batch_size:
            raise ValueError(
                f"Field '{field}' at timestep {timestep} has leading dimension "
                f"{value.shape[0]}, expected batch size {batch_size}."
            )
        else:
            value = value.clone()
        return value

    if isinstance(value, Mapping):
        raise TypeError(
            f"Nested mapping in field '{field}' at timestep {timestep} cannot be "
            "stacked as a per-trial array. Store its members as separate fields."
        )

    try:
        value = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"Field '{field}' at timestep {timestep} is not array-like."
        ) from error

    if value.ndim == 0:
        value = np.repeat(value[None], batch_size, axis=0)
    elif value.shape[0] != batch_size:
        raise ValueError(
            f"Field '{field}' at timestep {timestep} has leading dimension "
            f"{value.shape[0]}, expected batch size {batch_size}."
        )
    return value.copy()


def _infer_batch_size(store: Sequence[Mapping[str, Any]]) -> int:
    """Infer and validate batch size from the required padding mask."""
    if "valid_timestep" not in store[0]:
        raise KeyError(
            "ABCD store records must contain 'valid_timestep' from "
            "env.trajectory_metadata()."
        )
    valid = store[0]["valid_timestep"]
    if torch.is_tensor(valid):
        if valid.ndim != 1:
            raise ValueError("'valid_timestep' must have shape (batch,) per record.")
        return int(valid.shape[0])
    valid = np.asarray(valid)
    if valid.ndim != 1:
        raise ValueError("'valid_timestep' must have shape (batch,) per record.")
    return int(valid.shape[0])


def stack_store_records(
    store: Sequence[Mapping[str, Any]], *, as_numpy: bool = True
) -> dict[str, Any]:
    """Stack every field in an ABCD ``agent.store`` on CPU.

    Parameters
    ----------
    store
        Non-empty sequence of records produced by ``BaseAgent.forward(store=True)``.
        Every field must occur at every timestep and be scalar or batch-first.
    as_numpy
        If true (default), return NumPy arrays for every field.  If false,
        tensor-valued fields remain CPU tensors; string/object metadata remains
        represented by NumPy arrays.

    Returns
    -------
    dict
        All original store keys, without renaming or dropping metadata.  Each
        value has shape ``(time, batch, ...)``.  In particular this preserves
        ``rs``, ``zs``, ``xs``, ``action`` (the model sample), ``env_action``,
        ``optimal_actions``, ``pi``, and every pre/post-step ABCD metadata field.

    Notes
    -----
    Trials of unequal duration are naturally padded to the longest trial by the
    batched environment.  Always select entries with ``valid_timestep``; padding
    values are deliberately left unchanged so the original record is losslessly
    represented and integer/string dtypes do not need artificial NaN coercion.
    """
    if isinstance(store, (str, bytes)) or not isinstance(store, Sequence):
        raise TypeError("store must be a sequence of per-timestep mappings.")
    if len(store) == 0:
        raise ValueError("Cannot stack an empty agent.store.")
    if not all(isinstance(record, Mapping) for record in store):
        raise TypeError("Every agent.store record must be a mapping.")

    field_order = tuple(store[0].keys())
    expected_fields = set(field_order)
    if not expected_fields:
        raise ValueError("Agent store records cannot be empty mappings.")
    for timestep, record in enumerate(store[1:], start=1):
        record_fields = set(record)
        if record_fields != expected_fields:
            missing = sorted(expected_fields - record_fields)
            extra = sorted(record_fields - expected_fields)
            raise ValueError(
                f"Store fields differ at timestep {timestep}; "
                f"missing={missing}, extra={extra}."
            )

    batch_size = _infer_batch_size(store)
    if batch_size < 1:
        raise ValueError("Stored batch size must be positive.")

    stacked: dict[str, Any] = {}
    for field in field_order:
        values = [
            _batch_aligned_value(record[field], batch_size, field, timestep)
            for timestep, record in enumerate(store)
        ]
        if all(torch.is_tensor(value) for value in values):
            try:
                field_values = torch.stack(values, dim=0)
            except RuntimeError as error:
                shapes = [tuple(value.shape) for value in values]
                raise ValueError(
                    f"Field '{field}' has inconsistent per-timestep shapes: {shapes}."
                ) from error
            stacked[field] = (
                field_values.numpy().copy() if as_numpy else field_values
            )
        else:
            numpy_values = [
                value.numpy() if torch.is_tensor(value) else value for value in values
            ]
            try:
                field_values = np.stack(numpy_values, axis=0)
            except ValueError as error:
                shapes = [tuple(np.asarray(value).shape) for value in numpy_values]
                raise ValueError(
                    f"Field '{field}' has inconsistent per-timestep shapes: {shapes}."
                ) from error
            stacked[field] = field_values.copy()

    valid_timestep = stacked["valid_timestep"]
    if torch.is_tensor(valid_timestep):
        if valid_timestep.dtype != torch.bool:
            raise TypeError("'valid_timestep' must be boolean.")
        valid_numpy = valid_timestep.numpy()
    else:
        if valid_timestep.dtype != np.bool_:
            raise TypeError("'valid_timestep' must be boolean.")
        valid_numpy = valid_timestep
    if valid_numpy.shape != (len(store), batch_size):
        raise ValueError(
            "Stacked 'valid_timestep' must have shape "
            f"({len(store)}, {batch_size}), got {valid_numpy.shape}."
        )
    if len(store) > 1 and np.any(valid_numpy[1:] & ~valid_numpy[:-1]):
        raise ValueError(
            "A trial becomes valid again after padding began; ABCD store masks "
            "must be contiguous from the beginning of each block."
        )
    return stacked


def export_agent_store(agent: Any, *, as_numpy: bool = True) -> dict[str, Any]:
    """Export the current ``agent.store`` using :func:`stack_store_records`.

    The function does not run the agent or mutate its hidden/environment state.
    Call ``agent.forward(store=True)`` first.
    """
    if not hasattr(agent, "store"):
        raise TypeError("agent must expose a .store sequence.")
    return stack_store_records(agent.store, as_numpy=as_numpy)


def save_agent_store(
    path,
    agent_or_store: Any,
    *,
    compressed: bool = True,
    overwrite: bool = False,
) -> Path:
    """Write a complete stacked ABCD store to a named-field NPZ archive.

    Parameters
    ----------
    path
        Destination filename. ``.npz`` is appended if absent.
    agent_or_store
        An agent with a populated ``.store``, a raw store sequence, or an
        already stacked mapping returned by :func:`stack_store_records`.
    compressed
        Use ``numpy.savez_compressed`` when true (default), otherwise use
        uncompressed ``numpy.savez``.
    overwrite
        Existing archives are protected by default. Set true explicitly to
        replace one.

    Returns
    -------
    pathlib.Path
        Exact archive path written. Loading it with ``numpy.load`` exposes one
        named array per original store field, including all ABCD metadata and
        ``valid_timestep`` padding masks.
    """
    destination = Path(path).expanduser()
    if destination.suffix.lower() != ".npz":
        destination = Path(f"{destination}.npz")
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing ABCD store archive: {destination}"
        )
    if destination.exists() and not destination.is_file():
        raise ValueError(f"ABCD store destination is not a file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(agent_or_store, "store") and not isinstance(agent_or_store, Mapping):
        stacked = stack_store_records(agent_or_store.store, as_numpy=True)
    elif isinstance(agent_or_store, Mapping):
        stacked = _as_numpy_stacked(agent_or_store)
        if "valid_timestep" not in stacked:
            raise KeyError("Stacked ABCD store is missing 'valid_timestep'.")
    else:
        stacked = stack_store_records(agent_or_store, as_numpy=True)

    non_string_fields = [field for field in stacked if not isinstance(field, str)]
    if non_string_fields:
        raise TypeError(
            f"NPZ field names must be strings, got {non_string_fields!r}."
        )
    save_function = np.savez_compressed if compressed else np.savez
    save_function(destination, **stacked)
    return destination


def _as_numpy_stacked(store_or_stacked: Any) -> dict[str, np.ndarray]:
    """Normalize a raw store, agent, or stacked mapping to NumPy arrays."""
    if hasattr(store_or_stacked, "store") and not isinstance(
        store_or_stacked, Mapping
    ):
        return stack_store_records(store_or_stacked.store, as_numpy=True)
    if isinstance(store_or_stacked, Mapping):
        normalized = {}
        for field, value in store_or_stacked.items():
            if torch.is_tensor(value):
                normalized[field] = value.detach().cpu().numpy().copy()
            else:
                normalized[field] = np.asarray(value).copy()
        return normalized
    return stack_store_records(store_or_stacked, as_numpy=True)


def _normalize_future_lags(future_lags) -> tuple[int, ...]:
    """Return unique non-negative navigation-action lags in requested order."""
    if isinstance(future_lags, (int, np.integer)) and not isinstance(
        future_lags, (bool, np.bool_)
    ):
        future_lags = (int(future_lags),)
    else:
        try:
            future_lags = tuple(future_lags)
        except TypeError as error:
            raise TypeError("future_lags must be an integer or iterable of integers.") from error

    normalized = []
    for lag in future_lags:
        if isinstance(lag, (bool, np.bool_)) or not isinstance(
            lag, (int, np.integer)
        ):
            raise TypeError(f"Future lag must be an integer, got {lag!r}.")
        lag = int(lag)
        if lag < 0:
            raise ValueError(f"Future lags must be non-negative, got {lag}.")
        if lag not in normalized:
            normalized.append(lag)
    return tuple(normalized)


def _required_time_batch_field(
    stacked: Mapping[str, np.ndarray], field: str, time_steps: int, batch_size: int
) -> np.ndarray:
    """Fetch a required time-major field and validate its first two axes."""
    if field not in stacked:
        raise KeyError(f"Stacked ABCD store is missing required field '{field}'.")
    value = np.asarray(stacked[field])
    if value.ndim < 2 or value.shape[:2] != (time_steps, batch_size):
        raise ValueError(
            f"Field '{field}' must start with shape ({time_steps}, {batch_size}), "
            f"got {value.shape}."
        )
    return value


def extract_navigation_trajectories(
    store_or_stacked: Any,
    *,
    future_lags=(1,),
    invalid_location: int = -1,
) -> list[dict[str, Any]]:
    """Extract realised, per-trial ABCD navigation transitions and labels.

    Parameters
    ----------
    store_or_stacked
        An ``agent``, its raw ``agent.store``, or the output of
        :func:`stack_store_records`.
    future_lags
        Non-negative integer lags measured in **realised NAVIGATION actions**.
        Lag zero labels the pre-action location, lag one labels the location
        reached by the current action, and lag ``k`` skips exactly ``k``
        navigation transitions. Instruction and reward-dwell timesteps never
        increment this lag.
    invalid_location
        Integer sentinel placed in the tail when a requested future transition
        does not exist. Use the accompanying boolean validity masks rather than
        treating the sentinel as a location class.

    Returns
    -------
    list of dict
        One variable-length record per batch trial. Every original time-major
        field is sliced down to that trial's NAVIGATION-action timesteps. The
        following normalized keys are always included:

        ``pre_location``, ``post_location``
            Realised transition endpoints, each shape ``(navigation_steps,)``.
        ``action`` / ``model_action``, ``env_action``
            Unconstrained model samples and actions actually sent to the task.
        ``navigation_step_index`` / ``navigation_index``
            Zero-based action count within the complete ABCD block.
        ``locations``
            Contiguous realised path of length ``navigation_steps + 1``.
        ``future_location_by_lag`` and ``future_location_valid_by_lag``
            Dictionaries keyed by requested lag, each containing arrays of
            shape ``(navigation_steps,)`` with explicit invalid-tail masks.

    Notes
    -----
    Labels come only from ``current_location``, ``post_action_location``, and
    ``navigation_step_taken`` in the stored behavioral transitions. They do not
    use optimal paths, trial phase duration, cortical positions, or Csubs.
    """
    future_lags = _normalize_future_lags(future_lags)
    if isinstance(invalid_location, (bool, np.bool_)) or not isinstance(
        invalid_location, (int, np.integer)
    ):
        raise TypeError("invalid_location must be an integer sentinel.")
    invalid_location = int(invalid_location)

    stacked = _as_numpy_stacked(store_or_stacked)
    if "valid_timestep" not in stacked:
        raise KeyError("Stacked ABCD store is missing 'valid_timestep'.")
    valid_timestep = np.asarray(stacked["valid_timestep"])
    if valid_timestep.ndim != 2 or valid_timestep.dtype != np.bool_:
        raise ValueError("'valid_timestep' must be a boolean (time, batch) array.")
    time_steps, batch_size = valid_timestep.shape

    navigation_step_taken = _required_time_batch_field(
        stacked, "navigation_step_taken", time_steps, batch_size
    )
    if navigation_step_taken.dtype != np.bool_:
        raise TypeError("'navigation_step_taken' must be boolean.")
    if np.any(navigation_step_taken & ~valid_timestep):
        raise ValueError("A padded timestep is marked as a NAVIGATION action step.")

    required_fields = {
        field: _required_time_batch_field(stacked, field, time_steps, batch_size)
        for field in (
            "current_location",
            "post_action_location",
            "action",
            "env_action",
            "navigation_step_index",
        )
    }
    if "loss_mask" in stacked:
        loss_mask = _required_time_batch_field(
            stacked, "loss_mask", time_steps, batch_size
        )
        if loss_mask.dtype != np.bool_:
            raise TypeError("'loss_mask' must be boolean.")
        if not np.array_equal(loss_mask & valid_timestep, navigation_step_taken):
            raise ValueError(
                "'loss_mask' and 'navigation_step_taken' disagree on active "
                "ABCD policy timesteps."
            )

    trajectories = []
    for trial_index in range(batch_size):
        nav_times = np.flatnonzero(
            valid_timestep[:, trial_index]
            & navigation_step_taken[:, trial_index]
        )

        trajectory: dict[str, Any] = {"trial_index": trial_index}
        for field, value in stacked.items():
            value = np.asarray(value)
            if value.ndim >= 2 and value.shape[:2] == (time_steps, batch_size):
                trajectory[field] = value[nav_times, trial_index, ...].copy()

        pre_location = np.asarray(
            required_fields["current_location"][nav_times, trial_index]
        ).reshape(-1)
        post_location = np.asarray(
            required_fields["post_action_location"][nav_times, trial_index]
        ).reshape(-1)
        model_action = np.asarray(
            required_fields["action"][nav_times, trial_index]
        ).reshape(-1)
        env_action = np.asarray(
            required_fields["env_action"][nav_times, trial_index]
        ).reshape(-1)
        navigation_index = np.asarray(
            required_fields["navigation_step_index"][nav_times, trial_index]
        ).reshape(-1)

        for field, value in (
            ("current_location", pre_location),
            ("post_action_location", post_location),
            ("action", model_action),
            ("env_action", env_action),
            ("navigation_step_index", navigation_index),
        ):
            if not np.issubdtype(value.dtype, np.integer):
                raise TypeError(f"Navigation field '{field}' must contain integers.")

        num_navigation_steps = len(nav_times)
        expected_navigation_index = np.arange(
            num_navigation_steps, dtype=navigation_index.dtype
        )
        if not np.array_equal(navigation_index, expected_navigation_index):
            raise ValueError(
                f"Trial {trial_index} navigation_step_index is not contiguous "
                "from zero; instruction/reward timesteps may have been included."
            )
        if num_navigation_steps > 1 and not np.array_equal(
            pre_location[1:], post_location[:-1]
        ):
            raise ValueError(
                f"Trial {trial_index} has discontinuous realised NAVIGATION "
                "transition endpoints."
            )

        if num_navigation_steps:
            locations = np.concatenate(
                [pre_location[:1], post_location], axis=0
            ).astype(np.int64, copy=False)
        else:
            valid_times = np.flatnonzero(valid_timestep[:, trial_index])
            if len(valid_times):
                initial_location = required_fields["current_location"][
                    valid_times[0], trial_index
                ]
                if not np.issubdtype(np.asarray(initial_location).dtype, np.integer):
                    raise TypeError("Navigation field 'current_location' must be integer.")
                locations = np.asarray([initial_location], dtype=np.int64)
            else:
                locations = np.empty(0, dtype=np.int64)

        future_location_by_lag = {}
        future_location_valid_by_lag = {}
        for lag in future_lags:
            labels = np.full(
                num_navigation_steps, invalid_location, dtype=np.int64
            )
            valid_labels = (
                np.arange(num_navigation_steps, dtype=np.int64) + lag
                < len(locations)
            )
            if np.any(valid_labels):
                source_indices = (
                    np.arange(num_navigation_steps, dtype=np.int64)[valid_labels]
                    + lag
                )
                labels[valid_labels] = locations[source_indices]
            future_location_by_lag[lag] = labels
            future_location_valid_by_lag[lag] = valid_labels

        trajectory.update(
            {
                "store_timestep_index": nav_times,
                "num_navigation_steps": num_navigation_steps,
                "pre_location": pre_location.copy(),
                "post_location": post_location.copy(),
                "action": model_action.copy(),
                "model_action": model_action.copy(),
                "env_action": env_action.copy(),
                "navigation_step_index": navigation_index.copy(),
                "navigation_index": navigation_index.copy(),
                "locations": locations,
                "future_location_by_lag": future_location_by_lag,
                "future_location_valid_by_lag": future_location_valid_by_lag,
            }
        )
        trajectories.append(trajectory)

    return trajectories


def _one_dimensional_navigation_field(
    trajectory: Mapping[str, Any], field: str, *, boolean: bool = False
) -> np.ndarray:
    """Validate and return one scalar-valued field per navigation action."""
    if field not in trajectory:
        raise KeyError(
            "ABCD route-consistency analysis requires stored field "
            f"'{field}'."
        )
    value = np.asarray(trajectory[field]).reshape(-1)
    expected = int(trajectory["num_navigation_steps"])
    if len(value) != expected:
        raise ValueError(
            f"Navigation field '{field}' has {len(value)} values, expected "
            f"{expected}."
        )
    if boolean:
        if value.dtype != np.bool_:
            raise TypeError(f"Navigation field '{field}' must be boolean.")
    elif not np.issubdtype(value.dtype, np.integer):
        raise TypeError(f"Navigation field '{field}' must contain integers.")
    return value


def _route_record(
    trajectory: Mapping[str, Any], indices: list[int], *, completed: bool
) -> dict[str, Any]:
    """Build one JSON-safe realised leg record from navigation-row indices."""
    first = indices[0]
    last = indices[-1]
    success_count = int(trajectory["successful_goal_count"][first])
    target_abstract = int(
        trajectory["current_required_abstract_goal_index"][first]
    )
    target_physical = int(
        trajectory["current_required_physical_location"][first]
    )
    pre_locations = np.asarray(trajectory["pre_location"])[indices]
    post_locations = np.asarray(trajectory["post_location"])[indices]
    actions = np.asarray(trajectory["env_action"])[indices]

    for index in indices:
        if int(trajectory["successful_goal_count"][index]) != success_count:
            raise ValueError("successful_goal_count changed within an ABCD leg.")
        if (
            int(trajectory["current_required_abstract_goal_index"][index])
            != target_abstract
            or int(trajectory["current_required_physical_location"][index])
            != target_physical
        ):
            raise ValueError("The required ABCD target changed within a leg.")

    if "action_valid" in trajectory:
        action_valid = np.asarray(trajectory["action_valid"])[indices]
        if action_valid.dtype != np.bool_:
            raise TypeError("Navigation field 'action_valid' must be boolean.")
        invalid_boundary_count = int(np.count_nonzero(~action_valid))
    else:
        # ABCD has no stay action, so a realised no-movement transition is
        # necessarily an invalid boundary attempt.
        invalid_boundary_count = int(
            np.count_nonzero(pre_locations == post_locations)
        )

    locations = np.concatenate([pre_locations[:1], post_locations])
    minimum_actions = manhattan_distance(int(locations[0]), target_physical)
    action_count = len(indices)
    if completed:
        if int(locations[-1]) != target_physical:
            raise ValueError(
                "A leg marked target_reached does not end at its stored target."
            )
        excess_actions: int | None = action_count - minimum_actions
        if excess_actions < 0:
            raise ValueError(
                "A completed ABCD route is shorter than Manhattan distance; "
                "stored transitions are inconsistent."
            )
        shortest: bool | None = excess_actions == 0
    else:
        excess_actions = None
        shortest = None

    # successful_goal_count is the number of rewards before this movement.
    # Count zero is the block-start -> first-goal approach.  Later slot-zero
    # legs begin at the preceding fourth goal and are directly comparable.
    initial_approach = success_count == 0
    return {
        "successful_goal_count_before": success_count,
        "loop_index": success_count // 4,
        "execution_slot": success_count % 4,
        "target_abstract_goal_index": target_abstract,
        "target_physical_location": target_physical,
        "completed": bool(completed),
        "initial_approach": initial_approach,
        "included_in_repetition_consistency": bool(completed and not initial_approach),
        "navigation_start_index": int(
            np.asarray(trajectory["navigation_step_index"])[first]
        ),
        "navigation_end_index": int(
            np.asarray(trajectory["navigation_step_index"])[last]
        ),
        "start_location": int(locations[0]),
        "end_location": int(locations[-1]),
        "actions": [int(action) for action in actions],
        "locations": [int(location) for location in locations],
        "action_count": action_count,
        "minimum_action_count": minimum_actions,
        "shortest": shortest,
        "excess_actions": excess_actions,
        "invalid_boundary_action_count": invalid_boundary_count,
    }


def _segment_realised_legs(trajectory: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Segment navigation actions by pre-transition success count and reward."""
    num_steps = int(trajectory["num_navigation_steps"])
    if num_steps == 0:
        return []

    success_count = _one_dimensional_navigation_field(
        trajectory, "successful_goal_count"
    )
    target_reached = _one_dimensional_navigation_field(
        trajectory, "target_reached", boolean=True
    )
    _one_dimensional_navigation_field(
        trajectory, "current_required_abstract_goal_index"
    )
    _one_dimensional_navigation_field(
        trajectory, "current_required_physical_location"
    )
    if np.any(success_count < 0):
        raise ValueError("successful_goal_count cannot be negative.")

    routes: list[dict[str, Any]] = []
    current_indices: list[int] = []
    current_success_count: int | None = None
    for index in range(num_steps):
        row_success_count = int(success_count[index])
        if current_indices and row_success_count != current_success_count:
            # Preserve malformed/truncated material rather than silently
            # joining movements directed at different targets.
            routes.append(_route_record(trajectory, current_indices, completed=False))
            current_indices = []
        if not current_indices:
            current_success_count = row_success_count
        current_indices.append(index)

        if bool(target_reached[index]):
            routes.append(_route_record(trajectory, current_indices, completed=True))
            current_indices = []
            current_success_count = None

    if current_indices:
        routes.append(_route_record(trajectory, current_indices, completed=False))
    return routes


def _pair_counts(signatures: list[tuple[int, ...]]) -> tuple[int, int]:
    """Return exact-match and total unordered pair counts."""
    total = len(signatures) * (len(signatures) - 1) // 2
    matches = sum(count * (count - 1) // 2 for count in Counter(signatures).values())
    return matches, total


def _summary_from_groups(
    route_groups: Sequence[Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    """Aggregate routes while comparing repetition only within each group."""
    routes = [route for group in route_groups for route in group]
    completed = [route for route in routes if route["completed"]]
    incomplete = [route for route in routes if not route["completed"]]

    action_pair_matches = 0
    action_pair_count = 0
    location_pair_matches = 0
    location_pair_count = 0
    action_modal_count = 0
    location_modal_count = 0
    consistency_route_count = 0
    consistency_group_count = 0
    for group in route_groups:
        comparable = [
            route
            for route in group
            if route["included_in_repetition_consistency"]
        ]
        if comparable:
            consistency_group_count += 1
            consistency_route_count += len(comparable)
            action_signatures = [tuple(route["actions"]) for route in comparable]
            location_signatures = [tuple(route["locations"]) for route in comparable]
            action_matches, action_pairs = _pair_counts(action_signatures)
            location_matches, location_pairs = _pair_counts(location_signatures)
            action_pair_matches += action_matches
            action_pair_count += action_pairs
            location_pair_matches += location_matches
            location_pair_count += location_pairs
            action_modal_count += max(Counter(action_signatures).values())
            location_modal_count += max(Counter(location_signatures).values())

    shortest_count = sum(bool(route["shortest"]) for route in completed)
    total_excess = sum(int(route["excess_actions"]) for route in completed)
    action_count = sum(int(route["action_count"]) for route in routes)
    invalid_count = sum(
        int(route["invalid_boundary_action_count"]) for route in routes
    )
    return {
        "route_count": len(routes),
        "completed_route_count": len(completed),
        "incomplete_route_count": len(incomplete),
        "consistency_route_count": consistency_route_count,
        "consistency_group_count": consistency_group_count,
        "exact_action_pair_matches": action_pair_matches,
        "action_pair_count": action_pair_count,
        "exact_action_pairwise_rate": (
            action_pair_matches / action_pair_count if action_pair_count else None
        ),
        "exact_location_pair_matches": location_pair_matches,
        "location_pair_count": location_pair_count,
        "exact_location_pairwise_rate": (
            location_pair_matches / location_pair_count
            if location_pair_count
            else None
        ),
        "action_modal_fraction": (
            action_modal_count / consistency_route_count
            if consistency_route_count
            else None
        ),
        "location_modal_fraction": (
            location_modal_count / consistency_route_count
            if consistency_route_count
            else None
        ),
        "shortest_route_count": shortest_count,
        "shortest_route_fraction": (
            shortest_count / len(completed) if completed else None
        ),
        "total_excess_actions": total_excess,
        "mean_excess_actions": (
            total_excess / len(completed) if completed else None
        ),
        "navigation_action_count": action_count,
        "invalid_boundary_action_count": invalid_count,
        "invalid_boundary_action_fraction": (
            invalid_count / action_count if action_count else None
        ),
        "incomplete_navigation_action_count": sum(
            int(route["action_count"]) for route in incomplete
        ),
    }


def _last_valid_scalar(
    stacked: Mapping[str, np.ndarray],
    field: str,
    valid_timestep: np.ndarray,
    trial_index: int,
    default: Any,
) -> Any:
    """Read a trial's final valid scalar field without touching padded rows."""
    if field not in stacked:
        return default
    value = np.asarray(stacked[field])
    if value.ndim < 2 or value.shape[:2] != valid_timestep.shape:
        raise ValueError(
            f"Field '{field}' must start with shape {valid_timestep.shape}, "
            f"got {value.shape}."
        )
    valid_rows = np.flatnonzero(valid_timestep[:, trial_index])
    if not len(valid_rows):
        return default
    scalar = np.asarray(value[valid_rows[-1], trial_index])
    if scalar.ndim != 0:
        raise ValueError(f"Terminal field '{field}' must be scalar per trial.")
    return scalar.item()


def summarize_route_consistency(store_or_stacked: Any) -> dict[str, Any]:
    """Summarize realised ABCD route repetition, efficiency, and failures.

    Repetition is assessed using the environment action actually executed
    (``env_action``), never the unconstrained model action.  Navigation legs
    are delimited by their pre-transition ``successful_goal_count`` and the
    stored ``target_reached`` transition.  Instruction and reward dwell rows
    are excluded by ``navigation_step_taken``.

    Exact action- and location-route comparisons are made within each trial
    and execution slot.  The initial block-start-to-first-goal approach
    (pre-transition success count zero) is retained for shortest-path and
    boundary-error summaries but excluded from repetition comparisons because
    it has a different origin from later fourth-goal-to-first-goal legs.

    The returned nested mapping contains only JSON-serializable Python values.
    ``None`` denotes a rate without a denominator (for example, only one
    comparable repetition).  Both per-trial/per-slot and pooled summaries
    preserve their underlying counts so downstream aggregation is unambiguous.
    """
    agent_env = getattr(store_or_stacked, "env", None)
    stacked = _as_numpy_stacked(store_or_stacked)
    if "valid_timestep" not in stacked:
        raise KeyError("Stacked ABCD store is missing 'valid_timestep'.")
    valid_timestep = np.asarray(stacked["valid_timestep"])
    if valid_timestep.ndim != 2 or valid_timestep.dtype != np.bool_:
        raise ValueError("'valid_timestep' must be a boolean (time, batch) array.")
    _, batch_size = valid_timestep.shape
    trajectories = extract_navigation_trajectories(stacked, future_lags=())

    env_finished = getattr(agent_env, "finished", None)
    env_truncated = getattr(agent_env, "truncated", None)
    env_success_count = getattr(agent_env, "successful_goal_count", None)
    env_total_required = getattr(agent_env, "total_required_goals", None)
    if torch.is_tensor(env_finished):
        env_finished = env_finished.detach().cpu().numpy()
    if torch.is_tensor(env_truncated):
        env_truncated = env_truncated.detach().cpu().numpy()
    if torch.is_tensor(env_success_count):
        env_success_count = env_success_count.detach().cpu().numpy()

    trial_summaries: list[dict[str, Any]] = []
    routes_by_trial: list[list[dict[str, Any]]] = []
    for trial_index, trajectory in enumerate(trajectories):
        routes = _segment_realised_legs(trajectory)
        routes_by_trial.append(routes)
        slot_groups = {
            str(slot): [
                route for route in routes if route["execution_slot"] == slot
            ]
            for slot in range(4)
        }
        per_slot = {
            slot: _summary_from_groups([slot_routes])
            for slot, slot_routes in slot_groups.items()
        }

        stored_terminated = bool(
            _last_valid_scalar(
                stacked,
                "next_finished",
                valid_timestep,
                trial_index,
                _last_valid_scalar(
                    stacked,
                    "finished",
                    valid_timestep,
                    trial_index,
                    False,
                ),
            )
        )
        stored_truncated = bool(
            _last_valid_scalar(
                stacked,
                "next_truncated",
                valid_timestep,
                trial_index,
                False,
            )
        )
        if "truncated" in stacked:
            truncated = np.asarray(stacked["truncated"])
            if truncated.ndim < 2 or truncated.shape[:2] != valid_timestep.shape:
                raise ValueError(
                    "Field 'truncated' must start with the same (time, batch) "
                    "shape as valid_timestep."
                )
            stored_truncated = stored_truncated or bool(
                np.any(truncated[:, trial_index] & valid_timestep[:, trial_index])
            )
        final_success_count = int(
            _last_valid_scalar(
                stacked,
                "next_successful_goal_count",
                valid_timestep,
                trial_index,
                max(
                    (
                        int(route["successful_goal_count_before"])
                        + int(route["completed"])
                        for route in routes
                    ),
                    default=0,
                ),
            )
        )

        if env_finished is not None:
            terminated = bool(np.asarray(env_finished)[trial_index])
        else:
            terminated = stored_terminated
        if env_truncated is not None:
            truncated_flag = bool(np.asarray(env_truncated)[trial_index])
        else:
            truncated_flag = stored_truncated
            # The environment's pre-step truncated flag is normally false on
            # the final stored transition.  A terminal unfinished leg is the
            # lossless store-level signature of max-step truncation.
            if terminated and any(not route["completed"] for route in routes):
                truncated_flag = True
        if env_success_count is not None:
            final_success_count = int(np.asarray(env_success_count)[trial_index])

        expected_success_count = (
            int(env_total_required) if env_total_required is not None else None
        )
        if truncated_flag:
            status = "truncated"
        elif terminated:
            status = "complete"
        else:
            status = "unterminated"
        trial_summary = {
            "trial_index": trial_index,
            "status": status,
            "terminated": terminated,
            "truncated": truncated_flag,
            "has_incomplete_route": any(not route["completed"] for route in routes),
            "successful_goal_count": final_success_count,
            "expected_successful_goal_count": expected_success_count,
            "routes": routes,
            "by_execution_slot": per_slot,
            "overall": _summary_from_groups(list(slot_groups.values())),
        }
        trial_summaries.append(trial_summary)

    pooled_groups_by_slot = {
        str(slot): [
            [route for route in routes if route["execution_slot"] == slot]
            for routes in routes_by_trial
        ]
        for slot in range(4)
    }
    pooled_by_slot = {
        slot: _summary_from_groups(groups)
        for slot, groups in pooled_groups_by_slot.items()
    }
    pooled_groups = [
        group
        for slot_groups in pooled_groups_by_slot.values()
        for group in slot_groups
    ]
    return {
        "num_trials": batch_size,
        "trials": trial_summaries,
        "by_execution_slot": pooled_by_slot,
        "overall": _summary_from_groups(pooled_groups),
    }


__all__ = [
    "stack_store_records",
    "export_agent_store",
    "save_agent_store",
    "extract_navigation_trajectories",
    "summarize_route_consistency",
]
