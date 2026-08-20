"""Shared, reference-matched utilities for the ABCD proof-of-concept analyses.

The module deliberately keeps collection, normalization, provenance, and
cortical geometry in one small dependency surface.  It does not fit any raw,
Csubs, or RSA model.

Normalized navigation follows the pre-specified discrete RNN analogue of the
human DSR coordinate:

* every completed goal-directed leg is sampled at three deterministic
  midpoint phases;
* ``q = 3 * execution_ordinal + phase`` (not abstract A/B/C/D identity);
* future targets advance through the continuous normalized block sequence;
* the final source loop is withheld, leaving complete common support for all
  twelve horizons without wrapping a target inside a loop.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import pysta


REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIRNAME = "abcd_reference_analysis"
TRIAL_COLLECTION_DIRNAME = "trial_collection"
PHASES_PER_GOAL = 3
NUM_ABSTRACT_GOALS = 4
NUM_NORMALIZED_POSITIONS = NUM_ABSTRACT_GOALS * PHASES_PER_GOAL
NORMALIZED_HORIZONS = tuple(range(NUM_NORMALIZED_POSITIONS))


# ---------------------------------------------------------------------------
# JSON, hashing, and paths
# ---------------------------------------------------------------------------


def json_ready(value: Any) -> Any:
    """Recursively convert NumPy/torch/path objects to JSON-safe values."""
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        value = value.detach().cpu()
        return value.item() if value.ndim == 0 else value.numpy().tolist()
    return value


def write_json(path: Path | str, value: Mapping[str, Any]) -> Path:
    """Write an indented, deterministic JSON document."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf8") as stream:
        json.dump(json_ready(value), stream, indent=2, sort_keys=True)
        stream.write("\n")
    return destination


def load_analysis_manifest(path_or_root: Path | str) -> dict[str, Any]:
    """Load ``analysis_manifest.json`` from a file or analysis directory."""
    path = Path(path_or_root).expanduser().resolve()
    if path.is_dir():
        path = path / "analysis_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing ABCD analysis manifest: {path}")
    with path.open("r", encoding="utf8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema") != "abcd_reference_analysis/v1":
        raise ValueError(f"Unsupported ABCD analysis manifest schema in {path}.")
    return manifest


def file_sha256(path: Path | str, *, chunk_bytes: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for one regular file."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while True:
            block = stream.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray | torch.Tensor) -> str:
    """Hash array content together with its dtype and shape."""
    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous().numpy()
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("utf8"))
    digest.update(repr(tuple(array.shape)).encode("utf8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def state_dict_sha256(model_or_state: Any) -> str:
    """Hash a state dict without relying on torch serialization metadata."""
    state = (
        model_or_state.state_dict()
        if hasattr(model_or_state, "state_dict")
        else model_or_state
    )
    if not isinstance(state, Mapping):
        raise TypeError("model_or_state must expose a state_dict or be a mapping.")
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not torch.is_tensor(value) and not isinstance(value, np.ndarray):
            raise TypeError(f"State entry {name!r} is not an array/tensor.")
        digest.update(str(name).encode("utf8"))
        digest.update(array_sha256(value).encode("ascii"))
    return digest.hexdigest()


def _path_record(path: Path | str, *, hash_file: bool = True) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    record: dict[str, Any] = {"absolute": str(resolved)}
    try:
        record["repo_relative"] = str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        record["repo_relative"] = None
    if resolved.is_file():
        record["size_bytes"] = int(resolved.stat().st_size)
        if hash_file:
            record["sha256"] = file_sha256(resolved)
    return record


def _resolve_path_record(record: Mapping[str, Any] | str | Path) -> Path:
    if isinstance(record, (str, Path)):
        return Path(record).expanduser().resolve()
    relative = record.get("repo_relative")
    if relative is not None:
        candidate = (REPO_ROOT / str(relative)).resolve()
        if candidate.exists():
            return candidate
    absolute = record.get("absolute")
    if absolute is None:
        raise KeyError("Path record has neither repo_relative nor absolute path.")
    return Path(str(absolute)).expanduser().resolve()


def infer_portable_files(checkpoint: Path | str) -> tuple[Path, Path]:
    """Infer the sibling portable state dict and constructor JSON."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    suffix = "_best.pt"
    if not checkpoint.name.endswith(suffix):
        raise ValueError("Checkpoint identifier must end in '_best.pt'.")
    stem = checkpoint.name[: -len(suffix)]
    state_path = checkpoint.with_name(f"{stem}_best_portable_state_dict.pt")
    kwargs_path = checkpoint.with_name(f"{stem}_portable_kwargs.json")
    if not state_path.is_file():
        raise FileNotFoundError(f"Missing portable state_dict: {state_path}")
    if not kwargs_path.is_file():
        raise FileNotFoundError(f"Missing portable kwargs JSON: {kwargs_path}")
    return state_path, kwargs_path


def resolve_analysis_root(
    checkpoint: Path | str,
    output_dir: Path | str | None = None,
) -> Path:
    """Resolve the model-specific ``abcd_reference_analysis`` root."""
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    checkpoint = Path(checkpoint).expanduser().resolve()
    repo_root = Path(pysta.basedir).resolve()
    try:
        model_relative = checkpoint.parent.relative_to(repo_root / "models")
    except ValueError:
        model_relative = Path(checkpoint.parent.name)
    return (
        repo_root
        / "data"
        / "rnn_analyses"
        / model_relative
        / checkpoint.stem
        / ANALYSIS_DIRNAME
    )


def resolve_trial_collection_dir(path_or_root: Path | str) -> Path:
    """Resolve a collection directory from an analysis or collection root."""
    path = Path(path_or_root).expanduser().resolve()
    if path.name == TRIAL_COLLECTION_DIRNAME:
        return path
    return path / TRIAL_COLLECTION_DIRNAME


# ---------------------------------------------------------------------------
# Frozen portable model reconstruction and state preservation
# ---------------------------------------------------------------------------


def capture_global_rng_state() -> dict[str, Any]:
    """Capture NumPy plus available torch generator states."""
    state: dict[str, Any] = {
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state().clone(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [item.clone() for item in torch.cuda.get_rng_state_all()]
    if hasattr(torch, "mps") and hasattr(torch.mps, "get_rng_state"):
        try:
            state["torch_mps"] = torch.mps.get_rng_state().clone()
        except RuntimeError:
            pass
    return state


def restore_global_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a state produced by :func:`capture_global_rng_state`."""
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if "torch_mps" in state:
        torch.mps.set_rng_state(state["torch_mps"])


def global_rng_states_equal(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    """Compare captured RNG states exactly."""
    a, b = first["numpy"], second["numpy"]
    numpy_equal = (
        a[0] == b[0]
        and np.array_equal(a[1], b[1])
        and a[2:] == b[2:]
    )
    if not numpy_equal or not torch.equal(first["torch_cpu"], second["torch_cpu"]):
        return False
    for key in ("torch_cuda", "torch_mps"):
        if (key in first) != (key in second):
            return False
        if key == "torch_cuda" and key in first:
            if len(first[key]) != len(second[key]) or not all(
                torch.equal(x, y) for x, y in zip(first[key], second[key])
            ):
                return False
        elif key in first and not torch.equal(first[key], second[key]):
            return False
    return True


def reconstruct_trained_model(
    state_path: Path | str,
    kwargs_path: Path | str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Reconstruct and exactly load the portable frozen trained model.

    Constructor RNG use is locally contained so importing/constructing the
    analysis model does not perturb the caller's NumPy or torch streams.
    """
    state_path = Path(state_path).expanduser().resolve()
    kwargs_path = Path(kwargs_path).expanduser().resolve()
    with kwargs_path.open("r", encoding="utf8") as stream:
        kwargs = json.load(stream)
    if kwargs.get("task") != "abcd_fmri":
        raise ValueError(f"Expected task='abcd_fmri', got {kwargs.get('task')!r}.")

    rng_state = capture_global_rng_state()
    try:
        seed = int(kwargs["seed"])
        np.random.seed(seed)
        torch.manual_seed(seed)
        environment = pysta.tasks.make_environment(kwargs, split="train")
        model = pysta.train_rnn._make_rnn(environment, kwargs)
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        result = model.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                "Portable state_dict did not load exactly: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}."
            )
        if not all(torch.isfinite(value).all().item() for value in state.values()):
            raise ValueError("Portable trained state contains non-finite values.")
    finally:
        restore_global_rng_state(rng_state)
    return model, kwargs


def derive_repeat_seeds(model_seed: int) -> tuple[int, int]:
    """Return two deterministic, distinct recurrent-noise evaluation seeds."""
    modulus = 2**31 - 1
    first = (int(model_seed) + 104_729) % modulus
    second = (int(model_seed) + 209_759) % modulus
    if first == second:  # defensive for a future modulus change
        raise RuntimeError("Could not derive distinct repeat seeds.")
    return first, second


# ---------------------------------------------------------------------------
# Task-normalized navigation
# ---------------------------------------------------------------------------


def midpoint_phase_indices(n_steps: int, n_phases: int = PHASES_PER_GOAL) -> np.ndarray:
    """Indices for deterministic nearest-neighbour midpoint resampling.

    This is integer-exact ``floor((j + 0.5) * n_steps / n_phases)``.  Short
    paths intentionally repeat a realised row.
    """
    if isinstance(n_steps, (bool, np.bool_)) or int(n_steps) != n_steps:
        raise TypeError("n_steps must be an integer.")
    if isinstance(n_phases, (bool, np.bool_)) or int(n_phases) != n_phases:
        raise TypeError("n_phases must be an integer.")
    n_steps, n_phases = int(n_steps), int(n_phases)
    if n_steps < 1 or n_phases < 1:
        raise ValueError("n_steps and n_phases must be positive.")
    phase = np.arange(n_phases, dtype=np.int64)
    selected = ((2 * phase + 1) * n_steps) // (2 * n_phases)
    return np.minimum(selected, n_steps - 1)


def _activity_matrix(value: Any, n_steps: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2 or array.shape[0] != n_steps:
        raise ValueError(f"{name} must have shape (navigation_steps, units), got {array.shape}.")
    return array.astype(np.float32, copy=False)


def _scalar_navigation_field(
    trajectory: Mapping[str, Any], field: str, n_steps: int
) -> np.ndarray:
    if field not in trajectory:
        raise KeyError(f"Normalized navigation requires trajectory field {field!r}.")
    value = np.asarray(trajectory[field])
    if value.shape[0] != n_steps:
        raise ValueError(f"{field!r} has {value.shape[0]} rows, expected {n_steps}.")
    value = value.reshape(n_steps, -1)
    if value.shape[1] != 1:
        raise ValueError(f"{field!r} must be scalar per navigation row.")
    return value[:, 0]


def _constant_sequence(
    trajectory: Mapping[str, Any],
    block_metadata: Mapping[str, Any],
    n_steps: int,
) -> np.ndarray:
    sequence = block_metadata.get("effective_execution_abstract_sequence")
    if sequence is None and "effective_execution_abstract_sequence" in trajectory:
        values = np.asarray(trajectory["effective_execution_abstract_sequence"])
        if values.shape[0] != n_steps:
            raise ValueError("Execution-sequence metadata is not navigation aligned.")
        values = values.reshape(n_steps, -1)
        if not np.all(values == values[0]):
            raise ValueError("Effective execution sequence changes within a block.")
        sequence = values[0]
    if sequence is None:
        raise KeyError("Missing effective_execution_abstract_sequence metadata.")
    sequence = np.asarray(sequence, dtype=np.int64).reshape(-1)
    if not np.array_equal(np.sort(sequence), np.arange(NUM_ABSTRACT_GOALS)):
        raise ValueError(
            "ABCD analysis requires one execution-order permutation of four goals."
        )
    return sequence


def build_normalized_navigation(
    trajectory: Mapping[str, Any],
    block_metadata: Mapping[str, Any],
    *,
    repeat_index: int,
    expected_loops: int,
) -> dict[str, np.ndarray]:
    """Build flat, common-support normalized observations for one block.

    The returned source rows comprise ``expected_loops - 1`` complete loops.
    Their twelve targets are drawn from the full continuous normalized sequence,
    which still contains the withheld fifth loop.  No target wraps within its
    source loop, and every horizon has exactly the same source observations.
    """
    if int(expected_loops) != expected_loops or int(expected_loops) < 2:
        raise ValueError("expected_loops must be an integer of at least two.")
    expected_loops = int(expected_loops)
    n_steps = int(trajectory.get("num_navigation_steps", len(trajectory["rs"])))
    if n_steps < 1:
        raise ValueError("Cannot normalize an empty navigation trajectory.")

    rs = _activity_matrix(trajectory["rs"], n_steps, "rs")
    zs = _activity_matrix(trajectory["zs"], n_steps, "zs")
    pre_location = _scalar_navigation_field(trajectory, "pre_location", n_steps)
    post_location = _scalar_navigation_field(trajectory, "post_location", n_steps)
    action = _scalar_navigation_field(trajectory, "env_action", n_steps)
    model_action = _scalar_navigation_field(trajectory, "model_action", n_steps)
    success_count = _scalar_navigation_field(
        trajectory, "successful_goal_count", n_steps
    ).astype(np.int64)
    abstract_goal = _scalar_navigation_field(
        trajectory, "current_required_abstract_goal_index", n_steps
    ).astype(np.int64)
    target_reached = _scalar_navigation_field(
        trajectory, "target_reached", n_steps
    ).astype(bool)
    target_physical = _scalar_navigation_field(
        trajectory, "current_required_physical_location", n_steps
    ).astype(np.int64)
    store_index = _scalar_navigation_field(
        trajectory, "store_timestep_index", n_steps
    ).astype(np.int64)
    navigation_index = _scalar_navigation_field(
        trajectory, "navigation_index", n_steps
    ).astype(np.int64)
    execution_sequence = _constant_sequence(trajectory, block_metadata, n_steps)

    expected_counts = np.arange(expected_loops * NUM_ABSTRACT_GOALS)
    if not np.array_equal(np.unique(success_count), expected_counts):
        raise ValueError(
            "A normalized block must contain every completed goal leg; "
            f"observed counts={np.unique(success_count).tolist()}, "
            f"expected={expected_counts.tolist()}."
        )
    if np.any(np.diff(success_count) < 0):
        raise ValueError("successful_goal_count is not monotonic within the block.")

    full_selected: list[int] = []
    full_loop: list[int] = []
    full_q: list[int] = []
    full_phase: list[int] = []
    full_goal: list[int] = []
    for loop_index in range(expected_loops):
        observed_goals: list[int] = []
        for execution_ordinal in range(NUM_ABSTRACT_GOALS):
            count = loop_index * NUM_ABSTRACT_GOALS + execution_ordinal
            leg = np.flatnonzero(success_count == count)
            if len(leg) == 0 or not np.array_equal(
                leg, np.arange(leg[0], leg[-1] + 1)
            ):
                raise ValueError(f"Goal leg {count} is missing or non-contiguous.")
            if not bool(target_reached[leg[-1]]) or np.any(target_reached[leg[:-1]]):
                raise ValueError(f"Goal leg {count} does not end in exactly one reward.")
            goal_values = np.unique(abstract_goal[leg])
            if len(goal_values) != 1:
                raise ValueError(f"Abstract goal changes within leg {count}.")
            goal = int(goal_values[0])
            expected_goal = int(execution_sequence[execution_ordinal])
            if goal != expected_goal:
                raise ValueError(
                    f"Leg {count} targets abstract goal {goal}, expected "
                    f"execution-order goal {expected_goal}."
                )
            if np.any(target_physical[leg] != target_physical[leg[0]]):
                raise ValueError(f"Physical target changes within leg {count}.")
            observed_goals.append(goal)
            for phase, local_index in enumerate(midpoint_phase_indices(len(leg))):
                full_selected.append(int(leg[int(local_index)]))
                full_loop.append(loop_index)
                full_q.append(PHASES_PER_GOAL * execution_ordinal + phase)
                full_phase.append(phase)
                full_goal.append(goal)
        if observed_goals != execution_sequence.tolist():
            raise ValueError(f"Loop {loop_index} does not follow execution order.")

    selected_all = np.asarray(full_selected, dtype=np.int64)
    loop_all = np.asarray(full_loop, dtype=np.int16)
    q_all = np.asarray(full_q, dtype=np.int8)
    phase_all = np.asarray(full_phase, dtype=np.int8)
    goal_all = np.asarray(full_goal, dtype=np.int8)
    n_positions = NUM_NORMALIZED_POSITIONS
    if len(selected_all) != expected_loops * n_positions:
        raise RuntimeError("Normalized full-sequence size is internally inconsistent.")
    expected_q = np.tile(np.arange(n_positions, dtype=np.int8), expected_loops)
    if not np.array_equal(q_all, expected_q):
        raise RuntimeError("Normalized q does not advance 0..11 in execution order.")

    # Retain whole source loops only.  The final complete loop exists solely to
    # provide realized future targets, avoiding both terminal truncation and an
    # unbalanced single extra q=0 source row.
    source_count = (expected_loops - 1) * n_positions
    source_position = np.arange(source_count, dtype=np.int64)
    target_position = source_position[:, None] + np.asarray(
        NORMALIZED_HORIZONS, dtype=np.int64
    )[None, :]
    if int(target_position.max()) >= len(selected_all):
        raise RuntimeError("Common-support target extends beyond the realized block.")
    source_nav = selected_all[source_position]
    target_nav = selected_all[target_position]

    next_action = np.full(source_count, -1, dtype=np.int16)
    has_next = source_nav + 1 < n_steps
    next_action[has_next] = action[source_nav[has_next] + 1].astype(np.int16)

    factorial_index = int(block_metadata["factorial_index"])
    base_index = int(block_metadata["base_configuration_index"])
    instruction_direction = int(block_metadata["instruction_direction"])
    execution_relation = int(block_metadata["execution_relation"])
    configuration = np.asarray(
        block_metadata.get("base_configuration", block_metadata.get("configuration")),
        dtype=np.int16,
    ).reshape(-1)
    if configuration.size != NUM_ABSTRACT_GOALS:
        raise ValueError("Block configuration must map exactly four abstract goals.")

    result = {
        "rs": rs[source_nav].astype(np.float32, copy=False),
        "zs": zs[source_nav].astype(np.float32, copy=False),
        "current_location": pre_location[source_nav].astype(np.int16),
        "post_location": post_location[source_nav].astype(np.int16),
        "future_locations": pre_location[target_nav].astype(np.int16),
        "future_valid": np.ones(target_position.shape, dtype=bool),
        "current_action": action[source_nav].astype(np.int16),
        "next_action": next_action,
        "model_action": model_action[source_nav].astype(np.int16),
        "normalized_position": q_all[source_position],
        "q": q_all[source_position],
        "phase": phase_all[source_position],
        "abstract_goal": goal_all[source_position],
        "loop_index": loop_all[source_position],
        "future_normalized_position": q_all[target_position],
        "future_loop_index": loop_all[target_position],
        "future_abstract_goal": goal_all[target_position],
        "source_index": store_index[source_nav].astype(np.int32),
        "source_navigation_index": navigation_index[source_nav].astype(np.int32),
        "future_source_index": store_index[target_nav].astype(np.int32),
        "target_physical_location": target_physical[source_nav].astype(np.int16),
        "repeat_index": np.full(source_count, int(repeat_index), dtype=np.int8),
        "factorial_index": np.full(source_count, factorial_index, dtype=np.int16),
        "block_id": np.full(source_count, factorial_index, dtype=np.int16),
        "base_configuration_index": np.full(
            source_count, base_index, dtype=np.int16
        ),
        "instruction_direction": np.full(
            source_count, instruction_direction, dtype=np.int8
        ),
        "execution_relation": np.full(
            source_count, execution_relation, dtype=np.int8
        ),
        "configuration": np.broadcast_to(
            configuration, (source_count, NUM_ABSTRACT_GOALS)
        ).copy(),
        "execution_abstract_sequence": np.broadcast_to(
            execution_sequence.astype(np.int8),
            (source_count, NUM_ABSTRACT_GOALS),
        ).copy(),
        "sample_weight": np.full(source_count, 1.0 / source_count, dtype=np.float64),
    }
    return result


def concatenate_normalized_blocks(
    blocks: Sequence[Mapping[str, np.ndarray]],
    *,
    num_locations: int,
) -> dict[str, np.ndarray]:
    """Concatenate normalized blocks and add compact schema constants."""
    blocks = tuple(blocks)
    if not blocks:
        raise ValueError("At least one normalized block is required.")
    fields = set(blocks[0])
    for index, block in enumerate(blocks[1:], start=1):
        if set(block) != fields:
            raise ValueError(f"Normalized block {index} has a different schema.")
    data = {
        field: np.concatenate([np.asarray(block[field]) for block in blocks], axis=0)
        for field in sorted(fields)
    }
    data.update(
        {
            "horizons": np.asarray(NORMALIZED_HORIZONS, dtype=np.int8),
            "num_locations": np.asarray(int(num_locations), dtype=np.int16),
            "num_abstract_goals": np.asarray(NUM_ABSTRACT_GOALS, dtype=np.int8),
            "phases_per_goal": np.asarray(PHASES_PER_GOAL, dtype=np.int8),
            "num_normalized_positions": np.asarray(
                NUM_NORMALIZED_POSITIONS, dtype=np.int8
            ),
        }
    )
    validate_normalized_navigation(data)
    return data


_NORMALIZED_CONSTANT_FIELDS = {
    "horizons",
    "num_locations",
    "num_abstract_goals",
    "phases_per_goal",
    "num_normalized_positions",
}


def validate_normalized_navigation(data: Mapping[str, Any]) -> None:
    """Validate indexing, common support, and per-block weighting."""
    required = {
        "rs",
        "zs",
        "future_locations",
        "future_valid",
        "current_location",
        "current_action",
        "next_action",
        "normalized_position",
        "q",
        "phase",
        "abstract_goal",
        "loop_index",
        "repeat_index",
        "factorial_index",
        "block_id",
        "base_configuration_index",
        "instruction_direction",
        "execution_relation",
        "configuration",
        "source_index",
        "sample_weight",
        "horizons",
        "num_locations",
        "num_normalized_positions",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise KeyError(f"Normalized navigation is missing fields: {missing}")
    rs = np.asarray(data["rs"])
    zs = np.asarray(data["zs"])
    if rs.ndim != 2 or zs.shape != rs.shape:
        raise ValueError("rs and zs must have the same (observations, units) shape.")
    n_rows = rs.shape[0]
    horizons = np.asarray(data["horizons"], dtype=np.int64).reshape(-1)
    n_positions = int(np.asarray(data["num_normalized_positions"]).item())
    if n_positions != NUM_NORMALIZED_POSITIONS or not np.array_equal(
        horizons, np.arange(n_positions)
    ):
        raise ValueError("This reference analysis requires horizons 0..11.")
    future = np.asarray(data["future_locations"])
    valid = np.asarray(data["future_valid"])
    if future.shape != (n_rows, n_positions) or valid.shape != future.shape:
        raise ValueError("future_locations/future_valid have the wrong shape.")
    if valid.dtype != np.bool_ or not np.all(valid):
        raise ValueError("Every normalized horizon must use identical common support.")
    current = np.asarray(data["current_location"]).reshape(-1)
    if len(current) != n_rows or not np.array_equal(future[:, 0], current):
        raise ValueError("Horizon zero must equal current_location for every row.")
    q = np.asarray(data["q"], dtype=np.int64).reshape(-1)
    if not np.array_equal(q, np.asarray(data["normalized_position"]).reshape(-1)):
        raise ValueError("q and normalized_position aliases disagree.")
    future_q = np.asarray(data.get("future_normalized_position"))
    if future_q.shape != future.shape or not np.array_equal(
        future_q, (q[:, None] + horizons[None, :]) % n_positions
    ):
        raise ValueError("Future q values do not advance on the 12-position cycle.")
    block = np.asarray(data["block_id"]).reshape(-1)
    factorial = np.asarray(data["factorial_index"]).reshape(-1)
    repeat = np.asarray(data["repeat_index"]).reshape(-1)
    if not np.array_equal(block, factorial):
        raise ValueError("block_id and factorial_index aliases disagree.")
    weights = np.asarray(data["sample_weight"], dtype=float).reshape(-1)
    if len(weights) != n_rows or np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("sample_weight must be finite and positive per row.")
    for repeat_id, block_id in sorted(set(zip(repeat.tolist(), block.tolist()))):
        mask = (repeat == repeat_id) & (block == block_id)
        block_q = q[mask]
        if len(block_q) % n_positions or not np.array_equal(
            block_q, np.tile(np.arange(n_positions), len(block_q) // n_positions)
        ):
            raise ValueError("Each block must contain complete ordered source loops.")
        if not np.isclose(weights[mask].sum(), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("sample_weight must sum to one within each block.")
    for field, value in data.items():
        if field in _NORMALIZED_CONSTANT_FIELDS:
            continue
        array = np.asarray(value)
        if array.ndim == 0 or array.shape[0] != n_rows:
            raise ValueError(f"Row field {field!r} is not aligned to rs.")


def save_normalized_navigation(path: Path | str, data: Mapping[str, Any]) -> Path:
    """Validate and save one normalized-navigation archive."""
    validate_normalized_navigation(data)
    destination = Path(path).expanduser().resolve()
    if destination.suffix != ".npz":
        destination = destination.with_suffix(".npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **{key: np.asarray(value) for key, value in data.items()})
    return destination


def load_normalized_navigation(path_or_repeat_dir: Path | str) -> dict[str, np.ndarray]:
    """Load and strictly validate one repeat archive."""
    path = Path(path_or_repeat_dir).expanduser().resolve()
    if path.is_dir():
        path = path / "normalized_navigation.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing normalized navigation archive: {path}")
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    validate_normalized_navigation(data)
    return data


def load_normalized_repeats(
    analysis_root: Path | str,
    repeat_indices: Sequence[int] = (1, 2),
) -> dict[str, np.ndarray]:
    """Load and row-concatenate normalized archives from independent repeats."""
    collection = resolve_trial_collection_dir(analysis_root)
    datasets = [
        load_normalized_navigation(collection / f"repeat_{int(index):02d}")
        for index in repeat_indices
    ]
    constants: dict[str, np.ndarray] = {}
    for field in _NORMALIZED_CONSTANT_FIELDS:
        first = np.asarray(datasets[0][field])
        if not all(np.array_equal(first, np.asarray(dataset[field])) for dataset in datasets[1:]):
            raise ValueError(f"Repeat archives disagree on constant {field!r}.")
        constants[field] = first.copy()
    row_fields = set(datasets[0]).difference(_NORMALIZED_CONSTANT_FIELDS)
    if any(set(dataset).difference(_NORMALIZED_CONSTANT_FIELDS) != row_fields for dataset in datasets):
        raise ValueError("Repeat normalized-navigation schemas differ.")
    combined = {
        field: np.concatenate([np.asarray(dataset[field]) for dataset in datasets], axis=0)
        for field in sorted(row_fields)
    }
    combined.update(constants)
    validate_normalized_navigation(combined)
    return combined


# ---------------------------------------------------------------------------
# Dynamic provenance and cortical geometry
# ---------------------------------------------------------------------------


def _find_surface_path(species: str) -> Path:
    root = REPO_ROOT / "data" / "embedding" / "raw_surface_data" / str(species)
    candidates = sorted(root.glob("**/*l.midthickness.surf.gii"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one left midthickness surface under {root}, found {len(candidates)}."
        )
    return candidates[0].resolve()


def build_geometry_manifest(model: Any) -> dict[str, Any]:
    """Describe and hash the exact cortical embedding used by ``model``."""
    if not callable(getattr(model, "_embedding_dir", None)):
        raise TypeError("Reference spatial analyses require a cortical embedding model.")
    embedding_dir = Path(model._embedding_dir()).expanduser().resolve()
    if not embedding_dir.is_dir():
        raise FileNotFoundError(embedding_dir)
    species = str(getattr(model, "embedding_species", "unknown"))
    surface_path = _find_surface_path(species)
    files = {
        path.name: _path_record(path)
        for path in sorted(embedding_dir.iterdir())
        if path.is_file()
    }
    combined = hashlib.sha256()
    for name, record in sorted(files.items()):
        combined.update(name.encode("utf8"))
        combined.update(record["sha256"].encode("ascii"))
    combined.update(file_sha256(surface_path).encode("ascii"))
    distance = np.asarray(getattr(model, "distance_matrix").detach().cpu())
    vertices = np.asarray(getattr(model, "sampled_vertex_indices").detach().cpu())
    combined.update(array_sha256(distance).encode("ascii"))
    combined.update(array_sha256(vertices).encode("ascii"))
    return {
        "embedding_name": str(getattr(model, "embedding_name", embedding_dir.parent.name)),
        "embedding_species": species,
        "embedding_seed": int(getattr(model, "embedding_seed", -1)),
        "embedding_directory": _path_record(embedding_dir, hash_file=False),
        "surface_path": _path_record(surface_path),
        "embedding_files": files,
        "geometry_sha256": combined.hexdigest(),
        "model_distance_matrix_sha256": array_sha256(distance),
        "sampled_vertex_indices_sha256": array_sha256(vertices),
        "coordinate_note": (
            f"surface xyz from {surface_path.name}; surface coordinates, not MNI"
        ),
    }


def build_cortical_mechanism_manifest(model: Any) -> dict[str, Any]:
    """Record the resolved cortical mechanism and its two distinct unit sets.

    ``anatomical_anchor_seed`` is the fixed embedding-supplied reference used
    to compute anchor distance.  ``local_input_recipient_zone`` is the actual
    (possibly expanded) set receiving inputs routed as ``same_end``.  They are
    intentionally separate: changing ``local_fraction`` may change the latter
    without changing the anatomical reference.
    """

    required = (
        "local_fraction",
        "dist_reg",
        "line_decay",
        "line_init_scale",
        "use_local_init",
        "readout_mode",
        "same_end_unit_mask",
    )
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise TypeError(
            "Cortical mechanism provenance is unavailable; model is missing "
            f"{missing}."
        )

    n_units = int(model.Nrec)
    recipient_mask = np.asarray(
        torch.as_tensor(model.same_end_unit_mask).detach().cpu(), dtype=np.bool_
    ).reshape(-1)
    if recipient_mask.shape != (n_units,) or not np.any(recipient_mask):
        raise ValueError(
            "same_end_unit_mask must be a non-empty vector matching Nrec."
        )
    recipient_indices = np.flatnonzero(recipient_mask).astype(np.int64)

    anchor_source = "runtime anatomical_anchor_unit_indices"
    seed = getattr(model, "anatomical_anchor_unit_indices", None)
    anchor_path: Path | None = None
    if callable(getattr(model, "_embedding_dir", None)):
        candidate = Path(model._embedding_dir()).expanduser().resolve() / "anchor_unit_indices.npy"
        if candidate.is_file():
            anchor_path = candidate
    if seed is None and anchor_path is not None:
        seed = np.load(anchor_path)
        anchor_source = "embedding anchor_unit_indices.npy"
    elif seed is None:
        # This preserves provenance for custom cortical embeddings that use
        # the parcel-label fallback, while clearly not calling that fallback
        # an embedding-supplied Area-25-facing seed.
        seed = [int(getattr(model, "anchor_index"))]
        anchor_source = "runtime parcel-label representative fallback"
    seed_indices = np.unique(np.asarray(seed, dtype=np.int64).reshape(-1))
    if (
        len(seed_indices) == 0
        or np.any(seed_indices < 0)
        or np.any(seed_indices >= n_units)
    ):
        raise ValueError("Anatomical anchor seed indices are empty or out of range.")
    if anchor_path is not None:
        file_seed = np.unique(
            np.asarray(np.load(anchor_path), dtype=np.int64).reshape(-1)
        )
        if not np.array_equal(seed_indices, file_seed):
            raise ValueError(
                "Runtime anatomical anchor seed differs from the embedding file."
            )
        anchor_source = "embedding anchor_unit_indices.npy"

    routing_modes = {
        str(name): str(mode)
        for name, mode in dict(getattr(model, "input_routing_modes", {})).items()
    }
    observation_groups = {
        str(name): np.asarray(indices, dtype=np.int64).reshape(-1)
        for name, indices in dict(model.env.obs_inds()).items()
    }
    if set(observation_groups) != set(routing_modes):
        raise ValueError(
            "Runtime observation groups and input-routing modes disagree."
        )
    local_groups = sorted(
        name for name, mode in routing_modes.items() if mode == "same_end"
    )
    mask_buffers = dict(getattr(model, "input_mask_buffers", {}))
    for group in local_groups:
        buffer_name = mask_buffers.get(group)
        if buffer_name is None or not hasattr(model, buffer_name):
            raise ValueError(f"Missing runtime input mask for local group {group!r}.")
        group_mask = np.asarray(
            torch.as_tensor(getattr(model, buffer_name)).detach().cpu()
        )
        group_indices = observation_groups[group]
        if group_mask.shape != (n_units, int(model.Nin)) or not np.all(
            group_mask[:, group_indices] == recipient_mask[:, None]
        ):
            raise ValueError(
                f"Runtime input mask for {group!r} differs from same_end_unit_mask."
            )
    mask_int = recipient_mask.astype(np.uint8)
    seed_hash = array_sha256(seed_indices)
    return {
        "resolved_parameters": {
            "local_fraction": float(model.local_fraction),
            "dist_reg": float(model.dist_reg),
            "line_decay": float(model.line_decay),
            "use_local_init": bool(model.use_local_init),
            "line_init_scale": float(model.line_init_scale),
            "readout_mode": str(model.readout_mode),
        },
        "input_routing_modes": routing_modes,
        "observation_group_input_indices": observation_groups,
        "local_input_recipient_zone": {
            "definition": (
                "actual recurrent units receiving every observation group routed "
                "as same_end/local"
            ),
            "observation_groups": local_groups,
            "unit_mask": mask_int,
            "unit_mask_sha256": array_sha256(mask_int),
            "unit_indices": recipient_indices,
            "unit_indices_sha256": array_sha256(recipient_indices),
            "unit_count": int(len(recipient_indices)),
        },
        "anatomical_anchor_seed": {
            "definition": (
                "fixed Area-25-facing embedding seed; not the expanded input-"
                "recipient zone"
                if anchor_source == "embedding anchor_unit_indices.npy"
                else "parcel-label representative fallback; not an embedding-supplied seed"
            ),
            "source": anchor_source,
            "source_file": (
                _path_record(anchor_path) if anchor_path is not None else None
            ),
            "unit_indices": seed_indices,
            "unit_indices_sha256": seed_hash,
            "unit_count": int(len(seed_indices)),
        },
        "anchor_distance_reference": {
            "definition": (
                "mean unscaled surface-geodesic distance to the fixed anatomical "
                "anchor seed"
            ),
            "unit_indices": seed_indices,
            "unit_indices_sha256": seed_hash,
            "unit_count": int(len(seed_indices)),
            "uses_expanded_input_recipient_zone": False,
        },
    }


def build_analysis_manifest(
    *,
    checkpoint: Path | str,
    portable_state: Path | str,
    portable_kwargs: Path | str,
    model: Any,
    kwargs: Mapping[str, Any],
    schedule: Sequence[Any],
    repeat_seeds: Sequence[int],
    analysis_root: Path | str,
    geometry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact manifest using runtime model/task metadata."""
    schedule = tuple(schedule)
    if not schedule:
        raise ValueError("The factorial evaluation schedule is empty.")
    configurations = [tuple(int(value) for value in cell.configuration) for cell in schedule]
    if any(len(configuration) != NUM_ABSTRACT_GOALS for configuration in configurations):
        raise ValueError("ABCD reference analysis requires four abstract goals.")
    unique_base_indices = sorted({int(cell.base_configuration_index) for cell in schedule})
    base_configurations = []
    for base_index in unique_base_indices:
        observed = {
            tuple(int(value) for value in cell.configuration)
            for cell in schedule
            if int(cell.base_configuration_index) == base_index
        }
        if len(observed) != 1:
            raise ValueError(f"Base {base_index} changes configuration across conditions.")
        base_configurations.append(next(iter(observed)))
    loop_counts = {int(cell.environment.num_loops) for cell in schedule}
    if len(loop_counts) != 1:
        raise ValueError("Factorial schedule cells disagree on num_loops.")
    instruction_repeats = {int(cell.environment.instruction_repeats) for cell in schedule}
    if len(instruction_repeats) != 1:
        raise ValueError("Factorial schedule cells disagree on instruction_repeats.")
    repeat_seeds = tuple(int(seed) for seed in repeat_seeds)
    if len(repeat_seeds) != 2 or len(set(repeat_seeds)) != 2:
        raise ValueError("Exactly two distinct frozen-model repeat seeds are required.")
    geometry = dict(build_geometry_manifest(model) if geometry is None else geometry)
    cortical_mechanism = build_cortical_mechanism_manifest(model)
    num_locations = int(getattr(model.env, "num_locs"))
    manifest = {
        "schema": "abcd_reference_analysis/v1",
        "analysis_label": "ABCD reference-matched proof-of-concept mechanistic analysis",
        "model_description": (
            f"best-performing training-seed-{int(kwargs['seed'])} checkpoint; "
            "training completion is not implied"
        ),
        "analysis_root": _path_record(analysis_root, hash_file=False),
        "source": {
            "checkpoint_identifier": _path_record(checkpoint),
            "portable_state_dict": _path_record(portable_state),
            "portable_kwargs": _path_record(portable_kwargs),
            "loaded_state_dict_sha256": state_dict_sha256(model),
        },
        "model": {
            "class": model.__class__.__name__,
            "model_type": kwargs.get("model_type"),
            "seed": int(kwargs["seed"]),
            "n_recurrent_units": int(model.Nrec),
            "n_inputs": int(model.Nin),
            "n_outputs": int(model.Nout),
            "trained_recurrent_noise": float(model.rec_noise),
        },
        "task": {
            "task": kwargs.get("task"),
            "num_locations": num_locations,
            "num_abstract_goals": NUM_ABSTRACT_GOALS,
            "phases_per_goal": PHASES_PER_GOAL,
            "num_normalized_positions": NUM_NORMALIZED_POSITIONS,
            "normalized_horizons": list(NORMALIZED_HORIZONS),
            "num_loops_per_full_block": next(iter(loop_counts)),
            "num_common_support_source_loops": next(iter(loop_counts)) - 1,
            "instruction_repeats": next(iter(instruction_repeats)),
            "num_factorial_blocks": len(schedule),
            "num_base_configurations": len(unique_base_indices),
            "base_configuration_indices": unique_base_indices,
            "base_configurations": base_configurations,
        },
        "collection": {
            "repeat_seeds": list(repeat_seeds),
            "num_independent_repeats": 2,
            "recurrent_noise": float(model.rec_noise),
            "trial_collection_directory": _path_record(
                resolve_trial_collection_dir(analysis_root), hash_file=False
            ),
            "normalization": {
                "phase_rule": "floor((j + 0.5) * n_navigation_decisions / 3)",
                "q_rule": "3 * execution_order_ordinal + phase",
                "future_rule": "continuous normalized block position t+h; no within-loop wrap",
                "support_rule": "all complete source loops except final target-support loop",
                "common_support_all_horizons": True,
            },
        },
        "geometry": geometry,
        "cortical_mechanism": cortical_mechanism,
    }
    return manifest


def resolve_anchor_reference_units(
    manifest: Mapping[str, Any],
    embedding_dir: Path | str,
    n_units: int,
) -> np.ndarray:
    """Resolve and validate the fixed anchor-distance reference unit set.

    New manifests carry both the fixed anatomical seed and the expanded local
    input-recipient zone.  Older v1 manifests are still readable and fall back
    to the embedding's ``anchor_unit_indices.npy``.
    """

    mechanism = manifest.get("cortical_mechanism")
    anchor_path = Path(embedding_dir).expanduser().resolve() / "anchor_unit_indices.npy"
    if mechanism is None:
        if not anchor_path.is_file():
            raise FileNotFoundError(anchor_path)
        units = np.unique(
            np.asarray(np.load(anchor_path), dtype=np.int64).reshape(-1)
        )
    else:
        recipient = mechanism["local_input_recipient_zone"]
        recipient_mask = np.asarray(recipient["unit_mask"], dtype=np.uint8).reshape(-1)
        recipient_indices = np.asarray(
            recipient["unit_indices"], dtype=np.int64
        ).reshape(-1)
        if recipient_mask.shape != (int(n_units),) or np.any(
            (recipient_mask != 0) & (recipient_mask != 1)
        ):
            raise ValueError("Manifest local input-recipient mask is invalid.")
        if not np.array_equal(np.flatnonzero(recipient_mask), recipient_indices):
            raise ValueError(
                "Manifest local input-recipient mask and indices disagree."
            )
        if int(recipient["unit_count"]) != len(recipient_indices):
            raise ValueError("Manifest local input-recipient count is inconsistent.")
        if recipient.get("unit_mask_sha256") != array_sha256(recipient_mask):
            raise ValueError("Manifest local input-recipient mask hash is invalid.")
        if recipient.get("unit_indices_sha256") != array_sha256(recipient_indices):
            raise ValueError("Manifest local input-recipient index hash is invalid.")

        seed = mechanism["anatomical_anchor_seed"]
        reference = mechanism["anchor_distance_reference"]
        seed_units = np.asarray(seed["unit_indices"], dtype=np.int64).reshape(-1)
        units = np.asarray(reference["unit_indices"], dtype=np.int64).reshape(-1)
        if not np.array_equal(units, seed_units):
            raise ValueError(
                "Anchor-distance reference must equal the fixed anatomical seed."
            )
        expected_hash = array_sha256(units)
        if seed.get("unit_indices_sha256") != expected_hash or reference.get(
            "unit_indices_sha256"
        ) != expected_hash:
            raise ValueError("Manifest anatomical anchor hash is invalid.")
        if int(seed["unit_count"]) != len(units) or int(
            reference["unit_count"]
        ) != len(units):
            raise ValueError("Manifest anatomical anchor count is inconsistent.")
        if bool(reference.get("uses_expanded_input_recipient_zone", True)):
            raise ValueError(
                "Anchor-distance reference is incorrectly marked as the expanded "
                "input-recipient zone."
            )
        if anchor_path.is_file():
            file_units = np.unique(
                np.asarray(np.load(anchor_path), dtype=np.int64).reshape(-1)
            )
            if not np.array_equal(units, file_units):
                raise ValueError(
                    "Manifest anatomical anchor differs from the embedding file."
                )

    if len(units) == 0 or np.any(units < 0) or np.any(units >= int(n_units)):
        raise ValueError("A valid non-empty anchor-unit set is required.")
    return units.astype(np.int64, copy=False)


def load_analysis_geometry(
    manifest_or_root: Mapping[str, Any] | Path | str,
) -> dict[str, np.ndarray | Path | str]:
    """Load dynamic unit geometry used by all spatial summaries.

    Distances are the unscaled surface-geodesic values stored with the
    embedding (millimetres for the supplied fsLR resource), not the mean-scaled
    recurrent-initialization buffer.
    """
    manifest = (
        dict(manifest_or_root)
        if isinstance(manifest_or_root, Mapping)
        else load_analysis_manifest(manifest_or_root)
    )
    geometry = manifest["geometry"]
    embedding_dir = _resolve_path_record(geometry["embedding_directory"])
    surface_path = _resolve_path_record(geometry["surface_path"])
    distance = np.asarray(np.load(embedding_dir / "distance_matrix.npy"), dtype=float)
    vertices = np.asarray(np.load(embedding_dir / "sampled_indices.npy"), dtype=np.int64).reshape(-1)
    n_units = int(manifest["model"]["n_recurrent_units"])
    if distance.shape != (n_units, n_units) or vertices.shape != (n_units,):
        raise ValueError("Embedding geometry does not match manifest unit count.")
    if not np.allclose(distance, distance.T, rtol=0.0, atol=1e-6):
        raise ValueError("Geodesic distance matrix is not symmetric.")

    try:
        import nibabel as nib
    except ImportError as error:  # pragma: no cover - production env has nibabel
        raise ImportError("Loading fsLR geometry requires nibabel.") from error
    coords, faces = nib.load(str(surface_path)).agg_data()
    coords = np.asarray(coords, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if np.any(vertices < 0) or np.any(vertices >= len(coords)):
        raise ValueError("Unit vertex index lies outside the surface.")
    unit_xyz = coords[vertices]

    assignment_path = embedding_dir / "vertex_to_cluster.npy"
    adjacency = np.zeros((n_units, n_units), dtype=bool)
    if assignment_path.is_file():
        assignment = np.asarray(np.load(assignment_path), dtype=np.int64).reshape(-1)
        if len(assignment) != len(coords):
            raise ValueError("vertex_to_cluster length differs from surface vertex count.")
        for left, right in ((0, 1), (1, 2), (2, 0)):
            a = assignment[faces[:, left]]
            b = assignment[faces[:, right]]
            valid = (a >= 0) & (b >= 0) & (a < n_units) & (b < n_units) & (a != b)
            adjacency[a[valid], b[valid]] = True
            adjacency[b[valid], a[valid]] = True
    else:
        vertex_to_unit = {int(vertex): unit for unit, vertex in enumerate(vertices)}
        for triangle in faces:
            for left, right in ((0, 1), (1, 2), (2, 0)):
                a = vertex_to_unit.get(int(triangle[left]))
                b = vertex_to_unit.get(int(triangle[right]))
                if a is not None and b is not None and a != b:
                    adjacency[a, b] = adjacency[b, a] = True

    anchor_units = resolve_anchor_reference_units(manifest, embedding_dir, n_units)
    anchor_distance = distance[:, anchor_units].mean(axis=1)
    span = float(np.ptp(anchor_distance))
    anchor_distance_normalized = (
        (anchor_distance - anchor_distance.min()) / span
        if span > 0
        else np.zeros_like(anchor_distance)
    )
    return {
        "embedding_directory": embedding_dir,
        "surface_path": surface_path,
        "coordinate_note": str(geometry["coordinate_note"]),
        "unit_surface_xyz": unit_xyz,
        "unit_xyz": unit_xyz,
        "geodesic_distance": distance,
        "distance_matrix": distance,
        "surface_faces": faces,
        "unit_vertex_indices": vertices,
        "unit_adjacency": adjacency,
        "anchor_unit_indices": anchor_units,
        "anchor_distance": anchor_distance,
        "anchor_distance_normalized": anchor_distance_normalized,
    }


# Backward-readable alias for collaborators importing a cortical-specific name.
load_cortical_geometry = load_analysis_geometry
