"""Reference-matched local RSA for the ABCD cortical RNN.

This module implements the deliberately narrow *RNN analogue* documented in
``reports/ABCD_REFERENCE_MATCHED_ANALYSIS_REPORT.md``.  The human DSR
simulation source was not supplied, so the normalized categorical construction
below is explicit and testable rather than described as an exact replication.

One output beta at unit ``u`` and split ``s`` is the signed coefficient of the
split-DSR RDM in a joint OLS fit to the cross-repeat Spearman RDM of the
geodesic searchlight centred on ``u``.  The other three split-DSR RDMs and the
five reported visual/motor controls are fitted simultaneously.  Only
path--path and reward--reward entries are used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.stats import rankdata

from scripts.ABCD_task.abcd_analysis_common import (
    load_analysis_geometry,
    resolve_existing_analysis_root,
)


PHASE_INSTRUCTION = 0
PHASE_NAVIGATION = 1
PHASE_REWARD = 2
NO_ACTION = 4
N_GRID_LOCATIONS = 9
N_ABSTRACT_GOALS = 4
N_PROGRESS_PHASES = 3
N_NORMALIZED_POSITIONS = N_ABSTRACT_GOALS * N_PROGRESS_PHASES
RSA_SEQUENCE_LENGTH = 12
SPLIT_HORIZONS = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (9, 10, 11),
)
SPLIT_LABELS = (
    "current_immediate_h0_2",
    "future_25_50_h3_5",
    "future_50_75_h6_8",
    "future_75_99_h9_11",
)
CONTROL_LABELS = (
    "current_location",
    "grid_l2",
    "reward_a_feedback",
    "current_action",
    "next_action",
)


@dataclass(frozen=True)
class LoadedBlock:
    """One complete factorial evaluation block from a stored trajectory."""

    key: tuple[int, int, int]
    base_index: int
    instruction_direction: int
    execution_relation: int
    configuration: tuple[int, int, int, int]
    execution_sequence: tuple[int, int, int, int]
    r: np.ndarray
    phase: np.ndarray
    abstract_goal: np.ndarray
    loop: np.ndarray
    location: np.ndarray
    action: np.ndarray
    reward_event: np.ndarray
    valid: np.ndarray
    source_path: Path


@dataclass(frozen=True)
class ConditionDesign:
    """Neural condition patterns and model sequences for both repeats."""

    patterns: np.ndarray  # [repeat=2, condition, unit]
    base_index: np.ndarray
    instruction_direction: np.ndarray
    execution_relation: np.ndarray
    abstract_goal: np.ndarray
    is_reward: np.ndarray
    configuration: np.ndarray
    execution_order_position: np.ndarray
    current_location_sequence: np.ndarray  # [condition, 12]
    current_action_sequence: np.ndarray  # [condition, 12]
    next_action_sequence: np.ndarray  # [condition, 12]
    future_location_sequence: np.ndarray  # [condition, horizon=12, 12]
    condition_labels: tuple[str, ...]
    n_loops: int


def _one_dimensional(array: np.ndarray, *, name: str) -> np.ndarray:
    """Remove singleton batch/channel axes from a stored time series."""

    out = np.asarray(array)
    out = np.squeeze(out)
    if out.ndim != 1:
        raise ValueError(f"{name} must reduce to one dimension, got {out.shape}.")
    return out


def _activity_matrix(array: np.ndarray) -> np.ndarray:
    """Convert stored ``[time, 1, unit, 1]`` activity to ``[time, unit]``."""

    out = np.asarray(array, dtype=float)
    if out.ndim >= 2 and out.shape[1] == 1:
        out = np.squeeze(out, axis=1)
    if out.ndim >= 2 and out.shape[-1] == 1:
        out = np.squeeze(out, axis=-1)
    if out.ndim != 2:
        raise ValueError(f"Activity must reduce to [time, unit], got {out.shape}.")
    if not np.all(np.isfinite(out)):
        raise ValueError("Activity contains non-finite values.")
    return out


def _first_scalar(store: Mapping[str, np.ndarray], names: Sequence[str]) -> int:
    for name in names:
        if name in store:
            values = np.asarray(store[name]).reshape(-1)
            if values.size:
                return int(values[0])
    raise KeyError(f"None of the required scalar fields exists: {tuple(names)}")


def load_block(path: Path) -> LoadedBlock:
    """Load one complete block and validate the fields used by the RSA."""

    with np.load(path, allow_pickle=False) as raw:
        store = {name: np.asarray(raw[name]) for name in raw.files}

    activity_key = "rs" if "rs" in store else "r"
    r = _activity_matrix(store[activity_key])
    phase = _one_dimensional(store["phase"], name="phase").astype(int)
    goal = _one_dimensional(
        store["current_required_abstract_goal_index"],
        name="current_required_abstract_goal_index",
    ).astype(int)
    loop = _one_dimensional(store["loop_index"], name="loop_index").astype(int)
    location = _one_dimensional(store["current_location"], name="current_location").astype(int)
    action = _one_dimensional(store["action"], name="action").astype(int)
    reward = _one_dimensional(store["reward_event"], name="reward_event").astype(bool)
    valid = (
        _one_dimensional(store["valid_timestep"], name="valid_timestep").astype(bool)
        if "valid_timestep" in store
        else np.ones(len(phase), dtype=bool)
    )
    lengths = {len(x) for x in (r, phase, goal, loop, location, action, reward, valid)}
    if len(lengths) != 1:
        raise ValueError(f"Stored block fields have inconsistent lengths: {sorted(lengths)}")

    base = _first_scalar(
        store,
        ("base_configuration_index", "base_index", "configuration_index"),
    )
    idir = _first_scalar(store, ("instruction_direction",))
    relation = _first_scalar(store, ("execution_relation",))
    configuration = tuple(
        int(x) for x in np.asarray(store["configuration"])[0].reshape(-1)
    )
    execution = tuple(
        int(x)
        for x in np.asarray(store["effective_execution_abstract_sequence"])[0].reshape(-1)
    )
    if len(configuration) != N_ABSTRACT_GOALS or len(set(configuration)) != N_ABSTRACT_GOALS:
        raise ValueError(f"Invalid ABCD configuration in {path}: {configuration}")
    if sorted(execution) != list(range(N_ABSTRACT_GOALS)):
        raise ValueError(f"Invalid effective execution sequence in {path}: {execution}")
    if np.any((location < 0) | (location >= N_GRID_LOCATIONS)):
        raise ValueError(f"Invalid physical location in {path}.")

    return LoadedBlock(
        key=(base, idir, relation),
        base_index=base,
        instruction_direction=idir,
        execution_relation=relation,
        configuration=configuration,
        execution_sequence=execution,
        r=r,
        phase=phase,
        abstract_goal=goal,
        loop=loop,
        location=location,
        action=action,
        reward_event=reward,
        valid=valid,
        source_path=path,
    )


def load_repeat_blocks(repeat_dir: Path) -> dict[tuple[int, int, int], LoadedBlock]:
    """Load all factorial blocks from ``repeat_dir/blocks`` without relying on filenames."""

    blocks_dir = repeat_dir / "blocks" if (repeat_dir / "blocks").is_dir() else repeat_dir
    paths = sorted(blocks_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No block NPZ files found in {blocks_dir}")
    blocks: dict[tuple[int, int, int], LoadedBlock] = {}
    for path in paths:
        block = load_block(path)
        if block.key in blocks:
            raise ValueError(f"Duplicate factorial cell {block.key} in {blocks_dir}")
        blocks[block.key] = block
    return blocks


def midpoint_resample(sequence: Sequence[int], n_samples: int) -> np.ndarray:
    """Deterministic nearest-neighbour midpoint resampling of a categorical sequence."""

    values = np.asarray(sequence, dtype=int).reshape(-1)
    if values.size == 0:
        raise ValueError("Cannot resample an empty sequence.")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    indices = np.floor((np.arange(n_samples, dtype=float) + 0.5) * len(values) / n_samples)
    indices = np.clip(indices.astype(int), 0, len(values) - 1)
    return values[indices]


def modal_sequence(sequences: Iterable[Sequence[int]]) -> np.ndarray:
    """Select the most frequent complete sequence with a deterministic lexical tie-break."""

    tuples = [tuple(int(x) for x in seq) for seq in sequences if len(seq) > 0]
    if not tuples:
        raise ValueError("No non-empty sequences are available for the modal route.")
    counts = Counter(tuples)
    best_count = max(counts.values())
    return np.asarray(min(seq for seq, count in counts.items() if count == best_count), dtype=int)


def _next_navigation_actions(block: LoadedBlock) -> dict[int, int]:
    """Map each movement row to the next movement action, ignoring reward/instruction dwells.

    The final movement wraps to the first movement solely for the reference's looped-average
    button-sequence control; this does not alter neural activity or task trajectories.
    """

    nav = np.flatnonzero(block.valid & (block.phase == PHASE_NAVIGATION))
    if nav.size == 0:
        raise ValueError(f"No navigation rows in {block.source_path}")
    following = np.roll(block.action[nav], -1)
    return {int(row): int(value) for row, value in zip(nav, following)}


def _extract_block_material(block: LoadedBlock) -> dict:
    """Extract loop-first patterns and route/action candidates from a full block."""

    next_by_row = _next_navigation_actions(block)
    loops = np.unique(block.loop[block.valid & (block.phase != PHASE_INSTRUCTION)])
    loops = loops[loops >= 0]
    if loops.size == 0:
        raise ValueError(f"No execution loops in {block.source_path}")

    patterns: dict[tuple[int, bool], list[np.ndarray]] = {}
    routes: dict[int, list[np.ndarray]] = {g: [] for g in range(N_ABSTRACT_GOALS)}
    actions: dict[int, list[np.ndarray]] = {g: [] for g in range(N_ABSTRACT_GOALS)}
    next_actions: dict[int, list[np.ndarray]] = {g: [] for g in range(N_ABSTRACT_GOALS)}
    reward_next: dict[int, list[int]] = {g: [] for g in range(N_ABSTRACT_GOALS)}

    nav_rows = np.flatnonzero(block.valid & (block.phase == PHASE_NAVIGATION))
    for loop in loops:
        for goal in range(N_ABSTRACT_GOALS):
            path_rows = np.flatnonzero(
                block.valid
                & (block.phase == PHASE_NAVIGATION)
                & (block.loop == loop)
                & (block.abstract_goal == goal)
            )
            reward_rows = np.flatnonzero(
                block.valid
                & (block.phase == PHASE_REWARD)
                & block.reward_event
                & (block.loop == loop)
                & (block.abstract_goal == goal)
            )
            if path_rows.size == 0 or reward_rows.size != 1:
                raise ValueError(
                    f"Incomplete loop={loop}, goal={goal} in {block.source_path}: "
                    f"path rows={path_rows.size}, reward rows={reward_rows.size}."
                )
            patterns.setdefault((goal, False), []).append(np.mean(block.r[path_rows], axis=0))
            patterns.setdefault((goal, True), []).append(np.mean(block.r[reward_rows], axis=0))
            routes[goal].append(block.location[path_rows].copy())
            actions[goal].append(block.action[path_rows].copy())
            next_actions[goal].append(
                np.asarray([next_by_row[int(row)] for row in path_rows], dtype=int)
            )

            reward_row = int(reward_rows[0])
            later = nav_rows[nav_rows > reward_row]
            next_row = int(later[0]) if later.size else int(nav_rows[0])
            reward_next[goal].append(int(block.action[next_row]))

    averaged_patterns = {
        key: np.mean(np.stack(values, axis=0), axis=0) for key, values in patterns.items()
    }
    return {
        "patterns": averaged_patterns,
        "routes": routes,
        "actions": actions,
        "next_actions": next_actions,
        "reward_next": reward_next,
        "n_loops": int(len(loops)),
    }


def _actual_order_position(execution_sequence: Sequence[int], abstract_goal: int) -> int:
    """Return the goal's ordinal in the *actual* execution direction."""

    try:
        return tuple(execution_sequence).index(int(abstract_goal))
    except ValueError as exc:
        raise ValueError(f"Goal {abstract_goal} absent from execution sequence {execution_sequence}") from exc


def _future_sequences(
    *,
    normalized_cycle: np.ndarray,
    execution_order_position: int,
    is_reward: bool,
    rewarded_location: int,
) -> np.ndarray:
    """Construct ``[12 horizons, 12 aligned samples]`` categorical DSR targets.

    Path samples are aligned to the three normalized source phases (four of the
    12 representative samples per phase).  A reward sample owns the rewarded
    boundary at h0; h1 continues into the early phase of the next path.
    """

    cycle = np.asarray(normalized_cycle, dtype=int).reshape(-1)
    if cycle.shape != (N_NORMALIZED_POSITIONS,):
        raise ValueError(f"Expected a 12-position normalized cycle, got {cycle.shape}.")
    out = np.empty((N_NORMALIZED_POSITIONS, RSA_SEQUENCE_LENGTH), dtype=int)
    if is_reward:
        out[0] = int(rewarded_location)
        next_q = 3 * ((int(execution_order_position) + 1) % N_ABSTRACT_GOALS)
        for horizon in range(1, N_NORMALIZED_POSITIONS):
            out[horizon] = cycle[(next_q + horizon - 1) % N_NORMALIZED_POSITIONS]
    else:
        phase = np.floor(
            N_PROGRESS_PHASES * np.arange(RSA_SEQUENCE_LENGTH) / RSA_SEQUENCE_LENGTH
        ).astype(int)
        source_q = 3 * int(execution_order_position) + phase
        for horizon in range(N_NORMALIZED_POSITIONS):
            out[horizon] = cycle[(source_q + horizon) % N_NORMALIZED_POSITIONS]
    return out


def construct_condition_design(
    repeat_blocks: Sequence[Mapping[tuple[int, int, int], LoadedBlock]],
) -> ConditionDesign:
    """Create matched neural conditions and reference-style model sequences.

    Exactly two repeats are required because the primary data RDM is cross-repeat.
    Base count, unit count, execution order and loop count are inferred from data.
    """

    if len(repeat_blocks) != 2:
        raise ValueError(f"Reference-matched RSA requires exactly two repeats, got {len(repeat_blocks)}.")
    keys0 = set(repeat_blocks[0])
    keys1 = set(repeat_blocks[1])
    if keys0 != keys1:
        raise ValueError(f"Factorial cells differ between repeats: {keys0 ^ keys1}")
    keys = sorted(keys0)
    bases = sorted({key[0] for key in keys})
    expected = {(base, idir, relation) for base in bases for idir in (0, 1) for relation in (0, 1)}
    if set(keys) != expected:
        raise ValueError("Evaluation is not a complete base × direction × relation factorial.")

    material = [
        {key: _extract_block_material(repeat_blocks[rep][key]) for key in keys}
        for rep in range(2)
    ]
    n_units = next(iter(repeat_blocks[0].values())).r.shape[1]
    loop_counts = {
        material[rep][key]["n_loops"] for rep in range(2) for key in keys
    }
    if len(loop_counts) != 1:
        raise ValueError(f"Blocks have unequal loop counts: {sorted(loop_counts)}")
    n_loops = int(next(iter(loop_counts)))

    pattern_repeats: list[list[np.ndarray]] = [[], []]
    base_out: list[int] = []
    idir_out: list[int] = []
    relation_out: list[int] = []
    goal_out: list[int] = []
    reward_out: list[bool] = []
    configuration_out: list[tuple[int, int, int, int]] = []
    order_out: list[int] = []
    loc_sequences: list[np.ndarray] = []
    action_sequences: list[np.ndarray] = []
    next_action_sequences: list[np.ndarray] = []
    future_sequences: list[np.ndarray] = []
    labels: list[str] = []

    for key in keys:
        block0 = repeat_blocks[0][key]
        block1 = repeat_blocks[1][key]
        if (
            block0.configuration != block1.configuration
            or block0.execution_sequence != block1.execution_sequence
        ):
            raise ValueError(f"Task design differs between repeats for factorial cell {key}.")

        pooled_routes = {
            goal: material[0][key]["routes"][goal] + material[1][key]["routes"][goal]
            for goal in range(N_ABSTRACT_GOALS)
        }
        representative_routes = {
            goal: modal_sequence(pooled_routes[goal]) for goal in range(N_ABSTRACT_GOALS)
        }
        representative_actions = {
            goal: modal_sequence(
                material[0][key]["actions"][goal] + material[1][key]["actions"][goal]
            )
            for goal in range(N_ABSTRACT_GOALS)
        }
        representative_next_actions = {
            goal: modal_sequence(
                material[0][key]["next_actions"][goal]
                + material[1][key]["next_actions"][goal]
            )
            for goal in range(N_ABSTRACT_GOALS)
        }
        representative_reward_next = {
            goal: int(
                modal_sequence(
                    [[x] for x in material[0][key]["reward_next"][goal]
                     + material[1][key]["reward_next"][goal]]
                )[0]
            )
            for goal in range(N_ABSTRACT_GOALS)
        }

        normalized_cycle = np.concatenate(
            [
                midpoint_resample(representative_routes[goal], N_PROGRESS_PHASES)
                for goal in block0.execution_sequence
            ]
        )

        for goal in range(N_ABSTRACT_GOALS):
            order_position = _actual_order_position(block0.execution_sequence, goal)
            for is_reward in (False, True):
                for rep in range(2):
                    pattern_repeats[rep].append(material[rep][key]["patterns"][(goal, is_reward)])
                if is_reward:
                    location_sequence = np.full(
                        RSA_SEQUENCE_LENGTH, block0.configuration[goal], dtype=int
                    )
                    action_sequence = np.full(RSA_SEQUENCE_LENGTH, NO_ACTION, dtype=int)
                    next_sequence = np.full(
                        RSA_SEQUENCE_LENGTH, representative_reward_next[goal], dtype=int
                    )
                    section = f"reward_at_{chr(65 + goal)}"
                else:
                    location_sequence = midpoint_resample(
                        representative_routes[goal], RSA_SEQUENCE_LENGTH
                    )
                    action_sequence = midpoint_resample(
                        representative_actions[goal], RSA_SEQUENCE_LENGTH
                    )
                    next_sequence = midpoint_resample(
                        representative_next_actions[goal], RSA_SEQUENCE_LENGTH
                    )
                    section = f"path_to_{chr(65 + goal)}"

                base_out.append(block0.base_index)
                idir_out.append(block0.instruction_direction)
                relation_out.append(block0.execution_relation)
                goal_out.append(goal)
                reward_out.append(is_reward)
                configuration_out.append(block0.configuration)
                order_out.append(order_position)
                loc_sequences.append(location_sequence)
                action_sequences.append(action_sequence)
                next_action_sequences.append(next_sequence)
                future_sequences.append(
                    _future_sequences(
                        normalized_cycle=normalized_cycle,
                        execution_order_position=order_position,
                        is_reward=is_reward,
                        rewarded_location=block0.configuration[goal],
                    )
                )
                labels.append(
                    f"base{block0.base_index}_idir{block0.instruction_direction}_"
                    f"exec{block0.execution_relation}_{section}"
                )

    patterns = np.asarray(pattern_repeats, dtype=float)
    if patterns.shape != (2, len(labels), n_units):
        raise RuntimeError(f"Unexpected condition pattern shape {patterns.shape}.")
    return ConditionDesign(
        patterns=patterns,
        base_index=np.asarray(base_out, dtype=int),
        instruction_direction=np.asarray(idir_out, dtype=int),
        execution_relation=np.asarray(relation_out, dtype=int),
        abstract_goal=np.asarray(goal_out, dtype=int),
        is_reward=np.asarray(reward_out, dtype=bool),
        configuration=np.asarray(configuration_out, dtype=int),
        execution_order_position=np.asarray(order_out, dtype=int),
        current_location_sequence=np.asarray(loc_sequences, dtype=int),
        current_action_sequence=np.asarray(action_sequences, dtype=int),
        next_action_sequence=np.asarray(next_action_sequences, dtype=int),
        future_location_sequence=np.asarray(future_sequences, dtype=int),
        condition_labels=tuple(labels),
        n_loops=n_loops,
    )


def normalized_hamming_rdm(sequences: np.ndarray) -> np.ndarray:
    """Pairwise normalized Hamming distance after flattening non-condition axes."""

    values = np.asarray(sequences)
    if values.ndim < 2:
        raise ValueError("Sequences must have condition and feature axes.")
    flat = values.reshape(values.shape[0], -1)
    return np.mean(flat[:, None, :] != flat[None, :, :], axis=-1, dtype=float)


def cosine_rdm(features: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance with explicit zero-norm validation."""

    x = np.asarray(features, dtype=float)
    if x.ndim != 2:
        raise ValueError("Cosine features must be two-dimensional.")
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms <= 0):
        raise ValueError("Cosine features include a zero-norm condition.")
    similarity = (x / norms[:, None]) @ (x / norms[:, None]).T
    return np.clip(1.0 - similarity, 0.0, 2.0)


def build_model_rdms(design: ConditionDesign) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build the four split-DSR and five control RDMs described in the PDF."""

    rdms: list[np.ndarray] = []
    for horizons in SPLIT_HORIZONS:
        rdms.append(
            normalized_hamming_rdm(design.future_location_sequence[:, horizons, :])
        )

    rdms.append(normalized_hamming_rdm(design.current_location_sequence))

    coordinates = np.stack(
        np.meshgrid(np.arange(3), np.arange(3), indexing="ij"), axis=-1
    ).reshape(-1, 2)
    grid_distance = np.linalg.norm(
        coordinates[:, None, :] - coordinates[None, :, :], axis=-1
    )
    l2_features = -grid_distance[design.current_location_sequence].reshape(
        len(design.condition_labels), -1
    )
    rdms.append(cosine_rdm(l2_features))

    # Binary visual-feedback control: two conditions are dissimilar exactly
    # when one, but not both, is the special reward-A condition.  Two non-A
    # conditions therefore have zero feedback dissimilarity.
    reward_a = design.is_reward & (design.abstract_goal == 0)
    feedback = (reward_a[:, None] != reward_a[None, :]).astype(float)
    rdms.append(feedback)
    rdms.append(normalized_hamming_rdm(design.current_action_sequence))
    rdms.append(normalized_hamming_rdm(design.next_action_sequence))

    names = SPLIT_LABELS + CONTROL_LABELS
    result = np.stack(rdms, axis=0)
    if result.shape != (len(names), len(design.condition_labels), len(design.condition_labels)):
        raise RuntimeError(f"Unexpected model RDM shape {result.shape}.")
    return result, names


def valid_rdm_entries(is_reward: np.ndarray) -> np.ndarray:
    """Upper-triangle path--path and reward--reward mask; path--reward is excluded."""

    reward = np.asarray(is_reward, dtype=bool).reshape(-1)
    return np.triu(np.ones((len(reward), len(reward)), dtype=bool), k=1) & (
        reward[:, None] == reward[None, :]
    )


def cross_repeat_spearman_rdm(
    repeat_1_patterns: np.ndarray,
    repeat_2_patterns: np.ndarray,
) -> np.ndarray:
    """Symmetrized half1(i)↔half2(j) Spearman dissimilarity.

    Ranking is performed across local units independently for every condition
    and repeat.  No same-repeat correlation enters this primary RDM.
    """

    x = np.asarray(repeat_1_patterns, dtype=float)
    y = np.asarray(repeat_2_patterns, dtype=float)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError(f"Repeat patterns must share [condition, feature] shape, got {x.shape}/{y.shape}.")
    if x.shape[1] < 2:
        raise ValueError("A Spearman searchlight requires at least two units.")

    def normalized_ranks(values: np.ndarray) -> np.ndarray:
        ranks = rankdata(values, axis=1, method="average")
        ranks -= np.mean(ranks, axis=1, keepdims=True)
        norm = np.linalg.norm(ranks, axis=1)
        normalized = np.full_like(ranks, np.nan, dtype=float)
        valid = norm > 0
        normalized[valid] = ranks[valid] / norm[valid, None]
        return normalized

    x_rank = normalized_ranks(x)
    y_rank = normalized_ranks(y)
    cross_similarity = x_rank @ y_rank.T
    similarity = 0.5 * (cross_similarity + cross_similarity.T)
    return 1.0 - np.clip(similarity, -1.0, 1.0)


def vectorized_model_design(
    model_rdms: np.ndarray,
    entry_mask: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """Return raw dissimilarity predictors, design rank and condition number."""

    rdms = np.asarray(model_rdms, dtype=float)
    mask = np.asarray(entry_mask, dtype=bool)
    x = rdms[:, mask].T
    if not np.all(np.isfinite(x)):
        raise ValueError("Model RDM design contains non-finite entries.")
    augmented = np.column_stack([np.ones(len(x)), x])
    rank = int(np.linalg.matrix_rank(augmented))
    condition = float(np.linalg.cond(augmented))
    return x, rank, condition


def fit_joint_rsa(data_rdm: np.ndarray, x: np.ndarray, entry_mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit signed raw-scale OLS coefficients with an intercept."""

    y = np.asarray(data_rdm, dtype=float)[np.asarray(entry_mask, dtype=bool)]
    if x.shape[0] != len(y):
        raise ValueError("RSA design and data RDM have different numbers of selected entries.")
    finite = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    if np.sum(finite) <= x.shape[1] + 1:
        return np.full(x.shape[1], np.nan), np.nan
    augmented = np.column_stack([np.ones(np.sum(finite)), x[finite]])
    if np.linalg.matrix_rank(augmented) != augmented.shape[1]:
        return np.full(x.shape[1], np.nan), np.nan
    coefficients = np.linalg.lstsq(augmented, y[finite], rcond=None)[0]
    fitted = augmented @ coefficients
    residual = y[finite] - fitted
    total = y[finite] - np.mean(y[finite])
    r2 = 1.0 - float(residual @ residual) / float(total @ total) if total @ total > 0 else np.nan
    return coefficients[1:], r2


def make_searchlights(distance_matrix: np.ndarray, radius_mm: float = 6.0) -> tuple[np.ndarray, ...]:
    """Create inclusive geodesic searchlights from the saved unit distance matrix."""

    distance = np.asarray(distance_matrix, dtype=float)
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError(f"Distance matrix must be square, got {distance.shape}.")
    if radius_mm <= 0:
        raise ValueError("Searchlight radius must be positive.")
    lights = tuple(
        np.flatnonzero(np.isfinite(distance[centre]) & (distance[centre] <= radius_mm))
        for centre in range(len(distance))
    )
    if min(len(light) for light in lights) < 2:
        raise ValueError("At least one geodesic searchlight contains fewer than two units.")
    return lights


def run_local_rsa(
    patterns: np.ndarray,
    searchlights: Sequence[np.ndarray],
    model_rdms: np.ndarray,
    entry_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """Fit the same joint RSA once at every geodesic searchlight centre."""

    neural = np.asarray(patterns, dtype=float)
    if neural.ndim != 3 or neural.shape[0] != 2:
        raise ValueError(f"Expected patterns [2, condition, unit], got {neural.shape}.")
    x, design_rank, design_condition = vectorized_model_design(model_rdms, entry_mask)
    n_predictors = model_rdms.shape[0]
    if design_rank != n_predictors + 1:
        raise ValueError(
            "The joint split-DSR/control RDM design is rank deficient: "
            f"rank {design_rank}, expected {n_predictors + 1}. Signed partial "
            "betas would not be uniquely identified."
        )
    betas = np.full((n_predictors, neural.shape[2]), np.nan, dtype=float)
    fit_r2 = np.full(neural.shape[2], np.nan, dtype=float)
    finite_entry_count = np.zeros(neural.shape[2], dtype=int)
    if len(searchlights) != neural.shape[2]:
        raise ValueError("There must be one searchlight per unit.")
    for centre, units in enumerate(searchlights):
        data_rdm = cross_repeat_spearman_rdm(
            neural[0][:, units], neural[1][:, units]
        )
        finite_entry_count[centre] = int(
            np.sum(np.isfinite(data_rdm[np.asarray(entry_mask, dtype=bool)]))
        )
        beta, r2 = fit_joint_rsa(data_rdm, x, entry_mask)
        betas[:, centre] = beta
        fit_r2[centre] = r2
    return betas, fit_r2, finite_entry_count, design_rank, design_condition


def surface_unit_adjacency(unit_vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, ...]:
    """Induce the fsLR triangle graph on the embedded unit vertices."""

    vertices = np.asarray(unit_vertices, dtype=int).reshape(-1)
    vertex_to_unit = {int(vertex): unit for unit, vertex in enumerate(vertices)}
    adjacency = [set() for _ in vertices]
    for triangle in np.asarray(faces, dtype=int):
        for va, vb in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            ua = vertex_to_unit.get(int(va))
            ub = vertex_to_unit.get(int(vb))
            if ua is not None and ub is not None:
                adjacency[ua].add(ub)
                adjacency[ub].add(ua)
    return tuple(np.asarray(sorted(neighbors), dtype=int) for neighbors in adjacency)


def connected_components(mask: np.ndarray, adjacency: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Connected components of a boolean unit map on the induced surface graph."""

    selected = np.asarray(mask, dtype=bool).reshape(-1)
    if len(selected) != len(adjacency):
        raise ValueError("Mask and adjacency sizes differ.")
    visited = np.zeros(len(selected), dtype=bool)
    components: list[np.ndarray] = []
    for start in np.flatnonzero(selected):
        if visited[start]:
            continue
        queue: deque[int] = deque([int(start)])
        visited[start] = True
        component: list[int] = []
        while queue:
            unit = queue.popleft()
            component.append(unit)
            for neighbor in adjacency[unit]:
                neighbor = int(neighbor)
                if selected[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        components.append(np.asarray(component, dtype=int))
    return components


def weighted_com(
    values: np.ndarray,
    coordinates: np.ndarray,
    *,
    positive_only: bool,
) -> np.ndarray:
    """Beta-mass centre of mass in fsLR surface coordinates."""

    weight = np.asarray(values, dtype=float).reshape(-1)
    coords = np.asarray(coordinates, dtype=float)
    valid = np.isfinite(weight) & np.all(np.isfinite(coords), axis=1)
    if positive_only:
        valid &= weight > 0
    if not np.any(valid):
        return np.full(coords.shape[1], np.nan)
    denominator = float(np.sum(weight[valid]))
    if not np.isfinite(denominator) or abs(denominator) <= 1e-14:
        return np.full(coords.shape[1], np.nan)
    return np.sum(coords[valid] * weight[valid, None], axis=0) / denominator


def summarize_beta_map(
    beta: np.ndarray,
    unit_coordinates: np.ndarray,
    adjacency: Sequence[np.ndarray],
    anchor_distance: np.ndarray,
    percentile: float = 90.0,
) -> dict:
    """Human-code-matched strongest p90 component plus one all-positive sensitivity."""

    values = np.asarray(beta, dtype=float).reshape(-1)
    coords = np.asarray(unit_coordinates, dtype=float)
    anchor = np.asarray(anchor_distance, dtype=float).reshape(-1)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("Beta map has no finite units.")
    cutoff = float(np.percentile(values[finite], percentile))
    # Exact supplied human-code threshold analogue: strictly greater than the
    # within-map percentile.  The human implementation does not append an
    # extra positivity criterion.
    selected = finite & (values > cutoff)
    components = connected_components(selected, adjacency)
    if components:
        strongest = max(components, key=lambda units: float(np.sum(values[units])))
        primary_xyz = weighted_com(
            values[strongest], coords[strongest], positive_only=False
        )
        denominator = float(np.sum(values[strongest]))
        primary_anchor = (
            float(np.sum(anchor[strongest] * values[strongest]) / denominator)
            if abs(denominator) > 1e-14
            else np.nan
        )
        primary_mass = float(np.sum(values[strongest]))
    else:
        strongest = np.asarray([], dtype=int)
        primary_xyz = np.full(3, np.nan)
        primary_anchor = np.nan
        primary_mass = 0.0
    positive = finite & (values > 0)
    sensitivity_xyz = weighted_com(
        values[positive], coords[positive], positive_only=True
    )
    sensitivity_anchor = (
        float(np.average(anchor[positive], weights=values[positive]))
        if np.any(positive)
        else np.nan
    )
    return {
        "cutoff": cutoff,
        "selected_mask": selected,
        "strongest_units": strongest,
        "n_components": len(components),
        "primary_xyz": primary_xyz,
        "primary_surface_z": float(primary_xyz[2]),
        "primary_anchor_distance": primary_anchor,
        "primary_mass": primary_mass,
        "all_positive_xyz": sensitivity_xyz,
        "all_positive_surface_z": float(sensitivity_xyz[2]),
        "all_positive_anchor_distance": sensitivity_anchor,
    }


def load_geometry(embedding_dir: Path, surface_path: Path, n_units: int) -> dict:
    """Load geometry from manifest-resolved paths and assert unit identity/order."""

    sampled_path = embedding_dir / "sampled_indices.npy"
    distance_path = embedding_dir / "distance_matrix.npy"
    anchor_path = embedding_dir / "anchor_unit_indices.npy"
    for path in (sampled_path, distance_path, anchor_path, surface_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    vertices = np.asarray(np.load(sampled_path), dtype=int).reshape(-1)
    distance = np.asarray(np.load(distance_path), dtype=float)
    anchors = np.asarray(np.load(anchor_path), dtype=int).reshape(-1)
    if len(vertices) != n_units or distance.shape != (n_units, n_units):
        raise ValueError(
            f"Geometry/model mismatch: vertices={len(vertices)}, distance={distance.shape}, units={n_units}."
        )
    surface_coords, faces = nib.load(str(surface_path)).agg_data()
    surface_coords = np.asarray(surface_coords, dtype=float)
    faces = np.asarray(faces, dtype=int)
    unit_coords = surface_coords[vertices]
    anchor_distance = np.mean(distance[:, anchors], axis=1)
    assignment_path = embedding_dir / "vertex_to_cluster.npy"
    if assignment_path.is_file():
        assignment = np.asarray(np.load(assignment_path), dtype=int).reshape(-1)
        if len(assignment) != len(surface_coords):
            raise ValueError("vertex_to_cluster does not match the surface.")
        adjacency_matrix = np.zeros((n_units, n_units), dtype=bool)
        for left, right in ((0, 1), (1, 2), (2, 0)):
            first = assignment[faces[:, left]]
            second = assignment[faces[:, right]]
            valid = (
                (first >= 0)
                & (second >= 0)
                & (first < n_units)
                & (second < n_units)
                & (first != second)
            )
            adjacency_matrix[first[valid], second[valid]] = True
            adjacency_matrix[second[valid], first[valid]] = True
        adjacency = tuple(
            np.flatnonzero(adjacency_matrix[unit]) for unit in range(n_units)
        )
    else:
        adjacency = surface_unit_adjacency(vertices, faces)
    return {
        "unit_vertices": vertices,
        "distance_matrix": distance,
        "anchor_units": anchors,
        "anchor_distance": anchor_distance,
        "surface_coords": surface_coords,
        "faces": faces,
        "unit_coords": unit_coords,
        "adjacency": adjacency,
    }


def _manifest_value(manifest: Mapping, names: Sequence[str]):
    """Find a unique named value recursively in a compact analysis manifest."""

    found = []
    stack = [manifest]
    while stack:
        item = stack.pop()
        if isinstance(item, Mapping):
            for key, value in item.items():
                if key in names:
                    found.append(value)
                if isinstance(value, (Mapping, list, tuple)):
                    stack.append(value)
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    unique = []
    for value in found:
        if value not in unique:
            unique.append(value)
    return unique[0] if len(unique) == 1 else None


def resolve_geometry_paths(
    manifest: Mapping,
    embedding_dir: Path | None,
    surface_path: Path | None,
) -> tuple[Path, Path]:
    """Resolve geometry without hard-coding N480 or an embedding identity."""

    if embedding_dir is None:
        value = _manifest_value(manifest, ("embedding_dir", "embedding_directory"))
        if value is not None:
            embedding_dir = Path(value)
    if surface_path is None:
        value = _manifest_value(manifest, ("surface_path", "midthickness_surface"))
        if value is not None:
            surface_path = Path(value)
    if embedding_dir is None or surface_path is None:
        raise ValueError(
            "Geometry paths are absent from the manifest. Supply --embedding-dir and --surface-path."
        )
    if not embedding_dir.is_absolute():
        embedding_dir = REPO_ROOT / embedding_dir
    if not surface_path.is_absolute():
        surface_path = REPO_ROOT / surface_path
    return embedding_dir.resolve(), surface_path.resolve()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_summary(
    path: Path,
    betas: np.ndarray,
    unit_coords: np.ndarray,
    spatial: Sequence[dict],
    model_rdms: np.ndarray,
    entry_mask: np.ndarray,
    fit_r2: np.ndarray,
) -> None:
    """Write one compact multi-panel figure; no estimator/threshold figure grid."""

    figure = plt.figure(figsize=(14, 8), constrained_layout=True)
    grid = figure.add_gridspec(2, 4, height_ratios=(1.0, 0.85))
    vmax = float(np.nanpercentile(np.abs(betas[:4]), 98))
    vmax = vmax if vmax > 0 else 1.0
    map_axes = []
    map_titles = (
        "Current/immediate\nh=0–2",
        "25–50% future\nh=3–5",
        "50–75% future\nh=6–8",
        "75–99% future\nh=9–11",
    )
    for split in range(4):
        axis = figure.add_subplot(grid[0, split])
        map_axes.append(axis)
        scatter = axis.scatter(
            unit_coords[:, 1],
            unit_coords[:, 2],
            c=betas[split],
            s=14,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            linewidths=0,
        )
        strongest = spatial[split]["strongest_units"]
        if len(strongest):
            axis.scatter(
                unit_coords[strongest, 1], unit_coords[strongest, 2],
                facecolors="none", edgecolors="black", s=30, linewidths=0.7,
            )
        axis.set_title(map_titles[split], fontsize=10)
        axis.set_xlabel("fsLR y")
        if split == 0:
            axis.set_ylabel("fsLR surface z")
    figure.colorbar(
        scatter,
        ax=map_axes,
        shrink=0.72,
        pad=0.015,
        label="signed partial beta",
    )

    axis = figure.add_subplot(grid[1, 0])
    x = np.arange(4)
    primary_mass = np.asarray([item["primary_mass"] for item in spatial])
    nonpositive_component = primary_mass <= 0
    primary_z = np.asarray([item["primary_surface_z"] for item in spatial])
    axis.plot(x, primary_z, "o-", label="human-code p90 (signed)")
    axis.plot(x, [item["all_positive_surface_z"] for item in spatial], "s--", label="all-positive sensitivity")
    axis.scatter(
        x[nonpositive_component], primary_z[nonpositive_component], marker="x",
        s=75, linewidths=2, color="crimson", zorder=5,
        label="p90 component mass ≤0",
    )
    axis.set_xticks(x, ["0–2", "3–5", "6–8", "9–11"])
    axis.set_xlabel("normalized DSR horizon")
    axis.set_ylabel("fsLR surface z (not MNI)")
    axis.legend(fontsize=7)

    axis = figure.add_subplot(grid[1, 1])
    primary_anchor = np.asarray([item["primary_anchor_distance"] for item in spatial])
    axis.plot(x, primary_anchor, "o-", label="human-code p90 (signed)")
    axis.plot(x, [item["all_positive_anchor_distance"] for item in spatial], "s--", label="all-positive sensitivity")
    axis.scatter(
        x[nonpositive_component], primary_anchor[nonpositive_component], marker="x",
        s=75, linewidths=2, color="crimson", zorder=5,
    )
    axis.set_xticks(x, ["0–2", "3–5", "6–8", "9–11"])
    axis.set_xlabel("normalized DSR horizon")
    axis.set_ylabel("distance to fixed Area-25 seed (mm)")

    axis = figure.add_subplot(grid[1, 2])
    model_vectors = model_rdms[:, entry_mask]
    correlation = np.corrcoef(model_vectors)
    image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_title("Model-RDM correlations")
    short_names = ["S0", "S1", "S2", "S3", "loc", "L2", "A", "act", "next"]
    axis.set_xticks(range(len(correlation)), short_names, rotation=45, ha="right", fontsize=7)
    axis.set_yticks(range(len(correlation)), short_names, fontsize=7)
    figure.colorbar(image, ax=axis, shrink=0.7)

    axis = figure.add_subplot(grid[1, 3])
    axis.hist(fit_r2[np.isfinite(fit_r2)], bins=30, color="0.25")
    axis.axvline(np.nanmedian(fit_r2), color="tab:red", linestyle="--", label="median")
    axis.set_title("Joint RSA fit QC")
    axis.set_xlabel("searchlight R²")
    axis.set_ylabel("units")
    axis.legend(fontsize=7)
    figure.suptitle(
        "ABCD reference-matched local RSA\n"
        "cross-repeat Spearman; path–path/reward–reward entries only",
        fontsize=12,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_analysis(
    *,
    analysis_root: Path,
    repeat_dirs: Sequence[Path],
    embedding_dir: Path | None = None,
    surface_path: Path | None = None,
    radius_mm: float = 6.0,
) -> Path:
    """Run the complete local RSA and write one NPZ, one CSV and one figure."""
    analysis_root = resolve_existing_analysis_root(analysis_root)
    manifest_path = analysis_root / "analysis_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    if len(repeat_dirs) != 2:
        raise ValueError("Supply exactly two independent trained-noise repeat directories.")

    repeat_blocks = [load_repeat_blocks(path) for path in repeat_dirs]
    design = construct_condition_design(repeat_blocks)
    n_units = design.patterns.shape[2]
    if embedding_dir is None and surface_path is None:
        shared_geometry = load_analysis_geometry(analysis_root)
        adjacency_matrix = np.asarray(shared_geometry["unit_adjacency"], dtype=bool)
        geometry = {
            "unit_vertices": np.asarray(
                shared_geometry["unit_vertex_indices"], dtype=int
            ),
            "distance_matrix": np.asarray(
                shared_geometry["geodesic_distance"], dtype=float
            ),
            "anchor_units": np.asarray(
                shared_geometry["anchor_unit_indices"], dtype=int
            ),
            "anchor_distance": np.asarray(
                shared_geometry["anchor_distance"], dtype=float
            ),
            "unit_coords": np.asarray(
                shared_geometry["unit_surface_xyz"], dtype=float
            ),
            "adjacency": tuple(
                np.flatnonzero(adjacency_matrix[unit])
                for unit in range(len(adjacency_matrix))
            ),
        }
        embedding_dir = Path(shared_geometry["embedding_directory"])
        surface_path = Path(shared_geometry["surface_path"])
    else:
        embedding_dir, surface_path = resolve_geometry_paths(
            manifest, embedding_dir, surface_path
        )
        geometry = load_geometry(embedding_dir, surface_path, n_units)
    if len(geometry["unit_vertices"]) != n_units:
        raise ValueError("Manifest geometry does not match the neural unit count.")
    searchlights = make_searchlights(geometry["distance_matrix"], radius_mm)
    model_rdms, predictor_names = build_model_rdms(design)
    entry_mask = valid_rdm_entries(design.is_reward)
    betas, fit_r2, finite_entry_count, design_rank, design_condition = run_local_rsa(
        design.patterns, searchlights, model_rdms, entry_mask
    )
    if not np.any(np.all(np.isfinite(betas[:4]), axis=0)):
        raise ValueError("No searchlight yielded identifiable split-DSR beta maps.")
    spatial = [
        summarize_beta_map(
            betas[split], geometry["unit_coords"], geometry["adjacency"],
            geometry["anchor_distance"], percentile=90.0,
        )
        for split in range(4)
    ]

    output_dir = analysis_root / "local_rsa"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.npz"
    np.savez_compressed(
        results_path,
        split_beta=betas[:4],
        nuisance_beta=betas[4:],
        joint_fit_r2=fit_r2,
        finite_rdm_entry_count=finite_entry_count,
        model_rdms=model_rdms,
        rdm_entry_mask=entry_mask,
        predictor_names=np.asarray(predictor_names),
        condition_labels=np.asarray(design.condition_labels),
        condition_base_index=design.base_index,
        condition_instruction_direction=design.instruction_direction,
        condition_execution_relation=design.execution_relation,
        condition_abstract_goal=design.abstract_goal,
        condition_is_reward=design.is_reward,
        condition_execution_order_position=design.execution_order_position,
        current_location_sequence=design.current_location_sequence,
        current_action_sequence=design.current_action_sequence,
        next_action_sequence=design.next_action_sequence,
        future_location_sequence=design.future_location_sequence,
        unit_vertices=geometry["unit_vertices"],
        unit_coordinates_fslr=geometry["unit_coords"],
        anchor_distance=geometry["anchor_distance"],
        searchlight_size=np.asarray([len(light) for light in searchlights], dtype=int),
        primary_com_xyz_fslr=np.stack([item["primary_xyz"] for item in spatial]),
        primary_com_surface_z=np.asarray([item["primary_surface_z"] for item in spatial]),
        primary_com_anchor_distance=np.asarray([item["primary_anchor_distance"] for item in spatial]),
        sensitivity_all_positive_com_xyz_fslr=np.stack([item["all_positive_xyz"] for item in spatial]),
        sensitivity_all_positive_surface_z=np.asarray([item["all_positive_surface_z"] for item in spatial]),
        sensitivity_all_positive_anchor_distance=np.asarray([item["all_positive_anchor_distance"] for item in spatial]),
        primary_p90_cutoff=np.asarray([item["cutoff"] for item in spatial]),
        primary_component_mass=np.asarray([item["primary_mass"] for item in spatial]),
        primary_component_count=np.asarray([item["n_components"] for item in spatial]),
    )
    rows = []
    for split, item in enumerate(spatial):
        rows.append(
            {
                "split": split,
                "label": SPLIT_LABELS[split],
                "horizons": "-".join(str(x) for x in SPLIT_HORIZONS[split]),
                "mean_beta": float(np.nanmean(betas[split])),
                "median_beta": float(np.nanmedian(betas[split])),
                "min_beta": float(np.nanmin(betas[split])),
                "max_beta": float(np.nanmax(betas[split])),
                "positive_unit_fraction": float(np.nanmean(betas[split] > 0)),
                "p90_cutoff": item["cutoff"],
                "p90_n_components": item["n_components"],
                "p90_strongest_mass": item["primary_mass"],
                "p90_component_positive_mass": bool(item["primary_mass"] > 0),
                "p90_com_fslr_x": item["primary_xyz"][0],
                "p90_com_fslr_y": item["primary_xyz"][1],
                "p90_com_surface_z": item["primary_surface_z"],
                "p90_com_anchor_distance_mm": item["primary_anchor_distance"],
                "all_positive_com_surface_z": item["all_positive_surface_z"],
                "all_positive_com_anchor_distance_mm": item["all_positive_anchor_distance"],
            }
        )
    _write_csv(output_dir / "summary.csv", rows)
    _plot_summary(
        output_dir / "summary.png", betas, geometry["unit_coords"], spatial,
        model_rdms, entry_mask, fit_r2,
    )
    metadata = {
        "analysis": "reference_matched_local_rsa",
        "interpretation": (
            "Each split beta is the signed unique raw-scale OLS association between a local "
            "cross-repeat recurrent-pattern RDM and one normalized DSR horizon RDM, conditional "
            "on the other three horizons and five controls."
        ),
        "reference_status": (
            "RNN analogue: human DSR generator/resampling code was not supplied; the exact "
            "categorical midpoint construction is documented in the pre-specification report."
        ),
        "n_units": n_units,
        "n_bases": int(len(np.unique(design.base_index))),
        "n_conditions": len(design.condition_labels),
        "n_loops_averaged_per_block": design.n_loops,
        "repeat_directories": [str(path.resolve()) for path in repeat_dirs],
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": _file_sha256(manifest_path),
        "repeat_block_counts": [len(blocks) for blocks in repeat_blocks],
        "searchlight_radius_surface_mm": radius_mm,
        "searchlight_size_min_median_max": [
            int(min(map(len, searchlights))),
            float(np.median(list(map(len, searchlights)))),
            int(max(map(len, searchlights))),
        ],
        "split_horizons": [list(values) for values in SPLIT_HORIZONS],
        "rdm_entries": "upper triangle; path-path and reward-reward only",
        "data_rdm": "symmetrized cross-repeat Spearman distance",
        "regression": "joint unstandardized OLS with intercept",
        "predictor_names": list(predictor_names),
        "design_rank_with_intercept": design_rank,
        "design_columns_with_intercept": len(predictor_names) + 1,
        "design_condition_number": design_condition,
        "median_joint_fit_r2": float(np.nanmedian(fit_r2)),
        "finite_searchlight_beta_count": int(
            np.sum(np.all(np.isfinite(betas[:4]), axis=0))
        ),
        "finite_rdm_entry_count_min_median_max": [
            int(np.min(finite_entry_count)),
            float(np.median(finite_entry_count)),
            int(np.max(finite_entry_count)),
        ],
        "primary_spatial_rule": (
            "beta strictly above ROI p90; surface connected components; strongest "
            "summed-beta component; beta-mass COM"
        ),
        "single_spatial_sensitivity": "all-positive whole-map beta-mass COM",
        "coordinate_system": "fsLR/Conte69 surface coordinates; not human MNI",
        "embedding_dir": str(embedding_dir),
        "surface_path": str(surface_path),
        "geometry_hashes": {
            "distance_matrix_sha256": _file_sha256(embedding_dir / "distance_matrix.npy"),
            "sampled_indices_sha256": _file_sha256(embedding_dir / "sampled_indices.npy"),
            "surface_sha256": _file_sha256(surface_path),
        },
        "outputs": {
            "results": str(results_path),
            "summary_csv": str(output_dir / "summary.csv"),
            "summary_figure": str(output_dir / "summary.png"),
        },
    }
    with (output_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "analysis_root", type=Path,
        help=(
            "Collected analysis root, managed run directory, canonical "
            "checkpoint, or legacy checkpoint."
        ),
    )
    parser.add_argument("--repeat-1", type=Path, default=None)
    parser.add_argument("--repeat-2", type=Path, default=None)
    parser.add_argument("--embedding-dir", type=Path, default=None)
    parser.add_argument("--surface-path", type=Path, default=None)
    parser.add_argument("--radius-mm", type=float, default=6.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_root = resolve_existing_analysis_root(args.analysis_root)
    trial_root = analysis_root / "trial_collection"
    repeat_dirs = (
        args.repeat_1 or trial_root / "repeat_01",
        args.repeat_2 or trial_root / "repeat_02",
    )
    output = run_analysis(
        analysis_root=analysis_root,
        repeat_dirs=repeat_dirs,
        embedding_dir=args.embedding_dir,
        surface_path=args.surface_path,
        radius_mm=args.radius_mm,
    )
    print(f"Local RSA outputs: {output}")


if __name__ == "__main__":
    main()
