"""Human 7T-fMRI ABCD instruction-and-navigation task.

This module implements the behavioural state machine used for the human ABCD
task without depending on the main repo's original :class:`~pysta.envs.MazeEnv`. A complete
environment episode is one block: sequential instruction followed by a
continuous execution of a circular four-goal sequence for ``num_loops`` loops.

Locations use row-major indices on an open 3 x 3 grid::

    0 1 2
    3 4 5
    6 7 8

Actions are ordered ``up, down, left, right``.  Attempts to cross a boundary
leave the location unchanged and can never produce reward.
"""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha1
from itertools import combinations, permutations
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch


GRID_SIDE = 3
NUM_LOCATIONS = GRID_SIDE**2

ACTION_NAMES = ("up", "down", "left", "right")
UP, DOWN, LEFT, RIGHT = range(4)

PHASE_NAMES = ("INSTRUCTION", "NAVIGATION", "REWARD")
INSTRUCTION, NAVIGATION, REWARD = range(3)

INSTRUCTION_DIRECTION_NAMES = ("FORWARD", "BACKWARD")
FORWARD, BACKWARD = range(2)

EXECUTION_RELATION_NAMES = ("SAME", "REVERSE")
SAME, REVERSE = range(2)


# Synthetic, fallback for the final scanner-style evaluation.
# These are NOT original task coordinates from the laboratory experiment, which are not reported in the
# paper. The row-major location IDs use the grid shown above.  This
# deterministic bank satisfies the reported all-pairs Manhattan separation,
# has five distinct circular route classes and no direct reversal duplicates,
# and gives the mathematically best possible distribution of 20 goal
# appearances over nine cells: (3, 2, 2, 2, 2, 2, 2, 2, 3).  Each abstract
# label occupies five distinct cells with grid-centre centroid, and each of the
# four circular transition positions has the same aggregate distance.
# Its mean circular subpath length is 2.4 rather than the approximately 2.6
# reported in the paper; exhaustive search over valid five-cycle-class banks
# shows that exact 2.6 and optimal location balance cannot coexist under the
# open-grid/all-pairs constraints.
# Supply --fmri_base_configurations to replace this modelling fallback when
# the exact experimental coordinates become available.
DEFAULT_FMRI_BASE_CONFIGURATIONS = (
    (2, 6, 8, 0),
    (5, 1, 3, 7),
    (6, 8, 0, 4),
    (7, 3, 5, 1),
)


OBSERVATION_SLICES = {
    "current_location": slice(0, 9),
    "instruction_location": slice(9, 18),
    "execution_rule": slice(18, 20),
    "phase": slice(20, 23),
    "reward_event": slice(23, 24),
}


def location_to_row_col(location: int) -> tuple[int, int]:
    """Convert a row-major location ID to ``(row, column)``."""
    location = int(location)
    if not 0 <= location < NUM_LOCATIONS:
        raise ValueError(f"Location must be in 0..8, got {location}.")
    return divmod(location, GRID_SIDE)


def row_col_to_location(row: int, column: int) -> int:
    """Convert a valid ``(row, column)`` coordinate to a location ID."""
    row, column = int(row), int(column)
    if not (0 <= row < GRID_SIDE and 0 <= column < GRID_SIDE):
        raise ValueError(
            f"Grid coordinates must each be in 0..2, got ({row}, {column})."
        )
    return row * GRID_SIDE + column


def manhattan_distance(location_a: int, location_b: int) -> int:
    """Return Manhattan distance on the open 3 x 3 grid."""
    row_a, col_a = location_to_row_col(location_a)
    row_b, col_b = location_to_row_col(location_b)
    return abs(row_a - row_b) + abs(col_a - col_b)


def move_location(location: int, action: int) -> tuple[int, bool]:
    """Apply an action and return ``(new_location, action_was_valid)``.

    A boundary action returns the original location and ``False``.
    """
    location, action = int(location), int(action)
    row, column = location_to_row_col(location)
    if action == UP:
        new_row, new_column = row - 1, column
    elif action == DOWN:
        new_row, new_column = row + 1, column
    elif action == LEFT:
        new_row, new_column = row, column - 1
    elif action == RIGHT:
        new_row, new_column = row, column + 1
    else:
        raise ValueError(f"Action must be in 0..3, got {action}.")

    if not (0 <= new_row < GRID_SIDE and 0 <= new_column < GRID_SIDE):
        return location, False
    return row_col_to_location(new_row, new_column), True


def normalize_configuration(configuration: Sequence[int]) -> tuple[int, int, int, int]:
    """Validate and normalize one ordered physical ``(A, B, C, D)`` mapping."""
    try:
        normalized = tuple(int(location) for location in configuration)
    except TypeError as exc:
        raise TypeError("A configuration must be an iterable of four locations.") from exc
    if len(normalized) != 4:
        raise ValueError(
            f"An ABCD configuration must contain four locations, got {len(normalized)}."
        )
    if len(set(normalized)) != 4:
        raise ValueError(f"ABCD locations must be distinct, got {normalized}.")
    for location in normalized:
        location_to_row_col(location)
    return normalized


def configuration_has_minimum_distance(
    configuration: Sequence[int],
    min_manhattan_distance: int = 2,
    *,
    all_pairs: bool = False,
) -> bool:
    """Check goal separation, including the circular D-to-A transition.

    The behavioural interface requires consecutive circular goals to satisfy
    the threshold.  ``all_pairs=True`` applies the stronger constraint reported
    for the fMRI configuration set, whenever generated banks can support it.
    """
    configuration = normalize_configuration(configuration)
    min_manhattan_distance = int(min_manhattan_distance)
    if min_manhattan_distance < 0:
        raise ValueError("min_manhattan_distance must be non-negative.")

    if all_pairs:
        pairs = combinations(configuration, 2)
    else:
        pairs = (
            (configuration[index], configuration[(index + 1) % 4])
            for index in range(4)
        )
    return all(
        manhattan_distance(first, second) >= min_manhattan_distance
        for first, second in pairs
    )


def parse_configurations(configurations) -> tuple[tuple[int, int, int, int], ...]:
    """Parse an explicit configuration bank.

    Programmatic inputs may be one four-item sequence or a sequence of such
    configurations.  CLI strings use commas within configurations and
    semicolons between them, for example ``"0,2,6,8;8,6,2,0"``.
    """
    if configurations is None or (
        isinstance(configurations, str) and configurations == ""
    ):
        return ()

    if isinstance(configurations, str):
        parsed = []
        for item in configurations.split(";"):
            item = item.strip()
            if not item:
                continue
            parsed.append(normalize_configuration(item.split(",")))
        if not parsed:
            raise ValueError("The configuration string did not contain a configuration.")
        return tuple(parsed)

    configurations = list(configurations)
    if not configurations:
        return ()
    if len(configurations) == 4 and all(np.isscalar(item) for item in configurations):
        return (normalize_configuration(configurations),)
    return tuple(normalize_configuration(item) for item in configurations)


def configuration_cycle_variants(
    configuration: Sequence[int],
) -> tuple[tuple[int, int, int, int], ...]:
    """Return all rotations of a circular route in both directions.

    A configuration is an ordered A/B/C/D mapping, so these variants are not
    interchangeable for familiar-condition bookkeeping.  They *are* the same
    unoriented physical cycle for the stricter route-geometry held-out test.
    """
    configuration = normalize_configuration(configuration)
    reverse = tuple(reversed(configuration))
    variants = []
    for oriented in (configuration, reverse):
        for offset in range(4):
            variant = oriented[offset:] + oriented[:offset]
            if variant not in variants:
                variants.append(variant)
    return tuple(variants)


def canonical_configuration_cycle(
    configuration: Sequence[int],
) -> tuple[int, int, int, int]:
    """Canonical key for a circular route up to rotation and reversal."""
    return min(configuration_cycle_variants(configuration))


def configuration_bank_statistics(configurations) -> dict[str, object]:
    """Summarize a configuration bank without claiming experimental identity.

    The reported path length is the Manhattan distance for each of the four
    circular goal-to-goal transitions.  This is the statistic that can be
    compared with the paper's reported mean of approximately 2.6 once the exact
    scanner configurations are supplied explicitly.
    """
    bank = parse_configurations(configurations)
    if not bank:
        raise ValueError("Cannot summarize an empty ABCD configuration bank.")

    location_counts = np.bincount(
        np.asarray(bank, dtype=np.int64).reshape(-1), minlength=NUM_LOCATIONS
    )
    configuration_array = np.asarray(bank, dtype=np.int64)
    abstract_label_location_counts = np.stack(
        [
            np.bincount(configuration_array[:, label], minlength=NUM_LOCATIONS)
            for label in range(4)
        ]
    )
    circular_distances = np.asarray(
        [
            [
                manhattan_distance(configuration[index], configuration[(index + 1) % 4])
                for index in range(4)
            ]
            for configuration in bank
        ],
        dtype=np.int64,
    )
    bank_set = set(bank)
    physical_sets = {frozenset(configuration) for configuration in bank}
    reversal_keys = {
        min(configuration, tuple(reversed(configuration))) for configuration in bank
    }
    cycle_keys = {canonical_configuration_cycle(configuration) for configuration in bank}
    expected_location_count = float(4 * len(bank) / NUM_LOCATIONS)
    mean_circular_distance = float(circular_distances.mean())
    return {
        "num_configurations": len(bank),
        "num_unique_ordered_configurations": len(bank_set),
        "num_unique_physical_location_sets": len(physical_sets),
        "num_direct_reversal_classes": len(reversal_keys),
        "num_physical_cycle_classes": len(cycle_keys),
        "location_counts": tuple(int(value) for value in location_counts),
        "expected_location_count_if_uniform": expected_location_count,
        "location_count_range": int(location_counts.max() - location_counts.min()),
        "location_balance_squared_error": float(
            np.square(location_counts - expected_location_count).sum()
        ),
        "abstract_label_location_counts": tuple(
            tuple(int(value) for value in row)
            for row in abstract_label_location_counts
        ),
        "circular_manhattan_distances": tuple(
            tuple(int(value) for value in row) for row in circular_distances
        ),
        "per_configuration_mean_circular_manhattan_distance": tuple(
            float(value) for value in circular_distances.mean(axis=1)
        ),
        "mean_circular_manhattan_distance": mean_circular_distance,
        "difference_from_reported_mean_2p6": mean_circular_distance - 2.6,
        "mean_nonwrapping_abc_to_d_manhattan_distance": float(
            circular_distances[:, :3].mean()
        ),
        "all_pairs_minimum_distance_two": all(
            configuration_has_minimum_distance(configuration, 2, all_pairs=True)
            for configuration in bank
        ),
        "direct_reversals_complete": all(
            tuple(reversed(configuration)) in bank_set for configuration in bank
        ),
    }


def validate_fmri_configuration_bank(configurations) -> dict[str, object]:
    """Validate the structural controls of an explicit ten-config scanner bank.

    The paper does not report the ten coordinates, so this helper validates an
    explicitly supplied bank rather than embedding invented coordinates.  It
    enforces ten unique mappings arranged as five direct inverse pairs and the
    reported all-pairs separation, then returns balance/path-length statistics
    for transparent comparison with the reported controls.
    """
    bank = parse_configurations(configurations)
    if len(bank) != 10:
        raise ValueError(
            f"An fMRI comparison bank must contain exactly 10 configurations, got {len(bank)}."
        )
    if len(set(bank)) != len(bank):
        raise ValueError("An fMRI comparison bank must contain 10 unique mappings.")
    if not all(
        configuration_has_minimum_distance(configuration, 2, all_pairs=True)
        for configuration in bank
    ):
        raise ValueError(
            "Every pair of goal locations in an fMRI comparison configuration "
            "must be at least two Manhattan steps apart."
        )
    bank_set = set(bank)
    missing_reversals = [
        configuration
        for configuration in bank
        if tuple(reversed(configuration)) not in bank_set
    ]
    if missing_reversals:
        raise ValueError(
            "The ten fMRI comparison mappings must form five direct inverse "
            f"pairs; missing inverse for {missing_reversals}."
        )
    report = configuration_bank_statistics(bank)
    if report["num_direct_reversal_classes"] != 5:
        raise ValueError("The fMRI comparison bank must form exactly five inverse pairs.")
    return report


@lru_cache(maxsize=None)
def generate_synthetic_fmri_configuration_bank(
    *, seed: int = 0, objective: str = "balance_first"
) -> tuple[tuple[int, int, int, int], ...]:
    """Construct a labelled synthetic ten-config bank with an explicit trade-off.

    This is **not** the original task coordinates from the laboratory experiment. It exhaustively selects
    five distinct physical-cycle classes and includes each direct inverse.
    Under the open-grid/all-pairs assumptions, location balance and an exact
    circular mean of 2.6 cannot both be optimal, so callers must choose either
    ``"balance_first"`` or ``"distance_first"``. The seed deterministically
    breaks ties within the selected objective.
    """
    objective = str(objective).lower()
    if objective not in ("balance_first", "distance_first"):
        raise ValueError(
            "Synthetic fMRI bank objective must be 'balance_first' or "
            f"'distance_first', got {objective!r}."
        )

    cycle_representatives = generate_configuration_bank(
        18,
        seed=int(seed),
        min_manhattan_distance=2,
        prefer_all_pairs=True,
        unique_up_to_cycle=True,
    )
    rng = np.random.default_rng(int(seed))
    best_key = None
    best_bank = None
    for representative_indices in combinations(range(18), 5):
        bases = tuple(
            cycle_representatives[index] for index in representative_indices
        )
        bank = tuple(
            configuration
            for base in bases
            for configuration in (base, tuple(reversed(base)))
        )
        report = configuration_bank_statistics(bank)
        balance_error = float(report["location_balance_squared_error"])
        distance_error = abs(
            float(report["mean_circular_manhattan_distance"]) - 2.6
        )
        primary = (
            (balance_error, distance_error)
            if objective == "balance_first"
            else (distance_error, balance_error)
        )
        key = (*primary, float(rng.random()))
        if best_key is None or key < best_key:
            best_key = key
            best_bank = bank

    # The finite search space is non-empty by construction. Validate the
    # selected structural controls before exposing it.
    validate_fmri_configuration_bank(best_bank)
    return best_bank


def generate_configuration_bank(
    num_configurations: int,
    seed: int,
    min_manhattan_distance: int = 2,
    prefer_all_pairs: bool = True,
    exclude_configurations=None,
    exclude_reverse_equivalents: bool = False,
    unique_up_to_reversal: bool = False,
    exclude_cycle_equivalents: bool = False,
    unique_up_to_cycle: bool = False,
) -> tuple[tuple[int, int, int, int], ...]:
    """Generate a deterministic bank of ordered physical ABCD mappings.

    The requested consecutive (including wraparound) separation is always
    enforced.  When ``prefer_all_pairs`` is true, the stronger all-pairs
    separation reported for the fMRI configurations is enforced. Exact
    experimental coordinates are intentionally not encoded here.
    """
    num_configurations = int(num_configurations)
    if num_configurations < 1:
        raise ValueError("num_configurations must be positive.")

    excluded = set(parse_configurations(exclude_configurations))
    if exclude_reverse_equivalents:
        excluded |= {tuple(reversed(configuration)) for configuration in excluded}
    excluded_cycle_keys = (
        {canonical_configuration_cycle(configuration) for configuration in excluded}
        if exclude_cycle_equivalents
        else set()
    )
    candidates = [
        tuple(configuration)
        for configuration in permutations(range(NUM_LOCATIONS), 4)
        if configuration_has_minimum_distance(
            configuration, min_manhattan_distance, all_pairs=False
        )
        and tuple(configuration) not in excluded
        and (
            not exclude_cycle_equivalents
            or canonical_configuration_cycle(configuration) not in excluded_cycle_keys
        )
    ]
    if prefer_all_pairs:
        candidates = [
            configuration
            for configuration in candidates
            if configuration_has_minimum_distance(
                configuration, min_manhattan_distance, all_pairs=True
            )
        ]
    if len(candidates) < num_configurations:
        constraint_name = "all-pairs" if prefer_all_pairs else "circular consecutive"
        raise ValueError(
            f"Only {len(candidates)} configurations satisfy the {constraint_name} "
            f"minimum distance {min_manhattan_distance}; "
            f"{num_configurations} requested."
        )

    rng = np.random.default_rng(int(seed))
    ordered = [candidates[index] for index in rng.permutation(len(candidates))]
    if unique_up_to_cycle:
        selected = []
        selected_cycles = set()
        for configuration in ordered:
            cycle_key = canonical_configuration_cycle(configuration)
            if cycle_key in selected_cycles:
                continue
            selected.append(configuration)
            selected_cycles.add(cycle_key)
            if len(selected) == num_configurations:
                break
    elif unique_up_to_reversal:
        selected = []
        selected_or_reversed = set()
        for configuration in ordered:
            if configuration in selected_or_reversed:
                continue
            selected.append(configuration)
            selected_or_reversed.add(configuration)
            selected_or_reversed.add(tuple(reversed(configuration)))
            if len(selected) == num_configurations:
                break
    else:
        selected = ordered[:num_configurations]

    if len(selected) != num_configurations:
        equivalence_name = (
            "cycle" if unique_up_to_cycle else "reversal"
        )
        raise ValueError(
            f"Only {len(selected)} configurations remain after {equivalence_name}-equivalence "
            f"constraints; {num_configurations} requested."
        )
    return tuple(selected)


def split_configuration_bank(
    *,
    num_train: int,
    num_eval: int,
    seed: int,
    min_manhattan_distance: int = 2,
    prefer_all_pairs: bool = True,
) -> tuple[tuple[tuple[int, int, int, int], ...], tuple[tuple[int, int, int, int], ...]]:
    """Build deterministic banks held out by complete physical-cycle class."""
    full_bank = generate_configuration_bank(
        int(num_train) + int(num_eval),
        seed=seed,
        min_manhattan_distance=min_manhattan_distance,
        prefer_all_pairs=prefer_all_pairs,
        unique_up_to_cycle=True,
    )
    return full_bank[: int(num_train)], full_bank[int(num_train) :]


def _normalize_named_value(value, names: Sequence[str], label: str) -> int:
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized not in names:
            raise ValueError(f"Unknown {label} {value!r}; expected one of {tuple(names)}.")
        return names.index(normalized)
    value = int(value)
    if not 0 <= value < len(names):
        raise ValueError(f"{label} must be in 0..{len(names) - 1}, got {value}.")
    return value


def _normalize_choices(values, names: Sequence[str], label: str) -> tuple[int, ...]:
    if values is None:
        return tuple(range(len(names)))
    if isinstance(values, (str, int, np.integer)):
        values = [values]
    normalized = tuple(_normalize_named_value(value, names, label) for value in values)
    if not normalized:
        raise ValueError(f"At least one {label} must be allowed.")
    return normalized


class ABCDFMRIEnv:
    """Batched human ABCD task, with one recurrent episode per complete block."""

    side_length = GRID_SIDE
    num_locs = NUM_LOCATIONS
    num_actions = len(ACTION_NAMES)
    output_dim = num_actions
    obs_dim = 24
    output_format = "abcd_actions"

    action_names = ACTION_NAMES
    phase_names = PHASE_NAMES
    instruction_direction_names = INSTRUCTION_DIRECTION_NAMES
    execution_relation_names = EXECUTION_RELATION_NAMES
    observation_slices = OBSERVATION_SLICES

    def __init__(
        self,
        batch_size: int = 1,
        seed: int = 0,
        configuration_bank=None,
        num_configurations: int = 64,
        bank_name: str = "train",
        instruction_directions=None,
        execution_relations=None,
        num_loops: int = 5,
        instruction_repeats: int = 2,
        max_navigation_steps: int = 200,
        start_policy: str = "exclude_first_goal",
        fixed_start: int | None = None,
        min_manhattan_distance: int = 2,
        prefer_all_pairs: bool = True,
        **kwargs,
    ):
        self.batch = int(batch_size)
        if self.batch < 1:
            raise ValueError("batch_size must be positive.")
        self.batch_inds = torch.arange(self.batch)

        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.bank_name = str(bank_name)
        self.num_loops = int(num_loops)
        self.instruction_repeats = int(instruction_repeats)
        self.max_navigation_steps = int(max_navigation_steps)
        self.min_manhattan_distance = int(min_manhattan_distance)
        self.prefer_all_pairs = bool(prefer_all_pairs)
        if self.num_loops < 1:
            raise ValueError("num_loops must be positive.")
        if self.instruction_repeats < 1:
            raise ValueError("instruction_repeats must be positive.")
        if self.max_navigation_steps < 1:
            raise ValueError("max_navigation_steps must be positive.")

        self.allowed_instruction_directions = _normalize_choices(
            instruction_directions,
            INSTRUCTION_DIRECTION_NAMES,
            "instruction direction",
        )
        self.allowed_execution_relations = _normalize_choices(
            execution_relations,
            EXECUTION_RELATION_NAMES,
            "execution relation",
        )

        start_policy = str(start_policy).lower()
        start_aliases = {
            "exclude_first_goal": "exclude_first_goal",
            "exclude_first": "exclude_first_goal",
            "uniform_excluding_first": "exclude_first_goal",
            "uniform": "uniform",
            "anywhere": "uniform",
            "fixed": "fixed",
        }
        if start_policy not in start_aliases:
            raise ValueError(
                "start_policy must be 'exclude_first_goal', 'uniform', or 'fixed'."
            )
        self.start_policy = start_aliases[start_policy]
        self.fixed_start = None if fixed_start is None else int(fixed_start)
        if self.fixed_start is not None:
            location_to_row_col(self.fixed_start)
            self.start_policy = "fixed"
        if self.start_policy == "fixed" and self.fixed_start is None:
            raise ValueError("fixed_start is required when start_policy='fixed'.")

        parsed_bank = parse_configurations(configuration_bank)
        if not parsed_bank:
            parsed_bank = generate_configuration_bank(
                num_configurations=num_configurations,
                seed=self.seed,
                min_manhattan_distance=self.min_manhattan_distance,
                prefer_all_pairs=self.prefer_all_pairs,
            )
        for configuration in parsed_bank:
            if not configuration_has_minimum_distance(
                configuration,
                self.min_manhattan_distance,
                all_pairs=self.prefer_all_pairs,
            ):
                raise ValueError(
                    f"Configuration {configuration} violates the required minimum "
                    f"Manhattan distance {self.min_manhattan_distance}."
                )
        self.configuration_bank = tuple(parsed_bank)

        self._block_counter = -1

        # Static open-grid transition lookup. Boundary entries self-transition.
        self.neighbors = torch.empty(NUM_LOCATIONS, self.num_actions, dtype=torch.long)
        self.action_valid_lookup = torch.empty(
            NUM_LOCATIONS, self.num_actions, dtype=torch.bool
        )
        for location in range(NUM_LOCATIONS):
            for action in range(self.num_actions):
                new_location, valid = move_location(location, action)
                self.neighbors[location, action] = new_location
                self.action_valid_lookup[location, action] = valid

        self.reset()

    @property
    def name(self) -> str:
        bank_digest = sha1(repr(self.configuration_bank).encode("utf8")).hexdigest()[:8]
        instruction_signature = "-".join(
            INSTRUCTION_DIRECTION_NAMES[value].lower()
            for value in self.allowed_instruction_directions
        )
        execution_signature = "-".join(
            EXECUTION_RELATION_NAMES[value].lower()
            for value in self.allowed_execution_relations
        )
        start_signature = (
            f"fixed{self.fixed_start}"
            if self.start_policy == "fixed"
            else self.start_policy
        )
        return (
            f"ABCDFMRIEnv_L3_loops{self.num_loops}_instr{self.instruction_repeats}/"
            f"{self.bank_name}_cfg{bank_digest}_idir{instruction_signature}"
            f"_exec{execution_signature}_start{start_signature}"
            f"_navcap{self.max_navigation_steps}_taskseed{self.seed}"
        )

    @property
    def total_instruction_steps(self) -> int:
        return self.instruction_repeats * 4

    @property
    def total_required_goals(self) -> int:
        return self.num_loops * 4

    def obs_inds(self) -> dict[str, np.ndarray]:
        """Return the five semantic observation groups."""
        return {
            group: np.arange(group_slice.start, group_slice.stop, dtype=np.int64)
            for group, group_slice in OBSERVATION_SLICES.items()
        }

    def input_routing(self) -> Mapping[str, str]:
        """Describe cortical routing without encoding task semantics in the RNN."""
        return {
            "current_location": "local",
            "instruction_location": "local",
            "execution_rule": "global",
            "phase": "global",
            "reward_event": "global",
        }

    def _select_configurations(self, configurations) -> torch.Tensor:
        if configurations is None:
            bank_indices = self.rng.integers(
                0, len(self.configuration_bank), size=self.batch
            )
            self.configuration_index = torch.as_tensor(bank_indices, dtype=torch.long)
            return torch.tensor(
                [self.configuration_bank[index] for index in bank_indices],
                dtype=torch.long,
            )

        parsed = parse_configurations(configurations)
        if not parsed:
            raise ValueError("reset(configurations=...) cannot receive an empty bank.")
        for configuration in parsed:
            if not configuration_has_minimum_distance(
                configuration,
                self.min_manhattan_distance,
                all_pairs=self.prefer_all_pairs,
            ):
                raise ValueError(
                    f"Configuration {configuration} violates circular minimum distance."
                )
        if len(parsed) == 1:
            selected = parsed * self.batch
        elif len(parsed) == self.batch:
            selected = parsed
        else:
            sampled = self.rng.integers(0, len(parsed), size=self.batch)
            selected = tuple(parsed[index] for index in sampled)
        self.configuration_index = torch.full((self.batch,), -1, dtype=torch.long)
        return torch.tensor(selected, dtype=torch.long)

    def _select_conditions(self, values, allowed, names, label) -> torch.Tensor:
        if values is None:
            return torch.as_tensor(
                self.rng.choice(np.asarray(allowed), size=self.batch), dtype=torch.long
            )
        if isinstance(values, (str, int, np.integer)):
            normalized = [_normalize_named_value(values, names, label)]
        else:
            normalized = [
                _normalize_named_value(value, names, label) for value in values
            ]
        if len(normalized) == 1:
            normalized *= self.batch
        elif len(normalized) != self.batch:
            normalized = list(
                self.rng.choice(np.asarray(normalized), size=self.batch)
            )
        return torch.tensor(normalized, dtype=torch.long)

    def _sample_start_locations(self, start_locations=None) -> torch.Tensor:
        if start_locations is not None:
            if isinstance(start_locations, (int, np.integer)):
                starts = [int(start_locations)] * self.batch
            else:
                starts = [int(location) for location in start_locations]
                if len(starts) == 1:
                    starts *= self.batch
                elif len(starts) != self.batch:
                    raise ValueError(
                        f"start_locations must have length 1 or {self.batch}."
                    )
            for location in starts:
                location_to_row_col(location)
            return torch.tensor(starts, dtype=torch.long)

        if self.start_policy == "fixed":
            return torch.full((self.batch,), self.fixed_start, dtype=torch.long)

        starts = []
        first_targets = self._required_abstract_goal()
        first_physical = self.configuration[
            self.batch_inds, first_targets
        ]
        for batch_index in range(self.batch):
            choices = np.arange(NUM_LOCATIONS)
            if self.start_policy == "exclude_first_goal":
                choices = choices[choices != int(first_physical[batch_index])]
            starts.append(int(self.rng.choice(choices)))
        return torch.tensor(starts, dtype=torch.long)

    def reset(
        self,
        configurations=None,
        instruction_directions=None,
        execution_relations=None,
        start_locations=None,
    ):
        """Sample and initialize a new batch of complete blocks."""
        self._block_counter += 1
        self.configuration = self._select_configurations(configurations)
        self.instruction_direction = self._select_conditions(
            instruction_directions,
            self.allowed_instruction_directions,
            INSTRUCTION_DIRECTION_NAMES,
            "instruction direction",
        )
        self.execution_relation = self._select_conditions(
            execution_relations,
            self.allowed_execution_relations,
            EXECUTION_RELATION_NAMES,
            "execution relation",
        )

        forward_sequence = torch.arange(4, dtype=torch.long)
        backward_sequence = torch.flip(forward_sequence, dims=(0,))
        self.presented_sequence = torch.stack(
            [
                forward_sequence if direction == FORWARD else backward_sequence
                for direction in self.instruction_direction.tolist()
            ]
        )
        self.effective_execution_sequence = self.presented_sequence.clone()
        reverse_rows = self.execution_relation == REVERSE
        self.effective_execution_sequence[reverse_rows] = torch.flip(
            self.effective_execution_sequence[reverse_rows], dims=(1,)
        )

        self.phase = torch.full((self.batch,), INSTRUCTION, dtype=torch.long)
        self.instruction_presentation_index = torch.zeros(
            self.batch, dtype=torch.long
        )
        self.sequence_position = torch.zeros(self.batch, dtype=torch.long)
        self.loop_index = torch.zeros(self.batch, dtype=torch.long)
        self.successful_goal_count = torch.zeros(self.batch, dtype=torch.long)
        self.navigation_step_count = torch.zeros(self.batch, dtype=torch.long)
        self.block_timestep = torch.zeros(self.batch, dtype=torch.long)
        self.reward_event = torch.zeros(self.batch, dtype=torch.bool)
        self.latest_rew = torch.zeros(self.batch, dtype=torch.float32)
        self.finished = torch.zeros(self.batch, dtype=torch.bool)
        self.truncated = torch.zeros(self.batch, dtype=torch.bool)
        self._truncate_after_reward = torch.zeros(self.batch, dtype=torch.bool)

        self.loc = self._sample_start_locations(start_locations)
        self.start_location = self.loc.clone()
        # ``uniform`` genuinely samples all nine cells. The default
        # ``exclude_first_goal`` policy avoids this edge case. If an explicit
        # or uniform start already occupies the first target, execution onset
        # satisfies that goal and produces the ordinary explicit REWARD dwell,
        # without movement or a policy-loss timestep.
        self.started_on_first_goal = self.loc == self._required_physical_location()

        self._post_step = self._empty_post_step_metadata()
        return self.loc

    def _required_abstract_goal(self) -> torch.Tensor:
        return self.effective_execution_sequence[
            self.batch_inds, self.sequence_position.clamp(max=3)
        ]

    def _required_physical_location(self) -> torch.Tensor:
        abstract_goal = self._required_abstract_goal()
        return self.configuration[self.batch_inds, abstract_goal]

    @property
    def current_required_abstract_goal_index(self) -> torch.Tensor:
        """Current required A/B/C/D index for each batch row (internal target)."""
        return self._required_abstract_goal()

    @property
    def current_required_physical_location(self) -> torch.Tensor:
        """Current required grid location for each batch row (internal target)."""
        return self._required_physical_location()

    def observation(self) -> torch.Tensor:
        """Construct the exact 24-channel, phase-gated observation."""
        observation = torch.zeros(self.batch, self.obs_dim, dtype=torch.float32)
        active = ~self.finished

        instruction_rows = active & (self.phase == INSTRUCTION)
        if torch.any(instruction_rows):
            rows = torch.where(instruction_rows)[0]
            presentation_position = (
                self.instruction_presentation_index[rows] % 4
            )
            abstract_goal = self.presented_sequence[rows, presentation_position]
            physical_location = self.configuration[rows, abstract_goal]
            observation[
                rows,
                OBSERVATION_SLICES["instruction_location"].start + physical_location,
            ] = 1.0
            observation[
                rows,
                OBSERVATION_SLICES["execution_rule"].start
                + self.execution_relation[rows],
            ] = 1.0

        location_rows = active & (
            (self.phase == NAVIGATION) | (self.phase == REWARD)
        )
        if torch.any(location_rows):
            rows = torch.where(location_rows)[0]
            observation[
                rows,
                OBSERVATION_SLICES["current_location"].start + self.loc[rows],
            ] = 1.0

        reward_rows = active & (self.phase == REWARD) & self.reward_event
        observation[reward_rows, OBSERVATION_SLICES["reward_event"].start] = 1.0

        rows = torch.where(active)[0]
        observation[
            rows, OBSERVATION_SLICES["phase"].start + self.phase[rows]
        ] = 1.0
        return observation

    def policy_loss_mask(self) -> torch.Tensor:
        """Policy loss is defined only for active navigation timesteps."""
        return (~self.finished) & (self.phase == NAVIGATION)

    def action_sampling_mask(self) -> torch.Tensor:
        """Expose all four actions, including invalid boundary attempts."""
        return torch.ones(self.batch, self.output_dim, dtype=torch.float32)

    def optimal_actions(self) -> torch.Tensor:
        """Return every action that reduces target Manhattan distance by one."""
        optimal = torch.zeros(self.batch, self.output_dim, dtype=torch.float32)
        target_locations = self._required_physical_location()
        active_rows = torch.where(self.policy_loss_mask())[0]
        for batch_index in active_rows.tolist():
            location = int(self.loc[batch_index])
            target = int(target_locations[batch_index])
            current_distance = manhattan_distance(location, target)
            for action in range(self.num_actions):
                next_location, valid = move_location(location, action)
                if valid and manhattan_distance(next_location, target) == current_distance - 1:
                    optimal[batch_index, action] = 1.0
        return optimal

    def _empty_post_step_metadata(self) -> dict[str, torch.Tensor]:
        zeros_bool = torch.zeros(self.batch, dtype=torch.bool)
        return {
            "post_action_location": self.loc.clone(),
            "action_valid": zeros_bool.clone(),
            "movement_occurred": zeros_bool.clone(),
            "navigation_step_taken": zeros_bool.clone(),
            "reward_received": torch.zeros(self.batch, dtype=torch.float32),
            "target_reached": zeros_bool.clone(),
            "next_phase": self.phase.clone(),
            "next_finished": self.finished.clone(),
            "next_truncated": self.truncated.clone(),
            "next_loop_index": self.loop_index.clone(),
            "next_successful_goal_count": self.successful_goal_count.clone(),
            "next_required_abstract_goal_index": self._required_abstract_goal().clone(),
            "next_required_physical_location": self._required_physical_location().clone(),
        }

    def step(self, action) -> torch.Tensor:
        """Advance every unfinished block row by one abstract task timestep."""
        action = torch.as_tensor(action, dtype=torch.long).detach().cpu()
        if action.ndim == 0:
            action = action.expand(self.batch)
        if tuple(action.shape) != (self.batch,):
            raise ValueError(
                f"action must have shape ({self.batch},), got {tuple(action.shape)}."
            )
        if torch.any((action < 0) | (action >= self.num_actions)):
            raise ValueError("ABCD actions must be integer indices in 0..3.")

        pre_phase = self.phase.clone()
        was_active = ~self.finished
        self.latest_rew.zero_()
        post = self._empty_post_step_metadata()

        for batch_index in torch.where(was_active)[0].tolist():
            phase = int(pre_phase[batch_index])
            self.block_timestep[batch_index] += 1

            if phase == INSTRUCTION:
                self.instruction_presentation_index[batch_index] += 1
                if (
                    self.instruction_presentation_index[batch_index]
                    >= self.total_instruction_steps
                ):
                    if self.started_on_first_goal[batch_index]:
                        # Explicit modelling rule for the optional true-uniform
                        # start policy; the references does not specify this edge case.
                        self.latest_rew[batch_index] = 1.0
                        self.reward_event[batch_index] = True
                        self.successful_goal_count[batch_index] += 1
                        self.phase[batch_index] = REWARD
                        post["reward_received"][batch_index] = 1.0
                        post["target_reached"][batch_index] = True
                    else:
                        self.phase[batch_index] = NAVIGATION

            elif phase == REWARD:
                # The currently observed REWARD timestep is the one explicit
                # dwell. Advance the circular target only when that dwell ends.
                self.reward_event[batch_index] = False
                if self._truncate_after_reward[batch_index]:
                    self.truncated[batch_index] = True
                    self.finished[batch_index] = True
                elif (
                    self.successful_goal_count[batch_index]
                    >= self.total_required_goals
                ):
                    self.finished[batch_index] = True
                else:
                    if self.successful_goal_count[batch_index] % 4 == 0:
                        self.loop_index[batch_index] += 1
                    self.sequence_position[batch_index] = (
                        self.successful_goal_count[batch_index] % 4
                    )
                    self.phase[batch_index] = NAVIGATION

            elif phase == NAVIGATION:
                post["navigation_step_taken"][batch_index] = True
                self.navigation_step_count[batch_index] += 1
                old_location = int(self.loc[batch_index])
                next_location, valid = move_location(
                    old_location, int(action[batch_index])
                )
                self.loc[batch_index] = next_location
                post["action_valid"][batch_index] = valid
                post["movement_occurred"][batch_index] = next_location != old_location

                required_location = int(self._required_physical_location()[batch_index])
                # Boundary actions never reward, even if an unusual explicit
                # start happens to equal the first target.
                if valid and next_location == required_location:
                    self.latest_rew[batch_index] = 1.0
                    self.reward_event[batch_index] = True
                    self.successful_goal_count[batch_index] += 1
                    self.phase[batch_index] = REWARD
                    post["reward_received"][batch_index] = 1.0
                    post["target_reached"][batch_index] = True

                if self.navigation_step_count[batch_index] >= self.max_navigation_steps:
                    if self.phase[batch_index] == REWARD:
                        if (
                            self.successful_goal_count[batch_index]
                            < self.total_required_goals
                        ):
                            self._truncate_after_reward[batch_index] = True
                    else:
                        self.truncated[batch_index] = True
                        self.finished[batch_index] = True
            else:  # pragma: no cover - guarded by internal constants
                raise RuntimeError(f"Unknown task phase {phase}.")

        post["post_action_location"] = self.loc.clone()
        post["next_phase"] = self.phase.clone()
        post["next_finished"] = self.finished.clone()
        post["next_truncated"] = self.truncated.clone()
        post["next_loop_index"] = self.loop_index.clone()
        post["next_successful_goal_count"] = self.successful_goal_count.clone()
        post["next_required_abstract_goal_index"] = (
            self._required_abstract_goal().clone()
        )
        post["next_required_physical_location"] = (
            self._required_physical_location().clone()
        )
        self._post_step = post
        return self.latest_rew.clone()

    def trajectory_metadata(self) -> dict[str, object]:
        """Snapshot all task variables needed for later behavioral analyses."""
        required_abstract = self._required_abstract_goal()
        instruction_abstract = self.presented_sequence
        instruction_physical = torch.gather(
            self.configuration, 1, instruction_abstract
        )
        execution_abstract = self.effective_execution_sequence
        execution_physical = torch.gather(
            self.configuration, 1, execution_abstract
        )
        return {
            "valid_timestep": ~self.finished,
            "block_index": torch.full(
                (self.batch,), self._block_counter, dtype=torch.long
            ),
            "block_timestep": self.block_timestep,
            "phase": self.phase,
            "phase_name": tuple(PHASE_NAMES[int(value)] for value in self.phase),
            "current_location": self.loc,
            "start_location": self.start_location,
            "started_on_first_goal": self.started_on_first_goal,
            "configuration": self.configuration,
            "configuration_index": self.configuration_index,
            "instruction_direction": self.instruction_direction,
            "instruction_direction_name": tuple(
                INSTRUCTION_DIRECTION_NAMES[int(value)]
                for value in self.instruction_direction
            ),
            "execution_relation": self.execution_relation,
            "execution_relation_name": tuple(
                EXECUTION_RELATION_NAMES[int(value)]
                for value in self.execution_relation
            ),
            "presented_sequence": self.presented_sequence,
            "effective_execution_sequence": self.effective_execution_sequence,
            "effective_instruction_abstract_sequence": instruction_abstract,
            "effective_instruction_physical_sequence": instruction_physical,
            "effective_execution_abstract_sequence": execution_abstract,
            "effective_execution_physical_sequence": execution_physical,
            "instruction_presentation_index": torch.where(
                self.phase == INSTRUCTION,
                self.instruction_presentation_index,
                torch.full_like(self.instruction_presentation_index, -1),
            ),
            "instruction_index": torch.where(
                self.phase == INSTRUCTION,
                self.instruction_presentation_index,
                torch.full_like(self.instruction_presentation_index, -1),
            ),
            "current_required_abstract_goal_index": required_abstract,
            "current_required_physical_location": self.configuration[
                self.batch_inds, required_abstract
            ],
            "loop_index": self.loop_index,
            "successful_goal_count": self.successful_goal_count,
            "successes": self.successful_goal_count,
            "reward_event": self.reward_event,
            "navigation_step_index": torch.where(
                self.phase == NAVIGATION,
                self.navigation_step_count,
                torch.full_like(self.navigation_step_count, -1),
            ),
            "block_navigation_index": torch.where(
                self.phase == NAVIGATION,
                self.navigation_step_count,
                torch.full_like(self.navigation_step_count, -1),
            ),
            "finished": self.finished,
            "truncated": self.truncated,
        }

    def post_step_metadata(self) -> dict[str, torch.Tensor]:
        """Return metadata for the transition taken after the stored observation."""
        return self._post_step

    def evaluation_metrics(self) -> dict[str, torch.Tensor | float]:
        """Summarize the most recently completed/autonomously truncated blocks."""
        completed = (
            self.successful_goal_count == self.total_required_goals
        ) & ~self.truncated
        return {
            "completed": completed,
            "completion_rate": completed.to(torch.float32).mean(),
            "truncated": self.truncated,
            "successful_goal_count": self.successful_goal_count,
            "total_reward": self.successful_goal_count.to(torch.float32),
            "success_fraction": self.successful_goal_count.to(torch.float32)
            / float(self.total_required_goals),
            "required_goal_count": torch.full(
                (self.batch,), self.total_required_goals, dtype=torch.long
            ),
            "navigation_step_count": self.navigation_step_count,
            "final_location": self.loc,
        }


__all__ = [
    "ABCDFMRIEnv",
    "ACTION_NAMES",
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "PHASE_NAMES",
    "INSTRUCTION",
    "NAVIGATION",
    "REWARD",
    "INSTRUCTION_DIRECTION_NAMES",
    "FORWARD",
    "BACKWARD",
    "EXECUTION_RELATION_NAMES",
    "SAME",
    "REVERSE",
    "DEFAULT_FMRI_BASE_CONFIGURATIONS",
    "OBSERVATION_SLICES",
    "location_to_row_col",
    "row_col_to_location",
    "manhattan_distance",
    "move_location",
    "normalize_configuration",
    "configuration_has_minimum_distance",
    "parse_configurations",
    "configuration_cycle_variants",
    "canonical_configuration_cycle",
    "configuration_bank_statistics",
    "validate_fmri_configuration_bank",
    "generate_synthetic_fmri_configuration_bank",
    "generate_configuration_bank",
    "split_configuration_bank",
]
