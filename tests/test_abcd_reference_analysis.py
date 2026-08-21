"""Focused indexing, leakage, Csubs-map, and RSA tests."""

from __future__ import annotations

import numpy as np

from scripts.ABCD_task.abcd_analysis_common import (
    build_normalized_navigation,
    concatenate_normalized_blocks,
    midpoint_phase_indices,
    validate_normalized_navigation,
)
from scripts.ABCD_task.analyse_abcd_local_rsa import (
    ConditionDesign,
    _future_sequences,
    build_model_rdms,
    cross_repeat_spearman_rdm,
    run_local_rsa,
    summarize_beta_map,
    valid_rdm_entries,
)
from scripts.ABCD_task.analyse_abcd_normalized_csubs import (
    class_center_filter_map,
    haufe_logit_pattern_map,
    trajectory_label_null,
)


def _variable_leg_trajectory(execution=(3, 2, 1, 0), n_units=3):
    lengths = [1 + (leg % 4) for leg in range(20)]
    success = np.concatenate(
        [np.full(length, leg, dtype=int) for leg, length in enumerate(lengths)]
    )
    n = len(success)
    goal = np.concatenate(
        [
            np.full(length, execution[leg % 4], dtype=int)
            for leg, length in enumerate(lengths)
        ]
    )
    target = (goal * 2 + 1) % 9
    target_reached = np.zeros(n, dtype=bool)
    offset = 0
    for length in lengths:
        target_reached[offset + length - 1] = True
        offset += length
    pre = np.arange(n, dtype=int) % 9
    post = (pre + 1) % 9
    # Navigation-only continuity is not required by the normalizer itself; the
    # source indices make the exact normalized target construction observable.
    activity = np.column_stack(
        [np.arange(n, dtype=float) + 100 * unit for unit in range(n_units)]
    ).astype(np.float32)
    trajectory = {
        "num_navigation_steps": n,
        "rs": activity,
        "zs": activity + 0.5,
        "pre_location": pre,
        "post_location": post,
        "env_action": np.arange(n) % 4,
        "model_action": np.arange(n) % 4,
        "successful_goal_count": success,
        "current_required_abstract_goal_index": goal,
        "target_reached": target_reached,
        "current_required_physical_location": target,
        "store_timestep_index": np.arange(n) * 2,
        "navigation_index": np.arange(n),
    }
    metadata = {
        "factorial_index": 7,
        "base_configuration_index": 2,
        "instruction_direction": 1,
        "execution_relation": 0,
        "base_configuration": (0, 2, 4, 8),
        "effective_execution_abstract_sequence": execution,
    }
    return trajectory, metadata, lengths


def test_midpoint_progress_and_continuous_common_support():
    np.testing.assert_array_equal(midpoint_phase_indices(1), [0, 0, 0])
    np.testing.assert_array_equal(midpoint_phase_indices(2), [0, 1, 1])
    np.testing.assert_array_equal(midpoint_phase_indices(4), [0, 2, 3])

    trajectory, metadata, _ = _variable_leg_trajectory()
    block = build_normalized_navigation(
        trajectory, metadata, repeat_index=1, expected_loops=5
    )
    assert block["rs"].shape == (48, 3)
    np.testing.assert_array_equal(block["q"], np.tile(np.arange(12), 4))
    # Reverse execution keeps q ordinal while preserving literal goal identity.
    np.testing.assert_array_equal(block["abstract_goal"][:12], np.repeat([3, 2, 1, 0], 3))
    np.testing.assert_array_equal(
        block["future_normalized_position"],
        (block["q"][:, None] + np.arange(12)[None, :]) % 12,
    )
    np.testing.assert_array_equal(
        block["future_locations"],
        trajectory["pre_location"][block["future_source_index"] // 2],
    )
    assert np.max(block["future_loop_index"]) == 4
    assert np.all(block["future_valid"])

    combined = concatenate_normalized_blocks([block], num_locations=9)
    validate_normalized_navigation(combined)
    assert np.isclose(np.sum(combined["sample_weight"]), 1.0)


def test_csubs_gauge_inactive_units_haufe_and_null_are_well_defined():
    rng = np.random.default_rng(12)
    coefficients = rng.normal(size=(12, 9, 10))
    active = np.asarray([0, 2, 3, 4, 5, 6, 7, 8, 9, 11])
    score, normalized, centered = class_center_filter_map(
        coefficients, n_full_units=12, active_indices=active
    )
    assert score.shape == (12, 12)
    np.testing.assert_array_equal(score[:, [1, 10]], 0.0)
    np.testing.assert_allclose(np.mean(centered, axis=1), 0.0, atol=1e-12)
    # Adding the same softmax gauge vector to every class must not alter map.
    gauge = rng.normal(size=(12, 1, 10))
    score_gauge, _, _ = class_center_filter_map(
        coefficients + gauge, n_full_units=12, active_indices=active
    )
    np.testing.assert_allclose(score, score_gauge, atol=1e-12)
    assert normalized.shape == (12, 9, 12)

    x = rng.normal(size=(300, 10))
    haufe, rank, condition = haufe_logit_pattern_map(
        x, centered, n_full_units=12, active_indices=active
    )
    assert np.all(rank == 8)
    assert np.all(np.isfinite(condition))
    np.testing.assert_array_equal(haufe[:, [1, 10]], 0.0)
    np.testing.assert_allclose(np.linalg.norm(haufe, axis=1), 1.0, atol=1e-10)

    labels = (
        np.arange(90, dtype=int)[:, None]
        + np.arange(12, dtype=int)[None, :]
    ) % 9
    predictions = labels.copy()
    null_first = trajectory_label_null(
        labels, predictions, n_permutations=40, seed=44
    )
    null_second = trajectory_label_null(
        labels, predictions, n_permutations=40, seed=44
    )
    np.testing.assert_array_equal(null_first["null"], null_second["null"])
    assert np.all(null_first["observed"] == 1.0)
    assert np.all(null_first["p_value"] > 0)


def test_rsa_split_targets_entry_selection_and_cross_repeat_symmetry():
    cycle = np.arange(12) % 9
    path = _future_sequences(
        normalized_cycle=cycle,
        execution_order_position=2,
        is_reward=False,
        rewarded_location=4,
    )
    reward = _future_sequences(
        normalized_cycle=cycle,
        execution_order_position=2,
        is_reward=True,
        rewarded_location=4,
    )
    assert path.shape == reward.shape == (12, 12)
    np.testing.assert_array_equal(reward[0], np.full(12, 4))
    np.testing.assert_array_equal(reward[1], np.full(12, cycle[9]))
    assert not np.array_equal(path[1], path[0])

    is_reward = np.asarray([False, False, True, True])
    mask = valid_rdm_entries(is_reward)
    assert int(np.sum(mask)) == 2
    assert mask[0, 1] and mask[2, 3]
    assert not mask[0, 2]

    rng = np.random.default_rng(7)
    first = rng.normal(size=(8, 10))
    second = rng.normal(size=(8, 10))
    rdm = cross_repeat_spearman_rdm(first, second)
    np.testing.assert_allclose(rdm, rdm.T, atol=1e-12)
    assert np.all((rdm >= 0) & (rdm <= 2))

    patterns = rng.normal(size=(2, 8, 10))
    model_feature = np.arange(8, dtype=float)
    model_rdm = np.abs(model_feature[:, None] - model_feature[None, :])[None, ...]
    upper = np.triu(np.ones((8, 8), dtype=bool), 1)
    lights = tuple(np.arange(10, dtype=int) for _ in range(10))
    beta, fit_r2, finite_count, rank, _ = run_local_rsa(
        patterns, lights, model_rdm, upper
    )
    assert beta.shape == (1, 10)
    assert fit_r2.shape == finite_count.shape == (10,)
    assert rank == 2
    assert np.all(np.isfinite(beta))

    design = ConditionDesign(
        patterns=np.zeros((2, 4, 2)),
        base_index=np.zeros(4, dtype=int),
        instruction_direction=np.zeros(4, dtype=int),
        execution_relation=np.zeros(4, dtype=int),
        abstract_goal=np.asarray([0, 1, 0, 2]),
        is_reward=np.asarray([True, True, False, True]),
        configuration=np.tile(np.arange(4), (4, 1)),
        execution_order_position=np.arange(4),
        current_location_sequence=np.tile(np.arange(4)[:, None], (1, 12)),
        current_action_sequence=np.tile(np.arange(4)[:, None], (1, 12)),
        next_action_sequence=np.tile(np.arange(1, 5)[:, None], (1, 12)),
        future_location_sequence=np.tile(
            (np.arange(4)[:, None, None] + np.arange(12)[None, :, None]) % 9,
            (1, 1, 12),
        ),
        condition_labels=("reward_A", "reward_B", "path_A", "reward_C"),
        n_loops=5,
    )
    model_rdms, names = build_model_rdms(design)
    reward_a = design.is_reward & (design.abstract_goal == 0)
    expected_feedback = (reward_a[:, None] != reward_a[None, :]).astype(float)
    feedback_index = names.index("reward_a_feedback")
    np.testing.assert_array_equal(model_rdms[feedback_index], expected_feedback)
    assert model_rdms[feedback_index, 1, 3] == 0.0


def test_rsa_primary_com_matches_strict_percentile_surface_components():
    beta = np.asarray([0.0, 1.0, 2.0, 10.0, 9.0, -1.0])
    coords = np.column_stack((np.zeros(6), np.zeros(6), np.arange(6)))
    adjacency = (
        np.asarray([1]),
        np.asarray([0, 2]),
        np.asarray([1]),
        np.asarray([4]),
        np.asarray([3]),
        np.asarray([], dtype=int),
    )
    anchor = np.arange(6, dtype=float)
    result = summarize_beta_map(
        beta, coords, adjacency, anchor, percentile=60.0
    )
    np.testing.assert_array_equal(result["strongest_units"], [3, 4])
    expected_z = (10 * 3 + 9 * 4) / 19
    assert np.isclose(result["primary_surface_z"], expected_z)
