"""Pure-function tests for normalized-progress raw ABCD analysis."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "ABCD_task"
    / "analyse_abcd_normalized_raw.py"
)
SPEC = importlib.util.spec_from_file_location("analyse_abcd_normalized_raw", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
raw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(raw)


def _synthetic_normalized_data(seed: int = 19) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_bases = 5
    n_conditions = 4
    n_loops = 4
    n_positions = 4
    n_locations = 4
    configurations = np.asarray(
        [
            [0, 1, 2, 3],
            [1, 3, 0, 2],
            [2, 0, 3, 1],
            [3, 2, 1, 0],
            [0, 2, 3, 1],
        ],
        dtype=np.int16,
    )

    rows = []
    factorial = 0
    for base in range(n_bases):
        for condition in range(n_conditions):
            instruction = condition // 2
            execution = condition % 2
            for loop in range(n_loops):
                for q in range(n_positions):
                    current = (q + base + condition) % n_locations
                    future = np.empty(n_positions, dtype=np.int16)
                    future[0] = current
                    future[1:] = rng.integers(
                        0,
                        n_locations,
                        size=n_positions - 1,
                    )
                    current_action = int(rng.integers(0, 4))
                    next_action = int(rng.integers(0, 4))
                    rows.append(
                        (
                            base,
                            factorial,
                            condition,
                            instruction,
                            execution,
                            loop,
                            q,
                            current,
                            current_action,
                            next_action,
                            configurations[base],
                            future,
                        )
                    )
            factorial += 1

    n = len(rows)
    future_locations = np.stack([row[11] for row in rows])
    q = np.asarray([row[6] for row in rows], dtype=np.int8)
    # Unit 0 has a strong, cross-base-stable effect of horizon-1 location.
    class_effect = np.asarray([-1.5, -0.5, 0.5, 1.5])
    unit0 = class_effect[future_locations[:, 1]] + rng.normal(0.0, 0.08, n)
    # Unit 1 is explained by current progress alone; unit 2 is inactive.
    unit1 = 0.4 * q + rng.normal(0.0, 0.08, n)
    unit2 = np.full(n, 2.0)
    rs = np.column_stack((unit0, unit1, unit2)).astype(np.float32)

    block_size = n_loops * n_positions
    return {
        "rs": rs,
        "future_locations": future_locations,
        "normalized_position": q,
        "current_location": np.asarray([row[7] for row in rows], dtype=np.int16),
        "current_action": np.asarray([row[8] for row in rows], dtype=np.int16),
        "next_action": np.asarray([row[9] for row in rows], dtype=np.int16),
        "loop_index": np.asarray([row[5] for row in rows], dtype=np.int16),
        "repeat_index": np.ones(n, dtype=np.int8),
        "factorial_index": np.asarray([row[1] for row in rows], dtype=np.int16),
        "base_configuration_index": np.asarray([row[0] for row in rows], dtype=np.int16),
        "instruction_direction": np.asarray([row[3] for row in rows], dtype=np.int8),
        "execution_relation": np.asarray([row[4] for row in rows], dtype=np.int8),
        "configuration": np.stack([row[10] for row in rows]),
        "sample_weight": np.full(n, 1.0 / block_size, dtype=np.float64),
        "horizons": np.arange(n_positions, dtype=np.int8),
        "num_locations": np.asarray(n_locations),
        "num_normalized_positions": np.asarray(n_positions),
    }


def test_validation_enforces_common_support_and_horizon_zero_alignment():
    data = _synthetic_normalized_data()
    metadata = raw.validate_raw_inputs(data)
    assert metadata["n_bases"] == 5
    assert metadata["n_horizons"] == 4

    bad_target = {key: np.asarray(value).copy() for key, value in data.items()}
    bad_target["future_locations"][0, 0] = (
        bad_target["future_locations"][0, 0] + 1
    ) % 4
    with pytest.raises(ValueError, match="Horizon zero"):
        raw.validate_raw_inputs(bad_target)

    bad_weight = {key: np.asarray(value).copy() for key, value in data.items()}
    bad_weight["sample_weight"][0] *= 2
    with pytest.raises(ValueError, match="total sample_weight 1"):
        raw.validate_raw_inputs(bad_weight)


def test_weighted_lstsq_is_invariant_to_a_duplicated_column_space():
    rng = np.random.default_rng(4)
    x = np.column_stack((np.ones(80), rng.normal(size=(80, 4))))
    y = rng.normal(size=(80, 3))
    weights = rng.uniform(0.2, 1.3, size=80)
    beta, _, _ = raw.weighted_lstsq(x, y, weights)
    duplicated = np.column_stack((x, x[:, 2]))
    beta_duplicate, _, _ = raw.weighted_lstsq(duplicated, y, weights)
    np.testing.assert_allclose(
        x @ beta,
        duplicated @ beta_duplicate,
        rtol=1e-11,
        atol=1e-11,
    )


def test_crossfit_detects_future_effect_and_keeps_inactive_units_zero():
    result = raw.crossfit_raw_representation(_synthetic_normalized_data())
    partial = np.asarray(result["partial_r2"])
    native_qc = np.asarray(result["native_delta_rmse"])

    np.testing.assert_array_equal(partial[0], np.zeros(3))
    np.testing.assert_array_equal(native_qc[0], np.zeros(3))
    assert partial[1, 0] > 0.9
    assert partial[1, 0] > partial[1, 1] + 0.5
    np.testing.assert_array_equal(partial[:, 2], np.zeros(4))
    np.testing.assert_array_equal(native_qc[:, 2], np.zeros(4))
    assert np.asarray(result["unit_valid_fold_count"])[2] == 0


def test_heldout_activity_cannot_change_fitted_predictions_or_scaler():
    data = _synthetic_normalized_data()
    heldout = np.asarray(data["base_configuration_index"]) == 0
    first = raw.fit_raw_fold(data, ~heldout, heldout, return_predictions=True)

    changed = {key: np.asarray(value).copy() for key, value in data.items()}
    changed["rs"][heldout] += 10_000.0
    second = raw.fit_raw_fold(changed, ~heldout, heldout, return_predictions=True)

    np.testing.assert_array_equal(first["train_mean"], second["train_mean"])
    np.testing.assert_array_equal(first["train_scale"], second["train_scale"])
    np.testing.assert_allclose(
        first["nuisance_prediction"],
        second["nuisance_prediction"],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        first["full_prediction"],
        second["full_prediction"],
        rtol=0.0,
        atol=0.0,
    )
