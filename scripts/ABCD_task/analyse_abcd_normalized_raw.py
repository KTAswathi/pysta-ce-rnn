"""Nuisance-controlled raw-activity analysis on normalized ABCD progress.

This module deliberately contains one primary estimator and one native-scale
QC.  The primary quantity is a signed, out-of-fold partial R-squared from a
nested encoding comparison.  It is computed with leave-one-base-
configuration-out folds and equal block weights.

The neural observation is decision-time recurrent firing after the current
observation has entered the frozen RNN and before its current navigation
action is applied.  At horizon ``h``, the added predictor is future physical
location at ``h`` normalized task-progress advances.  The nuisance model uses
only current-time variables, except for next action, which is the explicit
motor control used by the human reference analysis.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ROW_FIELDS = (
    "rs",
    "future_locations",
    "normalized_position",
    "current_location",
    "current_action",
    "next_action",
    "loop_index",
    "repeat_index",
    "factorial_index",
    "base_configuration_index",
    "instruction_direction",
    "execution_relation",
    "configuration",
    "sample_weight",
)

OPTIONAL_ROW_FIELDS = (
    "source_index",
    "abstract_goal",
    "phase",
    "block_id",
)

DEFINITION = {
    "input_observation": (
        "decision-time recurrent firing r after the current observation and "
        "before the current navigation action, sampled on the common "
        "task-normalized progress grid"
    ),
    "target_model": (
        "for each unit and normalized horizon, compare weighted linear "
        "encoding models with and without one-hot future physical location"
    ),
    "nuisance_variables": (
        "current physical location; current and next realised action; "
        "normalized state-by-phase position; loop; frozen-model repeat; "
        "instruction direction; execution relation; and the four abstract-"
        "goal-to-physical-location configuration assignments"
    ),
    "crossfit_unit": (
        "leave one base configuration out, holding out every factorial block "
        "and frozen-model repeat for that base; each repeat-block has equal "
        "total weight"
    ),
    "primary_quantity": (
        "signed cross-fitted partial R2 = 1 - weighted_SSE_full / "
        "weighted_SSE_nuisance on training-fold-standardized firing"
    ),
    "native_qc": (
        "signed held-out delta-RMSE = RMSE_nuisance - RMSE_full after "
        "returning predictions to native firing-rate units"
    ),
    "interpretation": (
        "one primary value is the cross-configuration fractional held-out "
        "prediction-error reduction for one unit attributable to future "
        "physical location at one normalized horizon beyond the listed "
        "additive nuisances; it is predictive, not unique causal information"
    ),
}


def _require_row_length(
    name: str,
    value: np.ndarray,
    n_observations: int,
) -> None:
    if value.ndim == 0 or value.shape[0] != n_observations:
        raise ValueError(
            f"{name} must have first dimension {n_observations}, got "
            f"{value.shape}."
        )


def _block_keys(data: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return repeat-by-factorial keys for independently collected blocks."""
    return np.column_stack(
        (
            np.asarray(data["repeat_index"], dtype=np.int64),
            np.asarray(data["factorial_index"], dtype=np.int64),
        )
    )


def validate_raw_inputs(data: Mapping[str, np.ndarray]) -> dict[str, int]:
    """Validate the common normalized-navigation schema used by this analysis."""
    missing = [name for name in (*ROW_FIELDS, "horizons") if name not in data]
    if missing:
        raise KeyError(f"Normalized navigation data are missing fields: {missing}")

    rs = np.asarray(data["rs"])
    future = np.asarray(data["future_locations"])
    horizons = np.asarray(data["horizons"], dtype=np.int64)
    configuration = np.asarray(data["configuration"])

    if rs.ndim != 2 or rs.shape[0] == 0 or rs.shape[1] == 0:
        raise ValueError(f"rs must be a non-empty [observation, unit] array, got {rs.shape}.")
    if not np.all(np.isfinite(rs)):
        raise ValueError("rs contains non-finite values.")

    n_observations, n_units = rs.shape
    if future.ndim != 2 or future.shape[0] != n_observations:
        raise ValueError(
            "future_locations must have shape [observation, horizon], got "
            f"{future.shape}."
        )
    if horizons.ndim != 1 or len(horizons) != future.shape[1]:
        raise ValueError("horizons does not match the future-location axis.")
    if not np.array_equal(horizons, np.arange(len(horizons))):
        raise ValueError("horizons must enumerate normalized advances from zero.")
    if configuration.ndim != 2 or configuration.shape[0] != n_observations:
        raise ValueError(
            "configuration must have shape [observation, abstract_goal], got "
            f"{configuration.shape}."
        )

    for name in ROW_FIELDS:
        _require_row_length(name, np.asarray(data[name]), n_observations)
    for name in OPTIONAL_ROW_FIELDS:
        if name in data:
            _require_row_length(name, np.asarray(data[name]), n_observations)

    n_positions = int(np.asarray(data.get("num_normalized_positions", len(horizons))))
    if n_positions != len(horizons):
        raise ValueError(
            "num_normalized_positions must equal the number of horizons; got "
            f"{n_positions} and {len(horizons)}."
        )
    q = np.asarray(data["normalized_position"], dtype=np.int64)
    if np.any((q < 0) | (q >= n_positions)):
        raise ValueError("normalized_position contains an out-of-range value.")

    inferred_locations = int(
        max(
            np.max(configuration),
            np.max(future),
            np.max(np.asarray(data["current_location"])),
        )
        + 1
    )
    n_locations = int(np.asarray(data.get("num_locations", inferred_locations)))
    if n_locations < inferred_locations:
        raise ValueError("num_locations is smaller than an observed location label.")
    for name in ("current_location", "future_locations", "configuration"):
        values = np.asarray(data[name], dtype=np.int64)
        if np.any((values < 0) | (values >= n_locations)):
            raise ValueError(f"{name} contains an out-of-range physical location.")

    current = np.asarray(data["current_location"], dtype=np.int64)
    if not np.array_equal(future[:, 0], current):
        raise ValueError(
            "Horizon zero must equal current physical location exactly; the "
            "normalized target alignment is inconsistent."
        )

    weights = np.asarray(data["sample_weight"], dtype=np.float64)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("sample_weight must be finite and strictly positive.")

    keys = _block_keys(data)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    block_sums = np.bincount(inverse, weights=weights)
    if not np.allclose(block_sums, 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError(
            "Each repeat-by-factorial block must have total sample_weight 1; "
            f"observed range {block_sums.min():.9g}..{block_sums.max():.9g}."
        )

    base = np.asarray(data["base_configuration_index"], dtype=np.int64)
    if len(np.unique(base)) < 2:
        raise ValueError("Leave-one-base-out cross-fitting requires at least two bases.")

    # Every retained neural loop must contribute one observation at every q.
    loop_keys = np.column_stack(
        (
            keys,
            np.asarray(data["loop_index"], dtype=np.int64),
        )
    )
    _, loop_inverse = np.unique(loop_keys, axis=0, return_inverse=True)
    for group in range(int(loop_inverse.max()) + 1):
        observed = np.sort(q[loop_inverse == group])
        if not np.array_equal(observed, np.arange(n_positions)):
            raise ValueError(
                "Every retained repeat/block/loop must contain each normalized "
                "position exactly once."
            )

    return {
        "n_observations": n_observations,
        "n_units": n_units,
        "n_horizons": len(horizons),
        "n_positions": n_positions,
        "n_locations": n_locations,
        "n_abstract_goals": configuration.shape[1],
        "n_bases": len(np.unique(base)),
        "n_blocks": len(unique_keys),
    }


def concatenate_normalized_inputs(
    datasets: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Concatenate independently validated repeat stores without changing rows."""
    if not datasets:
        raise ValueError("At least one normalized-navigation dataset is required.")

    prepared = [{key: np.asarray(value) for key, value in item.items()} for item in datasets]
    for item in prepared:
        validate_raw_inputs(item)

    horizons = prepared[0]["horizons"]
    for item in prepared[1:]:
        if not np.array_equal(item["horizons"], horizons):
            raise ValueError("Repeat stores have different normalized horizons.")

    row_names = list(ROW_FIELDS)
    row_names.extend(
        name for name in OPTIONAL_ROW_FIELDS if all(name in item for item in prepared)
    )
    combined = {
        name: np.concatenate([item[name] for item in prepared], axis=0)
        for name in row_names
    }
    combined["horizons"] = np.asarray(horizons).copy()
    for scalar_name in ("num_locations", "num_normalized_positions"):
        present = [item[scalar_name] for item in prepared if scalar_name in item]
        if present:
            scalar_values = [int(np.asarray(value)) for value in present]
            if len(set(scalar_values)) != 1:
                raise ValueError(f"Repeat stores disagree on {scalar_name}.")
            combined[scalar_name] = np.asarray(scalar_values[0])

    validate_raw_inputs(combined)
    return combined


def _one_hot(values: np.ndarray, categories: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    categories = np.asarray(categories)
    return (values[:, None] == categories[None, :]).astype(np.float64)


def fit_nuisance_encoder(
    train: Mapping[str, np.ndarray],
    *,
    n_locations: int,
    n_positions: int,
) -> dict[str, object]:
    """Fit categorical levels on training observations only."""
    return {
        "n_locations": int(n_locations),
        "n_positions": int(n_positions),
        "current_action": np.unique(np.asarray(train["current_action"], dtype=np.int64)),
        "next_action": np.unique(np.asarray(train["next_action"], dtype=np.int64)),
        "loop_index": np.unique(np.asarray(train["loop_index"], dtype=np.int64)),
        "repeat_index": np.unique(np.asarray(train["repeat_index"], dtype=np.int64)),
        "instruction_direction": np.unique(
            np.asarray(train["instruction_direction"], dtype=np.int64)
        ),
        "execution_relation": np.unique(
            np.asarray(train["execution_relation"], dtype=np.int64)
        ),
        "n_abstract_goals": int(np.asarray(train["configuration"]).shape[1]),
    }


def transform_nuisance_design(
    data: Mapping[str, np.ndarray],
    encoder: Mapping[str, object],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Construct the fixed additive nuisance design described in DEFINITION."""
    n = len(np.asarray(data["current_location"]))
    n_locations = int(encoder["n_locations"])
    n_positions = int(encoder["n_positions"])
    pieces = [np.ones((n, 1), dtype=np.float64)]
    names = ["intercept"]

    def append_categorical(field: str, categories: np.ndarray) -> None:
        values = np.asarray(data[field], dtype=np.int64)
        pieces.append(_one_hot(values, categories))
        names.extend(f"{field}={int(category)}" for category in categories)

    append_categorical("current_location", np.arange(n_locations))
    append_categorical("current_action", np.asarray(encoder["current_action"]))
    append_categorical("next_action", np.asarray(encoder["next_action"]))
    append_categorical("normalized_position", np.arange(n_positions))
    append_categorical("loop_index", np.asarray(encoder["loop_index"]))
    append_categorical("repeat_index", np.asarray(encoder["repeat_index"]))
    append_categorical(
        "instruction_direction", np.asarray(encoder["instruction_direction"])
    )
    append_categorical(
        "execution_relation", np.asarray(encoder["execution_relation"])
    )

    configuration = np.asarray(data["configuration"], dtype=np.int64)
    n_abstract_goals = int(encoder["n_abstract_goals"])
    if configuration.shape[1] != n_abstract_goals:
        raise ValueError("Configuration width differs from the fitted encoder.")
    location_categories = np.arange(n_locations)
    for goal in range(n_abstract_goals):
        pieces.append(_one_hot(configuration[:, goal], location_categories))
        names.extend(
            f"configuration_goal{goal}={int(location)}"
            for location in location_categories
        )

    design = np.concatenate(pieces, axis=1)
    if not np.all(np.isfinite(design)):
        raise ValueError("The nuisance design contains non-finite values.")
    return design, tuple(names)


def weighted_mean_and_scale(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    activity_tolerance: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return training-weighted mean, SD and active-unit mask."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 2 or weights.shape != (values.shape[0],):
        raise ValueError("values/weights shapes are inconsistent.")
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Training weights must have positive finite total.")
    mean = np.sum(values * weights[:, None], axis=0) / total
    centered = values - mean
    variance = np.sum(centered * centered * weights[:, None], axis=0) / total
    scale = np.sqrt(np.maximum(variance, 0.0))
    active = scale > activity_tolerance
    safe_scale = np.where(active, scale, 1.0)
    return mean, safe_scale, active


def weighted_lstsq(
    design: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    *,
    rcond: float | None = None,
) -> tuple[np.ndarray, int, float]:
    """Fit an unpenalized weighted least-squares column space.

    A rank-revealing least-squares fit is intentional: unlike ridge, adding a
    duplicated horizon-zero location column cannot change the fitted column
    space merely by splitting a penalty across duplicate coefficients.
    """
    design = np.asarray(design, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if design.ndim != 2 or targets.ndim != 2:
        raise ValueError("design and targets must both be two-dimensional.")
    if design.shape[0] != targets.shape[0] or weights.shape != (design.shape[0],):
        raise ValueError("Weighted least-squares input shapes are inconsistent.")
    root_weight = np.sqrt(weights)
    weighted_design = design * root_weight[:, None]
    weighted_targets = targets * root_weight[:, None]
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        weighted_design,
        weighted_targets,
        rcond=rcond,
    )
    if rank == 0:
        condition = np.inf
    else:
        retained = singular_values[:rank]
        condition = float(retained[0] / retained[-1])
    return coefficients, int(rank), condition


def _subset(data: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(value)[mask]
        for name, value in data.items()
        if np.asarray(value).ndim > 0 and np.asarray(value).shape[0] == len(mask)
    }


def fit_raw_fold(
    data: Mapping[str, np.ndarray],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    activity_tolerance: float = 1e-12,
    return_predictions: bool = False,
) -> dict[str, np.ndarray | float | int | tuple[str, ...]]:
    """Fit one outer fold without consulting held-out activity or labels."""
    metadata = validate_raw_inputs(data)
    train_mask = np.asarray(train_mask, dtype=bool)
    test_mask = np.asarray(test_mask, dtype=bool)
    n = metadata["n_observations"]
    if train_mask.shape != (n,) or test_mask.shape != (n,):
        raise ValueError("Fold masks have the wrong shape.")
    if np.any(train_mask & test_mask) or not np.any(train_mask) or not np.any(test_mask):
        raise ValueError("Train/test folds must be non-empty and disjoint.")

    train = _subset(data, train_mask)
    test = _subset(data, test_mask)
    encoder = fit_nuisance_encoder(
        train,
        n_locations=metadata["n_locations"],
        n_positions=metadata["n_positions"],
    )
    x_train, design_names = transform_nuisance_design(train, encoder)
    x_test, test_names = transform_nuisance_design(test, encoder)
    if design_names != test_names:
        raise AssertionError("Train and test nuisance designs are misaligned.")

    y_train_native = np.asarray(train["rs"], dtype=np.float64)
    y_test_native = np.asarray(test["rs"], dtype=np.float64)
    w_train = np.asarray(train["sample_weight"], dtype=np.float64)
    w_test = np.asarray(test["sample_weight"], dtype=np.float64)
    train_mean, train_scale, active = weighted_mean_and_scale(
        y_train_native,
        w_train,
        activity_tolerance=activity_tolerance,
    )
    y_train = (y_train_native - train_mean) / train_scale
    y_test = (y_test_native - train_mean) / train_scale
    y_train[:, ~active] = 0.0

    nuisance_coef, nuisance_rank, nuisance_condition = weighted_lstsq(
        x_train,
        y_train,
        w_train,
    )
    nuisance_prediction = x_test @ nuisance_coef
    n_horizons = metadata["n_horizons"]
    n_units = metadata["n_units"]
    nuisance_sse_one = np.sum(
        w_test[:, None] * (y_test - nuisance_prediction) ** 2,
        axis=0,
    )
    nuisance_sse = np.repeat(nuisance_sse_one[None, :], n_horizons, axis=0)
    full_sse = np.empty((n_horizons, n_units), dtype=np.float64)
    nuisance_native_sse_one = np.sum(
        w_test[:, None]
        * (y_test_native - (nuisance_prediction * train_scale + train_mean)) ** 2,
        axis=0,
    )
    nuisance_native_sse = np.repeat(
        nuisance_native_sse_one[None, :], n_horizons, axis=0
    )
    full_native_sse = np.empty_like(full_sse)
    full_rank = np.empty(n_horizons, dtype=np.int64)
    full_condition = np.empty(n_horizons, dtype=np.float64)
    predictions = []

    future_train = np.asarray(train["future_locations"], dtype=np.int64)
    future_test = np.asarray(test["future_locations"], dtype=np.int64)
    current_train = np.asarray(train["current_location"], dtype=np.int64)
    current_test = np.asarray(test["current_location"], dtype=np.int64)

    for horizon_index in range(n_horizons):
        if horizon_index == 0:
            if not (
                np.array_equal(future_train[:, 0], current_train)
                and np.array_equal(future_test[:, 0], current_test)
            ):
                raise AssertionError("Horizon-zero target/current-location invariant failed.")
            # The added columns duplicate a nuisance term exactly.  Copying the
            # nuisance predictions makes this scientific null exact and avoids
            # a platform-dependent least-squares roundoff difference.
            full_prediction = nuisance_prediction.copy()
            full_rank[horizon_index] = nuisance_rank
            full_condition[horizon_index] = nuisance_condition
        else:
            target_train = _one_hot(
                future_train[:, horizon_index],
                np.arange(metadata["n_locations"]),
            )
            target_test = _one_hot(
                future_test[:, horizon_index],
                np.arange(metadata["n_locations"]),
            )
            full_train = np.concatenate((x_train, target_train), axis=1)
            full_test = np.concatenate((x_test, target_test), axis=1)
            full_coef, rank, condition = weighted_lstsq(
                full_train,
                y_train,
                w_train,
            )
            full_prediction = full_test @ full_coef
            full_rank[horizon_index] = rank
            full_condition[horizon_index] = condition

        full_sse[horizon_index] = np.sum(
            w_test[:, None] * (y_test - full_prediction) ** 2,
            axis=0,
        )
        full_prediction_native = full_prediction * train_scale + train_mean
        full_native_sse[horizon_index] = np.sum(
            w_test[:, None] * (y_test_native - full_prediction_native) ** 2,
            axis=0,
        )
        if return_predictions:
            predictions.append(full_prediction)

    fold_partial_r2 = np.zeros_like(full_sse)
    estimable = active[None, :] & (nuisance_sse > activity_tolerance)
    np.divide(
        nuisance_sse - full_sse,
        nuisance_sse,
        out=fold_partial_r2,
        where=estimable,
    )
    test_weight = float(np.sum(w_test))
    fold_native_qc = (
        np.sqrt(nuisance_native_sse / test_weight)
        - np.sqrt(full_native_sse / test_weight)
    )

    result: dict[str, np.ndarray | float | int | tuple[str, ...]] = {
        "nuisance_sse": nuisance_sse,
        "full_sse": full_sse,
        "nuisance_native_sse": nuisance_native_sse,
        "full_native_sse": full_native_sse,
        "partial_r2": fold_partial_r2,
        "native_delta_rmse": fold_native_qc,
        "active_units": active,
        "train_mean": train_mean,
        "train_scale": train_scale,
        "test_weight": test_weight,
        "nuisance_rank": nuisance_rank,
        "nuisance_condition": nuisance_condition,
        "full_rank": full_rank,
        "full_condition": full_condition,
        "design_names": design_names,
    }
    if return_predictions:
        result["nuisance_prediction"] = nuisance_prediction
        result["full_prediction"] = np.stack(predictions, axis=0)
        result["test_standardized_activity"] = y_test
    return result


def crossfit_raw_representation(
    data: Mapping[str, np.ndarray],
    *,
    activity_tolerance: float = 1e-12,
) -> dict[str, np.ndarray | str]:
    """Run the single pre-specified leave-one-base-out raw estimator."""
    metadata = validate_raw_inputs(data)
    bases = np.unique(np.asarray(data["base_configuration_index"], dtype=np.int64))
    base_values = np.asarray(data["base_configuration_index"], dtype=np.int64)
    fold_results = []

    for heldout_base in bases:
        test_mask = base_values == heldout_base
        train_mask = ~test_mask
        result = fit_raw_fold(
            data,
            train_mask,
            test_mask,
            activity_tolerance=activity_tolerance,
        )
        fold_results.append(result)

    fold_nuisance_sse = np.stack([item["nuisance_sse"] for item in fold_results])
    fold_full_sse = np.stack([item["full_sse"] for item in fold_results])
    fold_nuisance_native_sse = np.stack(
        [item["nuisance_native_sse"] for item in fold_results]
    )
    fold_full_native_sse = np.stack(
        [item["full_native_sse"] for item in fold_results]
    )
    fold_partial_r2 = np.stack([item["partial_r2"] for item in fold_results])
    fold_native_qc = np.stack(
        [item["native_delta_rmse"] for item in fold_results]
    )
    fold_active = np.stack([item["active_units"] for item in fold_results])
    fold_test_weight = np.asarray([item["test_weight"] for item in fold_results])

    active_3d = fold_active[:, None, :]
    nuisance_sse = np.sum(np.where(active_3d, fold_nuisance_sse, 0.0), axis=0)
    full_sse = np.sum(np.where(active_3d, fold_full_sse, 0.0), axis=0)
    primary = np.zeros_like(nuisance_sse)
    estimable = nuisance_sse > activity_tolerance
    np.divide(
        nuisance_sse - full_sse,
        nuisance_sse,
        out=primary,
        where=estimable,
    )

    nuisance_native_sse = np.sum(
        np.where(active_3d, fold_nuisance_native_sse, 0.0), axis=0
    )
    full_native_sse = np.sum(
        np.where(active_3d, fold_full_native_sse, 0.0), axis=0
    )
    weight_per_unit = np.sum(
        fold_test_weight[:, None] * fold_active,
        axis=0,
    )
    nuisance_native_rmse = np.sqrt(
        np.divide(
            nuisance_native_sse,
            weight_per_unit[None, :],
            out=np.zeros_like(nuisance_native_sse),
            where=weight_per_unit[None, :] > 0,
        )
    )
    full_native_rmse = np.sqrt(
        np.divide(
            full_native_sse,
            weight_per_unit[None, :],
            out=np.zeros_like(full_native_sse),
            where=weight_per_unit[None, :] > 0,
        )
    )
    native_qc = nuisance_native_rmse - full_native_rmse

    if not np.array_equal(primary[0], np.zeros(metadata["n_units"])):
        raise AssertionError("Horizon-zero partial R2 must be exactly zero.")
    if not np.array_equal(native_qc[0], np.zeros(metadata["n_units"])):
        raise AssertionError("Horizon-zero native delta-RMSE must be zero.")

    return {
        "horizons": np.asarray(data["horizons"], dtype=np.int64),
        "partial_r2": primary,
        "native_delta_rmse": native_qc,
        "fold_partial_r2": fold_partial_r2,
        "fold_native_delta_rmse": fold_native_qc,
        "fold_nuisance_sse": fold_nuisance_sse,
        "fold_full_sse": fold_full_sse,
        "fold_nuisance_native_sse": fold_nuisance_native_sse,
        "fold_full_native_sse": fold_full_native_sse,
        "fold_active_units": fold_active,
        "fold_test_weight": fold_test_weight,
        "heldout_base_configuration": bases,
        "nuisance_rank": np.asarray([item["nuisance_rank"] for item in fold_results]),
        "nuisance_condition": np.asarray(
            [item["nuisance_condition"] for item in fold_results]
        ),
        "full_rank": np.stack([item["full_rank"] for item in fold_results]),
        "full_condition": np.stack(
            [item["full_condition"] for item in fold_results]
        ),
        "unit_valid_fold_count": np.sum(fold_active, axis=0),
        "definition_json": json.dumps(DEFINITION, sort_keys=True),
    }


def _load_common_module():
    """Load shared utilities without requiring scripts to be a package."""
    module_path = SCRIPT_DIR / "abcd_analysis_common.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            f"Shared normalized-navigation loader is missing: {module_path}"
        )
    spec = importlib.util.spec_from_file_location("abcd_analysis_common", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_common_navigation(path: Path) -> dict[str, np.ndarray]:
    return _load_common_module().load_normalized_navigation(path)


def weighted_positive_coordinate(
    score: np.ndarray,
    coordinate: np.ndarray,
) -> np.ndarray:
    """Positive-effect-mass coordinate for each normalized horizon.

    The signed partial-R2 map remains the primary saved unit quantity.  Spatial
    location is defined only for validated positive incremental effects, in the
    same spirit as the positive human beta-map COM; negative cross-validated
    values are not reinterpreted as representational mass.
    """
    values = np.maximum(np.asarray(score, dtype=float), 0.0)
    coordinate = np.asarray(coordinate, dtype=float).reshape(-1)
    if values.ndim != 2 or values.shape[1] != len(coordinate):
        raise ValueError("Unit score and cortical coordinate shapes differ.")
    mass = np.sum(values, axis=1)
    return np.divide(
        np.sum(values * coordinate[None, :], axis=1),
        mass,
        out=np.full(values.shape[0], np.nan),
        where=mass > 0,
    )


def discover_normalized_inputs(path: Path) -> list[Path]:
    """Resolve normalized stores from collection, analysis, run, or checkpoint."""
    path = path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() == ".npz":
            return [path]
        analysis_root = _load_common_module().resolve_existing_analysis_root(path)
        path = analysis_root / "trial_collection"
    direct = path / "normalized_navigation.npz"
    if direct.is_file():
        return [direct]
    found = sorted(path.glob("repeat_*/normalized_navigation.npz"))
    if not found and (path / "analysis_manifest.json").is_file():
        path = path / "trial_collection"
        found = sorted(path.glob("repeat_*/normalized_navigation.npz"))
    if not found:
        try:
            analysis_root = _load_common_module().resolve_existing_analysis_root(path)
        except (FileNotFoundError, ValueError):
            analysis_root = None
        if analysis_root is not None:
            path = analysis_root / "trial_collection"
            found = sorted(path.glob("repeat_*/normalized_navigation.npz"))
    if not found:
        raise FileNotFoundError(
            f"No normalized_navigation.npz stores found below {path}."
        )
    return found


def save_summary_csv(path: Path, result: Mapping[str, np.ndarray | str]) -> None:
    horizons = np.asarray(result["horizons"])
    primary = np.asarray(result["partial_r2"], dtype=float)
    native = np.asarray(result["native_delta_rmse"], dtype=float)
    surface_z = np.asarray(result["positive_mass_surface_z"], dtype=float)
    anchor_distance = np.asarray(
        result["positive_mass_anchor_distance"], dtype=float
    )
    valid_count = np.asarray(result["unit_valid_fold_count"], dtype=int)
    active = valid_count > 0
    rows = []
    for index, horizon in enumerate(horizons):
        values = primary[index, active]
        qc_values = native[index, active]
        rows.append(
            {
                "normalized_horizon": int(horizon),
                "n_units_estimable": int(np.sum(active)),
                "mean_partial_r2": float(np.mean(values)),
                "median_partial_r2": float(np.median(values)),
                "q25_partial_r2": float(np.quantile(values, 0.25)),
                "q75_partial_r2": float(np.quantile(values, 0.75)),
                "fraction_positive_partial_r2": float(np.mean(values > 0)),
                "mean_native_delta_rmse": float(np.mean(qc_values)),
                "median_native_delta_rmse": float(np.median(qc_values)),
                "positive_mass_surface_z": float(surface_z[index]),
                "positive_mass_anchor_distance": float(anchor_distance[index]),
            }
        )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_summary_figure(path: Path, result: Mapping[str, np.ndarray | str]) -> None:
    """Save one compact primary-plus-QC multipanel figure."""
    horizons = np.asarray(result["horizons"])
    primary = np.asarray(result["partial_r2"], dtype=float)
    native = np.asarray(result["native_delta_rmse"], dtype=float)
    fold_primary = np.asarray(result["fold_partial_r2"], dtype=float)
    active = np.asarray(result["unit_valid_fold_count"]) > 0
    primary_active = primary[:, active]

    preferred = np.argmax(primary_active, axis=0)
    strength = np.max(primary_active, axis=0)
    order = np.lexsort((-strength, preferred))
    heatmap = primary_active[:, order]
    finite_max = float(np.max(np.abs(heatmap))) if heatmap.size else 1.0
    color_limit = max(finite_max, np.finfo(float).eps)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), constrained_layout=True)
    ax = axes[0, 0]
    median = np.median(primary_active, axis=1)
    lower = np.quantile(primary_active, 0.25, axis=1)
    upper = np.quantile(primary_active, 0.75, axis=1)
    ax.plot(horizons, median, marker="o", color="#244a73", label="unit median")
    ax.fill_between(horizons, lower, upper, color="#7ca4c9", alpha=0.35, label="unit IQR")
    ax.axhline(0.0, color="0.35", linewidth=0.8)
    ax.set(title="Primary nuisance-controlled effect", xlabel="Normalized future horizon", ylabel="Cross-fitted partial $R^2$")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    image = ax.imshow(
        heatmap,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=-color_limit,
        vmax=color_limit,
    )
    ax.set(title="Unit maps (sorted by strongest horizon)", xlabel="Estimable recurrent units", ylabel="Normalized future horizon")
    ax.set_yticks(np.arange(len(horizons)), labels=horizons)
    fig.colorbar(image, ax=ax, label="Partial $R^2$", shrink=0.8)

    ax = axes[0, 2]
    native_active = native[:, active]
    ax.plot(horizons, np.median(native_active, axis=1), marker="o", color="#9b4b35")
    ax.fill_between(
        horizons,
        np.quantile(native_active, 0.25, axis=1),
        np.quantile(native_active, 0.75, axis=1),
        color="#d79a86",
        alpha=0.35,
    )
    ax.axhline(0.0, color="0.35", linewidth=0.8)
    ax.set(title="Native-scale QC", xlabel="Normalized future horizon", ylabel=r"$\Delta$RMSE (firing units)")

    ax = axes[1, 0]
    ax.plot(
        horizons,
        np.asarray(result["positive_mass_surface_z"], dtype=float),
        marker="o",
        color="#5d3a9b",
    )
    ax.set(
        title="Positive-effect cortical location",
        xlabel="Normalized future horizon",
        ylabel="Partial-$R^2$-weighted fsLR z",
    )

    ax = axes[1, 1]
    ax.plot(
        horizons,
        np.asarray(result["positive_mass_anchor_distance"], dtype=float),
        marker="o",
        color="#167d77",
    )
    ax.set(
        title="Positive-effect distance from fixed Area-25 seed",
        xlabel="Normalized future horizon",
        ylabel="Partial-$R^2$-weighted geodesic distance",
    )

    ax = axes[1, 2]
    for fold_index in range(fold_primary.shape[0]):
        fold_values = fold_primary[fold_index][:, active]
        ax.plot(
            horizons,
            np.median(fold_values, axis=1),
            color="0.65",
            linewidth=0.9,
            alpha=0.8,
        )
    ax.plot(horizons, median, marker="o", color="#244a73", linewidth=2, label="pooled OOF")
    ax.axhline(0.0, color="0.35", linewidth=0.8)
    ax.set(title="Held-out-base stability", xlabel="Normalized future horizon", ylabel="Median unit partial $R^2$")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Normalized-progress raw recurrent representation", fontsize=13)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_results(
    output_dir: Path,
    result: Mapping[str, np.ndarray | str],
    *,
    source_paths: Iterable[Path],
    geometry: Mapping[str, np.ndarray | Path | str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    augmented = dict(result)
    primary = np.asarray(result["partial_r2"], dtype=float)
    xyz = np.asarray(geometry["unit_surface_xyz"], dtype=float)
    anchor = np.asarray(geometry["anchor_distance"], dtype=float)
    augmented["positive_partial_r2"] = np.maximum(primary, 0.0)
    augmented["positive_mass_surface_z"] = weighted_positive_coordinate(
        primary, xyz[:, 2]
    )
    augmented["positive_mass_anchor_distance"] = weighted_positive_coordinate(
        primary, anchor
    )
    archive = {
        key: value
        for key, value in augmented.items()
        if isinstance(value, (np.ndarray, str))
    }
    archive["source_paths"] = np.asarray([str(path) for path in source_paths])
    np.savez_compressed(output_dir / "results.npz", **archive)
    save_summary_csv(output_dir / "summary.csv", augmented)
    save_summary_figure(output_dir / "summary.png", augmented)
    with (output_dir / "analysis.json").open("w", encoding="utf8") as handle:
        manifest_path = output_dir.parent / "analysis_manifest.json"
        common = _load_common_module()
        json.dump(
            {
                "analysis": "nuisance_controlled_normalized_progress_raw_activity",
                "definition": DEFINITION,
                "spatial_summary": (
                    "The signed partial-R2 map is retained. Cortical coordinates "
                    "use only max(partial_R2,0) as positive validated effect mass; "
                    "negative held-out effects are not treated as representation."
                ),
                "source_paths": [str(path) for path in source_paths],
                "coordinate_note": str(geometry.get("coordinate_note", "")),
                "source_manifest": str(manifest_path),
                "source_manifest_sha256": common.file_sha256(manifest_path),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the one pre-specified nuisance-controlled raw-activity "
            "analysis on normalized ABCD progress."
        )
    )
    parser.add_argument(
        "trial_collection",
        type=Path,
        help=(
            "trial_collection/repeat directory, normalized_navigation.npz, "
            "collected analysis root, managed run, or checkpoint"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <analysis-root>/raw_activity).",
    )
    args = parser.parse_args()

    source_paths = discover_normalized_inputs(args.trial_collection)
    datasets = [_load_common_navigation(path) for path in source_paths]
    data = concatenate_normalized_inputs(datasets)
    repeats = np.unique(np.asarray(data["repeat_index"], dtype=np.int64))
    if not np.array_equal(repeats, np.asarray([1, 2])):
        raise ValueError(
            "The pre-specified primary analysis requires repeat_index values "
            f"[1, 2], got {repeats.tolist()}."
        )

    result = crossfit_raw_representation(data)
    if args.output_dir is None:
        collection = source_paths[0].parent
        if collection.name.startswith("repeat_"):
            collection = collection.parent
        if collection.name == "trial_collection":
            analysis_root = collection.parent
        else:
            analysis_root = _load_common_module().resolve_existing_analysis_root(
                args.trial_collection
            )
        output_dir = analysis_root / "raw_activity"
    else:
        output_dir = args.output_dir.expanduser().resolve()
        analysis_root = output_dir.parent
    geometry = _load_common_module().load_analysis_geometry(analysis_root)
    save_results(
        output_dir,
        result,
        source_paths=source_paths,
        geometry=geometry,
    )

    primary = np.asarray(result["partial_r2"], dtype=float)
    active = np.asarray(result["unit_valid_fold_count"]) > 0
    print(
        "Raw analysis complete: "
        f"{len(result['horizons'])} normalized horizons, "
        f"{int(np.sum(active))}/{len(active)} estimable units; "
        f"mean partial R2 range "
        f"{np.mean(primary[:, active], axis=1).min():.4g}.."
        f"{np.mean(primary[:, active], axis=1).max():.4g}."
    )
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
