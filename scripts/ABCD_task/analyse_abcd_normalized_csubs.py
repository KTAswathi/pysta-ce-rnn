"""Cross-fitted ABCD Csubs analysis on normalized task progress.

The primary estimator deliberately uses the repository's original joint
decoder objective, now isolated in :mod:`abcd_csubs_decoder`. Everything
around that fit (common support, leave-base-out maps, class gauge, inactive
features, held-out null and the two named sensitivities) is ABCD-local.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ABCD_task.abcd_analysis_common import (
    file_sha256,
    load_analysis_geometry,
    load_normalized_repeats as load_common_normalized_repeats,
    resolve_existing_analysis_root,
)

try:
    from scripts.ABCD_task.abcd_csubs_decoder import fit_original_joint_csubs
except ImportError:  # Direct execution from scripts/ABCD_task.
    from abcd_csubs_decoder import fit_original_joint_csubs


N_CLASSES = 9
HORIZONS = np.arange(12, dtype=np.int64)


@dataclass(frozen=True)
class FitSettings:
    L2_alpha: float = 1e-3
    overlap_alpha: float = 2e-3
    L1_alpha: float = 1e-4
    warmup: int = 500
    max_iters: int = 2000
    atol: float = 1e-3
    rtol: float = 2e-4
    lrate: float = 5e-3
    fit_seed: int = 7301
    inactive_tolerance: float = 1e-12
    device_name: str = "auto"


def _field(data: dict[str, np.ndarray], *names: str) -> np.ndarray:
    for name in names:
        if name in data:
            return np.asarray(data[name])
    raise KeyError(f"Missing required field; tried {names}.")


def load_normalized_repeats(analysis_root: Path) -> dict[str, np.ndarray]:
    """Load and concatenate the two normalized frozen-model repeats."""
    result = load_common_normalized_repeats(analysis_root, repeat_indices=(1, 2))
    validate_normalized_dataset(result)
    return result


def validate_normalized_dataset(data: dict[str, np.ndarray]) -> None:
    rs = _field(data, "rs")
    labels = _field(data, "future_locations")
    bases = _field(data, "base_index", "base_configuration_index")
    blocks = _field(data, "block_id", "factorial_index")
    repeats = _field(data, "repeat_index")
    q = _field(data, "q", "normalized_position")
    horizons = _field(data, "horizons")

    if rs.ndim != 2:
        raise ValueError(f"rs must be [sample, unit], got {rs.shape}.")
    if labels.shape != (len(rs), 12):
        raise ValueError(
            "future_locations must have common [sample,12] support; "
            f"got {labels.shape}."
        )
    if not np.array_equal(horizons, HORIZONS):
        raise ValueError(f"Expected normalized horizons 0..11, got {horizons}.")
    if not np.all((labels >= 0) & (labels < N_CLASSES)):
        raise ValueError("Future physical-location labels must be in 0..8.")
    if not np.all((q >= 0) & (q < 12)):
        raise ValueError("q must be the normalized executed-order position 0..11.")
    if len(np.unique(bases)) < 3:
        raise ValueError("Leave-base-out Csubs requires at least three bases.")

    block_uid = repeats.astype(np.int64) * 100_000 + blocks.astype(np.int64)
    counts = np.asarray(
        [np.sum(block_uid == uid) for uid in np.unique(block_uid)], dtype=int
    )
    if len(np.unique(counts)) != 1:
        raise ValueError(
            "Csubs requires exact equal block support; block row counts are "
            f"{sorted(np.unique(counts).tolist())}."
        )
    q_counts = np.bincount(q.astype(int), minlength=12)
    if not np.all(q_counts == q_counts[0]):
        raise ValueError(f"Normalized q support is not balanced: {q_counts}.")


def class_center_filter_map(
    coefficients: np.ndarray,
    *,
    n_full_units: int,
    active_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gauge-center, class-normalize, score, and restore unit order."""

    weights = np.asarray(coefficients, dtype=np.float64)
    if weights.ndim != 3 or weights.shape[1] != N_CLASSES:
        raise ValueError(
            f"Expected [horizon,{N_CLASSES},active_unit], got {weights.shape}."
        )
    centered = weights - weights.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=2, keepdims=True)
    if np.any(norms <= 1e-14):
        raise ValueError("A centered class filter has zero norm.")
    normalized = centered / norms
    active_score = np.linalg.norm(normalized, axis=1)

    full_score = np.zeros((weights.shape[0], n_full_units), dtype=np.float64)
    full_weights = np.zeros(
        (weights.shape[0], N_CLASSES, n_full_units), dtype=np.float64
    )
    full_score[:, active_indices] = active_score
    full_weights[:, :, active_indices] = normalized
    return full_score, full_weights, centered


def helmert_contrast(n_classes: int = N_CLASSES) -> np.ndarray:
    """Return a deterministic orthonormal class-contrast basis."""

    centered_identity = np.eye(n_classes) - np.ones((n_classes, n_classes)) / n_classes
    q, _ = np.linalg.qr(centered_identity[:, :-1])
    contrast = q[:, : n_classes - 1]
    if not np.allclose(contrast.T @ contrast, np.eye(n_classes - 1), atol=1e-10):
        raise RuntimeError("Helmert contrast is not orthonormal.")
    if not np.allclose(contrast.sum(axis=0), 0.0, atol=1e-10):
        raise RuntimeError("Helmert contrast is not class-centered.")
    return contrast


def haufe_logit_pattern_map(
    X_train: np.ndarray,
    centered_coefficients: np.ndarray,
    *,
    n_full_units: int,
    active_indices: np.ndarray,
    max_condition: float = 1e12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert centered multinomial filters to linear-logit patterns.

    Each returned unit value is its row-L2 pattern magnitude divided by
    the full pattern Frobenius norm.  No pseudoinverse fallback is used.
    """

    X = np.asarray(X_train, dtype=np.float64)
    Wc = np.asarray(centered_coefficients, dtype=np.float64)
    if X.ndim != 2 or Wc.shape[2] != X.shape[1]:
        raise ValueError("Training features and centered filters do not align.")

    covariance = np.cov(X, rowvar=False, ddof=1)
    if covariance.ndim == 0:
        covariance = np.asarray([[float(covariance)]])
    contrast = helmert_contrast(Wc.shape[1])
    scores = np.full((Wc.shape[0], n_full_units), np.nan, dtype=np.float64)
    ranks = np.zeros(Wc.shape[0], dtype=np.int64)
    conditions = np.full(Wc.shape[0], np.inf, dtype=np.float64)

    for horizon in range(Wc.shape[0]):
        B = Wc[horizon].T @ contrast
        score_covariance = B.T @ covariance @ B
        rank = int(np.linalg.matrix_rank(score_covariance))
        condition = float(np.linalg.cond(score_covariance))
        ranks[horizon] = rank
        conditions[horizon] = condition
        if rank != contrast.shape[1] or not np.isfinite(condition) or condition > max_condition:
            continue
        A = covariance @ B @ np.linalg.inv(score_covariance)
        denominator = float(np.linalg.norm(A))
        if not np.isfinite(denominator) or denominator <= 0:
            continue
        active_score = np.linalg.norm(A, axis=1) / denominator
        scores[horizon, active_indices] = active_score
        scores[horizon, np.setdiff1d(np.arange(n_full_units), active_indices)] = 0.0

    return scores, ranks, conditions


def _predict(coefficients: np.ndarray, biases: np.ndarray, X: np.ndarray) -> np.ndarray:
    logits = np.einsum("hcu,nu->hnc", coefficients, X)
    logits += np.asarray(biases)[..., 0][:, None, :]
    return np.argmax(logits, axis=2).T.astype(np.int64)


def _macro_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    truth = np.asarray(y_true, dtype=np.int64)
    prediction = np.asarray(y_pred, dtype=np.int64)
    if truth.shape != prediction.shape or truth.ndim != 2:
        raise ValueError("Balanced-accuracy inputs must share [sample,horizon] shape.")
    result = np.zeros(truth.shape[1], dtype=np.float64)
    for horizon in range(truth.shape[1]):
        recalls = []
        present_locations = np.unique(truth[:, horizon])
        for location in present_locations:
            selected = truth[:, horizon] == location
            recalls.append(float(np.mean(prediction[selected, horizon] == location)))
        result[horizon] = float(np.mean(recalls))
    return result


def trajectory_label_null(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_permutations: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Conditional held-out-label null with one global bijection/draw."""

    observed = _macro_balanced_accuracy(y_true, y_pred)
    rng = np.random.default_rng(seed)
    null = np.zeros((n_permutations, y_true.shape[1]), dtype=np.float64)
    for permutation_i in range(n_permutations):
        bijection = rng.permutation(N_CLASSES)
        null[permutation_i] = _macro_balanced_accuracy(
            bijection[y_true], y_pred
        )
    p_value = (
        1.0 + np.sum(null >= observed[None, :], axis=0)
    ) / (n_permutations + 1.0)
    max_null = np.max(null, axis=1)
    p_value_max_t = (
        1.0 + np.sum(max_null[:, None] >= observed[None, :], axis=0)
    ) / (n_permutations + 1.0)
    future_mean_observed = float(np.mean(observed[1:]))
    future_mean_null = np.mean(null[:, 1:], axis=1)
    future_mean_p_value = float(
        (1.0 + np.sum(future_mean_null >= future_mean_observed))
        / (n_permutations + 1.0)
    )
    return {
        "observed": observed,
        "null": null,
        "p_value": p_value,
        "p_value_max_t": p_value_max_t,
        "null_low": np.quantile(null, 0.025, axis=0),
        "null_high": np.quantile(null, 0.975, axis=0),
        "future_mean_observed": np.asarray(future_mean_observed),
        "future_mean_null": future_mean_null,
        "future_mean_p_value": np.asarray(future_mean_p_value),
    }


def _pairwise_map_stability(fold_maps: np.ndarray) -> np.ndarray:
    stability = np.full(fold_maps.shape[1], np.nan, dtype=np.float64)
    for horizon in range(fold_maps.shape[1]):
        correlations = []
        for first, second in combinations(range(fold_maps.shape[0]), 2):
            x = fold_maps[first, horizon]
            y = fold_maps[second, horizon]
            if np.std(x) > 0 and np.std(y) > 0:
                correlations.append(float(np.corrcoef(x, y)[0, 1]))
        if correlations:
            stability[horizon] = float(np.median(correlations))
    return stability


def run_cross_fitted_estimator(
    data: dict[str, np.ndarray],
    *,
    settings: FitSettings,
    standardized_features: bool,
    overlap_mode: str,
    compute_haufe: bool,
) -> dict[str, Any]:
    """Fit the actual joint estimator in leave-base-out folds."""

    validate_normalized_dataset(data)
    X_all = np.asarray(_field(data, "rs"), dtype=np.float64)
    y_all = np.asarray(_field(data, "future_locations"), dtype=np.int64)
    bases = np.asarray(
        _field(data, "base_index", "base_configuration_index"), dtype=np.int64
    )
    unique_bases = np.sort(np.unique(bases))
    n_samples, n_units = X_all.shape
    fold_maps = []
    fold_haufe = []
    fold_accuracy = []
    inactive_masks = []
    haufe_ranks = []
    haufe_conditions = []
    iterations = []
    losses = []
    logs = []
    oof_predictions = np.full((n_samples, 12), -1, dtype=np.int64)

    for fold_i, heldout_base in enumerate(unique_bases):
        train = bases != heldout_base
        test = ~train
        train_mean = X_all[train].mean(axis=0)
        train_scale = X_all[train].std(axis=0, ddof=0)
        active = train_scale > settings.inactive_tolerance
        active_indices = np.flatnonzero(active)
        if len(active_indices) == 0:
            raise ValueError(f"No active units in fold holding out base {heldout_base}.")

        X_train_native = X_all[train][:, active]
        X_test_native = X_all[test][:, active]
        if standardized_features:
            scale = train_scale[active]
            X_train = (X_train_native - train_mean[active]) / scale
            X_test = (X_test_native - train_mean[active]) / scale
        else:
            X_train = X_train_native
            X_test = X_test_native

        fit_stdout = io.StringIO()
        with contextlib.redirect_stdout(fit_stdout):
            fit = fit_original_joint_csubs(
                rs=X_train.astype(np.float32),
                future_locations=y_all[train],
                future_valid=np.ones_like(y_all[train], dtype=bool),
                future_lags=HORIZONS,
                L2_alpha=settings.L2_alpha,
                overlap_alpha=settings.overlap_alpha,
                L1_alpha=settings.L1_alpha,
                warmup=settings.warmup,
                max_iters=settings.max_iters,
                atol=settings.atol,
                rtol=settings.rtol,
                lrate=settings.lrate,
                fit_seed=settings.fit_seed + fold_i,
                overlap_mode=overlap_mode,
                device_name=settings.device_name,
            )
        logs.append(f"# held-out base {int(heldout_base)}\n{fit_stdout.getvalue()}")

        coefficients = np.asarray(fit["Csubs_raw"], dtype=np.float64)
        biases = np.asarray(fit["biases"], dtype=np.float64)
        biases = biases - biases.mean(axis=1, keepdims=True)
        score, _, centered = class_center_filter_map(
            coefficients,
            n_full_units=n_units,
            active_indices=active_indices,
        )
        predictions = _predict(coefficients, biases, X_test)
        oof_predictions[test] = predictions
        fold_maps.append(score)
        fold_accuracy.append(_macro_balanced_accuracy(y_all[test], predictions))
        inactive_masks.append(~active)
        iterations.append(int(fit["iterations_run"]))
        losses.append(float(fit["final_total_loss"]))

        if compute_haufe:
            haufe, rank, condition = haufe_logit_pattern_map(
                X_train,
                centered,
                n_full_units=n_units,
                active_indices=active_indices,
            )
            fold_haufe.append(haufe)
            haufe_ranks.append(rank)
            haufe_conditions.append(condition)

    if np.any(oof_predictions < 0):
        raise RuntimeError("Cross-fitting left samples without predictions.")

    fold_maps_array = np.asarray(fold_maps)
    result: dict[str, Any] = {
        "map": np.mean(fold_maps_array, axis=0),
        "fold_maps": fold_maps_array,
        "fold_accuracy": np.asarray(fold_accuracy),
        "oof_predictions": oof_predictions,
        "map_stability": _pairwise_map_stability(fold_maps_array),
        "inactive_masks": np.asarray(inactive_masks),
        "iterations": np.asarray(iterations),
        "losses": np.asarray(losses),
        "fit_log": "\n".join(logs),
        "heldout_bases": unique_bases,
    }
    if compute_haufe:
        fold_haufe_array = np.asarray(fold_haufe)
        if np.any(~np.isfinite(fold_haufe_array)):
            result["haufe_map"] = np.nanmean(fold_haufe_array, axis=0)
        else:
            result["haufe_map"] = np.mean(fold_haufe_array, axis=0)
        result["fold_haufe_maps"] = fold_haufe_array
        result["haufe_ranks"] = np.asarray(haufe_ranks)
        result["haufe_conditions"] = np.asarray(haufe_conditions)
    return result


def weighted_spatial_curve(score: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    values = np.asarray(score, dtype=np.float64)
    coord = np.asarray(coordinate, dtype=np.float64)
    if values.shape[1] != len(coord):
        raise ValueError("Unit map and coordinate length differ.")
    numerator = np.nansum(values * coord[None, :], axis=1)
    denominator = np.nansum(values, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(values), np.nan),
        where=denominator > 0,
    )


def _load_geometry(analysis_root: Path) -> dict[str, np.ndarray | Path | str]:
    return load_analysis_geometry(analysis_root)


def _save_summary_figure(
    output: Path,
    *,
    decodability: dict[str, np.ndarray],
    primary: dict[str, Any],
    sensitivity: dict[str, Any],
    z_curves: dict[str, np.ndarray],
    anchor_curves: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), constrained_layout=True)
    h = HORIZONS
    ax = axes[0, 0]
    ax.fill_between(
        h,
        decodability["null_low"],
        decodability["null_high"],
        color="0.85",
        label="trajectory-label null 95%",
    )
    ax.plot(h, decodability["observed"], "o-", color="#2459a6", label="held-out")
    ax.axhline(1 / 9, ls="--", color="0.3", label="uniform 1/9 (descriptive)")
    ax.set(title="Population decodability", xlabel="normalized future horizon", ylabel="balanced accuracy")
    ax.legend(frameon=False, fontsize=8)

    styles = {
        "original filter": ("#7b2cbf", "o-"),
        "scaled/corrected filter": ("#dd6e42", "s--"),
        "activation pattern": ("#2a9d8f", "^-.")
    }
    ax = axes[0, 1]
    for label, curve in z_curves.items():
        color, style = styles[label]
        ax.plot(h, curve, style, color=color, label=label)
    ax.set(title="Whole-map surface-z", xlabel="normalized future horizon", ylabel="score-weighted fsLR z")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    for label, curve in anchor_curves.items():
        color, style = styles[label]
        ax.plot(h, curve, style, color=color, label=label)
    ax.set(title="Distance from fixed Area-25 seed", xlabel="normalized future horizon", ylabel="score-weighted geodesic distance")

    ax = axes[1, 1]
    ax.plot(h, primary["map_stability"], "o-", color="#7b2cbf", label="original filter")
    ax.plot(h, sensitivity["map_stability"], "s--", color="#dd6e42", label="scaled/corrected")
    ax.axhline(0, color="0.7", lw=1)
    ax.set(title="Configuration-fold map stability", xlabel="normalized future horizon", ylabel="median fold-pair r")
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("ABCD normalized-progress Csubs", fontsize=14)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run_analysis(
    analysis_root: Path,
    *,
    settings: FitSettings,
    n_permutations: int,
    permutation_seed: int,
) -> Path:
    analysis_root = resolve_existing_analysis_root(analysis_root)
    data = load_normalized_repeats(analysis_root)
    geometry = _load_geometry(analysis_root)
    output_dir = analysis_root / "csubs"
    output_dir.mkdir(parents=True, exist_ok=True)

    primary = run_cross_fitted_estimator(
        data,
        settings=settings,
        standardized_features=False,
        overlap_mode="raw_dot",
        compute_haufe=True,
    )
    sensitivity = run_cross_fitted_estimator(
        data,
        settings=settings,
        standardized_features=True,
        overlap_mode="normalized",
        compute_haufe=False,
    )
    labels = np.asarray(_field(data, "future_locations"), dtype=np.int64)
    null = trajectory_label_null(
        labels,
        primary["oof_predictions"],
        n_permutations=n_permutations,
        seed=permutation_seed,
    )

    coords = _field(geometry, "unit_surface_xyz", "unit_coords", "unit_xyz")
    anchor_distance = _field(geometry, "anchor_distance", "anchor_mean_distance")
    maps = {
        "original filter": primary["map"],
        "scaled/corrected filter": sensitivity["map"],
        "activation pattern": primary["haufe_map"],
    }
    z_curves = {label: weighted_spatial_curve(score, coords[:, 2]) for label, score in maps.items()}
    anchor_curves = {label: weighted_spatial_curve(score, anchor_distance) for label, score in maps.items()}

    np.savez_compressed(
        output_dir / "results.npz",
        horizons=HORIZONS,
        primary_filter_map=primary["map"],
        primary_fold_maps=primary["fold_maps"],
        estimator_sensitivity_map=sensitivity["map"],
        estimator_sensitivity_fold_maps=sensitivity["fold_maps"],
        activation_pattern_map=primary["haufe_map"],
        activation_pattern_fold_maps=primary["fold_haufe_maps"],
        oof_predictions=primary["oof_predictions"],
        oof_balanced_accuracy=null["observed"],
        permutation_null=null["null"],
        permutation_p_value=null["p_value"],
        permutation_p_value_max_t=null["p_value_max_t"],
        future_horizon_mean_accuracy=null["future_mean_observed"],
        future_horizon_mean_permutation_null=null["future_mean_null"],
        future_horizon_mean_permutation_p=null["future_mean_p_value"],
        primary_map_stability=primary["map_stability"],
        sensitivity_map_stability=sensitivity["map_stability"],
        primary_fold_balanced_accuracy=primary["fold_accuracy"],
        sensitivity_fold_balanced_accuracy=sensitivity["fold_accuracy"],
        primary_inactive_masks=primary["inactive_masks"],
        sensitivity_inactive_masks=sensitivity["inactive_masks"],
        haufe_ranks=primary["haufe_ranks"],
        haufe_conditions=primary["haufe_conditions"],
        primary_surface_z=z_curves["original filter"],
        sensitivity_surface_z=z_curves["scaled/corrected filter"],
        activation_pattern_surface_z=z_curves["activation pattern"],
        primary_anchor_distance=anchor_curves["original filter"],
        sensitivity_anchor_distance=anchor_curves["scaled/corrected filter"],
        activation_pattern_anchor_distance=anchor_curves["activation pattern"],
    )

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "horizon", "balanced_accuracy", "permutation_p", "permutation_p_maxT", "primary_surface_z",
                "sensitivity_surface_z", "activation_pattern_surface_z",
                "primary_anchor_distance", "sensitivity_anchor_distance",
                "activation_pattern_anchor_distance", "primary_fold_stability",
                "sensitivity_fold_stability",
            ],
        )
        writer.writeheader()
        for h in HORIZONS:
            writer.writerow({
                "horizon": int(h),
                "balanced_accuracy": float(null["observed"][h]),
                "permutation_p": float(null["p_value"][h]),
                "permutation_p_maxT": float(null["p_value_max_t"][h]),
                "primary_surface_z": float(z_curves["original filter"][h]),
                "sensitivity_surface_z": float(z_curves["scaled/corrected filter"][h]),
                "activation_pattern_surface_z": float(z_curves["activation pattern"][h]),
                "primary_anchor_distance": float(anchor_curves["original filter"][h]),
                "sensitivity_anchor_distance": float(anchor_curves["scaled/corrected filter"][h]),
                "activation_pattern_anchor_distance": float(anchor_curves["activation pattern"][h]),
                "primary_fold_stability": float(primary["map_stability"][h]),
                "sensitivity_fold_stability": float(sensitivity["map_stability"][h]),
            })

    _save_summary_figure(
        output_dir / "summary.png",
        decodability=null,
        primary=primary,
        sensitivity=sensitivity,
        z_curves=z_curves,
        anchor_curves=anchor_curves,
    )
    (output_dir / "fit_log.txt").write_text(
        "# PRIMARY ORIGINAL-FILTER FITS\n" + primary["fit_log"]
        + "\n# STANDARDIZED/CORRECTED-OVERLAP FITS\n" + sensitivity["fit_log"],
        encoding="utf-8",
    )
    metadata = {
        "analysis": "normalized_progress_cross_fitted_csubs",
        "definition": (
            "Per-unit L2 magnitude across nine class-centered, per-class-normalized "
            "future-location filters, formed per leave-base-out fit then equal-fold averaged."
        ),
        "interpretation": (
            "Decoder-filter allocation, not unique information, causal contribution, "
            "intrinsic representational strength, or an RSA coefficient."
        ),
        "horizons": HORIZONS.tolist(),
        "primary": "raw features + original raw-dot overlap",
        "estimator_sensitivity": "training-fold z-scored features + cosine-normalized overlap",
        "estimator_sensitivity_scope": (
            "Scaling and corrected overlap are changed together as the single "
            "pre-specified estimator sensitivity; discrepancies cannot be "
            "attributed uniquely to either change."
        ),
        "interpretation_sensitivity": "full-rank linear-logit Haufe activation pattern",
        "n_permutations": int(n_permutations),
        "settings": settings.__dict__,
        "fit_iterations_primary": primary["iterations"].tolist(),
        "fit_iterations_sensitivity": sensitivity["iterations"].tolist(),
        "source_manifest": str(analysis_root / "analysis_manifest.json"),
        "source_manifest_sha256": file_sha256(
            analysis_root / "analysis_manifest.json"
        ),
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-fitted Csubs at ABCD normalized future-progress horizons."
    )
    parser.add_argument(
        "analysis_root",
        type=Path,
        help=(
            "Collected analysis root, managed run directory, canonical "
            "checkpoint, or legacy checkpoint."
        ),
    )
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--permutation-seed", type=int, default=881)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-iters", type=int, default=2000)
    args = parser.parse_args()
    settings = FitSettings(device_name=args.device, max_iters=args.max_iters)
    output = run_analysis(
        args.analysis_root,
        settings=settings,
        n_permutations=args.n_permutations,
        permutation_seed=args.permutation_seed,
    )
    print(f"Csubs outputs: {output}")


if __name__ == "__main__":
    main()
