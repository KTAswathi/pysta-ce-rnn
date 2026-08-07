#!/usr/bin/env python3
"""Plot population-preserving diagnostics for fMRI-analogue RNN maps.

This script consumes one ``raw_activity`` result folder and one normalized
``Csubs`` result folder produced by
``scripts/analyse_fmri_analogous_model_gradient.py``. It does not rerun or
alter either analysis.

It saves four complementary diagnostics:

1. all-unit, unthresholded score-weighted fsLR-z by future-location lag;
2. p90-survivor score-weighted fsLR-z after pooling every component;
3. p90 rank-1-component score-weighted COM z, retaining all survivors as
   faint spatial context;
4. unit preferred future lag versus fsLR-z, with undefined preferences
   explicitly excluded rather than assigned to lag zero.

Example
-------
python scripts/diagnose_fmri_analogous_population_gradient.py \
    RAW_RESULT_DIR SUBSPACE_RESULT_DIR \
    --embedding_dir data/embedding/subsampled/human/EMBEDDING/units=480_seed=42 \
    --output_dir data/rnn_analyses/.../fmri_analogous/population_gradient_diagnostics
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


DEFAULT_SURFACE = Path(
    "data/embedding/raw_surface_data/human/fs_lr32/surf/"
    "fs_lr32.l.midthickness.surf.gii"
)

VERTEX_FILE_CANDIDATES = (
    "sampled_indices.npy",
    "sampled_vertex_indices.npy",
    "unit_vertex_indices.npy",
    "vertex_indices.npy",
    "roi_vertex_indices.npy",
)

SOURCE_LABELS = {
    "raw_activity": "Raw future-location tuning",
    "subspace": "Normalized Csubs participation",
}

SOURCE_COLORS = {
    "raw_activity": "#2474B5",
    "subspace": "#D55E00",
}


def resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = repo_root / path

    return path.resolve()


def load_result(
    result_dir: Path,
    expected_source: str,
) -> dict[str, Any]:
    npz_path = result_dir / "lag_score_maps.npz"
    config_path = result_dir / "run_config.json"

    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing lag-score archive: {npz_path}")

    if not config_path.is_file():
        raise FileNotFoundError(f"Missing run configuration: {config_path}")

    archive = np.load(npz_path, allow_pickle=False)
    required = {
        "delay_scores",
        "delay_labels",
        "actual_lag_times",
        "score_source",
    }
    missing = required.difference(archive.files)

    if missing:
        raise KeyError(
            f"{npz_path} is missing required arrays: {sorted(missing)}"
        )

    source = str(archive["score_source"].item())

    if source != expected_source:
        raise ValueError(
            f"Expected score_source={expected_source!r} in {npz_path}, "
            f"found {source!r}."
        )

    scores = np.asarray(archive["delay_scores"], dtype=float)
    lag_values = np.asarray(archive["actual_lag_times"], dtype=float)
    lag_labels = np.asarray(archive["delay_labels"]).astype(str)

    if scores.ndim != 2:
        raise ValueError(
            f"Expected condition-by-unit scores in {npz_path}, got "
            f"{scores.shape}."
        )

    if (
        lag_values.ndim != 1
        or lag_labels.ndim != 1
        or lag_values.size != scores.shape[0]
        or lag_labels.size != scores.shape[0]
    ):
        raise ValueError(
            "Lag values/labels do not match the number of score maps in "
            f"{npz_path}."
        )

    if not np.all(np.isfinite(scores)):
        raise ValueError(f"Score maps contain non-finite values: {npz_path}")

    if np.min(scores) < -1e-12:
        raise ValueError(f"Score maps contain negative values: {npz_path}")

    scores = scores.copy()
    scores[scores < 0] = 0.0

    with open(config_path) as handle:
        config = json.load(handle)

    return {
        "result_dir": result_dir,
        "source": source,
        "scores": scores,
        "lag_values": lag_values,
        "lag_labels": lag_labels,
        "config": config,
    }


def find_unit_vertices(
    embedding_dir: Path,
    n_units: int,
) -> np.ndarray:
    for filename in VERTEX_FILE_CANDIDATES:
        path = embedding_dir / filename

        if not path.is_file():
            continue

        vertices = np.asarray(np.load(path), dtype=int).reshape(-1)

        if vertices.size == n_units:
            return vertices

    raise FileNotFoundError(
        f"Could not find a unit-vertex array of length {n_units} in "
        f"{embedding_dir}."
    )


def load_unit_z(
    surface_path: Path,
    embedding_dir: Path,
    n_units: int,
) -> np.ndarray:
    if not surface_path.is_file():
        raise FileNotFoundError(f"Missing fsLR surface: {surface_path}")

    surface = nib.load(str(surface_path))
    coordinates = np.asarray(surface.agg_data()[0], dtype=float)

    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(
            f"Surface coordinates must have shape (vertices, 3), got "
            f"{coordinates.shape}."
        )

    vertices = find_unit_vertices(embedding_dir, n_units)

    if np.unique(vertices).size != n_units:
        raise ValueError("Unit representative vertices are not unique.")

    if np.any((vertices < 0) | (vertices >= coordinates.shape[0])):
        raise ValueError("Unit vertex indices fall outside the fsLR surface.")

    unit_z = coordinates[vertices, 2]

    if not np.all(np.isfinite(unit_z)):
        raise ValueError("Unit fsLR-z coordinates contain non-finite values.")

    return unit_z


def load_unit_surface_adjacency(
    surface_path: Path,
    embedding_dir: Path,
    unit_vertices: np.ndarray,
) -> list[set[int]]:
    """Reproduce the main analysis's full-surface unit adjacency exactly."""
    surface = nib.load(str(surface_path))
    surface_data = surface.agg_data()
    coordinates = np.asarray(surface_data[0], dtype=float)
    faces = np.asarray(surface_data[1], dtype=int)
    assignment_path = embedding_dir / "vertex_to_cluster.npy"

    if not assignment_path.is_file():
        raise FileNotFoundError(
            "Rank-1 diagnostics require the same full-surface assignment "
            f"used by the main analysis: {assignment_path}"
        )

    assignments = np.asarray(
        np.load(assignment_path),
        dtype=int,
    ).reshape(-1)
    n_units = int(unit_vertices.size)

    if assignments.size != coordinates.shape[0]:
        raise ValueError(
            "vertex_to_cluster length does not match the fsLR surface: "
            f"{assignments.size} != {coordinates.shape[0]}."
        )

    represented = np.unique(assignments[assignments >= 0])

    if not np.array_equal(represented, np.arange(n_units, dtype=int)):
        raise ValueError(
            "vertex_to_cluster does not represent precisely all model units."
        )

    if not np.array_equal(
        assignments[unit_vertices],
        np.arange(n_units, dtype=int),
    ):
        raise ValueError(
            "Representative vertices do not map back to their model units."
        )

    adjacency: list[set[int]] = [set() for _ in range(n_units)]

    for triangle in faces:
        units = np.unique(assignments[np.asarray(triangle, dtype=int)])
        units = units[(units >= 0) & (units < n_units)]

        for first_index in range(len(units)):
            for second_index in range(first_index + 1, len(units)):
                first = int(units[first_index])
                second = int(units[second_index])
                adjacency[first].add(second)
                adjacency[second].add(first)

    return adjacency


def connected_unit_components(
    selected: np.ndarray,
    adjacency: list[set[int]],
) -> list[np.ndarray]:
    """Connected components using the same unit graph as the main analysis."""
    selected = np.asarray(selected, dtype=bool)
    visited = np.zeros(selected.size, dtype=bool)
    components: list[np.ndarray] = []

    for start in np.flatnonzero(selected):
        if visited[start]:
            continue

        queue = deque([int(start)])
        visited[start] = True
        component: list[int] = []

        while queue:
            unit = queue.popleft()
            component.append(unit)

            for neighbour in adjacency[unit]:
                if selected[neighbour] and not visited[neighbour]:
                    visited[neighbour] = True
                    queue.append(neighbour)

        components.append(np.asarray(component, dtype=int))

    return components


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    denominator = float(np.sum(weights))

    if not np.isfinite(denominator) or denominator <= 0:
        return np.nan

    return float(np.sum(values * weights) / denominator)


def calculate_lag_diagnostics(
    scores: np.ndarray,
    unit_z: np.ndarray,
    percentile: float,
) -> dict[str, np.ndarray]:
    n_lags = scores.shape[0]
    whole_z = np.full(n_lags, np.nan, dtype=float)
    top_z = np.full(n_lags, np.nan, dtype=float)
    cutoffs = np.full(n_lags, np.nan, dtype=float)
    selected_counts = np.zeros(n_lags, dtype=int)
    top_mass_fraction = np.full(n_lags, np.nan, dtype=float)
    top_masks = np.zeros_like(scores, dtype=bool)

    for lag_index, score_map in enumerate(scores):
        whole_z[lag_index] = weighted_mean(unit_z, score_map)
        cutoff = float(np.percentile(score_map, percentile))
        selected = score_map > cutoff

        cutoffs[lag_index] = cutoff
        selected_counts[lag_index] = int(np.sum(selected))
        top_masks[lag_index] = selected
        top_z[lag_index] = weighted_mean(
            unit_z[selected],
            score_map[selected],
        )

        total_mass = float(np.sum(score_map))
        top_mass_fraction[lag_index] = (
            float(np.sum(score_map[selected])) / total_mass
            if total_mass > 0
            else np.nan
        )

    return {
        "whole_z": whole_z,
        "top_z": top_z,
        "cutoffs": cutoffs,
        "selected_counts": selected_counts,
        "top_mass_fraction": top_mass_fraction,
        "top_masks": top_masks,
    }


def calculate_rank1_diagnostics(
    scores: np.ndarray,
    unit_z: np.ndarray,
    top_masks: np.ndarray,
    adjacency: list[set[int]],
) -> dict[str, np.ndarray]:
    """Rank p90 components by score mass and retain the strongest one."""
    n_lags = scores.shape[0]
    rank1_z = np.full(n_lags, np.nan, dtype=float)
    rank1_mass = np.full(n_lags, np.nan, dtype=float)
    rank1_mass_fraction = np.full(n_lags, np.nan, dtype=float)
    rank1_counts = np.zeros(n_lags, dtype=int)
    component_counts = np.zeros(n_lags, dtype=int)
    rank1_masks = np.zeros_like(top_masks, dtype=bool)

    for lag_index, score_map in enumerate(scores):
        selected = top_masks[lag_index]
        components = connected_unit_components(selected, adjacency)
        components = sorted(
            components,
            key=lambda component: float(np.nansum(score_map[component])),
            reverse=True,
        )
        component_counts[lag_index] = len(components)

        if not components:
            continue

        rank1 = components[0]
        rank1_masks[lag_index, rank1] = True
        rank1_counts[lag_index] = int(rank1.size)
        rank1_mass[lag_index] = float(np.sum(score_map[rank1]))
        rank1_z[lag_index] = weighted_mean(
            unit_z[rank1],
            score_map[rank1],
        )
        selected_mass = float(np.sum(score_map[selected]))

        if selected_mass > 0:
            rank1_mass_fraction[lag_index] = (
                rank1_mass[lag_index] / selected_mass
            )

    return {
        "rank1_z": rank1_z,
        "rank1_mass": rank1_mass,
        "rank1_mass_fraction": rank1_mass_fraction,
        "rank1_counts": rank1_counts,
        "component_counts": component_counts,
        "rank1_masks": rank1_masks,
    }


def verify_saved_rank1_com(
    result: dict[str, Any],
    rank1_diagnostics: dict[str, np.ndarray],
) -> None:
    """Require reconstructed rank-1 COM values to match the main analysis."""
    csv_path = (
        result["result_dir"]
        / "fmri_analogous_peak_cluster_com_coordinates.csv"
    )

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing saved coordinate table for rank-1 verification: {csv_path}"
        )

    saved_rows: dict[int, dict[str, str]] = {}

    with open(csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["mode"] == "cluster_com" and int(row["cluster_rank"]) == 1:
                saved_rows[int(row["delay_index"])] = row

    expected_indices = set(range(result["scores"].shape[0]))

    if set(saved_rows) != expected_indices:
        raise ValueError(
            "Saved coordinate table does not contain one rank-1 COM for "
            f"every lag: {csv_path}"
        )

    saved_z = np.asarray(
        [float(saved_rows[index]["z"]) for index in sorted(saved_rows)],
        dtype=float,
    )
    saved_counts = np.asarray(
        [
            int(saved_rows[index]["cluster_size"])
            for index in sorted(saved_rows)
        ],
        dtype=int,
    )
    saved_mass = np.asarray(
        [
            float(saved_rows[index]["cluster_mass"])
            for index in sorted(saved_rows)
        ],
        dtype=float,
    )

    if not np.allclose(
        saved_z,
        rank1_diagnostics["rank1_z"],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "Reconstructed rank-1 COM z does not match the main analysis."
        )

    if not np.array_equal(
        saved_counts,
        rank1_diagnostics["rank1_counts"],
    ):
        raise ValueError(
            "Reconstructed rank-1 sizes do not match the main analysis."
        )

    if not np.allclose(
        saved_mass,
        rank1_diagnostics["rank1_mass"],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "Reconstructed rank-1 masses do not match the main analysis."
        )


def calculate_preferred_lags(
    scores: np.ndarray,
    tie_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    maxima = np.max(scores, axis=0)
    totals = np.sum(scores, axis=0)
    at_maximum = np.isclose(
        scores,
        maxima[None, :],
        rtol=0.0,
        atol=tie_tolerance,
    )
    valid = (
        (totals > 0)
        & np.isfinite(maxima)
        & (np.sum(at_maximum, axis=0) == 1)
    )
    preferred = np.full(scores.shape[1], -1, dtype=int)
    preferred[valid] = np.argmax(scores[:, valid], axis=0)
    return preferred, valid


def descriptive_trend(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float]:
    finite = np.isfinite(x) & np.isfinite(y)

    if np.sum(finite) < 2 or np.unique(x[finite]).size < 2:
        return np.nan, np.nan, np.nan

    slope, intercept = np.polyfit(x[finite], y[finite], 1)

    if np.std(y[finite]) == 0:
        correlation = np.nan
    else:
        correlation = float(np.corrcoef(x[finite], y[finite])[0, 1])

    return float(slope), float(intercept), correlation


def annotate_values(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
) -> None:
    finite_y = y[np.isfinite(y)]
    span = float(np.ptp(finite_y)) if finite_y.size else 1.0
    offset = max(0.35, 0.045 * span)

    for x_value, y_value in zip(x, y):
        if np.isfinite(y_value):
            axis.text(
                x_value,
                y_value + offset,
                f"{y_value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )


def weighted_z_histogram(
    scores: np.ndarray,
    unit_z: np.ndarray,
    bin_edges: np.ndarray,
) -> np.ndarray:
    histograms = []

    for score_map in scores:
        histogram, _ = np.histogram(
            unit_z,
            bins=bin_edges,
            weights=score_map,
        )
        total = float(np.sum(histogram))

        if total > 0:
            histogram = histogram / total

        histograms.append(histogram)

    return np.asarray(histograms, dtype=float).T


def plot_all_unit_population_shift(
    results: list[dict[str, Any]],
    diagnostics: dict[str, dict[str, np.ndarray]],
    unit_z: np.ndarray,
    output_path: Path,
    n_z_bins: int,
) -> None:
    bin_edges = np.linspace(
        float(np.min(unit_z)),
        float(np.max(unit_z)),
        n_z_bins + 1,
    )
    histograms = {
        result["source"]: weighted_z_histogram(
            result["scores"],
            unit_z,
            bin_edges,
        )
        for result in results
    }
    common_vmax = max(
        float(np.max(histogram))
        for histogram in histograms.values()
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.0, 10.0),
        sharex="col",
        gridspec_kw={"height_ratios": [1.0, 1.25]},
    )
    heatmap_image = None

    for column, result in enumerate(results):
        source = result["source"]
        color = SOURCE_COLORS[source]
        lag_values = result["lag_values"]
        weighted_z = diagnostics[source]["whole_z"]
        slope, intercept, correlation = descriptive_trend(
            lag_values,
            weighted_z,
        )

        line_axis = axes[0, column]
        line_axis.plot(
            lag_values,
            weighted_z,
            marker="o",
            markersize=7,
            linewidth=2.6,
            color=color,
            label="All unit scores",
        )
        line_axis.plot(
            lag_values,
            intercept + slope * lag_values,
            linestyle="--",
            linewidth=1.8,
            color="0.35",
            label="Descriptive linear fit",
        )
        annotate_values(line_axis, lag_values, weighted_z)
        line_axis.set_title(
            f"{SOURCE_LABELS[source]}\n"
            f"all {result['scores'].shape[1]} unit positions; "
            f"{int(np.sum(np.sum(result['scores'], axis=0) > 0))} "
            "with positive total score\n"
            "no threshold/components; "
            f"slope={slope:.3f}, descriptive r={correlation:.3f}"
        )
        line_axis.set_ylabel("Score-weighted mean fsLR z-coordinate")
        line_axis.grid(alpha=0.25)
        line_axis.legend(frameon=False, fontsize=9)

        heatmap_axis = axes[1, column]
        histogram = histograms[source]
        heatmap_image = heatmap_axis.imshow(
            100.0 * histogram,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(
                float(lag_values[0]) - 0.5,
                float(lag_values[-1]) + 0.5,
                float(bin_edges[0]),
                float(bin_edges[-1]),
            ),
            cmap="magma",
            vmin=0.0,
            vmax=100.0 * common_vmax,
        )
        heatmap_axis.plot(
            lag_values,
            weighted_z,
            marker="o",
            linewidth=2.0,
            color="cyan",
            markeredgecolor="black",
            markeredgewidth=0.5,
            label="Weighted mean z",
        )
        heatmap_axis.set_xlabel(
            "Future-location lag (0=current; 1=+1 action; ...; 5=+5)"
        )
        heatmap_axis.set_ylabel("fsLR z-coordinate")
        heatmap_axis.set_xticks(lag_values)
        heatmap_axis.legend(frameon=True, fontsize=8, loc="upper left")

    if heatmap_image is not None:
        colorbar = fig.colorbar(
            heatmap_image,
            ax=axes[1, :].tolist(),
            fraction=0.025,
            pad=0.02,
        )
        colorbar.set_label("Percent of lag-map score mass per z bin")

    fig.suptitle(
        "N480 distributed population shift along fsLR z\n"
        "all-unit lag maps; no threshold and no cluster selection",
        fontsize=16,
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.91,
        bottom=0.08,
        top=0.79,
        hspace=0.28,
        wspace=0.22,
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_top_survivor_population_shift(
    results: list[dict[str, Any]],
    diagnostics: dict[str, dict[str, np.ndarray]],
    unit_z: np.ndarray,
    output_path: Path,
    percentile: float,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.8),
        sharey=True,
    )
    rng = np.random.default_rng(42)

    for axis, result in zip(axes, results):
        source = result["source"]
        color = SOURCE_COLORS[source]
        lag_values = result["lag_values"]
        score_maps = result["scores"]
        source_diagnostics = diagnostics[source]
        top_z = source_diagnostics["top_z"]
        top_masks = source_diagnostics["top_masks"]

        for lag_index, lag_value in enumerate(lag_values):
            selected = top_masks[lag_index]
            selected_scores = score_maps[lag_index, selected]
            score_range = float(np.ptp(selected_scores))
            scaled = (
                (selected_scores - float(np.min(selected_scores)))
                / score_range
                if score_range > 0
                else np.zeros_like(selected_scores)
            )
            jitter = rng.uniform(-0.12, 0.12, size=int(np.sum(selected)))
            axis.scatter(
                np.full(int(np.sum(selected)), lag_value) + jitter,
                unit_z[selected],
                s=9.0 + 22.0 * scaled,
                color=color,
                alpha=0.30,
                linewidths=0,
            )

        slope, intercept, correlation = descriptive_trend(
            lag_values,
            top_z,
        )
        axis.plot(
            lag_values,
            top_z,
            marker="o",
            markersize=7,
            linewidth=2.7,
            color="black",
            label=f"All p{percentile:g} survivors pooled",
        )
        axis.plot(
            lag_values,
            intercept + slope * lag_values,
            linestyle="--",
            linewidth=1.7,
            color="0.45",
            label="Descriptive linear fit",
        )
        annotate_values(axis, lag_values, top_z)
        counts = source_diagnostics["selected_counts"]
        axis.set_title(
            f"{SOURCE_LABELS[source]}\n"
            f"selected n={counts.tolist()}; all components retained\n"
            f"slope={slope:.3f}, descriptive r={correlation:.3f}"
        )
        axis.set_xlabel(
            "Future-location lag (0=current; 1=+1 action; ...; 5=+5)"
        )
        axis.set_xticks(lag_values)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=9, loc="upper left")

    axes[0].set_ylabel("fsLR z-coordinate")
    fig.suptitle(
        f"N480 p{percentile:g} population shift: every surviving component pooled\n"
        "points are selected units; black line is their score-weighted mean z",
        fontsize=15,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_rank1_population_shift(
    results: list[dict[str, Any]],
    diagnostics: dict[str, dict[str, np.ndarray]],
    rank1_diagnostics: dict[str, dict[str, np.ndarray]],
    unit_z: np.ndarray,
    output_path: Path,
    percentile: float,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 6.3),
        sharey=True,
    )
    rng = np.random.default_rng(42)

    for axis, result in zip(axes, results):
        source = result["source"]
        color = SOURCE_COLORS[source]
        lag_values = result["lag_values"]
        score_maps = result["scores"]
        top_masks = diagnostics[source]["top_masks"]
        source_rank1 = rank1_diagnostics[source]
        rank1_masks = source_rank1["rank1_masks"]
        rank1_z = source_rank1["rank1_z"]

        for lag_index, lag_value in enumerate(lag_values):
            selected_indices = np.flatnonzero(top_masks[lag_index])
            rank1_indices = np.flatnonzero(rank1_masks[lag_index])
            selected_scores = score_maps[lag_index, selected_indices]
            score_range = float(np.ptp(selected_scores))
            scaled_scores = (
                (selected_scores - float(np.min(selected_scores)))
                / score_range
                if score_range > 0
                else np.zeros_like(selected_scores)
            )
            jitter = rng.uniform(-0.12, 0.12, size=selected_indices.size)
            jitter_by_unit = np.zeros(score_maps.shape[1], dtype=float)
            jitter_by_unit[selected_indices] = jitter
            size_by_unit = np.zeros(score_maps.shape[1], dtype=float)
            size_by_unit[selected_indices] = 14.0 + 24.0 * scaled_scores

            axis.scatter(
                np.full(selected_indices.size, lag_value) + jitter,
                unit_z[selected_indices],
                s=size_by_unit[selected_indices],
                color=color,
                alpha=0.13,
                linewidths=0,
                label=(
                    f"All p{percentile:g} survivors"
                    if lag_index == 0
                    else None
                ),
                zorder=1,
            )
            axis.scatter(
                np.full(rank1_indices.size, lag_value)
                + jitter_by_unit[rank1_indices],
                unit_z[rank1_indices],
                s=size_by_unit[rank1_indices] + 24.0,
                color=color,
                alpha=0.90,
                edgecolors="black",
                linewidths=0.55,
                label=(
                    "Rank-1 component units"
                    if lag_index == 0
                    else None
                ),
                zorder=3,
            )

        slope, intercept, correlation = descriptive_trend(
            lag_values,
            rank1_z,
        )
        axis.plot(
            lag_values,
            rank1_z,
            marker="o",
            markersize=8,
            linewidth=2.8,
            color="black",
            label="Rank-1 score-weighted COM z",
            zorder=4,
        )
        axis.plot(
            lag_values,
            intercept + slope * lag_values,
            linestyle="--",
            linewidth=1.7,
            color="0.45",
            label="Descriptive linear fit",
            zorder=2,
        )
        annotate_values(axis, lag_values, rank1_z)
        axis.set_title(
            f"{SOURCE_LABELS[source]}\n"
            f"rank-1 n={source_rank1['rank1_counts'].tolist()}; "
            f"component counts={source_rank1['component_counts'].tolist()}\n"
            f"slope={slope:.3f}, descriptive r={correlation:.3f}"
        )
        axis.set_xlabel(
            "Future-location lag (0=current; 1=+1 action; ...; 5=+5)"
        )
        axis.set_xticks(lag_values)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8.5, loc="upper left")

    axes[0].set_ylabel("fsLR z-coordinate")
    fig.suptitle(
        f"N480 p{percentile:g}: rank-1 surface-component trajectory\n"
        "faint points are all 48 survivors; dark points alone determine COM z",
        fontsize=15,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_unit_preferred_lag(
    results: list[dict[str, Any]],
    unit_z: np.ndarray,
    output_path: Path,
    tie_tolerance: float,
) -> dict[str, dict[str, Any]]:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.8),
        sharey=True,
    )
    rng = np.random.default_rng(7)
    summaries: dict[str, dict[str, Any]] = {}

    for axis, result in zip(axes, results):
        source = result["source"]
        color = SOURCE_COLORS[source]
        lag_values = result["lag_values"]
        preferred_index, valid = calculate_preferred_lags(
            result["scores"],
            tie_tolerance=tie_tolerance,
        )
        preferred_values = np.full(preferred_index.size, np.nan, dtype=float)
        preferred_values[valid] = lag_values[preferred_index[valid]]
        jitter = rng.uniform(-0.14, 0.14, size=int(np.sum(valid)))

        axis.scatter(
            preferred_values[valid] + jitter,
            unit_z[valid],
            s=13,
            color=color,
            alpha=0.27,
            linewidths=0,
            label="Units with a unique preference",
        )

        means = []
        medians = []
        counts = []

        for lag_index, lag_value in enumerate(lag_values):
            members = valid & (preferred_index == lag_index)
            counts.append(int(np.sum(members)))
            means.append(
                float(np.mean(unit_z[members]))
                if np.any(members)
                else np.nan
            )
            medians.append(
                float(np.median(unit_z[members]))
                if np.any(members)
                else np.nan
            )

        means_array = np.asarray(means, dtype=float)
        medians_array = np.asarray(medians, dtype=float)
        axis.plot(
            lag_values,
            means_array,
            marker="o",
            markersize=7,
            linewidth=2.5,
            color="black",
            label="Mean z within preferred-lag group",
        )
        axis.plot(
            lag_values,
            medians_array,
            marker="s",
            markersize=5,
            linewidth=1.5,
            linestyle="--",
            color="0.45",
            label="Median z",
        )

        if np.sum(valid) >= 2 and np.std(preferred_values[valid]) > 0:
            correlation = float(
                np.corrcoef(preferred_values[valid], unit_z[valid])[0, 1]
            )
        else:
            correlation = np.nan

        axis.set_title(
            f"{SOURCE_LABELS[source]}\n"
            f"valid preference n={int(np.sum(valid))}/{valid.size}; "
            f"counts={counts}\n"
            f"unitwise descriptive r={correlation:.3f}"
        )
        axis.set_xlabel(
            "Preferred future-location lag (argmax; 0=current)"
        )
        axis.set_xticks(lag_values)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8, loc="upper left")

        summaries[source] = {
            "preferred_index": preferred_index,
            "valid": valid,
            "counts": counts,
            "means": means,
            "medians": medians,
            "correlation": correlation,
        }

    axes[0].set_ylabel("Unit fsLR z-coordinate")
    fig.suptitle(
        "N480 unit preferred lag versus fsLR z\n"
        "all units considered; no percentile threshold or components",
        fontsize=15,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return summaries


def write_lag_csv(
    output_path: Path,
    results: list[dict[str, Any]],
    diagnostics: dict[str, dict[str, np.ndarray]],
    percentile: float,
) -> None:
    rows = []

    for result in results:
        source = result["source"]
        source_diagnostics = diagnostics[source]

        for lag_index, lag_value in enumerate(result["lag_values"]):
            rows.append(
                {
                    "score_source": source,
                    "lag_index": lag_index,
                    "lag_value": float(lag_value),
                    "lag_label": str(result["lag_labels"][lag_index]),
                    "n_units": int(result["scores"].shape[1]),
                    "whole_map_weighted_z": float(
                        source_diagnostics["whole_z"][lag_index]
                    ),
                    "top_percentile": float(percentile),
                    "top_cutoff": float(
                        source_diagnostics["cutoffs"][lag_index]
                    ),
                    "n_top_units": int(
                        source_diagnostics["selected_counts"][lag_index]
                    ),
                    "top_all_components_weighted_z": float(
                        source_diagnostics["top_z"][lag_index]
                    ),
                    "top_score_mass_fraction": float(
                        source_diagnostics["top_mass_fraction"][lag_index]
                    ),
                }
            )

    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_rank1_csv(
    output_path: Path,
    results: list[dict[str, Any]],
    diagnostics: dict[str, dict[str, np.ndarray]],
    rank1_diagnostics: dict[str, dict[str, np.ndarray]],
    percentile: float,
) -> None:
    rows = []

    for result in results:
        source = result["source"]
        lag_diagnostics = diagnostics[source]
        source_rank1 = rank1_diagnostics[source]

        for lag_index, lag_value in enumerate(result["lag_values"]):
            rows.append(
                {
                    "score_source": source,
                    "lag_index": lag_index,
                    "lag_value": float(lag_value),
                    "lag_label": str(result["lag_labels"][lag_index]),
                    "top_percentile": float(percentile),
                    "top_cutoff": float(
                        lag_diagnostics["cutoffs"][lag_index]
                    ),
                    "n_top_units": int(
                        lag_diagnostics["selected_counts"][lag_index]
                    ),
                    "n_surface_components": int(
                        source_rank1["component_counts"][lag_index]
                    ),
                    "n_rank1_units": int(
                        source_rank1["rank1_counts"][lag_index]
                    ),
                    "rank1_score_mass": float(
                        source_rank1["rank1_mass"][lag_index]
                    ),
                    "rank1_fraction_of_selected_score_mass": float(
                        source_rank1["rank1_mass_fraction"][lag_index]
                    ),
                    "rank1_score_weighted_com_z": float(
                        source_rank1["rank1_z"][lag_index]
                    ),
                }
            )

    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_preferred_csv(
    output_path: Path,
    results: list[dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
    unit_z: np.ndarray,
) -> None:
    rows = []

    for result in results:
        source = result["source"]
        summary = summaries[source]
        preferred = summary["preferred_index"]
        valid = summary["valid"]

        for lag_index, lag_value in enumerate(result["lag_values"]):
            members = valid & (preferred == lag_index)
            member_z = unit_z[members]
            rows.append(
                {
                    "score_source": source,
                    "preferred_lag_index": lag_index,
                    "preferred_lag_value": float(lag_value),
                    "n_units": int(member_z.size),
                    "mean_z": (
                        float(np.mean(member_z)) if member_z.size else np.nan
                    ),
                    "median_z": (
                        float(np.median(member_z)) if member_z.size else np.nan
                    ),
                    "sd_z": (
                        float(np.std(member_z)) if member_z.size else np.nan
                    ),
                    "n_valid_preferences": int(np.sum(valid)),
                    "n_undefined_preferences": int(np.sum(~valid)),
                    "unitwise_pearson_r_descriptive": float(
                        summary["correlation"]
                    ),
                }
            )

    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot population-preserving spatial-gradient diagnostics from "
            "saved raw-activity and normalized-Csubs lag maps."
        )
    )
    parser.add_argument("raw_result_dir")
    parser.add_argument("subspace_result_dir")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument(
        "--embedding_dir",
        default=None,
        help=(
            "Embedding folder containing the unit vertex indices. If "
            "omitted, use embedding_dir from the subspace run_config.json."
        ),
    )
    parser.add_argument(
        "--surface",
        default=str(DEFAULT_SURFACE),
        help="fsLR surface GIFTI path, resolved from --repo_root.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Diagnostic output directory. Default: "
            "<common result parent>/population_gradient_diagnostics."
        ),
    )
    parser.add_argument("--top_percentile", type=float, default=90.0)
    parser.add_argument("--z_bins", type=int, default=24)
    parser.add_argument("--preference_tie_tolerance", type=float, default=1e-12)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 20 < args.top_percentile < 100:
        parser.error("--top_percentile must be strictly between 20 and 100.")

    if args.z_bins < 5:
        parser.error("--z_bins must be at least 5.")

    if (
        not np.isfinite(args.preference_tie_tolerance)
        or args.preference_tie_tolerance < 0
    ):
        parser.error("--preference_tie_tolerance must be finite and >= 0.")

    repo_root = Path(args.repo_root).expanduser().resolve()
    raw_dir = resolve_path(args.raw_result_dir, repo_root)
    subspace_dir = resolve_path(args.subspace_result_dir, repo_root)
    raw_result = load_result(raw_dir, "raw_activity")
    subspace_result = load_result(subspace_dir, "subspace")
    results = [raw_result, subspace_result]

    if subspace_result["config"].get("subspace_array") != "Csubs":
        raise ValueError(
            "The subspace result is not a Csubs run according to "
            f"run_config.json: {subspace_dir}"
        )

    for key in ("analysis_dir", "embedding_dir", "model_argument", "model_stem"):
        raw_value = raw_result["config"].get(key)
        subspace_value = subspace_result["config"].get(key)

        if raw_value is None or subspace_value is None:
            raise ValueError(
                f"Both result configurations must identify {key!r}."
            )

        if raw_value != subspace_value:
            raise ValueError(
                f"Raw and Csubs result configurations disagree on {key!r}: "
                f"{raw_value!r} versus {subspace_value!r}."
            )

    if raw_result["scores"].shape != subspace_result["scores"].shape:
        raise ValueError(
            "Raw and Csubs score maps have different shapes: "
            f"{raw_result['scores'].shape} versus "
            f"{subspace_result['scores'].shape}."
        )

    if not np.array_equal(
        raw_result["lag_values"],
        subspace_result["lag_values"],
    ):
        raise ValueError("Raw and Csubs lag values are not identical.")

    n_units = int(raw_result["scores"].shape[1])

    if args.embedding_dir is None:
        saved_embedding = subspace_result["config"].get("embedding_dir")

        if saved_embedding is None:
            raise ValueError(
                "Subspace run_config.json lacks embedding_dir; provide "
                "--embedding_dir explicitly."
            )

        embedding_dir = resolve_path(saved_embedding, repo_root)
    else:
        embedding_dir = resolve_path(args.embedding_dir, repo_root)

    surface_path = resolve_path(args.surface, repo_root)
    unit_vertices = find_unit_vertices(
        embedding_dir=embedding_dir,
        n_units=n_units,
    )
    unit_z = load_unit_z(
        surface_path=surface_path,
        embedding_dir=embedding_dir,
        n_units=n_units,
    )
    adjacency = load_unit_surface_adjacency(
        surface_path=surface_path,
        embedding_dir=embedding_dir,
        unit_vertices=unit_vertices,
    )

    if args.output_dir is None:
        if raw_dir.parent != subspace_dir.parent:
            raise ValueError(
                "Result folders do not share a parent; provide --output_dir."
            )

        output_dir = raw_dir.parent / "population_gradient_diagnostics"
    else:
        output_dir = resolve_path(args.output_dir, repo_root)

    if output_dir.is_dir() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Diagnostic output directory is non-empty: {output_dir}\n"
            "Use --overwrite only for an intentional exact rerun."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = {
        result["source"]: calculate_lag_diagnostics(
            result["scores"],
            unit_z,
            percentile=args.top_percentile,
        )
        for result in results
    }
    rank1_diagnostics = {
        result["source"]: calculate_rank1_diagnostics(
            scores=result["scores"],
            unit_z=unit_z,
            top_masks=diagnostics[result["source"]]["top_masks"],
            adjacency=adjacency,
        )
        for result in results
    }

    for result in results:
        verify_saved_rank1_com(
            result=result,
            rank1_diagnostics=rank1_diagnostics[result["source"]],
        )

    all_units_plot = output_dir / "all_units_population_shift_z.png"
    all_selected_counts = np.concatenate(
        [
            diagnostics[result["source"]]["selected_counts"]
            for result in results
        ]
    )
    unique_selected_counts = np.unique(all_selected_counts)
    top_stem = (
        f"top{int(unique_selected_counts[0])}"
        if unique_selected_counts.size == 1
        else f"p{args.top_percentile:g}_survivors"
    )
    top_plot = output_dir / (
        f"{top_stem}_all_components_population_shift_z.png"
    )
    rank1_plot = output_dir / (
        f"{top_stem}_rank1_all_components_population_shift_z.png"
    )
    preferred_plot = output_dir / "unit_preferred_lag_vs_z.png"

    plot_all_unit_population_shift(
        results=results,
        diagnostics=diagnostics,
        unit_z=unit_z,
        output_path=all_units_plot,
        n_z_bins=args.z_bins,
    )
    plot_top_survivor_population_shift(
        results=results,
        diagnostics=diagnostics,
        unit_z=unit_z,
        output_path=top_plot,
        percentile=args.top_percentile,
    )
    plot_rank1_population_shift(
        results=results,
        diagnostics=diagnostics,
        rank1_diagnostics=rank1_diagnostics,
        unit_z=unit_z,
        output_path=rank1_plot,
        percentile=args.top_percentile,
    )
    preferred_summaries = plot_unit_preferred_lag(
        results=results,
        unit_z=unit_z,
        output_path=preferred_plot,
        tie_tolerance=args.preference_tie_tolerance,
    )

    lag_csv = output_dir / "lag_population_weighted_z.csv"
    rank1_csv = output_dir / f"{top_stem}_rank1_population_shift_z.csv"
    preferred_csv = output_dir / "preferred_lag_z_summary.csv"
    write_lag_csv(
        output_path=lag_csv,
        results=results,
        diagnostics=diagnostics,
        percentile=args.top_percentile,
    )
    write_rank1_csv(
        output_path=rank1_csv,
        results=results,
        diagnostics=diagnostics,
        rank1_diagnostics=rank1_diagnostics,
        percentile=args.top_percentile,
    )
    write_preferred_csv(
        output_path=preferred_csv,
        results=results,
        summaries=preferred_summaries,
        unit_z=unit_z,
    )

    run_config = {
        "raw_result_dir": str(raw_dir),
        "subspace_result_dir": str(subspace_dir),
        "embedding_dir": str(embedding_dir),
        "surface": str(surface_path),
        "output_dir": str(output_dir),
        "n_units": n_units,
        "lag_values": raw_result["lag_values"].tolist(),
        "top_percentile": args.top_percentile,
        "z_bins": args.z_bins,
        "preference_tie_tolerance": args.preference_tie_tolerance,
        "interpretation": {
            "whole_map_weighted_z": (
                "score-weighted mean unit fsLR-z using every unit position; "
                "no threshold or connected-component selection"
            ),
            "top_all_components_weighted_z": (
                "score-weighted mean unit fsLR-z across every strict "
                "within-lag percentile survivor; no component discarded"
            ),
            "rank1_score_weighted_com_z": (
                "score-weighted fsLR-z centre of mass of the connected "
                "component with greatest raw score mass; all percentile "
                "survivors remain visible only as plot context"
            ),
            "preferred_lag": (
                "per-unit unique argmax across future-location lag score "
                "maps; all-zero and tied preferences are undefined"
            ),
        },
    }

    with open(output_dir / "diagnostic_run_config.json", "w") as handle:
        json.dump(run_config, handle, indent=2, sort_keys=True)

    print(f"[SAVE] {all_units_plot}")
    print(f"[SAVE] {top_plot}")
    print(f"[SAVE] {rank1_plot}")
    print(f"[SAVE] {preferred_plot}")
    print(f"[SAVE] {lag_csv}")
    print(f"[SAVE] {rank1_csv}")
    print(f"[SAVE] {preferred_csv}")
    print(f"[SAVE] {output_dir / 'diagnostic_run_config.json'}")

    for result in results:
        source = result["source"]
        source_diagnostics = diagnostics[source]
        print(f"\n{SOURCE_LABELS[source]}")
        print(
            "  whole-map weighted z: "
            + ", ".join(
                f"{value:.2f}" for value in source_diagnostics["whole_z"]
            )
        )
        print(
            f"  p{args.top_percentile:g}, all-components weighted z: "
            + ", ".join(
                f"{value:.2f}" for value in source_diagnostics["top_z"]
            )
        )
        print(
            "  selected counts: "
            f"{source_diagnostics['selected_counts'].tolist()}"
        )
        print(
            "  rank-1 weighted COM z: "
            + ", ".join(
                f"{value:.2f}"
                for value in rank1_diagnostics[source]["rank1_z"]
            )
        )
        print(
            "  rank-1 counts: "
            f"{rank1_diagnostics[source]['rank1_counts'].tolist()}"
        )

    print(f"\nDONE: {output_dir}")


if __name__ == "__main__":
    main()
