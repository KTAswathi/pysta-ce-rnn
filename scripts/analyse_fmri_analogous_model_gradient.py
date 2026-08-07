"""
fMRI-analogous gradient analysis for a cortically embedded RNN.

It mirrors the fMRI analysis script ('https://github.com/skuechenhoff/multiple_clocks_repo') 
as closely as possible, replacing voxel-volume operations with surface-unit operations:

    fMRI map per lag/condition
        -> model lag-specific score map over RNN units / fsLR vertices

The three fMRI-analogous spatial measures are:

    mode="voxel"
        -> unthresholded global maximum model unit / surface vertex

    mode="cluster_peak"
        -> strongest unit inside a thresholded connected component, after
           ranking components by score mass

    mode="cluster_com"
        -> score-weighted centre of mass of that same ranked component

Three alternative score-map sources are available before that common spatial
pipeline. They are alternative neural map definitions, not the three spatial
measures above:

    score_source="subspace" (existing/default analysis)
        -> norm of planning-subspace coefficients over locations

    score_source="raw_activity"
        -> RMS variation of mean planning activity across future-location
           conditions, computed separately for each lag

    score_source="raw_activity_by_planning_delay"
        -> assign each uniquely selective unit to the planning delay at which
           its mean raw activity is maximal. This is a separate internal-
           planning-clock analysis, not a future-state-lag analogue of the
           fMRI maps.

    MNI x/y/z coordinate
        -> fsLR/Conte69 surface x/y/z coordinate
           IMPORTANT: these are not labelled as MNI coordinates here.

The main fMRI analogous question is:

    Does one coordinate axis, e.g. surface z, change with delay/lag?

Prerequisites:
    # Needed by all score sources:
    python scripts/analyse_rnn.py "$MODEL" collect

    # Additionally needed only for score_source=subspace:
    python scripts/analyse_rnn.py "$MODEL" subspaces

Typical usage:
    MODEL="MazeEnv.../model0"

    # Existing analysis, unchanged in meaning:
    PYTHONPATH="$PWD" python scripts/analyse_fmri_analogous_model_gradient.py "$MODEL" \
        --score_source subspace --axis z --cluster_threshold 90 --n_clusters 3

    # Raw planning-activity condition-effect maps:
    PYTHONPATH="$PWD" python scripts/analyse_fmri_analogous_model_gradient.py "$MODEL" \
        --score_source raw_activity --planning_times=-2,-1 \
        --lag_times=0,1,2,3,4,5 --axis z --cluster_threshold 90 --n_clusters 3

    # Separate exploratory internal planning-clock preference analysis:
    PYTHONPATH="$PWD" python scripts/analyse_fmri_analogous_model_gradient.py "$MODEL" \
        --score_source raw_activity_by_planning_delay \
        --axis z --cluster_threshold 90 --n_clusters 3
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
from collections import deque
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.freesurfer.io import read_annot
from scipy.stats import pearsonr, spearmanr


SURF_REL = Path(
    "data/embedding/raw_surface_data/human/fs_lr32/surf/fs_lr32.l.midthickness.surf.gii"
)
ANNOT_REL = Path(
    "data/embedding/raw_surface_data/human/fs_lr32/label/lh.HCPMMP1.annot"
)
SULC_REL = Path(
    "data/embedding/raw_surface_data/human/fs_lr32/fs_lr32.l.sulc_data.func.gii"
)

ANALYSIS_REL = Path("data/rnn_analyses")
EMBEDDING_REL = Path("data/embedding/subsampled/human")

PEAK_MODES = ["voxel", "cluster_peak", "cluster_com"]
SCORE_SOURCES = ["subspace", "raw_activity", "raw_activity_by_planning_delay"]
SCORE_SOURCE_LABELS = {
    "subspace": "planning-subspace coefficient norm",
    "raw_activity": (
        "future-location tuning effect from raw planning activity"
    ),
    "raw_activity_by_planning_delay": (
        "preferred planning-clock delay selectivity"
    ),
}

VERTEX_FILE_CANDIDATES = [
    "sampled_indices.npy",
    "sampled_vertex_indices.npy",
    "unit_vertex_indices.npy",
    "vertex_indices.npy",
    "roi_vertex_indices.npy",
]

SUBSPACE_KEY_CANDIDATES = [
    "Csubs",
    "Csubs_raw",
    "C_subs",
    "Csub",
    "C_sub",
    "C",
    "coefficients",
    "subspaces",
]


def normalise_model_reference(
    model_arg: str,
    repo_root: Path,
) -> tuple[Path, str]:
    """
    Return ``(model_directory_relative_to_repo_data, model_stem)``.

    Keeping the stem is essential: analysis folders can contain model0,
    model1, ... artifacts, and silently falling back to model0 would mix
    independently trained networks. Both repository-relative and absolute
    paths below ``models/`` or ``data/rnn_analyses/`` are accepted.
    """
    p = Path(model_arg).expanduser()

    if p.is_absolute():
        resolved = p.resolve()
        relative = None

        for base in (
            repo_root / "models",
            repo_root / ANALYSIS_REL,
        ):
            try:
                relative = resolved.relative_to(base.resolve())
                break
            except ValueError:
                continue

        if relative is None:
            raise ValueError(
                "An absolute --model path must be inside either:\n"
                f"  {repo_root / 'models'}\n"
                f"  {repo_root / ANALYSIS_REL}\n"
                f"Received: {resolved}"
            )

        p = relative
    else:
        parts = list(p.parts)

        if parts and parts[0] == "models":
            p = Path(*parts[1:])
        elif (
            len(parts) >= 2
            and parts[0] == "data"
            and parts[1] == "rnn_analyses"
        ):
            p = Path(*parts[2:])

    if p.suffix:
        if p.suffix != ".p":
            raise ValueError(
                "Model files must be saved .p files; received "
                f"{p.name!r}."
            )

        model_stem = p.stem
        model_dir_rel = p.parent
    elif re.fullmatch(r"model\d+", p.name):
        model_stem = p.name
        model_dir_rel = p.parent
    else:
        raise ValueError(
            "The model argument must include an explicit model stem, for "
            "example '.../model0' or '.../model0.p'. This prevents a model "
            "directory containing several seeds from silently selecting "
            "model0."
        )

    if re.fullmatch(r"model\d+", model_stem) is None:
        raise ValueError(
            "Could not identify a model stem such as model0 or model31 "
            f"from {model_arg!r}."
        )

    return model_dir_rel, model_stem


def parse_model_name(model_dir_rel: Path) -> tuple[int, str, int]:
    """
    Parse N, embedding_name and embedding_seed from cortical model directory.

    Expected final folder example:
      N480_linout_cortical_mpfc_projected_mask_linear0p1_eseed42_ld0.12_...
    """
    name = model_dir_rel.name
    m = re.search(
        r"N(?P<n>\d+)_.*?_cortical_(?P<embedding>.+?)_eseed(?P<seed>\d+)",
        name,
    )
    if m is None:
        raise ValueError(
            "Could not parse N / embedding_name / embedding_seed from "
            "model folder name:\n"
            f"  {name}\n"
            "Expected something like:\n"
            "  N480_..._cortical_<embedding_name>_eseed42_..."
        )

    return (
        int(m.group("n")),
        m.group("embedding"),
        int(m.group("seed")),
    )


def find_embedding_dir(
    repo_root: Path,
    embedding_name: str,
    n_units: int,
    seed: int,
) -> Path:
    embedding_dir = (
        repo_root
        / EMBEDDING_REL
        / embedding_name
        / f"units={n_units}_seed={seed}"
    )

    if not embedding_dir.is_dir():
        raise FileNotFoundError(
            "Could not find embedding directory:\n"
            f"  {embedding_dir}\n"
            "Check model path / N / embedding name / seed."
        )

    return embedding_dir


def find_unit_vertices(
    embedding_dir: Path,
    n_units: int,
) -> np.ndarray:
    """Load fsLR vertex index for each model unit."""
    for fname in VERTEX_FILE_CANDIDATES:
        fpath = embedding_dir / fname

        if not fpath.is_file():
            continue

        arr = np.load(fpath)
        arr = np.asarray(arr).astype(int).ravel()

        if arr.size == n_units:
            print(f"[INFO] Loaded unit vertices from: {fpath}")
            return arr

        print(
            f"[WARN] {fname} has wrong size: "
            f"{arr.size} != {n_units}"
        )

    existing = "\n".join(
        f"  {p.name}"
        for p in sorted(embedding_dir.glob("*.npy"))
    )

    raise FileNotFoundError(
        "Could not find a vertex-index file with length n_units.\n"
        f"Embedding dir: {embedding_dir}\n"
        f"Existing npy files:\n{existing}"
    )


def find_planning_subspaces_pickle(
    analysis_dir: Path,
    model_stem: str,
) -> Path:
    exact = analysis_dir / f"{model_stem}_planning_subspaces.pickle"

    if exact.is_file():
        return exact

    legacy = analysis_dir / "planning_subspaces.pickle"

    if model_stem == "model0" and legacy.is_file():
        print(
            "[WARN] Using legacy unprefixed planning-subspaces file: "
            f"{legacy}"
        )
        return legacy

    existing = "\n".join(
        f"  {p.name}"
        for p in sorted(analysis_dir.glob("*.pickle"))
    )

    raise FileNotFoundError(
        "Could not find planning subspaces pickle. Run first:\n"
        f'  python scripts/analyse_rnn.py ".../{model_stem}" '
        "collect time decoding subspaces\n\n"
        f"Expected exact file: {exact}\n"
        f"Analysis dir: {analysis_dir}\n"
        f"Existing pickle files:\n"
        f"{existing if existing else '  <none>'}"
    )


def find_trial_data_pickle(
    analysis_dir: Path,
    model_stem: str,
) -> Path:
    """Find the trial-data pickle produced by analyse_rnn.py collect."""
    exact = analysis_dir / f"{model_stem}_trial_data.pickle"

    if exact.is_file():
        return exact

    legacy = analysis_dir / "trial_data.pickle"

    if model_stem == "model0" and legacy.is_file():
        print(
            "[WARN] Using legacy unprefixed trial-data file: "
            f"{legacy}"
        )
        return legacy

    existing = "\n".join(
        f"  {p.name}"
        for p in sorted(analysis_dir.glob("*.pickle"))
    )

    raise FileNotFoundError(
        "Could not find trial_data.pickle. Run first:\n"
        f'  python scripts/analyse_rnn.py ".../{model_stem}" collect\n\n'
        f"Expected exact file: {exact}\n"
        f"Analysis dir: {analysis_dir}\n"
        f"Existing pickle files:\n"
        f"{existing if existing else '  <none>'}"
    )


def load_surface(
    repo_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    surf_path = repo_root / SURF_REL

    if not surf_path.is_file():
        raise FileNotFoundError(
            f"Missing surface file: {surf_path}"
        )

    gii = nib.load(str(surf_path))
    coords, faces = gii.agg_data()

    return (
        np.asarray(coords, dtype=float),
        np.asarray(faces, dtype=int),
    )


def load_sulc(
    repo_root: Path,
) -> np.ndarray | None:
    """Load fsLR sulcal depth background for surface QC plots."""
    sulc_path = repo_root / SULC_REL

    if not sulc_path.is_file():
        print(
            f"[WARN] Missing sulc file for surface QC plot: "
            f"{sulc_path}"
        )
        return None

    sulc_gii = nib.load(str(sulc_path))

    return np.asarray(
        sulc_gii.darrays[0].data,
        dtype=float,
    ).reshape(-1)


def model_values_to_full_surface(
    values: np.ndarray,
    unit_vertices: np.ndarray,
    n_vertices: int,
    vertex_to_cluster: np.ndarray | None = None,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Map model-unit values back to the full fsLR surface.

    If a full vertex-to-unit assignment exists, broadcast each unit value over
    the vertices represented by that unit. Otherwise, paint only the sampled
    representative vertices. Non-model vertices receive ``fill_value``.
    """
    full = np.full(
        n_vertices,
        fill_value,
        dtype=float,
    )

    values = np.asarray(values, dtype=float).reshape(-1)

    if vertex_to_cluster is not None:
        assignments = np.asarray(
            vertex_to_cluster,
            dtype=int,
        ).reshape(-1)

        if assignments.size != n_vertices:
            raise ValueError(
                "vertex_to_cluster length does not match the surface: "
                f"{assignments.size} != {n_vertices}."
            )

        valid = (
            (assignments >= 0)
            & (assignments < values.size)
        )
        full[valid] = values[assignments[valid]]
    else:
        full[np.asarray(unit_vertices, dtype=int)] = values

    return full


def plot_delay_score_surface_qc(
    *,
    delay_scores: np.ndarray,
    delay_labels: list[str],
    surface_coords: np.ndarray,
    faces: np.ndarray,
    bg_sulc: np.ndarray | None,
    unit_vertices: np.ndarray,
    vertex_to_cluster: np.ndarray | None,
    out_dir: Path,
    score_source_label: str,
) -> None:
    """
    Surface QC plot for model analogue.

    Each panel shows the continuous delay-specific model score map used for
    voxel / cluster_peak / cluster_com extraction.

    This is NOT an anchor-distance map.
    """
    try:
        from nilearn import plotting as nilearn_plotting
    except ImportError as exc:
        raise ImportError(
            "--brain_plots 1 requires nilearn, but nilearn could not be "
            "imported. Install it in the active environment, or explicitly "
            "rerun with --brain_plots 0."
        ) from exc

    n_delays = delay_scores.shape[0]
    ncols = min(3, n_delays)
    nrows = int(np.ceil(n_delays / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.2 * ncols, 3.6 * nrows),
        subplot_kw={"projection": "3d"},
    )

    axes = np.asarray(axes).reshape(-1)

    all_finite = delay_scores[np.isfinite(delay_scores)]
    vmax = (
        float(np.nanmax(all_finite))
        if all_finite.size
        else 1.0
    )
    vmin = 0.0

    for d in range(n_delays):
        ax = axes[d]

        full_map = model_values_to_full_surface(
            values=delay_scores[d],
            unit_vertices=unit_vertices,
            n_vertices=surface_coords.shape[0],
            vertex_to_cluster=vertex_to_cluster,
            fill_value=0.0,
        )

        nilearn_plotting.plot_surf_stat_map(
            surf_mesh=(surface_coords, faces),
            stat_map=full_map,
            hemi="left",
            view="medial",
            bg_map=bg_sulc,
            bg_on_data=True,
            cmap="YlOrRd",
            threshold=1e-12,
            vmin=vmin,
            vmax=vmax,
            colorbar=False,
            axes=ax,
            title=f"delay {delay_labels[d]}",
        )

    for ax in axes[n_delays:]:
        ax.axis("off")

    fig.suptitle(
        "fMRI-style model analogue: lag-specific score maps\n"
        f"source: {score_source_label} | "
        "fsLR/Conte69 surface; not MNI",
        fontsize=13,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))

    out_path = (
        out_dir
        / "delay_score_maps_medial_surface.png"
    )

    fig.savefig(
        out_path,
        dpi=200,
    )
    plt.close(fig)

    print(f"[SAVE] {out_path}")


def plot_strongest_cluster_surface_qc(
    *,
    delay_scores: np.ndarray,
    delay_labels: list[str],
    surface_coords: np.ndarray,
    faces: np.ndarray,
    bg_sulc: np.ndarray | None,
    unit_vertices: np.ndarray,
    vertex_to_cluster: np.ndarray | None,
    adjacency: list[set[int]],
    cluster_threshold: str | float,
    out_dir: Path,
    score_source_label: str,
) -> None:
    """
    Surface QC plot showing only the strongest thresholded cluster per delay.

    This is the surface analogue of strongest cluster extraction.
    """
    try:
        from nilearn import plotting as nilearn_plotting
    except ImportError as exc:
        raise ImportError(
            "--brain_plots 1 requires nilearn, but nilearn could not be "
            "imported. Install it in the active environment, or explicitly "
            "rerun with --brain_plots 0."
        ) from exc

    n_delays = delay_scores.shape[0]
    ncols = min(3, n_delays)
    nrows = int(np.ceil(n_delays / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.2 * ncols, 3.6 * nrows),
        subplot_kw={"projection": "3d"},
    )

    axes = np.asarray(axes).reshape(-1)

    all_cluster_values = []
    cluster_maps = []
    cluster_titles = []

    for d in range(n_delays):
        score_map = np.asarray(
            delay_scores[d],
            dtype=float,
        )

        (
            _binary,
            actual_threshold,
            components,
        ) = threshold_and_rank_components(
            score_map=score_map,
            threshold=cluster_threshold,
            adjacency=adjacency,
        )

        cluster_values = np.zeros_like(
            score_map,
            dtype=float,
        )

        if len(components) > 0:
            strongest = components[0]
            cluster_values[strongest] = score_map[strongest]
            threshold_label = (
                f"{actual_threshold:.3g}"
                if isinstance(actual_threshold, float)
                else str(actual_threshold)
            )

            all_cluster_values.extend(
                score_map[strongest][
                    np.isfinite(score_map[strongest])
                ]
            )

            title = (
                f"delay {delay_labels[d]} | rank-1 n={len(strongest)}\n"
                f"mass={np.sum(score_map[strongest]):.3g}, "
                f"cut={threshold_label}"
            )
        else:
            title = (
                f"delay {delay_labels[d]}\n"
                "no cluster found"
            )

        cluster_maps.append(cluster_values)
        cluster_titles.append(title)

    all_cluster_values = np.asarray(
        all_cluster_values,
        dtype=float,
    )

    vmax = (
        float(np.nanmax(all_cluster_values))
        if all_cluster_values.size
        else 1.0
    )
    vmin = 0.0

    for d in range(n_delays):
        ax = axes[d]

        full_map = model_values_to_full_surface(
            values=cluster_maps[d],
            unit_vertices=unit_vertices,
            n_vertices=surface_coords.shape[0],
            vertex_to_cluster=vertex_to_cluster,
            fill_value=0.0,
        )

        nilearn_plotting.plot_surf_stat_map(
            surf_mesh=(surface_coords, faces),
            stat_map=full_map,
            hemi="left",
            view="medial",
            bg_map=bg_sulc,
            bg_on_data=True,
            cmap="YlOrRd",
            threshold=1e-12,
            vmin=vmin,
            vmax=vmax,
            colorbar=False,
            axes=ax,
            title=cluster_titles[d],
        )

    for ax in axes[n_delays:]:
        ax.axis("off")

    fig.suptitle(
        "fMRI-style model analogue: "
        "strongest thresholded cluster per lag\n"
        f"source: {score_source_label} | "
        f"cluster threshold = {cluster_threshold}",
        fontsize=13,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))

    out_path = (
        out_dir
        / "strongest_cluster_maps_medial_surface.png"
    )

    fig.savefig(
        out_path,
        dpi=200,
    )
    plt.close(fig)

    print(f"[SAVE] {out_path}")


def simplify_label_name(
    name: str | bytes,
) -> str:
    if isinstance(name, bytes):
        name = name.decode("utf-8")

    name = str(name)

    if name.startswith("L_") and name.endswith("_ROI"):
        return name[2:-4]

    return name


def load_unit_labels(
    repo_root: Path,
    unit_vertices: np.ndarray,
) -> np.ndarray:
    annot_path = repo_root / ANNOT_REL

    if not annot_path.is_file():
        print(
            f"[WARN] Missing annotation file: {annot_path}"
        )

        return np.array(
            ["unknown"] * len(unit_vertices),
            dtype=object,
        )

    labels, _ctab, raw_names = read_annot(
        str(annot_path),
        orig_ids=False,
    )

    names = [
        simplify_label_name(x)
        for x in raw_names
    ]

    unit_labels = []

    for vertex in unit_vertices:
        label_idx = int(labels[int(vertex)])

        if 0 <= label_idx < len(names):
            unit_labels.append(names[label_idx])
        else:
            unit_labels.append("unknown")

    return np.asarray(
        unit_labels,
        dtype=object,
    )


def load_vertex_to_cluster(
    embedding_dir: Path,
    n_vertices: int,
    n_units: int,
) -> np.ndarray | None:
    """
    Load the full-surface vertex-to-model-unit assignment, when available.
    """
    path = embedding_dir / "vertex_to_cluster.npy"

    if not path.is_file():
        print(
            "[WARN] vertex_to_cluster.npy was not found; "
            "falling back to exact sampled-vertex adjacency."
        )
        return None

    vertex_to_cluster = np.asarray(
        np.load(path),
        dtype=int,
    ).ravel()

    if vertex_to_cluster.size != n_vertices:
        raise ValueError(
            "vertex_to_cluster.npy has the wrong number "
            "of surface vertices: "
            f"{vertex_to_cluster.size} != {n_vertices}"
        )

    valid = vertex_to_cluster >= 0

    if np.any(
        vertex_to_cluster[valid] >= n_units
    ):
        raise ValueError(
            "vertex_to_cluster.npy contains cluster indices "
            "outside the model-unit range "
            f"0..{n_units - 1}."
        )

    print(
        "[INFO] Loaded full-surface unit assignments from: "
        f"{path}"
    )

    return vertex_to_cluster


def validate_surface_mapping(
    surface_coords: np.ndarray,
    faces: np.ndarray,
    unit_vertices: np.ndarray,
    vertex_to_cluster: np.ndarray | None,
    n_units: int,
) -> None:
    """Fail early on malformed or inconsistent surface/unit mappings."""
    if surface_coords.ndim != 2 or surface_coords.shape[1] != 3:
        raise ValueError(
            "Surface coordinates must have shape (vertices, 3), got "
            f"{surface_coords.shape}."
        )

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"Surface faces must have shape (faces, 3), got {faces.shape}."
        )

    if np.any((faces < 0) | (faces >= surface_coords.shape[0])):
        raise ValueError("Surface faces contain out-of-range vertex indices.")

    vertices = np.asarray(unit_vertices, dtype=int).reshape(-1)

    if vertices.size != n_units:
        raise ValueError(
            f"Expected {n_units} unit vertices, got {vertices.size}."
        )

    if np.unique(vertices).size != vertices.size:
        raise ValueError("Unit representative vertices are not unique.")

    if np.any((vertices < 0) | (vertices >= surface_coords.shape[0])):
        raise ValueError("Unit vertices contain out-of-range surface indices.")

    if vertex_to_cluster is None:
        return

    assignments = np.asarray(vertex_to_cluster, dtype=int).reshape(-1)
    represented = np.unique(assignments[assignments >= 0])
    expected = np.arange(n_units, dtype=int)

    if not np.array_equal(represented, expected):
        missing = np.setdiff1d(expected, represented)
        raise ValueError(
            "vertex_to_cluster does not represent every model unit. "
            f"Missing units: {missing[:20].tolist()}"
        )

    representative_assignments = assignments[vertices]

    if not np.array_equal(representative_assignments, expected):
        raise ValueError(
            "Each sampled unit vertex must map back to its own unit index in "
            "vertex_to_cluster.npy."
        )


def build_unit_adjacency(
    faces: np.ndarray,
    unit_vertices: np.ndarray,
    vertex_to_cluster: np.ndarray | None = None,
) -> list[set[int]]:
    """
    Build surface adjacency among model units.

    When vertex_to_cluster is available, two model units are adjacent when
    their full-surface clusters meet across a mesh triangle. Using only the
    sampled representative vertices would leave most units isolated.

    If the full-surface assignment is unavailable, fall back to the previous
    exact sampled-vertex adjacency.
    """
    n_units = len(unit_vertices)

    adjacency: list[set[int]] = [
        set()
        for _ in range(n_units)
    ]

    if vertex_to_cluster is not None:
        vertex_to_cluster = np.asarray(
            vertex_to_cluster,
            dtype=int,
        ).ravel()

        for tri in faces:
            clusters = np.unique(
                vertex_to_cluster[
                    np.asarray(tri, dtype=int)
                ]
            )

            clusters = clusters[
                (clusters >= 0)
                & (clusters < n_units)
            ]

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    a = int(clusters[i])
                    b = int(clusters[j])

                    if a == b:
                        continue

                    adjacency[a].add(b)
                    adjacency[b].add(a)

        return adjacency

    vertex_to_unit = {
        int(v): i
        for i, v in enumerate(unit_vertices)
    }

    for tri in faces:
        present = [
            vertex_to_unit[int(v)]
            for v in tri
            if int(v) in vertex_to_unit
        ]

        if len(present) < 2:
            continue

        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                a = present[i]
                b = present[j]

                adjacency[a].add(b)
                adjacency[b].add(a)

    return adjacency


def connected_components(
    mask: np.ndarray,
    adjacency: list[set[int]],
) -> list[np.ndarray]:
    mask = np.asarray(
        mask,
        dtype=bool,
    )

    visited = np.zeros(
        mask.size,
        dtype=bool,
    )

    components: list[np.ndarray] = []

    for start in np.where(mask)[0]:
        if visited[start]:
            continue

        queue = deque([int(start)])
        visited[start] = True
        comp = []

        while queue:
            u = queue.popleft()
            comp.append(u)

            for v in adjacency[u]:
                if mask[v] and not visited[v]:
                    visited[v] = True
                    queue.append(v)

        components.append(
            np.asarray(
                comp,
                dtype=int,
            )
        )

    return components


def find_ndarrays_with_unit_axis(
    obj: Any,
    n_units: int,
    path: str = "root",
) -> list[tuple[str, np.ndarray]]:
    hits: list[tuple[str, np.ndarray]] = []

    if isinstance(obj, np.ndarray):
        if obj.ndim == 3 and n_units in obj.shape:
            hits.append((path, obj))

        return hits

    if isinstance(obj, dict):
        for key, value in obj.items():
            hits.extend(
                find_ndarrays_with_unit_axis(
                    value,
                    n_units,
                    f"{path}.{key}",
                )
            )

    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            hits.extend(
                find_ndarrays_with_unit_axis(
                    value,
                    n_units,
                    f"{path}[{i}]",
                )
            )

    return hits


def load_pickle_with_dependency_hint(path: Path) -> Any:
    """Load a repository pickle with an actionable dependency error."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Could not unpickle {path} because Python module "
            f"{exc.name!r} is unavailable. Use the repository's fully "
            "provisioned environment (currently `conda run -n pysta`) or "
            "install the missing requirement."
        ) from exc


def load_csubs(
    planning_subspaces_path: Path,
    n_units: int,
    requested_key: str = "Csubs",
) -> tuple[np.ndarray, str]:
    obj = load_pickle_with_dependency_hint(
        planning_subspaces_path
    )

    if requested_key != "auto":
        if not isinstance(obj, dict):
            raise TypeError(
                "An explicit --subspace_array requires a dictionary "
                f"pickle, got {type(obj)!r}."
            )

        if requested_key not in obj:
            raise KeyError(
                f"Requested subspace array {requested_key!r} was not found "
                f"in {planning_subspaces_path}. Available keys: "
                f"{sorted(map(str, obj.keys()))}"
            )

        arr = np.asarray(obj[requested_key])

        if arr.ndim != 3 or n_units not in arr.shape:
            raise ValueError(
                f"Subspace array {requested_key!r} must be 3-D with an "
                f"axis of length {n_units}; got {arr.shape}."
            )

        return arr, requested_key

    if isinstance(obj, dict):
        for key in SUBSPACE_KEY_CANDIDATES:
            if (
                key in obj
                and isinstance(obj[key], np.ndarray)
            ):
                arr = np.asarray(obj[key])

                if (
                    arr.ndim == 3
                    and n_units in arr.shape
                ):
                    return arr, key

    hits = find_ndarrays_with_unit_axis(
        obj,
        n_units=n_units,
    )

    if not hits:
        raise ValueError(
            "Could not find a 3D subspace array with "
            "one axis equal to n_units="
            f"{n_units} in {planning_subspaces_path}"
        )

    print(
        "[INFO] Candidate subspace arrays found:"
    )

    for path, arr in hits:
        print(
            f"  {path}: shape={arr.shape}"
        )

    return (
        np.asarray(hits[0][1]),
        hits[0][0],
    )


def csubs_to_delay_scores(
    csubs: np.ndarray,
    n_units: int,
    delay_axis: str = "auto",
) -> np.ndarray:
    """
    Convert Csubs to a continuous delay-score map: delay x unit.

    Usual pysta shape is delay x location x unit.
    Score for delay k and unit i is the norm of that unit's
    subspace contribution over encoded locations.
    """
    arr = np.asarray(
        csubs,
        dtype=float,
    )

    if arr.ndim != 3:
        raise ValueError(
            f"Expected 3D Csubs, got shape {arr.shape}"
        )

    unit_axes = [
        ax
        for ax, size in enumerate(arr.shape)
        if size == n_units
    ]

    if not unit_axes:
        raise ValueError(
            f"No Csubs axis matches n_units={n_units}; "
            f"shape={arr.shape}"
        )

    if len(unit_axes) != 1:
        raise ValueError(
            "Csubs has more than one axis matching n_units, so the unit "
            f"axis is ambiguous: shape={arr.shape}, axes={unit_axes}."
        )

    unit_axis = unit_axes[0]

    if unit_axis != 2:
        arr = np.moveaxis(
            arr,
            unit_axis,
            2,
        )

        print(
            f"[INFO] Moved unit axis {unit_axis} "
            f"to last axis; new shape={arr.shape}"
        )

    if delay_axis == "auto":
        print(
            "[WARN] --delay_axis auto uses a shape heuristic. The pysta "
            "planning-subspace layout is delay,location,unit; prefer "
            "--delay_axis 0 for reproducibility."
        )
        delay_ax = (
            0
            if arr.shape[0] <= arr.shape[1]
            else 1
        )
    else:
        delay_ax = int(delay_axis)

        if delay_ax not in (0, 1):
            raise ValueError(
                "--delay_axis must be auto, 0 or 1"
            )

    if delay_ax == 0:
        delay_scores = np.linalg.norm(
            arr,
            axis=1,
        )
    else:
        delay_scores = np.linalg.norm(
            arr,
            axis=0,
        )

    print(
        f"[INFO] Csubs original shape: {csubs.shape}"
    )
    print(
        f"[INFO] delay_scores shape: "
        f"{delay_scores.shape}  # delay x unit"
    )

    return delay_scores


def validate_delay_scores(
    delay_scores: np.ndarray,
    n_units: int,
) -> np.ndarray:
    """Validate the common condition-by-unit score-map contract."""
    scores = np.asarray(delay_scores, dtype=float).copy()

    if scores.ndim != 2 or scores.shape[1] != n_units:
        raise ValueError(
            "Delay scores must have shape (conditions, units) with "
            f"{n_units} units; got {scores.shape}."
        )

    if scores.shape[0] < 2:
        raise ValueError("At least two condition maps are required.")

    if not np.all(np.isfinite(scores)):
        bad = np.argwhere(~np.isfinite(scores))
        raise ValueError(
            "Delay score maps must share one finite full-ROI support. "
            f"First non-finite entries: {bad[:10].tolist()}"
        )

    tolerance = 1e-12

    if np.nanmin(scores) < -tolerance:
        raise ValueError(
            "Supported score maps must be non-negative so component mass "
            "and weighted COM use the same values."
        )

    scores[scores < 0] = 0.0
    return scores


def parse_int_list(
    value: str | None,
    argument_name: str,
) -> list[int] | None:
    """Parse a comma-separated integer list."""
    if value is None or value.strip() == "":
        return None

    try:
        values = [
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        ]
    except ValueError as exc:
        raise ValueError(
            f"{argument_name} must be "
            "comma-separated integers."
        ) from exc

    if not values:
        raise ValueError(
            f"{argument_name} did not contain any integers."
        )

    if len(values) != len(set(values)):
        raise ValueError(
            f"{argument_name} contains duplicate values: "
            f"{values}"
        )

    return values


def load_trial_data(
    trial_data_path: Path,
    n_units: int,
) -> dict[str, Any]:
    trial_data = load_pickle_with_dependency_hint(
        trial_data_path
    )

    if not isinstance(trial_data, dict):
        raise TypeError(
            "Expected trial_data to be a dict, got "
            f"{type(trial_data)!r}"
        )

    required = [
        "rs",
        "step_nums",
        "locs",
    ]

    missing = [
        key
        for key in required
        if key not in trial_data
    ]

    if missing:
        raise KeyError(
            f"trial_data is missing required keys: {missing}"
        )

    rs = np.asarray(
        trial_data["rs"]
    )

    if (
        rs.ndim != 3
        or rs.shape[-1] != n_units
    ):
        raise ValueError(
            "Expected trial_data['rs'] shape "
            f"(trials, time, {n_units}), "
            f"got {rs.shape}"
        )

    return trial_data


def aligned_step_values(
    step_nums: np.ndarray,
) -> np.ndarray:
    """
    Return the common task-time value represented by each stored time index.
    """
    step_nums = np.asarray(
        step_nums,
        dtype=float,
    )

    if (
        step_nums.ndim == 3
        and step_nums.shape[-1] == 1
    ):
        step_nums = step_nums[..., 0]

    if step_nums.ndim != 2:
        raise ValueError(
            "Expected step_nums shape (trials, time) "
            "or (trials, time, 1), "
            f"got {step_nums.shape}"
        )

    step_values = np.nanmean(
        step_nums,
        axis=0,
    )

    variability = np.nanstd(
        step_nums,
        axis=0,
    )

    finite_variability = variability[
        np.isfinite(variability)
    ]

    if (
        finite_variability.size
        and np.nanmax(finite_variability) > 1e-6
    ):
        raise ValueError(
            "Trials are not aligned to the same step_num "
            "at each stored time index. "
            "This analysis currently requires aligned trial_data."
        )

    return step_values


def find_time_indices(
    step_values: np.ndarray,
    requested_times: list[int],
    name: str,
) -> list[int]:
    indices: list[int] = []

    for requested in requested_times:
        matches = np.where(
            np.isfinite(step_values)
            & np.isclose(step_values, requested)
        )[0]

        if matches.size != 1:
            available = [
                int(round(x))
                for x in step_values[
                    np.isfinite(step_values)
                ]
            ]

            raise ValueError(
                "Could not identify exactly one stored index for "
                f"{name}={requested}. "
                f"Available aligned step values: {available}"
            )

        indices.append(int(matches[0]))

    return indices


def prepare_activity_lag_data(
    trial_data: dict[str, Any],
    planning_times: list[int],
    lag_times: list[int] | None,
    activity_standardization: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[int],
    int,
]:
    """
    Extract one planning-activity vector and one future-location label
    per lag/trial.
    """
    rs = np.asarray(
        trial_data["rs"],
    )

    step_nums = np.asarray(
        trial_data["step_nums"],
        dtype=float,
    )

    locs = np.asarray(
        trial_data["locs"],
        dtype=float,
    )

    if (
        locs.ndim == 3
        and locs.shape[-1] == 1
    ):
        locs = locs[..., 0]

    if locs.ndim != 2:
        raise ValueError(
            "Expected locs shape (trials, time) "
            "or (trials, time, 1), "
            f"got {locs.shape}"
        )

    step_values = aligned_step_values(
        step_nums
    )

    planning_indices = find_time_indices(
        step_values,
        planning_times,
        "planning_time",
    )

    if lag_times is None:
        lag_times = sorted(
            {
                int(round(value))
                for value in step_values
                if (
                    np.isfinite(value)
                    and value >= 0
                    and np.isclose(value, round(value))
                )
            }
        )

        if not lag_times:
            raise ValueError(
                "No non-negative lag times were found "
                "in trial_data."
            )

    lag_indices = find_time_indices(
        step_values,
        lag_times,
        "lag_time",
    )

    selected_planning_activity = np.asarray(
        rs[:, planning_indices, :],
        dtype=float,
    )

    if activity_standardization == "zscore":
        mean = np.nanmean(
            selected_planning_activity,
            axis=(0, 1),
            keepdims=True,
        )
        sd = np.nanstd(
            selected_planning_activity,
            axis=(0, 1),
            keepdims=True,
        )
        selected_planning_activity = (
            selected_planning_activity - mean
        ) / (sd + 1e-10)
    elif activity_standardization != "none":
        raise ValueError(
            "activity_standardization must be 'none' or 'zscore'."
        )

    with np.errstate(invalid="ignore"):
        planning_activity = np.nanmean(
            selected_planning_activity,
            axis=1,
        )

    if not np.all(np.isfinite(planning_activity)):
        bad = np.argwhere(~np.isfinite(planning_activity))
        raise ValueError(
            "Planning activity contains non-finite trial/unit values. "
            "This implementation refuses to drop a whole trial because one "
            f"unit is missing. First entries: {bad[:10].tolist()}"
        )

    lag_locations = locs[:, lag_indices]

    finite_locs = locs[
        np.isfinite(locs)
    ]

    if finite_locs.size == 0:
        raise ValueError(
            "No finite location labels found in trial_data."
        )

    num_locs = int(
        np.nanmax(finite_locs)
    ) + 1

    print(
        f"[INFO] Planning times: {planning_times} "
        f"-> stored indices {planning_indices}"
    )

    print(
        f"[INFO] Lag times:      {lag_times} "
        f"-> stored indices {lag_indices}"
    )

    print(
        f"[INFO] planning_activity shape: "
        f"{planning_activity.shape}"
    )

    print(
        f"[INFO] lag_locations shape: "
        f"{lag_locations.shape}"
    )

    print(
        f"[INFO] inferred num_locs: {num_locs}"
    )

    print(
        "[INFO] activity standardization: "
        f"{activity_standardization}"
    )

    return (
        planning_activity,
        lag_locations,
        lag_times,
        num_locs,
    )


def raw_activity_to_delay_scores(
    planning_activity: np.ndarray,
    lag_locations: np.ndarray,
    num_locs: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Construct a raw-activity condition-effect map for every lag.

    For each lag and unit:
      1. compute mean planning activity for every future-location condition;
      2. centre those condition means across locations;
      3. take their root-mean-square magnitude.
    """
    n_lags = lag_locations.shape[1]
    n_units = planning_activity.shape[1]

    delay_scores = np.full(
        (n_lags, n_units),
        np.nan,
        dtype=float,
    )

    n_trials_per_lag: list[int] = []
    n_locations_per_lag: list[int] = []

    activity_valid = np.all(
        np.isfinite(planning_activity),
        axis=1,
    )

    for lag_index in range(n_lags):
        locations = lag_locations[:, lag_index]

        valid = (
            activity_valid
            & np.isfinite(locations)
        )

        y = planning_activity[valid]
        raw_labels = locations[valid]

        if not np.all(
            np.isclose(raw_labels, np.round(raw_labels))
        ):
            raise ValueError(
                f"Lag index {lag_index} contains non-integer location "
                "labels; refusing to truncate them."
            )

        labels = np.round(raw_labels).astype(int)

        if np.any((labels < 0) | (labels >= num_locs)):
            raise ValueError(
                f"Lag index {lag_index} contains location labels outside "
                f"0..{num_locs - 1}."
            )

        condition_means = np.full(
            (num_locs, n_units),
            np.nan,
            dtype=float,
        )

        for location in range(num_locs):
            condition_trials = labels == location

            if np.any(condition_trials):
                condition_means[location] = np.mean(
                    y[condition_trials],
                    axis=0,
                )

        present = np.all(
            np.isfinite(condition_means),
            axis=1,
        )

        if np.sum(present) < 2:
            raise ValueError(
                f"Lag index {lag_index} has fewer than "
                "two represented location conditions."
            )

        effects = condition_means[present]
        effects = (
            effects
            - np.mean(
                effects,
                axis=0,
                keepdims=True,
            )
        )

        delay_scores[lag_index] = np.sqrt(
            np.mean(
                effects**2,
                axis=0,
            )
        )

        n_trials_per_lag.append(
            int(np.sum(valid))
        )

        n_locations_per_lag.append(
            int(np.sum(present))
        )

        print(
            f"[RAW] lag_index={lag_index}: "
            f"n_trials={np.sum(valid)}, "
            f"represented_locations="
            f"{np.sum(present)}/{num_locs}"
        )

    metadata = {
        "n_trials_per_lag": n_trials_per_lag,
        "n_locations_per_lag": n_locations_per_lag,
        "definition": (
            "RMS of centred future-location-conditioned "
            "mean planning activity"
        ),
    }

    return delay_scores, metadata


def raw_activity_by_planning_delay_to_scores(
    trial_data: dict[str, Any],
    planning_times: list[int] | None,
    activity_standardization: str,
    preference_tie_tolerance: float,
) -> tuple[
    np.ndarray,
    list[int],
    dict[str, Any],
]:
    """
    Construct preferred-planning-delay raw-activity maps.

    1. Average every unit's activity across trials at every negative
       planning step.
    2. Assign a unit only when one delay has a unique maximum separated from
       the second-highest delay by more than ``preference_tie_tolerance``.
    3. Give every delay map the same full-ROI support: assigned units receive
       their max-minus-second-max margin and all other units receive zero.

    Delay order is nearest future first:
      step_num -1 -> delay 0
      step_num -2 -> delay 1
      and so on.
    """
    rs = np.asarray(
        trial_data["rs"],
    )

    step_values = aligned_step_values(
        np.asarray(
            trial_data["step_nums"],
            dtype=float,
        )
    )

    if planning_times is None:
        planning_times = sorted(
            {
                int(round(value))
                for value in step_values
                if (
                    np.isfinite(value)
                    and value < 0
                    and np.isclose(value, round(value))
                )
            },
            reverse=True,
        )

        if not planning_times:
            raise ValueError(
                "No negative planning step_nums were found "
                "in trial_data."
            )

    else:
        if any(
            step >= 0
            for step in planning_times
        ):
            raise ValueError(
                "--planning_times for "
                "raw_activity_by_planning_delay must contain "
                "only negative planning step_nums."
            )

        planning_times = sorted(
            planning_times,
            reverse=True,
        )

    planning_indices = find_time_indices(
        step_values,
        planning_times,
        "planning_time",
    )

    selected_activity = np.asarray(
        rs[:, planning_indices, :],
        dtype=float,
    )

    if activity_standardization == "zscore":
        mean = np.nanmean(
            selected_activity,
            axis=(0, 1),
            keepdims=True,
        )

        sd = np.nanstd(
            selected_activity,
            axis=(0, 1),
            keepdims=True,
        )

        selected_activity = (
            selected_activity - mean
        ) / (sd + 1e-10)

    elif activity_standardization != "none":
        raise ValueError(
            "activity_standardization must be "
            "'none' or 'zscore'."
        )

    with np.errstate(invalid="ignore"):
        mean_activity = np.nanmean(
            selected_activity,
            axis=0,
        )

    if mean_activity.ndim != 2:
        raise ValueError(
            "Expected mean planning activity to have "
            "shape delay x unit, got "
            f"{mean_activity.shape}."
        )

    finite_all = np.all(
        np.isfinite(mean_activity),
        axis=0,
    )

    filled_activity = mean_activity.copy()
    filled_activity[~np.isfinite(filled_activity)] = -np.inf

    n_delays, n_units = filled_activity.shape

    if n_delays < 2:
        raise ValueError(
            "Preferred planning delay requires at least two planning times."
        )

    argmax_delay = np.argmax(
        filled_activity,
        axis=0,
    )
    maximum = np.max(
        filled_activity,
        axis=0,
    )
    second_maximum = np.sort(
        filled_activity,
        axis=0,
    )[-2]
    preference_margin = maximum - second_maximum
    n_at_maximum = np.sum(
        np.isclose(
            filled_activity,
            maximum[None, :],
            rtol=0.0,
            atol=preference_tie_tolerance,
        ),
        axis=0,
    )
    preference_valid = (
        finite_all
        & (n_at_maximum == 1)
        & np.isfinite(preference_margin)
        & (preference_margin > preference_tie_tolerance)
    )
    preferred_delay = np.full(n_units, -1, dtype=int)
    preferred_delay[preference_valid] = argmax_delay[preference_valid]

    # Keep a fixed, finite 480-unit spatial support for every condition, as
    # the masked fMRI maps do. Non-preferring/excluded units have score zero.
    delay_scores = np.zeros(
        (n_delays, n_units),
        dtype=float,
    )

    preferred_counts = []

    for delay_index in range(n_delays):
        prefers_delay = (
            preferred_delay == delay_index
        )

        preferred_counts.append(
            int(np.sum(prefers_delay))
        )

        delay_scores[
            delay_index,
            prefers_delay,
        ] = preference_margin[prefers_delay]

    step_to_delay = {
        int(step): delay_index
        for delay_index, step
        in enumerate(planning_times)
    }

    delay_indices = list(
        range(n_delays)
    )

    print(
        "[RAW-BY-PLANNING-DELAY] "
        "step-to-delay mapping:"
    )

    for step, delay_index in step_to_delay.items():
        print(
            f"  step_num {step} "
            f"-> raw delay {delay_index}"
        )

    print(
        f"[INFO] Planning times: {planning_times} "
        f"-> stored indices {planning_indices}"
    )

    print(
        f"[INFO] mean_activity shape: "
        f"{mean_activity.shape}  # delay x unit"
    )

    print(
        f"[INFO] preferred-delay counts: "
        f"{preferred_counts}"
    )

    print(
        "[INFO] Excluded tied/unselective/non-finite units: "
        f"{int(np.sum(~preference_valid))}/{n_units} "
        f"(tolerance={preference_tie_tolerance:g})"
    )

    print(
        "[INFO] activity standardization: "
        f"{activity_standardization}"
    )

    metadata = {
        "planning_times": list(planning_times),
        "step_to_delay": step_to_delay,
        "preferred_delay": preferred_delay,
        "preference_margin": preference_margin,
        "preference_valid": preference_valid,
        "n_excluded_preference_units": int(
            np.sum(~preference_valid)
        ),
        "preference_tie_tolerance": preference_tie_tolerance,
        "preferred_counts": preferred_counts,
        "activity_standardization": activity_standardization,
        "definition": (
            "each unit is assigned to the negative planning "
            "step at which its mean raw activity across trials "
            "is uniquely maximal; each map has fixed full-ROI support, "
            "with max-minus-second-max activity as the preference score "
            "for units preferring that delay and zero elsewhere"
        ),
    }

    return (
        delay_scores,
        delay_indices,
        metadata,
    )


def make_activity_delay_scores(
    trial_data_path: Path,
    n_units: int,
    planning_times: list[int],
    lag_times: list[int] | None,
    activity_standardization: str,
) -> tuple[
    np.ndarray,
    list[int],
    dict[str, Any],
]:
    trial_data = load_trial_data(
        trial_data_path,
        n_units=n_units,
    )

    (
        planning_activity,
        lag_locations,
        actual_lag_times,
        num_locs,
    ) = prepare_activity_lag_data(
        trial_data=trial_data,
        planning_times=planning_times,
        lag_times=lag_times,
        activity_standardization=activity_standardization,
    )

    delay_scores, metadata = raw_activity_to_delay_scores(
        planning_activity=planning_activity,
        lag_locations=lag_locations,
        num_locs=num_locs,
    )

    metadata.update(
        {
            "trial_data_path": str(trial_data_path),
            "planning_times": list(planning_times),
            "lag_times": list(actual_lag_times),
            "activity_standardization": activity_standardization,
            "num_locs": int(num_locs),
            "interpretation_note": (
                "This is a marginal future-location tuning effect, not a "
                "partial or causal lag effect; current location, reward, "
                "trajectory, and neighboring future lags are not controlled."
            ),
        }
    )

    return (
        delay_scores,
        actual_lag_times,
        metadata,
    )


def parse_cluster_threshold(
    value: str | float | int,
) -> str | float:
    """Mirror CLUSTER_THRESHOLD options: 0, 'z', or percentile 20--100."""
    if isinstance(value, str):
        value = value.strip()

        if value.lower() == "z":
            return "z"

        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError(
                "--cluster_threshold must be 0, 'z', "
                "or a percentile strictly between 20 and 100"
            ) from exc
    else:
        parsed = float(value)

    if not np.isfinite(parsed):
        raise ValueError(
            "--cluster_threshold must be finite."
        )

    if parsed == 0 or 20 < parsed < 100:
        return parsed

    raise ValueError(
        "--cluster_threshold must be 0, 'z', or a percentile "
        "strictly between 20 and 100."
    )


def threshold_score_map(
    scores: np.ndarray,
    threshold: str | float,
) -> tuple[np.ndarray, float | str]:
    """Surface-unit version of thresholding."""
    scores = np.asarray(
        scores,
        dtype=float,
    )

    finite_scores = scores[
        np.isfinite(scores)
    ]

    if finite_scores.size == 0:
        raise ValueError(
            "Score map has no finite values."
        )

    if threshold == 0:
        binary = scores > 0
        actual_threshold: float | str = 0.0

    elif threshold == "z":
        sd = float(
            np.std(finite_scores)
        )

        if sd == 0:
            z = np.zeros_like(scores)
        else:
            z = (
                scores
                - float(np.mean(finite_scores))
            ) / sd

        binary = z > 1.0
        actual_threshold = "z>1"

    elif 20 < float(threshold) < 100:
        actual_threshold = float(
            np.percentile(
                finite_scores,
                float(threshold),
            )
        )

        binary = scores > actual_threshold

    else:
        raise ValueError(
            "Cluster threshold must be 0, 'z', "
            "or a percentile strictly between 20 and 100, "
            "matching the fMRI analysis script."
        )

    binary = (
        binary
        & np.isfinite(scores)
    )

    return (
        binary,
        actual_threshold,
    )


def threshold_and_rank_components(
    score_map: np.ndarray,
    threshold: str | float,
    adjacency: list[set[int]],
) -> tuple[np.ndarray, float | str, list[np.ndarray]]:
    """
    Threshold one map and rank connected components by raw score mass.

    This single helper is shared by coordinate extraction, fragmentation
    diagnostics, and surface QC so all three operate on identical clusters.
    """
    score_map = np.asarray(score_map, dtype=float)

    if score_map.ndim != 1:
        raise ValueError(
            f"Expected a one-dimensional score map, got {score_map.shape}."
        )

    if score_map.size != len(adjacency):
        raise ValueError(
            "Score-map length and adjacency size differ: "
            f"{score_map.size} != {len(adjacency)}."
        )

    binary, actual_threshold = threshold_score_map(
        score_map,
        threshold,
    )
    components = connected_components(binary, adjacency)
    components = sorted(
        components,
        key=lambda component: float(
            np.nansum(score_map[component])
        ),
        reverse=True,
    )

    return binary, actual_threshold, components


def weighted_com(
    coords: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    weights = np.asarray(
        weights,
        dtype=float,
    )

    valid = (
        np.all(np.isfinite(coords), axis=1)
        & np.isfinite(weights)
    )

    coords = coords[valid]
    weights = weights[valid]

    if coords.size == 0:
        raise ValueError(
            "Cannot compute a centre of mass without finite coordinates."
        )

    if np.isclose(np.sum(weights), 0.0):
        return np.mean(
            coords,
            axis=0,
        )

    return np.average(
        coords,
        axis=0,
        weights=weights,
    )


def extract_points_for_delay(
    score_map: np.ndarray,
    unit_coords: np.ndarray,
    unit_vertices: np.ndarray,
    unit_labels: np.ndarray,
    adjacency: list[set[int]],
    delay_index: int,
    delay_label: str,
    delay_value: float,
    cluster_threshold: str | float,
    n_clusters: int,
) -> list[dict[str, Any]]:
    """
    Extract voxel / cluster_peak / cluster_com analogues for one delay.
    """
    rows: list[dict[str, Any]] = []

    score_map = np.asarray(
        score_map,
        dtype=float,
    )

    has_positive_signal = bool(np.max(score_map) > 0)

    if not has_positive_signal:
        print(
            f"[WARN] delay={delay_label} has no positive score. A NaN voxel "
            "placeholder is emitted because an argmax coordinate would be "
            "an arbitrary tie; no clusters will be returned."
        )
        peak_unit: int | None = None
        rows.append(
            {
                "delay_index": delay_index,
                "delay_label": delay_label,
                "delay_value": delay_value,
                "mode": "voxel",
                "cluster_rank": 0,
                "unit_index": "",
                "vertex_index": "",
                "parcel_label": "no_positive_signal",
                "x": np.nan,
                "y": np.nan,
                "z": np.nan,
                "score": np.nan,
                "cluster_size": 0,
                "cluster_mass": np.nan,
                "cluster_threshold": "none_for_voxel_mode",
                "nearest_unit_to_com": "",
                "nearest_vertex_to_com": "",
            }
        )
    else:
        peak_unit = int(np.argmax(score_map))
        peak_coord = unit_coords[peak_unit]
        rows.append(
            {
                "delay_index": delay_index,
                "delay_label": delay_label,
                "delay_value": delay_value,
                "mode": "voxel",
                "cluster_rank": 0,
                "unit_index": peak_unit,
                "vertex_index": int(unit_vertices[peak_unit]),
                "parcel_label": str(unit_labels[peak_unit]),
                "x": float(peak_coord[0]),
                "y": float(peak_coord[1]),
                "z": float(peak_coord[2]),
                "score": float(score_map[peak_unit]),
                "cluster_size": 1,
                "cluster_mass": float(score_map[peak_unit]),
                "cluster_threshold": "none_for_voxel_mode",
                "nearest_unit_to_com": "",
                "nearest_vertex_to_com": "",
            }
        )

    (
        _binary,
        actual_threshold,
        components,
    ) = threshold_and_rank_components(
        score_map=score_map,
        threshold=cluster_threshold,
        adjacency=adjacency,
    )

    for rank, comp in enumerate(
        components[:n_clusters],
        start=1,
    ):
        if comp.size == 0:
            continue

        cluster_scores = score_map[comp]

        cluster_mass = float(
            np.nansum(cluster_scores)
        )

        peak_in_cluster = int(
            comp[
                np.nanargmax(cluster_scores)
            ]
        )

        peak_coord = unit_coords[
            peak_in_cluster
        ]

        rows.append(
            {
                "delay_index": delay_index,
                "delay_label": delay_label,
                "delay_value": delay_value,
                "mode": "cluster_peak",
                "cluster_rank": rank,
                "unit_index": peak_in_cluster,
                "vertex_index": int(
                    unit_vertices[peak_in_cluster]
                ),
                "parcel_label": str(
                    unit_labels[peak_in_cluster]
                ),
                "x": float(peak_coord[0]),
                "y": float(peak_coord[1]),
                "z": float(peak_coord[2]),
                "score": float(
                    score_map[peak_in_cluster]
                ),
                "cluster_size": int(comp.size),
                "cluster_mass": cluster_mass,
                "cluster_threshold": actual_threshold,
                "nearest_unit_to_com": "",
                "nearest_vertex_to_com": "",
            }
        )

        com_coord = weighted_com(
            unit_coords[comp],
            cluster_scores,
        )

        nearest_local = int(
            np.argmin(
                np.linalg.norm(
                    unit_coords[comp] - com_coord,
                    axis=1,
                )
            )
        )

        nearest_unit = int(
            comp[nearest_local]
        )

        rows.append(
            {
                "delay_index": delay_index,
                "delay_label": delay_label,
                "delay_value": delay_value,
                "mode": "cluster_com",
                "cluster_rank": rank,
                "unit_index": "",
                "vertex_index": "",
                "parcel_label": "weighted_surface_com",
                "x": float(com_coord[0]),
                "y": float(com_coord[1]),
                "z": float(com_coord[2]),
                "score": "",
                "cluster_size": int(comp.size),
                "cluster_mass": cluster_mass,
                "cluster_threshold": actual_threshold,
                "nearest_unit_to_com": nearest_unit,
                "nearest_vertex_to_com": int(
                    unit_vertices[nearest_unit]
                ),
            }
        )

    for rank in range(len(components) + 1, n_clusters + 1):
        common = {
            "delay_index": delay_index,
            "delay_label": delay_label,
            "delay_value": delay_value,
            "cluster_rank": rank,
            "unit_index": "",
            "vertex_index": "",
            "x": np.nan,
            "y": np.nan,
            "z": np.nan,
            "score": "",
            "cluster_size": 0,
            "cluster_mass": np.nan,
            "cluster_threshold": actual_threshold,
            "nearest_unit_to_com": "",
            "nearest_vertex_to_com": "",
        }
        rows.append(
            {
                **common,
                "mode": "cluster_peak",
                "parcel_label": "no_cluster",
            }
        )
        rows.append(
            {
                **common,
                "mode": "cluster_com",
                "parcel_label": "no_cluster",
            }
        )

    print(
        f"[INFO] delay={delay_label}: "
        f"voxel_peak_unit={peak_unit if peak_unit is not None else 'none'}, "
        f"surface_clusters_found={len(components)}, "
        f"cluster_threshold={actual_threshold}"
    )

    return rows


def extract_all_points(
    delay_scores: np.ndarray,
    delay_labels: list[str],
    delay_values: list[int] | np.ndarray,
    unit_coords: np.ndarray,
    unit_vertices: np.ndarray,
    unit_labels: np.ndarray,
    adjacency: list[set[int]],
    cluster_threshold: str | float,
    n_clusters: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if len(delay_labels) != delay_scores.shape[0]:
        raise ValueError("Delay-label count does not match delay-score maps.")

    if len(delay_values) != delay_scores.shape[0]:
        raise ValueError("Delay-value count does not match delay-score maps.")

    for delay_index in range(
        delay_scores.shape[0]
    ):
        rows.extend(
            extract_points_for_delay(
                score_map=delay_scores[delay_index],
                unit_coords=unit_coords,
                unit_vertices=unit_vertices,
                unit_labels=unit_labels,
                adjacency=adjacency,
                delay_index=delay_index,
                delay_label=delay_labels[delay_index],
                delay_value=float(delay_values[delay_index]),
                cluster_threshold=cluster_threshold,
                n_clusters=n_clusters,
            )
        )

    return rows


def fragmentation_rows(
    delay_scores: np.ndarray,
    delay_labels: list[str],
    delay_values: list[int] | np.ndarray,
    adjacency: list[set[int]],
    cluster_threshold: str | float,
    score_source: str,
) -> list[dict[str, Any]]:
    """
    Summarise spatial fragmentation of every thresholded lag map.
    """
    rows: list[dict[str, Any]] = []

    if (
        len(delay_labels) != delay_scores.shape[0]
        or len(delay_values) != delay_scores.shape[0]
    ):
        raise ValueError(
            "Delay labels/values do not match fragmentation score maps."
        )

    for delay_index, score_map in enumerate(
        delay_scores
    ):
        (
            binary,
            actual_threshold,
            components,
        ) = threshold_and_rank_components(
            score_map=score_map,
            threshold=cluster_threshold,
            adjacency=adjacency,
        )

        component_sizes = np.asarray(
            [
                len(comp)
                for comp in components
            ],
            dtype=int,
        )

        n_selected = int(
            np.sum(binary)
        )

        n_clusters = int(
            len(components)
        )

        largest_size = (
            int(component_sizes.max())
            if component_sizes.size
            else 0
        )

        n_singletons = (
            int(
                np.sum(component_sizes == 1)
            )
            if component_sizes.size
            else 0
        )

        component_masses = np.asarray(
            [
                float(np.sum(score_map[component]))
                for component in components
            ],
            dtype=float,
        )
        rank1_mass = (
            float(component_masses[0])
            if component_masses.size
            else np.nan
        )
        rank2_mass = (
            float(component_masses[1])
            if component_masses.size > 1
            else np.nan
        )
        rank1_mass_margin = (
            (rank1_mass - rank2_mass) / rank1_mass
            if (
                np.isfinite(rank1_mass)
                and np.isfinite(rank2_mass)
                and rank1_mass != 0
            )
            else np.nan
        )

        rows.append(
            {
                "score_source": score_source,
                "delay_index": delay_index,
                "delay_label": delay_labels[delay_index],
                "delay_value": float(delay_values[delay_index]),
                "cluster_threshold": actual_threshold,
                "n_selected_units": n_selected,
                "n_clusters": n_clusters,
                "largest_cluster_size": largest_size,
                "largest_cluster_fraction": (
                    largest_size / n_selected
                    if n_selected
                    else np.nan
                ),
                "n_singleton_clusters": n_singletons,
                "singleton_unit_fraction": (
                    n_singletons / n_selected
                    if n_selected
                    else np.nan
                ),
                "mean_cluster_size": (
                    float(np.mean(component_sizes))
                    if component_sizes.size
                    else np.nan
                ),
                "median_cluster_size": (
                    float(np.median(component_sizes))
                    if component_sizes.size
                    else np.nan
                ),
                "cluster_sizes": ";".join(
                    map(
                        str,
                        component_sizes.tolist(),
                    )
                ),
                "rank1_cluster_mass": rank1_mass,
                "rank2_cluster_mass": rank2_mass,
                "rank1_mass_margin_fraction": rank1_mass_margin,
            }
        )

    return rows


def print_fragmentation_summary(
    rows: list[dict[str, Any]],
) -> None:
    print(
        "\nlag | selected | clusters | largest | "
        "largest fraction | singletons | singleton fraction"
    )

    print("-" * 86)

    for row in rows:
        print(
            f"{str(row['delay_label']):>3s} | "
            f"{int(row['n_selected_units']):8d} | "
            f"{int(row['n_clusters']):8d} | "
            f"{int(row['largest_cluster_size']):7d} | "
            f"{float(row['largest_cluster_fraction']):16.3f} | "
            f"{int(row['n_singleton_clusters']):10d} | "
            f"{float(row['singleton_unit_fraction']):18.3f}"
        )


def rows_to_projection(
    rows: list[dict[str, Any]],
    mode: str,
    cluster_rank: int,
    axis: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
    np.ndarray,
]:
    selected = [
        r
        for r in rows
        if (
            r["mode"] == mode
            and int(r["cluster_rank"]) == int(cluster_rank)
        )
    ]

    selected = sorted(
        selected,
        key=lambda r: int(r["delay_index"]),
    )

    delay_indices = np.asarray(
        [int(r["delay_index"]) for r in selected],
        dtype=float,
    )
    delay_values = np.asarray(
        [float(r["delay_value"]) for r in selected],
        dtype=float,
    )

    labels = [
        str(r["delay_label"])
        for r in selected
    ]

    values = np.asarray(
        [
            float(r[axis])
            for r in selected
        ],
        dtype=float,
    )

    if np.unique(delay_indices).size != delay_indices.size:
        raise ValueError(
            f"Duplicate rows found for mode={mode}, rank={cluster_rank}."
        )

    return delay_indices, delay_values, labels, values


def plot_projection(
    rows: list[dict[str, Any]],
    out_dir: Path,
    mode: str,
    cluster_rank: int,
    axis: str,
    title_prefix: str,
    trend_x: str,
    x_label: str,
) -> dict[str, Any]:
    (
        delay_indices,
        delay_values,
        labels,
        values,
    ) = rows_to_projection(
        rows,
        mode=mode,
        cluster_rank=cluster_rank,
        axis=axis,
    )

    if len(values) == 0:
        return {
            "mode": mode,
            "cluster_rank": cluster_rank,
            "axis": axis,
            "trend_x": trend_x,
            "n_finite": 0,
        }

    if trend_x == "condition_index":
        x = delay_indices
    elif trend_x == "lag_value":
        x = delay_values
    else:
        raise ValueError(
            "trend_x must be 'condition_index' or 'lag_value'."
        )

    finite = np.isfinite(x) & np.isfinite(values)

    if finite.sum() >= 2 and np.unique(x[finite]).size >= 2:
        slope, intercept = np.polyfit(
            x[finite],
            values[finite],
            1,
        )

        if np.std(values[finite]) > 0:
            pearson_r, pearson_p = pearsonr(
                x[finite],
                values[finite],
            )
            spearman_r, spearman_p = spearmanr(
                x[finite],
                values[finite],
            )
        else:
            pearson_r = np.nan
            pearson_p = np.nan
            spearman_r = np.nan
            spearman_p = np.nan
    else:
        slope = np.nan
        intercept = np.nan
        pearson_r = np.nan
        pearson_p = np.nan
        spearman_r = np.nan
        spearman_p = np.nan

    fig, ax = plt.subplots(
        figsize=(5.8, 5.2)
    )

    ax.plot(
        x,
        values,
        marker="o",
        linewidth=2.5,
        color="black",
    )

    if finite.sum() >= 2:
        ax.plot(
            x[finite],
            intercept + slope * x[finite],
            linestyle="--",
            linewidth=2,
            color="gray",
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        labels,
        rotation=35,
        ha="right",
    )

    ax.set_xlabel(x_label)

    ax.set_ylabel(
        f"fsLR {axis}-coordinate"
    )

    ax.set_title(
        f"{title_prefix}\n"
        f"{mode}, cluster_rank={cluster_rank} | "
        f"slope={slope:.3g}, "
        f"descriptive r={pearson_r:.2f}",
        fontsize=11,
    )

    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    out_path = (
        out_dir
        / f"{mode}_rank{cluster_rank}_"
        f"delay_vs_surface_{axis}.png"
    )

    fig.savefig(
        out_path,
        dpi=200,
    )

    plt.close(fig)

    print(f"[SAVE] {out_path}")

    print(
        f"[TREND] mode={mode:12s} "
        f"rank={cluster_rank} "
        f"axis={axis} "
        f"slope={slope: .4g} "
        f"Pearson r={pearson_r: .3f} "
        f"naive_p={pearson_p: .4g} "
        f"Spearman rho={spearman_r: .3f} "
        f"naive_p={spearman_p: .4g}; descriptive single-model trend"
    )

    return {
        "mode": mode,
        "cluster_rank": cluster_rank,
        "axis": axis,
        "trend_x": trend_x,
        "n_finite": int(np.sum(finite)),
        "slope": float(slope),
        "intercept": float(intercept),
        "pearson_r": float(pearson_r),
        "pearson_p_naive_noninferential": float(pearson_p),
        "spearman_rho": float(spearman_r),
        "spearman_p_naive_noninferential": float(spearman_p),
    }


def make_primary_plots(
    rows: list[dict[str, Any]],
    out_dir: Path,
    axis: str,
    title_prefix: str,
    trend_x: str,
    x_label: str,
) -> list[dict[str, Any]]:
    metrics = []

    metrics.append(plot_projection(
        rows,
        out_dir,
        mode="voxel",
        cluster_rank=0,
        axis=axis,
        title_prefix=title_prefix,
        trend_x=trend_x,
        x_label=x_label,
    ))

    metrics.append(plot_projection(
        rows,
        out_dir,
        mode="cluster_peak",
        cluster_rank=1,
        axis=axis,
        title_prefix=title_prefix,
        trend_x=trend_x,
        x_label=x_label,
    ))

    metrics.append(plot_projection(
        rows,
        out_dir,
        mode="cluster_com",
        cluster_rank=1,
        axis=axis,
        title_prefix=title_prefix,
        trend_x=trend_x,
        x_label=x_label,
    ))

    return metrics


def write_csv(
    rows: list[dict[str, Any]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        raise ValueError(
            "No rows to write."
        )

    with open(
        out_path,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"[SAVE] {out_path}")


def parse_delay_labels(
    delay_labels: str | None,
    n_delays: int,
) -> list[str]:
    if (
        delay_labels is None
        or delay_labels.strip() == ""
    ):
        return [
            str(i)
            for i in range(n_delays)
        ]

    labels = [
        x.strip()
        for x in delay_labels.split(",")
        if x.strip()
    ]

    if len(labels) != n_delays:
        raise ValueError(
            f"--delay_labels supplied {len(labels)} labels, "
            f"but delay map has {n_delays} delays."
        )

    return labels


def value_tag(value: Any) -> str:
    """Create a filesystem-safe, readable configuration tag."""
    text = str(value).strip().replace("-", "m").replace(".", "p")
    text = re.sub(r"[^A-Za-z0-9_]+", "-", text)
    return text.strip("-") or "none"


def sequence_tag(values: list[int] | np.ndarray) -> str:
    return "-".join(value_tag(int(value)) for value in values)


def build_output_name(
    *,
    args: argparse.Namespace,
    model_dir_rel: Path,
    model_stem: str,
    csubs_key: str | None,
    actual_lag_times: list[int],
    score_metadata: dict[str, Any],
) -> str:
    """Name output folders from every result-changing CLI setting."""
    if args.score_source == "subspace":
        source_detail = (
            f"subspace-{value_tag(csubs_key)}_"
            f"daxis-{value_tag(args.delay_axis)}_"
            f"lags-{sequence_tag(actual_lag_times)}"
        )
    elif args.score_source == "raw_activity":
        source_detail = (
            "raw_activity_"
            f"plan-{sequence_tag(score_metadata['planning_times'])}_"
            f"lags-{sequence_tag(actual_lag_times)}_"
            f"std-{value_tag(args.activity_standardization)}"
        )
    else:
        source_detail = (
            "raw_activity_by_planning_delay_"
            f"plan-{sequence_tag(score_metadata['planning_times'])}_"
            f"std-{value_tag(args.activity_standardization)}_"
            f"tietol-{value_tag(args.preference_tie_tolerance)}"
        )

    label_detail = (
        f"_labels-{value_tag(args.delay_labels)}"
        if args.delay_labels is not None
        else ""
    )
    fingerprint_payload = {
        "model_dir_rel": str(model_dir_rel),
        "model_stem": model_stem,
        "score_source": args.score_source,
        "subspace_array_key": csubs_key,
        "delay_axis": args.delay_axis,
        "actual_lag_times": list(map(int, actual_lag_times)),
        "planning_times": score_metadata.get("planning_times"),
        "activity_standardization": args.activity_standardization,
        "preference_tie_tolerance": args.preference_tie_tolerance,
        "axis": args.axis,
        "trend_x": args.trend_x,
        "cluster_threshold": str(args.cluster_threshold),
        "n_clusters": args.n_clusters,
        "brain_plots": args.brain_plots,
        "delay_labels": args.delay_labels,
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:10]

    return (
        f"{model_stem}_fmri_score-{source_detail}_"
        f"axis-{args.axis}_trend-{args.trend_x}_"
        f"thr-{value_tag(args.cluster_threshold)}_"
        f"nclust-{args.n_clusters}_brain-{args.brain_plots}"
        f"{label_detail}_cfg-{fingerprint}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "fMRI-analogous peak/cluster/COM "
            "coordinate analysis for model units."
        )
    )

    parser.add_argument(
        "model",
        help=(
            "Model path, e.g. MazeEnv.../model0 "
            "or models/MazeEnv.../model0.p"
        ),
    )

    parser.add_argument(
        "--score_source",
        default="subspace",
        choices=SCORE_SOURCES,
        help=(
            "Source of delay-specific unit maps. "
            "'subspace' retains the existing planning-subspace analysis; "
            "'raw_activity' uses RMS variation of "
            "location-conditioned planning activity; "
            "'raw_activity_by_planning_delay' assigns uniquely selective "
            "units to a planning-delay preference and is a separate "
            "internal-clock analysis. "
            "Default: subspace."
        ),
    )

    parser.add_argument(
        "--subspace_array",
        default="Csubs",
        choices=["Csubs", "Csubs_raw", "auto"],
        help=(
            "Subspace tensor to analyse. Csubs is normalized across units "
            "for every location decoder and preserves prior behavior; "
            "Csubs_raw retains fitted coefficient magnitudes. Default: "
            "Csubs."
        ),
    )

    parser.add_argument(
        "--repo_root",
        default=".",
        help=(
            "Repository root. "
            "Default: current directory."
        ),
    )

    parser.add_argument(
        "--axis",
        default="z",
        choices=["x", "y", "z"],
        help=(
            "Coordinate axis to analyse. "
            "Default: z."
        ),
    )

    parser.add_argument(
        "--cluster_threshold",
        default="90",
        help=(
            "Cluster threshold, matching fmri analysis options: "
            "0, z, or a percentile strictly between 20 and 100. "
            "Default: 90."
        ),
    )

    parser.add_argument(
        "--n_clusters",
        type=int,
        default=3,
        help=(
            "Number of strongest clusters to save. "
            "Default: 3."
        ),
    )

    parser.add_argument(
        "--delay_axis",
        default="0",
        choices=["auto", "0", "1"],
        help=(
            "Which non-unit Csubs axis is delay after moving "
            "unit axis to last. Pysta stores delay,location,unit, so the "
            "validated default is 0."
        ),
    )

    parser.add_argument(
        "--trend_x",
        default="condition_index",
        choices=["condition_index", "lag_value"],
        help=(
            "X variable for the descriptive trend. condition_index exactly "
            "matches get_subj_gradients.py; lag_value uses the numeric task "
            "lag and is useful for non-contiguous selections. Default: "
            "condition_index."
        ),
    )

    parser.add_argument(
        "--delay_labels",
        default=None,
        help=(
            "Optional comma-separated display labels for lags, "
            "e.g. 0,1,2,3,4,5."
        ),
    )

    parser.add_argument(
        "--planning_times",
        default=None,
        help=(
            "Comma-separated planning step_nums. "
            "For raw_activity, the default remains -2,-1. "
            "For raw_activity_by_planning_delay, the default is "
            "all available negative planning steps. "
            "Not accepted for subspace."
        ),
    )

    parser.add_argument(
        "--lag_times",
        default=None,
        help=(
            "Comma-separated non-negative step_nums to analyse "
            "for raw_activity, e.g. 0,1,2,3,4,5. "
            "Default: all available non-negative integer steps. "
            "Only accepted for raw_activity; not accepted for subspace or "
            "raw_activity_by_planning_delay."
        ),
    )

    parser.add_argument(
        "--activity_standardization",
        default="none",
        choices=["none", "zscore"],
        help=(
            "Optional per-unit z-scoring before an "
            "activity-based analysis. "
            "For planning-clock preference it is fitted only on the selected "
            "planning observations. Default: none."
        ),
    )

    parser.add_argument(
        "--preference_tie_tolerance",
        type=float,
        default=1e-10,
        help=(
            "Max-versus-second-max activity margin required to assign a "
            "unit in raw_activity_by_planning_delay. Default: 1e-10."
        ),
    )

    parser.add_argument(
        "--brain_plots",
        type=int,
        default=1,
        help=(
            "If 1, save surface QC plots of delay score maps "
            "and strongest clusters. Default: 1."
        ),
    )

    parser.add_argument(
        "--output_root",
        default=None,
        help=(
            "Optional base directory for output folders. The default is "
            "<analysis_dir>/fmri_analogous. Relative paths are resolved "
            "from --repo_root."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow replacing files in an existing non-empty output folder. "
            "By default the script refuses to overwrite prior results."
        ),
    )

    args = parser.parse_args()

    if args.n_clusters < 1:
        parser.error("--n_clusters must be at least 1.")

    if args.brain_plots not in (0, 1):
        parser.error("--brain_plots must be exactly 0 or 1.")

    if (
        not np.isfinite(args.preference_tie_tolerance)
        or args.preference_tie_tolerance < 0
    ):
        parser.error("--preference_tie_tolerance must be finite and >= 0.")

    if args.score_source != "subspace":
        if args.subspace_array != "Csubs" or args.delay_axis != "0":
            parser.error(
                "--subspace_array and --delay_axis apply only to "
                "--score_source subspace."
            )
    elif args.activity_standardization != "none":
        parser.error(
            "--activity_standardization applies only to activity-based "
            "score sources."
        )

    if (
        args.score_source != "raw_activity_by_planning_delay"
        and args.preference_tie_tolerance != 1e-10
    ):
        parser.error(
            "--preference_tie_tolerance applies only to "
            "--score_source raw_activity_by_planning_delay."
        )

    if args.brain_plots:
        try:
            from nilearn import plotting as _nilearn_plotting  # noqa: F401
        except ImportError:
            parser.error(
                "--brain_plots 1 requires nilearn. Install it in the active "
                "environment, or explicitly rerun with --brain_plots 0."
            )

    try:
        cluster_threshold = parse_cluster_threshold(
            args.cluster_threshold
        )
    except ValueError as exc:
        parser.error(str(exc))

    repo_root = Path(
        args.repo_root
    ).resolve()

    try:
        model_dir_rel, model_stem = normalise_model_reference(
            args.model,
            repo_root,
        )

        (
            n_units,
            embedding_name,
            embedding_seed,
        ) = parse_model_name(
            model_dir_rel
        )
    except ValueError as exc:
        parser.error(str(exc))

    analysis_dir = (
        repo_root
        / ANALYSIS_REL
        / model_dir_rel
    )

    embedding_dir = find_embedding_dir(
        repo_root,
        embedding_name,
        n_units,
        embedding_seed,
    )

    if not analysis_dir.is_dir():
        raise FileNotFoundError(
            "Could not find analysis directory. "
            "Run analyse_rnn.py first.\n"
            f"Expected: {analysis_dir}"
        )

    print(f"repo_root:      {repo_root}")
    print(f"model_dir_rel:  {model_dir_rel}")
    print(f"model_stem:     {model_stem}")
    print(f"analysis_dir:   {analysis_dir}")
    print(f"embedding_dir:  {embedding_dir}")
    print(f"n_units:        {n_units}")
    print(f"embedding_name: {embedding_name}")
    print(f"axis:           {args.axis}")
    print(f"score_source:   {args.score_source}")

    print(
        "coordinate note: fsLR/Conte69 surface xyz coordinates; "
        "not labelled as MNI"
    )

    print(
        "anchor note:     "
        "no anchor/geodesic-distance analysis is performed here"
    )

    surface_coords, faces = load_surface(
        repo_root
    )

    unit_vertices = find_unit_vertices(
        embedding_dir,
        n_units,
    )

    vertex_to_cluster = load_vertex_to_cluster(
        embedding_dir=embedding_dir,
        n_vertices=surface_coords.shape[0],
        n_units=n_units,
    )

    validate_surface_mapping(
        surface_coords=surface_coords,
        faces=faces,
        unit_vertices=unit_vertices,
        vertex_to_cluster=vertex_to_cluster,
        n_units=n_units,
    )

    bg_sulc: np.ndarray | None = None

    if args.brain_plots:
        bg_sulc = load_sulc(
            repo_root
        )

        if (
            bg_sulc is not None
            and bg_sulc.size != surface_coords.shape[0]
        ):
            raise ValueError(
                "Sulcal background length does not match the surface: "
                f"{bg_sulc.size} != {surface_coords.shape[0]}."
            )

    unit_coords = surface_coords[
        unit_vertices
    ]

    unit_labels = load_unit_labels(
        repo_root,
        unit_vertices,
    )

    adjacency = build_unit_adjacency(
        faces=faces,
        unit_vertices=unit_vertices,
        vertex_to_cluster=vertex_to_cluster,
    )

    n_edges = (
        sum(
            len(x)
            for x in adjacency
        )
        // 2
    )

    n_isolated = sum(
        len(x) == 0
        for x in adjacency
    )

    print(
        f"[INFO] Surface-unit graph: "
        f"{n_units} units, "
        f"{n_edges} edges, "
        f"{n_isolated} isolated units"
    )

    graph_components = connected_components(
        np.ones(n_units, dtype=bool),
        adjacency,
    )
    graph_component_sizes = sorted(
        (len(component) for component in graph_components),
        reverse=True,
    )
    print(
        "[INFO] Full ROI graph component sizes: "
        f"{graph_component_sizes}"
    )

    score_metadata: dict[str, Any] = {}
    csubs_key: str | None = None

    if args.score_source == "subspace":
        if args.planning_times is not None or args.lag_times is not None:
            parser.error(
                "--planning_times/--lag_times are not used by the subspace "
                "source; remove them to avoid a misleading configuration."
            )

        ps_path = find_planning_subspaces_pickle(
            analysis_dir,
            model_stem,
        )

        print(
            "[INFO] Loaded planning subspaces from: "
            f"{ps_path}"
        )

        csubs, csubs_key = load_csubs(
            ps_path,
            n_units,
            requested_key=args.subspace_array,
        )

        print(
            f"[INFO] Using subspace array: "
            f"{csubs_key}"
        )

        delay_scores = csubs_to_delay_scores(
            csubs,
            n_units=n_units,
            delay_axis=args.delay_axis,
        )

        actual_lag_times = list(
            range(delay_scores.shape[0])
        )

        score_metadata = {
            "planning_subspaces_path": str(ps_path),
            "subspace_array_key": csubs_key,
            "delay_axis": args.delay_axis,
            "normalization_note": (
                "Csubs vectors are L2-normalized across units per location "
                "decoder; Csubs_raw retains fitted magnitudes."
            ),
            "definition": (
                "norm of planning-subspace coefficients "
                "over locations"
            ),
        }

    elif args.score_source == "raw_activity":
        try:
            planning_times = parse_int_list(
                args.planning_times,
                "--planning_times",
            )
        except ValueError as exc:
            parser.error(str(exc))

        if planning_times is None:
            planning_times = [-2, -1]

        if any(value >= 0 for value in planning_times):
            parser.error(
                "--planning_times for raw_activity must contain only "
                "negative planning step numbers."
            )

        if planning_times != sorted(planning_times):
            parser.error(
                "--planning_times for raw_activity must be in increasing "
                "task-time order, for example -2,-1."
            )

        try:
            requested_lag_times = parse_int_list(
                args.lag_times,
                "--lag_times",
            )
        except ValueError as exc:
            parser.error(str(exc))

        if requested_lag_times is not None:
            if any(value < 0 for value in requested_lag_times):
                parser.error("--lag_times must be non-negative.")

            if requested_lag_times != sorted(requested_lag_times):
                parser.error("--lag_times must be in increasing order.")

        trial_data_path = find_trial_data_pickle(
            analysis_dir,
            model_stem,
        )

        print(
            f"[INFO] Loaded trial data from: "
            f"{trial_data_path}"
        )

        (
            delay_scores,
            actual_lag_times,
            score_metadata,
        ) = make_activity_delay_scores(
            trial_data_path=trial_data_path,
            n_units=n_units,
            planning_times=planning_times,
            lag_times=requested_lag_times,
            activity_standardization=(
                args.activity_standardization
            ),
        )

    else:
        if args.lag_times is not None:
            parser.error(
                "--lag_times is not used by raw_activity_by_planning_delay."
            )

        try:
            planning_times = parse_int_list(
                args.planning_times,
                "--planning_times",
            )
        except ValueError as exc:
            parser.error(str(exc))

        if planning_times is not None:
            if any(value >= 0 for value in planning_times):
                parser.error(
                    "--planning_times for "
                    "raw_activity_by_planning_delay must contain only "
                    "negative planning step numbers."
                )

            if len(planning_times) < 2:
                parser.error(
                    "--planning_times for "
                    "raw_activity_by_planning_delay must contain at least "
                    "two distinct planning steps."
                )

        trial_data_path = find_trial_data_pickle(
            analysis_dir,
            model_stem,
        )

        print(
            f"[INFO] Loaded trial data from: "
            f"{trial_data_path}"
        )

        trial_data = load_trial_data(
            trial_data_path,
            n_units=n_units,
        )

        (
            delay_scores,
            actual_lag_times,
            score_metadata,
        ) = raw_activity_by_planning_delay_to_scores(
            trial_data=trial_data,
            planning_times=planning_times,
            activity_standardization=(
                args.activity_standardization
            ),
            preference_tie_tolerance=(
                args.preference_tie_tolerance
            ),
        )

        score_metadata[
            "trial_data_path"
        ] = str(trial_data_path)

    delay_scores = validate_delay_scores(
        delay_scores,
        n_units=n_units,
    )

    print(
        f"[INFO] Final delay_scores shape: "
        f"{delay_scores.shape}"
    )

    if args.delay_labels is None:
        delay_labels = [
            str(value)
            for value in actual_lag_times
        ]
    else:
        try:
            delay_labels = parse_delay_labels(
                args.delay_labels,
                n_delays=delay_scores.shape[0],
            )
        except ValueError as exc:
            parser.error(str(exc))

    rows = extract_all_points(
        delay_scores=delay_scores,
        delay_labels=delay_labels,
        delay_values=actual_lag_times,
        unit_coords=unit_coords,
        unit_vertices=unit_vertices,
        unit_labels=unit_labels,
        adjacency=adjacency,
        cluster_threshold=cluster_threshold,
        n_clusters=args.n_clusters,
    )

    for row in rows:
        row["score_source"] = args.score_source
        row["model_stem"] = model_stem

    frag_rows = fragmentation_rows(
        delay_scores=delay_scores,
        delay_labels=delay_labels,
        delay_values=actual_lag_times,
        adjacency=adjacency,
        cluster_threshold=cluster_threshold,
        score_source=args.score_source,
    )

    print_fragmentation_summary(
        frag_rows
    )

    print(
        "[WARN] This run contains one trained model. Slopes and correlations "
        "below are descriptive; the fMRI reference's inferential unit is the "
        "subject, whose model analogue would be independently trained seeds."
    )

    output_name = build_output_name(
        args=args,
        model_dir_rel=model_dir_rel,
        model_stem=model_stem,
        csubs_key=csubs_key,
        actual_lag_times=actual_lag_times,
        score_metadata=score_metadata,
    )

    if args.output_root is None:
        output_root = analysis_dir / "fmri_analogous"
    else:
        output_root = Path(args.output_root).expanduser()

        if not output_root.is_absolute():
            output_root = repo_root / output_root

        output_root = output_root.resolve()

    out_dir = output_root / output_name

    if out_dir.is_dir() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            "Output directory already exists and is non-empty:\n"
            f"  {out_dir}\n"
            "Use --overwrite only after confirming that this exact "
            "configuration should replace its files."
        )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        out_dir
        / "fmri_analogous_peak_cluster_com_coordinates.csv"
    )

    write_csv(
        rows,
        csv_path,
    )

    write_csv(
        frag_rows,
        out_dir / "spatial_fragmentation.csv",
    )

    np.savez_compressed(
        out_dir / "lag_score_maps.npz",
        delay_scores=delay_scores,
        delay_labels=np.asarray(
            delay_labels,
            dtype=str,
        ),
        actual_lag_times=np.asarray(
            actual_lag_times,
            dtype=int,
        ),
        score_source=np.asarray(
            args.score_source
        ),
    )

    with open(
        out_dir / "score_source_metadata.pickle",
        "wb",
    ) as f:
        pickle.dump(
            {
                "score_source": args.score_source,
                "model_stem": model_stem,
                "score_source_label": (
                    SCORE_SOURCE_LABELS[
                        args.score_source
                    ]
                ),
                "score_metadata": score_metadata,
                "delay_labels": delay_labels,
                "actual_lag_times": actual_lag_times,
                "cluster_threshold": cluster_threshold,
                "trend_x": args.trend_x,
            },
            f,
        )

    title_prefix = (
        f"{embedding_name}\n"
        f"{SCORE_SOURCE_LABELS[args.score_source]}"
    )

    if csubs_key is not None:
        title_prefix += f" ({csubs_key})"

    if args.score_source == "raw_activity_by_planning_delay":
        quantity_label = "planning-clock delay"
    else:
        quantity_label = "future execution lag"

    if args.trend_x == "condition_index":
        x_label = f"Ordered {quantity_label} condition (index)"
    else:
        x_label = quantity_label.capitalize()

    trend_rows = make_primary_plots(
        rows,
        out_dir,
        axis=args.axis,
        title_prefix=title_prefix,
        trend_x=args.trend_x,
        x_label=x_label,
    )

    write_csv(
        trend_rows,
        out_dir / "descriptive_trend_metrics.csv",
    )

    run_config = {
        "model_argument": args.model,
        "model_dir_rel": str(model_dir_rel),
        "model_stem": model_stem,
        "analysis_dir": str(analysis_dir),
        "embedding_dir": str(embedding_dir),
        "output_dir": str(out_dir),
        "score_source": args.score_source,
        "subspace_array": args.subspace_array,
        "delay_axis": args.delay_axis,
        "planning_times": score_metadata.get("planning_times"),
        "lag_times": actual_lag_times,
        "delay_labels": delay_labels,
        "activity_standardization": args.activity_standardization,
        "preference_tie_tolerance": args.preference_tie_tolerance,
        "axis": args.axis,
        "trend_x": args.trend_x,
        "cluster_threshold_requested": args.cluster_threshold,
        "cluster_threshold_parsed": cluster_threshold,
        "n_clusters": args.n_clusters,
        "brain_plots": bool(args.brain_plots),
        "single_model_inference_note": (
            "Trend slope/correlations are descriptive. The fMRI reference's "
            "group inference requires independently trained model seeds as "
            "the replicate unit."
        ),
    }

    with open(out_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2, sort_keys=True)

    if bool(args.brain_plots):
        plot_delay_score_surface_qc(
            delay_scores=delay_scores,
            delay_labels=delay_labels,
            surface_coords=surface_coords,
            faces=faces,
            bg_sulc=bg_sulc,
            unit_vertices=unit_vertices,
            vertex_to_cluster=vertex_to_cluster,
            out_dir=out_dir,
            score_source_label=(
                SCORE_SOURCE_LABELS[
                    args.score_source
                ]
            ),
        )

        plot_strongest_cluster_surface_qc(
            delay_scores=delay_scores,
            delay_labels=delay_labels,
            surface_coords=surface_coords,
            faces=faces,
            bg_sulc=bg_sulc,
            unit_vertices=unit_vertices,
            vertex_to_cluster=vertex_to_cluster,
            adjacency=adjacency,
            cluster_threshold=cluster_threshold,
            out_dir=out_dir,
            score_source_label=(
                SCORE_SOURCE_LABELS[
                    args.score_source
                ]
            ),
        )

    print("\nDONE")
    print(
        f"Outputs saved in: {out_dir}"
    )
    print(
        f"Main CSV: {csv_path}"
    )
    print(
        "Fragmentation CSV: "
        f"{out_dir / 'spatial_fragmentation.csv'}"
    )


if __name__ == "__main__":
    main()
