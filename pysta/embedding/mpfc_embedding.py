import json
from pathlib import Path

import numpy as np

from pysta.embedding.sampling import (
    sample_vertices_custom_roi,
    sample_vertices_roi,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SUBSAMPLED_DIR = REPO_ROOT / "data" / "embedding" / "subsampled"
CUSTOM_ROI_DIR = REPO_ROOT / "data" / "embedding" / "custom_roi_vertices"


CORE_MPFC = [
    "33pr", "p24pr", "a24pr", "p24", "a24",
    "p32pr", "a32pr", "d32", "p32", "s32",
    "8BM", "9m", "10v", "10r", "25",
]

BROAD_MPFC = CORE_MPFC + [
    "10d", "a10p", "p10p", "10pp", "OFC", "pOFC",
    "SCEF", "6ma", "6mp", "24dv", "24dd",
]

VERY_FOCUSED_MPFC = [
    "a32pr", "d32", "p32", "p32pr",
    "8BM", "9m",
]

DEFAULT_CUSTOM_ROI_NAME = "mpfc_union_gradient_nearest_a24_25_anchor25"
DEFAULT_CUSTOM_VERTEX_FILE = (
    CUSTOM_ROI_DIR / "mpfc_union_gradient_nearest_a24_25_anchor25_lh_vertex_indices.npy"
)


def save_embedding(
    embedding_dir,
    distance_matrix,
    sampled_indices,
    area_labels,
    vertex_to_cluster,
    roi_vertex_indices,
    roi_name,
    roi_area_names,
    seed,
    source_type="parcel",
    custom_vertex_path=None,
    geodesic_mode=None,
):
    embedding_dir.mkdir(parents=True, exist_ok=True)

    np.save(embedding_dir / "distance_matrix.npy", distance_matrix)
    np.save(embedding_dir / "sampled_indices.npy", sampled_indices)
    np.save(embedding_dir / "vertex_to_cluster.npy", vertex_to_cluster)
    np.save(embedding_dir / "roi_vertex_indices.npy", roi_vertex_indices)

    with open(embedding_dir / "area_labels.txt", "w") as f:
        for x in area_labels:
            f.write(f"{x}\n")

    meta = {
        "roi_name": roi_name,
        "roi_area_names": list(roi_area_names) if roi_area_names is not None else None,
        "n_units": int(len(sampled_indices)),
        "seed": int(seed),
        "source_type": str(source_type),
        "custom_vertex_path": None if custom_vertex_path is None else str(custom_vertex_path),
        "geodesic_mode": None if geodesic_mode is None else str(geodesic_mode),
    }

    with open(embedding_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_embedding(embedding_dir):
    embedding_dir = Path(embedding_dir)

    distance_matrix = np.load(embedding_dir / "distance_matrix.npy")
    sampled_indices = np.load(embedding_dir / "sampled_indices.npy")
    vertex_to_cluster = np.load(embedding_dir / "vertex_to_cluster.npy")
    roi_vertex_indices = np.load(embedding_dir / "roi_vertex_indices.npy")

    with open(embedding_dir / "area_labels.txt", "r") as f:
        area_labels = [line.strip() for line in f]

    with open(embedding_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    return {
        "distance_matrix": distance_matrix,
        "sampled_indices": sampled_indices,
        "vertex_to_cluster": vertex_to_cluster,
        "roi_vertex_indices": roi_vertex_indices,
        "area_labels": area_labels,
        "metadata": metadata,
    }


def create_mpfc_embedding(
    roi_name="mpfc_core",
    roi_area_names=None,
    n_units=800,
    seed=0,
    species="human",
    overwrite=False,
    geodesic_mode="roi",
):
    if roi_area_names is None:
        roi_area_names = CORE_MPFC

    embedding_dir = SUBSAMPLED_DIR / species / roi_name / f"units={n_units}_seed={seed}"

    if embedding_dir.exists() and not overwrite:
        print(f"Using existing embedding from {embedding_dir}")
        return load_embedding(embedding_dir)

    (
        distance_matrix,
        cortical_surface,
        sampled_indices,
        vertex_to_cluster,
        roi_vertex_indices,
    ) = sample_vertices_roi(
        n_units=n_units,
        seed=seed,
        species=species,
        roi_area_names=roi_area_names,
        geodesic_mode=geodesic_mode,
    )

    area_labels = cortical_surface.parcellation.vertex_names[sampled_indices].tolist()

    save_embedding(
        embedding_dir=embedding_dir,
        distance_matrix=distance_matrix,
        sampled_indices=sampled_indices,
        area_labels=area_labels,
        vertex_to_cluster=vertex_to_cluster,
        roi_vertex_indices=roi_vertex_indices,
        roi_name=roi_name,
        roi_area_names=roi_area_names,
        seed=seed,
        source_type="parcel",
        custom_vertex_path=None,
        geodesic_mode=geodesic_mode,
    )

    return {
        "distance_matrix": distance_matrix,
        "sampled_indices": sampled_indices,
        "vertex_to_cluster": vertex_to_cluster,
        "roi_vertex_indices": roi_vertex_indices,
        "area_labels": area_labels,
        "metadata": {
            "roi_name": roi_name,
            "roi_area_names": list(roi_area_names),
            "n_units": int(n_units),
            "seed": int(seed),
            "source_type": "parcel",
            "custom_vertex_path": None,
            "geodesic_mode": geodesic_mode,
        },
    }


def create_custom_vertex_embedding(
    roi_name=DEFAULT_CUSTOM_ROI_NAME,
    custom_vertex_path=DEFAULT_CUSTOM_VERTEX_FILE,
    n_units=None,
    seed=42,
    species="human",
    overwrite=False,
    geodesic_mode="full",
):
    """
    Create an embedding from a saved vertex-level ROI.

    If n_units is None, use all ROI vertices:
    one surface vertex = one recurrent unit.
    """
    custom_vertex_path = Path(custom_vertex_path)

    if not custom_vertex_path.exists():
        raise FileNotFoundError(f"Could not find custom vertex file: {custom_vertex_path}")

    roi_vertex_indices = np.load(custom_vertex_path).astype(np.int32).reshape(-1)
    roi_vertex_indices = np.unique(roi_vertex_indices).astype(np.int32)

    if n_units is None:
        n_units = int(len(roi_vertex_indices))

    if n_units > len(roi_vertex_indices):
        raise ValueError(
            f"Requested n_units={n_units}, but custom ROI has only "
            f"{len(roi_vertex_indices)} vertices."
        )

    embedding_dir = SUBSAMPLED_DIR / species / roi_name / f"units={n_units}_seed={seed}"

    if embedding_dir.exists() and not overwrite:
        print(f"Using existing embedding from {embedding_dir}")
        return load_embedding(embedding_dir)

    (
        distance_matrix,
        cortical_surface,
        sampled_indices,
        vertex_to_cluster,
        roi_vertex_indices,
    ) = sample_vertices_custom_roi(
        n_units=n_units,
        seed=seed,
        species=species,
        custom_vertex_path=custom_vertex_path,
        geodesic_mode=geodesic_mode,
    )

    area_labels = cortical_surface.parcellation.vertex_names[sampled_indices].tolist()

    save_embedding(
        embedding_dir=embedding_dir,
        distance_matrix=distance_matrix,
        sampled_indices=sampled_indices,
        area_labels=area_labels,
        vertex_to_cluster=vertex_to_cluster,
        roi_vertex_indices=roi_vertex_indices,
        roi_name=roi_name,
        roi_area_names=None,
        seed=seed,
        source_type="custom_vertex_file",
        custom_vertex_path=custom_vertex_path,
        geodesic_mode=geodesic_mode,
    )

    return {
        "distance_matrix": distance_matrix,
        "sampled_indices": sampled_indices,
        "vertex_to_cluster": vertex_to_cluster,
        "roi_vertex_indices": roi_vertex_indices,
        "area_labels": area_labels,
        "metadata": {
            "roi_name": roi_name,
            "roi_area_names": None,
            "n_units": int(n_units),
            "seed": int(seed),
            "source_type": "custom_vertex_file",
            "custom_vertex_path": str(custom_vertex_path),
            "geodesic_mode": geodesic_mode,
        },
    }


if __name__ == "__main__":
    out = create_custom_vertex_embedding(
        roi_name=DEFAULT_CUSTOM_ROI_NAME,
        custom_vertex_path=DEFAULT_CUSTOM_VERTEX_FILE,
        n_units=None,
        seed=42,
        species="human",
        overwrite=True,
        geodesic_mode="full",
    )

    print("\nDone.")
    print("roi_name:", out["metadata"]["roi_name"])
    print("n_units:", out["metadata"]["n_units"])
    print("source_type:", out["metadata"]["source_type"])
    print("custom_vertex_path:", out["metadata"]["custom_vertex_path"])
    print("geodesic_mode:", out["metadata"]["geodesic_mode"])
    print("distance_matrix shape:", out["distance_matrix"].shape)
    print("sampled_indices shape:", out["sampled_indices"].shape)
    print("roi_vertex_indices shape:", out["roi_vertex_indices"].shape)
    print("unique sampled parcels:", sorted(set(out["area_labels"])))