import argparse
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
ROI_SPEC_DIR = CUSTOM_ROI_DIR


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

DEFAULT_CUSTOM_ROI_NAME = "mpfc_projected_mask_linear0p1" # change the roi name to desired roi
DEFAULT_CUSTOM_VERTEX_FILE = (
    CUSTOM_ROI_DIR
    / DEFAULT_CUSTOM_ROI_NAME
    / f"{DEFAULT_CUSTOM_ROI_NAME}_lh_vertex_indices.npy"
)


def resolve_repo_path(path):
    path = Path(path)

    if path.is_absolute():
        return path

    return REPO_ROOT / path


def load_roi_spec(roi_spec):
    roi_spec = Path(roi_spec)

    if roi_spec.suffix == ".json":
        spec_path = roi_spec
        if not spec_path.is_absolute():
            spec_path = REPO_ROOT / spec_path
    else:
        spec_path = CUSTOM_ROI_DIR / str(roi_spec) / "roi_specs.json"

    if not spec_path.exists():
        raise FileNotFoundError(
            "Could not find ROI spec.\n"
            f"Looked for: {spec_path}\n"
            "Expected structure:\n"
            f"  {CUSTOM_ROI_DIR}/<roi_name>/roi_specs.json"
        )

    with open(spec_path, "r") as f:
        spec = json.load(f)

    spec["_spec_path"] = str(spec_path)

    return spec


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

    return load_embedding(embedding_dir)


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

    return load_embedding(embedding_dir)


def add_anchor_files_to_embedding(
    *,
    embedding_dir,
    roi_spec,
):
    embedding_dir = Path(embedding_dir)
    embedding = load_embedding(embedding_dir)

    sampled_indices = np.asarray(embedding["sampled_indices"], dtype=np.int64).reshape(-1)

    anchor_vertex_path = resolve_repo_path(roi_spec["anchor_vertex_file"])
    if not anchor_vertex_path.exists():
        raise FileNotFoundError(f"Could not find anchor vertex file: {anchor_vertex_path}")

    anchor_vertices = np.load(anchor_vertex_path).astype(np.int64).reshape(-1)
    anchor_vertices = np.unique(anchor_vertices).astype(np.int64)

    sampled_vertex_to_unit = {int(v): i for i, v in enumerate(sampled_indices)}

    anchor_unit_indices = []
    matched_anchor_vertices = []

    for vertex in anchor_vertices:
        vertex = int(vertex)
        if vertex in sampled_vertex_to_unit:
            anchor_unit_indices.append(sampled_vertex_to_unit[vertex])
            matched_anchor_vertices.append(vertex)

    anchor_unit_indices = np.asarray(anchor_unit_indices, dtype=np.int32)
    matched_anchor_vertices = np.asarray(matched_anchor_vertices, dtype=np.int32)

    if len(anchor_unit_indices) == 0:
        raise ValueError(
            "The anchor vertices do not overlap with the sampled model vertices.\n"
            f"Anchor file: {anchor_vertex_path}\n"
            f"Number of anchor vertices: {len(anchor_vertices)}\n"
            f"Number of sampled units: {len(sampled_indices)}"
        )

    np.save(embedding_dir / "anchor_unit_indices.npy", anchor_unit_indices)
    np.save(embedding_dir / "anchor_vertex_indices.npy", matched_anchor_vertices)

    external_ref_path = resolve_repo_path(roi_spec["external_reference_vertex_file"])
    external_reference_vertices = None

    if external_ref_path.exists():
        external_reference_vertices = np.load(external_ref_path).astype(np.int32).reshape(-1)
        external_reference_vertices = np.unique(external_reference_vertices).astype(np.int32)
        np.save(
            embedding_dir / "external_reference_vertex_indices.npy",
            external_reference_vertices,
        )

    metadata_path = embedding_dir / "metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    metadata.update(
        {
            "roi_spec_path": roi_spec["_spec_path"],
            "roi_spec": {k: v for k, v in roi_spec.items() if not k.startswith("_")},
            "model_space_vertex_file": roi_spec["model_space_vertex_file"],
            "anchor_vertex_file": roi_spec["anchor_vertex_file"],
            "external_reference_vertex_file": roi_spec["external_reference_vertex_file"],
            "anchor_definition": "closest_model_space_vertices_to_external_reference",
            "anchor_unit_indices_file": "anchor_unit_indices.npy",
            "anchor_vertex_indices_file": "anchor_vertex_indices.npy",
            "external_reference_vertex_indices_file": (
                "external_reference_vertex_indices.npy"
                if external_reference_vertices is not None
                else None
            ),
            "n_anchor_vertices_in_spec": int(len(anchor_vertices)),
            "n_anchor_units_matched": int(len(anchor_unit_indices)),
            "n_external_reference_vertices": (
                None
                if external_reference_vertices is None
                else int(len(external_reference_vertices))
            ),
        }
    )

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("Added anchor metadata to embedding:")
    print(f"  embedding_dir: {embedding_dir}")
    print(f"  anchor_vertex_file: {anchor_vertex_path}")
    print(f"  anchor vertices in spec: {len(anchor_vertices)}")
    print(f"  matched anchor RNN units: {len(anchor_unit_indices)}")
    print(f"  saved: {embedding_dir / 'anchor_unit_indices.npy'}")
    print(f"  saved: {embedding_dir / 'anchor_vertex_indices.npy'}")

    if external_reference_vertices is not None:
        print(f"  external reference vertices: {len(external_reference_vertices)}")
        print(f"  saved: {embedding_dir / 'external_reference_vertex_indices.npy'}")

    return load_embedding(embedding_dir)


def create_custom_vertex_embedding_from_spec(
    roi_spec,
    n_units=None,
    seed=42,
    species="human",
    overwrite=False,
    geodesic_mode="full",
):
    spec = load_roi_spec(roi_spec)

    roi_name = spec["roi_name"]
    custom_vertex_path = resolve_repo_path(spec["model_space_vertex_file"])

    if not custom_vertex_path.exists():
        raise FileNotFoundError(f"Could not find model-space vertex file: {custom_vertex_path}")

    if n_units is None:
        roi_vertex_indices = np.load(custom_vertex_path).astype(np.int32).reshape(-1)
        roi_vertex_indices = np.unique(roi_vertex_indices).astype(np.int32)
        n_units = int(len(roi_vertex_indices))

    embedding = create_custom_vertex_embedding(
        roi_name=roi_name,
        custom_vertex_path=custom_vertex_path,
        n_units=n_units,
        seed=seed,
        species=species,
        overwrite=overwrite,
        geodesic_mode=geodesic_mode,
    )

    embedding_dir = SUBSAMPLED_DIR / species / roi_name / f"units={n_units}_seed={seed}"

    embedding = add_anchor_files_to_embedding(
        embedding_dir=embedding_dir,
        roi_spec=spec,
    )

    return embedding


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--roi_spec",
        default=DEFAULT_CUSTOM_ROI_NAME,
        type=str,
        help=(
            "ROI spec name or JSON path, e.g. mpfc_projected_mask_linear0p1. "
            "If provided, model-space and anchor files are read from the spec."
        ),
    )
    parser.add_argument("--roi_name", default=DEFAULT_CUSTOM_ROI_NAME, type=str)
    parser.add_argument("--custom_vertex_path", default=str(DEFAULT_CUSTOM_VERTEX_FILE), type=str)
    parser.add_argument("--n_units", default=None, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--species", default="human", type=str)
    parser.add_argument("--overwrite", default=1, type=int)
    parser.add_argument("--geodesic_mode", default="full", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.roi_spec is not None:
        out = create_custom_vertex_embedding_from_spec(
            roi_spec=args.roi_spec,
            n_units=args.n_units,
            seed=args.seed,
            species=args.species,
            overwrite=bool(args.overwrite),
            geodesic_mode=args.geodesic_mode,
        )
    else:
        out = create_custom_vertex_embedding(
            roi_name=args.roi_name,
            custom_vertex_path=Path(args.custom_vertex_path),
            n_units=args.n_units,
            seed=args.seed,
            species=args.species,
            overwrite=bool(args.overwrite),
            geodesic_mode=args.geodesic_mode,
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

    if "anchor_unit_indices_file" in out["metadata"]:
        print("anchor_unit_indices_file:", out["metadata"]["anchor_unit_indices_file"])
        print("n_anchor_units_matched:", out["metadata"]["n_anchor_units_matched"])