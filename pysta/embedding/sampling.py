from pathlib import Path

import gdist
import nibabel as nib
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from pysta.embedding.ultimate_surface import Surface


def simplify_name(n):
    if isinstance(n, bytes):
        n = n.decode("utf-8")
    n = str(n)
    if n.startswith("L_") and n.endswith("_ROI"):
        return n[2:-4]
    return n


def load_annot(mesh_folder, species):
    if species != "human":
        raise ValueError(f"Only human is supported for now, got {species}")

    annot_path = Path(mesh_folder) / "label" / "lh.HCPMMP1.annot"
    labels, rgba, raw_names = nib.freesurfer.read_annot(str(annot_path))
    names = [simplify_name(x) for x in raw_names]
    return labels, rgba, names


class Parcellation:
    def __init__(self, species, mesh_folder):
        print(f"Loading parcellation for {species} from {mesh_folder}")
        self.labels, self.rgba, self.names = load_annot(mesh_folder, species)

        valid_mask = (self.labels >= 0) & (self.labels < len(self.names))
        self.vertex_names = np.array(["<invalid>"] * len(self.labels), dtype=object)
        self.vertex_names[valid_mask] = np.array(self.names, dtype=object)[
            self.labels[valid_mask]
        ]

        self.valid_mask = valid_mask
        self.cortex_mask = valid_mask
        self.cortex_vertices_indices = np.where(self.cortex_mask)[0]


class CorticalSurface:
    def __init__(self, species="human"):
        self.species = species

        if species == "human":
            mesh_name = "fs_lr32"
        else:
            raise ValueError(f"Unknown species: {species}")

        self.inflated_surface = Surface(
            mesh=mesh_name, specie=species, surface_name="sphere", hemisphere="l"
        )
        self.midthickness_surface = Surface(
            mesh=mesh_name, specie=species, surface_name="midthickness", hemisphere="l"
        )
        self.parcellation = Parcellation(species, self.inflated_surface.mesh_folder)


def get_roi_vertex_indices(cortical_surface, roi_area_names):
    vertex_names = cortical_surface.parcellation.vertex_names
    roi_mask = np.isin(vertex_names, roi_area_names)
    roi_vertex_indices = np.where(roi_mask)[0].astype(np.int32)

    if len(roi_vertex_indices) == 0:
        raise ValueError("ROI is empty. Check parcel names.")

    return roi_vertex_indices


def load_custom_roi_vertex_indices(custom_vertex_path):
    custom_vertex_path = Path(custom_vertex_path)

    if not custom_vertex_path.exists():
        raise FileNotFoundError(f"Could not find custom ROI vertices: {custom_vertex_path}")

    roi_vertex_indices = np.load(custom_vertex_path).astype(np.int32).reshape(-1)

    if len(roi_vertex_indices) == 0:
        raise ValueError(f"Custom ROI vertex file is empty: {custom_vertex_path}")

    roi_vertex_indices = np.unique(roi_vertex_indices).astype(np.int32)

    return roi_vertex_indices


def kmeans_clustering(vertices, n_units, seed):
    kmeans = KMeans(n_clusters=n_units, random_state=seed, n_init=10)
    kmeans.fit(vertices)
    return kmeans.cluster_centers_


def map_a_to_closest_b(coords_a, coords_b):
    return np.argmin(cdist(coords_a, coords_b), axis=1)


def unique_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        x = int(x)
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def fill_missing_unique_vertices(roi_coords, selected_local_indices, target_n):
    """
    If kmeans-centre -> nearest-vertex mapping gives duplicates,
    fill remaining slots by greedily adding vertices farthest from the current set.
    """
    selected = unique_keep_order(selected_local_indices)

    if len(selected) == 0:
        selected = [0]

    selected_set = set(selected)
    remaining = [i for i in range(len(roi_coords)) if i not in selected_set]

    while len(selected) < target_n:
        if len(remaining) == 0:
            raise RuntimeError("Ran out of remaining ROI vertices while filling sampled set.")

        selected_coords = roi_coords[selected]
        remaining_coords = roi_coords[remaining]

        dists = cdist(remaining_coords, selected_coords)
        min_dists = dists.min(axis=1)

        best_idx_in_remaining = int(np.argmax(min_dists))
        best_vertex = remaining[best_idx_in_remaining]

        selected.append(best_vertex)
        remaining.pop(best_idx_in_remaining)

    return np.array(selected[:target_n], dtype=np.int32)


def sample_vertices_kmeans_from_roi(cortical_surface, roi_vertex_indices, n_units, seed):
    """
    Sample n_units from an ROI.

    If n_units == number of ROI vertices, no kmeans is done:
    each ROI vertex becomes one RNN unit.
    """
    roi_vertex_indices = np.asarray(roi_vertex_indices, dtype=np.int32).reshape(-1)
    roi_vertex_indices = np.unique(roi_vertex_indices).astype(np.int32)

    if len(roi_vertex_indices) < n_units:
        raise ValueError(
            f"ROI has only {len(roi_vertex_indices)} vertices, "
            f"cannot sample {n_units} unique units."
        )

    sphere_coords = cortical_surface.inflated_surface.positions[roi_vertex_indices]

    if len(roi_vertex_indices) == n_units:
        sampled_roi_local_indices = np.arange(len(roi_vertex_indices), dtype=np.int32)
        sampled_vertex_indices = roi_vertex_indices.copy()

        full_vertex_to_cluster = np.full(
            cortical_surface.inflated_surface.vertex_number, -1, dtype=np.int32
        )
        full_vertex_to_cluster[roi_vertex_indices] = sampled_roi_local_indices

        print(
            f"Using all ROI vertices as RNN units: "
            f"{len(sampled_vertex_indices)} vertices / units."
        )

        return sampled_vertex_indices, full_vertex_to_cluster, sampled_roi_local_indices

    sampled_centers = kmeans_clustering(sphere_coords, n_units=n_units, seed=seed)
    nearest_local_indices = map_a_to_closest_b(sampled_centers, sphere_coords)
    unique_local_indices = unique_keep_order(nearest_local_indices.tolist())

    if len(unique_local_indices) < n_units:
        print(
            f"KMeans mapped to only {len(unique_local_indices)} unique ROI vertices. "
            f"Filling remaining {n_units - len(unique_local_indices)} greedily."
        )
        sampled_roi_local_indices = fill_missing_unique_vertices(
            sphere_coords, unique_local_indices, n_units
        )
    else:
        sampled_roi_local_indices = np.array(unique_local_indices[:n_units], dtype=np.int32)

    sampled_vertex_indices = roi_vertex_indices[sampled_roi_local_indices]

    sampled_roi_coords = sphere_coords[sampled_roi_local_indices]
    roi_vertex_to_cluster = map_a_to_closest_b(sphere_coords, sampled_roi_coords)

    full_vertex_to_cluster = np.full(
        cortical_surface.inflated_surface.vertex_number, -1, dtype=np.int32
    )
    full_vertex_to_cluster[roi_vertex_indices] = roi_vertex_to_cluster.astype(np.int32)

    return sampled_vertex_indices, full_vertex_to_cluster, sampled_roi_local_indices


def build_roi_surface_mesh(cortical_surface, roi_vertex_indices):
    """
    Build an ROI-only mesh on midthickness surface:
    - keep only ROI vertices
    - keep only triangles fully inside ROI
    """
    surface = cortical_surface.midthickness_surface

    vertex_mask = np.zeros(surface.vertex_number, dtype=bool)
    vertex_mask[roi_vertex_indices] = True

    roi_vertices = surface.positions[roi_vertex_indices].astype(np.float64)
    roi_triangles = surface.get_filtered_triangles(vertex_mask).astype(np.int32)

    old_to_new = np.full(surface.vertex_number, -1, dtype=np.int32)
    old_to_new[roi_vertex_indices] = np.arange(len(roi_vertex_indices), dtype=np.int32)

    return roi_vertices, roi_triangles, old_to_new


def compute_geodesic_distance_matrix_on_roi_mesh(
    roi_vertices,
    roi_triangles,
    sampled_roi_local_indices,
    progress_interval=100,
):
    sampled_roi_local_indices = np.asarray(sampled_roi_local_indices, dtype=np.int32)
    n = len(sampled_roi_local_indices)

    dist_mat = np.zeros((n, n), dtype=np.float32)
    target_indices = sampled_roi_local_indices.astype(np.int32)

    print(f"Computing ROI-only geodesic distance matrix for {n} sampled vertices...")

    for i, src_idx in enumerate(sampled_roi_local_indices):
        distances = gdist.compute_gdist(
            roi_vertices,
            roi_triangles,
            source_indices=np.array([src_idx], dtype=np.int32),
            target_indices=target_indices,
        )
        dist_mat[i, :] = distances.astype(np.float32)

        if (i + 1) % progress_interval == 0 or i == n - 1:
            print(f"  done {i + 1}/{n}")

    if not np.all(np.isfinite(dist_mat)):
        n_bad = int(np.sum(~np.isfinite(dist_mat)))
        raise ValueError(
            f"ROI-only geodesic distance matrix contains {n_bad} non-finite values. "
            "This usually means the ROI mesh is disconnected. "
            "Use full-surface distances for this ROI."
        )

    return dist_mat


def compute_geodesic_distance_matrix_on_full_surface(
    cortical_surface,
    sampled_vertex_indices,
    progress_interval=100,
):
    """
    Compute geodesics on the full LH midthickness mesh.

    This is safer for custom ROIs that have several islands, because distances
    can travel through cortex outside the ROI instead of becoming infinite.
    """
    surface = cortical_surface.midthickness_surface

    vertices = surface.positions.astype(np.float64)
    triangles = surface.triangles.astype(np.int32)

    sampled_vertex_indices = np.asarray(sampled_vertex_indices, dtype=np.int32)
    n = len(sampled_vertex_indices)

    dist_mat = np.zeros((n, n), dtype=np.float32)
    target_indices = sampled_vertex_indices.astype(np.int32)

    print(f"Computing full-surface geodesic distance matrix for {n} sampled vertices...")

    for i, src_idx in enumerate(sampled_vertex_indices):
        distances = gdist.compute_gdist(
            vertices,
            triangles,
            source_indices=np.array([src_idx], dtype=np.int32),
            target_indices=target_indices,
        )
        dist_mat[i, :] = distances.astype(np.float32)

        if (i + 1) % progress_interval == 0 or i == n - 1:
            print(f"  done {i + 1}/{n}")

    if not np.all(np.isfinite(dist_mat)):
        n_bad = int(np.sum(~np.isfinite(dist_mat)))
        raise ValueError(
            f"Full-surface geodesic distance matrix contains {n_bad} non-finite values."
        )

    return dist_mat


def sample_vertices_roi(
    n_units,
    seed,
    species,
    roi_area_names,
    geodesic_mode="roi",
):
    cortical_surface = CorticalSurface(species=species)

    roi_vertex_indices = get_roi_vertex_indices(cortical_surface, roi_area_names)

    sampled_vertex_indices, vertex_to_cluster, sampled_roi_local_indices = (
        sample_vertices_kmeans_from_roi(
            cortical_surface=cortical_surface,
            roi_vertex_indices=roi_vertex_indices,
            n_units=n_units,
            seed=seed,
        )
    )

    if geodesic_mode == "roi":
        roi_vertices, roi_triangles, old_to_new = build_roi_surface_mesh(
            cortical_surface, roi_vertex_indices
        )

        sampled_roi_local_indices = old_to_new[sampled_vertex_indices]
        if np.any(sampled_roi_local_indices < 0):
            raise RuntimeError("Some sampled vertices were not found in ROI-local indexing.")

        dist_mat = compute_geodesic_distance_matrix_on_roi_mesh(
            roi_vertices=roi_vertices,
            roi_triangles=roi_triangles,
            sampled_roi_local_indices=sampled_roi_local_indices,
        )

    elif geodesic_mode == "full":
        dist_mat = compute_geodesic_distance_matrix_on_full_surface(
            cortical_surface=cortical_surface,
            sampled_vertex_indices=sampled_vertex_indices,
        )

    else:
        raise ValueError("geodesic_mode must be 'roi' or 'full'.")

    return (
        dist_mat,
        cortical_surface,
        sampled_vertex_indices,
        vertex_to_cluster,
        roi_vertex_indices,
    )


def sample_vertices_custom_roi(
    n_units,
    seed,
    species,
    custom_vertex_path,
    geodesic_mode="full",
):
    """
    Sample from a saved vertex-level ROI.

    Default geodesic_mode='full' because custom masks from projected volumes
    can be disconnected; ROI-only geodesics would then contain infinities.
    """
    cortical_surface = CorticalSurface(species=species)

    roi_vertex_indices = load_custom_roi_vertex_indices(custom_vertex_path)

    if np.max(roi_vertex_indices) >= cortical_surface.midthickness_surface.vertex_number:
        raise ValueError(
            f"Custom ROI contains vertex index {int(np.max(roi_vertex_indices))}, "
            f"but surface has only {cortical_surface.midthickness_surface.vertex_number} vertices."
        )

    sampled_vertex_indices, vertex_to_cluster, sampled_roi_local_indices = (
        sample_vertices_kmeans_from_roi(
            cortical_surface=cortical_surface,
            roi_vertex_indices=roi_vertex_indices,
            n_units=n_units,
            seed=seed,
        )
    )

    if geodesic_mode == "full":
        dist_mat = compute_geodesic_distance_matrix_on_full_surface(
            cortical_surface=cortical_surface,
            sampled_vertex_indices=sampled_vertex_indices,
        )

    elif geodesic_mode == "roi":
        roi_vertices, roi_triangles, old_to_new = build_roi_surface_mesh(
            cortical_surface, roi_vertex_indices
        )

        sampled_roi_local_indices = old_to_new[sampled_vertex_indices]
        if np.any(sampled_roi_local_indices < 0):
            raise RuntimeError("Some sampled vertices were not found in ROI-local indexing.")

        dist_mat = compute_geodesic_distance_matrix_on_roi_mesh(
            roi_vertices=roi_vertices,
            roi_triangles=roi_triangles,
            sampled_roi_local_indices=sampled_roi_local_indices,
        )

    else:
        raise ValueError("geodesic_mode must be 'roi' or 'full'.")

    return (
        dist_mat,
        cortical_surface,
        sampled_vertex_indices,
        vertex_to_cluster,
        roi_vertex_indices,
    )