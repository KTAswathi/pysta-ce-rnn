from pathlib import Path
from collections import Counter, deque
import csv
import json

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from nilearn import plotting as nilearn_plotting
from nilearn import surface as nilearn_surface


REPO_ROOT = Path(__file__).resolve().parents[1]

ANNOT_PATH = (
    REPO_ROOT
    / "data"
    / "embedding"
    / "raw_surface_data"
    / "human"
    / "fs_lr32"
    / "label"
    / "lh.HCPMMP1.annot"
)

SURF_PATH = (
    REPO_ROOT
    / "data"
    / "embedding"
    / "raw_surface_data"
    / "human"
    / "fs_lr32"
    / "surf"
    / "fs_lr32.l.midthickness.surf.gii"
)

SULC_PATH = (
    REPO_ROOT
    / "data"
    / "embedding"
    / "raw_surface_data"
    / "human"
    / "fs_lr32"
    / "fs_lr32.l.sulc_data.func.gii"
)

GRADIENT_MASK_PATH = (
    REPO_ROOT
    / "data"
    / "embedding"
    / "gradient_mask_bin.nii"
)

# Projection options:
#   "nearest_most_frequent"
#   "linear"
GRADIENT_PROJECTION_MODE = "linear"
GRADIENT_LINEAR_THRESHOLD = 0.10

# External anatomical reference.
# Area 25 is NOT added to the model space here.
ANCHOR_REFERENCE_AREAS = ["25"]

# This defines the input/anchor zone inside projected mask.
# If ANCHOR_ZONE_N is None, the script uses ANCHOR_ZONE_FRACTION.
ANCHOR_ZONE_FRACTION = 1.0 / 6.0
ANCHOR_ZONE_N = None

# Distance mode:
#   "geodesic" = surface geodesic distance on full LH mesh
#   "euclidean" = straight-line xyz distance fallback
DISTANCE_MODE = "geodesic"


def threshold_tag():
    if GRADIENT_PROJECTION_MODE == "nearest_most_frequent":
        return "nearest"

    if GRADIENT_PROJECTION_MODE == "linear":
        return f"linear{str(GRADIENT_LINEAR_THRESHOLD).replace('.', 'p')}"

    return str(GRADIENT_PROJECTION_MODE)


def fraction_tag():
    if ANCHOR_ZONE_N is not None:
        return f"n{ANCHOR_ZONE_N}"
    return f"frac{str(round(float(ANCHOR_ZONE_FRACTION), 6)).replace('.', 'p')}"


def roi_name():
    return f"mpfc_projected_mask_{threshold_tag()}"


def repo_relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(REPO_ROOT.resolve()))


OUT_PREFIX = (
    f"mpfc_anchorcheck_mask_{threshold_tag()}"
    f"_closest_to_area25_{fraction_tag()}"
)


def simplify_name(n):
    if isinstance(n, bytes):
        n = n.decode("utf-8")
    n = str(n)

    if n.startswith("L_") and n.endswith("_ROI"):
        return n[2:-4]

    return n


def build_vertex_names():
    if not ANNOT_PATH.exists():
        raise FileNotFoundError(f"Could not find: {ANNOT_PATH}")

    labels, _ctab, raw_names = nib.freesurfer.read_annot(str(ANNOT_PATH))
    parsed_names = [simplify_name(x) for x in raw_names]
    unique_names = sorted(set(parsed_names))

    valid_vertex_mask = (labels >= 0) & (labels < len(parsed_names))

    vertex_names = np.array(["<invalid>"] * len(labels), dtype=object)
    vertex_names[valid_vertex_mask] = np.array(parsed_names, dtype=object)[
        labels[valid_vertex_mask]
    ]

    return unique_names, vertex_names


def load_surface_and_sulc():
    if not SURF_PATH.exists():
        raise FileNotFoundError(f"Could not find: {SURF_PATH}")

    surf_gii = nib.load(str(SURF_PATH))
    coords, faces = surf_gii.agg_data()

    coords = np.asarray(coords, dtype=float)
    faces = np.asarray(faces, dtype=np.int32)

    bg_sulc = None
    if SULC_PATH.exists():
        sulc_gii = nib.load(str(SULC_PATH))
        bg_sulc = np.asarray(sulc_gii.darrays[0].data, dtype=float).reshape(-1)

    return coords, faces, bg_sulc


def print_volume_info(img, data):
    print(f"Mask path: {GRADIENT_MASK_PATH}")
    print(f"Image shape: {img.shape}")
    print(f"Data shape after squeeze: {data.shape}")
    print(f"Data dtype: {data.dtype}")

    zooms = img.header.get_zooms()[:3]
    voxel_volume = float(np.prod(zooms))
    nonzero_voxels = int(np.sum(data > 0))

    print(f"Voxel sizes: {zooms}")
    print(f"Voxel volume: {voxel_volume:.3f} mm^3")
    print(f"Nonzero voxels: {nonzero_voxels}")
    print(f"Approx nonzero volume: {nonzero_voxels * voxel_volume:.3f} mm^3")
    print(f"Min value: {float(np.nanmin(data)):.6f}")
    print(f"Max value: {float(np.nanmax(data)):.6f}")
    print("Affine:")
    print(img.affine)


def project_volume_to_surface_all_modes(img, coords, faces):
    nearest_values = nilearn_surface.vol_to_surf(
        img,
        surf_mesh=(coords, faces),
        interpolation="nearest_most_frequent",
    )
    nearest_values = np.asarray(nearest_values, dtype=float)

    linear_values = nilearn_surface.vol_to_surf(
        img,
        surf_mesh=(coords, faces),
        interpolation="linear",
    )
    linear_values = np.asarray(linear_values, dtype=float)

    return nearest_values, linear_values


def load_gradient_mask_to_surface(coords, faces):
    if not GRADIENT_MASK_PATH.exists():
        raise FileNotFoundError(f"Could not find: {GRADIENT_MASK_PATH}")

    img = nib.load(str(GRADIENT_MASK_PATH))
    data = np.asarray(img.get_fdata()).squeeze()
    n_vertices = int(coords.shape[0])

    print_volume_info(img, data)

    flat = data.reshape(-1)

    print(f"LH surface vertices: {n_vertices}")
    print(f"Chosen projection mode: {GRADIENT_PROJECTION_MODE}")
    print(f"Chosen linear threshold: {GRADIENT_LINEAR_THRESHOLD}")

    if flat.size == n_vertices:
        print("Interpreting gradient mask as left-surface vertex data.")
        nearest_values = flat.astype(float)
        linear_values = flat.astype(float)

    elif flat.size == 2 * n_vertices:
        print("Interpreting gradient mask as two-hemisphere surface data.")
        print("Using first half as left hemisphere.")
        nearest_values = flat[:n_vertices].astype(float)
        linear_values = flat[:n_vertices].astype(float)

    elif data.ndim == 3:
        print("Interpreting gradient mask as volume.")
        print("Projecting volume to LH midthickness surface.")

        nearest_values, linear_values = project_volume_to_surface_all_modes(
            img, coords, faces
        )

    else:
        raise ValueError(
            "Could not interpret gradient mask. "
            f"Mask shape={data.shape}, flat size={flat.size}, surface vertices={n_vertices}."
        )

    print("Projection diagnostics:")
    print(f"  nearest_most_frequent > 0: {int(np.sum(nearest_values > 0))}")

    for thr in [0.0, 0.01, 0.05, 0.10, 0.25, 0.50]:
        print(
            f"  linear > {thr:0.2f}: "
            f"{int(np.sum(linear_values > thr))}"
        )

    if GRADIENT_PROJECTION_MODE == "nearest_most_frequent":
        gradient_values = nearest_values
        gradient_mask = nearest_values > 0
        projection_description = "nearest_most_frequent > 0"

    elif GRADIENT_PROJECTION_MODE == "linear":
        gradient_values = linear_values
        gradient_mask = linear_values > float(GRADIENT_LINEAR_THRESHOLD)
        projection_description = f"linear > {GRADIENT_LINEAR_THRESHOLD}"

    else:
        raise ValueError(
            "GRADIENT_PROJECTION_MODE must be 'nearest_most_frequent' or 'linear'."
        )

    print("Chosen projected mask:")
    print(f"  projection: {projection_description}")
    print(f"  vertices: {int(np.sum(gradient_mask))}")
    print(f"  fraction of LH surface: {np.mean(gradient_mask):.6f}")
    print(f"  projected value min: {float(np.nanmin(gradient_values)):.6f}")
    print(f"  projected value max: {float(np.nanmax(gradient_values)):.6f}")

    return np.asarray(gradient_mask, dtype=bool), gradient_values, projection_description


def make_parcel_mask(vertex_names, areas):
    return np.isin(vertex_names, areas)


def build_mesh_adjacency(n_vertices, faces):
    neighbors = [set() for _ in range(n_vertices)]

    for tri in faces:
        a, b, c = map(int, tri)

        neighbors[a].add(b)
        neighbors[a].add(c)

        neighbors[b].add(a)
        neighbors[b].add(c)

        neighbors[c].add(a)
        neighbors[c].add(b)

    return [np.array(sorted(x), dtype=np.int32) for x in neighbors]


def connected_components_for_mask(mask, adjacency):
    mask = np.asarray(mask, dtype=bool)
    labels = np.full(mask.shape[0], -1, dtype=int)
    components = []

    vertices = np.where(mask)[0]

    for start in vertices:
        if labels[start] >= 0:
            continue

        comp_id = len(components)
        queue = deque([int(start)])
        labels[start] = comp_id
        comp = []

        while queue:
            v = queue.popleft()
            comp.append(v)

            for nb in adjacency[v]:
                if not mask[nb]:
                    continue
                if labels[nb] >= 0:
                    continue

                labels[nb] = comp_id
                queue.append(int(nb))

        components.append(np.array(comp, dtype=np.int32))

    return labels, components


def report_mask_composition(name, mask, vertex_names, top_n=30):
    names = vertex_names[mask]
    counts = Counter(names)

    print(f"{name} parcel composition:")
    print(f"Total vertices: {int(np.sum(mask))}")

    items = sorted(counts.items(), key=lambda x: -x[1])
    for area, count in items[:top_n]:
        print(f"{str(area):>8} : {count}")

    if len(items) > top_n:
        print(f"  ... {len(items) - top_n} more parcels not shown")

    return counts


def report_connected_components(name, mask, adjacency, vertex_names, max_components=10):
    labels, components = connected_components_for_mask(mask, adjacency)

    sizes = np.array([len(c) for c in components], dtype=int)
    order = np.argsort(-sizes) if len(sizes) > 0 else np.array([], dtype=int)

    print(f"Connected components for {name}:")
    print(f"Total vertices: {int(np.sum(mask))}")
    print(f"Number of connected components: {len(components)}")

    if len(components) == 0:
        return labels, components

    print(f"Largest component size: {int(sizes[order[0]])}")
    print(f"Smallest component size: {int(sizes[order[-1]])}")
    print("Component sizes, largest first:")
    print(sizes[order[:max_components]])

    for rank, comp_id in enumerate(order[:max_components]):
        comp = components[comp_id]
        comp_counts = Counter(vertex_names[comp])
        top_parcels = sorted(comp_counts.items(), key=lambda x: -x[1])[:8]

        print(f"  Component rank {rank + 1}")
        print(f"  component id: {comp_id}")
        print(f"  size: {len(comp)}")
        print(f"  first 20 vertex indices: {comp[:20]}")
        print("  top parcels:")
        for area, count in top_parcels:
            print(f"    {str(area):>8} : {count}")

    return labels, components


def compute_distances_to_anchor(coords, faces, target_vertices, anchor_vertices):
    target_vertices = np.asarray(target_vertices, dtype=np.int32)
    anchor_vertices = np.asarray(anchor_vertices, dtype=np.int32)

    if len(target_vertices) == 0:
        raise ValueError("target_vertices is empty.")
    if len(anchor_vertices) == 0:
        raise ValueError("anchor_vertices is empty.")

    if DISTANCE_MODE == "geodesic":
        try:
            import gdist

            distances = gdist.compute_gdist(
                coords.astype(np.float64),
                faces.astype(np.int32),
                source_indices=anchor_vertices,
                target_indices=target_vertices,
            )
            distances = np.asarray(distances, dtype=float)
            return distances, "surface geodesic distance to area 25"

        except Exception as exc:
            print("WARNING: geodesic distance failed.")
            print(f"Reason: {repr(exc)}")
            print("Falling back to Euclidean xyz distance.")

    if DISTANCE_MODE not in {"geodesic", "euclidean"}:
        raise ValueError("DISTANCE_MODE must be 'geodesic' or 'euclidean'.")

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(coords[anchor_vertices])
        distances, _nearest = tree.query(coords[target_vertices], k=1)
        distances = np.asarray(distances, dtype=float)
        return distances, "Euclidean xyz distance to area 25"

    except Exception:
        anchor_coords = coords[anchor_vertices]
        distances = np.empty(len(target_vertices), dtype=float)

        chunk = 2048
        for start in range(0, len(target_vertices), chunk):
            stop = min(start + chunk, len(target_vertices))
            diff = coords[target_vertices[start:stop], None, :] - anchor_coords[None, :, :]
            distances[start:stop] = np.sqrt(np.sum(diff * diff, axis=-1)).min(axis=1)

        return distances, "Euclidean xyz distance to area 25"


def choose_anchor_zone_within_mask(mask_vertices, distances):
    mask_vertices = np.asarray(mask_vertices, dtype=np.int32)
    distances = np.asarray(distances, dtype=float)

    if ANCHOR_ZONE_N is not None:
        n_zone = int(ANCHOR_ZONE_N)
    else:
        n_zone = int(np.ceil(float(ANCHOR_ZONE_FRACTION) * len(mask_vertices)))

    n_zone = max(1, min(n_zone, len(mask_vertices)))

    order = np.argsort(distances)
    chosen = mask_vertices[order[:n_zone]]
    chosen_distances = distances[order[:n_zone]]

    return chosen.astype(np.int32), chosen_distances, order


def print_anchor_info(
    *,
    mask,
    area25_mask,
    closest_mask,
    mask_vertices,
    distances,
    distance_description,
    vertex_names,
):
    print("Anchor/input-zone info:")
    print("Area 25 is used as an external reference and is not added to the model-space mask.")
    print(f"Distance definition: {distance_description}")
    print(f"Projected model-space mask vertices: {int(np.sum(mask))}")
    print(f"Area 25 reference vertices: {int(np.sum(area25_mask))}")
    print(f"Closest-within-mask anchor/input-zone vertices: {int(np.sum(closest_mask))}")

    print("Distance from projected mask vertices to area 25:")
    print(f"  min:    {float(np.nanmin(distances)):.3f}")
    print(f"  median: {float(np.nanmedian(distances)):.3f}")
    print(f"  mean:   {float(np.nanmean(distances)):.3f}")
    print(f"  max:    {float(np.nanmax(distances)):.3f}")

    closest_vertices = np.where(closest_mask)[0]
    closest_distances = distances[np.isin(mask_vertices, closest_vertices)]

    print("Distance for chosen closest-within-mask vertices:")
    print(f"  min:    {float(np.nanmin(closest_distances)):.3f}")
    print(f"  median: {float(np.nanmedian(closest_distances)):.3f}")
    print(f"  mean:   {float(np.nanmean(closest_distances)):.3f}")
    print(f"  max:    {float(np.nanmax(closest_distances)):.3f}")

    report_mask_composition("PROJECTED_MASK", mask, vertex_names)
    report_mask_composition("AREA25_REFERENCE", area25_mask, vertex_names)
    report_mask_composition("CLOSEST_WITHIN_MASK_TO_AREA25", closest_mask, vertex_names)


def save_outputs(
    *,
    mask,
    area25_mask,
    closest_mask,
    mask_vertices,
    distances,
    coords,
    vertex_names,
    distance_description,
    projection_description,
):
    out_base_dir = REPO_ROOT / "data" / "embedding" / "custom_roi_vertices"
    out_roi_dir = out_base_dir / roi_name()
    out_roi_dir.mkdir(parents=True, exist_ok=True)

    out_check_dir = REPO_ROOT / "data" / "embedding" / "anchor_checks"
    out_check_dir.mkdir(parents=True, exist_ok=True)

    model_space_filename = f"{roi_name()}_lh_vertex_indices.npy"
    area25_filename = "area25_reference_lh_vertex_indices.npy"
    anchor_filename = (
        f"{roi_name()}_closest_to_area25_{fraction_tag()}_lh_vertex_indices.npy"
    )

    model_space_path = out_roi_dir / model_space_filename
    area25_path = out_roi_dir / area25_filename
    anchor_path = out_roi_dir / anchor_filename

    files = {
        model_space_path: np.where(mask)[0],
        area25_path: np.where(area25_mask)[0],
        anchor_path: np.where(closest_mask)[0],
    }

    print("Saving vertex index files:")

    for path, inds in files.items():
        np.save(path, inds.astype(np.int32))
        print(f"{path.name}: {len(inds)} vertices")
        print(f"  {path}")

    order = np.argsort(distances)
    csv_path = out_check_dir / f"{OUT_PREFIX}_mask_vertices_ranked_by_distance.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "vertex_index",
                "distance_to_area25",
                "x",
                "y",
                "z",
                "parcel_label",
                "projection",
                "distance_description",
            ]
        )

        for rank, idx_in_mask in enumerate(order, start=1):
            vertex = int(mask_vertices[idx_in_mask])
            x, y, z = coords[vertex]
            writer.writerow(
                [
                    rank,
                    vertex,
                    float(distances[idx_in_mask]),
                    float(x),
                    float(y),
                    float(z),
                    str(vertex_names[vertex]),
                    projection_description,
                    distance_description,
                ]
            )

    print("Saving distance table:")
    print(csv_path)

    spec = {
        "roi_name": roi_name(),
        "projection_mode": GRADIENT_PROJECTION_MODE,
        "linear_threshold": (
            float(GRADIENT_LINEAR_THRESHOLD)
            if GRADIENT_PROJECTION_MODE == "linear"
            else None
        ),
        "projection_description": projection_description,
        "model_space_vertex_file": repo_relative(model_space_path),
        "anchor_vertex_file": repo_relative(anchor_path),
        "external_reference_vertex_file": repo_relative(area25_path),
        "anchor_reference_areas": list(ANCHOR_REFERENCE_AREAS),
        "anchor_zone_fraction": (
            None if ANCHOR_ZONE_N is not None else float(ANCHOR_ZONE_FRACTION)
        ),
        "anchor_zone_n_requested": (
            None if ANCHOR_ZONE_N is None else int(ANCHOR_ZONE_N)
        ),
        "distance_mode": DISTANCE_MODE,
        "distance_description": distance_description,
        "n_model_vertices": int(np.sum(mask)),
        "n_anchor_vertices": int(np.sum(closest_mask)),
        "n_external_reference_vertices": int(np.sum(area25_mask)),
        "distance_to_area25_min": float(np.nanmin(distances)),
        "distance_to_area25_median": float(np.nanmedian(distances)),
        "distance_to_area25_mean": float(np.nanmean(distances)),
        "distance_to_area25_max": float(np.nanmax(distances)),
    }

    spec_path = out_roi_dir / "roi_specs.json"
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)

    print("Saving ROI spec:")
    print(spec_path)


def plot_single_mask(ax, coords, faces, bg_sulc, mask, title, color):
    vals = np.full(mask.shape, np.nan, dtype=float)
    vals[mask] = 1.0

    cmap = ListedColormap(["#ffffff", color])

    nilearn_plotting.plot_surf_stat_map(
        surf_mesh=(coords, faces),
        stat_map=vals,
        hemi="left",
        view="medial",
        bg_map=bg_sulc,
        bg_on_data=True,
        cmap=cmap,
        threshold=0.5,
        vmin=0.0,
        vmax=1.0,
        axes=ax,
        title=title,
        colorbar=False,
    )


def plot_combined_mask_anchor(ax, coords, faces, bg_sulc, mask, area25_mask, closest_mask, title):
    vals = np.full(mask.shape, np.nan, dtype=float)

    vals[mask] = 1.0
    vals[area25_mask] = 2.0
    vals[closest_mask] = 3.0

    cmap = ListedColormap(
        [
            "#e6953f",
            "#7b3294",
            "#111111",
        ]
    )

    nilearn_plotting.plot_surf_stat_map(
        surf_mesh=(coords, faces),
        stat_map=vals,
        hemi="left",
        view="medial",
        bg_map=bg_sulc,
        bg_on_data=True,
        cmap=cmap,
        threshold=0.5,
        vmin=1.0,
        vmax=3.0,
        axes=ax,
        title=title,
        colorbar=False,
    )


def plot_distance_map(ax, coords, faces, bg_sulc, mask_vertices, distances, title):
    vals = np.full(coords.shape[0], np.nan, dtype=float)
    vals[mask_vertices] = distances

    vmax = float(np.nanpercentile(distances, 95))

    nilearn_plotting.plot_surf_stat_map(
        surf_mesh=(coords, faces),
        stat_map=vals,
        hemi="left",
        view="medial",
        bg_map=bg_sulc,
        bg_on_data=True,
        cmap="viridis",
        threshold=0.0,
        vmin=0.0,
        vmax=vmax,
        axes=ax,
        title=title,
        colorbar=True,
    )


def save_figure(
    *,
    coords,
    faces,
    bg_sulc,
    mask,
    area25_mask,
    closest_mask,
    mask_vertices,
    distances,
    projection_description,
    distance_description,
):
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3)

    ax00 = fig.add_subplot(gs[0, 0], projection="3d")
    ax01 = fig.add_subplot(gs[0, 1], projection="3d")
    ax02 = fig.add_subplot(gs[0, 2], projection="3d")

    ax10 = fig.add_subplot(gs[1, 0], projection="3d")
    ax11 = fig.add_subplot(gs[1, 1], projection="3d")
    ax12 = fig.add_subplot(gs[1, 2], projection="3d")

    plot_single_mask(
        ax00,
        coords,
        faces,
        bg_sulc,
        mask,
        f"Projected mask\n{projection_description}\nn={int(np.sum(mask))}",
        color="#e6953f",
    )

    plot_single_mask(
        ax01,
        coords,
        faces,
        bg_sulc,
        area25_mask,
        f"Area 25 reference\nnot in model space\nn={int(np.sum(area25_mask))}",
        color="#7b3294",
    )

    plot_single_mask(
        ax02,
        coords,
        faces,
        bg_sulc,
        closest_mask,
        (
            "Closest mask vertices to area 25\n"
            f"candidate input/anchor zone\nn={int(np.sum(closest_mask))}"
        ),
        color="#111111",
    )

    plot_combined_mask_anchor(
        ax10,
        coords,
        faces,
        bg_sulc,
        mask,
        area25_mask,
        closest_mask,
        (
            "Mask + area 25 + closest zone\n"
            "orange=mask, purple=25, black=closest"
        ),
    )

    plot_distance_map(
        ax11,
        coords,
        faces,
        bg_sulc,
        mask_vertices,
        distances,
        "Distance from mask vertices to area 25",
    )

    plot_combined_mask_anchor(
        ax12,
        coords,
        faces,
        bg_sulc,
        mask,
        area25_mask,
        closest_mask,
        "Candidate model input boundary",
    )

    fig.suptitle(
        "Projected mask and area-25-derived anchor/input zone\n"
        f"projection = {projection_description}; distance = {distance_description}",
        fontsize=15,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    out_fig = REPO_ROOT / "figures" / f"{OUT_PREFIX}_medial.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saving figure:")
    print(out_fig)


def main():
    unique_names, vertex_names = build_vertex_names()
    coords, faces, bg_sulc = load_surface_and_sulc()

    print("Annotation and surface info:")
    print(f"Loaded annot: {ANNOT_PATH}")
    print(f"Unique parcel names in annot table: {len(unique_names)}")
    print(f"LH surface coords shape: {coords.shape}")
    print(f"LH surface faces shape: {faces.shape}")

    for area in ANCHOR_REFERENCE_AREAS:
        print(f"Parcel {area!r} exists in annot? {area in unique_names}")

    mask, _gradient_values, projection_description = load_gradient_mask_to_surface(
        coords, faces
    )

    area25_mask = make_parcel_mask(vertex_names, ANCHOR_REFERENCE_AREAS)

    if int(np.sum(mask)) == 0:
        raise ValueError("Projected mask is empty.")
    if int(np.sum(area25_mask)) == 0:
        raise ValueError(f"Anchor reference mask is empty: {ANCHOR_REFERENCE_AREAS}")

    mask_vertices = np.where(mask)[0].astype(np.int32)
    area25_vertices = np.where(area25_mask)[0].astype(np.int32)

    distances, distance_description = compute_distances_to_anchor(
        coords,
        faces,
        target_vertices=mask_vertices,
        anchor_vertices=area25_vertices,
    )

    closest_vertices, _closest_distances, _order = choose_anchor_zone_within_mask(
        mask_vertices,
        distances,
    )

    closest_mask = np.zeros(mask.shape[0], dtype=bool)
    closest_mask[closest_vertices] = True

    adjacency = build_mesh_adjacency(coords.shape[0], faces)

    print_anchor_info(
        mask=mask,
        area25_mask=area25_mask,
        closest_mask=closest_mask,
        mask_vertices=mask_vertices,
        distances=distances,
        distance_description=distance_description,
        vertex_names=vertex_names,
    )

    report_connected_components(
        "PROJECTED_MASK",
        mask,
        adjacency,
        vertex_names,
    )

    report_connected_components(
        "AREA25_REFERENCE",
        area25_mask,
        adjacency,
        vertex_names,
    )

    report_connected_components(
        "CLOSEST_WITHIN_MASK_TO_AREA25",
        closest_mask,
        adjacency,
        vertex_names,
    )

    save_outputs(
        mask=mask,
        area25_mask=area25_mask,
        closest_mask=closest_mask,
        mask_vertices=mask_vertices,
        distances=distances,
        coords=coords,
        vertex_names=vertex_names,
        distance_description=distance_description,
        projection_description=projection_description,
    )

    save_figure(
        coords=coords,
        faces=faces,
        bg_sulc=bg_sulc,
        mask=mask,
        area25_mask=area25_mask,
        closest_mask=closest_mask,
        mask_vertices=mask_vertices,
        distances=distances,
        projection_description=projection_description,
        distance_description=distance_description,
    )


if __name__ == "__main__":
    main()