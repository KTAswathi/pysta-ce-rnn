from pathlib import Path
from collections import Counter, deque

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
GRADIENT_PROJECTION_MODE = "nearest_most_frequent"
GRADIENT_LINEAR_THRESHOLD = 0.25

# GRADIENT_LINEAR_THRESHOLD is used only when GRADIENT_PROJECTION_MODE = "linear".
# Examples:
#   0.00 -> max vertices
#   0.01 -> almost max, but removes tiny numerical traces
#   0.50 -> very high confidence only

# mPFC = projected gradient mask ∪ a24 ∪ 25
MPFC_EXTRA_AREAS = ["a24", "25"]

# Anchor = 25
ANCHOR_AREAS = ["25"]


def threshold_tag():
    if GRADIENT_PROJECTION_MODE == "nearest_most_frequent":
        return "nearest"

    if GRADIENT_PROJECTION_MODE == "linear":
        return f"linear{str(GRADIENT_LINEAR_THRESHOLD).replace('.', 'p')}"

    return str(GRADIENT_PROJECTION_MODE)


OUT_PREFIX = f"mpfc_union_gradient_{threshold_tag()}_a24_25_anchor25"
LEGACY_OUT_PREFIX = "mpfc_union_gradient_a24_25_anchor25"


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


def print_volume_debug(img, data):
    print("VOLUME MASK DEBUG")
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

    if data.ndim == 3 and nonzero_voxels > 0:
        ijk = np.argwhere(data > 0)
        ijk_min = ijk.min(axis=0)
        ijk_max = ijk.max(axis=0)

        corners = np.array(
            [
                [ijk_min[0], ijk_min[1], ijk_min[2]],
                [ijk_min[0], ijk_min[1], ijk_max[2]],
                [ijk_min[0], ijk_max[1], ijk_min[2]],
                [ijk_min[0], ijk_max[1], ijk_max[2]],
                [ijk_max[0], ijk_min[1], ijk_min[2]],
                [ijk_max[0], ijk_min[1], ijk_max[2]],
                [ijk_max[0], ijk_max[1], ijk_min[2]],
                [ijk_max[0], ijk_max[1], ijk_max[2]],
            ],
            dtype=float,
        )

        world_corners = nib.affines.apply_affine(img.affine, corners)

        print(f"Nonzero voxel bbox ijk min: {ijk_min}")
        print(f"Nonzero voxel bbox ijk max: {ijk_max}")
        print(f"Nonzero bbox world min xyz: {world_corners.min(axis=0)}")
        print(f"Nonzero bbox world max xyz: {world_corners.max(axis=0)}")


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

    print_volume_debug(img, data)

    flat = data.reshape(-1)

    print("SURFACE PROJECTION DEBUG")
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

    print("\nProjection diagnostics:")
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

    print("\nChosen projected gradient mask:")
    print(f"  projection: {projection_description}")
    print(f"  vertices: {int(np.sum(gradient_mask))}")
    print(f"  fraction of LH surface: {np.mean(gradient_mask):.6f}")
    print(f"  value min: {float(np.nanmin(gradient_values)):.6f}")
    print(f"  value max: {float(np.nanmax(gradient_values)):.6f}")

    return np.asarray(gradient_mask, dtype=bool), gradient_values, projection_description


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


def report_mask_composition(name, mask, vertex_names, top_n=25):
    names = vertex_names[mask]
    counts = Counter(names)

    print(f"\n{name} parcel composition:")
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

    print(f"CONNECTED COMPONENT DEBUG: {name}")
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

        print(f"\n  Component rank {rank + 1}")
        print(f"  component id: {comp_id}")
        print(f"  size: {len(comp)}")
        print(f"  first 20 vertex indices: {comp[:20]}")
        print("  top parcels:")
        for area, count in top_parcels:
            print(f"    {str(area):>8} : {count}")

    return labels, components


def make_parcel_mask(vertex_names, areas):
    return np.isin(vertex_names, areas)


def print_union_debug(
    *,
    gradient_mask,
    a24_mask,
    area25_mask,
    anchor_mask,
    union_mask,
    adjacency,
    vertex_names,
    projection_description,
):
    print("FINAL MPFC UNION DEBUG")

    print(f"gradient projection:       {projection_description}")
    print(f"gradient projected mask vertices:   {int(np.sum(gradient_mask))}")
    print(f"a24 parcel vertices:              {int(np.sum(a24_mask))}")
    print(f"25 parcel vertices:               {int(np.sum(area25_mask))}")
    print(f"Anchor areas:                     {ANCHOR_AREAS}")
    print(f"Anchor vertices:                  {int(np.sum(anchor_mask))}")
    print(f"Final mPFC union vertices:        {int(np.sum(union_mask))}")

    print("\nOverlap / inclusion:")
    print(f"gradient ∩ a24:                  {int(np.sum(gradient_mask & a24_mask))}")
    print(f"gradient ∩ 25:                   {int(np.sum(gradient_mask & area25_mask))}")
    print(f"a24 ∩ 25:                        {int(np.sum(a24_mask & area25_mask))}")
    print(f"anchor ∩ gradient:               {int(np.sum(anchor_mask & gradient_mask))}")
    print(f"anchor outside gradient:         {int(np.sum(anchor_mask & ~gradient_mask))}")
    print(f"anchor inside final union:        {int(np.sum(anchor_mask & union_mask))}")
    print(f"anchor outside final union:       {int(np.sum(anchor_mask & ~union_mask))}")

    report_mask_composition("PROJECTED_GRADIENT_MASK", gradient_mask, vertex_names)
    report_mask_composition("A24_MASK", a24_mask, vertex_names)
    report_mask_composition("AREA_25_MASK / ANCHOR", area25_mask, vertex_names)
    report_mask_composition("FINAL_MPFC_UNION", union_mask, vertex_names)

    report_connected_components(
        "PROJECTED_GRADIENT_MASK",
        gradient_mask,
        adjacency,
        vertex_names,
    )

    report_connected_components(
        "A24_MASK",
        a24_mask,
        adjacency,
        vertex_names,
    )

    report_connected_components(
        "AREA_25_MASK / ANCHOR",
        area25_mask,
        adjacency,
        vertex_names,
    )

    report_connected_components(
        "FINAL_MPFC_UNION",
        union_mask,
        adjacency,
        vertex_names,
    )


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


def plot_union_with_anchor(ax, coords, faces, bg_sulc, union_mask, anchor_mask, title):
    vals = np.full(union_mask.shape, np.nan, dtype=float)
    vals[union_mask] = 1.0
    vals[anchor_mask] = 2.0

    cmap = ListedColormap(
        [
            "#e6953f",  # mPFC union
            "#111111",  # anchor 25
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
        vmax=2.0,
        axes=ax,
        title=title,
        colorbar=False,
    )


def save_vertex_indices(
    *,
    gradient_mask,
    a24_mask,
    area25_mask,
    anchor_mask,
    union_mask,
):
    out_data_dir = REPO_ROOT / "data" / "embedding" / "custom_roi_vertices"
    out_data_dir.mkdir(parents=True, exist_ok=True)

    files = {
        f"gradient_mask_{threshold_tag()}_lh_vertex_indices.npy": np.where(gradient_mask)[0],
        "a24_lh_vertex_indices.npy": np.where(a24_mask)[0],
        "area25_lh_vertex_indices.npy": np.where(area25_mask)[0],
        "anchor25_lh_vertex_indices.npy": np.where(anchor_mask)[0],
        f"{OUT_PREFIX}_lh_vertex_indices.npy": np.where(union_mask)[0],
        # legacy filename used by current mpfc_embedding.py default
        f"{LEGACY_OUT_PREFIX}_lh_vertex_indices.npy": np.where(union_mask)[0],
    }

    print("SAVED VERTEX INDEX FILES")

    for filename, inds in files.items():
        path = out_data_dir / filename
        np.save(path, inds.astype(np.int32))
        print(f"{filename}: {len(inds)} vertices")
        print(f"  {path}")


def main():
    unique_names, vertex_names = build_vertex_names()
    coords, faces, bg_sulc = load_surface_and_sulc()

    print("ANNOTATION / SURFACE DEBUG")
    print(f"Loaded annot: {ANNOT_PATH}")
    print(f"Unique parcel names in annot table: {len(unique_names)}")
    print(f"LH surface coords shape: {coords.shape}")
    print(f"LH surface faces shape: {faces.shape}")

    for area in MPFC_EXTRA_AREAS + ANCHOR_AREAS:
        print(f"Parcel {area!r} exists in annot? {area in unique_names}")

    gradient_mask, _gradient_values, projection_description = load_gradient_mask_to_surface(
        coords, faces
    )

    a24_mask = make_parcel_mask(vertex_names, ["a24"])
    area25_mask = make_parcel_mask(vertex_names, ["25"])
    anchor_mask = make_parcel_mask(vertex_names, ANCHOR_AREAS)

    union_mask = gradient_mask | a24_mask | area25_mask

    adjacency = build_mesh_adjacency(coords.shape[0], faces)

    print_union_debug(
        gradient_mask=gradient_mask,
        a24_mask=a24_mask,
        area25_mask=area25_mask,
        anchor_mask=anchor_mask,
        union_mask=union_mask,
        adjacency=adjacency,
        vertex_names=vertex_names,
        projection_description=projection_description,
    )

    save_vertex_indices(
        gradient_mask=gradient_mask,
        a24_mask=a24_mask,
        area25_mask=area25_mask,
        anchor_mask=anchor_mask,
        union_mask=union_mask,
    )

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
        gradient_mask,
        f"gradient mask\n{projection_description}\nn={int(np.sum(gradient_mask))}",
        color="#e6953f",
    )

    plot_single_mask(
        ax01,
        coords,
        faces,
        bg_sulc,
        a24_mask,
        f"a24 parcel\nn={int(np.sum(a24_mask))}",
        color="#7b3294",
    )

    plot_single_mask(
        ax02,
        coords,
        faces,
        bg_sulc,
        area25_mask,
        f"25 anchor parcel\nn={int(np.sum(area25_mask))}",
        color="#111111",
    )

    plot_single_mask(
        ax10,
        coords,
        faces,
        bg_sulc,
        union_mask,
        (
            "mPFC union\n"
            f"gradient ∪ a24 ∪ 25\nn={int(np.sum(union_mask))}"
        ),
        color="#e6953f",
    )

    plot_union_with_anchor(
        ax11,
        coords,
        faces,
        bg_sulc,
        union_mask,
        anchor_mask,
        (
            "mPFC union + anchor\n"
            f"orange=union, black=25\nn={int(np.sum(union_mask))}"
        ),
    )

    plot_union_with_anchor(
        ax12,
        coords,
        faces,
        bg_sulc,
        gradient_mask | a24_mask,
        area25_mask,
        (
            "gradient ∪ a24 + 25 anchor\n"
            "orange=gradient/a24, black=25"
        ),
    )

    fig.suptitle(
        f"mPFC union from the gradient mask, a24, and 25 anchor\n"
        f"projection = {projection_description}",
        fontsize=15,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_fig = REPO_ROOT / "figures" / f"{OUT_PREFIX}_medial.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("SAVED FIGURE")
    print(out_fig)
    print()


if __name__ == "__main__":
    main()