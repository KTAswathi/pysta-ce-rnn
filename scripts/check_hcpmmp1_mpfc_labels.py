from pathlib import Path
from collections import Counter, deque

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from nilearn import plotting as nilearn_plotting


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


CORE_MPFC = [
    "33pr", "p24pr", "a24pr", "p24", "a24",
    "p32pr", "a32pr", "d32", "p32", "s32",
    "8BM", "9m", "10v", "10r", "25",
]

BROAD_MPFC = CORE_MPFC + [
    "10d", "a10p", "p10p", "10pp", "OFC", "pOFC",
    "SCEF", "6ma", "6mp", "24dv", "24dd",
]

FOCUSED_MPFC = [
    "a24pr", "a24", "p24pr", "p24",
    "a32pr", "d32", "p32", "p32pr",
    "8BM", "9m",
]

VERY_FOCUSED_MPFC = [
    "a32pr", "d32", "p32", "p32pr",
    "8BM", "9m",
]

DORSAL_32_ONLY = [
    "a32pr", "d32", "p32", "p32pr",
]

DORSAL_32_PLUS_8BM = DORSAL_32_ONLY + ["8BM"]
DORSAL_32_PLUS_9M = DORSAL_32_ONLY + ["9m"]

# Anchor: bottom dark patch from your diagnostic plot
ANCHOR_AREAS = ["p32"]

# Custom vertex-level ROI:
# grow from p32, inside VERY_FOCUSED_MPFC, until we have enough vertices for Nrec=800
CUSTOM_ALLOWED_AREAS = VERY_FOCUSED_MPFC
CUSTOM_SEED_AREAS = ANCHOR_AREAS
CUSTOM_TARGET_N_VERTICES = 800
CUSTOM_ROI_NAME = "mpfc_custom_connected_p32_seed_n800"


def simplify_name(n):
    if isinstance(n, bytes):
        n = n.decode("utf-8")
    n = str(n)

    # e.g. L_10r_ROI -> 10r
    if n.startswith("L_") and n.endswith("_ROI"):
        return n[2:-4]

    return n


def build_vertex_names():
    if not ANNOT_PATH.exists():
        raise FileNotFoundError(f"Could not find: {ANNOT_PATH}")

    labels, ctab, raw_names = nib.freesurfer.read_annot(str(ANNOT_PATH))
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


def report_roi(name, roi_list, vertex_names, unique_names):
    vertex_counts = Counter(vertex_names[vertex_names != "<invalid>"])

    print(f"\n{name} labels:")
    for area in roi_list:
        print(f"{area:>6} : {'YES' if area in unique_names else 'NO'}")

    print(f"\nVertex counts per {name} parcel:")
    total = 0
    for area in roi_list:
        c = vertex_counts.get(area, 0)
        total += c
        print(f"{area:>6} : {c}")

    print(f"\nTotal {name} vertices: {total}")
    print(f"Enough for 800 units? {'YES' if total >= 800 else 'NO'}")

    mask = np.isin(vertex_names, roi_list)
    inds = np.where(mask)[0]

    print(f"\nNumber of {name} surface vertices: {len(inds)}")
    print(f"First 20 {name} vertex indices:")
    print(inds[:20])

    return mask, inds


def report_mask_composition(name, mask, vertex_names):
    names = vertex_names[mask]
    counts = Counter(names)

    print(f"\n{name} parcel composition:")
    total = int(np.sum(mask))
    print(f"Total vertices: {total}")

    for area, count in sorted(counts.items(), key=lambda x: str(x[0])):
        print(f"{str(area):>8} : {count}")

    return counts


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


def grow_connected_roi_from_seed(
    *,
    allowed_mask,
    seed_mask,
    coords,
    faces,
    target_n_vertices,
):
    """
    Build a connected custom ROI.

    Start from seed vertices, expand by mesh-neighbour distance,
    stay inside allowed_mask, and stop after target_n_vertices.

    This is useful when HCP parcels are too coarse but we still need
    a continuous vertex-level ROI with at least Nrec vertices.
    """

    allowed_mask = np.asarray(allowed_mask, dtype=bool)
    seed_mask = np.asarray(seed_mask, dtype=bool) & allowed_mask

    allowed_vertices = np.where(allowed_mask)[0]
    seed_vertices = np.where(seed_mask)[0]

    if len(seed_vertices) == 0:
        raise ValueError("Seed mask has zero vertices inside allowed_mask.")

    if len(allowed_vertices) < target_n_vertices:
        raise ValueError(
            f"Allowed mask has only {len(allowed_vertices)} vertices, "
            f"but target_n_vertices={target_n_vertices}."
        )

    adjacency = build_mesh_adjacency(coords.shape[0], faces)

    distances = np.full(coords.shape[0], np.inf, dtype=float)
    queue = deque()

    for v in seed_vertices:
        distances[v] = 0.0
        queue.append(int(v))

    while queue:
        v = queue.popleft()
        next_dist = distances[v] + 1.0

        for nb in adjacency[v]:
            if not allowed_mask[nb]:
                continue
            if np.isfinite(distances[nb]):
                continue

            distances[nb] = next_dist
            queue.append(int(nb))

    reachable_mask = allowed_mask & np.isfinite(distances)
    reachable_vertices = np.where(reachable_mask)[0]

    if len(reachable_vertices) < target_n_vertices:
        raise ValueError(
            f"Only {len(reachable_vertices)} allowed vertices are reachable from seed, "
            f"but target_n_vertices={target_n_vertices}. The allowed ROI is disconnected."
        )

    seed_centroid = coords[seed_vertices].mean(axis=0)
    centroid_dist = np.linalg.norm(coords[reachable_vertices] - seed_centroid[None, :], axis=1)

    # Primary sort: mesh-hop distance from seed.
    # Secondary sort: Euclidean distance to seed centroid.
    # Tertiary sort: vertex index for reproducibility.
    order = np.lexsort(
        (
            reachable_vertices,
            centroid_dist,
            distances[reachable_vertices],
        )
    )

    selected_vertices = reachable_vertices[order[:target_n_vertices]]

    custom_mask = np.zeros(coords.shape[0], dtype=bool)
    custom_mask[selected_vertices] = True

    return custom_mask, selected_vertices, distances


def plot_roi(ax, coords, faces, bg_sulc, roi_mask, title):
    roi_vals = roi_mask.astype(int)

    nilearn_plotting.plot_surf_roi(
        surf_mesh=(coords, faces),
        roi_map=roi_vals,
        hemi="left",
        view="medial",
        bg_map=bg_sulc,
        bg_on_data=True,
        cmap="YlOrRd",
        axes=ax,
        title=title,
        colorbar=False,
    )


def plot_roi_with_anchor(ax, coords, faces, bg_sulc, roi_mask, anchor_mask, title):
    """
    0 = outside ROI
    1 = ROI
    2 = anchor
    """
    vals = np.zeros_like(roi_mask, dtype=int)
    vals[roi_mask] = 1
    vals[anchor_mask] = 2

    cmap = ListedColormap(
        [
            "#f5f2c7",  # pale background
            "#e6953f",  # ROI orange
            "#111111",  # anchor black
        ]
    )

    nilearn_plotting.plot_surf_roi(
        surf_mesh=(coords, faces),
        roi_map=vals,
        hemi="left",
        view="medial",
        bg_map=bg_sulc,
        bg_on_data=True,
        cmap=cmap,
        axes=ax,
        title=title,
        colorbar=False,
    )


def plot_allowed_custom_seed(
    ax,
    coords,
    faces,
    bg_sulc,
    allowed_mask,
    custom_mask,
    seed_mask,
    title,
):
    """
    0 = outside
    1 = allowed parent ROI
    2 = custom connected ROI
    3 = seed / anchor
    """

    vals = np.zeros_like(allowed_mask, dtype=int)
    vals[allowed_mask] = 1
    vals[custom_mask] = 2
    vals[seed_mask] = 3

    cmap = ListedColormap(
        [
            "#f5f2c7",  # outside
            "#ead79b",  # allowed parent ROI
            "#e6953f",  # custom ROI
            "#111111",  # seed anchor
        ]
    )

    nilearn_plotting.plot_surf_roi(
        surf_mesh=(coords, faces),
        roi_map=vals,
        hemi="left",
        view="medial",
        bg_map=bg_sulc,
        bg_on_data=True,
        cmap=cmap,
        axes=ax,
        title=title,
        colorbar=False,
    )


def main():
    unique_names, vertex_names = build_vertex_names()
    coords, faces, bg_sulc = load_surface_and_sulc()

    print(f"\nLoaded: {ANNOT_PATH}")
    print(f"Unique parcel names in annot table: {len(unique_names)}\n")

    # Main ROIs
    core_mask, core_inds = report_roi("CORE_MPFC", CORE_MPFC, vertex_names, unique_names)
    broad_mask, broad_inds = report_roi("BROAD_MPFC", BROAD_MPFC, vertex_names, unique_names)
    focused_mask, focused_inds = report_roi("FOCUSED_MPFC", FOCUSED_MPFC, vertex_names, unique_names)
    very_focused_mask, very_focused_inds = report_roi(
        "VERY_FOCUSED_MPFC", VERY_FOCUSED_MPFC, vertex_names, unique_names
    )

    # Smaller candidates
    dorsal32_mask, dorsal32_inds = report_roi(
        "DORSAL_32_ONLY", DORSAL_32_ONLY, vertex_names, unique_names
    )
    d32_8bm_mask, d32_8bm_inds = report_roi(
        "DORSAL_32_PLUS_8BM", DORSAL_32_PLUS_8BM, vertex_names, unique_names
    )
    d32_9m_mask, d32_9m_inds = report_roi(
        "DORSAL_32_PLUS_9M", DORSAL_32_PLUS_9M, vertex_names, unique_names
    )

    # Anchor
    anchor_mask, anchor_inds = report_roi(
        "ANCHOR_AREAS", ANCHOR_AREAS, vertex_names, unique_names
    )

    # Custom connected vertex ROI
    allowed_mask, allowed_inds = report_roi(
        "CUSTOM_ALLOWED_AREAS", CUSTOM_ALLOWED_AREAS, vertex_names, unique_names
    )

    seed_mask, seed_inds = report_roi(
        "CUSTOM_SEED_AREAS", CUSTOM_SEED_AREAS, vertex_names, unique_names
    )

    custom_mask, custom_inds, custom_distances = grow_connected_roi_from_seed(
        allowed_mask=allowed_mask,
        seed_mask=seed_mask,
        coords=coords,
        faces=faces,
        target_n_vertices=CUSTOM_TARGET_N_VERTICES,
    )

    report_mask_composition(
        f"CUSTOM_CONNECTED_ROI {CUSTOM_ROI_NAME}",
        custom_mask,
        vertex_names,
    )

    # Save custom vertices for later embedding generation
    custom_out_dir = REPO_ROOT / "data" / "embedding" / "custom_roi_vertices"
    custom_out_dir.mkdir(parents=True, exist_ok=True)

    custom_vertex_path = custom_out_dir / f"{CUSTOM_ROI_NAME}_vertex_indices.npy"
    np.save(custom_vertex_path, custom_inds.astype(np.int32))

    custom_dist_path = custom_out_dir / f"{CUSTOM_ROI_NAME}_mesh_hop_distance_from_seed.npy"
    np.save(custom_dist_path, custom_distances.astype(np.float32))

    print(f"\nSaved custom ROI vertex indices to:\n{custom_vertex_path}")
    print(f"\nSaved mesh-hop distances from seed to:\n{custom_dist_path}")

    # Figure 1: main ROI candidates
    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(2, 2)

    ax00 = fig.add_subplot(gs[0, 0], projection="3d")
    ax01 = fig.add_subplot(gs[0, 1], projection="3d")
    ax10 = fig.add_subplot(gs[1, 0], projection="3d")
    ax11 = fig.add_subplot(gs[1, 1], projection="3d")

    plot_roi(ax00, coords, faces, bg_sulc, core_mask, f"CORE_MPFC\nn={len(core_inds)}")
    plot_roi(ax01, coords, faces, bg_sulc, broad_mask, f"BROAD_MPFC\nn={len(broad_inds)}")
    plot_roi(ax10, coords, faces, bg_sulc, focused_mask, f"FOCUSED_MPFC\nn={len(focused_inds)}")
    plot_roi(
        ax11,
        coords,
        faces,
        bg_sulc,
        very_focused_mask,
        f"VERY_FOCUSED_MPFC\nn={len(very_focused_inds)}",
    )

    fig.suptitle("Candidate MPFC ROIs on LH medial surface", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = REPO_ROOT / "figures" / "roi_candidates_medial.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: smaller candidate / continuity diagnostic
    fig2 = plt.figure(figsize=(14, 14))
    gs2 = fig2.add_gridspec(2, 2)

    ax20 = fig2.add_subplot(gs2[0, 0], projection="3d")
    ax21 = fig2.add_subplot(gs2[0, 1], projection="3d")
    ax30 = fig2.add_subplot(gs2[1, 0], projection="3d")
    ax31 = fig2.add_subplot(gs2[1, 1], projection="3d")

    plot_roi(
        ax20,
        coords,
        faces,
        bg_sulc,
        dorsal32_mask,
        f"DORSAL_32_ONLY\nn={len(dorsal32_inds)}",
    )
    plot_roi(
        ax21,
        coords,
        faces,
        bg_sulc,
        d32_8bm_mask,
        f"DORSAL_32_PLUS_8BM\nn={len(d32_8bm_inds)}",
    )
    plot_roi(
        ax30,
        coords,
        faces,
        bg_sulc,
        d32_9m_mask,
        f"DORSAL_32_PLUS_9M\nn={len(d32_9m_inds)}",
    )
    plot_roi(
        ax31,
        coords,
        faces,
        bg_sulc,
        very_focused_mask,
        f"VERY_FOCUSED_MPFC\nn={len(very_focused_inds)}",
    )

    fig2.suptitle("Smaller diagnostic MPFC candidates", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path2 = REPO_ROOT / "figures" / "roi_candidates_medial_small.png"
    plt.savefig(out_path2, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # Figure 3: anchor diagnostics
    fig3 = plt.figure(figsize=(16, 6))
    gs3 = fig3.add_gridspec(1, 3)

    ax40 = fig3.add_subplot(gs3[0, 0], projection="3d")
    ax41 = fig3.add_subplot(gs3[0, 1], projection="3d")
    ax42 = fig3.add_subplot(gs3[0, 2], projection="3d")

    p32_mask = np.isin(vertex_names, ["p32"])
    p32pr_mask = np.isin(vertex_names, ["p32pr"])
    both_anchor_mask = np.isin(vertex_names, ["p32", "p32pr"])

    plot_roi_with_anchor(
        ax40,
        coords,
        faces,
        bg_sulc,
        dorsal32_mask,
        p32_mask,
        "DORSAL_32_ONLY + p32\nanchor=p32",
    )
    plot_roi_with_anchor(
        ax41,
        coords,
        faces,
        bg_sulc,
        dorsal32_mask,
        p32pr_mask,
        "DORSAL_32_ONLY + p32pr\nanchor=p32pr",
    )
    plot_roi_with_anchor(
        ax42,
        coords,
        faces,
        bg_sulc,
        dorsal32_mask,
        both_anchor_mask,
        "DORSAL_32_ONLY + both\nanchor=p32+p32pr",
    )

    fig3.suptitle("Anchor diagnostics on smaller candidate", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_path3 = REPO_ROOT / "figures" / "anchor_candidates_medial.png"
    plt.savefig(out_path3, dpi=300, bbox_inches="tight")
    plt.close(fig3)

    # Figure 4: custom connected ROI grown from p32
    fig4 = plt.figure(figsize=(16, 6))
    gs4 = fig4.add_gridspec(1, 3)

    ax50 = fig4.add_subplot(gs4[0, 0], projection="3d")
    ax51 = fig4.add_subplot(gs4[0, 1], projection="3d")
    ax52 = fig4.add_subplot(gs4[0, 2], projection="3d")

    plot_allowed_custom_seed(
        ax50,
        coords,
        faces,
        bg_sulc,
        allowed_mask,
        custom_mask,
        seed_mask,
        (
            f"Custom connected ROI\n"
            f"seed={CUSTOM_SEED_AREAS}, n={len(custom_inds)}"
        ),
    )

    plot_roi_with_anchor(
        ax51,
        coords,
        faces,
        bg_sulc,
        custom_mask,
        seed_mask,
        (
            f"Custom ROI + anchor\n"
            f"{CUSTOM_ROI_NAME}\n"
            f"anchor={CUSTOM_SEED_AREAS}"
        ),
    )

    plot_roi(
        ax52,
        coords,
        faces,
        bg_sulc,
        custom_mask,
        (
            f"Custom ROI only\n"
            f"{CUSTOM_ROI_NAME}\n"
            f"n={len(custom_inds)}"
        ),
    )

    fig4.suptitle("Custom connected vertex-level MPFC ROI", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_path4 = REPO_ROOT / "figures" / "custom_connected_mpfc_roi.png"
    plt.savefig(out_path4, dpi=300, bbox_inches="tight")
    plt.close(fig4)

    print(f"\nSaved figure to:\n{out_path}")
    print(f"\nSaved smaller diagnostic figure to:\n{out_path2}")
    print(f"\nSaved anchor diagnostic figure to:\n{out_path3}")
    print(f"\nSaved custom connected ROI figure to:\n{out_path4}\n")


if __name__ == "__main__":
    main()