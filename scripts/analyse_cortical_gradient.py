import argparse
import pickle
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pysta
from nilearn import plotting as nilearn_plotting
from scipy.stats import pearsonr


KNOWN_SUFFIXES = {
    "decoder_generalization_performance": "decoder",
    "decode_time_of_loc": "timeofloc",
    "planning_subspaces": "subspaces",
    "connectivity_data": "connectivity",
    "trial_data": "trial",
}


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def collect_model_groups(analyses_root):
    groups = defaultdict(dict)

    for p in analyses_root.glob("**/*.pickle"):
        name = p.name
        for suffix, short in KNOWN_SUFFIXES.items():
            tag = f"_{suffix}.pickle"
            if name.endswith(tag):
                prefix = name[: -len(tag)]
                groups[(p.parent, prefix)][short] = p
                break

    return groups


def model_relpath_no_suffix(models_root, model_pickle_path):
    return str(model_pickle_path.relative_to(models_root).with_suffix("")).replace("\\", "/")


def find_best_matching_model_pickle(repo_root, query):
    models_root = repo_root / "models"

    if query is not None:
        query = str(query).replace("\\", "/")

        # direct path hit
        direct = models_root / (query + ".p" if not query.endswith(".p") else query)
        if direct.exists():
            return direct

        # absolute path hit
        qpath = Path(query)
        if qpath.exists():
            return qpath

    candidates = list(models_root.glob("**/model*.p"))
    if not candidates:
        return None

    flat_map = []
    for c in candidates:
        rel = model_relpath_no_suffix(models_root, c)
        flat_map.append((c, rel))

    # exact relative-path match
    if query is not None:
        exact = [c for c, rel in flat_map if rel == query]
        if exact:
            return exact[0]

        contains = [c for c, rel in flat_map if (query in rel) or (rel in query)]
        if contains:
            contains.sort(key=lambda p: len(str(p)))
            return contains[0]

    scored = []
    for c, rel in flat_map:
        score = SequenceMatcher(None, str(query), rel).ratio() if query is not None else 0.0
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def choose_group(groups, query=None):
    items = list(groups.items())
    if len(items) == 0:
        return None, None

    if query is None:
        if len(items) == 1:
            return items[0]
        raise ValueError(
            "More than one analysis group found. Pass a query string to choose one."
        )

    query = str(query).replace("\\", "/")

    scored = []
    for (folder, prefix), files_dict in items:
        key = f"{folder}/{prefix}".replace("\\", "/")
        exact_bonus = 2.0 if (query == prefix or query == key) else 0.0
        contains_bonus = 1.0 if (query in prefix or query in key or key in query) else 0.0
        score = SequenceMatcher(None, query, key).ratio() + exact_bonus + contains_bonus
        scored.append((score, (folder, prefix), files_dict))

    scored.sort(key=lambda x: x[0], reverse=True)
    _, group_key, files_dict = scored[0]
    return group_key, files_dict


def extract_delay_preferences(subspace_obj, n_units):
    if not isinstance(subspace_obj, dict) or "Csubs" not in subspace_obj:
        return None, None, None

    Csubs = np.asarray(subspace_obj["Csubs"])
    if Csubs.ndim != 3:
        return None, None, None

    if n_units is not None and Csubs.shape[-1] != n_units:
        return None, None, None

    # Csubs: (delay, location, neuron)
    delay_score = np.linalg.norm(Csubs, axis=1)   # (delay, neuron)
    preferred = np.argmax(delay_score, axis=0)    # (neuron,)
    strength = np.max(delay_score, axis=0)        # (neuron,)

    return preferred, strength, delay_score


def _find_first_key(d, candidates):
    for k in candidates:
        if k in d:
            return k
    return None


def _prepare_activity_and_step_arrays(trial_obj, n_units):
    if not isinstance(trial_obj, dict):
        raise ValueError("trial_data pickle is not a dict.")

    rs_key = _find_first_key(trial_obj, ["rs", "r", "activity", "hidden", "hid"])
    step_key = _find_first_key(trial_obj, ["step_num", "step_nums", "steps"])

    if rs_key is None or step_key is None:
        raise ValueError(
            f"Could not find raw activity / step keys in trial_data. "
            f"Available keys: {list(trial_obj.keys())}"
        )

    rs = trial_obj[rs_key]
    step_num = trial_obj[step_key]

    if isinstance(rs, list):
        rs = np.stack([np.asarray(x) for x in rs], axis=0)
    else:
        rs = np.asarray(rs)

    if isinstance(step_num, list):
        step_num = np.stack([np.asarray(x) for x in step_num], axis=0)
    else:
        step_num = np.asarray(step_num)

    rs = np.asarray(rs)
    step_num = np.asarray(step_num)

    # remove trailing singleton unit dim if present
    if rs.ndim >= 1 and rs.shape[-1] == 1:
        rs = np.squeeze(rs, axis=-1)

    # collapse step_num if it has unnecessary singleton / repeated dimensions
    step_num = np.squeeze(step_num)

    # ensure units dim is last
    if rs.ndim == 2:
        if rs.shape[-1] != n_units and rs.shape[0] == n_units:
            rs = rs.T

        if rs.shape[-1] != n_units:
            raise ValueError(
                f"Could not align raw activity with Nrec={n_units}. rs.shape={rs.shape}"
            )

        if step_num.ndim == 1 and step_num.shape[0] == rs.shape[0]:
            step_flat = step_num.reshape(-1)
            rs_flat = rs.reshape(-1, n_units)
            return rs_flat, step_flat

        raise ValueError(
            f"Unsupported 2D rs / step_num combination. "
            f"rs.shape={rs.shape}, step_num.shape={step_num.shape}"
        )

    elif rs.ndim == 3:
        # aim for rs: (T, B, H)
        if rs.shape[-1] != n_units:
            if rs.shape[0] == n_units:
                rs = np.transpose(rs, (1, 2, 0))
            elif rs.shape[1] == n_units:
                rs = np.transpose(rs, (0, 2, 1))
            elif rs.shape[2] == n_units:
                pass
            else:
                raise ValueError(f"Could not identify units dim in rs.shape={rs.shape}")

        if rs.shape[-1] != n_units:
            raise ValueError(
                f"Could not align raw activity with Nrec={n_units}. rs.shape={rs.shape}"
            )

        T, B, H = rs.shape

        if step_num.ndim == 3:
            if step_num.shape[:2] == (T, B):
                step_num = step_num[:, :, 0]
            elif step_num.shape[-2:] == (T, B):
                step_num = step_num[0, :, :]
            elif step_num.shape == rs.shape:
                step_num = step_num[:, :, 0]
            else:
                raise ValueError(
                    f"Unsupported 3D step_num.shape={step_num.shape} for rs.shape={rs.shape}"
                )

        if step_num.ndim == 1:
            if step_num.shape[0] == T:
                step_grid = np.broadcast_to(step_num[:, None], (T, B))
            elif step_num.shape[0] == B:
                step_grid = np.broadcast_to(step_num[None, :], (T, B))
            else:
                raise ValueError(
                    f"step_num shape {step_num.shape} does not match rs.shape={rs.shape}"
                )

        elif step_num.ndim == 2:
            if step_num.shape == (T, B):
                step_grid = step_num
            elif step_num.shape == (B, T):
                step_grid = step_num.T
            else:
                raise ValueError(
                    f"Unsupported step_num.shape={step_num.shape} for rs.shape={rs.shape}"
                )
        else:
            raise ValueError(
                f"Unsupported step_num ndim={step_num.ndim}, shape={step_num.shape}"
            )

        rs_flat = rs.reshape(T * B, H)
        step_flat = step_grid.reshape(T * B)

        return rs_flat, step_flat

    else:
        raise ValueError(f"Unsupported raw activity array shape: {rs.shape}")


def extract_raw_activity_preferred_delay(trial_obj, n_units, n_delays):
    rs_flat, step_flat = _prepare_activity_and_step_arrays(trial_obj, n_units)

    # planning phase = negative step_num
    planning_mask = step_flat < 0
    if np.sum(planning_mask) == 0:
        raise ValueError("No planning-phase samples found in trial_data (no negative step_num).")

    rs_plan = rs_flat[planning_mask]
    step_plan = step_flat[planning_mask].astype(int)

    # convert step_num to planning delay index:
    # step_num = -1 -> delay 0 (next action)
    # step_num = -2 -> delay 1
    planning_steps = np.sort(np.unique(step_plan))
    step_to_delay = {step: i for i, step in enumerate(planning_steps)}
    delay_idx = np.array([step_to_delay[s] for s in step_plan], dtype=int)

    print("Raw activity step-to-delay mapping:")
    for step, delay in step_to_delay.items():
        print(f"  step_num {step} -> raw delay {delay}")

    valid = (delay_idx >= 0) & (delay_idx < n_delays)
    rs_plan = rs_plan[valid]
    delay_idx = delay_idx[valid]

    if len(delay_idx) == 0:
        raise ValueError("No valid planning-delay samples survived delay-index conversion.")

    delay_means = np.full((n_delays, n_units), np.nan, dtype=float)

    for d in range(n_delays):
        mask = delay_idx == d
        if np.sum(mask) > 0:
            delay_means[d] = np.nanmean(rs_plan[mask], axis=0)

    # fill empty delays with very small value so argmax still works
    fill_val = np.nanmin(delay_means[np.isfinite(delay_means)]) if np.any(np.isfinite(delay_means)) else 0.0
    delay_means = np.where(np.isfinite(delay_means), delay_means, fill_val - 1e-8)

    preferred = np.argmax(delay_means, axis=0)
    strength = np.max(delay_means, axis=0)

    return preferred, strength, delay_means


def load_surface_and_sulc(repo_root):
    surf_path = (
        repo_root
        / "data"
        / "embedding"
        / "raw_surface_data"
        / "human"
        / "fs_lr32"
        / "surf"
        / "fs_lr32.l.midthickness.surf.gii"
    )
    sulc_path = (
        repo_root
        / "data"
        / "embedding"
        / "raw_surface_data"
        / "human"
        / "fs_lr32"
        / "fs_lr32.l.sulc_data.func.gii"
    )

    if not surf_path.exists():
        raise FileNotFoundError(f"Missing surface file: {surf_path}")

    surf_gii = nib.load(str(surf_path))
    coords, faces = surf_gii.agg_data()
    coords = np.asarray(coords, dtype=float)
    faces = np.asarray(faces, dtype=np.int32)

    bg_sulc = None
    if sulc_path.exists():
        sulc_gii = nib.load(str(sulc_path))
        bg_sulc = np.asarray(sulc_gii.darrays[0].data, dtype=float).reshape(-1)

    return coords, faces, bg_sulc


def load_cortical_model_and_geometry(model_pickle_path, repo_root):
    training_result = load_pickle(model_pickle_path)
    rnn = training_result.get("rnn", None)
    if rnn is None:
        raise ValueError("Could not find 'rnn' in model pickle.")

    if not hasattr(rnn, "sampled_vertex_indices"):
        raise ValueError("This model does not look cortically embedded.")

    if not all(
        hasattr(rnn, x)
        for x in ["embedding_species", "embedding_name", "embedding_seed", "Nrec"]
    ):
        raise ValueError("Model is missing embedding metadata attributes.")

    embedding_dir = (
        repo_root
        / "data"
        / "embedding"
        / "subsampled"
        / str(rnn.embedding_species)
        / str(rnn.embedding_name)
        / f"units={int(rnn.Nrec)}_seed={int(rnn.embedding_seed)}"
    )

    roi_path = embedding_dir / "roi_vertex_indices.npy"
    area_path = embedding_dir / "area_labels.txt"
    v2c_path = embedding_dir / "vertex_to_cluster.npy"

    if not roi_path.exists():
        raise FileNotFoundError(f"Missing ROI vertex file: {roi_path}")
    if not area_path.exists():
        raise FileNotFoundError(f"Missing area label file: {area_path}")
    if not v2c_path.exists():
        raise FileNotFoundError(f"Missing vertex_to_cluster file: {v2c_path}")

    roi_vertex_indices = np.load(roi_path).astype(np.int32)
    sampled_indices = np.asarray(rnn.sampled_vertex_indices).astype(np.int32).squeeze()
    vertex_to_cluster = np.load(v2c_path).astype(np.int32)

    with open(area_path, "r") as f:
        area_labels = [line.strip() for line in f]

    distance_matrix = (
        np.asarray(rnn.distance_matrix).astype(float)
        if hasattr(rnn, "distance_matrix")
        else None
    )

    geometry = {
        "sampled_indices": sampled_indices,
        "roi_vertex_indices": roi_vertex_indices,
        "vertex_to_cluster": vertex_to_cluster,
        "area_labels": area_labels,
        "distance_matrix": distance_matrix,
        "anchor_index": int(rnn.anchor_index) if hasattr(rnn, "anchor_index") else None,
        "opposite_anchor_index": int(rnn.opposite_anchor_index)
        if hasattr(rnn, "opposite_anchor_index")
        else None,
        "embedding_dir": embedding_dir,
    }

    return rnn, geometry


def sampled_to_full_vertex_map(sampled_values, vertex_to_cluster, roi_vertex_indices, n_vertices):
    full_map = np.full(n_vertices, np.nan, dtype=float)

    roi_vertex_indices = np.asarray(roi_vertex_indices, dtype=int)
    roi_cluster_idx = np.asarray(vertex_to_cluster[roi_vertex_indices], dtype=int)

    valid = roi_cluster_idx >= 0
    full_map[roi_vertex_indices[valid]] = sampled_values[roi_cluster_idx[valid]]

    return full_map


def make_strength_keep_mask(strength, keep_frac=1.0):
    strength = np.asarray(strength, dtype=float)

    if not (0 < keep_frac <= 1):
        raise ValueError("keep_frac must be in (0, 1].")

    if keep_frac == 1.0:
        return np.ones(len(strength), dtype=bool), float(np.min(strength))

    cutoff = np.quantile(strength, 1.0 - keep_frac)
    keep_mask = strength >= cutoff
    return keep_mask, float(cutoff)


def mask_values(values, keep_mask):
    values = np.asarray(values).astype(float).copy()
    values[~keep_mask] = np.nan
    return values


def render_single_medial_surface(
    fig,
    ax,
    coords,
    faces,
    stat_map,
    bg_sulc,
    title,
    cmap="YlOrRd",
    vmin=None,
    vmax=None,
    threshold=1e-12,
    cbar_label=None,
):
    nilearn_plotting.plot_surf_stat_map(
        surf_mesh=(coords, faces),
        stat_map=stat_map,
        hemi="left",
        view="medial",
        bg_map=bg_sulc,
        bg_on_data=True,
        cmap=cmap,
        threshold=threshold,
        vmin=vmin,
        vmax=vmax,
        colorbar=False,
        axes=ax,
        title=title,
    )

    if cbar_label is not None:
        finite = np.asarray(stat_map, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            finite = np.array([0.0, 1.0])

        norm = mpl.colors.Normalize(
            vmin=np.min(finite) if vmin is None else vmin,
            vmax=np.max(finite) if vmax is None else vmax,
        )
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(cbar_label)


def plot_distance_vs_delay(ax, anchor_distance, preferred, strength, title):
    """
    axis convention:
    x = preferred delay / lag
    y = distance from anchor

    asks: where along the cortical anchor-distance axis do units preferring each delay sit?
    """

    x = np.asarray(preferred, dtype=float)
    y = np.asarray(anchor_distance, dtype=float)
    s = np.asarray(strength, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(s)
    x = x[valid]
    y = y[valid]
    s = s[valid]

    if len(x) == 0:
        ax.text(0.5, 0.5, "No units survived threshold", ha="center", va="center")
        ax.axis("off")
        return

    # normalize distance to make different models/ROIs easier to compare
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)

    # jitter x slightly so repeated integer delays do not sit exactly on top of each other
    rng = np.random.default_rng(0)
    x_jitter = x + rng.normal(0.0, 0.045, size=len(x))

    s_plot = 20 + 80 * (s / (np.max(s) + 1e-8))

    ax.scatter(x_jitter, y_norm, s=s_plot, c=x, cmap="YlOrRd", alpha=0.65)

    # correlation between preferred delay and normalized anchor distance
    r, p = pearsonr(x, y_norm)

    if len(np.unique(x)) > 1:
        coeffs = np.polyfit(x, y_norm, deg=1)
        xfit = np.linspace(np.min(x), np.max(x), 100)
        yfit = coeffs[0] * xfit + coeffs[1]
        ax.plot(xfit, yfit, linestyle="--", linewidth=2)

    ax.set_xlabel("preferred delay")
    ax.set_ylabel("distance from anchor (normalized)")
    ax.set_title(f"{title} (N={len(x)})")
    ax.set_xticks(np.arange(int(np.nanmin(x)), int(np.nanmax(x)) + 1))

    txt = f"r = {r:.3f}\np = {p:.3g}"

    ax.text(
        0.97,
        0.03,
        txt,
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
    )

    ax.grid(alpha=0.25)


def plot_mean_distance_by_preferred_delay(ax, anchor_distance, preferred, title, n_delays):
    """
    summary plot:
    x = preferred delay / lag
    y = mean distance from anchor among units preferring that delay

    """

    distance = np.asarray(anchor_distance, dtype=float)
    pref = np.asarray(preferred, dtype=float)

    valid = np.isfinite(distance) & np.isfinite(pref)
    distance = distance[valid]
    pref = pref[valid].astype(int)

    if len(distance) == 0:
        ax.text(0.5, 0.5, "No units survived threshold", ha="center", va="center")
        ax.axis("off")
        return

    distance_norm = (distance - distance.min()) / (distance.max() - distance.min() + 1e-8)

    delays = np.arange(n_delays)
    means = np.full(n_delays, np.nan, dtype=float)
    sems = np.full(n_delays, np.nan, dtype=float)
    counts = np.zeros(n_delays, dtype=int)

    for d in delays:
        vals = distance_norm[pref == d]
        counts[d] = len(vals)

        if len(vals) > 0:
            means[d] = np.mean(vals)
            sems[d] = np.std(vals) / np.sqrt(len(vals))

    ax.errorbar(delays, means, yerr=sems, marker="o", linewidth=2)

    ax.set_xlabel("preferred delay")
    ax.set_ylabel("mean distance from anchor (normalized)")
    ax.set_title(title)
    ax.set_xticks(delays)
    ax.grid(alpha=0.25)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="String used to choose one analysis group under data/rnn_analyses/MazeEnv_L4_max6",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=6,
        help="Number of distance bins for the binned-gradient plots",
    )
    parser.add_argument(
        "--keep_frac",
        type=float,
        default=1.0,
        help="Keep only the top fraction of units by subspace delay-selectivity strength",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    analyses_root = repo_root / "data" / "rnn_analyses" / "MazeEnv_L4_max6"

    groups = collect_model_groups(analyses_root)
    if not groups:
        raise FileNotFoundError(f"No analysis pickles found under {analyses_root}")

    (folder, prefix), files_dict = choose_group(groups, args.query)
    if "subspaces" not in files_dict:
        raise FileNotFoundError(
            "Could not find planning_subspaces pickle for the chosen group."
        )
    if "trial" not in files_dict:
        raise FileNotFoundError(
            "Could not find trial_data pickle for the chosen group."
        )

    print(f"Using analysis group:\n  folder = {folder}\n  prefix = {prefix}\n")

    model_query = args.query
    if model_query is None:
        rel_folder = folder.relative_to(analyses_root)
        model_query = str(rel_folder / prefix).replace("\\", "/")

    model_pickle_path = find_best_matching_model_pickle(repo_root, model_query)
    if model_pickle_path is None:
        raise FileNotFoundError("Could not find matching trained model pickle.")

    print(f"Matched model pickle:\n  {model_pickle_path}\n")

    coords, faces, bg_sulc = load_surface_and_sulc(repo_root)
    rnn, geometry = load_cortical_model_and_geometry(model_pickle_path, repo_root)

    print("embedding_name from model:", rnn.embedding_name)
    print("roi_vertex_count:", len(geometry["roi_vertex_indices"]))
    print("num vertices with cluster assignment:", int(np.sum(geometry["vertex_to_cluster"] >= 0)))
    print("unique sampled parcels:", sorted(set(geometry["area_labels"])))

    # subspace-derived preferred delay
    subspace_obj = load_pickle(files_dict["subspaces"])
    sub_pref, sub_strength, delay_score = extract_delay_preferences(
        subspace_obj, getattr(rnn, "Nrec", None)
    )
    if sub_pref is None:
        raise ValueError("Could not extract preferred delay from planning_subspaces.")

    n_delays = delay_score.shape[0]

    # raw-activity-derived preferred delay
    trial_obj = load_pickle(files_dict["trial"])
    raw_pref, raw_strength, raw_delay_means = extract_raw_activity_preferred_delay(
        trial_obj, getattr(rnn, "Nrec", None), n_delays
    )

    if geometry["distance_matrix"] is None or geometry["anchor_index"] is None:
        raise ValueError("Missing cortical distance matrix or anchor index in model.")

    anchor_distance = geometry["distance_matrix"][geometry["anchor_index"]]

    # thresholding based on subspace strength
    keep_mask, cutoff = make_strength_keep_mask(sub_strength, keep_frac=args.keep_frac)
    sub_pref_thr = mask_values(sub_pref, keep_mask)
    sub_strength_thr = mask_values(sub_strength, keep_mask)
    raw_pref_thr = mask_values(raw_pref, keep_mask)

    print(
        f"Thresholding by subspace delay strength: keep_frac={args.keep_frac:.3f} | "
        f"cutoff={cutoff:.6f} | kept {int(np.sum(keep_mask))}/{len(keep_mask)} units"
    )

    # brain maps
    anchor_distance_map = sampled_to_full_vertex_map(
        anchor_distance,
        geometry["vertex_to_cluster"],
        geometry["roi_vertex_indices"],
        coords.shape[0],
    )
    sub_pref_map = sampled_to_full_vertex_map(
        sub_pref_thr,
        geometry["vertex_to_cluster"],
        geometry["roi_vertex_indices"],
        coords.shape[0],
    )
    raw_pref_map = sampled_to_full_vertex_map(
        raw_pref_thr,
        geometry["vertex_to_cluster"],
        geometry["roi_vertex_indices"],
        coords.shape[0],
    )

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3)

    # top row: 3 brain plots
    ax00 = fig.add_subplot(gs[0, 0], projection="3d")
    ax01 = fig.add_subplot(gs[0, 1], projection="3d")
    ax02 = fig.add_subplot(gs[0, 2], projection="3d")

    # bottom row: scatter + 2 binned plots
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax12 = fig.add_subplot(gs[1, 2])

    render_single_medial_surface(
        fig,
        ax00,
        coords,
        faces,
        anchor_distance_map,
        bg_sulc,
        "Distance from anchor (medial)",
        cmap="YlOrRd",
        threshold=1e-12,
        cbar_label="anchor distance",
    )

    render_single_medial_surface(
        fig,
        ax01,
        coords,
        faces,
        sub_pref_map,
        bg_sulc,
        "Subspace-derived preferred delay (medial)",
        cmap="YlOrRd",
        vmin=0,
        vmax=max(n_delays - 1, 1),
        threshold=1e-12,
        cbar_label="preferred delay",
    )

    render_single_medial_surface(
        fig,
        ax02,
        coords,
        faces,
        raw_pref_map,
        bg_sulc,
        "Raw-activity-derived preferred delay (medial)",
        cmap="YlOrRd",
        vmin=0,
        vmax=max(n_delays - 1, 1),
        threshold=1e-12,
        cbar_label="preferred delay",
    )

    plot_distance_vs_delay(
        ax10,
        anchor_distance,
        sub_pref_thr,
        sub_strength_thr,
        "Subspace distance vs preferred delay",
    )

    plot_mean_distance_by_preferred_delay(
        ax11,
        anchor_distance,
        sub_pref_thr,
        "Subspace mean distance by preferred delay",
        n_delays=n_delays,
    )

    plot_mean_distance_by_preferred_delay(
        ax12,
        anchor_distance,
        raw_pref_thr,
        "Raw-activity mean distance by preferred delay",
        n_delays=n_delays,
    )

    n_kept = int(np.sum(keep_mask))

    fig.suptitle(
        f"Cortical gradient analysis\n{prefix}\n"
        f"top {int(round(100 * args.keep_frac))}% by subspace delay strength "
        f"(N={n_kept})",
        fontsize=15,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    keep_tag = f"top{int(round(100 * args.keep_frac))}"
    save_path = folder / f"{prefix}_cortical_gradient_nilearn_{keep_tag}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved:\n  {save_path}\n")


if __name__ == "__main__":
    main()