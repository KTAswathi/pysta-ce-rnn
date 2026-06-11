import pickle
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def wrap_text(s, width=55):
    words = s.replace("_", " ").split()
    lines = []
    curr = []
    n = 0
    for w in words:
        if n + len(w) + len(curr) > width:
            lines.append(" ".join(curr))
            curr = [w]
            n = len(w)
        else:
            curr.append(w)
            n += len(w)
    if curr:
        lines.append(" ".join(curr))
    return "\n".join(lines)


def collect_model_groups(analyses_root):
    groups = defaultdict(dict)

    for p in analyses_root.glob("**/*.pickle"):
        name = p.name
        for suffix, short in KNOWN_SUFFIXES.items():
            tag = f"_{suffix}.pickle"
            if name.endswith(tag):
                prefix = name[:-len(tag)]
                groups[(p.parent, prefix)][short] = p
                break

    return groups


def find_best_matching_model_pickle(repo_root, folder_hint):
    models_root = (
        repo_root
        / "models"
        / "MazeEnv_L4_max6"
        / "landscape_changing-rew_dynamic-rew_constant-maze"
        / "allo_planrew_plan5-6-7"
    )

    folder_map = {
        "baseline": "VanillaRNN(baseline)",
        "lineembed(regu)": "lineembed(regu)",
        "lineembed(init_regu)": "lineembed(init_regu)",
        "lineembed(1.5init_regu)": "lineembed(1.5init_regu)",
        "LE(init_regu_loc-loc)": "LE(init_regu_loc-loc)",
        "LE(init_regu_loc-loc-rew)": "LE(init_regu_loc-loc-rew)",
        "LE(init_regu_loc-loc-rew-out)": "LE(init_regu_loc-loc-rew-out)",
    }

    model_folder = folder_map.get(folder_hint, None)
    if model_folder is None:
        return None

    candidates = list((models_root / model_folder).glob("**/model*.p"))
    if not candidates:
        return None

    for c in candidates:
        if c.name == "model31.p":
            return c

    return candidates[0]


def load_model_and_positions(model_pickle_path):
    if model_pickle_path is None:
        return None, None

    try:
        training_result = load_pickle(model_pickle_path)
        rnn = training_result.get("rnn", None)
        if rnn is None:
            return None, None

        nrec = getattr(rnn, "Nrec", None)

        pos = None
        for attr in [
            "positions",
            "unit_positions",
            "line_positions",
            "line_pos",
            "pos",
            "coords",
            "xpos",
            "x_pos",
        ]:
            if hasattr(rnn, attr):
                try:
                    pos = np.asarray(getattr(rnn, attr)).astype(float).squeeze()
                    break
                except Exception:
                    pass

        if pos is None and nrec is not None:
            pos = np.arange(int(nrec), dtype=float)

        return rnn, pos
    except Exception:
        return None, None


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


def clustering_score_1d(positions, labels, n_shuffle=500, seed=0):
    rng = np.random.default_rng(seed)
    positions = np.asarray(positions).astype(float)
    labels = np.asarray(labels).astype(int)

    uniq = np.unique(labels)

    def mean_group_std(lbls):
        vals = []
        for u in uniq:
            p = positions[lbls == u]
            if len(p) >= 2:
                vals.append(np.std(p))
        if len(vals) == 0:
            return np.nan
        return np.mean(vals)

    obs = mean_group_std(labels)
    if np.isnan(obs):
        return np.nan, np.nan, np.nan

    shufs = []
    for _ in range(n_shuffle):
        lbl = labels.copy()
        rng.shuffle(lbl)
        shufs.append(mean_group_std(lbl))
    shufs = np.asarray(shufs)

    return obs, np.nanmean(shufs), (np.nanmean(shufs) - obs) / (np.nanstd(shufs) + 1e-8)


def has_real_positions(pos, preferred):
    return (
        pos is not None
        and len(pos) == len(preferred)
        and not np.allclose(pos, np.arange(len(preferred)))
    )


def normalize_positions(pos):
    pos = np.asarray(pos).astype(float)
    return (pos - pos.min()) / (pos.max() - pos.min() + 1e-8)


def plot_future_decoding_line(ax, decoder_obj):
    if not isinstance(decoder_obj, dict):
        ax.text(0.5, 0.5, "decoder file not dict", ha="center", va="center")
        ax.axis("off")
        return

    scores = decoder_obj.get("nongen_scores", None)
    neural_times = decoder_obj.get("neural_times", None)
    loc_times = decoder_obj.get("loc_times", None)

    if scores is None or neural_times is None or loc_times is None:
        ax.text(0.5, 0.5, "missing decoder keys", ha="center", va="center")
        ax.axis("off")
        return

    scores = np.asarray(scores)
    loc_times = np.asarray(loc_times)

    if scores.ndim != 2:
        ax.text(0.5, 0.5, f"unexpected score shape: {scores.shape}", ha="center", va="center")
        ax.axis("off")
        return

    cmap = plt.get_cmap("tab10")

    for i in range(scores.shape[0]):
        ax.plot(loc_times, scores[i], marker="o", linewidth=2, color=cmap(i % 10))

    ax.set_xlabel("predict location at this time")
    ax.set_ylabel("accuracy")
    ax.set_title("Future decoding")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)


def plot_future_decoding_heatmap(ax, decoder_obj):
    if not isinstance(decoder_obj, dict):
        ax.text(0.5, 0.5, "decoder file not dict", ha="center", va="center")
        ax.axis("off")
        return

    scores = decoder_obj.get("nongen_scores", None)
    neural_times = decoder_obj.get("neural_times", None)
    loc_times = decoder_obj.get("loc_times", None)

    if scores is None or neural_times is None or loc_times is None:
        ax.text(0.5, 0.5, "missing decoder keys", ha="center", va="center")
        ax.axis("off")
        return

    scores = np.asarray(scores)
    neural_times = np.asarray(neural_times)
    loc_times = np.asarray(loc_times)

    im = ax.imshow(scores, aspect="auto", origin="lower", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(loc_times)))
    ax.set_xticklabels(loc_times)
    ax.set_yticks(np.arange(len(neural_times)))
    ax.set_yticklabels(neural_times)
    ax.set_xlabel("location time")
    ax.set_ylabel("neural time")
    ax.set_title("Future decoding heatmap")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def find_numeric_arrays(obj, prefix="root", out=None):
    if out is None:
        out = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            find_numeric_arrays(v, f"{prefix}.{k}", out)
        return out

    if isinstance(obj, (list, tuple)):
        if len(obj) > 0:
            for i, v in enumerate(obj):
                find_numeric_arrays(v, f"{prefix}[{i}]", out)
        return out

    try:
        arr = np.asarray(obj)
    except Exception:
        return out

    if arr.dtype == object:
        return out

    out.append((prefix, arr))
    return out


def choose_best_2d_array(obj, preferred_terms=()):
    candidates = []
    for key, arr in find_numeric_arrays(obj):
        if arr.ndim == 2 and min(arr.shape) >= 2:
            score = 0
            key_low = key.lower()
            for term in preferred_terms:
                if term.lower() in key_low:
                    score += 10
            score += min(arr.shape) * 0.1
            candidates.append((score, key, arr))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, key, arr = candidates[0]
    return key, arr


def plot_time_of_location(ax, time_obj):
    key, arr = choose_best_2d_array(time_obj, preferred_terms=("score", "acc", "time"))
    if arr is None:
        ax.text(0.5, 0.5, "no 2D array found\nfor Time-of-location", ha="center", va="center")
        ax.axis("off")
        return

    im = ax.imshow(arr, aspect="auto", origin="lower")
    ax.set_title("Time-of-location")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_preferred_delay_histogram(ax, preferred, n_delays):
    if preferred is None or n_delays == 0:
        ax.text(0.5, 0.5, "No preferred-delay data", ha="center", va="center")
        ax.axis("off")
        return

    counts = np.bincount(preferred, minlength=n_delays)
    ax.bar(np.arange(len(counts)), counts)
    ax.set_xlabel("preferred delay")
    ax.set_ylabel("number of neurons")
    ax.set_title("Overall preferred-delay histogram")


def plot_separate_population_scatter(ax, preferred, strength, pos):
    if preferred is None:
        ax.text(0.5, 0.5, "No preferred-delay data", ha="center", va="center")
        ax.axis("off")
        return

    if not has_real_positions(pos, preferred):
        ax.text(0.5, 0.5, "No spatial positions in this model", ha="center", va="center")
        ax.axis("off")
        return

    pos_norm = normalize_positions(pos)
    jitter = np.linspace(-0.08, 0.08, len(preferred))
    y = preferred + jitter

    sc = ax.scatter(
        pos_norm,
        y,
        c=preferred,
        s=20 + 80 * (strength / (strength.max() + 1e-8)),
        alpha=0.85,
    )

    obs, shuf, z = clustering_score_1d(pos_norm, preferred)

    ax.set_xlabel("unit position")
    ax.set_ylabel("preferred delay")
    ax.set_title(f"Preferred delay vs position\nclustering z = {z:.2f}")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04).set_label("preferred delay")


def plot_delay_composition_heatmap(ax, preferred, pos, n_delays, n_bins=6):
    if preferred is None or n_delays == 0:
        ax.text(0.5, 0.5, "No preferred-delay data", ha="center", va="center")
        ax.axis("off")
        return

    if not has_real_positions(pos, preferred):
        ax.text(0.5, 0.5, "No spatial positions in this model", ha="center", va="center")
        ax.axis("off")
        return

    pos_norm = normalize_positions(pos)
    edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(pos_norm, edges) - 1, 0, n_bins - 1)

    frac = np.zeros((n_delays, n_bins), dtype=float)

    for b in range(n_bins):
        mask = bin_idx == b
        if np.sum(mask) == 0:
            continue
        counts = np.bincount(preferred[mask], minlength=n_delays)
        frac[:, b] = counts / counts.sum()

    im = ax.imshow(frac, aspect="auto", origin="lower", vmin=0, vmax=1)
    ax.set_xlabel("position bin")
    ax.set_ylabel("preferred delay")
    ax.set_title("Delay composition across 6 line bins")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("fraction")


def plot_bin_histograms(axs, preferred, pos, n_delays, n_bins=6):
    axs = list(axs)

    if preferred is None or n_delays == 0:
        for ax in axs:
            ax.text(0.5, 0.5, "No preferred-delay data", ha="center", va="center")
            ax.axis("off")
        return

    if not has_real_positions(pos, preferred):
        for ax in axs:
            ax.text(0.5, 0.5, "No spatial positions\nin this model", ha="center", va="center")
            ax.axis("off")
        return

    pos_norm = normalize_positions(pos)
    edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(pos_norm, edges) - 1, 0, n_bins - 1)

    ymax = 1
    all_counts = []

    for b in range(n_bins):
        mask = bin_idx == b
        counts = np.bincount(preferred[mask], minlength=n_delays)
        all_counts.append(counts)
        ymax = max(ymax, counts.max() if len(counts) > 0 else 1)

    for b, ax in enumerate(axs):
        counts = all_counts[b]
        ax.bar(np.arange(n_delays), counts)
        ax.set_ylim(0, ymax * 1.1)
        ax.set_xticks(np.arange(n_delays))
        ax.set_xlabel("delay")
        ax.set_ylabel("count")
        left = edges[b]
        right = edges[b + 1]
        ax.set_title(f"Bin {b+1}: [{left:.2f}, {right:.2f})")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    analyses_root = repo_root / "data" / "rnn_analyses"

    groups = collect_model_groups(analyses_root)

    if not groups:
        print(f"No analysis pickle groups found under {analyses_root}")
        return

    print(f"Found {len(groups)} model analysis groups.\n")

    for (folder, prefix), files_dict in sorted(groups.items()):
        print(f"Building summary for: {folder.name} / {prefix}")

        decoder_obj = load_pickle(files_dict["decoder"]) if "decoder" in files_dict else {}
        time_obj = load_pickle(files_dict["timeofloc"]) if "timeofloc" in files_dict else {}
        subspace_obj = load_pickle(files_dict["subspaces"]) if "subspaces" in files_dict else {}

        model_pickle_path = find_best_matching_model_pickle(repo_root, folder.name)
        rnn, pos = load_model_and_positions(model_pickle_path)
        n_units = getattr(rnn, "Nrec", None) if rnn is not None else None

        preferred, strength, delay_score = extract_delay_preferences(subspace_obj, n_units)
        n_delays = 0 if delay_score is None else delay_score.shape[0]

        fig, axs = plt.subplots(4, 3, figsize=(18, 16))

        plot_future_decoding_line(axs[0, 0], decoder_obj)
        plot_future_decoding_heatmap(axs[0, 1], decoder_obj)
        plot_time_of_location(axs[0, 2], time_obj)

        plot_preferred_delay_histogram(axs[1, 0], preferred, n_delays)
        plot_separate_population_scatter(axs[1, 1], preferred, strength, pos)
        plot_delay_composition_heatmap(axs[1, 2], preferred, pos, n_delays, n_bins=6)

        # 6 bin histograms
        plot_bin_histograms(axs[2:, :].ravel(), preferred, pos, n_delays, n_bins=6)

        title = f"{folder.name}\n{wrap_text(prefix, width=90)}"
        fig.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        save_path = folder / f"{prefix}_summary.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {save_path}\n")


if __name__ == "__main__":
    main()