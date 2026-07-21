#!/usr/bin/env python3
"""
Analyse whether future locations become progressively less decodable from the
RNN's late-planning activity.

This script does NOT refit decoders. It reads the existing
`*_decoder_generalization_performance.pickle` produced by:

    python scripts/analyse_rnn.py <MODEL> decoding

The key quantity is `nongen_scores[neural_time, loc_time]`: cross-location
held-out decoding accuracy when neural activity and the decoded location are
both evaluated at the indicated times.

For the standard analysis:
  * neural time -2 / -1 = late planning activity
  * location time 0     = current/start location at execution onset
  * location time 1     = location after the first action (+1 future state)
  * location time 2     = location after two actions (+2), etc.

Example
-------
MODEL="MazeEnv_L4_max6/landscape_changing-rew_dynamic-rew_constant-maze/allo_planrew_plan5-6-7/CorticallyEmbeddedRNN/iter10_tau5.0_opt/N480_linout_cortical_mpfc_projected_mask_linear0p1_eseed42_ld0.12_dr1e-07_lfrac0.16666666666666666_loc1_rew0_wall0_roglobal_anchorautoanchor/model0"

PYTHONPATH="$PWD" python scripts/analyse_future_state_decodability.py "$MODEL"
"""

from __future__ import annotations

import argparse
import csv
import pickle
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


DECODER_SUFFIX = "_decoder_generalization_performance.pickle"
TRIAL_SUFFIX = "_trial_data.pickle"


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def normalise_query(query: str) -> str:
    query = str(query).replace("\\", "/").strip()
    if query.endswith(".p"):
        query = query[:-2]
    return query.strip("/")


def candidate_key(path: Path, analyses_root: Path) -> str:
    rel = path.relative_to(analyses_root).as_posix()
    return rel[: -len(DECODER_SUFFIX)]


def find_decoder_pickle(
    repo_root: Path,
    query: str,
    analyses_root: Path | None = None,
) -> Path:
    """Find the best matching decoder pickle for a model query."""
    query_path = Path(query)
    if query_path.exists() and query_path.is_file():
        return query_path.resolve()

    if analyses_root is None:
        analyses_root = repo_root / "data" / "rnn_analyses" / "MazeEnv_L4_max6"

    if not analyses_root.exists():
        raise FileNotFoundError(f"Analysis root does not exist: {analyses_root}")

    candidates = sorted(analyses_root.glob(f"**/*{DECODER_SUFFIX}"))
    if not candidates:
        raise FileNotFoundError(
            f"No *{DECODER_SUFFIX} files found under {analyses_root}"
        )

    query_norm = normalise_query(query)

    exact = [p for p in candidates if candidate_key(p, analyses_root) == query_norm]
    if exact:
        return exact[0]

    scored: list[tuple[float, Path]] = []
    for path in candidates:
        key = candidate_key(path, analyses_root)
        filename_prefix = path.name[: -len(DECODER_SUFFIX)]

        score = SequenceMatcher(None, query_norm, key).ratio()
        if query_norm in key or key in query_norm:
            score += 2.0
        if query_norm == filename_prefix:
            score += 3.0
        if query_norm.endswith("/" + filename_prefix):
            score += 1.0
        scored.append((score, path))

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_path = scored[0]

    if best_score < 0.25:
        raise ValueError(
            f"Could not confidently match query {query!r} to a decoder pickle."
        )

    return best_path


def extract_decoder_matrix(result: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {"nongen_scores", "neural_times", "loc_times"}
    missing = required.difference(result)
    if missing:
        raise KeyError(
            f"Decoder pickle is missing keys {sorted(missing)}. "
            f"Available keys: {sorted(result.keys())}"
        )

    accuracy = np.asarray(result["nongen_scores"], dtype=float)
    neural_times = np.asarray(result["neural_times"], dtype=int).reshape(-1)
    loc_times = np.asarray(result["loc_times"], dtype=int).reshape(-1)

    expected = (len(neural_times), len(loc_times))
    if accuracy.shape == expected:
        pass
    elif accuracy.T.shape == expected:
        accuracy = accuracy.T
    else:
        raise ValueError(
            "Could not align nongen_scores with neural_times and loc_times: "
            f"nongen_scores.shape={accuracy.shape}, expected={expected}."
        )

    return accuracy, neural_times, loc_times


def infer_nominal_chance(decoder_pickle: Path) -> tuple[float | None, int | None]:
    """Infer 1 / number_of_locations from the sibling trial-data pickle."""
    prefix = decoder_pickle.name[: -len(DECODER_SUFFIX)]
    trial_path = decoder_pickle.parent / f"{prefix}{TRIAL_SUFFIX}"
    if not trial_path.exists():
        return None, None

    try:
        trial = load_pickle(trial_path)
        locs = np.asarray(trial["locs"], dtype=float)
        unique_locs = np.unique(locs[np.isfinite(locs)]).astype(int)
        n_locs = len(unique_locs)
        if n_locs > 0:
            return 1.0 / n_locs, n_locs
    except (KeyError, TypeError, ValueError, pickle.UnpicklingError):
        pass

    return None, None


def time_label(t: int) -> str:
    if t == 0:
        return "current\n(0)"
    if t > 0:
        return f"+{t}"
    return str(t)


def select_planning_rows(
    accuracy: np.ndarray,
    neural_times: np.ndarray,
    requested_times: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    selected_indices: list[int] = []
    selected_times: list[int] = []

    for requested in requested_times:
        matches = np.where(neural_times == requested)[0]
        if len(matches):
            selected_indices.append(int(matches[0]))
            selected_times.append(requested)
        else:
            print(f"Warning: neural time {requested} is not present; skipping it.")

    if not selected_indices:
        raise ValueError(
            f"None of the requested planning times {requested_times} occur in "
            f"neural_times={neural_times.tolist()}"
        )

    return accuracy[selected_indices, :], np.asarray(selected_times, dtype=int)


def write_csv(
    save_path: Path,
    planning_scores: np.ndarray,
    planning_times: np.ndarray,
    loc_times: np.ndarray,
) -> None:
    with save_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "planning_neural_time",
                "decoded_location_time",
                "future_offset",
                "decoding_accuracy",
            ],
        )
        writer.writeheader()
        for row_idx, neural_time in enumerate(planning_times):
            for col_idx, loc_time in enumerate(loc_times):
                writer.writerow(
                    {
                        "planning_neural_time": int(neural_time),
                        "decoded_location_time": int(loc_time),
                        "future_offset": int(loc_time),
                        "decoding_accuracy": float(planning_scores[row_idx, col_idx]),
                    }
                )


def make_heatmap(
    save_path: Path,
    accuracy: np.ndarray,
    neural_times: np.ndarray,
    loc_times: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    image = ax.imshow(accuracy, aspect="auto", origin="upper")
    ax.set_xticks(np.arange(len(loc_times)))
    ax.set_xticklabels([time_label(int(t)) for t in loc_times])
    ax.set_yticks(np.arange(len(neural_times)))
    ax.set_yticklabels([str(int(t)) for t in neural_times])
    ax.set_xlabel("Decoded location time relative to execution onset")
    ax.set_ylabel("Neural activity time")
    ax.set_title("Cross-location-generalised location decoding")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Decoding accuracy")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_planning_curve(
    save_path: Path,
    planning_scores: np.ndarray,
    planning_times: np.ndarray,
    loc_times: np.ndarray,
    chance: float | None,
    next_loc_time: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for idx, neural_time in enumerate(planning_times):
        ax.plot(
            loc_times,
            planning_scores[idx],
            marker="o",
            linewidth=1.8,
            label=f"planning activity t={int(neural_time)}",
        )

    if len(planning_times) > 1:
        mean_curve = np.nanmean(planning_scores, axis=0)
        ax.plot(
            loc_times,
            mean_curve,
            marker="o",
            linewidth=3.0,
            label="mean across selected planning times",
        )

    if chance is not None:
        ax.axhline(chance, linestyle="--", linewidth=1.5, label="nominal chance")

    if next_loc_time in loc_times:
        ax.axvline(next_loc_time, linestyle=":", linewidth=1.5)

    ax.set_xticks(loc_times)
    ax.set_xticklabels([time_label(int(t)) for t in loc_times])
    ax.set_xlabel("Decoded state: current, then future locations")
    ax.set_ylabel("Cross-location-generalised decoding accuracy")
    ax.set_title("Future-location decodability from late-planning activity")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_next_relative_curve(
    save_path: Path,
    planning_scores: np.ndarray,
    planning_times: np.ndarray,
    loc_times: np.ndarray,
    next_loc_time: int,
) -> None:
    matches = np.where(loc_times == next_loc_time)[0]
    if not len(matches):
        return

    next_idx = int(matches[0])
    future_mask = loc_times >= next_loc_time
    future_times = loc_times[future_mask]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for row_idx, neural_time in enumerate(planning_times):
        delta = (
            planning_scores[row_idx, future_mask]
            - planning_scores[row_idx, next_idx]
        )
        ax.plot(
            future_times,
            delta,
            marker="o",
            linewidth=1.8,
            label=f"planning activity t={int(neural_time)}",
        )

    if len(planning_times) > 1:
        mean_scores = np.nanmean(planning_scores, axis=0)
        mean_delta = mean_scores[future_mask] - mean_scores[next_idx]
        ax.plot(
            future_times,
            mean_delta,
            marker="o",
            linewidth=3.0,
            label="mean across selected planning times",
        )

    ax.axhline(0.0, linestyle="--", linewidth=1.5)
    ax.set_xticks(future_times)
    ax.set_xticklabels([f"+{int(t)}" for t in future_times])
    ax.set_xlabel("Future location")
    ax.set_ylabel(f"Accuracy difference from +{next_loc_time}")
    ax.set_title("Decodability relative to the first future location")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_summary(
    decoder_pickle: Path,
    planning_scores: np.ndarray,
    planning_times: np.ndarray,
    loc_times: np.ndarray,
    next_loc_time: int,
    chance: float | None,
    n_locs: int | None,
) -> str:
    lines = [
        "Future-state decodability analysis",
        "==================================",
        f"Decoder pickle: {decoder_pickle}",
        f"Planning neural times: {planning_times.tolist()}",
        f"Decoded location times: {loc_times.tolist()}",
        "",
        "Interpretation of location times:",
        "  0 = current/start location at execution onset",
        "  1 = location after the first action (+1 future state)",
        "  2 = location after two actions (+2 future state), etc.",
        "",
    ]

    if chance is not None and n_locs is not None:
        lines.append(f"Nominal chance: 1/{n_locs} = {chance:.6f}")
        lines.append("")

    next_matches = np.where(loc_times == next_loc_time)[0]
    future_mask = loc_times >= next_loc_time
    later_mask = loc_times > next_loc_time

    if not len(next_matches):
        lines.append(
            f"Requested next-location time {next_loc_time} is absent; "
            "next-versus-later summaries were not computed."
        )
        return "\n".join(lines) + "\n"

    next_idx = int(next_matches[0])
    future_times = loc_times[future_mask]

    for row_idx, neural_time in enumerate(planning_times):
        curve = planning_scores[row_idx]
        next_acc = curve[next_idx]
        later_values = curve[later_mask]
        slope = np.polyfit(future_times, curve[future_mask], 1)[0]
        rho, rho_p = spearmanr(future_times, curve[future_mask])
        num_later_below = int(np.sum(later_values < next_acc))

        lines.extend(
            [
                f"Planning time {int(neural_time)}:",
                f"  +{next_loc_time} accuracy: {next_acc:.6f}",
                f"  mean later (+>{next_loc_time}) accuracy: "
                f"{np.nanmean(later_values):.6f}",
                f"  mean later minus +{next_loc_time}: "
                f"{np.nanmean(later_values) - next_acc:.6f}",
                f"  linear slope across +{int(future_times[0])}..+{int(future_times[-1])}: "
                f"{slope:.6f} accuracy units/state",
                f"  Spearman rho across future states: {rho:.6f} "
                f"(descriptive p={rho_p:.6g})",
                f"  later states below +{next_loc_time}: "
                f"{num_later_below}/{len(later_values)}",
                "",
            ]
        )

    if len(planning_times) > 1:
        mean_curve = np.nanmean(planning_scores, axis=0)
        next_acc = mean_curve[next_idx]
        later_values = mean_curve[later_mask]
        slope = np.polyfit(future_times, mean_curve[future_mask], 1)[0]
        rho, rho_p = spearmanr(future_times, mean_curve[future_mask])
        lines.extend(
            [
                "Mean across selected planning times:",
                f"  +{next_loc_time} accuracy: {next_acc:.6f}",
                f"  mean later (+>{next_loc_time}) accuracy: "
                f"{np.nanmean(later_values):.6f}",
                f"  mean later minus +{next_loc_time}: "
                f"{np.nanmean(later_values) - next_acc:.6f}",
                f"  linear slope: {slope:.6f} accuracy units/state",
                f"  Spearman rho: {rho:.6f} (descriptive p={rho_p:.6g})",
                "",
            ]
        )

    lines.extend(
        [
            "Important limitation:",
            "  This is a descriptive analysis of one trained model. The pickle",
            "  contains decoder performance already averaged across held-out",
            "  current-location folds. Treat slopes/p-values as descriptive,",
            "  not as a population-level inferential test. For inference, repeat",
            "  across independently trained model seeds and use model as the",
            "  statistical unit.",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read decoder_generalization_performance pickle and test "
            "whether later future locations are less decodable from planning activity."
        )
    )
    parser.add_argument(
        "query",
        help=(
            "Model-relative path, model name, or direct path to a "
            "*_decoder_generalization_performance.pickle file."
        ),
    )
    parser.add_argument(
        "--planning-times",
        type=int,
        nargs="+",
        default=[-2, -1],
        help="Planning neural times to extract (default: -2 -1).",
    )
    parser.add_argument(
        "--next-loc-time",
        type=int,
        default=1,
        help=(
            "Location time treated as the first future state. In stored "
            "data, 0 is current location and 1 is after the first action."
        ),
    )
    parser.add_argument(
        "--analyses-root",
        type=Path,
        default=None,
        help=(
            "Optional analysis root. Default: "
            "<repo>/data/rnn_analyses/MazeEnv_L4_max6"
        ),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    analyses_root = args.analyses_root
    if analyses_root is not None and not analyses_root.is_absolute():
        analyses_root = repo_root / analyses_root

    decoder_pickle = find_decoder_pickle(
        repo_root=repo_root,
        query=args.query,
        analyses_root=analyses_root,
    )
    print(f"Using decoder pickle:\n  {decoder_pickle}\n")

    result = load_pickle(decoder_pickle)
    if not isinstance(result, dict):
        raise TypeError("Decoder pickle must contain a dictionary.")

    accuracy, neural_times, loc_times = extract_decoder_matrix(result)
    planning_scores, planning_times = select_planning_rows(
        accuracy,
        neural_times,
        args.planning_times,
    )

    chance, n_locs = infer_nominal_chance(decoder_pickle)

    prefix = decoder_pickle.name[: -len(DECODER_SUFFIX)]
    output_dir = decoder_pickle.parent / "future_state_decodability"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{prefix}_future_state_decodability.csv"
    heatmap_path = output_dir / f"{prefix}_decoder_heatmap.png"
    curve_path = output_dir / f"{prefix}_planning_future_decodability.png"
    relative_path = output_dir / f"{prefix}_future_decodability_relative_to_plus1.png"
    summary_path = output_dir / f"{prefix}_future_state_decodability_summary.txt"

    write_csv(csv_path, planning_scores, planning_times, loc_times)
    make_heatmap(heatmap_path, accuracy, neural_times, loc_times)
    make_planning_curve(
        curve_path,
        planning_scores,
        planning_times,
        loc_times,
        chance,
        args.next_loc_time,
    )
    make_next_relative_curve(
        relative_path,
        planning_scores,
        planning_times,
        loc_times,
        args.next_loc_time,
    )

    summary = build_summary(
        decoder_pickle,
        planning_scores,
        planning_times,
        loc_times,
        args.next_loc_time,
        chance,
        n_locs,
    )
    summary_path.write_text(summary)

    print(summary)
    print("Saved:")
    for path in [csv_path, heatmap_path, curve_path, relative_path, summary_path]:
        if path.exists():
            print(f"  {path}")


if __name__ == "__main__":
    main()
