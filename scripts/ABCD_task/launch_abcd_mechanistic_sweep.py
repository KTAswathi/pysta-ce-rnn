"""Build or launch managed training commands from the frozen sweep spec.

The specification is JSON-compatible YAML so this thin launcher needs no
optional YAML dependency.  Scientific values live only in that file: this
module validates structure, selects declared cells/seeds, and serializes them
to the existing ``pysta.train_rnn`` CLI.  The default is a read-only dry run;
``--execute`` is required to start training.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = REPO_ROOT / "configs" / "abcd_fmri_mechanistic_sweep.yaml"
MECHANISM_KEYS = {
    "local_fraction",
    "dist_reg",
    "line_decay",
    "line_init_scale",
    "use_local_init",
}


def load_spec(path: Path | str) -> dict[str, Any]:
    """Load and structurally validate one frozen sweep declaration."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    if spec.get("schema") != "abcd_fmri_mechanistic_sweep/v1":
        raise ValueError("Unsupported ABCD mechanistic sweep schema.")
    if spec.get("status") != "frozen_pre_sweep":
        raise ValueError("Sweep specification is not marked frozen_pre_sweep.")
    declared = set(spec.get("mechanism_parameters", ()))
    if declared != MECHANISM_KEYS:
        raise ValueError("The declared mechanism parameter set is invalid.")
    seeds = spec.get("model_seeds")
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int for seed in seeds):
        raise ValueError("model_seeds must be a non-empty list of integers.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("model_seeds must be unique.")
    conditions = spec.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("conditions must be a non-empty list.")
    tags: list[str] = []
    for condition in conditions:
        if not isinstance(condition, Mapping):
            raise ValueError("Each condition must be a mapping.")
        tag = condition.get("tag")
        if not isinstance(tag, str) or not tag:
            raise ValueError("Every condition needs a non-empty tag.")
        if set(condition).difference({"tag"} | MECHANISM_KEYS):
            raise ValueError(f"Condition {tag!r} contains undeclared parameters.")
        if set(condition).difference({"tag"}) != MECHANISM_KEYS:
            raise ValueError(f"Condition {tag!r} is missing a mechanism parameter.")
        tags.append(tag)
    if len(set(tags)) != len(tags):
        raise ValueError("Condition tags must be unique.")
    fixed = spec.get("fixed_cli_arguments")
    if not isinstance(fixed, Mapping) or fixed.get("num_epochs") != spec.get(
        "training_updates"
    ):
        raise ValueError("training_updates must equal fixed num_epochs.")
    if not isinstance(spec.get("seed_coupled_cli_arguments"), Mapping):
        raise ValueError("seed_coupled_cli_arguments must be a mapping.")
    if not spec.get("fixed_design", {}).get("analysis_pipeline_frozen", False):
        raise ValueError("The analysis pipeline must be explicitly frozen.")
    return spec


def _append_cli_argument(command: list[str], key: str, value: Any) -> None:
    option = f"--{key}"
    command.append(option)
    if isinstance(value, list):
        command.extend(str(item) for item in value)
    elif isinstance(value, bool):
        command.append("1" if value else "0")
    else:
        command.append(str(value))


def build_training_command(
    spec: Mapping[str, Any],
    condition: Mapping[str, Any],
    seed: int,
    *,
    python_executable: str = sys.executable,
) -> list[str]:
    """Serialize one declared condition/seed without deriving science values."""

    if seed not in spec["model_seeds"]:
        raise ValueError(f"Seed {seed} is not declared by the sweep specification.")
    command = [python_executable, "-m", "pysta.train_rnn"]
    for key, value in spec["fixed_cli_arguments"].items():
        _append_cli_argument(command, key, value)
    for key, template in spec["seed_coupled_cli_arguments"].items():
        value = template.format(seed=seed) if isinstance(template, str) else template
        _append_cli_argument(command, key, value)
    for key in spec["mechanism_parameters"]:
        _append_cli_argument(command, key, condition[key])
    run_name = spec["run_name_template"].format(tag=condition["tag"], seed=seed)
    if len(run_name) > 48:
        raise ValueError(f"Managed run name exceeds 48 characters: {run_name!r}")
    _append_cli_argument(command, "run_name", run_name)
    return command


def selected_commands(
    spec: Mapping[str, Any],
    *,
    tags: Sequence[str] | None = None,
    excluded_tags: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    python_executable: str = sys.executable,
) -> list[list[str]]:
    """Return commands for the requested declared subset in spec order."""

    requested_tags = set(tags or ())
    rejected_tags = set(excluded_tags or ())
    requested_seeds = set(seeds or ())
    known_tags = {condition["tag"] for condition in spec["conditions"]}
    unknown_tags = requested_tags.union(rejected_tags).difference(known_tags)
    unknown_seeds = requested_seeds.difference(spec["model_seeds"])
    if unknown_tags:
        raise ValueError(f"Unknown condition tag(s): {sorted(unknown_tags)}")
    if unknown_seeds:
        raise ValueError(f"Undeclared model seed(s): {sorted(unknown_seeds)}")
    conditions = [
        condition
        for condition in spec["conditions"]
        if (not requested_tags or condition["tag"] in requested_tags)
        and condition["tag"] not in rejected_tags
    ]
    selected_seeds = [
        seed
        for seed in spec["model_seeds"]
        if not requested_seeds or seed in requested_seeds
    ]
    return [
        build_training_command(
            spec,
            condition,
            seed,
            python_executable=python_executable,
        )
        for condition in conditions
        for seed in selected_seeds
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--condition", action="append", dest="conditions")
    parser.add_argument(
        "--exclude-condition",
        action="append",
        dest="excluded_conditions",
        help=(
            "Skip a declared cell, for example baseline after it has already "
            "passed the end-to-end smoke test."
        ),
    )
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Start the selected managed training runs (default: print only).",
    )
    args = parser.parse_args(argv)

    spec = load_spec(args.spec)
    commands = selected_commands(
        spec,
        tags=args.conditions,
        excluded_tags=args.excluded_conditions,
        seeds=args.seeds,
    )
    for command in commands:
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
