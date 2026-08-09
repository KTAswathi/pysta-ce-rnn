"""Task selection and deterministic train/evaluation environment factories."""

from __future__ import annotations

from typing import Mapping

from .envs import MazeEnv


SUPPORTED_TASKS = ("maze", "abcd_fmri")


def _value(kwargs: Mapping[str, object], key: str, default):
    """Return a configured value while treating ``None`` as unspecified."""
    value = kwargs.get(key)
    return default if value is None else value


def _abcd_configuration_banks(kwargs: Mapping[str, object]):
    """Build deterministic, non-overlapping ABCD train/evaluation banks."""
    from . import abcd_env

    explicit_train = abcd_env.parse_configurations(kwargs.get("train_configurations"))
    explicit_eval = abcd_env.parse_configurations(kwargs.get("eval_configurations"))

    base_seed = int(_value(kwargs, "configuration_seed", 0))
    train_seed = int(_value(kwargs, "train_configuration_seed", base_seed))
    eval_seed = int(_value(kwargs, "eval_configuration_seed", base_seed + 1))
    min_distance = int(_value(kwargs, "min_goal_distance", 2))

    # There are 72 all-pair-separated route classes after identifying a route
    # with its direct reverse. Defaults leave genuinely unseen reversed-route
    # classes for held-out evaluation while still training on many mappings.
    num_train = int(_value(kwargs, "num_train_configurations", 48))
    num_eval = int(_value(kwargs, "num_eval_configurations", 12))
    if num_train < 1 or num_eval < 1:
        raise ValueError("ABCD train and evaluation configuration counts must be positive.")

    if explicit_train:
        train_bank = explicit_train
    else:
        train_bank = abcd_env.generate_configuration_bank(
            num_configurations=num_train,
            seed=train_seed,
            min_manhattan_distance=min_distance,
            prefer_all_pairs=True,
            unique_up_to_reversal=True,
        )

    if explicit_eval:
        eval_bank = explicit_eval
    else:
        eval_bank = abcd_env.generate_configuration_bank(
            num_configurations=num_eval,
            seed=eval_seed,
            min_manhattan_distance=min_distance,
            prefer_all_pairs=True,
            exclude_configurations=train_bank,
            exclude_reverse_equivalents=True,
            unique_up_to_reversal=True,
        )
        if len(eval_bank) != num_eval:
            raise ValueError(
                "Could not construct the requested held-out ABCD evaluation "
                "bank; reduce the train/evaluation counts or distance constraint."
            )

    overlap = set(train_bank).intersection(eval_bank)
    if overlap:
        raise ValueError(
            "ABCD training and evaluation configurations must be disjoint; "
            f"found {len(overlap)} overlapping configuration(s)."
        )

    reversed_train = {tuple(reversed(configuration)) for configuration in train_bank}
    reverse_overlap = reversed_train.intersection(eval_bank)
    if reverse_overlap:
        raise ValueError(
            "ABCD evaluation configurations must also exclude direct reversals "
            "of training configurations; found "
            f"{len(reverse_overlap)} equivalent configuration(s)."
        )

    return {"train": tuple(train_bank), "eval": tuple(eval_bank)}


def make_environment(kwargs: Mapping[str, object], split: str = "train"):
    """Construct the configured behavioural environment.

    Parameters
    ----------
    kwargs
        Parsed training/task arguments.
    split
        ``"train"`` or ``"eval"``. MazeEnv is intentionally unchanged by
        the split; ABCD uses a deterministic held-out configuration bank and
        a split-specific task RNG.
    """
    if split not in ("train", "eval"):
        raise ValueError(f"Unknown environment split: {split}")

    task = kwargs.get("task", "maze")
    if task == "maze":
        return MazeEnv(**dict(kwargs))
    if task != "abcd_fmri":
        raise ValueError(f"Unknown task: {task}")

    from .abcd_env import ABCDFMRIEnv

    banks = _abcd_configuration_banks(kwargs)
    base_seed = int(_value(kwargs, "configuration_seed", 0))
    if split == "train":
        task_seed = int(_value(kwargs, "train_task_seed", kwargs.get("seed", 0)))
    else:
        # Evaluation defaults to a seed independent of model/training seed so
        # different models see the same held-out block sequence.
        task_seed = int(_value(kwargs, "eval_task_seed", base_seed + 2))

    environment_kwargs = dict(kwargs)
    environment_kwargs.update({
        "batch_size": int(_value(kwargs, "batch_size", 8)),
        "seed": task_seed,
        "configuration_bank": banks[split],
        "num_configurations": len(banks[split]),
        "bank_name": split,
        "instruction_directions": kwargs.get("instruction_directions"),
        "execution_relations": kwargs.get("execution_relations"),
        "num_loops": int(_value(kwargs, "n_loops", 5)),
        "instruction_repeats": int(_value(kwargs, "instruction_repeats", 2)),
        "max_navigation_steps": int(_value(kwargs, "max_navigation_steps", 200)),
        "start_policy": _value(kwargs, "start_position_policy", "exclude_first_goal"),
        "fixed_start": kwargs.get("start_position"),
        "min_manhattan_distance": int(_value(kwargs, "min_goal_distance", 2)),
        "prefer_all_pairs": True,
    })
    return ABCDFMRIEnv(**environment_kwargs)
