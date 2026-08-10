"""Task selection and deterministic train/evaluation environment factories."""

from __future__ import annotations

from typing import Mapping

from .envs import MazeEnv


SUPPORTED_TASKS = ("maze", "abcd_fmri")


def _value(kwargs: Mapping[str, object], key: str, default):
    """Return a configured value while treating ``None`` as unspecified."""
    value = kwargs.get(key)
    return default if value is None else value


def _abcd_evaluation_mode(kwargs: Mapping[str, object]) -> str:
    """Return and validate the requested ABCD evaluation configuration set."""
    mode = str(_value(kwargs, "evaluation_mode", "familiar")).lower()
    if mode not in ("familiar", "heldout"):
        raise ValueError(
            "ABCD evaluation_mode must be 'familiar' or 'heldout'; "
            f"got {mode!r}."
        )
    return mode


def _abcd_configuration_banks(kwargs: Mapping[str, object]):
    """Build the ABCD training bank and selected evaluation bank.

    Familiar evaluation deliberately reuses exact ordered
    training/familiarisation configurations, matching the primary
    scanner-style evaluation. Held-out evaluation is the stricter
    schema-generalisation test: it excludes all cyclic rotations in both
    directions, i.e. the complete physical-route geometry class.
    """
    from . import abcd_env

    explicit_train = abcd_env.parse_configurations(kwargs.get("train_configurations"))
    synthetic_objective = kwargs.get("synthetic_fmri_bank_objective")
    if explicit_train and synthetic_objective is not None:
        raise ValueError(
            "Choose either explicit train_configurations or a "
            "synthetic_fmri_bank_objective, not both."
        )

    base_seed = int(_value(kwargs, "configuration_seed", 0))
    train_seed = int(_value(kwargs, "train_configuration_seed", base_seed))
    min_distance = int(_value(kwargs, "min_goal_distance", 2))
    evaluation_mode = _abcd_evaluation_mode(kwargs)

    # With all-pairs distance >= 2 there are only 18 physical-cycle classes on
    # this grid. Reserve six by default so ``--evaluation_mode heldout`` works
    # out of the box and means genuinely unseen route geometry.
    num_train = int(_value(kwargs, "num_train_configurations", 12))
    if num_train < 1:
        raise ValueError("ABCD training configuration count must be positive.")

    if explicit_train:
        train_bank = explicit_train
    elif synthetic_objective is not None:
        if kwargs.get("num_train_configurations") not in (None, 10):
            raise ValueError(
                "The synthetic fMRI comparison profile always contains exactly "
                "10 ordered configurations (five inverse pairs)."
            )
        train_bank = abcd_env.generate_synthetic_fmri_configuration_bank(
            seed=train_seed,
            objective=str(synthetic_objective),
        )
    else:
        train_bank = abcd_env.generate_configuration_bank(
            num_configurations=num_train,
            seed=train_seed,
            min_manhattan_distance=min_distance,
            prefer_all_pairs=True,
            unique_up_to_cycle=True,
        )

    # The familiar condition is not a second generated bank: it is the same
    # set of configurations on which the model was trained/familiarised. The
    # environment itself still has an independent evaluation RNG (below).
    if evaluation_mode == "familiar":
        train_bank = tuple(train_bank)
        explicit_familiar = abcd_env.parse_configurations(
            kwargs.get("familiar_configurations")
        )
        if explicit_familiar:
            unfamiliar = set(explicit_familiar).difference(train_bank)
            if unfamiliar:
                raise ValueError(
                    "Every familiar evaluation configuration must occur as the "
                    "same ordered mapping in the training/familiarisation bank; "
                    f"missing {sorted(unfamiliar)}."
                )
            familiar_bank = tuple(explicit_familiar)
        else:
            familiar_bank = train_bank
        return {"train": train_bank, "eval": familiar_bank}

    explicit_eval = abcd_env.parse_configurations(kwargs.get("eval_configurations"))
    eval_seed = int(_value(kwargs, "eval_configuration_seed", base_seed + 1))
    num_eval = int(_value(kwargs, "num_eval_configurations", 6))
    if num_eval < 1:
        raise ValueError("ABCD held-out evaluation configuration count must be positive.")

    if explicit_eval:
        eval_bank = explicit_eval
    else:
        eval_bank = abcd_env.generate_configuration_bank(
            num_configurations=num_eval,
            seed=eval_seed,
            min_manhattan_distance=min_distance,
            prefer_all_pairs=True,
            exclude_configurations=train_bank,
            exclude_cycle_equivalents=True,
            unique_up_to_cycle=True,
        )
        if len(eval_bank) != num_eval:
            raise ValueError(
                "Could not construct the requested held-out ABCD evaluation "
                "bank; reduce the train/evaluation counts or distance constraint."
            )

    train_cycle_keys = {
        abcd_env.canonical_configuration_cycle(configuration)
        for configuration in train_bank
    }
    eval_cycle_keys = {
        abcd_env.canonical_configuration_cycle(configuration)
        for configuration in eval_bank
    }
    cycle_overlap = train_cycle_keys.intersection(eval_cycle_keys)
    if cycle_overlap:
        raise ValueError(
            "ABCD held-out configurations must exclude every cyclic rotation "
            "and reversed rotation of each training physical route; found "
            f"{len(cycle_overlap)} overlapping cycle class(es)."
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
        the split. ABCD evaluation uses either the familiar training bank or
        a deterministic held-out bank, plus a split-specific task RNG.
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
    evaluation_mode = _abcd_evaluation_mode(kwargs)
    base_seed = int(_value(kwargs, "configuration_seed", 0))
    if split == "train":
        task_seed = int(_value(kwargs, "train_task_seed", kwargs.get("seed", 0)))
    else:
        # Evaluation defaults to a seed independent of model/training seed so
        # different models see the same evaluation block sequence, whether
        # the selected configurations are familiar or held out.
        task_seed = int(_value(kwargs, "eval_task_seed", base_seed + 2))

    environment_kwargs = dict(kwargs)
    environment_kwargs.update({
        "batch_size": int(_value(kwargs, "batch_size", 8)),
        "seed": task_seed,
        "configuration_bank": banks[split],
        "num_configurations": len(banks[split]),
        "bank_name": "train" if split == "train" else f"eval_{evaluation_mode}",
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
