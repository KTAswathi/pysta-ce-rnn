"""Task selection and deterministic train/evaluation environment factories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .envs import MazeEnv


SUPPORTED_TASKS = ("maze", "abcd_fmri")


@dataclass(frozen=True)
class ABCDFMRIFactorialCell:
    """One deterministic cell of the final scanner-style factorial design."""

    factorial_index: int
    base_configuration_index: int
    configuration: tuple[int, int, int, int]
    instruction_direction: int
    execution_relation: int
    seed: int
    environment: object


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


def _configured_fmri_bases(
    kwargs: Mapping[str, object], *, use_default: bool = False
):
    """Resolve and structurally validate a five-base scanner bank.

    An explicit ``fmri_base_configurations`` value always wins.  The labelled
    synthetic fallback is used only when requested by the final factorial
    design, so ordinary ABCD runs and the older synthetic comparison profile
    retain their previous configuration-bank behaviour.
    """
    from . import abcd_env

    raw_bases = kwargs.get("fmri_base_configurations")
    if raw_bases is None and use_default:
        bases = abcd_env.DEFAULT_FMRI_BASE_CONFIGURATIONS
    else:
        bases = tuple(abcd_env.parse_configurations(raw_bases))
    if not bases:
        return ()
    if len(bases) != 5:
        raise ValueError(
            "Final factorial fMRI evaluation requires exactly five "
            "fmri_base_configurations."
        )
    if len(set(bases)) != 5:
        raise ValueError("fmri_base_configurations must contain five distinct bases.")

    # Direct sequence reversal belongs in the crossed condition variables
    # below, so it must not also appear as another base. Other non-reversal
    # A/B/C/D assignments using the same four cells remain legitimate distinct
    # configurations; the PDFs/specification do not justify excluding them.
    base_set = set(bases)
    inverse_duplicates = [
        configuration
        for configuration in bases
        if tuple(reversed(configuration)) in base_set
    ]
    if inverse_duplicates:
        raise ValueError(
            "fmri_base_configurations must not contain direct inverse copies; "
            "represent reversal with instruction_direction/execution_relation."
        )

    min_distance = int(_value(kwargs, "min_goal_distance", 2))
    for configuration in bases:
        if not abcd_env.configuration_has_minimum_distance(
            configuration,
            min_distance,
            all_pairs=True,
        ):
            raise ValueError(
                f"fMRI base configuration {configuration} violates the "
                f"all-pairs Manhattan-distance minimum {min_distance}."
            )
    return bases


def _abcd_configuration_banks(kwargs: Mapping[str, object]):
    """Build the ABCD training bank and selected evaluation bank.

    Familiar evaluation deliberately reuses exact ordered
    training/familiarisation configurations as an ordinary performance
    monitor. Held-out evaluation is the stricter schema-generalisation test:
    it excludes all cyclic rotations in both directions, i.e. the complete
    physical-route geometry class. The distinct final scanner design is built
    by :func:`make_fmri_evaluation_schedule`.
    """
    from . import abcd_env

    explicit_train = abcd_env.parse_configurations(kwargs.get("train_configurations"))
    synthetic_objective = kwargs.get("synthetic_fmri_bank_objective")
    fmri_bases = _configured_fmri_bases(
        kwargs,
        use_default=bool(kwargs.get("run_final_fmri_evaluation", False)),
    )
    if explicit_train and synthetic_objective is not None:
        raise ValueError(
            "Choose either explicit train_configurations or a "
            "synthetic_fmri_bank_objective, not both."
        )
    if fmri_bases and synthetic_objective is not None:
        raise ValueError(
            "The synthetic inverse-paired bank is not the final factorial fMRI "
            "design; do not combine synthetic_fmri_bank_objective with final "
            "factorial base configurations."
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
    if fmri_bases and not explicit_train and num_train < len(fmri_bases):
        raise ValueError(
            "num_train_configurations must be at least five when final "
            "factorial base configurations are used."
        )

    if explicit_train:
        train_bank = explicit_train
        if fmri_bases:
            unfamiliar = set(fmri_bases).difference(train_bank)
            if unfamiliar:
                raise ValueError(
                    "Every final fMRI base configuration must occur as the same "
                    "ordered mapping in explicit train_configurations; missing "
                    f"{sorted(unfamiliar)}."
                )
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
    elif fmri_bases:
        num_fill = num_train - len(fmri_bases)
        filler = ()
        if num_fill:
            filler = abcd_env.generate_configuration_bank(
                num_configurations=num_fill,
                seed=train_seed,
                min_manhattan_distance=min_distance,
                prefer_all_pairs=True,
                exclude_configurations=fmri_bases,
                exclude_cycle_equivalents=True,
                unique_up_to_cycle=True,
            )
        # Put the five scanner bases first so the relationship is visible in
        # saved task metadata, then retain a larger deterministic training bank.
        train_bank = tuple(fmri_bases) + tuple(filler)
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


def _fmri_base_configuration_bank(kwargs: Mapping[str, object]):
    """Resolve and validate the five familiar scanner bases.

    The final design crosses these five bases with the two instruction
    directions and two execution relations. Direct reversal is therefore a
    condition manipulation, not another entry in this configuration bank. If
    no explicit bank is supplied, the documented synthetic fallback from
    :mod:`pysta.abcd_env` is used.
    """
    if kwargs.get("task", "maze") != "abcd_fmri":
        raise ValueError("Final factorial fMRI evaluation is only defined for abcd_fmri.")

    using_default = kwargs.get("fmri_base_configurations") is None
    if using_default and not bool(kwargs.get("run_final_fmri_evaluation", False)):
        raise ValueError(
            "The default fMRI base configurations are made familiar only when "
            "run_final_fmri_evaluation is enabled. Set that opt-in flag, or "
            "supply explicit fmri_base_configurations already present in the "
            "training bank."
        )
    bases = _configured_fmri_bases(kwargs, use_default=True)
    if not bases:
        raise ValueError(
            "Final factorial fMRI evaluation requires exactly five "
            "fmri_base_configurations."
        )

    # Scanner configurations were familiarised. Keep that scientific meaning
    # explicit instead of allowing a nominally fMRI-like held-out bank.
    train_bank = tuple(_abcd_configuration_banks(kwargs)["train"])
    unfamiliar = set(bases).difference(train_bank)
    if unfamiliar:
        raise ValueError(
            "Every final fMRI base configuration must occur as the same ordered "
            "mapping in the training/familiarisation bank; missing "
            f"{sorted(unfamiliar)}."
        )
    return bases


def make_fmri_evaluation_schedule(kwargs: Mapping[str, object]):
    """Build the fixed 5 x 2 x 2 scanner-style evaluation schedule.

    Ordering is base-major, then FORWARD/BACKWARD instruction direction, then
    SAME/REVERSE execution relation. Each cell is an independent batch-one
    environment, ensuring that ``BaseAgent.forward()`` resets the hidden state
    once per block without randomly resampling any factorial variable.
    """
    from .abcd_env import (
        ABCDFMRIEnv,
        BACKWARD,
        FORWARD,
        REVERSE,
        SAME,
    )

    bases = _fmri_base_configuration_bank(kwargs)
    base_seed = int(
        _value(
            kwargs,
            "fmri_evaluation_seed",
            _value(kwargs, "eval_task_seed", int(_value(kwargs, "configuration_seed", 0)) + 3),
        )
    )
    schedule = []
    factorial_index = 0
    for base_index, configuration in enumerate(bases):
        # Reuse the same task seed across all four conditions of a spatial
        # base. This makes the schedule reproducible and aligns stochastic
        # start sampling as closely as target-exclusion permits.
        cell_seed = base_seed + base_index
        for instruction_direction in (FORWARD, BACKWARD):
            for execution_relation in (SAME, REVERSE):
                environment = ABCDFMRIEnv(
                    batch_size=1,
                    seed=cell_seed,
                    configuration_bank=(configuration,),
                    num_configurations=1,
                    bank_name=(
                        f"fmri_factorial_base{base_index}_"
                        f"idir{instruction_direction}_exec{execution_relation}"
                    ),
                    instruction_directions=(instruction_direction,),
                    execution_relations=(execution_relation,),
                    num_loops=int(_value(kwargs, "n_loops", 5)),
                    instruction_repeats=int(_value(kwargs, "instruction_repeats", 2)),
                    max_navigation_steps=int(
                        _value(kwargs, "max_navigation_steps", 200)
                    ),
                    start_policy=_value(
                        kwargs, "start_position_policy", "exclude_first_goal"
                    ),
                    fixed_start=kwargs.get("start_position"),
                    min_manhattan_distance=int(
                        _value(kwargs, "min_goal_distance", 2)
                    ),
                    prefer_all_pairs=True,
                )
                schedule.append(
                    ABCDFMRIFactorialCell(
                        factorial_index=factorial_index,
                        base_configuration_index=base_index,
                        configuration=configuration,
                        instruction_direction=instruction_direction,
                        execution_relation=execution_relation,
                        seed=cell_seed,
                        environment=environment,
                    )
                )
                factorial_index += 1
    return tuple(schedule)
