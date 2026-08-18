"""Focused tests for the explicit 5 x 2 x 2 scanner evaluation design."""

import sys

import numpy as np
import pytest
import torch

import pysta
from pysta.abcd_env import BACKWARD, FORWARD, REVERSE, SAME
from pysta.abcd_env import (
    DEFAULT_FMRI_BASE_CONFIGURATIONS,
    configuration_bank_statistics,
    manhattan_distance,
)


# Five deterministic examples used only by these tests. They are not claimed
# to be Svenja's coordinates; production evaluation requires explicit inputs.
BASES = (
    (7, 5, 3, 1),
    (2, 8, 6, 0),
    (6, 8, 4, 0),
    (2, 8, 6, 4),
    (2, 6, 0, 4),
)


def _kwargs(**overrides):
    values = dict(
        task="abcd_fmri",
        batch_size=2,
        seed=11,
        configuration_seed=17,
        train_configuration_seed=19,
        fmri_base_configurations=BASES,
        fmri_evaluation_seed=23,
        num_train_configurations=12,
        n_loops=5,
        instruction_repeats=2,
        max_navigation_steps=20,
        start_position_policy="exclude_first_goal",
        min_goal_distance=2,
    )
    values.update(overrides)
    return values


def test_scanner_cli_is_opt_in_and_distinct_from_familiar_monitor(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train"])
    args = pysta.argparser.parse_args(task="abcd_fmri")
    assert args["evaluation_mode"] == "familiar"
    assert args["fmri_base_configurations"] is None
    assert args["run_final_fmri_evaluation"] is False


def test_documented_default_bases_are_balanced_and_used_for_final_design():
    report = configuration_bank_statistics(DEFAULT_FMRI_BASE_CONFIGURATIONS)
    assert DEFAULT_FMRI_BASE_CONFIGURATIONS == (
        (0, 2, 4, 8),
        (2, 6, 8, 0),
        (5, 1, 3, 7),
        (6, 8, 0, 4),
        (7, 3, 5, 1),
    )
    assert report["num_unique_ordered_configurations"] == 5
    assert report["num_physical_cycle_classes"] == 5
    assert report["location_counts"] == (3, 2, 2, 2, 2, 2, 2, 2, 3)
    assert report["location_count_range"] == 1
    assert report["all_pairs_minimum_distance_two"] is True
    assert report["mean_circular_manhattan_distance"] == pytest.approx(2.4)

    # Each abstract label is spread across five locations with its centroid at
    # the grid centre; each circular transition position has the same total
    # path length across bases.
    for abstract_index in range(4):
        locations = [base[abstract_index] for base in DEFAULT_FMRI_BASE_CONFIGURATIONS]
        assert len(set(locations)) == 5
        assert np.mean([location // 3 for location in locations]) == 1
        assert np.mean([location % 3 for location in locations]) == 1
    assert [
        sum(
            manhattan_distance(base[index], base[(index + 1) % 4])
            for base in DEFAULT_FMRI_BASE_CONFIGURATIONS
        )
        for index in range(4)
    ] == [12, 12, 12, 12]

    kwargs = _kwargs(run_final_fmri_evaluation=True)
    kwargs.pop("fmri_base_configurations")
    banks = pysta.tasks._abcd_configuration_banks(kwargs)
    assert banks["train"][:5] == DEFAULT_FMRI_BASE_CONFIGURATIONS
    assert len(banks["train"]) == 12
    assert banks["eval"] == banks["train"]

    schedule = pysta.tasks.make_fmri_evaluation_schedule(kwargs)
    assert len(schedule) == 20
    assert tuple(cell.configuration for cell in schedule[::4]) == (
        DEFAULT_FMRI_BASE_CONFIGURATIONS
    )

    # Without the opt-in, an ordinary run uses its previous generated bank;
    # do not silently mislabel the fallback configurations as familiar.
    kwargs["run_final_fmri_evaluation"] = False
    with pytest.raises(ValueError, match="made familiar only when"):
        pysta.tasks.make_fmri_evaluation_schedule(kwargs)


def test_default_fmri_bases_do_not_change_ordinary_or_synthetic_bank_modes():
    kwargs = _kwargs()
    kwargs.pop("fmri_base_configurations")
    ordinary = pysta.tasks._abcd_configuration_banks(kwargs)
    assert pysta.tasks._configured_fmri_bases(kwargs) == ()
    assert ordinary["train"] != DEFAULT_FMRI_BASE_CONFIGURATIONS

    kwargs.update(
        synthetic_fmri_bank_objective="balance_first",
        num_train_configurations=10,
    )
    synthetic = pysta.tasks._abcd_configuration_banks(kwargs)
    assert len(synthetic["train"]) == 10


def test_five_bases_are_familiar_and_not_inverse_paired():
    kwargs = _kwargs()
    banks = pysta.tasks._abcd_configuration_banks(kwargs)
    assert banks["train"][:5] == BASES
    assert len(banks["train"]) == 12
    assert banks["eval"] == banks["train"]  # ordinary familiar monitor unchanged

    with pytest.raises(ValueError, match="direct inverse copies"):
        pysta.tasks.make_fmri_evaluation_schedule(
            _kwargs(
                fmri_base_configurations=(
                    BASES[0],
                    tuple(reversed(BASES[0])),
                    *BASES[2:],
                )
            )
        )

    # A non-reversal A/B/C/D reassignment over the same physical cells is a
    # distinct configuration and is not excluded by the experimental facts.
    relabelled = (7, 3, 5, 1)
    relabelled_schedule = pysta.tasks.make_fmri_evaluation_schedule(
        _kwargs(
            fmri_base_configurations=(BASES[0], relabelled, *BASES[2:]),
            num_train_configurations=5,
        )
    )
    assert relabelled_schedule[4].configuration == relabelled

    with pytest.raises(ValueError, match="synthetic inverse-paired"):
        pysta.tasks.make_fmri_evaluation_schedule(
            _kwargs(synthetic_fmri_bank_objective="balance_first")
        )

    with pytest.raises(ValueError, match="explicit train_configurations"):
        pysta.tasks.make_fmri_evaluation_schedule(
            _kwargs(train_configurations=BASES[:-1])
        )


def test_schedule_is_exact_deterministic_base_major_factorial():
    first = pysta.tasks.make_fmri_evaluation_schedule(_kwargs())
    second = pysta.tasks.make_fmri_evaluation_schedule(_kwargs())
    assert len(first) == len(second) == 20

    expected = [
        (base_index, instruction_direction, execution_relation)
        for base_index in range(5)
        for instruction_direction in (FORWARD, BACKWARD)
        for execution_relation in (SAME, REVERSE)
    ]
    observed = [
        (
            cell.base_configuration_index,
            cell.instruction_direction,
            cell.execution_relation,
        )
        for cell in first
    ]
    assert observed == expected
    assert [cell.factorial_index for cell in first] == list(range(20))

    for cell, repeated in zip(first, second):
        assert cell.configuration == BASES[cell.base_configuration_index]
        assert cell.environment.batch == 1
        assert cell.environment.num_loops == 5
        assert cell.environment.instruction_repeats == 2
        assert tuple(cell.environment.configuration[0].tolist()) == cell.configuration
        assert int(cell.environment.instruction_direction[0]) == cell.instruction_direction
        assert int(cell.environment.execution_relation[0]) == cell.execution_relation
        assert int(cell.environment.start_location[0]) == int(
            repeated.environment.start_location[0]
        )


def test_factorial_evaluation_is_frozen_autonomous_and_restores_state_and_rng():
    # Keep this behavioral test short; the schedule/default test above guards
    # the intended five-loop scanner setting.
    kwargs = _kwargs(n_loops=1, instruction_repeats=1, max_navigation_steps=1)
    train_env = pysta.tasks.make_environment(kwargs, split="train")
    agent = pysta.agents.VanillaRNN(
        train_env,
        Nrec=8,
        rec_noise=0,
        # Exercise and then verify restoration of the legacy global NumPy RNG
        # path used to select a recurrent-microstep count from a list.
        iters_per_action=[1, 2],
        force_optimal=True,
        greedy=False,
    )
    parameters_before = [parameter.detach().clone() for parameter in agent.parameters()]
    rng_before = torch.random.get_rng_state().clone()
    numpy_rng_before = np.random.get_state()
    transient_names = pysta.train_rnn._AGENT_TRANSIENT_STATE_NAMES
    transient_before = {
        name: getattr(agent, name)
        for name in transient_names
        if hasattr(agent, name)
    }

    schedule = pysta.tasks.make_fmri_evaluation_schedule(kwargs)
    with pytest.raises(ValueError, match="exact base-major order"):
        pysta.train_rnn.evaluate_abcd_fmri_factorial(
            agent, (schedule[1], schedule[0], *schedule[2:])
        )
    loss, accuracy, metrics = pysta.train_rnn.evaluate_abcd_fmri_factorial(
        agent, schedule
    )

    assert np.isfinite(loss)
    assert np.isfinite(accuracy)
    assert metrics["evaluation_design"] == (
        "five_base_x_instruction_direction_x_execution_relation"
    )
    assert metrics["num_base_configurations"] == 5
    assert metrics["num_blocks"] == 20
    assert metrics["base_configurations"] == BASES
    assert metrics["weights_frozen"] is True
    assert metrics["autonomous"] is True
    assert metrics["greedy"] is True
    assert metrics["force_optimal"] is False
    assert len(metrics["factorial_cells"]) == 20
    assert len(metrics["trajectory_stores"]) == 20
    assert [cell["factorial_index"] for cell in metrics["factorial_cells"]] == list(
        range(20)
    )
    assert {
        (
            cell["base_configuration_index"],
            cell["instruction_direction"],
            cell["execution_relation"],
        )
        for cell in metrics["factorial_cells"]
    } == {
        (base_index, instruction_direction, execution_relation)
        for base_index in range(5)
        for instruction_direction in (FORWARD, BACKWARD)
        for execution_relation in (SAME, REVERSE)
    }

    assert agent.env is train_env
    assert agent.force_optimal is True
    assert agent.greedy is False
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    numpy_rng_after = np.random.get_state()
    assert numpy_rng_after[0] == numpy_rng_before[0]
    assert np.array_equal(numpy_rng_after[1], numpy_rng_before[1])
    assert numpy_rng_after[2:] == numpy_rng_before[2:]
    for name, value in transient_before.items():
        assert getattr(agent, name) is value
    assert agent.r.shape[0] == train_env.batch
    assert agent.z.shape[0] == train_env.batch
    for before, after in zip(parameters_before, agent.parameters()):
        assert torch.equal(before, after)


def test_maze_factory_does_not_require_or_use_scanner_settings():
    env = pysta.tasks.make_environment(
        {"task": "maze", "batch_size": 1, "side_length": 3, "max_steps": 1}
    )
    assert isinstance(env, pysta.envs.MazeEnv)
    with pytest.raises(ValueError, match="only defined for abcd_fmri"):
        pysta.tasks.make_fmri_evaluation_schedule(
            {"task": "maze", "fmri_base_configurations": BASES}
        )
