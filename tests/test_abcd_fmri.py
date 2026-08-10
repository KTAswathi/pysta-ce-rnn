"""Focused regression tests for the additive human 7T-fMRI ABCD task."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

import pysta
from pysta.abcd_analysis_utils import (
    extract_navigation_trajectories,
    save_agent_store,
    stack_store_records,
)
from pysta.abcd_env import (
    ABCDFMRIEnv,
    ACTION_NAMES,
    BACKWARD,
    DOWN,
    EXECUTION_RELATION_NAMES,
    FORWARD,
    INSTRUCTION,
    LEFT,
    NAVIGATION,
    OBSERVATION_SLICES,
    REVERSE,
    REWARD,
    RIGHT,
    SAME,
    UP,
    canonical_configuration_cycle,
    configuration_bank_statistics,
    configuration_cycle_variants,
    configuration_has_minimum_distance,
    generate_configuration_bank,
    generate_synthetic_fmri_configuration_bank,
    location_to_row_col,
    manhattan_distance,
    move_location,
    parse_configurations,
    row_col_to_location,
    split_configuration_bank,
    validate_fmri_configuration_bank,
)


CONFIGURATION = (0, 2, 8, 6)


def make_env(**overrides):
    kwargs = dict(
        batch_size=1,
        seed=7,
        configuration_bank=[CONFIGURATION],
        instruction_directions=[FORWARD],
        execution_relations=[SAME],
        num_loops=1,
        instruction_repeats=2,
        max_navigation_steps=100,
        start_policy="fixed",
        fixed_start=4,
    )
    kwargs.update(overrides)
    return ABCDFMRIEnv(**kwargs)


def finish_instruction(env):
    for _ in range(env.total_instruction_steps):
        assert int(env.phase[0]) == INSTRUCTION
        env.step(torch.tensor([UP]))
    assert int(env.phase[0]) == NAVIGATION


def take_one_optimal_action(env):
    mask = env.optimal_actions()[0]
    assert mask.sum() >= 1
    action = int(torch.where(mask > 0)[0][0])
    old_location = int(env.loc[0])
    reward = float(env.step(torch.tensor([action]))[0])
    return action, old_location, int(env.loc[0]), reward


def reach_current_goal(env):
    actions = []
    while int(env.phase[0]) == NAVIGATION:
        action, _, _, reward = take_one_optimal_action(env)
        actions.append(action)
    assert int(env.phase[0]) == REWARD
    assert reward == 1.0
    return actions


def test_grid_indexing_movement_and_boundaries():
    assert [row_col_to_location(*location_to_row_col(i)) for i in range(9)] == list(
        range(9)
    )
    assert move_location(4, UP) == (1, True)
    assert move_location(4, DOWN) == (7, True)
    assert move_location(4, LEFT) == (3, True)
    assert move_location(4, RIGHT) == (5, True)
    assert move_location(0, UP) == (0, False)
    assert move_location(0, LEFT) == (0, False)
    assert move_location(8, DOWN) == (8, False)
    assert move_location(8, RIGHT) == (8, False)
    assert manhattan_distance(0, 8) == 4
    assert ACTION_NAMES == ("up", "down", "left", "right")


def test_configuration_generation_is_distinct_separated_and_deterministic():
    first = generate_configuration_bank(40, seed=123)
    second = generate_configuration_bank(40, seed=123)
    different_seed = generate_configuration_bank(40, seed=124)
    assert first == second
    assert first != different_seed
    assert len(first) == len(set(first)) == 40
    for configuration in first:
        assert len(set(configuration)) == 4
        assert configuration_has_minimum_distance(configuration, 2)
        # The generator preferentially satisfies the stronger reported fMRI
        # all-pairs separation for banks of this size.
        assert configuration_has_minimum_distance(configuration, 2, all_pairs=True)

    train, held_out = split_configuration_bank(
        num_train=10, num_eval=8, seed=19
    )
    assert set(train).isdisjoint(held_out)
    assert {
        canonical_configuration_cycle(configuration) for configuration in train
    }.isdisjoint(
        canonical_configuration_cycle(configuration) for configuration in held_out
    )
    assert parse_configurations("0,2,8,6;2,8,6,0") == (
        (0, 2, 8, 6),
        (2, 8, 6, 0),
    )
    with pytest.raises(ValueError, match="distinct"):
        ABCDFMRIEnv(configuration_bank=[(0, 0, 2, 8)])
    with pytest.raises(ValueError, match="minimum"):
        ABCDFMRIEnv(configuration_bank=[(0, 1, 8, 6)])
    with pytest.raises(ValueError, match="minimum"):
        ABCDFMRIEnv(configuration_bank=[(0, 5, 1, 6)])


def test_cycle_equivalence_and_scanner_bank_statistics_are_explicit():
    configuration = (0, 2, 8, 6)
    variants = configuration_cycle_variants(configuration)
    assert len(variants) == 8
    assert len({canonical_configuration_cycle(value) for value in variants}) == 1
    assert canonical_configuration_cycle((0, 2, 6, 8)) != (
        canonical_configuration_cycle(configuration)
    )

    all_cycle_classes = generate_configuration_bank(
        18, seed=4, unique_up_to_cycle=True
    )
    assert len(
        {canonical_configuration_cycle(value) for value in all_cycle_classes}
    ) == 18
    with pytest.raises(ValueError, match="cycle-equivalence"):
        generate_configuration_bank(19, seed=4, unique_up_to_cycle=True)

    bases = generate_configuration_bank(5, seed=9, unique_up_to_cycle=True)
    scanner_like = tuple(
        value for base in bases for value in (base, tuple(reversed(base)))
    )
    report = validate_fmri_configuration_bank(scanner_like)
    assert report == configuration_bank_statistics(scanner_like)
    assert report["num_configurations"] == 10
    assert report["num_direct_reversal_classes"] == 5
    assert report["direct_reversals_complete"]
    assert sum(report["location_counts"]) == 40
    assert len(report["circular_manhattan_distances"]) == 10
    assert isinstance(report["difference_from_reported_mean_2p6"], float)

    with pytest.raises(ValueError, match="inverse"):
        validate_fmri_configuration_bank(scanner_like[:-1] + (all_cycle_classes[7],))

    balance_bank = generate_synthetic_fmri_configuration_bank(
        seed=12, objective="balance_first"
    )
    assert balance_bank == generate_synthetic_fmri_configuration_bank(
        seed=12, objective="balance_first"
    )
    distance_bank = generate_synthetic_fmri_configuration_bank(
        seed=12, objective="distance_first"
    )
    balance_report = validate_fmri_configuration_bank(balance_bank)
    distance_report = validate_fmri_configuration_bank(distance_bank)
    assert balance_report["location_balance_squared_error"] <= (
        distance_report["location_balance_squared_error"]
    )
    assert abs(distance_report["difference_from_reported_mean_2p6"]) <= abs(
        balance_report["difference_from_reported_mean_2p6"]
    )


@pytest.mark.parametrize(
    "instruction_direction,execution_relation,presented,effective",
    [
        (FORWARD, SAME, (0, 1, 2, 3), (0, 1, 2, 3)),
        (FORWARD, REVERSE, (0, 1, 2, 3), (3, 2, 1, 0)),
        (BACKWARD, SAME, (3, 2, 1, 0), (3, 2, 1, 0)),
        (BACKWARD, REVERSE, (3, 2, 1, 0), (0, 1, 2, 3)),
    ],
)
def test_all_instruction_execution_combinations(
    instruction_direction, execution_relation, presented, effective
):
    env = make_env(
        instruction_directions=[instruction_direction],
        execution_relations=[execution_relation],
    )
    assert tuple(env.presented_sequence[0].tolist()) == presented
    assert tuple(env.effective_execution_sequence[0].tolist()) == effective

    displayed_locations = []
    for presentation_index in range(8):
        observation = env.observation()[0]
        instruction_input = observation[OBSERVATION_SLICES["instruction_location"]]
        displayed_locations.append(int(torch.argmax(instruction_input)))
        assert int(env.instruction_presentation_index[0]) == presentation_index
        env.step(torch.tensor([RIGHT]))
    expected_once = [CONFIGURATION[index] for index in presented]
    assert displayed_locations == expected_once * 2
    assert int(env.phase[0]) == NAVIGATION


def test_phase_gated_observations_and_no_navigation_leakage():
    env = make_env()
    instruction_obs = env.observation()[0]
    assert instruction_obs[OBSERVATION_SLICES["current_location"]].sum() == 0
    assert instruction_obs[OBSERVATION_SLICES["instruction_location"]].sum() == 1
    assert instruction_obs[OBSERVATION_SLICES["execution_rule"]].tolist() == [1, 0]
    assert instruction_obs[OBSERVATION_SLICES["phase"]].tolist() == [1, 0, 0]
    assert instruction_obs[OBSERVATION_SLICES["reward_event"]].item() == 0
    assert not env.policy_loss_mask()[0]

    finish_instruction(env)
    navigation_obs = env.observation()[0]
    assert navigation_obs[OBSERVATION_SLICES["current_location"]].sum() == 1
    assert navigation_obs[OBSERVATION_SLICES["instruction_location"]].sum() == 0
    assert navigation_obs[OBSERVATION_SLICES["execution_rule"]].sum() == 0
    assert navigation_obs[OBSERVATION_SLICES["phase"]].tolist() == [0, 1, 0]
    assert navigation_obs[OBSERVATION_SLICES["reward_event"]].item() == 0
    assert env.policy_loss_mask()[0]

    # Two different hidden configurations with the same physical state are
    # indistinguishable at the navigation interface.
    other = make_env(configuration_bank=[(2, 8, 6, 0)])
    finish_instruction(other)
    assert torch.equal(navigation_obs, other.observation()[0])


def test_reward_required_goal_only_boundary_and_out_of_order_do_not_advance():
    env = make_env(start_policy="fixed", fixed_start=1)
    finish_instruction(env)
    required_before = int(env._required_physical_location()[0])
    assert required_before == 0

    # Visit B=2 out of order.
    reward = env.step(torch.tensor([RIGHT]))
    assert int(env.loc[0]) == 2
    assert reward.item() == 0
    assert int(env.successful_goal_count[0]) == 0
    assert int(env._required_physical_location()[0]) == required_before

    # Right from cell 2 is a boundary action: no movement, reward, or advance.
    reward = env.step(torch.tensor([RIGHT]))
    assert int(env.loc[0]) == 2
    assert reward.item() == 0
    assert not env.post_step_metadata()["action_valid"][0]
    assert int(env.successful_goal_count[0]) == 0

    env.step(torch.tensor([LEFT]))
    reward = env.step(torch.tensor([LEFT]))
    assert reward.item() == 1
    assert int(env.loc[0]) == 0
    assert int(env.phase[0]) == REWARD
    assert int(env.successful_goal_count[0]) == 1
    # The just-reached target remains current throughout the explicit dwell.
    assert int(env._required_physical_location()[0]) == 0


def test_reward_dwell_has_fixed_location_no_policy_loss_and_then_advances():
    env = make_env(start_policy="fixed", fixed_start=1)
    finish_instruction(env)
    env.step(torch.tensor([LEFT]))
    assert int(env.phase[0]) == REWARD
    reached_location = int(env.loc[0])
    reward_obs = env.observation()[0]
    assert reward_obs[OBSERVATION_SLICES["current_location"]].sum() == 1
    assert reward_obs[OBSERVATION_SLICES["instruction_location"]].sum() == 0
    assert reward_obs[OBSERVATION_SLICES["execution_rule"]].sum() == 0
    assert reward_obs[OBSERVATION_SLICES["phase"]].tolist() == [0, 0, 1]
    assert reward_obs[OBSERVATION_SLICES["reward_event"]].item() == 1
    assert not env.policy_loss_mask()[0]

    env.step(torch.tensor([RIGHT]))  # ignored during the reward dwell
    assert int(env.loc[0]) == reached_location
    assert int(env.phase[0]) == NAVIGATION
    assert int(env._required_abstract_goal()[0]) == 1
    assert int(env._required_physical_location()[0]) == 2


def test_uniform_start_is_uniform_over_all_cells_and_target_start_is_well_defined():
    uniform = make_env(start_policy="uniform", fixed_start=None)
    counts = np.zeros(9, dtype=int)
    for _ in range(900):
        uniform.reset()
        counts[int(uniform.loc[0])] += 1
    assert np.all(counts > 0)
    assert counts.max() - counts.min() < 60

    starts_at_target = make_env(start_policy="fixed", fixed_start=0)
    assert starts_at_target.started_on_first_goal.tolist() == [True]
    for _ in range(starts_at_target.total_instruction_steps - 1):
        assert starts_at_target.step(torch.tensor([UP])).item() == 0
    reward = starts_at_target.step(torch.tensor([UP]))
    assert reward.item() == 1
    assert starts_at_target.phase.tolist() == [REWARD]
    assert starts_at_target.successful_goal_count.tolist() == [1]
    assert starts_at_target.navigation_step_count.tolist() == [0]
    assert not starts_at_target.policy_loss_mask()[0]
    assert starts_at_target.post_step_metadata()["target_reached"].tolist() == [True]

    # Explicit reset locations use the same documented onset rule.
    starts_at_target.reset(start_locations=[0])
    assert starts_at_target.started_on_first_goal.tolist() == [True]


def test_asynchronous_batch_rows_can_occupy_different_phases():
    env = make_env(batch_size=2)
    env.reset(
        configurations=[CONFIGURATION],
        instruction_directions=[FORWARD],
        execution_relations=[SAME],
        start_locations=[1, 4],
    )
    for _ in range(env.total_instruction_steps):
        env.step(torch.tensor([UP, UP]))

    env.step(torch.tensor([LEFT, UP]))
    assert env.phase.tolist() == [REWARD, NAVIGATION]
    assert env.policy_loss_mask().tolist() == [False, True]
    observations = env.observation()
    assert observations[0, OBSERVATION_SLICES["phase"]].tolist() == [0, 0, 1]
    assert observations[1, OBSERVATION_SLICES["phase"]].tolist() == [0, 1, 0]

    # The first row consumes its fixed reward dwell while the second reaches A.
    env.step(torch.tensor([RIGHT, LEFT]))
    assert env.phase.tolist() == [NAVIGATION, REWARD]
    assert env.loc.tolist() == [0, 0]


def test_navigation_limit_miss_reward_dwell_and_final_completion_timing():
    missed = make_env(max_navigation_steps=1)
    finish_instruction(missed)
    missed.step(torch.tensor([UP]))
    assert bool(missed.finished[0])
    assert bool(missed.truncated[0])
    assert int(missed.successful_goal_count[0]) == 0

    intermediate = make_env(
        start_policy="fixed", fixed_start=1, max_navigation_steps=1
    )
    finish_instruction(intermediate)
    assert intermediate.step(torch.tensor([LEFT])).item() == 1
    assert int(intermediate.phase[0]) == REWARD
    assert not bool(intermediate.finished[0])
    intermediate.step(torch.tensor([UP]))
    assert bool(intermediate.finished[0])
    assert bool(intermediate.truncated[0])
    assert int(intermediate.successful_goal_count[0]) == 1

    final = make_env(max_navigation_steps=100)
    finish_instruction(final)
    for _ in range(3):
        reach_current_goal(final)
        final.step(torch.tensor([UP]))
    assert int(final._required_abstract_goal()[0]) == 3
    # Move one step toward D, then arrange for the final goal to be reached on
    # the exact navigation limit.
    final.step(torch.tensor([LEFT]))
    final.max_navigation_steps = int(final.navigation_step_count[0]) + 1
    assert final.step(torch.tensor([LEFT])).item() == 1
    assert int(final.successful_goal_count[0]) == 4
    assert int(final.phase[0]) == REWARD
    assert not bool(final.truncated[0])
    final.step(torch.tensor([UP]))
    assert bool(final.finished[0])
    assert not bool(final.truncated[0])


def test_goal_advancement_wraparound_and_five_continuous_loops():
    env = make_env(num_loops=5, start_policy="fixed", fixed_start=4)
    initial_start = int(env.loc[0])
    finish_instruction(env)
    reward_targets = []
    locations_after_loop_dwell = []

    while not bool(env.finished[0]):
        if int(env.phase[0]) == NAVIGATION:
            target = int(env._required_abstract_goal()[0])
            _, _, _, reward = take_one_optimal_action(env)
            if reward:
                reward_targets.append(target)
        else:
            assert int(env.phase[0]) == REWARD
            location_before_dwell = int(env.loc[0])
            successes = int(env.successful_goal_count[0])
            env.step(torch.tensor([UP]))
            assert int(env.loc[0]) == location_before_dwell
            if successes % 4 == 0:
                locations_after_loop_dwell.append(int(env.loc[0]))
                if successes < 20:
                    assert int(env._required_abstract_goal()[0]) == 0

    assert initial_start == 4
    assert reward_targets == [0, 1, 2, 3] * 5
    assert int(env.successful_goal_count[0]) == 20
    assert len(locations_after_loop_dwell) == 5
    # Every loop ends at D and the next begins from exactly that same position.
    assert locations_after_loop_dwell == [CONFIGURATION[3]] * 5
    assert not bool(env.truncated[0])


@pytest.mark.parametrize(
    "location,configuration,expected",
    [
        (4, (0, 2, 8, 6), (1, 0, 1, 0)),  # up or left
        (0, (8, 6, 0, 2), (0, 1, 0, 1)),  # down or right
        (1, (0, 2, 8, 6), (0, 0, 1, 0)),
        (3, (0, 2, 8, 6), (1, 0, 0, 0)),
    ],
)
def test_shortest_path_optimal_action_sets_include_ties(
    location, configuration, expected
):
    env = make_env(configuration_bank=[configuration])
    finish_instruction(env)
    env.loc[0] = location
    assert tuple(env.optimal_actions()[0].to(torch.int64).tolist()) == expected
    # Boundaries are not silently masked from model sampling/evaluation.
    assert tuple(env.action_sampling_mask()[0].tolist()) == (1, 1, 1, 1)


def test_four_action_readout_loss_masks_and_teacher_forced_action_are_stored():
    torch.manual_seed(3)
    env = make_env(batch_size=3, num_loops=1)
    agent = pysta.agents.VanillaRNN(
        env,
        Nrec=12,
        rec_noise=0,
        iters_per_action=1,
        tau=2,
        force_optimal=True,
    )
    loss = agent.forward(store=True)
    assert torch.isfinite(loss)
    assert agent.Nin == 24
    assert agent.Nout == agent.Wout.shape[0] == 4
    assert len(agent.store) > env.total_instruction_steps

    for state in agent.store:
        phase = state["phase"]
        expected_loss_mask = state["valid_timestep"] & (phase == NAVIGATION)
        assert torch.equal(state["loss_mask"].cpu(), expected_loss_mask)
        assert torch.isnan(state["corrects"][~state["loss_mask"]]).all()
        if torch.any(state["loss_mask"]):
            rows = torch.where(state["loss_mask"])[0]
            env_actions = state["env_action"][rows]
            assert torch.all(
                state["optimal_actions"][rows, env_actions] == 1
            )
        assert state["xs"].shape[-1] == 24
        assert "post_action_location" in state
        assert "navigation_step_taken" in state


def test_store_export_and_future_lags_count_navigation_steps_only(tmp_path):
    torch.manual_seed(13)
    env = make_env(batch_size=2, num_loops=1)
    agent = pysta.agents.VanillaRNN(
        env,
        Nrec=10,
        rec_noise=0,
        force_optimal=True,
        iters_per_action=1,
    )
    agent.forward(store=True)
    stacked = stack_store_records(agent.store)
    assert stacked["rs"].shape[:2] == stacked["valid_timestep"].shape
    assert stacked["xs"].shape[-1] == 24
    assert "effective_execution_physical_sequence" in stacked
    assert "block_navigation_index" in stacked

    trajectories = extract_navigation_trajectories(
        stacked, future_lags=(0, 1, 2)
    )
    assert len(trajectories) == 2
    for trajectory in trajectories:
        n_steps = trajectory["num_navigation_steps"]
        assert n_steps > 0
        assert np.array_equal(
            trajectory["navigation_index"], np.arange(n_steps)
        )
        assert np.array_equal(
            trajectory["future_location_by_lag"][0],
            trajectory["pre_location"],
        )
        assert np.array_equal(
            trajectory["future_location_by_lag"][1],
            trajectory["post_location"],
        )
        assert np.array_equal(
            trajectory["future_location_by_lag"][2][:-1],
            trajectory["post_location"][1:],
        )
        assert not trajectory["future_location_valid_by_lag"][2][-1]
        # There are four intervening reward-dwell records, proving store time
        # is not being used as the future-location lag clock.
        assert len(agent.store) - n_steps >= env.total_instruction_steps + 4

    destination = save_agent_store(tmp_path / "abcd_block", stacked)
    with np.load(destination) as saved:
        assert np.array_equal(saved["xs"], stacked["xs"])
        assert np.array_equal(
            saved["valid_timestep"], stacked["valid_timestep"]
        )


def test_intended_semantic_and_embedded_input_routing():
    env = make_env()
    assert env.input_routing() == {
        "current_location": "local",
        "instruction_location": "local",
        "execution_rule": "global",
        "phase": "global",
        "reward_event": "global",
    }
    agent = pysta.agents.LineEmbeddedRNN(
        env,
        Nrec=12,
        local_fraction=0.25,
        rec_noise=0,
    )
    assert agent.input_routing_modes == {
        "current_location": "same_end",
        "instruction_location": "same_end",
        "execution_rule": "global",
        "phase": "global",
        "reward_event": "global",
    }
    local_units = int(agent.same_end_unit_mask.sum())
    for group in ("current_location", "instruction_location"):
        mask = getattr(agent, agent.input_mask_buffers[group])
        assert int(mask.sum()) == local_units * len(env.obs_inds()[group])
    for group in ("execution_rule", "phase", "reward_event"):
        mask = getattr(agent, agent.input_mask_buffers[group])
        assert int(mask.sum()) == agent.Nrec * len(env.obs_inds()[group])
    assert torch.all(agent.readout_unit_mask == 1)


def test_abcd_cli_familiar_primary_and_strict_heldout_task_factory(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train"])
    raw_defaults = pysta.argparser.apply_task_defaults({"task": "abcd_fmri"})
    assert raw_defaults["model_type"] == "corticallyembedded"
    assert raw_defaults["Nrec"] == 480
    abcd_defaults = pysta.argparser.parse_args(task="abcd_fmri")
    assert abcd_defaults["batch_size"] == 8
    assert abcd_defaults["evaluation_mode"] == "familiar"
    familiar_kwargs = pysta.argparser.parse_args(
        task="abcd_fmri",
        batch_size=2,
        num_train_configurations=8,
        num_eval_configurations=7,
    )
    assert familiar_kwargs["Nrec"] == 480
    assert familiar_kwargs["embedding_name"] == "mpfc_projected_mask_linear0p1"
    train = pysta.tasks.make_environment(familiar_kwargs, split="train")
    evaluation = pysta.tasks.make_environment(familiar_kwargs, split="eval")
    assert train.instruction_repeats == evaluation.instruction_repeats == 2
    assert train.num_loops == evaluation.num_loops == 5
    assert train.configuration_bank == evaluation.configuration_bank
    assert train.name != evaluation.name

    heldout_kwargs = dict(familiar_kwargs, evaluation_mode="heldout")
    heldout_train = pysta.tasks.make_environment(heldout_kwargs, split="train")
    heldout_eval = pysta.tasks.make_environment(heldout_kwargs, split="eval")
    assert {
        canonical_configuration_cycle(configuration)
        for configuration in heldout_train.configuration_bank
    }.isdisjoint(
        canonical_configuration_cycle(configuration)
        for configuration in heldout_eval.configuration_bank
    )

    familiar_subset = ";".join(
        ",".join(str(value) for value in configuration)
        for configuration in train.configuration_bank[:3]
    )
    subset_kwargs = dict(familiar_kwargs, familiar_configurations=familiar_subset)
    subset_eval = pysta.tasks.make_environment(subset_kwargs, split="eval")
    assert subset_eval.configuration_bank == train.configuration_bank[:3]

    synthetic_kwargs = dict(
        familiar_kwargs,
        num_train_configurations=None,
        synthetic_fmri_bank_objective="balance_first",
    )
    synthetic_train = pysta.tasks.make_environment(synthetic_kwargs, split="train")
    synthetic_eval = pysta.tasks.make_environment(synthetic_kwargs, split="eval")
    assert synthetic_train.configuration_bank == synthetic_eval.configuration_bank
    validate_fmri_configuration_bank(synthetic_train.configuration_bank)

    alien = generate_configuration_bank(
        1,
        seed=91,
        exclude_configurations=train.configuration_bank,
    )[0]
    with pytest.raises(ValueError, match="familiar evaluation"):
        pysta.tasks.make_environment(
            dict(familiar_kwargs, familiar_configurations=[alien]), split="eval"
        )


def test_familiar_evaluation_is_autonomous_records_routes_and_restores_training_state():
    train_env = make_env(num_loops=1, max_navigation_steps=5)
    eval_env = make_env(num_loops=1, max_navigation_steps=5)
    eval_env.bank_name = "eval_familiar"
    agent = pysta.agents.VanillaRNN(
        train_env,
        Nrec=8,
        rec_noise=0,
        iters_per_action=1,
        force_optimal=True,
        greedy=False,
    )
    loss, accuracy, metrics = pysta.train_rnn._evaluate_abcd(
        agent, eval_env, num_eval=1, evaluation_mode="familiar"
    )
    assert np.isfinite(loss)
    assert np.isfinite(accuracy)
    assert metrics["evaluation_mode"] == "familiar"
    assert metrics["configuration_bank"] == eval_env.configuration_bank
    assert metrics["configuration_bank_statistics"]["num_configurations"] == 1
    assert metrics["route_consistency"]["action_source"] == "env_action"
    assert metrics["evaluation_recurrent_noise"] == 0
    assert agent.env is train_env
    assert agent.force_optimal is True
    assert agent.greedy is False


def test_jensen_maze_and_legacy_routing_still_run(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train"])
    defaults = pysta.argparser.parse_args()
    assert defaults["task"] == "maze"
    assert defaults["batch_size"] == 200
    assert defaults["Nrec"] == 347
    assert defaults["embedding_name"] == "mpfc_union_gradient_nearest_a24_25_anchor25"

    env = pysta.envs.MazeEnv(
        side_length=3,
        max_steps=2,
        planning_steps=0,
        batch_size=2,
        output_format="allocentric",
    )
    agent = pysta.agents.VanillaRNN(
        env, Nrec=10, rec_noise=0, force_optimal=True
    )
    assert torch.isfinite(agent.forward(store=True))
    assert agent.Nout == 9

    embedded = pysta.agents.LineEmbeddedRNN(
        env,
        Nrec=12,
        local_fraction=0.25,
        localize_loc_input=True,
        localize_rew_input=False,
        localize_wall_input=False,
    )
    assert hasattr(embedded, "mask_loc")
    assert hasattr(embedded, "mask_rew")
    assert hasattr(embedded, "mask_wall")
    # Whole-object checkpoints predating generic routing do not contain the
    # newly registered combined buffer; the legacy three-mask path still runs.
    del embedded._buffers["mask_input"]
    assert torch.isfinite(embedded.forward())


def test_default_n480_cortical_artifact_routing_when_available():
    embedding_dir = (
        Path(pysta.basedir)
        / "data/embedding/subsampled/human/mpfc_projected_mask_linear0p1/units=480_seed=42"
    )
    if not embedding_dir.exists():
        pytest.skip("The preserved N480 cortical embedding artifact is not installed.")

    env = make_env()
    agent = pysta.agents.CorticallyEmbeddedRNN(
        env,
        Nrec=480,
        embedding_name="mpfc_projected_mask_linear0p1",
        embedding_species="human",
        embedding_seed=42,
        readout_mode="global",
        rec_noise=0,
    )
    assert agent.sampled_vertex_indices.shape == (480,)
    assert agent.distance_matrix.shape == (480, 480)
    assert agent.Wout.shape == (4, 480)
    assert torch.all(agent.readout_unit_mask == 1)
    assert agent.input_routing_modes["current_location"] == "same_end"
    assert agent.input_routing_modes["instruction_location"] == "same_end"
    for group in ("execution_rule", "phase", "reward_event"):
        assert agent.input_routing_modes[group] == "global"
