"""Regression tests for task-generic recurrent-microstep activity storage."""

import numpy as np
import pytest

from pysta.abcd_env import ABCDFMRIEnv, FORWARD, SAME
from pysta.agents import LineEmbeddedRNN, VanillaRNN
from pysta.envs import MazeEnv


@pytest.mark.parametrize("agent_class", [VanillaRNN, LineEmbeddedRNN])
def test_abcd_all_activity_uses_native_block_time_and_phase_metadata(agent_class):
    env = ABCDFMRIEnv(
        batch_size=1,
        seed=7,
        configuration_bank=[(0, 2, 8, 6)],
        instruction_directions=[FORWARD],
        execution_relations=[SAME],
        num_loops=1,
        instruction_repeats=2,
        max_navigation_steps=50,
        start_policy="fixed",
        fixed_start=4,
    )
    agent = agent_class(
        env,
        Nrec=12,
        W_reg=0.0,
        r_reg=0.0,
        rec_noise=0.0,
        iters_per_action=2,
        force_optimal=True,
    )
    agent.store_all_activity = True

    loss = agent.forward()

    assert np.isfinite(float(loss.detach().cpu()))
    assert bool(env.finished[0])
    assert not hasattr(env, "step_num")
    assert agent.all_acts_time_name == "block_timestep"
    assert len(agent.all_acts[0]) == len(agent.all_acts[1])
    assert len(agent.all_acts[0]) == len(agent.all_acts[2])
    assert len(agent.all_acts[0]) == len(agent.all_acts_metadata)
    assert len(agent.all_acts[0]) > 0

    assert all(isinstance(activity, np.ndarray) for activity in agent.all_acts[0])
    assert all(isinstance(location, np.ndarray) for location in agent.all_acts[1])
    assert all(isinstance(time, np.ndarray) for time in agent.all_acts[2])

    phases = set()
    navigation_indices = []
    for index, metadata in enumerate(agent.all_acts_metadata):
        assert metadata["rnn_microstep_index"] == index
        assert metadata["activity_time_name"] == "block_timestep"
        np.testing.assert_array_equal(
            metadata["activity_time"], agent.all_acts[2][index]
        )
        np.testing.assert_array_equal(
            metadata["location"], agent.all_acts[1][index]
        )
        phases.update(np.asarray(metadata["phase"]).reshape(-1).tolist())
        navigation_indices.extend(
            np.asarray(metadata["navigation_step_index"]).reshape(-1).tolist()
        )

    # Instruction, navigation, and reward-dwell microsteps remain explicitly
    # distinguishable; navigation-only time is retained independently.
    assert phases == {0, 1, 2}
    assert any(index >= 0 for index in navigation_indices)
    assert any(index == -1 for index in navigation_indices)


def test_original_task_all_activity_keeps_legacy_step_num_coordinate():
    env = MazeEnv(
        side_length=2,
        max_steps=1,
        batch_size=2,
        sample_wall_num=1,
        planning_steps=1,
    )
    agent = VanillaRNN(
        env,
        Nrec=8,
        W_reg=0.0,
        r_reg=0.0,
        rec_noise=0.0,
        iters_per_action=2,
    )
    agent.store_all_activity = True
    expected_step_num = env.step_num

    agent.step(env.observation())

    assert agent.all_acts_time_name == "step_num"
    assert agent.all_acts[2] == [expected_step_num, expected_step_num]
    assert [record["activity_time_name"] for record in agent.all_acts_metadata] == [
        "step_num",
        "step_num",
    ]

    # Legacy scripts sometimes clear only the original positional cache.
    # Old pickles also lack the new metadata attributes.  Both cases must
    # initialise lazily rather than retaining stale rows or raising.
    agent.all_acts = [[], [], []]
    del agent.all_acts_metadata
    del agent.all_acts_time_name
    agent.step(env.observation())
    assert len(agent.all_acts_metadata) == 2
    assert [record["rnn_microstep_index"] for record in agent.all_acts_metadata] == [
        0,
        1,
    ]
