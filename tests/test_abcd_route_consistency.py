"""Focused tests for realised ABCD route-consistency summaries."""

import json

import numpy as np

from pysta.abcd_analysis_utils import summarize_route_consistency
from pysta.abcd_env import move_location


CONFIGURATION = (0, 2, 8, 6)


def _record(
    *,
    location,
    post_location,
    model_action,
    env_action,
    navigation_index,
    navigation,
    success_count,
    target_reached=False,
    action_valid=False,
    next_finished=False,
    next_truncated=False,
    truncated=False,
):
    target = CONFIGURATION[success_count % 4]
    return {
        "valid_timestep": np.asarray([True], dtype=bool),
        "navigation_step_taken": np.asarray([navigation], dtype=bool),
        "current_location": np.asarray([location], dtype=np.int64),
        "post_action_location": np.asarray([post_location], dtype=np.int64),
        "action": np.asarray([model_action], dtype=np.int64),
        "env_action": np.asarray([env_action], dtype=np.int64),
        "navigation_step_index": np.asarray([navigation_index], dtype=np.int64),
        "successful_goal_count": np.asarray([success_count], dtype=np.int64),
        "current_required_abstract_goal_index": np.asarray(
            [success_count % 4], dtype=np.int64
        ),
        "current_required_physical_location": np.asarray(
            [target], dtype=np.int64
        ),
        "target_reached": np.asarray([target_reached], dtype=bool),
        "action_valid": np.asarray([action_valid], dtype=bool),
        "finished": np.asarray([False], dtype=bool),
        "next_finished": np.asarray([next_finished], dtype=bool),
        "next_truncated": np.asarray([next_truncated], dtype=bool),
        "truncated": np.asarray([truncated], dtype=bool),
        "next_successful_goal_count": np.asarray(
            [success_count + int(target_reached)], dtype=np.int64
        ),
    }


def _make_store(route_specs, *, terminate=True, truncated=False):
    """Build a one-trial store with instruction/reward rows around real routes."""
    store = [
        _record(
            location=4,
            post_location=4,
            model_action=0,
            env_action=0,
            navigation_index=-1,
            navigation=False,
            success_count=0,
        )
    ]
    location = 4
    navigation_index = 0
    success_count = 0
    for route_number, (actions, complete) in enumerate(route_specs):
        assert route_number == success_count
        target = CONFIGURATION[success_count % 4]
        for action_index, env_action in enumerate(actions):
            post_location, valid = move_location(location, env_action)
            reached = bool(
                complete
                and action_index == len(actions) - 1
                and post_location == target
            )
            is_last_store_row = (
                route_number == len(route_specs) - 1
                and action_index == len(actions) - 1
                and not complete
            )
            store.append(
                _record(
                    location=location,
                    post_location=post_location,
                    # Deliberately disagree: summaries must use realised
                    # teacher/environment actions, not model samples.
                    model_action=(env_action + 1) % 4,
                    env_action=env_action,
                    navigation_index=navigation_index,
                    navigation=True,
                    success_count=success_count,
                    target_reached=reached,
                    action_valid=valid,
                    next_finished=terminate and is_last_store_row,
                    truncated=truncated and is_last_store_row,
                )
            )
            navigation_index += 1
            location = post_location
        if complete:
            assert location == target
            success_count += 1
            is_final_reward = route_number == len(route_specs) - 1
            store.append(
                _record(
                    location=location,
                    post_location=location,
                    model_action=0,
                    env_action=0,
                    navigation_index=-1,
                    navigation=False,
                    success_count=success_count,
                    next_finished=terminate and is_final_reward,
                    truncated=truncated and is_final_reward,
                )
            )
    return store


def test_route_consistency_uses_realised_actions_and_excludes_initial_approach():
    route_specs = []
    for success_count in range(12):
        slot = success_count % 4
        if success_count == 0:
            actions = [0, 2]  # arbitrary block start 4 -> first goal 0
        elif slot == 0:
            actions = [0, 0]  # D=6 -> A=0
        elif slot == 1:
            actions = [3, 3]  # A=0 -> B=2
        elif slot == 2:
            actions = [1, 1]  # B=2 -> C=8
        else:
            actions = [2, 2]  # C=8 -> D=6
        if success_count == 5:
            actions = [0, 3, 3]  # invalid up at 0, then reach B
        route_specs.append((actions, True))

    summary = summarize_route_consistency(_make_store(route_specs))
    trial = summary["trials"][0]
    overall = summary["overall"]

    assert trial["status"] == "complete"
    assert trial["successful_goal_count"] == 12
    assert len(trial["routes"]) == 12
    assert trial["routes"][0]["initial_approach"]
    assert not trial["routes"][0]["included_in_repetition_consistency"]
    assert trial["routes"][1]["actions"] == [3, 3]
    assert trial["routes"][1]["actions"] != [0, 0]  # model actions were shifted

    slot_zero = summary["by_execution_slot"]["0"]
    assert slot_zero["completed_route_count"] == 3
    assert slot_zero["consistency_route_count"] == 2
    assert slot_zero["action_pair_count"] == 1
    assert slot_zero["exact_action_pairwise_rate"] == 1.0

    slot_one = summary["by_execution_slot"]["1"]
    assert slot_one["action_pair_count"] == 3
    assert slot_one["exact_action_pair_matches"] == 1
    assert slot_one["exact_action_pairwise_rate"] == 1 / 3
    assert slot_one["exact_location_pairwise_rate"] == 1 / 3
    assert slot_one["action_modal_fraction"] == 2 / 3
    assert slot_one["shortest_route_fraction"] == 2 / 3
    assert slot_one["total_excess_actions"] == 1
    assert slot_one["invalid_boundary_action_count"] == 1

    assert overall["consistency_route_count"] == 11
    assert overall["action_pair_count"] == 10
    assert overall["exact_action_pair_matches"] == 8
    assert overall["exact_action_pairwise_rate"] == 0.8
    assert overall["exact_location_pairwise_rate"] == 0.8
    assert overall["action_modal_fraction"] == 10 / 11
    assert overall["shortest_route_fraction"] == 11 / 12
    assert overall["total_excess_actions"] == 1
    assert overall["invalid_boundary_action_count"] == 1

    # No NumPy scalar/array leaks into saved JSON reports.
    json.dumps(summary)


def test_incomplete_terminal_leg_is_reported_as_truncated_not_compared():
    route_specs = [
        ([0, 2], True),   # initial approach to A
        ([3, 3], True),   # A -> B
        ([1], False),     # partial B -> C, then max-step termination
    ]
    summary = summarize_route_consistency(
        _make_store(route_specs, terminate=True, truncated=False)
    )
    trial = summary["trials"][0]
    incomplete = trial["routes"][-1]

    assert trial["status"] == "truncated"
    assert trial["terminated"]
    assert trial["truncated"]
    assert trial["has_incomplete_route"]
    assert incomplete["successful_goal_count_before"] == 2
    assert not incomplete["completed"]
    assert incomplete["shortest"] is None
    assert incomplete["excess_actions"] is None
    assert not incomplete["included_in_repetition_consistency"]
    assert trial["overall"]["incomplete_route_count"] == 1
    assert trial["overall"]["incomplete_navigation_action_count"] == 1
    assert trial["by_execution_slot"]["2"]["consistency_route_count"] == 0
    assert (
        trial["by_execution_slot"]["2"]["exact_action_pairwise_rate"] is None
    )
    json.dumps(summary)


def test_five_loop_block_has_the_expected_19_routes_and_36_comparisons():
    route_specs = []
    for success_count in range(20):
        slot = success_count % 4
        if success_count == 0:
            actions = [0, 2]  # block start -> first goal, not comparable
        elif slot == 0:
            actions = [0, 0]
        elif slot == 1:
            actions = [3, 3]
        elif slot == 2:
            actions = [1, 1]
        else:
            actions = [2, 2]
        route_specs.append((actions, True))

    summary = summarize_route_consistency(_make_store(route_specs))
    overall = summary["overall"]
    assert overall["completed_route_count"] == 20
    assert overall["consistency_route_count"] == 19
    assert overall["action_pair_count"] == 36
    assert overall["exact_action_pairwise_rate"] == 1.0
    assert overall["exact_location_pairwise_rate"] == 1.0
    assert overall["shortest_route_fraction"] == 1.0
    assert summary["by_execution_slot"]["0"]["consistency_route_count"] == 4
    assert summary["by_execution_slot"]["0"]["action_pair_count"] == 6
    for slot in ("1", "2", "3"):
        assert summary["by_execution_slot"][slot]["consistency_route_count"] == 5
        assert summary["by_execution_slot"][slot]["action_pair_count"] == 10


def test_offline_store_preserves_truncation_after_a_completed_reward_leg():
    # This is the max-navigation-cap edge case: the rewarding movement is a
    # complete leg, then the following reward dwell sets finished+truncated.
    store = _make_store([([0, 2], True)], terminate=True)
    store[-1]["truncated"][:] = False
    store[-1]["next_truncated"][:] = True
    summary = summarize_route_consistency(store)
    trial = summary["trials"][0]
    assert trial["routes"][0]["completed"]
    assert not trial["has_incomplete_route"]
    assert trial["terminated"]
    assert trial["truncated"]
    assert trial["status"] == "truncated"
