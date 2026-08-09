"""Print a compact, human-readable example of one complete ABCD block."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pysta.abcd_env import (
    ABCDFMRIEnv,
    ACTION_NAMES,
    BACKWARD,
    INSTRUCTION_DIRECTION_NAMES,
    NAVIGATION,
    REVERSE,
    REWARD,
    EXECUTION_RELATION_NAMES,
)


GOAL_NAMES = "ABCD"


def main():
    # These are illustrative model coordinates, not Svenja's experimental set.
    configuration = (0, 2, 8, 6)
    env = ABCDFMRIEnv(
        batch_size=1,
        seed=11,
        configuration_bank=[configuration],
        instruction_directions=[BACKWARD],
        execution_relations=[REVERSE],
        instruction_repeats=2,
        num_loops=5,
        start_policy="fixed",
        fixed_start=4,
        max_navigation_steps=200,
    )

    presented = env.presented_sequence[0].tolist()
    effective = env.effective_execution_sequence[0].tolist()
    print("ABCD human 7T-fMRI example block")
    print(
        "configuration: "
        + ", ".join(
            f"{GOAL_NAMES[index]}=cell {location}"
            for index, location in enumerate(configuration)
        )
    )
    print(
        "instruction_direction: "
        f"{INSTRUCTION_DIRECTION_NAMES[int(env.instruction_direction[0])]}"
    )
    print(
        "execution_relation: "
        f"{EXECUTION_RELATION_NAMES[int(env.execution_relation[0])]}"
    )
    print("presented abstract sequence: " + "-".join(GOAL_NAMES[i] for i in presented))
    print("effective execution sequence: " + "-".join(GOAL_NAMES[i] for i in effective))
    print(f"start location: cell {int(env.loc[0])}")
    print("instruction displays:")

    while int(env.phase[0]) == 0:
        presentation_index = int(env.instruction_presentation_index[0])
        sequence_position = presentation_index % 4
        abstract_goal = presented[sequence_position]
        physical_location = configuration[abstract_goal]
        repeat = presentation_index // 4 + 1
        print(
            f"  repeat {repeat}, item {sequence_position + 1}: "
            f"{GOAL_NAMES[abstract_goal]} at cell {physical_location}"
        )
        env.step(torch.tensor([0]))  # ignored during instruction

    print("navigation/reward trace:")
    while not bool(env.finished[0]):
        assert int(env.phase[0]) == NAVIGATION
        loop_number = int(env.loop_index[0]) + 1
        abstract_goal = int(env.current_required_abstract_goal_index[0])
        physical_goal = int(env.current_required_physical_location[0])
        route = [int(env.loc[0])]
        actions = []

        while int(env.phase[0]) == NAVIGATION:
            optimal = env.optimal_actions()[0]
            action = int(torch.where(optimal > 0)[0][0])
            actions.append(ACTION_NAMES[action])
            reward = float(env.step(torch.tensor([action]))[0])
            route.append(int(env.loc[0]))

        assert int(env.phase[0]) == REWARD and reward == 1.0
        print(
            f"  loop {loop_number}, goal {GOAL_NAMES[abstract_goal]} "
            f"(cell {physical_goal}): actions={actions}; route={route}; reward=+1"
        )

        successes = int(env.successful_goal_count[0])
        reward_location = int(env.loc[0])
        env.step(torch.tensor([0]))  # ignored during the explicit reward dwell
        assert int(env.loc[0]) == reward_location
        if successes % 4 == 0:
            if bool(env.finished[0]):
                print(
                    f"  loop {successes // 4} complete at cell {reward_location}; "
                    "final reward dwell -> block terminated"
                )
            else:
                print(
                    f"  loop {successes // 4} complete at cell {reward_location}; "
                    f"continue directly to loop {successes // 4 + 1}"
                )

    print(
        "termination: "
        f"finished={bool(env.finished[0])}, "
        f"successful_goals={int(env.successful_goal_count[0])}, "
        f"navigation_steps={int(env.navigation_step_count[0])}, "
        f"truncated={bool(env.truncated[0])}"
    )


if __name__ == "__main__":
    main()
