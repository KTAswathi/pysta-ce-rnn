"""Small full-block ABCD overfit smoke test (never instantiates/trains N480)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pysta.abcd_env import ABCDFMRIEnv, FORWARD, SAME
from pysta.agents import VanillaRNN


def main(epochs: int = 120):
    """Overfit one one-loop block and verify autonomous greedy completion."""
    torch.manual_seed(5)
    np.random.seed(5)

    # This deliberately tiny fixture uses the real two-repeat instruction and
    # reward-dwell state machine but one loop, one condition, and one generated
    # model configuration. It is a graph/integration check, not a fitted model.
    env = ABCDFMRIEnv(
        batch_size=16,
        seed=5,
        configuration_bank=[(0, 2, 8, 6)],
        instruction_directions=[FORWARD],
        execution_relations=[SAME],
        num_loops=1,
        instruction_repeats=2,
        start_policy="fixed",
        fixed_start=4,
        max_navigation_steps=80,
    )
    rnn = VanillaRNN(
        env,
        Nrec=64,
        rec_noise=0,
        force_optimal=True,
        iters_per_action=1,
        tau=2,
        W_reg=0,
        r_reg=0,
        ent_reg=0,
    )
    optimizer = torch.optim.Adam(rnn.parameters(), lr=3e-3)
    losses = []
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        loss = rnn.forward()  # one complete block; no BPTT truncation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))

    first_mean = float(np.mean(losses[:10]))
    final_mean = float(np.mean(losses[-10:]))
    if not np.isfinite(losses).all() or final_mean >= first_mean * 0.25:
        raise AssertionError(
            f"Overfit loss did not decrease enough: {first_mean:.4f} -> {final_mean:.4f}."
        )

    rnn.greedy = True
    rnn.force_optimal = False
    with torch.no_grad():
        autonomous_loss = float(rnn.forward(store=True))
    completed = (
        (env.successful_goal_count == env.total_required_goals) & ~env.truncated
    )
    if not torch.all(completed):
        raise AssertionError(
            "Tiny overfit model did not autonomously complete every fixture block."
        )

    print("Tiny ABCD full-block overfit smoke passed")
    print("  model: VanillaRNN, Nrec=64 (not the N480 cortical model)")
    print("  fixture: one configuration, FORWARD/SAME, one loop, batch=16")
    print(f"  training epochs: {epochs}")
    print(f"  mean loss, first 10: {first_mean:.6f}")
    print(f"  mean loss, final 10: {final_mean:.6f}")
    print(f"  autonomous loss: {autonomous_loss:.6f}")
    print(
        "  autonomous completion: "
        f"{int(completed.sum())}/{len(completed)} blocks; "
        f"goals={env.successful_goal_count.tolist()}; "
        f"truncated={env.truncated.tolist()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    arguments = parser.parse_args()
    main(epochs=arguments.epochs)
