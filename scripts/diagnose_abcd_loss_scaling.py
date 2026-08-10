#!/usr/bin/env python3
"""Run one N480 ABCD forward pass and report objective-component scaling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pysta.abcd_env import (
    ABCDFMRIEnv,
    generate_configuration_bank,
    parse_configurations,
)
from pysta.agents import CorticallyEmbeddedRNN
from pysta.training_diagnostics import (
    diagnose_abcd_loss_scaling,
    format_loss_scaling_diagnostic,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure existing ABCD loss components on one full recurrent block "
            "batch. This command does not train or alter coefficients."
        )
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-loops", type=int, default=5)
    parser.add_argument("--instruction-repeats", type=int, default=2)
    parser.add_argument(
        "--num-configurations",
        type=int,
        default=12,
        help="generated physical-cycle representatives (matches the ABCD task default)",
    )
    parser.add_argument("--configuration-seed", type=int, default=0)
    parser.add_argument(
        "--configurations",
        type=str,
        default=None,
        help="optional semicolon-separated explicit configurations",
    )
    parser.add_argument("--max-navigation-steps", type=int, default=200)
    parser.add_argument(
        "--start-position-policy",
        choices=("exclude_first_goal", "uniform", "fixed"),
        default="exclude_first_goal",
    )
    parser.add_argument("--start-position", type=int, default=None)

    parser.add_argument("--Nrec", type=int, default=480)
    parser.add_argument("--iters-per-action", type=int, default=10)
    parser.add_argument("--tau", type=float, default=5.0)
    parser.add_argument("--rec-noise", type=float, default=1e-3)
    parser.add_argument("--ent-reg", type=float, default=1e-4)
    parser.add_argument("--r-reg", type=float, default=1e-5)
    parser.add_argument("--W-reg", type=float, default=2e-7)
    parser.add_argument("--dist-reg", type=float, default=1e-7)
    parser.add_argument("--line-decay", type=float, default=0.12)
    parser.add_argument("--line-init-scale", type=float, default=1.0)
    parser.add_argument("--embedding-name", default="mpfc_projected_mask_linear0p1")
    parser.add_argument("--embedding-species", default="human")
    parser.add_argument("--embedding-seed", type=int, default=42)
    parser.add_argument(
        "--force-optimal",
        type=int,
        choices=(0, 1),
        default=1,
        help="1 matches teacher-forced training trajectories; 0 is autonomous",
    )
    parser.add_argument("--greedy", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    configurations = parse_configurations(args.configurations)
    if not configurations:
        configurations = generate_configuration_bank(
            num_configurations=args.num_configurations,
            seed=args.configuration_seed,
            min_manhattan_distance=2,
            prefer_all_pairs=True,
            unique_up_to_cycle=True,
        )

    environment = ABCDFMRIEnv(
        batch_size=args.batch_size,
        seed=args.seed,
        configuration_bank=configurations,
        num_loops=args.n_loops,
        instruction_repeats=args.instruction_repeats,
        max_navigation_steps=args.max_navigation_steps,
        start_policy=args.start_position_policy,
        fixed_start=args.start_position,
    )
    agent = CorticallyEmbeddedRNN(
        environment,
        Nrec=args.Nrec,
        iters_per_action=args.iters_per_action,
        tau=args.tau,
        rec_noise=args.rec_noise,
        ent_reg=args.ent_reg,
        r_reg=args.r_reg,
        W_reg=args.W_reg,
        dist_reg=args.dist_reg,
        line_decay=args.line_decay,
        line_init_scale=args.line_init_scale,
        embedding_name=args.embedding_name,
        embedding_species=args.embedding_species,
        embedding_seed=args.embedding_seed,
        force_optimal=bool(args.force_optimal),
        greedy=bool(args.greedy),
        readout_mode="global",
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable.")
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    agent = agent.to(device)

    report = diagnose_abcd_loss_scaling(agent)
    if args.as_json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        print(format_loss_scaling_diagnostic(report))
    return report


if __name__ == "__main__":
    main()
