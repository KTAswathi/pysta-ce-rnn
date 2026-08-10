#!/usr/bin/env python3
"""Run exactly one full-block ABCD forward/backward/optimizer sanity step.

The defaults mirror the intended cortical N480 training setup.  This command
is deliberately a one-step diagnostic: there is no epoch loop and no model is
saved.  A nonzero exit is raised before (or immediately after) the optimizer
step if the block, loss, gradients, or updated parameters are invalid.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pysta.abcd_env import (
    ABCDFMRIEnv,
    EXECUTION_RELATION_NAMES,
    INSTRUCTION_DIRECTION_NAMES,
    generate_configuration_bank,
    parse_configurations,
)
from pysta.agents import CorticallyEmbeddedRNN, LineEmbeddedRNN, VanillaRNN


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one, and only one, complete ABCD block through forward(), "
            "loss.backward(), and Adam.step(), with numerical and memory checks."
        )
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-loops", type=int, default=5)
    parser.add_argument("--instruction-repeats", type=int, default=2)
    parser.add_argument("--max-navigation-steps", type=int, default=200)
    parser.add_argument("--num-configurations", type=int, default=12)
    parser.add_argument("--configuration-seed", type=int, default=0)
    parser.add_argument(
        "--configurations",
        default=None,
        help="optional semicolon-separated configurations, e.g. '0,2,8,6'",
    )
    parser.add_argument(
        "--start-position-policy",
        choices=("exclude_first_goal", "uniform", "fixed"),
        default="exclude_first_goal",
    )
    parser.add_argument("--start-position", type=int, default=None)

    parser.add_argument(
        "--model-type",
        choices=("corticallyembedded", "lineembedded", "vanilla"),
        default="corticallyembedded",
        help="corticallyembedded is the production default; smaller alternatives aid tests",
    )
    parser.add_argument("--Nrec", "--nrec", dest="Nrec", type=int, default=480)
    parser.add_argument("--iters-per-action", type=int, default=10)
    parser.add_argument("--tau", type=float, default=5.0)
    parser.add_argument("--rec-noise", type=float, default=1e-3)
    parser.add_argument("--ent-reg", type=float, default=1e-4)
    parser.add_argument("--r-reg", type=float, default=1e-5)
    parser.add_argument("--W-reg", dest="W_reg", type=float, default=2e-7)
    parser.add_argument("--dist-reg", type=float, default=1e-7)
    parser.add_argument("--line-decay", type=float, default=0.12)
    parser.add_argument("--line-init-scale", type=float, default=1.0)
    parser.add_argument("--use-local-init", type=int, choices=(0, 1), default=1)
    parser.add_argument("--embedding-name", default="mpfc_projected_mask_linear0p1")
    parser.add_argument("--embedding-species", default="human")
    parser.add_argument("--embedding-seed", type=int, default=42)
    parser.add_argument("--force-optimal", type=int, choices=(0, 1), default=1)
    parser.add_argument("--greedy", type=int, choices=(0, 1), default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _process_peak_rss() -> dict[str, Any]:
    """Return resource.ru_maxrss with its platform-dependent units normalized."""
    try:
        import resource
    except ImportError:  # pragma: no cover - resource is available on Unix CI/HPC
        return {
            "available": False,
            "source": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
            "reason": "Python resource module unavailable",
        }

    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # POSIX does not standardize ru_maxrss units.  Darwin reports bytes; Linux
    # (including ordinary Slurm nodes) reports KiB.
    if sys.platform == "darwin":
        raw_unit = "bytes"
        peak_bytes = int(raw)
    elif sys.platform.startswith("linux"):
        raw_unit = "KiB"
        peak_bytes = int(raw * 1024.0)
    else:  # retain the raw value rather than guessing on uncommon platforms
        return {
            "available": True,
            "source": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
            "raw_value": raw,
            "raw_unit": "platform-dependent",
            "peak_rss_bytes": None,
            "peak_rss_mib": None,
        }
    return {
        "available": True,
        "source": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
        "raw_value": raw,
        "raw_unit": raw_unit,
        "peak_rss_bytes": peak_bytes,
        "peak_rss_mib": peak_bytes / (1024.0**2),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_environment(args: argparse.Namespace) -> ABCDFMRIEnv:
    configurations = parse_configurations(args.configurations)
    if not configurations:
        configurations = generate_configuration_bank(
            num_configurations=args.num_configurations,
            seed=args.configuration_seed,
            min_manhattan_distance=2,
            prefer_all_pairs=True,
            unique_up_to_cycle=True,
        )
    return ABCDFMRIEnv(
        batch_size=args.batch_size,
        seed=args.seed,
        configuration_bank=configurations,
        num_loops=args.n_loops,
        instruction_repeats=args.instruction_repeats,
        max_navigation_steps=args.max_navigation_steps,
        start_policy=args.start_position_policy,
        fixed_start=args.start_position,
    )


def _make_agent(args: argparse.Namespace, environment: ABCDFMRIEnv):
    common = dict(
        Nrec=args.Nrec,
        iters_per_action=args.iters_per_action,
        tau=args.tau,
        rec_noise=args.rec_noise,
        ent_reg=args.ent_reg,
        r_reg=args.r_reg,
        W_reg=args.W_reg,
        force_optimal=bool(args.force_optimal),
        greedy=bool(args.greedy),
    )
    if args.model_type == "vanilla":
        return VanillaRNN(environment, **common)

    embedded = dict(
        dist_reg=args.dist_reg,
        line_decay=args.line_decay,
        line_init_scale=args.line_init_scale,
        use_local_init=bool(args.use_local_init),
        readout_mode="global",
    )
    if args.model_type == "lineembedded":
        return LineEmbeddedRNN(environment, **common, **embedded)
    return CorticallyEmbeddedRNN(
        environment,
        **common,
        **embedded,
        embedding_name=args.embedding_name,
        embedding_species=args.embedding_species,
        embedding_seed=args.embedding_seed,
    )


def _loss_components(agent) -> dict[str, float]:
    batch = float(agent.env.batch)
    return {
        "policy_mean_per_block": float(agent.acc_loss.detach().cpu()) / batch,
        "entropy_mean_per_block": float(agent.ent_loss.detach().cpu()) / batch,
        "parameter_mean_per_block": float(agent.weight_loss.detach().cpu()) / batch,
        "firing_rate_mean_per_block": float(agent.rate_loss.detach().cpu()) / batch,
    }


def _gradient_report(agent) -> dict[str, Any]:
    by_parameter: dict[str, Any] = {}
    missing: list[str] = []
    nonfinite: list[str] = []
    total_squared_norm = 0.0
    global_max_abs = 0.0
    present_elements = 0
    trainable_elements = 0
    trainable_tensors = 0

    for name, parameter in agent.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable_tensors += 1
        trainable_elements += parameter.numel()
        gradient = parameter.grad
        if gradient is None:
            missing.append(name)
            continue
        finite = bool(torch.isfinite(gradient).all().detach().cpu())
        if not finite:
            nonfinite.append(name)
        grad64 = gradient.detach().to(dtype=torch.float64)
        squared_norm = float(torch.sum(grad64 * grad64).cpu())
        max_abs = (
            float(torch.max(torch.abs(gradient)).cpu())
            if gradient.numel()
            else 0.0
        )
        total_squared_norm += squared_norm
        global_max_abs = max(global_max_abs, max_abs)
        present_elements += gradient.numel()
        by_parameter[name] = {
            "shape": list(parameter.shape),
            "elements": parameter.numel(),
            "finite": finite,
            "l2_norm": math.sqrt(squared_norm),
            "max_abs": max_abs,
        }

    return {
        "trainable_parameter_tensors": trainable_tensors,
        "trainable_parameter_elements": trainable_elements,
        "gradient_tensors_present": len(by_parameter),
        "gradient_elements_present": present_elements,
        "missing_gradient_parameters": missing,
        "nonfinite_gradient_parameters": nonfinite,
        "all_trainable_gradients_present": not missing,
        "all_present_gradients_finite": not nonfinite,
        "global_l2_norm": math.sqrt(total_squared_norm),
        "global_max_abs": global_max_abs,
        "by_parameter": by_parameter,
    }


def _parameter_report(agent, before: dict[str, torch.Tensor]) -> dict[str, Any]:
    nonfinite: list[str] = []
    changed: list[str] = []
    max_abs_update = 0.0
    for name, parameter in agent.named_parameters():
        value = parameter.detach()
        if not bool(torch.isfinite(value).all().cpu()):
            nonfinite.append(name)
        difference = value - before[name]
        this_max = (
            float(torch.max(torch.abs(difference)).cpu())
            if difference.numel()
            else 0.0
        )
        if this_max > 0.0:
            changed.append(name)
        max_abs_update = max(max_abs_update, this_max)
    return {
        "all_parameters_finite_after_step": not nonfinite,
        "nonfinite_parameters_after_step": nonfinite,
        "changed_parameter_tensors": len(changed),
        "changed_parameter_names": changed,
        "max_abs_parameter_update": max_abs_update,
    }


def _verify_complete_block(environment: ABCDFMRIEnv) -> dict[str, Any]:
    expected_rewards = environment.num_loops * 4
    finished = environment.finished.detach().cpu()
    truncated = environment.truncated.detach().cpu()
    successes = environment.successful_goal_count.detach().cpu()
    navigation_steps = environment.navigation_step_count.detach().cpu()
    block_timesteps = environment.block_timestep.detach().cpu()
    expected_timesteps = (
        environment.total_instruction_steps + navigation_steps + expected_rewards
    )
    failures = []
    if not bool(torch.all(finished)):
        failures.append("not every batch row finished")
    if bool(torch.any(truncated)):
        failures.append("at least one batch row was truncated")
    if not bool(torch.all(successes == expected_rewards)):
        failures.append("not every batch row reached every required goal")
    if not bool(torch.all(block_timesteps == expected_timesteps)):
        failures.append("phase counts do not describe instruction + movement + reward dwell")
    if failures:
        raise RuntimeError("Incomplete full-block forward pass: " + "; ".join(failures))

    return {
        "full_block_verified": True,
        "batch_size": environment.batch,
        "num_loops": environment.num_loops,
        "instruction_repeats": environment.instruction_repeats,
        "instruction_steps_per_row": environment.total_instruction_steps,
        "required_rewards_per_row": expected_rewards,
        "successful_goal_count_per_row": successes.tolist(),
        "navigation_steps_per_row": navigation_steps.tolist(),
        "reward_dwell_steps_per_row": expected_rewards,
        "block_timesteps_per_row": block_timesteps.tolist(),
        "finished_per_row": finished.tolist(),
        "truncated_per_row": truncated.tolist(),
    }


def run_backward_check(args: argparse.Namespace) -> dict[str, Any]:
    """Execute exactly one optimizer step and return a JSON-serializable report."""
    if args.batch_size < 1 or args.n_loops < 1 or args.instruction_repeats < 1:
        raise ValueError("batch size, loops, and instruction repeats must be positive")
    if args.iters_per_action < 1 or args.Nrec < 1:
        raise ValueError("Nrec and iters per action must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _select_device(args.device)
    rss_before = _process_peak_rss()

    environment = _make_environment(args)
    agent = _make_agent(args, environment).to(device)
    agent.train()
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate)
    optimizer_steps = 0

    before = {
        name: parameter.detach().clone()
        for name, parameter in agent.named_parameters()
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    total_start = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)
    forward_start = time.perf_counter()
    loss = agent.forward()
    _synchronize(device)
    forward_seconds = time.perf_counter() - forward_start
    if loss.ndim != 0 or not bool(torch.isfinite(loss).detach().cpu()):
        raise FloatingPointError(f"Forward loss must be one finite scalar, got {loss}.")
    sequence = _verify_complete_block(environment)
    # Now that the actual timestep count is known, report the exact recurrent
    # unroll length.  The production defaults use a fixed integer microstep count.
    sequence["recurrent_microsteps_per_row"] = [
        int(value) * int(args.iters_per_action)
        for value in sequence["block_timesteps_per_row"]
    ]
    components = _loss_components(agent)

    backward_start = time.perf_counter()
    loss.backward()
    _synchronize(device)
    backward_seconds = time.perf_counter() - backward_start
    gradients = _gradient_report(agent)
    if not gradients["all_trainable_gradients_present"]:
        raise RuntimeError(
            "Trainable parameters without gradients: "
            + ", ".join(gradients["missing_gradient_parameters"])
        )
    if not gradients["all_present_gradients_finite"]:
        raise FloatingPointError(
            "Non-finite gradients: "
            + ", ".join(gradients["nonfinite_gradient_parameters"])
        )

    optimizer_start = time.perf_counter()
    optimizer.step()
    optimizer_steps += 1
    _synchronize(device)
    optimizer_seconds = time.perf_counter() - optimizer_start
    parameters = _parameter_report(agent, before)
    if not parameters["all_parameters_finite_after_step"]:
        raise FloatingPointError(
            "Non-finite parameters after optimizer step: "
            + ", ".join(parameters["nonfinite_parameters_after_step"])
        )
    if optimizer_steps != 1:  # defensive assertion against future refactors
        raise AssertionError(
            f"Expected exactly one optimizer step, got {optimizer_steps}."
        )

    _synchronize(device)
    total_seconds = time.perf_counter() - total_start
    rss_after = _process_peak_rss()
    if device.type == "cuda":
        cuda_memory: dict[str, Any] | None = {
            "source": "torch.cuda peak memory stats after model placement",
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024.0**2),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024.0**2),
        }
    else:
        cuda_memory = None

    return {
        "status": "passed",
        "optimizer_steps": optimizer_steps,
        "device": str(device),
        "platform": platform.platform(),
        "seed": args.seed,
        "model": {
            "type": args.model_type,
            "recurrent_units": args.Nrec,
            "input_dim": environment.obs_dim,
            "output_dim": environment.output_dim,
            "iters_per_action": args.iters_per_action,
            "tau": args.tau,
            "force_optimal": bool(args.force_optimal),
            "recurrent_noise": args.rec_noise,
        },
        "task": sequence,
        "sampled_block": {
            "configuration_bank_size": len(environment.configuration_bank),
            "configurations": environment.configuration.detach().cpu().tolist(),
            "instruction_directions": [
                INSTRUCTION_DIRECTION_NAMES[int(value)]
                for value in environment.instruction_direction
            ],
            "execution_relations": [
                EXECUTION_RELATION_NAMES[int(value)]
                for value in environment.execution_relation
            ],
            "start_locations": environment.start_location.detach().cpu().tolist(),
        },
        "loss": {
            "mean_per_block": float(loss.detach().cpu()),
            "finite": True,
            "components": components,
        },
        "gradients": gradients,
        "optimizer": {
            "type": "Adam",
            "learning_rate": args.learning_rate,
            **parameters,
        },
        "timing_seconds": {
            "forward": forward_seconds,
            "backward": backward_seconds,
            "optimizer": optimizer_seconds,
            "total_step": total_seconds,
        },
        "memory": {
            "process_peak_rss_before": rss_before,
            "process_peak_rss_after": rss_after,
            "cuda": cuda_memory,
            "note": (
                "ru_maxrss is a process-lifetime high-water mark; its before/after "
                "difference is not an isolated allocation measurement"
            ),
        },
    }


def format_report(report: dict[str, Any]) -> str:
    task = report["task"]
    gradients = report["gradients"]
    memory = report["memory"]
    lines = [
        "ABCD single full-block backward sanity check",
        f"status: {report['status']} (optimizer steps: {report['optimizer_steps']})",
        (
            f"model/device: {report['model']['type']} N{report['model']['recurrent_units']} "
            f"batch={task['batch_size']} on {report['device']}"
        ),
        (
            f"sequence: instruction={task['instruction_steps_per_row']} "
            f"rewards/dwells={task['required_rewards_per_row']} "
            f"navigation={task['navigation_steps_per_row']} "
            f"timesteps={task['block_timesteps_per_row']} "
            f"microsteps={task['recurrent_microsteps_per_row']}"
        ),
        f"loss: {report['loss']['mean_per_block']:.9g} (finite=True)",
        (
            f"gradients: finite={gradients['all_present_gradients_finite']} "
            f"present={gradients['gradient_tensors_present']}/"
            f"{gradients['trainable_parameter_tensors']} tensors "
            f"global_l2={gradients['global_l2_norm']:.9g} "
            f"max_abs={gradients['global_max_abs']:.9g}"
        ),
        (
            "updated parameters: "
            f"finite={report['optimizer']['all_parameters_finite_after_step']} "
            f"changed={report['optimizer']['changed_parameter_tensors']} tensors"
        ),
        (
            "timing (s): "
            f"forward={report['timing_seconds']['forward']:.3f}, "
            f"backward={report['timing_seconds']['backward']:.3f}, "
            f"optimizer={report['timing_seconds']['optimizer']:.3f}"
        ),
    ]
    if memory["cuda"] is not None:
        lines.append(
            "CUDA peak: "
            f"allocated={memory['cuda']['peak_allocated_mib']:.2f} MiB, "
            f"reserved={memory['cuda']['peak_reserved_mib']:.2f} MiB"
        )
    else:
        rss = memory["process_peak_rss_after"]
        if rss.get("peak_rss_mib") is not None:
            lines.append(
                f"process peak RSS: {rss['peak_rss_mib']:.2f} MiB "
                f"(raw ru_maxrss unit: {rss['raw_unit']})"
            )
        else:
            lines.append("process peak RSS: unavailable or platform-dependent")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    report = run_backward_check(args)
    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_report(report))
    return report


if __name__ == "__main__":
    main()
