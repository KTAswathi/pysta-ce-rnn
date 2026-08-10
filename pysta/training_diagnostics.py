"""Forward-only diagnostics for ABCD sequence-loss scaling.

This module deliberately reports the objective exactly as it is currently
implemented.  It does not rescale any term, build an optimiser, call
``backward()``, or recommend replacement coefficients.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import torch


LOSS_COMPONENTS = (
    "policy",
    "entropy",
    "firing_rate",
    "l2_parameter",
    "cortical_distance",
)


@dataclass(frozen=True)
class LossComponentDiagnostic:
    """One existing loss term under several diagnostic normalisations."""

    raw_batch_sum: float
    mean_per_block: float
    per_policy_active_step: float | None
    signed_ratio_to_policy: float | None
    magnitude_ratio_to_policy: float | None
    share_of_component_magnitude: float | None


@dataclass(frozen=True)
class LossScalingDiagnostic:
    """Machine-readable result of one no-gradient full-block forward pass."""

    model_class: str
    recurrent_units: int
    batch_size: int
    coefficients: Mapping[str, float]
    counts: Mapping[str, int | float]
    components: Mapping[str, LossComponentDiagnostic]
    component_raw_batch_sum: float
    component_mean_per_block: float
    forward_returned_loss: float
    parameter_regularizer_residual: float
    normalization_note: str

    def as_dict(self) -> dict[str, Any]:
        """Return a recursively JSON-serialisable representation."""
        return asdict(self)


def _scalar(value: Any) -> float:
    """Convert a scalar tensor/NumPy value to a Python float."""
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(
                f"Expected a scalar tensor, got shape {tuple(value.shape)}."
            )
        return float(value.detach().cpu())
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected a scalar value, got shape {array.shape}.")
    return float(array.reshape(-1)[0])


def _batch_bool_count(value: Any) -> int:
    """Count true entries in a stored per-batch mask."""
    if torch.is_tensor(value):
        return int(value.detach().to(dtype=torch.bool).sum().cpu())
    return int(np.asarray(value, dtype=bool).sum())


def _split_parameter_regularizer(agent) -> tuple[float, float, float]:
    """Return raw-batch L2, cortical-distance, and unexplained residual terms.

    ``BaseAgent.reset`` multiplies ``calc_parameter_reg()`` by batch size.  We
    apply that same multiplier to each constituent here, then compare their
    sum with the cached ``weight_loss``.  The residual is normally only
    floating-point roundoff for Vanilla/Line/CorticallyEmbeddedRNN; exposing
    it makes the diagnostic fail visibly rather than silently mislabelling a
    future custom regularizer.
    """
    batch_size = int(agent.env.batch)
    device = next(agent.parameters()).device
    dtype = next(agent.parameters()).dtype

    coefficient = float(getattr(agent, "W_reg", 0.0))
    if coefficient:
        l2_base = torch.stack(
            [torch.square(parameter).sum() for parameter in agent.parameters()]
        ).sum()
        l2 = coefficient * l2_base
    else:
        l2 = torch.zeros((), dtype=dtype, device=device)

    distance_coefficient = float(getattr(agent, "dist_reg", 0.0))
    if distance_coefficient and hasattr(agent, "distance_matrix"):
        distance = distance_coefficient * (
            torch.abs(agent.Wrec) * agent.distance_matrix
        ).sum()
    else:
        distance = torch.zeros((), dtype=dtype, device=device)

    raw_l2 = batch_size * _scalar(l2)
    raw_distance = batch_size * _scalar(distance)
    cached_parameter = _scalar(agent.weight_loss)
    residual = cached_parameter - raw_l2 - raw_distance
    return raw_l2, raw_distance, residual


def _trajectory_counts(agent) -> dict[str, int | float]:
    """Count batch calls, active task rows, phases, and recurrent microsteps."""
    records = agent.store
    batch_size = int(agent.env.batch)

    environment_row_timesteps = 0
    policy_active_steps = 0
    phase_counts = {0: 0, 1: 0, 2: 0}
    for record in records:
        valid = record.get("valid_timestep")
        if valid is None:
            valid = ~torch.as_tensor(record["finished"], dtype=torch.bool)
        environment_row_timesteps += _batch_bool_count(valid)
        policy_active_steps += _batch_bool_count(
            record.get("policy_loss_mask", record["loss_mask"])
        )

        if "phase" in record:
            phase_value = record["phase"]
            if torch.is_tensor(phase_value):
                phase_value = phase_value.detach().cpu()
            phase = np.asarray(phase_value)
            valid_value = valid.detach().cpu() if torch.is_tensor(valid) else valid
            valid_array = np.asarray(valid_value, dtype=bool)
            for phase_index in phase_counts:
                phase_counts[phase_index] += int(
                    np.logical_and(valid_array, phase == phase_index).sum()
                )

    microstep_metadata = getattr(agent, "all_acts_metadata", [])
    recurrent_microstep_calls = len(
        getattr(agent, "all_acts", [[], [], []])[0]
    )
    if len(microstep_metadata) != recurrent_microstep_calls:
        raise RuntimeError(
            "Exact recurrent-microstep counting requires aligned all_acts_metadata; "
            f"got {recurrent_microstep_calls} activity rows and "
            f"{len(microstep_metadata)} metadata rows."
        )

    active_recurrent_row_microsteps = 0
    for metadata in microstep_metadata:
        if metadata is None:
            active_recurrent_row_microsteps += batch_size
            continue
        valid = metadata.get("valid_timestep")
        if valid is None:
            finished = metadata.get("finished")
            valid = (
                np.ones(batch_size, dtype=bool)
                if finished is None
                else ~np.asarray(finished, dtype=bool)
            )
        active_recurrent_row_microsteps += _batch_bool_count(valid)

    return {
        "batched_environment_timestep_calls": int(len(records)),
        "active_environment_row_timesteps": int(environment_row_timesteps),
        "mean_environment_timesteps_per_block": (
            float(environment_row_timesteps) / batch_size
        ),
        "instruction_row_timesteps": int(phase_counts[0]),
        "navigation_row_timesteps": int(phase_counts[1]),
        "reward_row_timesteps": int(phase_counts[2]),
        "policy_active_navigation_steps": int(policy_active_steps),
        "mean_policy_active_steps_per_block": (
            float(policy_active_steps) / batch_size
        ),
        "recurrent_microstep_calls": int(recurrent_microstep_calls),
        "executed_recurrent_row_microsteps": int(
            recurrent_microstep_calls * batch_size
        ),
        "active_recurrent_row_microsteps": int(active_recurrent_row_microsteps),
        "mean_active_recurrent_microsteps_per_block": (
            float(active_recurrent_row_microsteps) / batch_size
        ),
        # VanillaRNN.reset() evaluates calc_activity_reg() once on the initial
        # recurrent state before any environment/recurrent step, then adds one
        # evaluation at every active recurrent microstep.
        "initial_state_firing_rate_evaluation_rows": int(batch_size),
        "total_firing_rate_evaluation_rows": int(
            active_recurrent_row_microsteps + batch_size
        ),
    }


def diagnose_abcd_loss_scaling(agent) -> LossScalingDiagnostic:
    """Run one complete ABCD block batch and report current objective scaling.

    The call performs one ordinary ``agent.forward(store=True)`` under
    ``torch.no_grad()``.  Thus it resets and advances the agent/environment in
    the normal way, but does not alter parameters, gradients, coefficients, or
    optimiser state.  ``store_all_activity`` is enabled temporarily so counts
    remain exact even when ``iters_per_action`` is sampled separately at each
    environment timestep.
    """
    environment = agent.env
    required_environment_attributes = (
        "policy_loss_mask",
        "phase",
        "block_timestep",
    )
    missing = [
        name
        for name in required_environment_attributes
        if not hasattr(environment, name)
    ]
    if missing:
        raise TypeError(
            "diagnose_abcd_loss_scaling requires an ABCD-compatible environment; "
            f"missing {missing}."
        )

    previous_store_all_activity = bool(agent.store_all_activity)
    agent.store_all_activity = True
    try:
        with torch.no_grad():
            forward_loss = agent.forward(store=True)
            raw_l2, raw_distance, parameter_residual = (
                _split_parameter_regularizer(agent)
            )
    finally:
        agent.store_all_activity = previous_store_all_activity

    counts = _trajectory_counts(agent)
    batch_size = int(environment.batch)
    raw_components = {
        "policy": _scalar(agent.acc_loss),
        "entropy": _scalar(agent.ent_loss),
        "firing_rate": _scalar(agent.rate_loss),
        "l2_parameter": raw_l2,
        "cortical_distance": raw_distance,
    }

    policy = raw_components["policy"]
    total_magnitude = sum(abs(value) for value in raw_components.values())
    active_policy_steps = int(counts["policy_active_navigation_steps"])
    components: dict[str, LossComponentDiagnostic] = {}
    for name in LOSS_COMPONENTS:
        raw_value = raw_components[name]
        components[name] = LossComponentDiagnostic(
            raw_batch_sum=raw_value,
            mean_per_block=raw_value / batch_size,
            per_policy_active_step=(
                raw_value / active_policy_steps if active_policy_steps else None
            ),
            signed_ratio_to_policy=(raw_value / policy if policy else None),
            magnitude_ratio_to_policy=(
                abs(raw_value) / abs(policy) if policy else None
            ),
            share_of_component_magnitude=(
                abs(raw_value) / total_magnitude if total_magnitude else None
            ),
        )

    raw_total = sum(raw_components.values())
    returned = _scalar(forward_loss)
    # The residual is reported separately because the five named components
    # are meant to remain semantically exact. For current RNNs it should be
    # numerical roundoff only.
    expected_returned = (raw_total + parameter_residual) / batch_size
    if not np.isclose(returned, expected_returned, rtol=1e-5, atol=1e-7):
        raise RuntimeError(
            "Separated loss components do not reconstruct forward() loss: "
            f"reported={returned:.9g}, reconstructed={expected_returned:.9g}."
        )

    coefficients = {
        "entropy": float(getattr(agent, "ent_reg", 0.0)),
        "firing_rate": float(getattr(agent, "r_reg", 0.0)),
        "l2_parameter": float(getattr(agent, "W_reg", 0.0)),
        "cortical_distance": float(getattr(agent, "dist_reg", 0.0)),
    }
    return LossScalingDiagnostic(
        model_class=type(agent).__name__,
        recurrent_units=int(getattr(agent, "Nrec", 0)),
        batch_size=batch_size,
        coefficients=coefficients,
        counts=counts,
        components=components,
        component_raw_batch_sum=raw_total,
        component_mean_per_block=raw_total / batch_size,
        forward_returned_loss=returned,
        parameter_regularizer_residual=parameter_residual,
        normalization_note=(
            "Policy, entropy, and firing-rate losses accumulate over their active "
            "sequence steps; L2 and cortical-distance penalties are evaluated once "
            "per block and multiplied by batch size. BaseAgent.forward divides the "
            "combined raw batch sum by batch size only."
        ),
    )


def format_loss_scaling_diagnostic(report: LossScalingDiagnostic) -> str:
    """Format a compact human-readable diagnostic table."""
    lines = [
        "ABCD loss-scaling diagnostic (forward only; no backward/optimizer)",
        (
            f"model={report.model_class}, N={report.recurrent_units}, "
            f"batch={report.batch_size}"
        ),
        "",
        "Sequence counts",
    ]
    for name, value in report.counts.items():
        lines.append(f"  {name}: {value}")

    lines.extend(
        [
            "",
            "Loss components",
            (
                "  component                 raw batch sum      mean/block   "
                "|term|/policy  magnitude share"
            ),
        ]
    )
    for name in LOSS_COMPONENTS:
        component = report.components[name]
        ratio = component.magnitude_ratio_to_policy
        share = component.share_of_component_magnitude
        ratio_text = "n/a" if ratio is None else f"{ratio:.6g}"
        share_text = "n/a" if share is None else f"{share:.2%}"
        lines.append(
            f"  {name:<24} {component.raw_batch_sum:>14.7g} "
            f"{component.mean_per_block:>15.7g} {ratio_text:>14} "
            f"{share_text:>16}"
        )

    lines.extend(
        [
            "",
            f"component mean/block: {report.component_mean_per_block:.9g}",
            f"forward returned loss: {report.forward_returned_loss:.9g}",
            (
                "parameter split residual: "
                f"{report.parameter_regularizer_residual:.9g}"
            ),
            report.normalization_note,
        ]
    )
    return "\n".join(lines)


__all__ = [
    "LOSS_COMPONENTS",
    "LossComponentDiagnostic",
    "LossScalingDiagnostic",
    "diagnose_abcd_loss_scaling",
    "format_loss_scaling_diagnostic",
]
