"""Focused tests for the forward-only ABCD loss-scaling diagnostic."""

import numpy as np
import pytest
import torch

from pysta.abcd_env import ABCDFMRIEnv, FORWARD, SAME
from pysta.agents import LineEmbeddedRNN
from pysta.training_diagnostics import (
    LOSS_COMPONENTS,
    diagnose_abcd_loss_scaling,
    format_loss_scaling_diagnostic,
)


def make_agent(**overrides):
    env = ABCDFMRIEnv(
        batch_size=2,
        seed=3,
        configuration_bank=[(0, 2, 8, 6)],
        instruction_directions=[FORWARD],
        execution_relations=[SAME],
        num_loops=1,
        instruction_repeats=2,
        max_navigation_steps=50,
        start_policy="fixed",
        fixed_start=4,
    )
    kwargs = dict(
        Nrec=8,
        W_reg=2e-4,
        r_reg=3e-4,
        dist_reg=5e-5,
        ent_reg=7e-4,
        rec_noise=0.0,
        iters_per_action=2,
        tau=2,
        force_optimal=True,
    )
    kwargs.update(overrides)
    return LineEmbeddedRNN(env, **kwargs)


def test_diagnostic_separates_existing_components_and_exact_sequence_counts():
    torch.manual_seed(3)
    np.random.seed(3)
    agent = make_agent()
    report = diagnose_abcd_loss_scaling(agent)

    assert tuple(report.components) == LOSS_COMPONENTS
    assert report.batch_size == 2
    assert report.recurrent_units == 8

    # Per row: 8 instructions + 8 shortest movements + 4 reward dwells.
    assert report.counts["batched_environment_timestep_calls"] == 20
    assert report.counts["active_environment_row_timesteps"] == 40
    assert report.counts["instruction_row_timesteps"] == 16
    assert report.counts["navigation_row_timesteps"] == 16
    assert report.counts["reward_row_timesteps"] == 8
    assert report.counts["policy_active_navigation_steps"] == 16
    assert report.counts["recurrent_microstep_calls"] == 40
    assert report.counts["active_recurrent_row_microsteps"] == 80
    assert report.counts["initial_state_firing_rate_evaluation_rows"] == 2
    assert report.counts["total_firing_rate_evaluation_rows"] == 82

    assert report.components["policy"].raw_batch_sum == pytest.approx(
        float(agent.acc_loss)
    )
    assert report.components["entropy"].raw_batch_sum == pytest.approx(
        float(agent.ent_loss)
    )
    assert report.components["firing_rate"].raw_batch_sum == pytest.approx(
        float(agent.rate_loss)
    )
    assert report.components["l2_parameter"].raw_batch_sum > 0
    assert report.components["cortical_distance"].raw_batch_sum > 0
    assert report.parameter_regularizer_residual == pytest.approx(0.0, abs=2e-6)
    assert report.component_mean_per_block == pytest.approx(
        report.forward_returned_loss, rel=2e-6
    )
    assert report.components["policy"].magnitude_ratio_to_policy == pytest.approx(
        1.0
    )
    assert sum(
        component.share_of_component_magnitude
        for component in report.components.values()
    ) == pytest.approx(1.0)

    rendered = format_loss_scaling_diagnostic(report)
    assert "no backward/optimizer" in rendered
    assert "cortical_distance" in rendered
    assert "divides the combined raw batch sum by batch size only" in rendered
    assert report.as_dict()["counts"]["policy_active_navigation_steps"] == 16


def test_diagnostic_does_not_change_parameters_coefficients_or_gradients():
    torch.manual_seed(11)
    np.random.seed(11)
    agent = make_agent(iters_per_action=[1, 3])
    original_parameters = [parameter.detach().clone() for parameter in agent.parameters()]
    original_coefficients = (agent.W_reg, agent.r_reg, agent.ent_reg, agent.dist_reg)
    agent.store_all_activity = False

    report = diagnose_abcd_loss_scaling(agent)

    assert agent.store_all_activity is False
    assert (agent.W_reg, agent.r_reg, agent.ent_reg, agent.dist_reg) == (
        original_coefficients
    )
    for before, after in zip(original_parameters, agent.parameters()):
        assert torch.equal(before, after.detach())
        assert after.grad is None

    # A sampled iters_per_action list is counted from actual microstep records,
    # not approximated from a presumed fixed number.
    assert report.counts["recurrent_microstep_calls"] == len(agent.all_acts[0])
    assert report.counts["recurrent_microstep_calls"] in range(20, 61)


def test_diagnostic_rejects_non_abcd_environment():
    class NotABCD:
        pass

    agent = make_agent()
    agent.env = NotABCD()
    with pytest.raises(TypeError, match="ABCD-compatible"):
        diagnose_abcd_loss_scaling(agent)
