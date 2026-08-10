"""Focused tests for the single-step ABCD backward sanity command."""

import math

from scripts.check_abcd_n480_backward import (
    build_parser,
    format_report,
    run_backward_check,
)


def test_production_defaults_encode_one_full_n480_training_step():
    args = build_parser().parse_args([])
    assert args.model_type == "corticallyembedded"
    assert args.Nrec == 480
    assert args.batch_size == 1
    assert args.n_loops == 5
    assert args.instruction_repeats == 2
    assert args.iters_per_action == 10
    assert args.tau == 5.0
    assert args.force_optimal == 1
    assert args.learning_rate == 3e-4


def test_tiny_full_block_runs_exactly_one_finite_optimizer_step():
    args = build_parser().parse_args(
        [
            "--model-type",
            "lineembedded",
            "--Nrec",
            "8",
            "--batch-size",
            "2",
            "--n-loops",
            "1",
            "--instruction-repeats",
            "2",
            "--iters-per-action",
            "1",
            "--tau",
            "2",
            "--rec-noise",
            "0",
            "--configurations",
            "0,2,8,6",
            "--start-position-policy",
            "fixed",
            "--start-position",
            "4",
            "--device",
            "cpu",
        ]
    )
    report = run_backward_check(args)

    assert report["status"] == "passed"
    assert report["optimizer_steps"] == 1
    assert report["task"]["full_block_verified"] is True
    assert report["task"]["finished_per_row"] == [True, True]
    assert report["task"]["truncated_per_row"] == [False, False]
    assert report["task"]["successful_goal_count_per_row"] == [4, 4]
    assert report["task"]["instruction_steps_per_row"] == 8
    assert report["task"]["reward_dwell_steps_per_row"] == 4
    assert report["task"]["navigation_steps_per_row"] == [8, 8]
    assert report["task"]["block_timesteps_per_row"] == [20, 20]
    assert report["task"]["recurrent_microsteps_per_row"] == [20, 20]

    assert report["loss"]["finite"] is True
    assert math.isfinite(report["loss"]["mean_per_block"])
    assert report["gradients"]["all_trainable_gradients_present"] is True
    assert report["gradients"]["all_present_gradients_finite"] is True
    assert report["gradients"]["missing_gradient_parameters"] == []
    assert report["gradients"]["nonfinite_gradient_parameters"] == []
    assert report["gradients"]["gradient_tensors_present"] > 0
    assert report["gradients"]["global_l2_norm"] > 0
    assert report["optimizer"]["all_parameters_finite_after_step"] is True
    assert report["optimizer"]["changed_parameter_tensors"] > 0
    assert report["memory"]["process_peak_rss_after"]["available"] is True

    rendered = format_report(report)
    assert "optimizer steps: 1" in rendered
    assert "gradients: finite=True" in rendered
    assert "process peak RSS" in rendered
