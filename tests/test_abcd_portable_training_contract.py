"""Focused checks for ABCD routing and portable analysis checkpoints."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

import pysta
from scripts.ABCD_task.abcd_analysis_common import (
    infer_portable_files,
    reconstruct_trained_model,
)


def _parsed_abcd(monkeypatch, **overrides):
    monkeypatch.setattr(sys, "argv", ["train"])
    defaults = {
        "task": "abcd_fmri",
        "batch_size": 1,
        "num_train_configurations": 1,
        "num_eval": 1,
    }
    defaults.update(overrides)
    return pysta.argparser.parse_args(**defaults)


def test_cli_default_n480_routes_both_spatial_groups_to_same_120_units(
    monkeypatch,
):
    embedding_dir = (
        Path(pysta.basedir)
        / "data"
        / "embedding"
        / "subsampled"
        / "human"
        / "mpfc_projected_mask_linear0p1"
        / "units=480_seed=42"
    )
    if not embedding_dir.is_dir():
        pytest.skip("The preserved N480 cortical embedding artifact is not installed.")

    kwargs = _parsed_abcd(monkeypatch)
    maze_kwargs = pysta.argparser.parse_args(task="maze")
    assert maze_kwargs["local_fraction"] == pytest.approx(1.0 / 6.0)
    assert kwargs["Nrec"] == 480
    assert kwargs["local_fraction"] == pytest.approx(0.25)

    environment = pysta.tasks.make_environment(kwargs, split="train")
    model = pysta.train_rnn._make_rnn(environment, kwargs)
    same_end = model.same_end_unit_mask.to(dtype=torch.bool)
    assert model._local_band_size() == 120
    assert int(same_end.sum()) == 120
    assert len(model.anchor_unit_indices) == 120

    anatomical_seed = torch.as_tensor(
        model.anatomical_anchor_unit_indices, dtype=torch.long
    )
    assert len(anatomical_seed) == 80
    assert torch.all(same_end[anatomical_seed])
    distance_to_seed = model.distance_matrix[:, anatomical_seed].min(dim=1).values
    expected_nearest = torch.argsort(distance_to_seed)[:120]
    assert torch.equal(
        torch.sort(torch.where(same_end)[0]).values,
        torch.sort(expected_nearest).values,
    )

    recipients = {}
    for group, indices in environment.obs_inds().items():
        group_mask = getattr(model, model.input_mask_buffers[group])
        indices = torch.as_tensor(indices, dtype=torch.long)
        recipients[group] = torch.any(group_mask[:, indices] != 0, dim=1)

    assert torch.equal(recipients["current_location"], same_end)
    assert torch.equal(recipients["instruction_location"], same_end)
    assert torch.equal(
        recipients["current_location"], recipients["instruction_location"]
    )
    for group in ("execution_rule", "phase", "reward_event"):
        assert int(recipients[group].sum()) == 480


def test_saved_abcd_best_checkpoint_has_exact_reconstructable_portable_artifacts(
    monkeypatch, tmp_path
):
    kwargs = _parsed_abcd(
        monkeypatch,
        model_type="vanilla",
        Nrec=6,
        batch_size=1,
        n_loops=1,
        instruction_repeats=1,
        max_navigation_steps=1,
        num_epochs=1,
        eval_freq=1,
        iters_per_action=1,
        tau=1.0,
        rec_noise=0.0,
        run_final_fmri_evaluation=False,
        prefix="portable_contract_",
        seed=13,
        save_results=True,
    )
    monkeypatch.setattr(pysta.utils, "basedir", str(tmp_path))

    pysta.train_rnn.main_train(kwargs)

    checkpoints = list((tmp_path / "models").rglob("*_best.pt"))
    assert len(checkpoints) == 1
    checkpoint = checkpoints[0]
    state_path, kwargs_path = infer_portable_files(checkpoint)
    assert state_path.name == checkpoint.name.replace(
        "_best.pt", "_best_portable_state_dict.pt"
    )
    assert kwargs_path.name == checkpoint.name.replace(
        "_best.pt", "_portable_kwargs.json"
    )

    reconstructed, loaded_kwargs = reconstruct_trained_model(state_path, kwargs_path)
    portable_state = torch.load(state_path, map_location="cpu", weights_only=True)
    whole_checkpoint = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    ).state_dict()
    assert loaded_kwargs["task"] == "abcd_fmri"
    assert loaded_kwargs["seed"] == 13
    assert loaded_kwargs["model_type"] == "vanilla"
    assert set(portable_state) == set(reconstructed.state_dict())
    for name, expected in portable_state.items():
        assert expected.device.type == "cpu"
        assert torch.equal(expected, whole_checkpoint[name].detach().cpu())
        assert torch.equal(expected, reconstructed.state_dict()[name].detach().cpu())
