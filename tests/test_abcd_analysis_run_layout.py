"""Managed-run and legacy compatibility for the ABCD analysis entry points."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import pysta
from pysta import run_manager
from scripts.ABCD_task.abcd_analysis_common import (
    infer_portable_files,
    load_training_config,
    reconstruct_trained_model,
    resolve_analysis_root,
    resolve_checkpoint_identifier,
    resolve_existing_analysis_root,
)
from scripts.ABCD_task.analyse_abcd_normalized_raw import (
    discover_normalized_inputs,
)


def _managed_run(tmp_path: Path, name: str = "mechanism_baseline_abc123def0"):
    run_dir = tmp_path / "models" / "abcd_fmri" / name
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    document = {
        "schema": "pysta_managed_run/v1",
        "config_hash": {"full": "a" * 64, "short": "a" * 10},
        "training_config": {
            "arguments": {"task": "abcd_fmri", "seed": 17},
        },
    }
    (run_dir / "config.yaml").write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf8"
    )
    return run_dir, document


def test_managed_run_checkpoint_and_output_resolution(monkeypatch, tmp_path):
    run_dir, document = _managed_run(tmp_path)
    best = run_dir / "checkpoints" / "best.pt"
    latest = run_dir / "checkpoints" / "latest.pt"
    torch.save({"model_state_dict": {"weight": torch.ones(1)}, "config_hash": "a" * 64}, best)
    torch.save({"model_state_dict": {"weight": torch.zeros(1)}, "config_hash": "a" * 64}, latest)

    assert resolve_checkpoint_identifier(run_dir) == best.resolve()
    assert resolve_checkpoint_identifier(run_dir / "checkpoints") == best.resolve()
    assert resolve_checkpoint_identifier(latest) == latest.resolve()
    state_path, config_path = infer_portable_files(run_dir)
    assert state_path == best.resolve()
    assert config_path == (run_dir / "config.yaml").resolve()
    kwargs, loaded = load_training_config(config_path)
    assert kwargs == {"task": "abcd_fmri", "seed": 17}
    assert loaded == document

    monkeypatch.setattr(pysta.utils, "basedir", str(tmp_path))
    expected = (
        tmp_path / "data" / "abcd_task_analyses" / run_dir.name
    ).resolve()
    assert resolve_analysis_root(run_dir) == expected
    assert resolve_analysis_root(best) == expected

    expected.mkdir(parents=True)
    (expected / "analysis_manifest.json").write_text("{}\n", encoding="utf8")
    assert resolve_existing_analysis_root(run_dir) == expected
    assert resolve_existing_analysis_root(best) == expected


def test_legacy_portable_pair_and_disambiguated_output(monkeypatch, tmp_path):
    legacy_dir = tmp_path / "models" / "long" / "legacy"
    legacy_dir.mkdir(parents=True)
    checkpoint = legacy_dir / "model4_best.pt"
    state = legacy_dir / "model4_best_portable_state_dict.pt"
    kwargs = legacy_dir / "model4_portable_kwargs.json"
    checkpoint.write_bytes(b"identifier")
    torch.save({"weight": torch.ones(1)}, state)
    kwargs.write_text('{"seed": 4, "task": "abcd_fmri"}\n', encoding="utf8")

    assert infer_portable_files(checkpoint) == (state.resolve(), kwargs.resolve())
    monkeypatch.setattr(pysta.utils, "basedir", str(tmp_path))
    output = resolve_analysis_root(checkpoint)
    assert output.parent == (tmp_path / "data" / "abcd_task_analyses").resolve()
    assert output.name.startswith("model4_best_")


def test_structured_managed_and_unhashed_legacy_state_dicts_reconstruct(
    monkeypatch, tmp_path
):
    run_dir, document = _managed_run(tmp_path)
    config = run_dir / "config.yaml"
    reference = torch.nn.Linear(2, 1)
    expected = {name: value.detach().clone() for name, value in reference.state_dict().items()}

    monkeypatch.setattr(
        pysta.tasks, "make_environment", lambda kwargs, split: object()
    )
    monkeypatch.setattr(
        pysta.train_rnn,
        "_make_rnn",
        lambda environment, kwargs: torch.nn.Linear(2, 1),
    )

    structured = run_dir / "checkpoints" / "best.pt"
    torch.save(
        {"model_state_dict": expected, "config_hash": document["config_hash"]["full"]},
        structured,
    )
    model, kwargs = reconstruct_trained_model(structured, config)
    assert kwargs["seed"] == 17
    for name, value in expected.items():
        assert torch.equal(model.state_dict()[name], value)

    optimizer = torch.optim.Adam(reference.parameters())
    latest = run_dir / "checkpoints" / "latest.pt"
    latest_payload = run_manager.checkpoint_payload(
        kind="latest",
        config_hash=document["config_hash"]["full"],
        resume_provenance_hash="b" * 64,
        completed_updates=3,
        model=reference,
        best_validation_loss=0.2,
        best_update=2,
        optimizer=optimizer,
        validation_history=({"update": 2, "loss": 0.2},),
        env=SimpleNamespace(rng=np.random.default_rng(2), _block_counter=4),
        eval_env=None,
    )
    torch.save(latest_payload, latest)
    latest_model, _ = reconstruct_trained_model(latest, config)
    for name, value in expected.items():
        assert torch.equal(latest_model.state_dict()[name], value)

    raw = run_dir / "checkpoints" / "imported_raw.pt"
    torch.save(expected, raw)
    # Raw state dictionaries remain readable for the legacy/unhashed contract,
    # but cannot bypass a managed run's embedded config-hash binding.
    imported = dict(document)
    imported.pop("config_hash")
    imported_config = run_dir / "imported_config.yaml"
    imported_config.write_text(json.dumps(imported), encoding="utf8")
    model, _ = reconstruct_trained_model(raw, imported_config)
    for name, value in expected.items():
        assert torch.equal(model.state_dict()[name], value)

    with pytest.raises(ValueError, match="missing config_hash"):
        reconstruct_trained_model(raw, config)

    torch.save({"model_state_dict": expected, "config_hash": "wrong"}, structured)
    with pytest.raises(ValueError, match="hash mismatch"):
        reconstruct_trained_model(structured, config)


def test_raw_input_discovery_accepts_managed_run(monkeypatch, tmp_path):
    run_dir, _ = _managed_run(tmp_path)
    best = run_dir / "checkpoints" / "best.pt"
    torch.save({"model_state_dict": {"weight": torch.ones(1)}, "config_hash": "a" * 64}, best)
    monkeypatch.setattr(pysta.utils, "basedir", str(tmp_path))
    root = tmp_path / "data" / "abcd_task_analyses" / run_dir.name
    repeat_1 = root / "trial_collection" / "repeat_01"
    repeat_2 = root / "trial_collection" / "repeat_02"
    repeat_1.mkdir(parents=True)
    repeat_2.mkdir(parents=True)
    (root / "analysis_manifest.json").write_text("{}\n", encoding="utf8")
    for directory in (repeat_1, repeat_2):
        (directory / "normalized_navigation.npz").write_bytes(b"test")
    assert discover_normalized_inputs(run_dir) == [
        (repeat_1 / "normalized_navigation.npz").resolve(),
        (repeat_2 / "normalized_navigation.npz").resolve(),
    ]
