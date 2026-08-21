"""Independent identity/provenance contracts for managed ABCD runs."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

import pysta
from pysta import run_manager
from scripts.ABCD_task.abcd_analysis_common import (
    infer_portable_files,
    load_training_config,
    reconstruct_trained_model,
)


def _write_fake_source_tree(root: Path, marker: str) -> Path:
    """Create only the critical files used by source fingerprinting."""
    for relative in run_manager.SOURCE_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{marker}:{relative}\n", encoding="utf8")
    return root


def _scientific_training_config() -> dict:
    return {
        "arguments": {
            "task": "abcd_fmri",
            "model_type": "vanilla",
            "seed": 5,
            "lrate": 1e-4,
        },
        "summary": {"identity": {"task": "abcd_fmri", "seed": 5}},
        "derived": {"recurrent_units": 8, "observation_dim": 24, "output_dim": 4},
    }


def test_experiment_identity_excludes_source_runtime_path_and_device_provenance(
    monkeypatch, tmp_path
):
    scientific = _scientific_training_config()
    old_style_a = {
        **scientific,
        "code_fingerprint": {
            "combined_sha256": "source-a",
            "files": {"pysta/agents.py": "source-a"},
        },
        "runtime_dependencies": {
            "python": "3.11-a",
            "numpy": "1-a",
            "torch": "2-a",
        },
        "legacy_source_artifacts": {
            "checkpoint": {
                "absolute": "/machine-a/very/long/model.pt",
                "sha256": "artifact-a",
            }
        },
    }
    old_style_b = {
        **scientific,
        "code_fingerprint": {
            "combined_sha256": "source-b",
            "files": {"pysta/agents.py": "source-b"},
        },
        "runtime_dependencies": {
            "python": "3.12-b",
            "numpy": "2-b",
            "torch": "3-b",
        },
        "legacy_source_artifacts": {
            "checkpoint": {
                "absolute": "/machine-b/other/model.pt",
                "sha256": "artifact-b",
            }
        },
    }
    expected_hash = run_manager.experiment_config_hash(scientific)
    assert run_manager.experiment_config_hash(old_style_a) == expected_hash
    assert run_manager.experiment_config_hash(old_style_b) == expected_hash

    repo_a = _write_fake_source_tree(tmp_path / "source_a", "source-a")
    repo_b = _write_fake_source_tree(tmp_path / "source_b", "source-b")
    cwd_a = tmp_path / "launch_a"
    cwd_b = tmp_path / "launch_b"
    cwd_a.mkdir()
    cwd_b.mkdir()
    monkeypatch.chdir(cwd_a)
    first_dir, first_document = run_manager.prepare_run_directory(
        basedir=tmp_path / "storage_a",
        repo_root=repo_a,
        run_name="same-experiment",
        training_config=scientific,
        launch_controls={"training_device": "cpu"},
        resume=False,
    )
    monkeypatch.chdir(cwd_b)
    second_dir, second_document = run_manager.prepare_run_directory(
        basedir=tmp_path / "storage_b",
        repo_root=repo_b,
        run_name="same-experiment",
        training_config=scientific,
        launch_controls={"training_device": "mps"},
        resume=False,
    )
    assert first_dir.name == second_dir.name
    assert first_document["config_hash"] == second_document["config_hash"]
    assert first_document["config_hash"]["full"] == expected_hash
    assert (
        first_document["execution_provenance"]["compatibility"]["source"][
            "combined_sha256"
        ]
        != second_document["execution_provenance"]["compatibility"]["source"][
            "combined_sha256"
        ]
    )
    assert (
        first_document["execution_provenance"]["compatibility"]["device"]
        != second_document["execution_provenance"]["compatibility"]["device"]
    )
    assert (
        first_document["execution_provenance"]["informational"]["cwd"]
        != second_document["execution_provenance"]["informational"]["cwd"]
    )


def test_strict_resume_rejects_changed_training_source_provenance(tmp_path):
    scientific = _scientific_training_config()
    repo_root = _write_fake_source_tree(tmp_path / "source", "original")
    basedir = tmp_path / "storage"
    run_manager.prepare_run_directory(
        basedir=basedir,
        repo_root=repo_root,
        run_name="strict-resume",
        training_config=scientific,
        launch_controls={"training_device": "cpu"},
        resume=False,
    )

    # agents.py is training-critical, so an exact continuation may not silently
    # proceed after this change. Run-management source is also fingerprinted.
    (repo_root / "pysta" / "agents.py").write_text(
        "scientifically incompatible source\n", encoding="utf8"
    )
    with pytest.raises(ValueError, match="provenance incompatibility"):
        run_manager.prepare_run_directory(
            basedir=basedir,
            repo_root=repo_root,
            run_name="strict-resume",
            training_config=scientific,
            launch_controls={"training_device": "cpu"},
            resume=True,
        )


def test_resume_treats_host_as_informational_but_runtime_as_strict(
    monkeypatch, tmp_path
):
    scientific = _scientific_training_config()
    repo_root = _write_fake_source_tree(tmp_path / "source", "unchanged")
    basedir = tmp_path / "storage"
    baseline_runtime = run_manager.runtime_provenance("cpu")
    monkeypatch.setattr(
        run_manager,
        "runtime_provenance",
        lambda training_device: copy.deepcopy(baseline_runtime),
    )
    monkeypatch.setattr(run_manager.socket, "gethostname", lambda: "host-a")
    run_dir, document = run_manager.prepare_run_directory(
        basedir=basedir,
        repo_root=repo_root,
        run_name="runtime-contract",
        training_config=scientific,
        launch_controls={"training_device": "cpu"},
        resume=False,
    )

    # Host identity is retained for audit but does not make a compatible
    # execution unsafe to continue.
    monkeypatch.setattr(run_manager.socket, "gethostname", lambda: "host-b")
    resumed_dir, resumed_document = run_manager.prepare_run_directory(
        basedir=basedir,
        repo_root=repo_root,
        run_name="runtime-contract",
        training_config=scientific,
        launch_controls={"training_device": "cpu"},
        resume=True,
    )
    assert resumed_dir == run_dir
    assert resumed_document == document

    changed_runtime = copy.deepcopy(baseline_runtime)
    changed_runtime["dependencies"]["python"] = "incompatible-python"
    monkeypatch.setattr(
        run_manager,
        "runtime_provenance",
        lambda training_device: copy.deepcopy(changed_runtime),
    )
    with pytest.raises(ValueError, match="Python/NumPy/PyTorch runtime"):
        run_manager.prepare_run_directory(
            basedir=basedir,
            repo_root=repo_root,
            run_name="runtime-contract",
            training_config=scientific,
            launch_controls={"training_device": "cpu"},
            resume=True,
        )


def test_v1_resume_discovery_never_matches_a_prefix_overlapping_run_name(tmp_path):
    scientific = {
        "arguments": {"task": "abcd_fmri", "seed": 5},
        "derived": {"recurrent_units": 8},
    }
    repo_root = Path(__file__).resolve().parents[1]
    current = run_manager.build_execution_provenance(
        repo_root, {"training_device": "cpu"}
    )
    compatibility = current["compatibility"]
    v1_training = {
        **scientific,
        "code_fingerprint": compatibility["source"],
        "runtime_dependencies": compatibility["runtime_dependencies"],
    }
    old_hash = run_manager.sha256_json(v1_training)
    candidate = (
        tmp_path
        / "models"
        / "abcd_fmri"
        / f"foo_bar_{old_hash[:run_manager.SHORT_HASH_LENGTH]}"
    )
    candidate.mkdir(parents=True)
    document = {
        "schema_version": 1,
        "run": {
            "name": "foo_bar",
            "id": old_hash[:run_manager.SHORT_HASH_LENGTH],
            "task": "abcd_fmri",
        },
        "training_config": v1_training,
        "config_hash": {
            "algorithm": "sha256",
            "full": old_hash,
            "short": old_hash[:run_manager.SHORT_HASH_LENGTH],
        },
        "launch_controls": {"training_device": "cpu"},
        "provenance": {"platform": compatibility["platform"]},
    }
    (candidate / "config.yaml").write_text(
        json.dumps(document), encoding="utf8"
    )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        run_manager.prepare_run_directory(
            basedir=tmp_path,
            repo_root=repo_root,
            run_name="foo",
            training_config=scientific,
            launch_controls={"training_device": "cpu"},
            resume=True,
        )


def test_v2_analysis_recomputes_identity_and_binds_checkpoint_provenance(
    monkeypatch, tmp_path
):
    scientific = {
        "arguments": {"task": "abcd_fmri", "seed": 7},
        "derived": {
            "recurrent_units": 2,
            "observation_dim": 24,
            "output_dim": 4,
        },
    }
    run_dir, document = run_manager.prepare_run_directory(
        basedir=tmp_path,
        repo_root=Path(__file__).resolve().parents[1],
        run_name="analysis-binding",
        training_config=scientific,
        launch_controls={"training_device": "cpu"},
        resume=False,
    )
    reference = torch.nn.Linear(2, 1)
    state = {
        name: value.detach().clone()
        for name, value in reference.state_dict().items()
    }
    checkpoint = run_dir / "checkpoints" / "best.pt"
    torch.save(
        {
            "schema_version": 2,
            "checkpoint_kind": "best",
            "config_hash": document["config_hash"]["full"],
            "resume_provenance_hash": "wrong-provenance",
            "model_state_dict": state,
        },
        checkpoint,
    )
    monkeypatch.setattr(
        pysta.tasks, "make_environment", lambda kwargs, split: object()
    )
    monkeypatch.setattr(
        pysta.train_rnn,
        "_make_rnn",
        lambda environment, kwargs: torch.nn.Linear(2, 1),
    )
    with pytest.raises(ValueError, match="execution-provenance mismatch"):
        reconstruct_trained_model(checkpoint, run_dir / "config.yaml")

    tampered = json.loads((run_dir / "config.yaml").read_text(encoding="utf8"))
    tampered["training_config"]["arguments"]["seed"] = 999
    (run_dir / "config.yaml").write_text(json.dumps(tampered), encoding="utf8")
    with pytest.raises(ValueError, match="config identity hash mismatch"):
        reconstruct_trained_model(checkpoint, run_dir / "config.yaml")


def test_v1_managed_config_and_checkpoint_remain_analysis_readable(
    monkeypatch, tmp_path
):
    """Schema-v1 provenance inside training_config must remain loadable."""
    run_dir = tmp_path / "models" / "abcd_fmri" / "historical_v1_0123456789"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    training_config = {
        **_scientific_training_config(),
        "code_fingerprint": {
            "algorithm": "sha256",
            "combined_sha256": "old-source-fingerprint",
        },
        "runtime_dependencies": {
            "python": "3.11.0",
            "numpy": "1.26.0",
            "torch": "2.2.0",
        },
    }
    old_hash = run_manager.sha256_json(training_config)
    document = {
        "schema_version": 1,
        "run": {"name": "historical_v1", "id": old_hash[:10], "task": "abcd_fmri"},
        "training_config": training_config,
        "config_hash": {
            "algorithm": "sha256",
            "full": old_hash,
            "short": old_hash[:10],
        },
        "launch_controls": {"training_device": "cpu"},
        "provenance": {"note": "historical fixture"},
    }
    (run_dir / "config.yaml").write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf8"
    )

    reference = torch.nn.Linear(2, 1)
    state = {
        name: value.detach().cpu().clone()
        for name, value in reference.state_dict().items()
    }
    best = checkpoint_dir / "best.pt"
    torch.save(
        {
            "schema_version": 1,
            "checkpoint_kind": "best",
            "config_hash": old_hash,
            "resume_capable": False,
            "model_state_dict": state,
        },
        best,
    )

    state_path, config_path = infer_portable_files(run_dir)
    assert state_path == best.resolve()
    kwargs, loaded_document = load_training_config(config_path)
    assert kwargs == training_config["arguments"]
    assert loaded_document == document

    monkeypatch.setattr(
        pysta.tasks, "make_environment", lambda kwargs, split: object()
    )
    monkeypatch.setattr(
        pysta.train_rnn,
        "_make_rnn",
        lambda environment, kwargs: torch.nn.Linear(2, 1),
    )
    reconstructed, reconstructed_kwargs = reconstruct_trained_model(
        state_path, config_path
    )
    assert reconstructed_kwargs == training_config["arguments"]
    for name, expected in state.items():
        assert torch.equal(reconstructed.state_dict()[name], expected)
