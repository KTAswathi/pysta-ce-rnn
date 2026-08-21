"""Focused contracts for opt-in short ABCD training-run directories."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import pysta
from pysta import run_manager


class _DummyEnvironment:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self._block_counter = 0


class _ImportEnvironment(_DummyEnvironment):
    obs_dim = 24
    output_dim = 4
    batch = 1
    configuration_bank = ((0, 2, 6, 8),)
    num_loops = 5
    instruction_repeats = 2
    max_navigation_steps = 200
    min_manhattan_distance = 2
    start_policy = "exclude_first_goal"
    fixed_start = None
    allowed_instruction_directions = (0, 1)
    allowed_execution_relations = (0, 1)


class _ImportModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Nrec = 3
        self.env = _ImportEnvironment(3)
        self.weight = torch.nn.Parameter(torch.randn(3, 3))


def _training_step(model, optimizer, environment):
    optimizer.zero_grad()
    x = torch.randn(3)
    target = float(np.random.normal()) + float(environment.rng.normal())
    loss = torch.square(model(x).squeeze() - target)
    loss.backward()
    optimizer.step()
    environment._block_counter += 1


def _optimizer_tensors(optimizer):
    return [
        value.detach().cpu()
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    ]


def test_managed_directory_is_hash_guarded_and_never_silently_reused(tmp_path):
    config = {
        "arguments": {"task": "abcd_fmri", "seed": 3, "lrate": 1e-4},
        "derived": {"recurrent_units": 480},
        "code_fingerprint": {"combined_sha256": "abc"},
    }
    run_dir, document = run_manager.prepare_run_directory(
        basedir=tmp_path,
        repo_root=Path(__file__).resolve().parents[1],
        run_name="mechanism_baseline",
        training_config=config,
        launch_controls={"training_device": "cpu"},
        resume=False,
    )
    assert run_dir.parent == tmp_path / "models" / "abcd_fmri"
    assert run_dir.name == f"mechanism_baseline_{document['config_hash']['short']}"
    assert len(document["config_hash"]["short"]) == 10
    with (run_dir / "config.yaml").open(encoding="utf8") as stream:
        assert json.load(stream) == document

    with pytest.raises(FileExistsError, match="--resume 1"):
        run_manager.prepare_run_directory(
            basedir=tmp_path,
            repo_root=Path(__file__).resolve().parents[1],
            run_name="mechanism_baseline",
            training_config=config,
            launch_controls={"training_device": "cpu"},
            resume=False,
        )

    resumed_dir, resumed_document = run_manager.prepare_run_directory(
        basedir=tmp_path,
        repo_root=Path(__file__).resolve().parents[1],
        run_name="mechanism_baseline",
        training_config=config,
        launch_controls={"training_device": "cpu"},
        resume=True,
    )
    assert resumed_dir == run_dir
    assert resumed_document == document

    swapped = json.loads(json.dumps(document))
    swapped["run"]["name"] = "different-run"
    with (run_dir / "config.yaml").open("w", encoding="utf8") as stream:
        json.dump(swapped, stream)
    with pytest.raises(ValueError, match="document/name mismatch"):
        run_manager.prepare_run_directory(
            basedir=tmp_path,
            repo_root=Path(__file__).resolve().parents[1],
            run_name="mechanism_baseline",
            training_config=config,
            launch_controls={"training_device": "cpu"},
            resume=True,
        )
    with (run_dir / "config.yaml").open("w", encoding="utf8") as stream:
        json.dump(document, stream)

    with pytest.raises(ValueError, match="original training device"):
        run_manager.prepare_run_directory(
            basedir=tmp_path,
            repo_root=Path(__file__).resolve().parents[1],
            run_name="mechanism_baseline",
            training_config=config,
            launch_controls={"training_device": "cuda:0"},
            resume=True,
        )

    tampered = dict(document)
    tampered["training_config"] = dict(document["training_config"])
    tampered["training_config"]["arguments"] = {
        **document["training_config"]["arguments"],
        "lrate": 9e-4,
    }
    with (run_dir / "config.yaml").open("w", encoding="utf8") as stream:
        json.dump(tampered, stream)
    with pytest.raises(ValueError, match="configuration mismatch"):
        run_manager.prepare_run_directory(
            basedir=tmp_path,
            repo_root=Path(__file__).resolve().parents[1],
            run_name="mechanism_baseline",
            training_config=config,
            launch_controls={"training_device": "cpu"},
            resume=True,
        )


def test_latest_checkpoint_is_restricted_loadable_and_exactly_resumable(tmp_path):
    torch.manual_seed(31)
    np.random.seed(32)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    environment = _DummyEnvironment(33)
    eval_environment = _DummyEnvironment(34)
    _training_step(model, optimizer, environment)

    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    payload = run_manager.checkpoint_payload(
        kind="latest",
        config_hash="f" * 64,
        resume_provenance_hash="e" * 64,
        completed_updates=1,
        model=model,
        optimizer=optimizer,
        best_validation_loss=0.5,
        best_update=0,
        validation_history=[
            {
                "update": 0,
                "loss": 0.5,
                "accuracy": 0.25,
                "best_loss": 0.5,
                "elapsed_minutes": 0.0,
            }
        ],
        env=environment,
        eval_env=eval_environment,
    )
    latest = run_manager.save_checkpoint(run_dir, payload)
    restricted = torch.load(latest, map_location="cpu", weights_only=True)
    assert restricted["resume_capable"] is True
    assert all(value.device.type == "cpu" for value in restricted["model_state_dict"].values())

    # Uninterrupted reference continuation.
    _training_step(model, optimizer, environment)
    _training_step(model, optimizer, environment)
    reference_parameters = [value.detach().clone() for value in model.parameters()]
    reference_optimizer = [value.clone() for value in _optimizer_tensors(optimizer)]
    reference_torch = torch.randn(4)
    reference_numpy = np.random.normal(size=4)
    reference_environment = environment.rng.normal(size=4)

    # Fresh construction may consume RNG; loading must restore the boundary.
    torch.manual_seed(999)
    np.random.seed(998)
    resumed_model = torch.nn.Linear(3, 1)
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=0.01)
    resumed_environment = _DummyEnvironment(997)
    resumed_eval_environment = _DummyEnvironment(996)
    restored = run_manager.load_latest_checkpoint(
        run_dir,
        expected_config_hash="f" * 64,
        expected_resume_provenance_hash="e" * 64,
        model=resumed_model,
        optimizer=resumed_optimizer,
        env=resumed_environment,
        eval_env=resumed_eval_environment,
        device=torch.device("cpu"),
    )
    assert restored[:3] == (1, 0.5, 0)
    _training_step(resumed_model, resumed_optimizer, resumed_environment)
    _training_step(resumed_model, resumed_optimizer, resumed_environment)

    for expected, actual in zip(reference_parameters, resumed_model.parameters()):
        assert torch.equal(expected, actual)
    for expected, actual in zip(reference_optimizer, _optimizer_tensors(resumed_optimizer)):
        assert torch.equal(expected, actual)
    assert torch.equal(reference_torch, torch.randn(4))
    assert np.array_equal(reference_numpy, np.random.normal(size=4))
    assert np.array_equal(reference_environment, resumed_environment.rng.normal(size=4))


def test_latest_provenance_mismatch_is_rejected_before_state_mutation(tmp_path):
    source_model = torch.nn.Linear(3, 1)
    with torch.no_grad():
        source_model.weight.fill_(7.0)
        source_model.bias.fill_(8.0)
    source_optimizer = torch.optim.Adam(source_model.parameters(), lr=0.01)
    source_environment = _DummyEnvironment(43)
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    run_manager.save_checkpoint(
        run_dir,
        run_manager.checkpoint_payload(
            kind="latest",
            config_hash="c" * 64,
            resume_provenance_hash="a" * 64,
            completed_updates=0,
            model=source_model,
            optimizer=source_optimizer,
            best_validation_loss=1.0,
            best_update=None,
            validation_history=(),
            env=source_environment,
            eval_env=None,
        ),
    )

    target_model = torch.nn.Linear(3, 1)
    target_optimizer = torch.optim.Adam(target_model.parameters(), lr=0.01)
    target_environment = _DummyEnvironment(44)
    parameters_before = {
        name: value.detach().clone()
        for name, value in target_model.state_dict().items()
    }
    rng_before = target_environment.rng.bit_generator.state
    with pytest.raises(ValueError, match="execution provenance"):
        run_manager.load_latest_checkpoint(
            run_dir,
            expected_config_hash="c" * 64,
            expected_resume_provenance_hash="b" * 64,
            model=target_model,
            optimizer=target_optimizer,
            env=target_environment,
            eval_env=None,
            device=torch.device("cpu"),
        )
    for name, expected in parameters_before.items():
        assert torch.equal(target_model.state_dict()[name], expected)
    assert target_optimizer.state == {}
    assert target_environment.rng.bit_generator.state == rng_before


def test_model_summary_exposes_actual_spatial_recipient_identity():
    arguments = {
        "task": "abcd_fmri",
        "model_type": "corticallyembedded",
        "seed": 1,
        "local_fraction": 0.25,
    }
    derived = {
        "model_class": "pysta.agents.CorticallyEmbeddedRNN",
        "recurrent_units": 480,
        "observation_dim": 24,
        "output_dim": 4,
        "parameter_count": 123,
        "training_configuration_bank": [[0, 2, 6, 8]],
        "input_routing": {
            "current_location": {"count": 120, "indices_sha256": "recipient-hash"}
        },
        "anatomical_anchor_unit_indices": {
            "count": 80,
            "indices_sha256": "anchor-hash",
        },
    }
    summary = run_manager.build_model_summary(arguments, derived)
    mechanism = summary["cortical_mechanism"]
    assert mechanism["local_fraction"] == 0.25
    assert mechanism["actual_spatial_input_recipients"] == {
        "count": 120,
        "indices_sha256": "recipient-hash",
    }
    assert mechanism["anatomical_anchor"]["count"] == 80


def test_opt_in_managed_training_writes_short_complete_layout(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(sys, "argv", ["train"])
    kwargs = pysta.argparser.parse_args(
        task="abcd_fmri",
        model_type="vanilla",
        Nrec=6,
        batch_size=1,
        n_loops=1,
        instruction_repeats=1,
        max_navigation_steps=1,
        num_train_configurations=1,
        num_eval=1,
        num_epochs=1,
        eval_freq=1,
        iters_per_action=1,
        tau=1.0,
        rec_noise=0.0,
        run_final_fmri_evaluation=False,
        run_name="managed-smoke",
        seed=17,
    )
    monkeypatch.setattr(pysta.utils, "basedir", str(tmp_path))
    pysta.train_rnn.main_train(kwargs)

    run_dirs = list((tmp_path / "models" / "abcd_fmri").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert run_dir.name.startswith("managed-smoke_")
    assert len(run_dir.name) < 64
    expected = {
        "config.yaml",
        "validation_curve.csv",
        "validation_curve.png",
        "validation_metrics.json",
    }
    assert expected.issubset({path.name for path in run_dir.iterdir()})
    assert (run_dir / "checkpoints" / "best.pt").is_file()
    assert (run_dir / "checkpoints" / "latest.pt").is_file()
    best = torch.load(
        run_dir / "checkpoints" / "best.pt",
        map_location="cpu",
        weights_only=True,
    )
    latest = torch.load(
        run_dir / "checkpoints" / "latest.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert best["checkpoint_kind"] == "best"
    assert latest["checkpoint_kind"] == "latest"
    assert latest["completed_updates"] == 1
    with (run_dir / "config.yaml").open(encoding="utf8") as stream:
        config = json.load(stream)
    assert set(config["training_config"]) == {"arguments", "derived"}
    assert config["summary"]["task"]["n_loops"] == 1
    assert config["summary"]["identity"]["Nout"] == 4
    assert best["config_hash"] == config["config_hash"]["full"]
    assert latest["config_hash"] == config["config_hash"]["full"]
    assert best["resume_provenance_hash"] == config["execution_provenance"][
        "compatibility_hash"
    ]["full"]
    assert latest["resume_provenance_hash"] == best["resume_provenance_hash"]


def test_legacy_import_is_read_only_and_explicitly_non_resumable(tmp_path):
    source = tmp_path / "stable_legacy_snapshot.pt"
    source.write_bytes(b"read-only source identity")
    source_before = source.read_bytes()
    model = _ImportModel()
    history = [
        {
            "update": 0,
            "loss": 1.0,
            "accuracy": 0.1,
            "best_loss": 1.0,
            "elapsed_minutes": 0.0,
        }
    ]
    run_dir = run_manager.import_legacy_run(
        basedir=tmp_path,
        repo_root=Path(__file__).resolve().parents[1],
        run_name="imported-example",
        resolved_kwargs={
            "task": "abcd_fmri",
            "model_type": "vanilla",
            "seed": 3,
            "lrate": 3e-4,
            "num_epochs": 10,
        },
        model=model,
        eval_env=None,
        fmri_factorial_schedule=None,
        best_state_dict=model.state_dict(),
        latest_state_dict=model.state_dict(),
        source_artifacts={"legacy_checkpoint": source},
        completed_updates=10,
        best_validation_loss=1.0,
        best_update=0,
        validation_history=history,
    )
    assert source.read_bytes() == source_before
    best = torch.load(
        run_dir / "checkpoints" / "best.pt", weights_only=True
    )
    latest = torch.load(
        run_dir / "checkpoints" / "latest.pt", weights_only=True
    )
    assert best["resume_capable"] is False
    assert latest["resume_capable"] is False
    with (run_dir / "config.yaml").open(encoding="utf8") as stream:
        config = json.load(stream)
    assert config["legacy_import"]["resume_supported"] is False
    assert config["legacy_import"]["source_artifacts"]["legacy_checkpoint"][
        "sha256"
    ] == run_manager.sha256_file(source)

    copied_source = tmp_path / "different_snapshot_name.pt"
    copied_source.write_bytes(source_before)
    second_dir = run_manager.import_legacy_run(
        basedir=tmp_path,
        repo_root=Path(__file__).resolve().parents[1],
        run_name="same-import-another-label",
        resolved_kwargs={
            "task": "abcd_fmri",
            "model_type": "vanilla",
            "seed": 3,
            "lrate": 3e-4,
            "num_epochs": 10,
        },
        model=model,
        eval_env=None,
        fmri_factorial_schedule=None,
        best_state_dict=model.state_dict(),
        latest_state_dict=model.state_dict(),
        source_artifacts={"legacy_checkpoint": copied_source},
        completed_updates=10,
        best_validation_loss=1.0,
        best_update=0,
        validation_history=history,
    )
    assert run_dir.name.rsplit("_", 1)[1] == second_dir.name.rsplit("_", 1)[1]
