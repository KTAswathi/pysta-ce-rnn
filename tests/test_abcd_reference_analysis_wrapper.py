"""Lightweight orchestration tests for the frozen ABCD reference pipeline."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.ABCD_task import run_abcd_reference_analysis as wrapper


TRAINING_CONFIG = {
    "arguments": {"task": "abcd_fmri", "seed": 1, "num_epochs": 10},
    "derived": {"observation_dim": 24, "output_dim": 4},
}
CONFIG_HASH = hashlib.sha256(
    json.dumps(
        TRAINING_CONFIG,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf8")
).hexdigest()
RUN_PREFIX = "mech_baseline_seed1"
RUN_NAME = f"{RUN_PREFIX}_{CONFIG_HASH[:10]}"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_npz(path: Path) -> None:
    np.savez_compressed(path, value=np.asarray([1], dtype=np.int64))


def _write_png(path: Path) -> None:
    # The wrapper deliberately validates only the PNG signature/nonzero body;
    # scientific figure content belongs to the frozen analysis entry points.
    path.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic-test-figure")


def _make_managed_run(
    root: Path,
    *,
    best: bool = True,
    schema_version: int = 1,
    completed_updates: int = 10,
) -> Path:
    """Create just enough of either managed-run schema for orchestration."""
    source_files = {
        relative
        for stage_files in wrapper.STAGE_SOURCE_FILES.values()
        for relative in stage_files
    }
    for relative in source_files:
        source = root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"synthetic source: {relative}\n", encoding="utf8")
    run_dir = root / "models" / "abcd_fmri" / RUN_NAME
    (run_dir / "checkpoints").mkdir(parents=True)
    document = {
        "schema": "pysta_managed_run/v1",
        "schema_version": schema_version,
        "run": {
            "name": RUN_PREFIX,
            "id": CONFIG_HASH[:10],
            "task": "abcd_fmri",
        },
        "config_hash": {"full": CONFIG_HASH, "short": CONFIG_HASH[:10]},
        "training_config": TRAINING_CONFIG,
    }
    if schema_version == 1:
        document["legacy_import"] = {"resume_supported": False}
    (run_dir / "config.yaml").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf8",
    )
    if best:
        (run_dir / "checkpoints" / "best.pt").write_bytes(b"checkpoint")
    if schema_version >= 2:
        import torch

        rng_dtype = getattr(torch, "uint32", torch.int64)
        torch.save(
            {
                "schema_version": schema_version,
                "checkpoint_kind": "latest",
                "config_hash": CONFIG_HASH,
                "completed_updates": completed_updates,
                "model_state_dict": {
                    "weight": torch.ones(1, dtype=torch.float32),
                },
                # This mirrors the dtype responsible for the real cross-version
                # failure: Torch 2.2 cannot deserialize a uint32 RNG tensor
                # written by Torch 2.6, although the scalar header is readable
                # without unpickling either the RNG state or model tensors.
                "rng_state": {
                    "numpy_uint32_like": torch.ones(4, dtype=rng_dtype),
                },
            },
            run_dir / "checkpoints" / "latest.pt",
        )
    return run_dir


def _analysis_root(root: Path) -> Path:
    return root / "data" / "abcd_task_analyses" / RUN_NAME


def _pipeline_status(root: Path) -> dict:
    return json.loads(
        (_analysis_root(root) / wrapper.PIPELINE_STATUS_NAME).read_text(
            encoding="utf8"
        )
    )


def _write_pipeline_status(root: Path, document: dict) -> None:
    (_analysis_root(root) / wrapper.PIPELINE_STATUS_NAME).write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf8",
    )


def _mark_complete(root: Path, stage: str) -> None:
    analysis_root = _analysis_root(root)
    if stage == "collect":
        run_dir = root / "models" / "abcd_fmri" / RUN_NAME
        checkpoint_hash = _file_sha256(run_dir / "checkpoints" / "best.pt")
        config_hash = _file_sha256(run_dir / "config.yaml")
        (analysis_root / "trial_collection" / "repeat_01").mkdir(
            parents=True, exist_ok=True
        )
        (analysis_root / "trial_collection" / "repeat_02").mkdir(
            parents=True, exist_ok=True
        )
        (analysis_root / "analysis_manifest.json").write_text(
            json.dumps(
                {
                    "schema": "abcd_reference_analysis/v1",
                    "task": {"num_factorial_blocks": 1},
                    "source": {
                        "checkpoint_identifier": {"sha256": checkpoint_hash},
                        "resolved_config": {"sha256": config_hash},
                    },
                }
            )
            + "\n"
        )
        summaries = [
            {
                "repeat_index": repeat_index,
                "autonomous": True,
                "greedy": True,
                "force_optimal": False,
                "weights_frozen": True,
                "num_blocks": 1,
                "completed_blocks": 1,
            }
            for repeat_index in (1, 2)
        ]
        (analysis_root / "trial_collection" / "collection_qc.json").write_text(
            json.dumps(
                {
                    "checkpoint_sha256": checkpoint_hash,
                    "resolved_config_sha256": config_hash,
                    "weights_unchanged": True,
                    "same_trained_recurrent_noise": True,
                    "distinct_recurrent_noise_seeds": True,
                    "matched_factorial_cells": True,
                    "matched_start_locations": True,
                    "repeats": summaries,
                }
            )
            + "\n"
        )
        for summary, repeat in zip(summaries, ("repeat_01", "repeat_02")):
            repeat_dir = analysis_root / "trial_collection" / repeat
            _write_npz(repeat_dir / "normalized_navigation.npz")
            (repeat_dir / "behaviour_summary.json").write_text(
                json.dumps(summary) + "\n"
            )
            (repeat_dir / "blocks").mkdir(exist_ok=True)
            _write_npz(repeat_dir / "blocks" / "block_000.npz")
        return

    output = analysis_root / stage
    output.mkdir(parents=True, exist_ok=True)
    _write_npz(output / "results.npz")
    (output / "summary.csv").write_text("metric,value\ncomplete,1\n")
    _write_png(output / "summary.png")
    if stage == "csubs":
        (output / "fit_log.txt").write_text("complete\n")
    metadata_name = "metadata.json" if stage == "local_rsa" else "analysis.json"
    manifest_hash = _file_sha256(analysis_root / "analysis_manifest.json")
    metadata = {
        "analysis": {
            "raw_activity": "nuisance_controlled_normalized_progress_raw_activity",
            "csubs": "normalized_progress_cross_fitted_csubs",
            "local_rsa": "reference_matched_local_rsa",
        }[stage],
        "source_manifest_sha256": manifest_hash,
    }
    if stage == "csubs":
        metadata.update(
            {"n_permutations": 1000, "settings": {"max_iters": 2000}}
        )
    if stage == "local_rsa":
        metadata["searchlight_radius_surface_mm"] = 6.0
    (output / metadata_name).write_text(json.dumps(metadata) + "\n")


class RecordingRunner:
    """Record subprocesses and materialize their nominal completion markers."""

    def __init__(self, root: Path):
        self.root = root
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, command, **kwargs):
        command = [str(item) for item in command]
        self.calls.append((command, kwargs))
        joined = " ".join(command)
        if "collect_abcd_reference_trials" in joined:
            _mark_complete(self.root, "collect")
        elif "analyse_abcd_normalized_raw" in joined:
            _mark_complete(self.root, "raw_activity")
        elif "analyse_abcd_normalized_csubs" in joined:
            _mark_complete(self.root, "csubs")
        elif "analyse_abcd_local_rsa" in joined:
            _mark_complete(self.root, "local_rsa")
        else:  # pragma: no cover - makes an unexpected scientific entry point loud
            raise AssertionError(f"Unexpected command: {command}")


def _invoke(monkeypatch, root: Path, runner: RecordingRunner, *arguments: str):
    monkeypatch.setattr(wrapper, "REPO_ROOT", root)
    monkeypatch.setattr(wrapper.subprocess, "run", runner)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_abcd_reference_analysis.py", RUN_NAME, *arguments],
    )
    return wrapper.main()


def _stage_names(calls: list[tuple[list[str], dict]]) -> list[str]:
    names = []
    for command, _ in calls:
        joined = " ".join(command)
        if "collect_abcd_reference_trials" in joined:
            names.append("collect")
        elif "analyse_abcd_normalized_raw" in joined:
            names.append("raw")
        elif "analyse_abcd_normalized_csubs" in joined:
            names.append("csubs")
        elif "analyse_abcd_local_rsa" in joined:
            names.append("local_rsa")
    return names


def test_one_command_resolves_best_checkpoint_and_runs_frozen_order(
    monkeypatch, tmp_path
):
    run_dir = _make_managed_run(tmp_path)
    runner = RecordingRunner(tmp_path)

    _invoke(monkeypatch, tmp_path, runner)

    assert _stage_names(runner.calls) == ["collect", "raw", "csubs", "local_rsa"]
    commands = [call[0] for call in runner.calls]
    checkpoint = (run_dir / "checkpoints" / "best.pt").resolve()
    analysis_root = _analysis_root(tmp_path).resolve()
    collection = analysis_root / "trial_collection"

    # Collection alone reads the frozen checkpoint. Every analysis is routed
    # explicitly to the collection/root produced by that first command.
    assert str(checkpoint) in commands[0]
    assert str(analysis_root) in commands[0]
    assert str(checkpoint) not in commands[1] + commands[2] + commands[3]
    assert str(collection) in commands[1]
    assert str(analysis_root / "raw_activity") in commands[1]
    assert str(analysis_root) in commands[2]
    assert str(analysis_root) in commands[3]
    assert str(collection / "repeat_01") in commands[3]
    assert str(collection / "repeat_02") in commands[3]
    assert commands[2][-8:] == [
        "--n-permutations",
        "1000",
        "--permutation-seed",
        "881",
        "--device",
        "auto",
        "--max-iters",
        "2000",
    ]
    assert commands[3][-2:] == ["--radius-mm", "6.0"]
    assert all(command[0] == sys.executable for command in commands)
    assert all(kwargs.get("check") is True for _, kwargs in runner.calls)


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        ("collect", ["collect"]),
        ("raw", ["raw"]),
        ("csubs", ["csubs"]),
        ("local_rsa", ["local_rsa"]),
    ],
)
def test_only_runs_exactly_one_selected_stage(
    monkeypatch, tmp_path, selection, expected
):
    _make_managed_run(tmp_path)
    runner = RecordingRunner(tmp_path)
    if selection != "collect":
        wrapper.run_pipeline(
            RUN_NAME,
            only="collect",
            repo_root=tmp_path,
            runner=runner,
        )
        runner.calls.clear()

    _invoke(monkeypatch, tmp_path, runner, "--only", selection)

    assert _stage_names(runner.calls) == expected


def test_skip_existing_uses_complete_artifacts_not_directory_presence(
    monkeypatch, tmp_path
):
    _make_managed_run(tmp_path)
    runner = RecordingRunner(tmp_path)
    wrapper.run_pipeline(
        RUN_NAME, only="collect", repo_root=tmp_path, runner=runner
    )
    wrapper.run_pipeline(RUN_NAME, only="raw", repo_root=tmp_path, runner=runner)
    # A partial Csubs directory must not count as a completed analysis.
    (_analysis_root(tmp_path) / "csubs").mkdir(parents=True)
    (_analysis_root(tmp_path) / "csubs" / "results.npz").write_bytes(b"partial")
    runner.calls.clear()

    _invoke(monkeypatch, tmp_path, runner, "--skip-existing")

    assert _stage_names(runner.calls) == ["csubs", "local_rsa"]


def test_skip_existing_does_not_trust_outputs_without_pipeline_status(
    monkeypatch, tmp_path
):
    _make_managed_run(tmp_path)
    runner = RecordingRunner(tmp_path)
    wrapper.run_pipeline(
        RUN_NAME, only="collect", repo_root=tmp_path, runner=runner
    )
    # These look structurally complete, but were copied/created outside the
    # wrapper and therefore have no reusable completion certificate.
    _mark_complete(tmp_path, "raw_activity")
    runner.calls.clear()

    wrapper.run_pipeline(
        RUN_NAME,
        only="raw",
        skip_existing=True,
        repo_root=tmp_path,
        runner=runner,
    )

    assert _stage_names(runner.calls) == ["raw"]
    assert _pipeline_status(tmp_path)["stages"]["raw"]["status"] == "complete"


def test_skip_existing_rejects_mismatched_pipeline_stage_signature(
    monkeypatch, tmp_path
):
    _make_managed_run(tmp_path)
    runner = RecordingRunner(tmp_path)
    wrapper.run_pipeline(
        RUN_NAME, only="collect", repo_root=tmp_path, runner=runner
    )
    wrapper.run_pipeline(RUN_NAME, only="raw", repo_root=tmp_path, runner=runner)
    status = _pipeline_status(tmp_path)
    status["stages"]["raw"]["signature"]["stage"] = "csubs"
    _write_pipeline_status(tmp_path, status)
    runner.calls.clear()

    wrapper.run_pipeline(
        RUN_NAME,
        only="raw",
        skip_existing=True,
        repo_root=tmp_path,
        runner=runner,
    )

    assert _stage_names(runner.calls) == ["raw"]


def test_copied_stage_certificate_cannot_certify_another_analysis(
    monkeypatch, tmp_path
):
    _make_managed_run(tmp_path)
    runner = RecordingRunner(tmp_path)
    wrapper.run_pipeline(
        RUN_NAME, only="collect", repo_root=tmp_path, runner=runner
    )
    wrapper.run_pipeline(RUN_NAME, only="raw", repo_root=tmp_path, runner=runner)
    _mark_complete(tmp_path, "csubs")
    status = _pipeline_status(tmp_path)
    status["stages"]["csubs"] = dict(status["stages"]["raw"])
    _write_pipeline_status(tmp_path, status)
    runner.calls.clear()

    wrapper.run_pipeline(
        RUN_NAME,
        only="csubs",
        skip_existing=True,
        repo_root=tmp_path,
        runner=runner,
    )

    assert _stage_names(runner.calls) == ["csubs"]
    assert _pipeline_status(tmp_path)["stages"]["csubs"]["signature"][
        "stage"
    ] == "csubs"


def test_copied_scientific_metadata_cannot_certify_another_analysis(
    monkeypatch, tmp_path
):
    _make_managed_run(tmp_path)
    runner = RecordingRunner(tmp_path)
    wrapper.run_pipeline(
        RUN_NAME, only="collect", repo_root=tmp_path, runner=runner
    )
    wrapper.run_pipeline(RUN_NAME, only="raw", repo_root=tmp_path, runner=runner)
    wrapper.run_pipeline(
        RUN_NAME, only="csubs", repo_root=tmp_path, runner=runner
    )
    analysis_root = _analysis_root(tmp_path)
    (analysis_root / "csubs" / "analysis.json").write_bytes(
        (analysis_root / "raw_activity" / "analysis.json").read_bytes()
    )
    runner.calls.clear()

    wrapper.run_pipeline(
        RUN_NAME,
        only="csubs",
        skip_existing=True,
        repo_root=tmp_path,
        runner=runner,
    )

    assert _stage_names(runner.calls) == ["csubs"]


@pytest.mark.parametrize("damage", ["zero_byte", "corrupt_npz", "corrupt_json"])
def test_skip_existing_recomputes_zero_byte_or_corrupt_artifacts(
    monkeypatch, tmp_path, damage
):
    _make_managed_run(tmp_path)
    runner = RecordingRunner(tmp_path)
    wrapper.run_pipeline(
        RUN_NAME, only="collect", repo_root=tmp_path, runner=runner
    )
    wrapper.run_pipeline(RUN_NAME, only="raw", repo_root=tmp_path, runner=runner)
    raw_dir = _analysis_root(tmp_path) / "raw_activity"
    if damage == "zero_byte":
        (raw_dir / "summary.csv").write_bytes(b"")
    elif damage == "corrupt_npz":
        (raw_dir / "results.npz").write_bytes(b"not a zip archive")
    else:
        (raw_dir / "analysis.json").write_text("{not-json")
    runner.calls.clear()

    wrapper.run_pipeline(
        RUN_NAME,
        only="raw",
        skip_existing=True,
        repo_root=tmp_path,
        runner=runner,
    )

    assert _stage_names(runner.calls) == ["raw"]


def test_recollecting_marks_all_downstream_completion_records_stale(
    monkeypatch, tmp_path
):
    _make_managed_run(tmp_path)
    runner = RecordingRunner(tmp_path)
    wrapper.run_pipeline(RUN_NAME, repo_root=tmp_path, runner=runner)
    assert all(
        _pipeline_status(tmp_path)["stages"][stage]["status"] == "complete"
        for stage in wrapper.STAGE_ORDER
    )

    wrapper.run_pipeline(
        RUN_NAME,
        only="collect",
        force=True,
        repo_root=tmp_path,
        runner=runner,
    )

    status = _pipeline_status(tmp_path)["stages"]
    assert status["collect"]["status"] == "complete"
    assert all(status[stage]["status"] == "stale" for stage in wrapper.STAGE_ORDER[1:])
    paths = wrapper.resolve_managed_run(RUN_NAME, repo_root=tmp_path)
    assert all(
        wrapper.stage_status(paths, stage) == "partial"
        for stage in wrapper.STAGE_ORDER[1:]
    )


def test_pipeline_source_mutation_leaves_stage_failed_not_complete(
    monkeypatch, tmp_path
):
    _make_managed_run(tmp_path)
    recorder = RecordingRunner(tmp_path)
    changed_source = tmp_path / wrapper.STAGE_SOURCE_FILES["collect"][0]

    def mutate_source(command, **kwargs):
        recorder(command, **kwargs)
        changed_source.write_text("mutated during stage\n", encoding="utf8")

    with pytest.raises(RuntimeError, match="changed"):
        wrapper.run_pipeline(
            RUN_NAME,
            only="collect",
            repo_root=tmp_path,
            runner=mutate_source,
        )

    status = _pipeline_status(tmp_path)["stages"]["collect"]
    assert status["status"] == "failed"
    paths = wrapper.resolve_managed_run(RUN_NAME, repo_root=tmp_path)
    assert wrapper.stage_status(paths, "collect") == "partial"
    assert not (_analysis_root(tmp_path) / ".run_abcd_reference_analysis.lock").exists()


def test_force_reruns_all_complete_stages(monkeypatch, tmp_path):
    _make_managed_run(tmp_path)
    for stage in ("collect", "raw_activity", "csubs", "local_rsa"):
        _mark_complete(tmp_path, stage)
    runner = RecordingRunner(tmp_path)

    _invoke(monkeypatch, tmp_path, runner, "--force")

    assert _stage_names(runner.calls) == ["collect", "raw", "csubs", "local_rsa"]


def test_default_refuses_to_overwrite_existing_output(monkeypatch, tmp_path):
    _make_managed_run(tmp_path)
    _mark_complete(tmp_path, "collect")
    runner = RecordingRunner(tmp_path)

    with pytest.raises(SystemExit):
        _invoke(monkeypatch, tmp_path, runner)

    assert runner.calls == []


def test_skip_existing_and_force_are_mutually_exclusive(monkeypatch, tmp_path):
    _make_managed_run(tmp_path)
    runner = RecordingRunner(tmp_path)

    with pytest.raises(SystemExit):
        _invoke(
            monkeypatch,
            tmp_path,
            runner,
            "--skip-existing",
            "--force",
        )
    assert runner.calls == []


def test_subprocess_failure_propagates_and_releases_analysis_lock(
    monkeypatch, tmp_path
):
    _make_managed_run(tmp_path)
    calls = []

    def fail(command, **kwargs):
        calls.append((command, kwargs))
        raise subprocess.CalledProcessError(returncode=7, cmd=command)

    monkeypatch.setattr(wrapper, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(wrapper.subprocess, "run", fail)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_abcd_reference_analysis.py", RUN_NAME, "--only", "collect"],
    )

    with pytest.raises(subprocess.CalledProcessError) as error:
        wrapper.main()

    assert error.value.returncode == 7
    assert len(calls) == 1
    assert not (
        _analysis_root(tmp_path) / ".run_abcd_reference_analysis.lock"
    ).exists()
    assert _pipeline_status(tmp_path)["stages"]["collect"]["status"] == "failed"
    paths = wrapper.resolve_managed_run(RUN_NAME, repo_root=tmp_path)
    assert wrapper.stage_status(paths, "collect") == "partial"


def test_existing_lock_prevents_launch_and_is_not_removed(monkeypatch, tmp_path):
    _make_managed_run(tmp_path)
    analysis_root = _analysis_root(tmp_path)
    analysis_root.mkdir(parents=True)
    lock = analysis_root / ".run_abcd_reference_analysis.lock"
    lock.write_text("pid=123\n")
    runner = RecordingRunner(tmp_path)

    with pytest.raises(SystemExit):
        _invoke(monkeypatch, tmp_path, runner, "--only", "collect")

    assert runner.calls == []
    assert lock.read_text() == "pid=123\n"


@pytest.mark.parametrize(
    "unsafe_name",
    ["../escape", "nested/run", ".", "..", "/absolute/run"],
)
def test_folder_name_input_cannot_escape_managed_models_root(
    monkeypatch, tmp_path, unsafe_name
):
    runner = RecordingRunner(tmp_path)
    monkeypatch.setattr(wrapper, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(wrapper.subprocess, "run", runner)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_abcd_reference_analysis.py", unsafe_name, "--only", "collect"],
    )

    with pytest.raises((SystemExit, ValueError)):
        wrapper.main()
    assert runner.calls == []


def test_symlinked_analysis_root_is_rejected(monkeypatch, tmp_path):
    _make_managed_run(tmp_path)
    external = tmp_path / "external_analysis"
    external.mkdir()
    analysis_root = _analysis_root(tmp_path)
    analysis_root.parent.mkdir(parents=True)
    analysis_root.symlink_to(external, target_is_directory=True)
    runner = RecordingRunner(tmp_path)

    with pytest.raises(ValueError, match="symlink"):
        wrapper.run_pipeline(
            RUN_NAME,
            only="collect",
            repo_root=tmp_path,
            runner=runner,
        )

    assert runner.calls == []


@pytest.mark.parametrize(
    ("stage", "directory_name"),
    [
        ("collect", "trial_collection"),
        ("raw", "raw_activity"),
        ("csubs", "csubs"),
        ("local_rsa", "local_rsa"),
    ],
)
def test_symlinked_stage_output_directory_is_rejected(
    monkeypatch, tmp_path, stage, directory_name
):
    _make_managed_run(tmp_path)
    runner = RecordingRunner(tmp_path)
    if stage != "collect":
        wrapper.run_pipeline(
            RUN_NAME,
            only="collect",
            repo_root=tmp_path,
            runner=runner,
        )
        runner.calls.clear()
    analysis_root = _analysis_root(tmp_path)
    analysis_root.mkdir(parents=True, exist_ok=True)
    external = tmp_path / f"external_{directory_name}"
    external.mkdir()
    (analysis_root / directory_name).symlink_to(
        external, target_is_directory=True
    )

    with pytest.raises(ValueError, match="symlink"):
        wrapper.run_pipeline(
            RUN_NAME,
            only=stage,
            repo_root=tmp_path,
            runner=runner,
        )

    assert runner.calls == []


def test_missing_run_or_best_checkpoint_fails_before_launch(monkeypatch, tmp_path):
    runner = RecordingRunner(tmp_path)
    with pytest.raises((SystemExit, FileNotFoundError)):
        _invoke(monkeypatch, tmp_path, runner, "--only", "collect")
    assert runner.calls == []

    _make_managed_run(tmp_path, best=False)
    with pytest.raises((SystemExit, FileNotFoundError)):
        _invoke(monkeypatch, tmp_path, runner, "--only", "collect")
    assert runner.calls == []


def test_analysis_only_requires_the_shared_trial_collection(
    monkeypatch, tmp_path
):
    _make_managed_run(tmp_path)
    runner = RecordingRunner(tmp_path)

    with pytest.raises((SystemExit, FileNotFoundError)):
        _invoke(monkeypatch, tmp_path, runner, "--only", "raw")
    assert runner.calls == []


def test_skip_existing_rejects_collection_from_another_checkpoint(
    monkeypatch, tmp_path
):
    run_dir = _make_managed_run(tmp_path)
    _mark_complete(tmp_path, "collect")
    (run_dir / "checkpoints" / "best.pt").write_bytes(b"different checkpoint")
    runner = RecordingRunner(tmp_path)

    with pytest.raises(RuntimeError, match="does not match"):
        _invoke(monkeypatch, tmp_path, runner, "--skip-existing")

    assert runner.calls == []


def test_schema_v1_managed_folder_remains_accepted(monkeypatch, tmp_path):
    """The wrapper must not make legacy managed imports path-version specific."""
    _make_managed_run(tmp_path)
    runner = RecordingRunner(tmp_path)

    _invoke(monkeypatch, tmp_path, runner, "--only", "collect")

    assert _stage_names(runner.calls) == ["collect"]


def test_schema_v2_managed_folder_is_accepted(monkeypatch, tmp_path):
    _make_managed_run(tmp_path, schema_version=2)
    runner = RecordingRunner(tmp_path)

    _invoke(monkeypatch, tmp_path, runner, "--only", "collect")

    assert _stage_names(runner.calls) == ["collect"]


def test_schema_v2_resolution_never_deserializes_latest_tensor_payload(
    monkeypatch, tmp_path
):
    """The completion guard must remain usable across Torch dtype versions."""

    run_dir = _make_managed_run(tmp_path, schema_version=2)
    import torch

    def forbidden_torch_load(*_args, **_kwargs):
        raise AssertionError("latest.pt tensors must not be deserialized")

    monkeypatch.setattr(torch, "load", forbidden_torch_load)

    paths = wrapper.resolve_managed_run(RUN_NAME, repo_root=tmp_path)

    assert paths.run_dir == run_dir.resolve()
    assert paths.checkpoint == (run_dir / "checkpoints" / "best.pt").resolve()


def test_malformed_native_latest_header_is_rejected(tmp_path):
    run_dir = _make_managed_run(tmp_path, schema_version=2)
    import torch

    # Omit completed_updates: a valid ZIP/pickle container must not be enough
    # to certify that native managed training reached its configured boundary.
    torch.save(
        {
            "schema_version": 2,
            "checkpoint_kind": "latest",
            "config_hash": CONFIG_HASH,
            "model_state_dict": {"weight": torch.ones(1)},
        },
        run_dir / "checkpoints" / "latest.pt",
    )

    with pytest.raises(ValueError, match="missing.*completed_updates"):
        wrapper.resolve_managed_run(RUN_NAME, repo_root=tmp_path)


def test_incomplete_native_managed_run_is_rejected_before_analysis(
    monkeypatch, tmp_path
):
    _make_managed_run(tmp_path, schema_version=2, completed_updates=9)
    runner = RecordingRunner(tmp_path)

    with pytest.raises(RuntimeError, match="not complete"):
        wrapper.run_pipeline(
            RUN_NAME,
            only="collect",
            repo_root=tmp_path,
            runner=runner,
        )

    assert runner.calls == []
