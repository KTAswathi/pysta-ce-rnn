"""Run the frozen ABCD reference-analysis pipeline for one managed run.

The sole positional input is the exact basename of a directory below
``models/abcd_fmri``.  This orchestration layer always selects
``checkpoints/best.pt`` and invokes the four existing scientific entry points
as subprocesses; it contains no analysis estimator itself.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import pickletools
import re
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]


STAGE_ORDER = ("collect", "raw", "csubs", "local_rsa")
FOLDER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SCRIPT_NAMES = {
    "collect": "collect_abcd_reference_trials.py",
    "raw": "analyse_abcd_normalized_raw.py",
    "csubs": "analyse_abcd_normalized_csubs.py",
    "local_rsa": "analyse_abcd_local_rsa.py",
}
PIPELINE_STATUS_NAME = "pipeline_status.json"
PIPELINE_STATUS_SCHEMA = "abcd_reference_analysis_pipeline/v1"
ORCHESTRATOR_SOURCE_FILE = (
    "scripts/ABCD_task/abcd_reference_analysis_orchestrator.py"
)

# These are execution-provenance inputs, not additional estimator choices.  A
# completed stage is reusable only while the frozen entry points and their
# direct shared implementation/training-reconstruction dependencies are
# byte-identical.  This keeps ``--skip-existing`` conservative without moving
# any scientific code into this wrapper.
COLLECTION_SOURCE_FILES = (
    "scripts/ABCD_task/collect_abcd_reference_trials.py",
    "scripts/ABCD_task/abcd_analysis_common.py",
    "pysta/__init__.py",
    "pysta/abcd_env.py",
    "pysta/abcd_analysis_utils.py",
    "pysta/agents.py",
    "pysta/argparser.py",
    "pysta/embedding/mpfc_embedding.py",
    "pysta/embedding/sampling.py",
    "pysta/embedding/ultimate_surface.py",
    "pysta/envs.py",
    "pysta/run_manager.py",
    "pysta/tasks.py",
    "pysta/train_rnn.py",
    "pysta/utils.py",
)
STAGE_SOURCE_FILES = {
    "collect": (*COLLECTION_SOURCE_FILES, ORCHESTRATOR_SOURCE_FILE),
    "raw": (
        "scripts/ABCD_task/analyse_abcd_normalized_raw.py",
        "scripts/ABCD_task/abcd_analysis_common.py",
        ORCHESTRATOR_SOURCE_FILE,
    ),
    "csubs": (
        "scripts/ABCD_task/analyse_abcd_normalized_csubs.py",
        "scripts/ABCD_task/abcd_analysis_common.py",
        "scripts/ABCD_task/abcd_csubs_decoder.py",
        ORCHESTRATOR_SOURCE_FILE,
    ),
    "local_rsa": (
        "scripts/ABCD_task/analyse_abcd_local_rsa.py",
        "scripts/ABCD_task/abcd_analysis_common.py",
        ORCHESTRATOR_SOURCE_FILE,
    ),
}


@dataclass(frozen=True)
class PipelinePaths:
    """Canonical model, collection, and output paths for one managed run."""

    repo_root: Path
    folder_name: str
    run_dir: Path
    config: Path
    checkpoint: Path
    analysis_root: Path
    trial_collection: Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf8") as stream:
        value = json.load(stream)
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected a mapping in {path}.")
    return dict(value)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    """Atomically replace one small wrapper-owned provenance document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _reject_symlinks_below(path: Path) -> None:
    """Reject existing symlinks anywhere in the canonical output tree."""

    if path.is_symlink():
        raise ValueError(f"Analysis output path may not be a symlink: {path}")
    if not path.exists():
        return
    for directory, directories, files in os.walk(path, followlinks=False):
        directory_path = Path(directory)
        for name in (*directories, *files):
            candidate = directory_path / name
            if candidate.is_symlink():
                raise ValueError(
                    f"Analysis output tree may not contain symlinks: {candidate}"
                )


def _validate_npz_container(path: Path) -> None:
    """Validate the ZIP container without materializing its numerical arrays."""

    if path.stat().st_size <= 0 or not zipfile.is_zipfile(path):
        raise ValueError(f"Expected a non-empty NPZ archive: {path}")
    with zipfile.ZipFile(path, "r") as archive:
        members = archive.namelist()
        if not members or not all(name.endswith(".npy") for name in members):
            raise ValueError(f"NPZ archive has no valid array members: {path}")
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"Corrupt NPZ member {bad_member!r} in {path}")


def _validate_png(path: Path) -> None:
    with path.open("rb") as stream:
        signature = stream.read(8)
    if path.stat().st_size <= 8 or signature != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Expected a non-empty PNG image: {path}")


def _validate_config_identity(document: Mapping[str, object], path: Path) -> None:
    """Recompute the schema-aware identity without importing analysis code."""

    training = document.get("training_config")
    config_hash = document.get("config_hash")
    if not isinstance(training, Mapping) or not isinstance(config_hash, Mapping):
        raise ValueError(f"Managed config identity is incomplete: {path}")
    stored = config_hash.get("full")
    if not isinstance(stored, str):
        raise ValueError(f"Managed config has no full identity hash: {path}")
    schema_version = int(document.get("schema_version", 1))
    if schema_version >= 2:
        arguments = training.get("arguments")
        derived = training.get("derived")
        if not isinstance(arguments, Mapping) or not isinstance(derived, Mapping):
            raise ValueError(f"Schema-v2 managed identity is incomplete: {path}")
        identity: object = {"arguments": arguments, "derived": derived}
    else:
        identity = training
    recomputed = hashlib.sha256(_canonical_json(identity).encode("utf8")).hexdigest()
    if recomputed != stored:
        raise ValueError(
            "Managed config identity hash mismatch: "
            f"stored={stored!r}, recomputed={recomputed!r}, config={path}."
        )


def _validate_frozen_managed_run(
    document: Mapping[str, object], run_dir: Path
) -> None:
    """Refuse a native run whose best checkpoint may still be changing."""

    legacy_import = document.get("legacy_import")
    if isinstance(legacy_import, Mapping) and legacy_import.get(
        "resume_supported"
    ) is False:
        # Imported historical snapshots are explicitly frozen/non-resumable;
        # their recorded update count need not equal the originally requested
        # training length.
        return

    training = document.get("training_config")
    config_hash = document.get("config_hash")
    if not isinstance(training, Mapping) or not isinstance(config_hash, Mapping):
        raise ValueError(f"Managed run metadata is incomplete: {run_dir}")
    arguments = training.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError(f"Managed run arguments are incomplete: {run_dir}")
    expected_updates = int(arguments.get("num_epochs", -1))
    latest = run_dir / "checkpoints" / "latest.pt"
    if latest.is_symlink() or not latest.is_file():
        raise FileNotFoundError(
            "A native managed run must have checkpoints/latest.pt so completion "
            f"can be verified before analysis: {latest}"
        )
    try:
        payload = _read_managed_checkpoint_header(latest)
    except Exception as error:
        raise ValueError(
            f"Could not safely inspect managed latest.pt: {latest}. "
            f"Underlying error: {type(error).__name__}: {error}"
        ) from error
    if payload.get("checkpoint_kind") != "latest":
        raise ValueError(f"Not a managed latest checkpoint: {latest}")
    completed = int(payload.get("completed_updates", -1))
    if payload.get("config_hash") != config_hash.get("full"):
        raise ValueError(f"latest.pt does not match config.yaml: {latest}")
    if expected_updates < 0 or completed != expected_updates:
        raise RuntimeError(
            "Native managed run is not complete and its best checkpoint may still "
            f"change: completed_updates={completed}, configured={expected_updates}, "
            f"run={run_dir}."
        )


def _read_managed_checkpoint_header(path: Path) -> dict[str, object]:
    """Read scalar managed metadata without loading tensors or executing pickle.

    Managed ``latest.pt`` archives may contain optimizer/RNG tensor dtypes that
    older Torch versions cannot deserialize even with ``weights_only=True``.
    The completion guard does not need those objects: collection uses
    ``best.pt``.  This routine validates the ZIP CRCs and uses ``pickletools``
    (which disassembles but never executes pickle opcodes) to read the four
    scalar header fields written before ``model_state_dict`` by
    :func:`pysta.run_manager.checkpoint_payload`.
    """

    required = {
        "schema_version",
        "checkpoint_kind",
        "config_hash",
        "completed_updates",
    }
    string_ops = {"BINUNICODE", "SHORT_BINUNICODE", "UNICODE", "STRING"}
    integer_ops = {"BININT", "BININT1", "BININT2", "INT", "LONG", "LONG1", "LONG4"}
    memo_ops = {"BINPUT", "LONG_BINPUT", "PUT", "MEMOIZE"}

    if path.stat().st_size <= 0 or not zipfile.is_zipfile(path):
        raise ValueError("checkpoint is not a non-empty PyTorch ZIP archive")
    with zipfile.ZipFile(path, "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"checkpoint ZIP member {bad_member!r} failed CRC")
        pickle_members = [
            name
            for name in archive.namelist()
            if name == "data.pkl" or name.endswith("/data.pkl")
        ]
        if len(pickle_members) != 1:
            raise ValueError(
                f"expected one data.pkl member, found {len(pickle_members)}"
            )
        serialized = archive.read(pickle_members[0])
    if len(serialized) > 16 * 1024 * 1024:
        raise ValueError("managed checkpoint pickle header exceeds 16 MiB")

    operations = list(pickletools.genops(serialized))
    result: dict[str, object] = {}
    for index, (operation, argument, _position) in enumerate(operations):
        if operation.name in string_ops and argument == "model_state_dict":
            break
        if operation.name not in string_ops or argument not in required:
            continue
        if str(argument) in result:
            raise ValueError(f"duplicate managed header field {argument!r}")
        cursor = index + 1
        while cursor < len(operations) and operations[cursor][0].name in memo_ops:
            cursor += 1
        if cursor >= len(operations):
            raise ValueError(f"header field {argument!r} has no value")
        value_operation, value, _ = operations[cursor]
        if argument in {"checkpoint_kind", "config_hash"}:
            if value_operation.name not in string_ops or not isinstance(value, str):
                raise ValueError(f"header field {argument!r} is not a string")
            result[str(argument)] = value
        else:
            if value_operation.name not in integer_ops or isinstance(value, bool):
                raise ValueError(f"header field {argument!r} is not an integer")
            result[str(argument)] = int(value)
        if result.keys() >= required:
            break
    missing = required.difference(result)
    if missing:
        raise ValueError(f"managed checkpoint header is missing {sorted(missing)}")
    return result


def _validate_folder_name(folder_name: str) -> str:
    name = str(folder_name)
    if (
        not name
        or name in {".", ".."}
        or Path(name).is_absolute()
        or "/" in name
        or "\\" in name
        or not FOLDER_NAME_PATTERN.fullmatch(name)
    ):
        raise ValueError(
            "managed_folder must be one exact directory basename below "
            "models/abcd_fmri (not a path, prefix, or glob)."
        )
    return name


def resolve_managed_run(
    folder_name: str,
    *,
    repo_root: Path | str | None = None,
) -> PipelinePaths:
    """Resolve and validate an exact managed-folder basename and best checkpoint."""

    name = _validate_folder_name(folder_name)
    root = Path(REPO_ROOT if repo_root is None else repo_root).expanduser().resolve()
    managed_root = (root / "models" / "abcd_fmri").resolve()
    candidate = managed_root / name
    if candidate.is_symlink():
        raise ValueError(f"Managed run directory may not be a symlink: {candidate}")
    if not candidate.is_dir():
        raise FileNotFoundError(f"Managed run directory does not exist: {candidate}")
    run_dir = candidate.resolve()
    if run_dir.parent != managed_root:
        raise ValueError(f"Managed run escaped models/abcd_fmri: {candidate}")

    config = run_dir / "config.yaml"
    checkpoints = run_dir / "checkpoints"
    checkpoint = checkpoints / "best.pt"
    for path, kind in (
        (config, "config.yaml"),
        (checkpoints, "checkpoints directory"),
        (checkpoint, "checkpoints/best.pt"),
    ):
        if path.is_symlink():
            raise ValueError(f"Managed {kind} may not be a symlink: {path}")
    if not config.is_file():
        raise FileNotFoundError(f"Managed run is missing config.yaml: {run_dir}")
    if not checkpoints.is_dir():
        raise FileNotFoundError(f"Managed run is missing checkpoints/: {run_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "Managed run has no checkpoints/best.pt; this wrapper never falls "
            f"back to latest.pt: {run_dir}"
        )

    document = _load_json(config)
    _validate_config_identity(document, config)
    run = document.get("run")
    config_hash = document.get("config_hash")
    training = document.get("training_config")
    if not all(isinstance(value, Mapping) for value in (run, config_hash, training)):
        raise ValueError(f"Managed config is missing run/config identity: {config}")
    full_hash = config_hash.get("full")
    short_hash = config_hash.get("short")
    if (
        not isinstance(full_hash, str)
        or not isinstance(short_hash, str)
        or short_hash != full_hash[:10]
        or run.get("id") != short_hash
        or name != f"{run.get('name')}_{short_hash}"
    ):
        raise ValueError(
            "Managed folder/config name or hash binding is inconsistent: "
            f"{run_dir}"
        )
    arguments = training.get("arguments")
    if not isinstance(arguments, Mapping) or arguments.get("task") != "abcd_fmri":
        raise ValueError(f"Managed run is not an ABCD task configuration: {config}")
    if run.get("task") not in (None, "abcd_fmri"):
        raise ValueError(f"Managed run metadata is not task='abcd_fmri': {config}")
    _validate_frozen_managed_run(document, run_dir)

    data_root = root / "data"
    analysis_parent = data_root / "abcd_task_analyses"
    for path in (data_root, analysis_parent):
        if path.is_symlink():
            raise ValueError(f"Canonical analysis parent may not be a symlink: {path}")
    resolved_parent = analysis_parent.resolve()
    analysis_candidate = analysis_parent / name
    _reject_symlinks_below(analysis_candidate)
    analysis_root = analysis_candidate.resolve()
    if analysis_root.parent != resolved_parent:
        raise ValueError(
            "Managed analysis output escaped data/abcd_task_analyses: "
            f"{analysis_candidate}"
        )
    return PipelinePaths(
        repo_root=root,
        folder_name=name,
        run_dir=run_dir,
        config=config.resolve(),
        checkpoint=checkpoint.resolve(),
        analysis_root=analysis_root.resolve(),
        trial_collection=analysis_root / "trial_collection",
    )


# A descriptive alias for callers that think in folder rather than run terms.
resolve_run_folder = resolve_managed_run


def build_stage_commands(
    paths: PipelinePaths,
    *,
    python_executable: str | Path | None = None,
) -> dict[str, tuple[str, ...]]:
    """Build the exact, fixed commands for the four frozen entry points."""

    python = str(sys.executable if python_executable is None else python_executable)
    scripts = paths.repo_root / "scripts" / "ABCD_task"
    return {
        "collect": (
            python,
            str(scripts / SCRIPT_NAMES["collect"]),
            str(paths.checkpoint),
            "--output-dir",
            str(paths.analysis_root),
        ),
        "raw": (
            python,
            str(scripts / SCRIPT_NAMES["raw"]),
            str(paths.trial_collection),
            "--output-dir",
            str(paths.analysis_root / "raw_activity"),
        ),
        "csubs": (
            python,
            str(scripts / SCRIPT_NAMES["csubs"]),
            str(paths.analysis_root),
            "--n-permutations",
            "1000",
            "--permutation-seed",
            "881",
            "--device",
            "auto",
            "--max-iters",
            "2000",
        ),
        "local_rsa": (
            python,
            str(scripts / SCRIPT_NAMES["local_rsa"]),
            str(paths.analysis_root),
            "--repeat-1",
            str(paths.trial_collection / "repeat_01"),
            "--repeat-2",
            str(paths.trial_collection / "repeat_02"),
            "--radius-mm",
            "6.0",
        ),
    }


def _source_hashes(paths: PipelinePaths, stage: str) -> dict[str, str]:
    records: dict[str, str] = {}
    for relative in STAGE_SOURCE_FILES[stage]:
        source = paths.repo_root / relative
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"Missing frozen pipeline source: {source}")
        records[relative] = _sha256_file(source)
    return records


def _portable_command_contract(
    paths: PipelinePaths, command: Sequence[str]
) -> list[str]:
    """Remove host-specific absolute prefixes from a recorded command."""

    substitutions = (
        (str(paths.analysis_root), "$ANALYSIS_ROOT"),
        (str(paths.run_dir), "$RUN_DIR"),
        (str(paths.repo_root), "$REPO_ROOT"),
    )
    result: list[str] = []
    for index, raw in enumerate(command):
        value = str(raw)
        if index == 0:
            result.append("$PYTHON")
            continue
        for source, replacement in substitutions:
            if value == source or value.startswith(source + os.sep):
                value = replacement + value[len(source) :]
                break
        result.append(value)
    return result


def _stage_signature(
    paths: PipelinePaths,
    stage: str,
    commands: Mapping[str, Sequence[str]],
) -> dict[str, object]:
    signature: dict[str, object] = {
        "stage": stage,
        "command_contract": _portable_command_contract(paths, commands[stage]),
        "pipeline_source_sha256": _source_hashes(paths, stage),
        "checkpoint_sha256": _sha256_file(paths.checkpoint),
        "config_sha256": _sha256_file(paths.config),
    }
    manifest = paths.analysis_root / "analysis_manifest.json"
    if stage != "collect":
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing shared collection manifest: {manifest}")
        signature["input_manifest_sha256"] = _sha256_file(manifest)
    return signature


def _new_pipeline_status(paths: PipelinePaths) -> dict[str, object]:
    return {
        "schema": PIPELINE_STATUS_SCHEMA,
        "managed_folder": paths.folder_name,
        "source": {
            "checkpoint_sha256": _sha256_file(paths.checkpoint),
            "config_sha256": _sha256_file(paths.config),
        },
        "stages": {},
    }


def _load_pipeline_status(paths: PipelinePaths) -> dict[str, object]:
    status_path = paths.analysis_root / PIPELINE_STATUS_NAME
    if not status_path.is_file():
        return _new_pipeline_status(paths)
    document = _load_json(status_path)
    if document.get("schema") != PIPELINE_STATUS_SCHEMA:
        raise ValueError(f"Unsupported pipeline status document: {status_path}")
    if document.get("managed_folder") != paths.folder_name:
        raise ValueError(f"Pipeline status belongs to another managed run: {status_path}")
    expected_source = _new_pipeline_status(paths)["source"]
    if document.get("source") != expected_source:
        raise RuntimeError(
            "Pipeline status does not match the requested best checkpoint/config: "
            f"{status_path}"
        )
    stages = document.get("stages")
    if not isinstance(stages, Mapping):
        raise ValueError(f"Pipeline status has no stage mapping: {status_path}")
    document["stages"] = dict(stages)
    return document


def _save_pipeline_status(paths: PipelinePaths, status: Mapping[str, object]) -> None:
    _write_json_atomic(paths.analysis_root / PIPELINE_STATUS_NAME, status)


def _required_outputs(paths: PipelinePaths, stage: str) -> tuple[Path, ...]:
    if stage == "collect":
        collection = paths.trial_collection
        return (
            paths.analysis_root / "analysis_manifest.json",
            collection / "collection_qc.json",
            collection / "repeat_01" / "normalized_navigation.npz",
            collection / "repeat_01" / "behaviour_summary.json",
            collection / "repeat_02" / "normalized_navigation.npz",
            collection / "repeat_02" / "behaviour_summary.json",
        )
    directory = paths.analysis_root / {
        "raw": "raw_activity",
        "csubs": "csubs",
        "local_rsa": "local_rsa",
    }[stage]
    names = {
        "raw": ("results.npz", "summary.csv", "summary.png", "analysis.json"),
        "csubs": (
            "results.npz",
            "summary.csv",
            "summary.png",
            "fit_log.txt",
            "analysis.json",
        ),
        "local_rsa": ("results.npz", "summary.csv", "summary.png", "metadata.json"),
    }[stage]
    return tuple(directory / name for name in names)


def _collection_block_paths(paths: PipelinePaths) -> tuple[Path, ...]:
    return tuple(
        path
        for repeat in ("repeat_01", "repeat_02")
        for path in sorted(
            (paths.trial_collection / repeat / "blocks").glob("*.npz")
        )
    )


def _stage_output_paths(paths: PipelinePaths, stage: str) -> tuple[Path, ...]:
    required = _required_outputs(paths, stage)
    if stage == "collect":
        return (*required, *_collection_block_paths(paths))
    return required


def _artifact_records(paths: PipelinePaths, stage: str) -> dict[str, str]:
    records: dict[str, str] = {}
    for path in _stage_output_paths(paths, stage):
        if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
            raise ValueError(f"Missing, empty, or symlinked stage artifact: {path}")
        try:
            relative = path.relative_to(paths.analysis_root).as_posix()
        except ValueError as error:
            raise ValueError(f"Stage artifact escaped canonical output root: {path}") from error
        records[relative] = _sha256_file(path)
    return records


def _collection_fingerprint(paths: PipelinePaths) -> str:
    """Hash the realized collection, including both stores and every block."""

    return hashlib.sha256(
        _canonical_json(_artifact_records(paths, "collect")).encode("utf8")
    ).hexdigest()


def _validate_stage_artifacts(paths: PipelinePaths, stage: str) -> None:
    """Perform cheap structural/QC checks before certifying or reusing a stage."""

    for path in _required_outputs(paths, stage):
        if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
            raise ValueError(f"Incomplete {stage} output: {path}")

    manifest_path = paths.analysis_root / "analysis_manifest.json"
    if stage == "collect":
        _validate_collection_provenance(paths, require_complete=True)
        manifest = _load_json(manifest_path)
        task = manifest.get("task")
        if not isinstance(task, Mapping):
            raise ValueError(f"Collection manifest has no task metadata: {manifest_path}")
        expected_blocks = int(task.get("num_factorial_blocks", 0))
        if expected_blocks <= 0:
            raise ValueError(f"Invalid factorial block count in {manifest_path}")
        qc_path = paths.trial_collection / "collection_qc.json"
        qc = _load_json(qc_path)
        for field in (
            "weights_unchanged",
            "same_trained_recurrent_noise",
            "distinct_recurrent_noise_seeds",
            "matched_factorial_cells",
            "matched_start_locations",
        ):
            if qc.get(field) is not True:
                raise ValueError(f"Collection QC invariant {field!r} failed: {qc_path}")
        repeats = qc.get("repeats")
        if not isinstance(repeats, list) or len(repeats) != 2:
            raise ValueError(f"Collection QC must contain exactly two repeats: {qc_path}")
        for repeat_index, repeat in enumerate(("repeat_01", "repeat_02"), start=1):
            repeat_dir = paths.trial_collection / repeat
            summary_path = repeat_dir / "behaviour_summary.json"
            summary = _load_json(summary_path)
            if (
                summary.get("repeat_index") != repeat_index
                or summary.get("autonomous") is not True
                or summary.get("greedy") is not True
                or summary.get("force_optimal") is not False
                or summary.get("weights_frozen") is not True
                or int(summary.get("num_blocks", -1)) != expected_blocks
                or int(summary.get("completed_blocks", -1)) != expected_blocks
            ):
                raise ValueError(f"Invalid collection repeat summary: {summary_path}")
            if repeats[repeat_index - 1] != summary:
                raise ValueError(
                    f"Collection QC and repeat summary disagree: {summary_path}"
                )
            normalized_path = repeat_dir / "normalized_navigation.npz"
            _validate_npz_container(normalized_path)
            block_paths = tuple(sorted((repeat_dir / "blocks").glob("*.npz")))
            if len(block_paths) != expected_blocks:
                raise ValueError(
                    f"Expected {expected_blocks} block archives in {repeat_dir}, "
                    f"found {len(block_paths)}."
                )
            for block_path in block_paths:
                _validate_npz_container(block_path)
        return

    result, summary_csv, summary_png, metadata_path = (
        _required_outputs(paths, stage)[:4]
        if stage != "csubs"
        else (
            _required_outputs(paths, stage)[0],
            _required_outputs(paths, stage)[1],
            _required_outputs(paths, stage)[2],
            _required_outputs(paths, stage)[4],
        )
    )
    _validate_npz_container(result)
    if not summary_csv.read_text(encoding="utf8").strip():
        raise ValueError(f"Empty analysis summary table: {summary_csv}")
    _validate_png(summary_png)
    metadata = _load_json(metadata_path)
    manifest_hash = _sha256_file(manifest_path)
    if metadata.get("source_manifest_sha256") != manifest_hash:
        raise ValueError(
            f"{stage} metadata is not bound to the shared collection manifest: "
            f"{metadata_path}"
        )
    expected_analysis = {
        "raw": "nuisance_controlled_normalized_progress_raw_activity",
        "csubs": "normalized_progress_cross_fitted_csubs",
        "local_rsa": "reference_matched_local_rsa",
    }[stage]
    if metadata.get("analysis") != expected_analysis:
        raise ValueError(f"Unexpected {stage} analysis identity: {metadata_path}")
    if stage == "csubs":
        if int(metadata.get("n_permutations", -1)) != 1000:
            raise ValueError(f"Unexpected Csubs permutation count: {metadata_path}")
        settings = metadata.get("settings")
        if not isinstance(settings, Mapping) or int(settings.get("max_iters", -1)) != 2000:
            raise ValueError(f"Unexpected Csubs fit settings: {metadata_path}")
    if stage == "local_rsa" and float(
        metadata.get("searchlight_radius_surface_mm", float("nan"))
    ) != 6.0:
        raise ValueError(f"Unexpected local-RSA radius: {metadata_path}")


def stage_status(
    paths: PipelinePaths,
    stage: str,
    *,
    status_document: Mapping[str, object] | None = None,
    commands: Mapping[str, Sequence[str]] | None = None,
) -> str:
    """Return ``absent``, ``partial``, or provenance-certified ``complete``."""

    if stage not in STAGE_ORDER:
        raise ValueError(f"Unknown pipeline stage: {stage}")
    required = _required_outputs(paths, stage)
    if stage == "collect":
        footprint = paths.trial_collection.exists() or required[0].exists()
    else:
        directory = required[0].parent
        footprint = directory.is_dir() and any(directory.iterdir())
    document = (
        _load_pipeline_status(paths)
        if status_document is None
        else dict(status_document)
    )
    stage_records = document.get("stages", {})
    record = stage_records.get(stage) if isinstance(stage_records, Mapping) else None
    if isinstance(record, Mapping) and record.get("status") == "complete":
        command_map = build_stage_commands(paths) if commands is None else commands
        try:
            signature = _stage_signature(paths, stage, command_map)
            if stage != "collect":
                signature["collection_fingerprint"] = _collection_fingerprint(paths)
            _validate_stage_artifacts(paths, stage)
            outputs = _artifact_records(paths, stage)
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile):
            pass
        else:
            if record.get("signature") == signature and record.get("outputs") == outputs:
                return "complete"
    return "partial" if footprint or record is not None else "absent"


def _validate_collection_provenance(
    paths: PipelinePaths,
    *,
    require_complete: bool,
) -> None:
    manifest_path = paths.analysis_root / "analysis_manifest.json"
    if not manifest_path.is_file():
        if require_complete:
            raise FileNotFoundError(f"Missing analysis manifest: {manifest_path}")
        return
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != "abcd_reference_analysis/v1":
        raise ValueError(f"Unsupported or incomplete analysis manifest: {manifest_path}")
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise ValueError(f"Analysis manifest has no source binding: {manifest_path}")
    checkpoint_hash = _sha256_file(paths.checkpoint)
    config_hash = _sha256_file(paths.config)
    checkpoint_record = source.get("checkpoint_identifier")
    config_record = source.get("resolved_config", source.get("portable_kwargs"))
    if (
        not isinstance(checkpoint_record, Mapping)
        or checkpoint_record.get("sha256") != checkpoint_hash
        or not isinstance(config_record, Mapping)
        or config_record.get("sha256") != config_hash
    ):
        raise RuntimeError(
            "Existing trial collection does not match the requested managed "
            f"best checkpoint/config: {paths.analysis_root}"
        )
    qc_path = paths.trial_collection / "collection_qc.json"
    if require_complete:
        qc = _load_json(qc_path)
        if (
            qc.get("checkpoint_sha256") != checkpoint_hash
            or qc.get("resolved_config_sha256") != config_hash
        ):
            raise RuntimeError(
                "Trial-collection QC does not match the requested managed "
                f"best checkpoint/config: {qc_path}"
            )


@contextlib.contextmanager
def _exclusive_analysis_lock(paths: PipelinePaths):
    """Prevent concurrent wrappers from writing the same analysis root."""

    if paths.analysis_root.is_symlink():
        raise ValueError(
            f"Analysis output path may not be a symlink: {paths.analysis_root}"
        )
    paths.analysis_root.mkdir(parents=True, exist_ok=True)
    if (
        paths.analysis_root.is_symlink()
        or paths.analysis_root.resolve().parent != paths.analysis_root.parent.resolve()
    ):
        raise ValueError(
            f"Analysis output escaped its canonical parent: {paths.analysis_root}"
        )
    lock_path = paths.analysis_root / ".run_abcd_reference_analysis.lock"
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise FileExistsError(
            "Another wrapper may be using this analysis root. The lock is never "
            f"removed automatically as stale: {lock_path}"
        ) from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf8") as stream:
            stream.write(f"pid={os.getpid()}\n")
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        if lock_path.is_file() and not lock_path.is_symlink():
            lock_path.unlink()


def run_pipeline(
    folder_name: str,
    *,
    skip_existing: bool = False,
    force: bool = False,
    only: str | None = None,
    repo_root: Path | str | None = None,
    runner: Callable[..., object] | None = None,
) -> Path:
    """Run selected frozen stages and return the canonical analysis root."""

    if skip_existing and force:
        raise ValueError("--skip-existing and --force are mutually exclusive.")
    if only is not None and only not in STAGE_ORDER:
        raise ValueError(f"Unknown --only stage: {only}")
    paths = resolve_managed_run(folder_name, repo_root=repo_root)
    commands = build_stage_commands(paths)
    selected = (only,) if only is not None else STAGE_ORDER
    invoke = subprocess.run if runner is None else runner
    initial_checkpoint_hash = _sha256_file(paths.checkpoint)
    initial_config_hash = _sha256_file(paths.config)

    def require_stable_source() -> None:
        if (
            _sha256_file(paths.checkpoint) != initial_checkpoint_hash
            or _sha256_file(paths.config) != initial_config_hash
        ):
            raise RuntimeError(
                "Managed best.pt or config.yaml changed while the frozen "
                "analysis pipeline was running; outputs must not be treated "
                "as valid."
            )

    with _exclusive_analysis_lock(paths):
        _reject_symlinks_below(paths.analysis_root)
        status_document = _load_pipeline_status(paths)
        stage_records = status_document["stages"]
        if not isinstance(stage_records, dict):  # normalized by loader
            raise ValueError("Pipeline status stages must be a mutable mapping.")

        if only in {"raw", "csubs", "local_rsa"}:
            if stage_status(
                paths,
                "collect",
                status_document=status_document,
                commands=commands,
            ) != "complete":
                raise FileNotFoundError(
                    f"--only {only} requires a complete, provenance-certified "
                    f"shared trial collection at {paths.trial_collection}."
                )
            _validate_collection_provenance(paths, require_complete=True)

        for stage in selected:
            status = stage_status(
                paths,
                stage,
                status_document=status_document,
                commands=commands,
            )
            if stage != "collect":
                if stage_status(
                    paths,
                    "collect",
                    status_document=status_document,
                    commands=commands,
                ) != "complete":
                    raise FileNotFoundError(
                        f"Stage {stage} requires the provenance-certified shared "
                        f"collection at {paths.trial_collection}."
                    )
                _validate_collection_provenance(paths, require_complete=True)

            if skip_existing and status == "complete":
                print(f"[{stage}] complete; skipping")
                if stage == "collect":
                    _validate_collection_provenance(paths, require_complete=True)
                continue
            if not skip_existing and not force and status != "absent":
                raise FileExistsError(
                    f"Stage {stage} has {status} output. Use --skip-existing to "
                    "continue around complete stages or --force to rerun it: "
                    f"{paths.analysis_root}"
                )
            if stage == "collect" and status != "absent":
                _validate_collection_provenance(paths, require_complete=False)

            if stage == "collect":
                # A recollection replaces the one shared input.  Any downstream
                # completion certificate is stale immediately, even if old
                # result files remain on disk after --only collect or failure.
                for downstream in STAGE_ORDER[1:]:
                    if downstream in stage_records or stage_status(
                        paths,
                        downstream,
                        status_document=status_document,
                        commands=commands,
                    ) != "absent":
                        stage_records[downstream] = {
                            "status": "stale",
                            "reason": "trial collection was rerun",
                        }

            signature = _stage_signature(paths, stage, commands)
            if stage != "collect":
                signature["collection_fingerprint"] = _collection_fingerprint(paths)
            stage_records[stage] = {
                "status": "running",
                "signature": signature,
            }
            _save_pipeline_status(paths, status_document)

            print(f"[{stage}] running")
            try:
                invoke(
                    list(commands[stage]),
                    cwd=str(paths.repo_root),
                    check=True,
                )
                require_stable_source()
                _validate_stage_artifacts(paths, stage)
                end_signature = _stage_signature(paths, stage, commands)
                if stage != "collect":
                    end_signature["collection_fingerprint"] = (
                        _collection_fingerprint(paths)
                    )
                if end_signature != signature:
                    raise RuntimeError(
                        f"Frozen inputs changed while stage {stage} was running."
                    )
                outputs = _artifact_records(paths, stage)
            except BaseException as error:
                stage_records[stage] = {
                    "status": "failed",
                    "signature": signature,
                    "error_type": type(error).__name__,
                }
                _save_pipeline_status(paths, status_document)
                raise

            stage_records[stage] = {
                "status": "complete",
                "signature": signature,
                "outputs": outputs,
            }
            _save_pipeline_status(paths, status_document)

        require_stable_source()
    return paths.analysis_root


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "managed_folder",
        help=(
            "Exact folder basename below models/abcd_fmri, for example "
            "mech_baseline_seed1_a1b2c3d4e5. Paths and prefixes are not accepted."
        ),
    )
    behavior = parser.add_mutually_exclusive_group()
    behavior.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip only provenance-certified, structurally complete stages.",
    )
    behavior.add_argument(
        "--force",
        action="store_true",
        help="Rerun selected stages and replace their standard output files.",
    )
    parser.add_argument(
        "--only",
        choices=STAGE_ORDER,
        default=None,
        help="Run exactly one stage; analysis-only choices require collection first.",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    repo_root: Path | str | None = None,
    runner: Callable[..., object] | None = None,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        output = run_pipeline(
            args.managed_folder,
            skip_existing=args.skip_existing,
            force=args.force,
            only=args.only,
            repo_root=repo_root,
            runner=runner,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(f"ABCD reference analysis root: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
