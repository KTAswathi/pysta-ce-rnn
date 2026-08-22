"""Short, immutable run directories for future ABCD training launches.

The managed layout is deliberately opt-in.  Existing commands that do not
provide ``--run_name`` continue to use the historical directory/checkpoint
contract unchanged.
"""

from __future__ import annotations

import copy
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import re
import socket
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SCHEMA_VERSION = 2
SHORT_HASH_LENGTH = 10
RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$")

# These options control where/how a launch is written; they cannot alter the
# trained function.  Everything else in the resolved argument mapping is part
# of the immutable configuration identity.
NON_CONFIG_ARGUMENTS = frozenset(
    {
        "overwrite",
        "prefix",
        "resume",
        "run_name",
        "save_results",
    }
)

SOURCE_FILES = (
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

# Exact hashes for every direct training/run-management dependency are part of
# the strict resume contract.  They remain outside experiment identity: source
# changes cannot redirect the same scientific configuration to a new hash, but
# they also cannot silently continue an existing optimizer/RNG trajectory.
RESUME_CRITICAL_SOURCE_FILES = SOURCE_FILES

def json_safe(value: Any) -> Any:
    """Convert resolved configuration values to canonical JSON values."""
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("Managed-run configuration cannot contain NaN/Inf.")
        return value
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if torch.is_tensor(value):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((json_safe(item) for item in value), key=repr)
    raise TypeError(
        f"Unsupported managed-run configuration value {value!r} "
        f"({type(value).__name__})."
    )


def canonical_json(value: Any) -> str:
    """Return the stable byte-equivalent representation used for hashes."""
    return json.dumps(
        json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def metrics_json_safe(value: Any) -> Any:
    """JSON conversion for observations, mapping undefined NaN/Inf to null."""
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if torch.is_tensor(value):
        return metrics_json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return metrics_json_safe(value.tolist())
    if isinstance(value, np.generic):
        return metrics_json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): metrics_json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [metrics_json_safe(item) for item in value]
    if isinstance(value, list):
        return [metrics_json_safe(item) for item in value]
    return json_safe(value)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def validate_run_name(run_name: str) -> str:
    run_name = str(run_name)
    if not RUN_NAME_PATTERN.fullmatch(run_name):
        raise ValueError(
            "--run_name must be 1-48 characters, start with an alphanumeric "
            "character, and contain only letters, numbers, '.', '_' or '-'."
        )
    return run_name


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    # Avoid forking ``git`` from a process that may already have initialized
    # torch/OpenMP runtimes.  The exact critical-source hashes above make the
    # dirty-state ambiguity harmless for reproducibility.
    git_dir = repo_root / ".git"
    if git_dir.is_file():
        text = git_dir.read_text(encoding="utf8").strip()
        git_dir = (repo_root / text.removeprefix("gitdir:").strip()).resolve()
    commit = None
    head = git_dir / "HEAD"
    if head.is_file():
        head_value = head.read_text(encoding="utf8").strip()
        if head_value.startswith("ref:"):
            reference = head_value.split(":", 1)[1].strip()
            ref_path = git_dir / reference
            if ref_path.is_file():
                commit = ref_path.read_text(encoding="utf8").strip()
            else:
                packed_refs = git_dir / "packed-refs"
                if packed_refs.is_file():
                    for line in packed_refs.read_text(encoding="utf8").splitlines():
                        if line and not line.startswith(("#", "^")):
                            candidate, name = line.split(" ", 1)
                            if name == reference:
                                commit = candidate
                                break
        else:
            commit = head_value
    return {
        "commit": commit,
        "tracked_worktree_dirty": None,
        "dirty_state_note": (
            "not inferred with a subprocess; exact critical-source file hashes "
            "are stored in unhashed provenance and checked on resume"
        ),
    }


def source_fingerprint(
    repo_root: Path, *, files: Sequence[str] = SOURCE_FILES
) -> dict[str, Any]:
    """Fingerprint selected source files without making them experiment identity."""
    repo_root = Path(repo_root).resolve()
    file_hashes: dict[str, str] = {}
    for relative in files:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing training source file: {path}")
        file_hashes[relative] = sha256_file(path)
    return {
        "algorithm": "sha256",
        "combined_sha256": sha256_json(file_hashes),
        "files": file_hashes,
    }


def normalize_training_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return only resolved scientific/training configuration identity.

    This also normalizes schema-v1 configurations, where source, package
    versions, and legacy artifact hashes were accidentally nested in (and
    therefore hashed with) the scientific configuration.
    """
    # The projection is explicit rather than "everything except provenance":
    # future display/operational fields therefore cannot accidentally become
    # directory identity.  Both mappings are present in every schema-v2 run.
    arguments = config.get("arguments", {})
    derived = config.get("derived", {})
    if not isinstance(arguments, Mapping) or not isinstance(derived, Mapping):
        raise TypeError("training_config arguments and derived must be mappings")
    return {
        "arguments": json_safe(arguments),
        "derived": json_safe(derived),
    }


def experiment_config_hash(config: Mapping[str, Any]) -> str:
    """Hash only the resolved scientific/training configuration."""
    return sha256_json(normalize_training_config(config))


def runtime_provenance(training_device: Any) -> dict[str, Any]:
    """Record the selected runtime/device facts used by strict resume checks."""
    device = str(training_device)
    selected_device: dict[str, Any] = {"specifier": device}
    if device.startswith("cuda"):
        selected_device.update(
            {
                "torch_cuda_version": torch.version.cuda,
                "cudnn_version": (
                    None
                    if not torch.backends.cudnn.is_available()
                    else int(torch.backends.cudnn.version())
                ),
            }
        )
        if torch.cuda.is_available():
            index = torch.device(device).index
            if index is None:
                index = torch.cuda.current_device()
            selected_device.update(
                {
                    "index": int(index),
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    elif device.startswith("mps"):
        selected_device["mps_available"] = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    elif device.startswith("cpu"):
        selected_device.update(
            {
                "machine": platform.machine(),
                "processor": platform.processor(),
                "logical_cpu_count": os.cpu_count(),
            }
        )

    backend_settings: dict[str, Any] = {
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "default_dtype": str(torch.get_default_dtype()),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    if hasattr(torch, "is_deterministic_algorithms_warn_only_enabled"):
        backend_settings["deterministic_warn_only"] = bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        )
    if hasattr(torch.backends, "cuda") and hasattr(
        torch.backends.cuda, "matmul"
    ):
        backend_settings["cuda_matmul_allow_tf32"] = bool(
            torch.backends.cuda.matmul.allow_tf32
        )

    execution_environment = {
        name: os.environ.get(name)
        for name in (
            "CUBLAS_WORKSPACE_CONFIG",
            "CUDA_VISIBLE_DEVICES",
            "MKL_NUM_THREADS",
            "OMP_NUM_THREADS",
            "PYTORCH_ENABLE_MPS_FALLBACK",
        )
    }
    return {
        "dependencies": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
        },
        "platform": platform.platform(),
        "device": selected_device,
        "backend_settings": backend_settings,
        "execution_environment": execution_environment,
    }


def build_execution_provenance(
    repo_root: Path, launch_controls: Mapping[str, Any]
) -> dict[str, Any]:
    """Build unhashed execution provenance for creation/resume validation."""
    repo_root = Path(repo_root).resolve()
    source = source_fingerprint(repo_root, files=RESUME_CRITICAL_SOURCE_FILES)
    runtime = runtime_provenance(launch_controls.get("training_device"))
    compatibility = {
        "source": source,
        "runtime_dependencies": runtime["dependencies"],
        "platform": runtime["platform"],
        "device": runtime["device"],
        "backend_settings": runtime["backend_settings"],
        "execution_environment": runtime["execution_environment"],
    }
    return {
        "compatibility": compatibility,
        "compatibility_hash": {
            "algorithm": "sha256",
            "full": sha256_json(compatibility),
        },
        "informational": {
            "git": _git_provenance(repo_root),
            "argv": list(sys.argv),
            "cwd": str(Path.cwd().resolve()),
            "hostname": socket.gethostname(),
        },
        "resume_rule": (
            "compatibility_hash must match exactly before model, optimizer, "
            "environment, or RNG state is restored"
        ),
        "hardware_scope_note": (
            "hostname is informational; the strict signature records the "
            "selected accelerator/CPU descriptors and backend/thread settings, "
            "but is not a serial-number-level hardware identity"
        ),
    }


def _index_record(mask: torch.Tensor) -> dict[str, Any]:
    indices = torch.where(mask.detach().cpu().to(dtype=torch.bool))[0]
    return {
        "count": int(indices.numel()),
        "indices_sha256": sha256_tensor(indices.to(dtype=torch.int64)),
    }


def derive_runtime_config(
    env: Any,
    model: torch.nn.Module,
    *,
    eval_env: Any | None = None,
    fmri_factorial_schedule: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Record instantiated facts that are otherwise hidden behind defaults."""
    result: dict[str, Any] = {
        "environment_class": f"{type(env).__module__}.{type(env).__qualname__}",
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "observation_dim": int(env.obs_dim),
        "output_dim": int(env.output_dim),
        "batch_size": int(env.batch),
        "recurrent_units": int(model.Nrec),
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "trainable_parameter_count": int(
            sum(value.numel() for value in model.parameters() if value.requires_grad)
        ),
    }

    configuration_bank = getattr(env, "configuration_bank", None)
    if configuration_bank is not None:
        result["training_configuration_bank"] = json_safe(configuration_bank)
    for name in (
        "seed",
        "num_loops",
        "instruction_repeats",
        "max_navigation_steps",
        "min_manhattan_distance",
        "start_policy",
        "fixed_start",
        "allowed_instruction_directions",
        "allowed_execution_relations",
    ):
        if hasattr(env, name):
            result[f"effective_{name}"] = json_safe(getattr(env, name))
    if eval_env is not None:
        eval_bank = getattr(eval_env, "configuration_bank", None)
        if eval_bank is not None:
            result["evaluation_configuration_bank"] = json_safe(eval_bank)
        result["effective_evaluation_task_seed"] = int(eval_env.seed)
        result["effective_evaluation_bank_name"] = str(eval_env.bank_name)
    if fmri_factorial_schedule is not None:
        result["final_factorial_evaluation"] = {
            "num_cells": len(fmri_factorial_schedule),
            "base_configurations": json_safe(
                tuple(
                    cell.configuration
                    for index, cell in enumerate(fmri_factorial_schedule)
                    if index % 4 == 0
                )
            ),
            "cell_seeds": [int(cell.seed) for cell in fmri_factorial_schedule],
        }

    routing: dict[str, Any] = {}
    input_buffers = getattr(model, "input_mask_buffers", None)
    if isinstance(input_buffers, Mapping):
        for group, buffer_name in sorted(input_buffers.items()):
            group_mask = getattr(model, str(buffer_name))
            recipients = torch.any(group_mask.detach() != 0, dim=1)
            routing[str(group)] = _index_record(recipients)
    if routing:
        result["input_routing"] = routing

    for name in ("same_end_unit_mask", "anatomical_anchor_unit_indices"):
        value = getattr(model, name, None)
        if value is None:
            continue
        if name.endswith("unit_mask"):
            result[name] = _index_record(torch.as_tensor(value))
        else:
            indices = torch.as_tensor(value, dtype=torch.int64)
            result[name] = {
                "count": int(indices.numel()),
                "indices_sha256": sha256_tensor(indices),
            }

    for name in ("sampled_vertex_indices", "distance_matrix"):
        value = getattr(model, name, None)
        if torch.is_tensor(value):
            result[f"{name}_sha256"] = sha256_tensor(value)
    return result


def build_model_summary(
    arguments: Mapping[str, Any], derived: Mapping[str, Any]
) -> dict[str, Any]:
    """Select effective scientific/training controls for quick comparison."""
    configuration_bank = derived.get("training_configuration_bank", [])
    evaluation_bank = derived.get("evaluation_configuration_bank", [])
    spatial_routing = derived.get("input_routing", {}).get("current_location")
    base_configuration_seed = int(arguments.get("configuration_seed") or 0)
    train_configuration_seed = arguments.get("train_configuration_seed")
    if train_configuration_seed is None:
        train_configuration_seed = base_configuration_seed
    eval_configuration_seed = arguments.get("eval_configuration_seed")
    if eval_configuration_seed is None and arguments.get("evaluation_mode") == "heldout":
        eval_configuration_seed = base_configuration_seed + 1
    return {
        "identity": {
            "task": arguments.get("task"),
            "model_type": arguments.get("model_type"),
            "model_class": derived.get("model_class"),
            "seed": arguments.get("seed"),
            "Nrec": derived.get("recurrent_units"),
            "Nin": derived.get("observation_dim"),
            "Nout": derived.get("output_dim"),
            "parameter_count": derived.get("parameter_count"),
        },
        "optimization": {
            "batch_size": arguments.get("batch_size"),
            "learning_rate": arguments.get("lrate"),
            "num_epochs": arguments.get("num_epochs"),
            "eval_freq": arguments.get("eval_freq"),
            "num_eval": arguments.get("num_eval"),
            "evaluation_mode": arguments.get("evaluation_mode"),
            "force_optimal": arguments.get("force_optimal"),
        },
        "task": {
            "n_loops": derived.get("effective_num_loops"),
            "instruction_repeats": derived.get("effective_instruction_repeats"),
            "base_configuration_seed": base_configuration_seed,
            "train_configuration_seed": int(train_configuration_seed),
            "evaluation_configuration_seed": (
                None
                if eval_configuration_seed is None
                else int(eval_configuration_seed)
            ),
            "train_task_seed": derived.get("effective_seed"),
            "evaluation_task_seed": derived.get("effective_evaluation_task_seed"),
            "configuration_count": len(configuration_bank),
            "configuration_bank_sha256": (
                sha256_json(configuration_bank) if configuration_bank else None
            ),
            "evaluation_configuration_count": len(evaluation_bank),
            "evaluation_configuration_bank_sha256": (
                sha256_json(evaluation_bank) if evaluation_bank else None
            ),
            "instruction_directions": derived.get(
                "effective_allowed_instruction_directions"
            ),
            "execution_relations": derived.get(
                "effective_allowed_execution_relations"
            ),
            "start_position_policy": derived.get("effective_start_policy"),
            "max_navigation_steps": derived.get("effective_max_navigation_steps"),
            "min_goal_distance": derived.get("effective_min_manhattan_distance"),
            "final_factorial_evaluation": derived.get(
                "final_factorial_evaluation"
            ),
        },
        "dynamics_and_losses": {
            "tau": arguments.get("tau"),
            "iters_per_action": arguments.get("iters_per_action"),
            "rec_noise": arguments.get("rec_noise"),
            "r_reg": arguments.get("r_reg"),
            "W_reg": arguments.get("W_reg"),
            "ent_reg": arguments.get("ent_reg"),
            "dist_reg": arguments.get("dist_reg"),
        },
        "cortical_mechanism": {
            "embedding_name": arguments.get("embedding_name"),
            "embedding_species": arguments.get("embedding_species"),
            "embedding_seed": arguments.get("embedding_seed"),
            "anchor_area_names_fallback": arguments.get("anchor_area_names"),
            "anatomical_anchor": derived.get("anatomical_anchor_unit_indices"),
            "local_fraction": arguments.get("local_fraction"),
            "actual_spatial_input_recipients": spatial_routing,
            "use_local_init": arguments.get("use_local_init"),
            "line_decay": arguments.get("line_decay"),
            "line_init_scale": arguments.get("line_init_scale"),
            "readout_mode": arguments.get("readout_mode"),
        },
    }


def build_training_config(
    kwargs: Mapping[str, Any],
    env: Any,
    model: torch.nn.Module,
    repo_root: Path,
    *,
    eval_env: Any | None = None,
    fmri_factorial_schedule: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Resolve the scientific/training configuration used for run identity.

    ``repo_root`` is retained in the public signature for compatibility with
    callers of schema v1.  Source and runtime facts are now deliberately
    collected later by :func:`build_execution_provenance`, outside this
    returned (and hashed) mapping.
    """
    _ = repo_root
    arguments = {
        str(key): json_safe(value)
        for key, value in sorted(kwargs.items())
        if key not in NON_CONFIG_ARGUMENTS
    }
    derived = derive_runtime_config(
        env,
        model,
        eval_env=eval_env,
        fmri_factorial_schedule=fmri_factorial_schedule,
    )
    return {
        "arguments": arguments,
        "derived": derived,
    }


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(json_safe(payload), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_run_document(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf8") as stream:
        document = json.load(stream)
    if not isinstance(document, Mapping):
        raise ValueError(f"Managed config is not a mapping: {config_path}")
    return dict(document)


def _verify_stored_identity(
    document: Mapping[str, Any], current_config: Mapping[str, Any]
) -> None:
    """Validate a v1/v2 config document against current experiment identity."""
    stored_config = document.get("training_config")
    if not isinstance(stored_config, Mapping):
        raise ValueError("Managed run configuration mismatch: missing training_config.")
    stored_hash = document.get("config_hash", {}).get("full")
    if not isinstance(stored_hash, str):
        raise ValueError("Managed run configuration mismatch: missing full config hash.")

    schema_version = int(document.get("schema_version", 1))
    if schema_version >= 2:
        recomputed_stored_hash = experiment_config_hash(stored_config)
    else:
        # Schema v1 bound checkpoints/directories to the complete historical
        # mapping, including its misplaced provenance fields.
        recomputed_stored_hash = sha256_json(stored_config)
    if stored_hash != recomputed_stored_hash:
        raise ValueError(
            "Managed run configuration mismatch: config.yaml hash validation failed."
        )

    if normalize_training_config(stored_config) != normalize_training_config(
        current_config
    ):
        raise ValueError(
            "Managed run configuration mismatch or short-hash collision; "
            "refusing to reuse the directory."
        )


def _verify_run_document_binding(
    document: Mapping[str, Any], run_name: str, run_dir: Path
) -> None:
    """Ensure a config document is bound to its requested managed directory."""
    config_hash = document.get("config_hash", {})
    full_hash = config_hash.get("full") if isinstance(config_hash, Mapping) else None
    short_hash = config_hash.get("short") if isinstance(config_hash, Mapping) else None
    expected_short = (
        full_hash[:SHORT_HASH_LENGTH] if isinstance(full_hash, str) else None
    )
    run = document.get("run", {})
    if not isinstance(run, Mapping) or run.get("name") != run_name:
        raise ValueError(
            "Managed run document/name mismatch; refusing to reuse "
            f"{run_dir}."
        )
    if short_hash != expected_short:
        raise ValueError(
            "Managed run short/full config-hash mismatch; refusing to reuse "
            f"{run_dir}."
        )
    if "schema_version" in document and run.get("id") != short_hash:
        raise ValueError(
            "Managed run id/config-hash mismatch; refusing to reuse "
            f"{run_dir}."
        )
    if run_dir.name != f"{run_name}_{short_hash}":
        raise ValueError(
            "Managed config is stored under the wrong directory name; refusing "
            f"to reuse {run_dir}."
        )


def _v1_resume_compatibility_values(
    document: Mapping[str, Any]
) -> dict[str, Any]:
    """Extract strict fields from the historical schema-v1 arrangement."""
    training = document.get("training_config", {})
    old_source = training.get("code_fingerprint", {})
    old_files = old_source.get("files", {}) if isinstance(old_source, Mapping) else {}
    strict_files = {
        relative: old_files.get(relative)
        for relative in RESUME_CRITICAL_SOURCE_FILES
    }
    old_runtime = training.get("runtime_dependencies", {})
    provenance = document.get("provenance", {})
    launch = document.get("launch_controls", {})
    return {
        "strict_source_files": strict_files,
        "runtime_dependencies": json_safe(old_runtime),
        "platform": provenance.get("platform"),
        "device_specifier": launch.get("training_device"),
    }


def _provenance_incompatibilities(
    document: Mapping[str, Any], current: Mapping[str, Any]
) -> list[str]:
    """Return strict resume incompatibilities; informational metadata is ignored."""
    incompatibilities: list[str] = []
    schema_version = int(document.get("schema_version", 1))
    if schema_version >= 2:
        stored = document.get("execution_provenance", {})
        stored_compatibility = stored.get("compatibility", {})
        recorded_hash = stored.get("compatibility_hash", {}).get("full")
        if not isinstance(stored_compatibility, Mapping) or recorded_hash != sha256_json(
            stored_compatibility
        ):
            incompatibilities.append(
                "execution provenance hash: config.yaml validation failed"
            )
        current_compatibility = current.get("compatibility", {})
        comparisons = (
            (
                "critical source files",
                stored_compatibility.get("source", {}).get("files"),
                current_compatibility.get("source", {}).get("files"),
            ),
            (
                "Python/NumPy/PyTorch runtime",
                stored_compatibility.get("runtime_dependencies"),
                current_compatibility.get("runtime_dependencies"),
            ),
            (
                "platform",
                stored_compatibility.get("platform"),
                current_compatibility.get("platform"),
            ),
            (
                "original training device",
                stored_compatibility.get("device"),
                current_compatibility.get("device"),
            ),
            (
                "backend/thread/determinism settings",
                stored_compatibility.get("backend_settings"),
                current_compatibility.get("backend_settings"),
            ),
            (
                "execution environment",
                stored_compatibility.get("execution_environment"),
                current_compatibility.get("execution_environment"),
            ),
        )
    else:
        stored = _v1_resume_compatibility_values(document)
        current_compatibility = current.get("compatibility", {})
        comparisons = (
            (
                "critical source files",
                stored["strict_source_files"],
                current_compatibility.get("source", {}).get("files"),
            ),
            (
                "Python/NumPy/PyTorch runtime",
                stored["runtime_dependencies"],
                current_compatibility.get("runtime_dependencies"),
            ),
            (
                "platform",
                stored["platform"],
                current_compatibility.get("platform"),
            ),
            (
                "original training device",
                stored["device_specifier"],
                current_compatibility.get("device", {}).get("specifier"),
            ),
        )

    for label, original, resumed in comparisons:
        if json_safe(original) != json_safe(resumed):
            incompatibilities.append(
                f"{label}: recorded={original!r}, current={resumed!r}"
            )
    return incompatibilities


def _find_v1_resume_directory(
    parent: Path,
    run_name: str,
    training_config: Mapping[str, Any],
    current_provenance: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]] | None:
    """Find one compatible historical directory whose old hash changed by provenance."""
    if not parent.is_dir():
        return None
    identity_matches: list[tuple[Path, dict[str, Any], list[str]]] = []
    prefix = f"{run_name}_"
    for candidate in sorted(parent.iterdir()):
        if not candidate.is_dir() or not candidate.name.startswith(prefix):
            continue
        config_path = candidate / "config.yaml"
        if not config_path.is_file():
            continue
        try:
            document = _load_run_document(config_path)
            if int(document.get("schema_version", 1)) >= 2:
                continue
            if document.get("run", {}).get("name") != run_name:
                continue
            _verify_run_document_binding(document, run_name, candidate)
            _verify_stored_identity(document, training_config)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        identity_matches.append(
            (
                candidate,
                document,
                _provenance_incompatibilities(document, current_provenance),
            )
        )

    compatible = [record for record in identity_matches if not record[2]]
    if len(compatible) == 1:
        return compatible[0][0], compatible[0][1]
    if len(compatible) > 1:
        paths = ", ".join(str(record[0]) for record in compatible)
        raise ValueError(
            "Multiple provenance-compatible schema-v1 managed runs match this "
            f"configuration; refusing to choose silently: {paths}"
        )
    if identity_matches:
        details = "; ".join(
            f"{record[0].name}: {', '.join(record[2])}" for record in identity_matches
        )
        raise ValueError(
            "Managed resume provenance incompatibility; refusing to reuse an "
            f"existing schema-v1 run. {details}"
        )
    return None


def _require_resume_supported(document: Mapping[str, Any], run_dir: Path) -> None:
    legacy = document.get("legacy_import")
    resume_semantics = document.get("resume_semantics")
    explicitly_disabled = (
        isinstance(legacy, Mapping) and legacy.get("resume_supported") is False
    ) or (
        isinstance(resume_semantics, Mapping)
        and resume_semantics.get("resume_supported") is False
    )
    if explicitly_disabled:
        raise ValueError(
            f"Managed run is explicitly non-resumable: {run_dir}. "
            "Its frozen checkpoints remain available for analysis."
        )


def prepare_run_directory(
    *,
    basedir: Path | str,
    run_name: str,
    training_config: Mapping[str, Any],
    launch_controls: Mapping[str, Any],
    resume: bool,
    repo_root: Path | str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Create or validate one immutable managed run directory."""
    run_name = validate_run_name(run_name)
    training_config = normalize_training_config(training_config)
    full_hash = experiment_config_hash(training_config)
    short_hash = full_hash[:SHORT_HASH_LENGTH]
    parent = Path(basedir).resolve() / "models" / "abcd_fmri"
    run_dir = parent / f"{run_name}_{short_hash}"
    config_path = run_dir / "config.yaml"
    repo_root = Path(repo_root if repo_root is not None else basedir).resolve()
    current_provenance = build_execution_provenance(repo_root, launch_controls)

    if resume:
        if not run_dir.is_dir():
            historical = _find_v1_resume_directory(
                parent,
                run_name,
                training_config,
                current_provenance,
            )
            if historical is None:
                raise FileNotFoundError(
                    f"Managed run directory does not exist: {run_dir}"
                )
            _require_resume_supported(historical[1], historical[0])
            return historical
        if not config_path.is_file():
            raise FileNotFoundError(f"Managed run is missing config.yaml: {run_dir}")
        document = _load_run_document(config_path)
        _verify_run_document_binding(document, run_name, run_dir)
        _verify_stored_identity(document, training_config)
        _require_resume_supported(document, run_dir)
        incompatibilities = _provenance_incompatibilities(
            document, current_provenance
        )
        if incompatibilities:
            raise ValueError(
                "Managed resume provenance incompatibility; refusing to reuse "
                f"{run_dir}. " + "; ".join(incompatibilities)
            )
        return run_dir, document

    if run_dir.exists():
        raise FileExistsError(
            f"Managed run already exists: {run_dir}. Use --resume 1 to continue "
            "that exact configuration, or choose a different --run_name/configuration."
        )

    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir()
    (run_dir / "checkpoints").mkdir()
    document = {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "name": run_name,
            "id": short_hash,
            "task": "abcd_fmri",
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
        "training_config": json_safe(training_config),
        "summary": build_model_summary(
            training_config.get("arguments", {}),
            training_config.get("derived", {}),
        ),
        "config_hash": {
            "algorithm": "sha256",
            "full": full_hash,
            "short": short_hash,
            "scope": "resolved scientific/training configuration only",
            "excludes": [
                "source code and git metadata",
                "host/platform/device metadata",
                "Python/NumPy/PyTorch versions",
                "runtime paths, command line, and launch controls",
            ],
        },
        "launch_controls": json_safe(launch_controls),
        "execution_provenance": current_provenance,
        "resume_semantics": {
            "latest_boundary": "immediately after completed_updates optimizer steps",
            "restored": [
                "model_state_dict",
                "optimizer_state_dict",
                "NumPy global RNG",
                "torch CPU/CUDA RNG",
                "training/evaluation environment Generator state and block counter",
            ],
            "bitwise_guarantee": (
                "same code/configuration and deterministic kernels are required; "
                "cross-device or nondeterministic accelerator kernels are not bitwise guaranteed"
            ),
            "atomic_ordering_note": (
                "best.pt and validation files may be one evaluation ahead of "
                "latest.pt after a crash; resume trusts latest.pt and discards "
                "uncheckpointed validation rows before reproducing that evaluation"
            ),
        },
    }
    _atomic_json_write(config_path, document)
    return run_dir, document


def resume_provenance_hash(document: Mapping[str, Any]) -> str | None:
    """Return and validate the schema-v2 resume-provenance binding.

    Schema-v1 checkpoints predate this redundant checkpoint binding.  A stable
    hash of the compatibility facts they did record is returned so that a
    subsequently saved latest.pt can be bound without rewriting config.yaml.
    """
    if int(document.get("schema_version", 1)) < 2:
        return sha256_json(_v1_resume_compatibility_values(document))
    provenance = document.get("execution_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Managed config is missing execution_provenance.")
    compatibility = provenance.get("compatibility")
    stored_hash = provenance.get("compatibility_hash", {}).get("full")
    if not isinstance(compatibility, Mapping) or not isinstance(stored_hash, str):
        raise ValueError("Managed config has incomplete execution provenance.")
    recomputed = sha256_json(compatibility)
    if stored_hash != recomputed:
        raise ValueError("Managed config execution provenance hash validation failed.")
    return stored_hash


def capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        # Tensor/primitives only so the complete latest checkpoint remains
        # loadable under torch's restricted weights_only=True unpickler.
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "keys": torch.as_tensor(
                numpy_state[1].astype(np.int64, copy=True),
                dtype=torch.int64,
            ),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            numpy_state["keys"].detach().cpu().numpy().astype(np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.random.set_rng_state(state["torch_cpu"].cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def capture_environment_state(env: Any, eval_env: Any | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, item in (("train", env), ("evaluation", eval_env)):
        if item is None:
            continue
        payload: dict[str, Any] = {}
        rng = getattr(item, "rng", None)
        if rng is not None and hasattr(rng, "bit_generator"):
            payload["rng_bit_generator_state"] = json_safe(
                copy.deepcopy(rng.bit_generator.state)
            )
        if hasattr(item, "_block_counter"):
            payload["block_counter"] = int(item._block_counter)
        result[label] = payload
    return result


def restore_environment_state(
    state: Mapping[str, Any], env: Any, eval_env: Any | None
) -> None:
    for label, item in (("train", env), ("evaluation", eval_env)):
        if item is None or label not in state:
            continue
        payload = state[label]
        if "rng_bit_generator_state" in payload:
            item.rng.bit_generator.state = copy.deepcopy(payload["rng_bit_generator_state"])
        if "block_counter" in payload:
            item._block_counter = int(payload["block_counter"])


def _to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    return copy.deepcopy(value)


def checkpoint_payload(
    *,
    kind: str,
    config_hash: str,
    resume_provenance_hash: str,
    completed_updates: int,
    model: torch.nn.Module,
    best_validation_loss: float,
    best_update: int | None,
    optimizer: torch.optim.Optimizer | None = None,
    validation_history: Sequence[Mapping[str, Any]] | None = None,
    env: Any | None = None,
    eval_env: Any | None = None,
) -> dict[str, Any]:
    if kind not in {"best", "latest"}:
        raise ValueError(f"Unknown checkpoint kind: {kind}")
    model_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_kind": kind,
        "config_hash": str(config_hash),
        "resume_provenance_hash": str(resume_provenance_hash),
        "completed_updates": int(completed_updates),
        "best_validation_loss": float(best_validation_loss),
        "best_update": None if best_update is None else int(best_update),
        "resume_capable": kind == "latest",
        "model_state_dict": model_state,
    }
    if kind == "latest":
        if optimizer is None or validation_history is None or env is None:
            raise ValueError("latest checkpoint requires optimizer/history/environment state")
        payload.update(
            {
                "optimizer_state_dict": _to_cpu(optimizer.state_dict()),
                "validation_history": metrics_json_safe(validation_history),
                "rng_state": capture_rng_state(),
                "environment_state": capture_environment_state(env, eval_env),
            }
        )
    return payload


def save_checkpoint(run_dir: Path, payload: Mapping[str, Any]) -> Path:
    kind = str(payload["checkpoint_kind"])
    path = Path(run_dir) / "checkpoints" / f"{kind}.pt"
    _atomic_torch_save(path, payload)
    return path


def load_latest_checkpoint(
    run_dir: Path,
    *,
    expected_config_hash: str,
    expected_resume_provenance_hash: str | None,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    env: Any,
    eval_env: Any | None,
    device: torch.device,
) -> tuple[int, float, int | None, list[dict[str, Any]]]:
    path = Path(run_dir) / "checkpoints" / "latest.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Managed run has no latest checkpoint: {path}")
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("checkpoint_kind") != "latest":
        raise ValueError(f"Not a latest checkpoint: {path}")
    if not payload.get("resume_capable", False):
        raise ValueError(f"Checkpoint is explicitly marked non-resumable: {path}")
    if payload.get("config_hash") != expected_config_hash:
        raise ValueError("latest.pt configuration hash does not match config.yaml")
    checkpoint_schema = int(payload.get("schema_version", 1))
    checkpoint_provenance_hash = payload.get("resume_provenance_hash")
    if checkpoint_schema >= 2:
        if expected_resume_provenance_hash is None:
            raise ValueError(
                "Schema-v2 latest.pt cannot be resumed without validated "
                "execution provenance from config.yaml."
            )
        if checkpoint_provenance_hash != expected_resume_provenance_hash:
            raise ValueError(
                "latest.pt execution provenance does not match config.yaml"
            )
    elif (
        checkpoint_provenance_hash is not None
        and expected_resume_provenance_hash is not None
        and checkpoint_provenance_hash != expected_resume_provenance_hash
    ):
        raise ValueError("latest.pt execution provenance does not match config.yaml")

    # All identity/provenance checks deliberately precede any state mutation.
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    restore_environment_state(payload["environment_state"], env, eval_env)
    restore_rng_state(payload["rng_state"])
    return (
        int(payload["completed_updates"]),
        float(payload["best_validation_loss"]),
        payload.get("best_update"),
        list(payload["validation_history"]),
    )


def _validation_curve_checkpoint_summary(
    validation_history: Sequence[Mapping[str, Any]],
    *,
    best_update: int | None = None,
    latest_checkpoint_update: int | None = None,
) -> dict[str, Any]:
    """Bind checkpoint roles to the existing validation measurements."""
    if not validation_history:
        raise ValueError("A validation curve requires at least one evaluation.")
    rows = list(validation_history)
    if best_update is None:
        # ABCD training ranks checkpoints by highest validation accuracy, then
        # lower total validation loss. Keep the earliest point on a complete
        # tie because neither declared criterion distinguishes it.
        def checkpoint_rank(index: int) -> tuple[float, float, int]:
            row = rows[index]
            accuracy = row.get("accuracy")
            accuracy = -np.inf if accuracy is None else float(accuracy)
            if np.isnan(accuracy):
                accuracy = -np.inf
            return (-accuracy, float(row["loss"]), index)

        best_index = min(
            range(len(rows)),
            key=checkpoint_rank,
        )
        best_update = int(rows[best_index]["update"])
    else:
        best_update = int(best_update)
        matches = [
            index
            for index, row in enumerate(rows)
            if int(row["update"]) == best_update
        ]
        if not matches:
            raise ValueError(
                f"best.pt update {best_update} has no validation-history row."
            )
        best_index = matches[-1]

    latest_validation_index = len(rows) - 1
    latest_validation_update = int(rows[latest_validation_index]["update"])
    if latest_checkpoint_update is None:
        latest_checkpoint_update = latest_validation_update
    return {
        "best_index": int(best_index),
        "best_update": int(best_update),
        "best_row": rows[best_index],
        "latest_validation_index": int(latest_validation_index),
        "latest_validation_update": int(latest_validation_update),
        "latest_validation_row": rows[latest_validation_index],
        "latest_checkpoint_update": int(latest_checkpoint_update),
    }


def save_validation_curve(
    run_dir: Path,
    validation_history: Sequence[Mapping[str, Any]],
    *,
    best_update: int | None = None,
    latest_checkpoint_update: int | None = None,
) -> tuple[Path, Path]:
    """Persist metrics plus an explicit best/latest checkpoint curve."""
    run_dir = Path(run_dir)
    checkpoint_summary = _validation_curve_checkpoint_summary(
        validation_history,
        best_update=best_update,
        latest_checkpoint_update=latest_checkpoint_update,
    )
    csv_path = run_dir / "validation_curve.csv"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf8",
        newline="",
        dir=run_dir,
        prefix=".validation_curve.csv.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_csv = Path(stream.name)
        columns = (
            "update",
            "loss",
            "accuracy",
            "best_loss",
            "elapsed_minutes",
            "checkpoint_role",
            "latest_checkpoint_update",
        )
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for index, row in enumerate(validation_history):
            roles = []
            if index == checkpoint_summary["best_index"]:
                roles.append("best.pt")
            if index == checkpoint_summary["latest_validation_index"]:
                roles.append("latest.pt:last_validation")
            csv_row = {
                key: row.get(key)
                for key in columns
                if key not in {"checkpoint_role", "latest_checkpoint_update"}
            }
            csv_row["checkpoint_role"] = ";".join(roles)
            csv_row["latest_checkpoint_update"] = (
                checkpoint_summary["latest_checkpoint_update"]
                if index == checkpoint_summary["latest_validation_index"]
                else ""
            )
            writer.writerow(csv_row)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_csv, csv_path)

    # Import lazily so importing training utilities remains lightweight.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    updates = np.asarray([row["update"] for row in validation_history], dtype=int)
    losses = np.asarray([row["loss"] for row in validation_history], dtype=float)
    accuracies = np.asarray(
        [row["accuracy"] for row in validation_history], dtype=float
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), constrained_layout=True)
    axes[0].plot(updates, losses, color="#305c89", marker="o", ms=3)
    axes[0].set(xlabel="optimizer updates", ylabel="total validation loss")
    axes[0].grid(alpha=0.2)
    axes[1].plot(updates, accuracies, color="#9b3d4a", marker="o", ms=3)
    axes[1].set(xlabel="optimizer updates", ylabel="validation accuracy")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(alpha=0.2)

    best_index = checkpoint_summary["best_index"]
    latest_index = checkpoint_summary["latest_validation_index"]
    same_validation_point = best_index == latest_index
    for axis, values in zip(axes, (losses, accuracies)):
        if same_validation_point:
            axis.scatter(
                updates[best_index],
                values[best_index],
                marker="D",
                s=65,
                facecolor="#f2c14e",
                edgecolor="black",
                linewidth=0.7,
                zorder=5,
                label="best.pt / latest validation",
            )
        else:
            axis.scatter(
                updates[best_index],
                values[best_index],
                marker="*",
                s=105,
                facecolor="#f2c14e",
                edgecolor="black",
                linewidth=0.7,
                zorder=5,
                label="best.pt",
            )
            axis.scatter(
                updates[latest_index],
                values[latest_index],
                marker="s",
                s=46,
                facecolor="#62b6cb",
                edgecolor="black",
                linewidth=0.7,
                zorder=5,
                label="latest.pt: latest validation",
            )
        axis.legend(loc="best", fontsize=7.5, framealpha=0.9)

    best_row = checkpoint_summary["best_row"]
    latest_row = checkpoint_summary["latest_validation_row"]
    best_text = (
        f"best.pt | update {checkpoint_summary['best_update']} | "
        f"performance={float(best_row['accuracy']):.5g} | "
        f"total validation loss={float(best_row['loss']):.5g}"
    )
    same_checkpoint_point = (
        same_validation_point
        and checkpoint_summary["best_update"]
        == checkpoint_summary["latest_checkpoint_update"]
    )
    if same_checkpoint_point:
        checkpoint_text = best_text.replace("best.pt", "best.pt = latest.pt", 1)
    else:
        latest_text = (
            f"latest.pt | update {checkpoint_summary['latest_checkpoint_update']} | "
            f"latest validation at update "
            f"{checkpoint_summary['latest_validation_update']}: "
            f"performance={float(latest_row['accuracy']):.5g} | "
            f"total validation loss={float(latest_row['loss']):.5g}"
        )
        checkpoint_text = f"{best_text}\n{latest_text}"
    figure.suptitle(checkpoint_text, fontsize=9.0)

    png_path = run_dir / "validation_curve.png"
    with tempfile.NamedTemporaryFile(
        dir=run_dir,
        prefix=".validation_curve.png.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_png = Path(stream.name)
    figure.savefig(temporary_png, format="png", dpi=160)
    plt.close(figure)
    os.replace(temporary_png, png_path)
    return csv_path, png_path


def load_validation_metrics(run_dir: Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "validation_metrics.json"
    if not path.is_file():
        return []
    with path.open("r", encoding="utf8") as stream:
        payload = json.load(stream)
    return list(payload.get("evaluations", []))


def save_validation_metrics(
    run_dir: Path, evaluations: Sequence[Mapping[str, Any]]
) -> Path:
    """Save compact evaluation QC while omitting per-block duplicate payloads."""
    keep = {
        "update",
        "evaluation_mode",
        "configuration_bank_name",
        "configuration_bank_statistics",
        "evaluation_task_seed",
        "validation_recurrent_noise_seed",
        "validation_rng_replayed",
        "evaluation_recurrent_noise",
        "loss",
        "accuracy",
        "num_blocks",
        "environment",
        "route_consistency",
    }
    compact = [
        {
            str(key): metrics_json_safe(value)
            for key, value in item.items()
            if key in keep
        }
        for item in evaluations
    ]
    path = Path(run_dir) / "validation_metrics.json"
    _atomic_json_write(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "evaluations": compact,
        },
    )
    return path


def save_final_evaluation(run_dir: Path, metrics: Mapping[str, Any]) -> Path:
    """Save compact factorial-evaluation provenance without trajectory duplication."""
    excluded = {"trajectory_stores"}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "note": (
            "Compact final evaluation summary; full trajectories are collected by "
            "the frozen ABCD analysis pipeline from checkpoints/best.pt."
        ),
        "metrics": {
            str(key): metrics_json_safe(value)
            for key, value in metrics.items()
            if key not in excluded
        },
    }
    path = Path(run_dir) / "final_evaluation.json"
    _atomic_json_write(path, payload)
    return path


def import_legacy_run(
    *,
    basedir: Path | str,
    repo_root: Path | str,
    run_name: str,
    resolved_kwargs: Mapping[str, Any],
    model: torch.nn.Module,
    eval_env: Any | None,
    fmri_factorial_schedule: Sequence[Any] | None,
    best_state_dict: Mapping[str, torch.Tensor],
    latest_state_dict: Mapping[str, torch.Tensor] | None,
    source_artifacts: Mapping[str, Path | str],
    completed_updates: int,
    best_validation_loss: float,
    best_update: int | None,
    validation_history: Sequence[Mapping[str, Any]] = (),
    limitations: Sequence[str] = (),
) -> Path:
    """Create a non-resumable managed illustration from stable legacy snapshots.

    Source artifacts are opened read-only and fingerprinted; they are never
    moved, renamed, or modified.  Legacy files generally lack complete global
    and environment RNG state, so ``latest.pt`` is explicitly non-resumable.
    This helper must not be used to claim exact continuation support.
    """
    source_records: dict[str, Any] = {}
    for label, source in sorted(source_artifacts.items()):
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        source_records[str(label)] = {
            "name": path.name,
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
            "absolute_source_at_import": str(path),
        }

    arguments = {
        str(key): json_safe(value)
        for key, value in sorted(resolved_kwargs.items())
        if key not in NON_CONFIG_ARGUMENTS
    }
    derived = derive_runtime_config(
        model.env,
        model,
        eval_env=eval_env,
        fmri_factorial_schedule=fmri_factorial_schedule,
    )
    training_config = {
        "arguments": arguments,
        "derived": derived,
    }
    run_dir, document = prepare_run_directory(
        basedir=basedir,
        run_name=run_name,
        training_config=training_config,
        launch_controls={"operation": "legacy_import", "resume": False},
        resume=False,
        repo_root=repo_root,
    )
    stated_limitations = list(limitations) or [
        "Legacy artifacts do not contain all global/environment RNG state.",
        "Imported latest.pt is a frozen weight snapshot, not a resumable boundary.",
        "Historical package versions/source cleanliness may be unavailable.",
    ]
    document["legacy_import"] = {
        "source_artifacts": source_records,
        "import_tool_source": source_fingerprint(Path(repo_root).resolve()),
        "import_runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
        },
        "resume_supported": False,
        "limitations": stated_limitations,
    }
    document["resume_semantics"] = {
        "resume_supported": False,
        "reason": stated_limitations,
    }
    _atomic_json_write(run_dir / "config.yaml", document)

    best_payload = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_kind": "best",
        "config_hash": document["config_hash"]["full"],
        "resume_provenance_hash": resume_provenance_hash(document),
        "completed_updates": int(0 if best_update is None else best_update),
        "best_validation_loss": float(best_validation_loss),
        "best_update": None if best_update is None else int(best_update),
        "resume_capable": False,
        "model_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in best_state_dict.items()
        },
    }
    latest_source = latest_state_dict if latest_state_dict is not None else best_state_dict
    latest_payload = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_kind": "latest",
        "config_hash": document["config_hash"]["full"],
        "resume_provenance_hash": resume_provenance_hash(document),
        "completed_updates": int(completed_updates),
        "best_validation_loss": float(best_validation_loss),
        "best_update": None if best_update is None else int(best_update),
        "resume_capable": False,
        "resume_limitations": stated_limitations,
        "validation_history": metrics_json_safe(validation_history),
        "model_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in latest_source.items()
        },
    }
    save_checkpoint(run_dir, best_payload)
    save_checkpoint(run_dir, latest_payload)
    if validation_history:
        save_validation_curve(
            run_dir,
            validation_history,
            best_update=best_update,
            latest_checkpoint_update=completed_updates,
        )
    return run_dir
