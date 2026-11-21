"""Safe support utilities for the isolated IBM QPU pilot.

This module deliberately does not import Qiskit or create circuits. Account
queries and sampler submissions are separate, explicit operations.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np


PILOT_EXPERIMENT_ID = "QNIM_IBM_QPU_PILOT_V1"
PILOT_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "scaling_study"
    / "qnim_ibm_qpu_pilot_v1"
    / "protocol.json"
)
PILOT_JOB_RECORD_DIR = PILOT_PROTOCOL_PATH.parent / "jobs"


class PilotPreflightError(RuntimeError):
    """Raised when a pilot precondition cannot be established."""


class PilotBudgetError(ValueError):
    """Raised when the declared QPU budget would be exceeded."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_sha256(parameters: np.ndarray) -> str:
    values = np.ascontiguousarray(parameters, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def load_protocol(path: Path = PILOT_PROTOCOL_PATH) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("experiment_id") != PILOT_EXPERIMENT_ID:
        raise PilotPreflightError("Unexpected pilot experiment identity")
    if not isinstance(protocol.get("submission_enabled"), bool):
        raise PilotPreflightError("Protocol submission_enabled must be a boolean")
    return protocol


def _installed_version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def local_preflight(protocol: Mapping[str, Any]) -> dict[str, Any]:
    pins = protocol["software"]
    installed = {
        "qiskit": _installed_version("qiskit"),
        "qiskit-ibm-runtime": _installed_version("qiskit-ibm-runtime"),
        "qiskit-aer": _installed_version("qiskit-aer"),
        "qiskit-machine-learning": _installed_version("qiskit-machine-learning"),
    }
    pin_matches = {
        name: installed[name] == pins[name]
        for name in ("qiskit", "qiskit-ibm-runtime")
    }
    if installed["qiskit-ibm-runtime"] is None:
        compatibility = "unverified_runtime_not_installed"
    elif installed["qiskit-ibm-runtime"] != pins["qiskit-ibm-runtime"]:
        compatibility = "runtime_version_mismatch"
    elif installed["qiskit"] != pins["qiskit"]:
        compatibility = "qiskit_version_mismatch"
    else:
        compatibility = "declared_versions_match_api_compatibility_unverified"
    return {
        "experiment_id": protocol["experiment_id"],
        "installed_versions": installed,
        "declared_pins": {
            "qiskit": pins["qiskit"],
            "qiskit-ibm-runtime": pins["qiskit-ibm-runtime"],
        },
        "declared_versions_match": pin_matches,
        "runtime_api_compatibility": compatibility,
        "account_query_performed": False,
        "sampler_job_submitted": False,
    }


def safe_instance_metadata(service: Any) -> dict[str, Optional[str]]:
    result: dict[str, Optional[str]] = {"instance": None, "region": None}
    for source in (service, getattr(service, "_account", None)):
        if source is None:
            continue
        for key in result:
            value = (
                source.get(key)
                if isinstance(source, Mapping)
                else getattr(source, key, None)
            )
            if isinstance(value, str) and value:
                result[key] = value
    return result


def _backend_edges(backend: Any) -> set[tuple[int, int]]:
    coupling = getattr(backend, "coupling_map", None)
    if coupling is None:
        target = getattr(backend, "target", None)
        builder = getattr(target, "build_coupling_map", None)
        if callable(builder):
            coupling = builder()
    get_edges = getattr(coupling, "get_edges", None)
    if not callable(get_edges):
        return set()
    return {
        (int(left), int(right))
        for left, right in get_edges()
        if int(left) != int(right)
    }


def validate_connected_layout(
    layout: Sequence[int],
    edges: Sequence[Sequence[int]] | set[tuple[int, int]],
    *,
    logical_qubits: int = 8,
    backend_qubits: Optional[int] = None,
) -> tuple[int, ...]:
    values = tuple(layout)
    if len(values) != logical_qubits:
        raise ValueError(f"Layout must contain exactly {logical_qubits} physical qubits")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("Layout entries must be integer physical-qubit indices")
    if len(set(values)) != len(values):
        raise ValueError("Layout physical-qubit indices must be unique")
    if any(value < 0 for value in values):
        raise ValueError("Layout indices cannot be negative")
    if backend_qubits is not None and any(value >= backend_qubits for value in values):
        raise ValueError("Layout contains a qubit outside the selected backend")
    undirected = {
        (min(int(left), int(right)), max(int(left), int(right)))
        for left, right in edges
    }
    for left, right in zip(values, values[1:]):
        if (min(left, right), max(left, right)) not in undirected:
            raise ValueError("Layout is not a connected path in the backend coupling map")
    return values


def find_connected_linear_layouts(
    edges: Sequence[Sequence[int]] | set[tuple[int, int]],
    backend_qubits: int,
    *,
    logical_qubits: int = 8,
    limit: int = 20,
    max_expansions: int = 100_000,
) -> dict[str, Any]:
    adjacency = {qubit: set() for qubit in range(backend_qubits)}
    for left, right in edges:
        left, right = int(left), int(right)
        if left == right or left not in adjacency or right not in adjacency:
            continue
        adjacency[left].add(right)
        adjacency[right].add(left)

    layouts: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    expansions = 0
    truncated = False

    def visit(path: tuple[int, ...]) -> None:
        nonlocal expansions, truncated
        if len(layouts) >= limit or expansions >= max_expansions:
            truncated = True
            return
        if len(path) == logical_qubits:
            reverse = tuple(reversed(path))
            canonical = min(path, reverse)
            if canonical not in seen:
                seen.add(canonical)
                layouts.append(list(path))
            return
        for neighbor in sorted(adjacency[path[-1]] - set(path)):
            expansions += 1
            if expansions > max_expansions:
                truncated = True
                return
            visit(path + (neighbor,))
            if len(layouts) >= limit or truncated:
                return

    for start in range(backend_qubits):
        visit((start,))
        if len(layouts) >= limit or truncated:
            break
    return {
        "layouts": layouts,
        "truncated": truncated,
        "search_expansions": expansions,
        "logical_qubits": logical_qubits,
    }


def query_ibm_preflight(token: str) -> dict[str, Any]:
    """Query account-visible backend metadata; call only on explicit request."""
    if not token:
        return {
            "authenticated": False,
            "diagnostic": "No IBM token was supplied; no login was attempted.",
            "backends": [],
            "sampler_job_submitted": False,
        }
    try:
        import qiskit_ibm_runtime
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService(
            channel="ibm_quantum_platform",
            token=token,
        )
        backends = service.backends()
    except Exception as exc:
        return {
            "authenticated": False,
            "diagnostic": f"IBM preflight failed ({type(exc).__name__}); no job was submitted.",
            "backends": [],
            "sampler_job_submitted": False,
        }

    reports = []
    for backend in backends:
        try:
            status_obj = backend.status()
            status = {
                "operational": getattr(status_obj, "operational", None),
                "pending_jobs": getattr(status_obj, "pending_jobs", None),
                "message": getattr(status_obj, "status_msg", None),
            }
        except Exception as exc:
            status = {"error": type(exc).__name__}
        try:
            edges = _backend_edges(backend)
            backend_qubits = int(backend.num_qubits)
            layouts = find_connected_linear_layouts(edges, backend_qubits)
            target = getattr(backend, "target", None)
            operations = sorted(
                str(name) for name in getattr(target, "operation_names", ())
            )
            reports.append(
                {
                    "name": str(backend.name),
                    "qubit_count": backend_qubits,
                    "status": status,
                    "target_instructions": operations,
                    "coupling_edges": [list(edge) for edge in sorted(edges)],
                    "connected_eight_qubit_linear_layouts": layouts,
                }
            )
        except Exception as exc:
            reports.append(
                {
                    "name": str(getattr(backend, "name", "unknown")),
                    "diagnostic": f"Metadata unavailable ({type(exc).__name__})",
                }
            )

    try:
        session_available = hasattr(qiskit_ibm_runtime, "Session")
        batch_available = hasattr(qiskit_ibm_runtime, "Batch")
    except NameError:
        session_available = False
        batch_available = False
    return {
        "authenticated": True,
        **safe_instance_metadata(service),
        "backends": reports,
        "runtime_client_session_symbol": session_available,
        "runtime_client_batch_symbol": batch_available,
        "plan_session_capability": "not established by client metadata",
        "resource_quota": "not exposed by this preflight",
        "sampler_job_submitted": False,
    }


def verify_dataset_provenance(
    dataset_dir: Path,
    protocol: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    dataset_cfg = protocol["dataset"]
    expected_manifest_hash = dataset_cfg["dataset_manifest_sha256"].lower()
    manifest_path = dataset_dir / "dataset_manifest.json"
    metadata_path = dataset_dir / "dataset_metadata.json"
    if not manifest_path.is_file() or not metadata_path.is_file():
        raise PilotPreflightError("Dataset manifest or metadata is missing")
    actual_manifest_hash = sha256_file(manifest_path)
    if actual_manifest_hash != expected_manifest_hash:
        raise PilotPreflightError("Dataset manifest hash does not match the pilot protocol")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != dataset_cfg["dataset_id"]:
        raise PilotPreflightError("Unexpected dataset identity")
    if metadata.get("dataset_id") != dataset_cfg["dataset_id"]:
        raise PilotPreflightError("Dataset metadata identity mismatch")
    if manifest.get("protocol", {}).get("test_created_or_loaded") is not False:
        raise PilotPreflightError("Pilot input must have no test split or test arrays")
    if metadata.get("final_test", {}).get("generated") is not False:
        raise PilotPreflightError("Dataset metadata does not seal the test split")
    if metadata.get("final_test", {}).get("loaded") is not False:
        raise PilotPreflightError("Dataset metadata reports test data was loaded")

    generation = metadata.get("generation", {})
    if generation.get("strict_physics") is not True:
        raise PilotPreflightError("Dataset was not generated with strict_physics=True")
    if generation.get("fallback_allowed") is not False:
        raise PilotPreflightError("Dataset provenance permits a waveform fallback")
    if generation.get("waveform_source") != "PyCBC":
        raise PilotPreflightError("Dataset waveform source is not PyCBC")
    if not generation.get("approximant"):
        raise PilotPreflightError("Dataset approximant is not recorded")
    if not generation.get("class_injectors"):
        raise PilotPreflightError("Dataset injection configuration is missing")
    if generation.get("class_count") != protocol["model"]["classes"]:
        raise PilotPreflightError("Dataset class count differs from the pilot protocol")
    if generation.get("detector_noise_included") is not False:
        raise PilotPreflightError("Unexpected detector-noise provenance")

    pca = metadata.get("pca", {})
    if pca.get("fit_split") != "train_only":
        raise PilotPreflightError("PCA must be fitted on training data only")
    if pca.get("scaler_fit_split") != "train_only":
        raise PilotPreflightError("Scaler must be fitted on training data only")

    manifest_files = manifest.get("file_sha256", {})
    expected_files = dataset_cfg["file_sha256"]
    for filename, expected_hash in expected_files.items():
        if filename not in manifest_files:
            raise PilotPreflightError(f"Dataset manifest does not list {filename}")
        if manifest_files[filename].lower() != expected_hash.lower():
            raise PilotPreflightError(f"Protocol and dataset manifest disagree for {filename}")
        path = dataset_dir / filename
        if not path.is_file() or sha256_file(path) != expected_hash.lower():
            raise PilotPreflightError(f"Dataset file hash mismatch: {filename}")
    if any("test" in name.lower() for name in manifest_files):
        raise PilotPreflightError("Dataset manifest lists a test artifact")

    source_hashes = metadata.get("generator_source_hashes", {})
    if not source_hashes:
        raise PilotPreflightError("Generator source hashes are absent")
    for relative_path, expected_hash in source_hashes.items():
        source_path = repository_root / relative_path
        if not source_path.is_file() or sha256_file(source_path) != expected_hash.lower():
            raise PilotPreflightError(f"Generator source hash mismatch: {relative_path}")

    runtime = metadata.get("runtime_versions", {})
    if not runtime.get("pycbc") or not runtime.get("numpy"):
        raise PilotPreflightError("Waveform-generation runtime versions are incomplete")
    return {
        "dataset_id": dataset_cfg["dataset_id"],
        "dataset_manifest_sha256": actual_manifest_hash,
        "file_sha256": {
            filename: expected_hash.lower()
            for filename, expected_hash in expected_files.items()
        },
        "pca_sha256": expected_files["pca_state.npz"],
        "strict_physics": True,
        "fallback_allowed": False,
        "waveform_source": generation["waveform_source"],
        "approximant": generation["approximant"],
        "class_names": generation["class_names"],
        "dataset_seed": generation["dataset_seed"],
        "injection_parameter_ranges": {
            "mass1_msun_range": generation["mass1_msun_range"],
            "mass2_msun_range": generation["mass2_msun_range"],
            "distance_mpc_range": generation["distance_mpc_range"],
            "duration_seconds": generation["duration_seconds"],
            "sample_rate_hz": generation["sample_rate_hz"],
            "f_lower_hz": generation["f_lower_hz"],
        },
        "label_files": {
            "y_train": expected_files["y_train.npy"],
            "y_validation": expected_files["y_val.npy"],
        },
        "injectors": generation["class_injectors"],
        "whitening": generation["whitening"],
        "generator_source_hashes": source_hashes,
        "generation_runtime_versions": runtime,
        "test_split_created_or_loaded": False,
        "detector_noise_included": False,
        "event_level_parameter_ledger_present": bool(
            generation.get("event_parameter_ledger")
        ),
        "scientific_scope": "strict PyCBC simulated injections; no detector noise or observed events",
        "physical_astrophysical_claim_authorized": False,
    }


@dataclass(frozen=True)
class PilotSplits:
    """Train/validation-only container; intentionally has no test attributes."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray

    @property
    def n_classes(self) -> int:
        return int(np.max(self.y_train)) + 1


def load_train_validation_splits(
    dataset_dir: Path,
    protocol: Mapping[str, Any],
    repository_root: Path,
) -> tuple[PilotSplits, dict[str, Any]]:
    provenance = verify_dataset_provenance(dataset_dir, protocol, repository_root)
    files = protocol["dataset"]["input_files"]
    X_train = np.load(dataset_dir / files["X_train"], allow_pickle=False)
    y_train = np.load(dataset_dir / files["y_train"], allow_pickle=False)
    X_val = np.load(dataset_dir / files["X_val"], allow_pickle=False)
    y_val = np.load(dataset_dir / files["y_val"], allow_pickle=False)

    expected = protocol["dataset"]["shapes"]
    if list(X_train.shape) != expected["X_train"] or list(X_val.shape) != expected["X_val"]:
        raise PilotPreflightError("Train/validation feature shapes differ from protocol")
    if list(y_train.shape) != expected["y_train"] or list(y_val.shape) != expected["y_val"]:
        raise PilotPreflightError("Train/validation label shapes differ from protocol")
    if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
        raise PilotPreflightError("Train/validation features contain non-finite values")
    classes = protocol["model"]["classes"]
    if not np.array_equal(np.unique(y_train), np.arange(classes)):
        raise PilotPreflightError("Training labels do not match the declared classes")
    if not np.array_equal(np.unique(y_val), np.arange(classes)):
        raise PilotPreflightError("Validation labels do not match the declared classes")
    n_features = protocol["model"]["input_dimension"]
    return (
        PilotSplits(
            X_train=np.asarray(X_train[:, :n_features], dtype=np.float64),
            y_train=np.asarray(y_train, dtype=np.int64),
            X_val=np.asarray(X_val[:, :n_features], dtype=np.float64),
            y_val=np.asarray(y_val, dtype=np.int64),
        ),
        provenance,
    )


def estimate_resources(protocol: Mapping[str, Any]) -> dict[str, int]:
    local = protocol["local_pretraining"]
    hardware = protocol["hardware_spsa"]
    validation = protocol["validation"]
    budget = protocol["resource_budget"]
    n_train = int(protocol["dataset"]["shapes"]["y_train"][0])
    n_classes = int(protocol["model"]["classes"])
    iterations = int(local["max_iterations"])
    feynman_parameters = min(int(local["n_feynman_params"]), int(protocol["model"]["logical_qubits"]))
    batch_size = min(int(local["batch_size"]), n_train)
    reference_examples = min(max(128, 10 * n_classes), n_train)
    readout_samples = min(200, n_train)
    refit_every = int(local["readout_refit_every"])
    refits = iterations // refit_every if refit_every else 0

    local_batch_calls = 1 + iterations * (7 + 8 * feynman_parameters)
    local_reference_runs = iterations
    local_readout_runs = 1 + refits
    local_objective_calls = local_batch_calls + local_reference_runs
    local_pubs = (
        local_batch_calls * batch_size
        + local_reference_runs * reference_examples
        + local_readout_runs * readout_samples
    )
    local_runs = local_batch_calls + local_reference_runs + local_readout_runs
    local_shots = (
        local_batch_calls * batch_size * int(local["shots"])
        + local_reference_runs * reference_examples * int(local["reference_shots"])
        + local_readout_runs * readout_samples * int(local["shots"])
    )

    steps = int(hardware["steps"])
    hardware_batch = int(hardware["batch_size"])
    hardware_shots = int(hardware["shots"])
    reference_every = int(hardware["reference_every"])
    reference_runs = steps // reference_every if reference_every else 0
    reference_examples_hw = int(hardware["reference_examples"])
    reference_shots = int(hardware["reference_shots"])
    zne_multiplier = len(validation["zne_scale_factors"]) if validation["use_zne"] else 1
    validation_examples = int(validation["examples"])
    validation_shots = int(validation["shots"])
    validation_runs = 1 if validation_examples else 0

    hardware_jobs = 2 * steps + reference_runs
    hardware_pubs = (
        2 * steps * hardware_batch
        + reference_runs * reference_examples_hw
    )
    hardware_shots_total = (
        2 * steps * hardware_batch * hardware_shots
        + reference_runs * reference_examples_hw * reference_shots
    )
    validation_pubs = validation_runs * validation_examples * zne_multiplier
    validation_shots_total = validation_pubs * validation_shots
    total_qpu_jobs = hardware_jobs + validation_runs
    total_qpu_pubs = hardware_pubs + validation_pubs
    total_qpu_shots = hardware_shots_total + validation_shots_total

    estimate = {
        "local_qnspsa_objective_calls": local_objective_calls,
        "local_sampler_runs": local_runs,
        "local_pubs_and_circuits": local_pubs,
        "local_shots": local_shots,
        "local_reference_evaluations": local_reference_runs,
        "local_readout_warm_start_and_refit_runs": local_readout_runs,
        "hardware_spsa_steps": steps,
        "hardware_plus_minus_evaluations": 2 * steps,
        "hardware_reference_evaluations": reference_runs,
        "hardware_qpu_jobs": hardware_jobs,
        "hardware_pubs_and_circuits": hardware_pubs,
        "hardware_shots": hardware_shots_total,
        "validation_zne_multiplier": zne_multiplier,
        "validation_qpu_jobs": validation_runs,
        "validation_pubs_and_circuits": validation_pubs,
        "validation_shots": validation_shots_total,
        "total_qpu_jobs": total_qpu_jobs,
        "total_qpu_pubs_and_circuits": total_qpu_pubs,
        "total_qpu_shots": total_qpu_shots,
        "qpu_seconds": -1,
        "resource_units": -1,
    }
    enforce_resource_budget(protocol, estimate)
    return estimate


def enforce_resource_budget(
    protocol: Mapping[str, Any], estimate: Mapping[str, int]
) -> None:
    budget = protocol["resource_budget"]
    checks = (
        ("total_qpu_jobs", "max_qpu_jobs"),
        ("total_qpu_shots", "max_total_qpu_shots"),
        ("hardware_spsa_steps", "max_optimization_steps"),
        ("validation_examples", "max_validation_examples"),
    )
    for estimate_key, budget_key in checks:
        actual = (
            int(protocol["validation"]["examples"])
            if estimate_key == "validation_examples"
            else int(estimate[estimate_key])
        )
        limit = int(budget[budget_key])
        if actual > limit:
            raise PilotBudgetError(
                f"{estimate_key}={actual} exceeds {budget_key}={limit}"
            )


def require_submission_enabled(
    protocol: Mapping[str, Any], submit_requested: bool
) -> None:
    if not submit_requested:
        raise PilotPreflightError("Submission was not explicitly requested")
    if protocol.get("submission_enabled") is not True:
        raise PilotPreflightError(
            "QPU submission is disabled in this development-stage protocol"
        )


def validate_pilot_job_context(
    store: Any,
    metadata: Mapping[str, Any],
    *,
    optimization_steps: Optional[int] = None,
    validation_examples: Optional[int] = None,
) -> None:
    if metadata.get("experiment_id") != PILOT_EXPERIMENT_ID:
        return
    protocol = load_protocol()
    require_submission_enabled(protocol, submit_requested=True)
    repository_root = Path(__file__).resolve().parents[2]
    required_directory = (
        repository_root / protocol["job_recording"]["directory"]
    ).resolve()
    if store.directory.resolve() != required_directory:
        raise PilotPreflightError(
            "Pilot jobs must use the shared protocol job ledger"
        )
    budget = protocol["resource_budget"]
    if (
        store.max_jobs is None
        or store.max_jobs > int(budget["max_qpu_jobs"])
        or store.max_shots is None
        or store.max_shots > int(budget["max_total_qpu_shots"])
    ):
        raise PilotPreflightError("Pilot job ledger is missing or weakens hard budgets")
    if metadata.get("protocol_hash") != sha256_file(PILOT_PROTOCOL_PATH):
        raise PilotPreflightError("Pilot job protocol hash does not match the current protocol")
    if optimization_steps is not None and optimization_steps > int(
        budget["max_optimization_steps"]
    ):
        raise PilotBudgetError("Hardware SPSA steps exceed the pilot limit")
    if validation_examples is not None and validation_examples > int(
        budget["max_validation_examples"]
    ):
        raise PilotBudgetError("Hardware validation examples exceed the pilot limit")


def save_parameter_checkpoint(
    directory: Path,
    checkpoint_name: str,
    parameters: np.ndarray,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if not checkpoint_name or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for char in checkpoint_name
    ):
        raise ValueError("checkpoint_name contains unsafe filename characters")
    values = np.asarray(parameters, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Checkpoint parameters must be a finite vector")
    directory.mkdir(parents=True, exist_ok=True)
    array_path = directory / f"{checkpoint_name}.npz"
    metadata_path = directory / f"{checkpoint_name}.json"
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=directory,
            prefix=f".{checkpoint_name}.",
            suffix=".npz.tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            np.savez_compressed(stream, parameters=values)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, array_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    payload = {
        **dict(metadata),
        "checkpoint_name": checkpoint_name,
        "parameter_sha256": parameter_sha256(values),
        "array_file_sha256": sha256_file(array_path),
        "created_at": utc_now(),
    }
    payload["checkpoint_metadata_sha256"] = canonical_json_sha256(payload)
    _atomic_json(metadata_path, payload)
    return payload


def verify_parameter_checkpoint(directory: Path, checkpoint_name: str) -> dict[str, Any]:
    array_path = directory / f"{checkpoint_name}.npz"
    metadata_path = directory / f"{checkpoint_name}.json"
    if not array_path.is_file() or not metadata_path.is_file():
        raise PilotPreflightError("Checkpoint array or metadata is missing")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    recorded_metadata_hash = payload.pop("checkpoint_metadata_sha256", None)
    if canonical_json_sha256(payload) != recorded_metadata_hash:
        raise PilotPreflightError("Checkpoint metadata integrity check failed")
    if sha256_file(array_path) != payload.get("array_file_sha256"):
        raise PilotPreflightError("Checkpoint array hash does not match metadata")
    with np.load(array_path, allow_pickle=False) as archive:
        parameters = archive["parameters"]
    if parameter_sha256(parameters) != payload.get("parameter_sha256"):
        raise PilotPreflightError("Checkpoint parameter hash does not match metadata")
    payload["checkpoint_metadata_sha256"] = recorded_metadata_hash
    return payload


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, indent=2, allow_nan=False, default=_json_default)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_json_metadata(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_json(path, payload)


@contextmanager
def _exclusive_budget_lock(directory: Path):
    lock_path = directory / ".budget.lock"
    try:
        descriptor = os.open(
            str(lock_path),
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as exc:
        raise PilotPreflightError(
            "Pilot job ledger is locked or has a stale lock; reconcile before submitting"
        ) from exc
    try:
        os.write(descriptor, f"pid={os.getpid()} time={utc_now()}".encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


class DurableJobRecordStore:
    """Persist submission intent before network submission and block retries."""

    _UNRESOLVED = {
        "submission_intent",
        "submission_unknown",
        "submitted",
        "completion_unknown",
        "completed_unhashed",
        "evaluation_incomplete",
    }

    def __init__(
        self,
        directory: Path,
        *,
        max_jobs: Optional[int] = None,
        max_shots: Optional[int] = None,
    ):
        self.directory = directory
        self.max_jobs = max_jobs
        self.max_shots = max_shots
        self.directory.mkdir(parents=True, exist_ok=True)

    def create_intent(
        self,
        record_id: str,
        metadata: Mapping[str, Any],
        *,
        pubs: int,
        shots_per_pub: int,
        circuit_sha256: str,
    ) -> Path:
        with _exclusive_budget_lock(self.directory):
            return self._create_intent_locked(
                record_id,
                metadata,
                pubs=pubs,
                shots_per_pub=shots_per_pub,
                circuit_sha256=circuit_sha256,
            )

    def _create_intent_locked(
        self,
        record_id: str,
        metadata: Mapping[str, Any],
        *,
        pubs: int,
        shots_per_pub: int,
        circuit_sha256: str,
    ) -> Path:
        if not record_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in record_id):
            raise ValueError("record_id contains unsafe filename characters")
        existing_records = []
        for path in self.directory.glob("*.json"):
            existing = json.loads(path.read_text(encoding="utf-8"))
            existing_records.append(existing)
            if existing.get("status") in self._UNRESOLVED:
                raise PilotPreflightError(
                    "An unresolved QPU submission record exists; reconcile it before submitting"
                )
        path = self.directory / f"{record_id}.json"
        if path.exists():
            raise PilotPreflightError(
                "This submission record already exists; automatic resubmission is blocked"
            )
        required = (
            "run_id",
            "experiment_id",
            "protocol_hash",
            "dataset_hashes",
            "dataset_provenance",
            "pca_hash",
            "parameter_hash",
            "logical_qubits",
            "backend",
            "physical_layout",
            "transpiler_configuration",
            "optimizer_state",
            "random_seed",
            "sampling_policy",
            "calibration_metadata",
        )
        missing = [key for key in required if key not in metadata]
        if missing:
            raise PilotPreflightError(f"Job metadata is missing fields: {missing}")
        next_shots = int(pubs) * int(shots_per_pub)
        if metadata.get("experiment_id") == PILOT_EXPERIMENT_ID:
            protocol = load_protocol()
            optimizer_state = metadata.get("optimizer_state", {})
            evaluation_role = optimizer_state.get("evaluation_role")
            sampling_policy = metadata.get("sampling_policy", {})
            if evaluation_role in {"spsa_plus", "spsa_minus", "fixed_reference"}:
                step = int(optimizer_state.get("step", 0))
                if not 1 <= step <= int(
                    protocol["resource_budget"]["max_optimization_steps"]
                ):
                    raise PilotBudgetError("Hardware SPSA step exceeds the pilot limit")
                if evaluation_role == "spsa_plus":
                    plus_count = sum(
                        record.get("experiment_id") == PILOT_EXPERIMENT_ID
                        and record.get("optimizer_state", {}).get("evaluation_role")
                        == "spsa_plus"
                        for record in existing_records
                    )
                    if plus_count >= int(
                        protocol["resource_budget"]["max_optimization_steps"]
                    ):
                        raise PilotBudgetError(
                            "Cumulative Hardware SPSA step budget exceeded"
                        )
                same_step_roles = {
                    record.get("optimizer_state", {}).get("evaluation_role")
                    for record in existing_records
                    if record.get("experiment_id") == PILOT_EXPERIMENT_ID
                    and record.get("run_id") == metadata.get("run_id")
                    and int(record.get("optimizer_state", {}).get("step", 0)) == step
                }
                run_roles_by_step: dict[int, set[Any]] = {}
                for record in existing_records:
                    state = record.get("optimizer_state", {})
                    if (
                        record.get("experiment_id") == PILOT_EXPERIMENT_ID
                        and record.get("run_id") == metadata.get("run_id")
                        and state.get("evaluation_role")
                        in {"spsa_plus", "spsa_minus", "fixed_reference"}
                    ):
                        recorded_step = int(state.get("step", 0))
                        run_roles_by_step.setdefault(recorded_step, set()).add(
                            state["evaluation_role"]
                        )
                if evaluation_role == "spsa_plus":
                    expected_step = 1
                    reference_every = int(
                        protocol["hardware_spsa"]["reference_every"]
                    )
                    while True:
                        completed_roles = run_roles_by_step.get(expected_step, set())
                        step_complete = {"spsa_plus", "spsa_minus"}.issubset(
                            completed_roles
                        )
                        reference_complete = (
                            not reference_every
                            or expected_step % reference_every != 0
                            or "fixed_reference" in completed_roles
                        )
                        if not step_complete or not reference_complete:
                            break
                        expected_step += 1
                    if step != expected_step:
                        raise PilotPreflightError(
                            "SPSA plus evaluations must follow completed steps in order"
                        )
                if evaluation_role in same_step_roles:
                    raise PilotPreflightError(
                        "This SPSA evaluation role is already recorded for the step"
                    )
                if evaluation_role == "spsa_minus" and "spsa_plus" not in same_step_roles:
                    raise PilotPreflightError("SPSA minus requires its plus evaluation")
                if evaluation_role == "fixed_reference":
                    reference_every = int(
                        protocol["hardware_spsa"]["reference_every"]
                    )
                    if (
                        not reference_every
                        or step % reference_every
                        or not {"spsa_plus", "spsa_minus"}.issubset(same_step_roles)
                    ):
                        raise PilotPreflightError(
                            "Reference evaluation is not valid for this SPSA step"
                        )
            elif sampling_policy.get("split") in {"validation", "final_test"}:
                examples = int(metadata.get("validation_examples", 0))
                if examples < 1:
                    raise PilotPreflightError(
                        "Pilot validation metadata must record its example count"
                    )
                recorded_examples = sum(
                    int(record.get("validation_examples", 0))
                    for record in existing_records
                    if record.get("experiment_id") == PILOT_EXPERIMENT_ID
                )
                if recorded_examples + examples > int(
                    protocol["resource_budget"]["max_validation_examples"]
                ):
                    raise PilotBudgetError(
                        "Cumulative hardware validation example budget exceeded"
                    )
            else:
                raise PilotPreflightError(
                    "Pilot submission has an unsupported evaluation role"
                )
        if self.max_jobs is not None and len(existing_records) + 1 > self.max_jobs:
            raise PilotBudgetError(
                f"QPU job budget exceeded: {len(existing_records) + 1}>{self.max_jobs}"
            )
        recorded_shots = sum(int(record.get("shots", 0)) for record in existing_records)
        if self.max_shots is not None and recorded_shots + next_shots > self.max_shots:
            raise PilotBudgetError(
                f"QPU shot budget exceeded: {recorded_shots + next_shots}>{self.max_shots}"
            )
        record = {
            **dict(metadata),
            "record_id": record_id,
            "status": "submission_intent",
            "job_id": None,
            "submission_time": utc_now(),
            "completion_time": None,
            "pubs": int(pubs),
            "circuits": int(pubs),
            "shots_per_pub": int(shots_per_pub),
            "shots": next_shots,
            "circuit_sha256": circuit_sha256,
            "result_sha256": None,
            "qpu_seconds": None,
            "resource_units": None,
            "failure_type": None,
        }
        _atomic_json(path, record)
        return path

    def check_budget(self, additional_jobs: int, additional_shots: int) -> None:
        with _exclusive_budget_lock(self.directory):
            records = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in self.directory.glob("*.json")
            ]
            if any(record.get("status") in self._UNRESOLVED for record in records):
                raise PilotPreflightError(
                    "An unresolved QPU submission record exists; reconcile it before continuing"
                )
            total_jobs = len(records) + int(additional_jobs)
            total_shots = sum(int(record.get("shots", 0)) for record in records)
            total_shots += int(additional_shots)
            if self.max_jobs is not None and total_jobs > self.max_jobs:
                raise PilotBudgetError(
                    f"QPU job budget exceeded: {total_jobs}>{self.max_jobs}"
                )
            if self.max_shots is not None and total_shots > self.max_shots:
                raise PilotBudgetError(
                    f"QPU shot budget exceeded: {total_shots}>{self.max_shots}"
                )

    def save_result(self, record_path: Path, data: bytes) -> str:
        if record_path.parent.resolve() != self.directory.resolve():
            raise PilotPreflightError("Job record path is not owned by this store")
        results_dir = self.directory / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        result_path = results_dir / f"{record_path.stem}.json"
        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=results_dir,
                prefix=f".{record_path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, result_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        return str(result_path.relative_to(self.directory))

    def update(self, path: Path, updates: Mapping[str, Any]) -> None:
        if path.parent.resolve() != self.directory.resolve() or not path.is_file():
            raise PilotPreflightError("Job record path is not owned by this store")
        record = json.loads(path.read_text(encoding="utf-8"))
        record.update(dict(updates))
        _atomic_json(path, record)


def _hash_pub_circuits(pubs: Sequence[Any]) -> str:
    import qiskit.qpy

    output = io.BytesIO()
    for pub in pubs:
        circuit = pub[0] if isinstance(pub, (tuple, list)) else pub
        qiskit.qpy.dump(circuit, output)
    return hashlib.sha256(output.getvalue()).hexdigest()


def _result_bytes(result: Any) -> bytes:
    if isinstance(result, (dict, list, tuple, str, int, float, bool)):
        return json.dumps(
            result, sort_keys=True, separators=(",", ":"), allow_nan=False,
            default=_json_default,
        ).encode("utf-8")
    rows = []
    for pub_result in result:
        measurement = getattr(getattr(pub_result, "data", None), "meas", None)
        get_counts = getattr(measurement, "get_counts", None)
        if not callable(get_counts):
            raise TypeError("Sampler result does not expose measurement counts")
        counts = get_counts()
        rows.append({str(key): int(value) for key, value in sorted(counts.items())})
    return json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


class _DurableJob:
    def __init__(self, job: Any, store: DurableJobRecordStore, record_path: Path):
        self._job = job
        self._store = store
        self._record_path = record_path

    def job_id(self) -> str:
        return str(self._job.job_id())

    def metrics(self) -> Any:
        return self._job.metrics()

    def result(self) -> Any:
        try:
            result = self._job.result()
        except Exception as exc:
            status = "completion_unknown"
            try:
                job_status = str(self._job.status()).lower()
                if "cancel" in job_status:
                    status = "cancelled"
                elif any(
                    terminal in job_status
                    for terminal in ("error", "failed", "jobstatus.error")
                ):
                    status = "failed"
            except Exception:
                pass
            self._store.update(
                self._record_path,
                {
                    "status": status,
                    "completion_time": utc_now(),
                    "failure_type": type(exc).__name__,
                },
            )
            raise

        try:
            result_data = _result_bytes(result)
            result_hash = hashlib.sha256(result_data).hexdigest()
            result_file = self._store.save_result(self._record_path, result_data)
        except Exception as exc:
            self._store.update(
                self._record_path,
                {
                    "status": "completed_unhashed",
                    "completion_time": utc_now(),
                    "failure_type": type(exc).__name__,
                },
            )
            raise

        metrics: dict[str, Any] = {}
        try:
            raw_metrics = self._job.metrics()
            usage = raw_metrics.get("usage", {}) if isinstance(raw_metrics, Mapping) else {}
            metrics["qpu_seconds"] = usage.get(
                "quantum_seconds",
                raw_metrics.get("quantum_seconds") if isinstance(raw_metrics, Mapping) else None,
            )
            metrics["resource_units"] = usage.get(
                "resource_units",
                raw_metrics.get("resource_units") if isinstance(raw_metrics, Mapping) else None,
            )
        except Exception as exc:
            metrics["metrics_unavailable_type"] = type(exc).__name__
        self._store.update(
            self._record_path,
            {
                **metrics,
                "status": (
                    "evaluation_incomplete"
                    if json.loads(
                        self._record_path.read_text(encoding="utf-8")
                    ).get("experiment_id") == PILOT_EXPERIMENT_ID
                    else "completed"
                ),
                "completion_time": utc_now(),
                "result_sha256": result_hash,
                "result_file": result_file,
            },
        )
        return result

    def cancel(self) -> Any:
        result = self._job.cancel()
        self._store.update(
            self._record_path,
            {"status": "cancelled", "completion_time": utc_now()},
        )
        return result


class DurableSampler:
    """Sampler proxy that writes an intent and job ID before awaiting results."""

    def __init__(
        self,
        sampler: Any,
        store: DurableJobRecordStore,
        base_metadata: Mapping[str, Any],
        *,
        circuit_hasher: Callable[[Sequence[Any]], str] = _hash_pub_circuits,
    ):
        self._sampler = sampler
        self._store = store
        self._base_metadata = dict(base_metadata)
        validate_pilot_job_context(store, self._base_metadata)
        self._circuit_hasher = circuit_hasher
        self._context: Optional[dict[str, Any]] = None
        self._sequence = 0
        self._last_record_path: Optional[Path] = None

    def set_execution_context(self, context: Mapping[str, Any]) -> None:
        self._context = dict(context)

    def update_last_record(self, updates: Mapping[str, Any]) -> None:
        if self._last_record_path is None:
            raise PilotPreflightError("No sampler record is available to update")
        record_updates = dict(updates)
        if (
            self._base_metadata.get("experiment_id") == PILOT_EXPERIMENT_ID
            and ("evaluation_loss" in record_updates or "validation_metrics" in record_updates)
        ):
            record_updates["status"] = "completed"
        self._store.update(self._last_record_path, record_updates)

    def run(self, pubs: Sequence[Any], *, shots: int) -> _DurableJob:
        if self._context is None:
            raise PilotPreflightError("Durable sampler has no optimizer/evaluation context")
        if shots < 1 or not pubs:
            raise ValueError("Sampler submissions require non-empty PUBs and positive shots")
        circuit_hash = self._circuit_hasher(pubs)
        self._sequence += 1
        run_id = str(self._base_metadata["run_id"])
        record_id = f"{run_id}-job-{self._sequence:04d}"
        record_metadata = {**self._base_metadata, **self._context}
        if self._base_metadata.get("experiment_id") == PILOT_EXPERIMENT_ID:
            protocol = load_protocol()
            optimizer_state = record_metadata.get("optimizer_state", {})
            sampling_policy = record_metadata.get("sampling_policy", {})
            evaluation_role = optimizer_state.get("evaluation_role")
            if evaluation_role in {"spsa_plus", "spsa_minus", "fixed_reference"}:
                step = int(optimizer_state.get("step", 0))
                validate_pilot_job_context(
                    self._store,
                    self._base_metadata,
                    optimization_steps=step,
                )
                expected_batch = (
                    protocol["hardware_spsa"]["reference_examples"]
                    if evaluation_role == "fixed_reference"
                    else protocol["hardware_spsa"]["batch_size"]
                )
                sample_indices = sampling_policy.get("sample_indices", ())
                expected_shots = (
                    protocol["hardware_spsa"]["reference_shots"]
                    if evaluation_role == "fixed_reference"
                    else protocol["hardware_spsa"]["shots"]
                )
                if (
                    len(sample_indices) != int(expected_batch)
                    or len(pubs) != int(expected_batch)
                    or shots != int(expected_shots)
                ):
                    raise PilotPreflightError(
                        "Pilot SPSA sample count or shots differ from the protocol"
                    )
            elif sampling_policy.get("split") in {"validation", "final_test"}:
                selected_indices = sampling_policy.get("selected_indices", ())
                record_metadata["validation_examples"] = len(selected_indices)
                expected_scales = protocol["validation"]["zne_scale_factors"]
                scale_factors = sampling_policy.get("zne_scale_factors", [1])
                if (
                    len(pubs) != len(selected_indices) * len(expected_scales)
                    or scale_factors != expected_scales
                    or shots != int(protocol["validation"]["shots"])
                ):
                    raise PilotPreflightError(
                        "Pilot validation circuit count or shots differ from the protocol"
                    )
                validate_pilot_job_context(
                    self._store,
                    self._base_metadata,
                    validation_examples=len(selected_indices),
                )
            else:
                raise PilotPreflightError(
                    "Pilot sampler context must identify SPSA or validation work"
                )
        record_path = self._store.create_intent(
            record_id,
            record_metadata,
            pubs=len(pubs),
            shots_per_pub=shots,
            circuit_sha256=circuit_hash,
        )
        self._last_record_path = record_path
        try:
            job = self._sampler.run(pubs, shots=shots)
        except Exception as exc:
            self._store.update(
                record_path,
                {
                    "status": "submission_unknown",
                    "failure_type": type(exc).__name__,
                },
            )
            raise
        try:
            job_id = str(job.job_id())
            if not job_id:
                raise RuntimeError("IBM returned an empty job ID")
        except Exception as exc:
            self._store.update(
                record_path,
                {
                    "status": "submission_unknown",
                    "failure_type": type(exc).__name__,
                },
            )
            raise PilotPreflightError(
                "Submission may have reached IBM but its job ID was not persisted; reconcile manually"
            ) from exc
        self._store.update(
            record_path,
            {
                "status": "submitted",
                "job_id": job_id,
                "job_id_persisted_time": utc_now(),
            },
        )
        return _DurableJob(job, self._store, record_path)
