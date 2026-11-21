"""Run one frozen QNIM PCA-scaling configuration at a time."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENT_DIR = (
    ROOT / "results" / "scaling_study" / "qnim_quantum_pca_scaling_v1"
)
PROTOCOL_PATH = EXPERIMENT_DIR / "protocol.json"
DATASET_DIR = ROOT / "results" / "scaling_study" / "qnim_q1_6_dataset_v2_pca64"
PROTOCOL_SHA256 = "87581f0b3d0a435ecfd44be59145a13bf824977c732f03d401f30520eb335d8e"
EXPERIMENT_ID = "QNIM_QUANTUM_PCA_SCALING_V1"
CONFIGURATIONS = (16, 20)
SEED = 42
TRAINING_RANDOM_SEED = 42
DIAGNOSTIC_SIMULATOR_SEED = 42042
DIAGNOSTIC_SAMPLES_PER_CLASS = 5
MAX_ITERATIONS = 20
BATCH_SIZE = 32
SHOTS = 512
LEARNING_RATE = 0.03
FEATURE_MAP_REPS = 2
ANSATZ_REPS = 2
ENTANGLEMENT = "linear"
READOUT_MODE = "marginals_zz_neighbors"
READOUT_HIDDEN_SIZE = 16
N_CLASSES = 13
CHECKPOINT_NAMES = {16: "q16", 20: "q20"}
ATTEMPT_POLICY = (
    "Each fresh execution gets a unique attempt directory. --new-attempt "
    "restarts from seed 42; optimizer state is not restored."
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array, dtype=np.float64).tobytes()
    ).hexdigest()


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_json(path: Path, payload: dict) -> None:
    content = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    _atomic_bytes(path, content)


@contextmanager
def _output_lock(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".runner.lock"
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
    except FileExistsError as exc:
        raise FileExistsError(
            "Output directory is locked by another run or a stale lock; "
            "verify no process is active before removing the lock."
        ) from exc
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        lock_path.unlink()


def _output_file(output_dir: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError("Execution manifest result paths must be relative")
    root = output_dir.resolve()
    result_path = (root / relative).resolve()
    try:
        result_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Execution manifest result path escapes output directory") from exc
    return result_path


def _read_protocol(path: Path = PROTOCOL_PATH) -> tuple[dict, bytes, str]:
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != PROTOCOL_SHA256:
        raise ValueError("Frozen experiment protocol SHA-256 mismatch")
    protocol = json.loads(content)
    if protocol.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Unexpected experiment ID in frozen protocol")
    return protocol, content, digest


def _load_dataset(protocol: dict, dataset_dir: Path = DATASET_DIR) -> tuple[dict, dict, bytes]:
    dataset_config = protocol["dataset"]
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_digest != dataset_config["dataset_manifest_sha256"]:
        raise ValueError("Immutable PCA64 dataset manifest SHA-256 mismatch")
    manifest = json.loads(manifest_bytes)
    if (
        manifest.get("dataset_id") != dataset_config["dataset_id"]
        or manifest.get("status") != "immutable_complete"
    ):
        raise ValueError("Unexpected or incomplete PCA64 dataset manifest")

    required_files = dataset_config["allowed_files"]
    if set(required_files) != set(dataset_config["file_sha256"]) | {
        "dataset_manifest.json"
    }:
        raise ValueError("Frozen dataset file allowlist is inconsistent")
    verified_hashes = {}
    for filename, expected_hash in dataset_config["file_sha256"].items():
        if manifest.get("file_sha256", {}).get(filename) != expected_hash:
            raise ValueError(f"Dataset manifest hash pin differs for {filename}")
        actual_hash = _sha256_file(dataset_dir / filename)
        if actual_hash != expected_hash:
            raise ValueError(f"PCA64 source file SHA-256 mismatch: {filename}")
        verified_hashes[filename] = actual_hash

    arrays = {
        "X_train": np.load(
            dataset_dir / "X_train_scaled64.npy", allow_pickle=False
        ),
        "X_val": np.load(
            dataset_dir / "X_val_scaled64.npy", allow_pickle=False
        ),
        "y_train": np.load(dataset_dir / "y_train.npy", allow_pickle=False),
        "y_val": np.load(dataset_dir / "y_val.npy", allow_pickle=False),
    }
    with np.load(dataset_dir / "pca_state.npz", allow_pickle=False) as state:
        pca_info = {
            "mean_shape": list(state["mean"].shape),
            "components_shape": list(state["components"].shape),
            "n_samples_seen": int(state["n_samples_seen"]),
            "n_features_in": int(state["n_features_in"]),
            "n_components": int(state["components"].shape[0]),
        }

    if arrays["X_train"].shape != tuple(dataset_config["train_shape"]):
        raise ValueError(f"Unexpected X_train shape: {arrays['X_train'].shape}")
    if arrays["X_val"].shape != tuple(dataset_config["validation_shape"]):
        raise ValueError(f"Unexpected X_val shape: {arrays['X_val'].shape}")
    if arrays["y_train"].shape != (dataset_config["train_shape"][0],):
        raise ValueError("Unexpected y_train shape")
    if arrays["y_val"].shape != (dataset_config["validation_shape"][0],):
        raise ValueError("Unexpected y_val shape")
    if pca_info != {
        "mean_shape": [16384],
        "components_shape": [64, 16384],
        "n_samples_seen": 2600,
        "n_features_in": 16384,
        "n_components": 64,
    }:
        raise ValueError(f"Unexpected persisted PCA state: {pca_info}")
    for name in ("X_train", "X_val"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} contains non-finite values")
        if np.min(arrays[name]) < -1.0 - 1e-12 or np.max(
            arrays[name]
        ) > 1.0 + 1e-12:
            raise ValueError(f"{name} is outside the frozen MinMax range [-1,1]")
    expected_labels = np.arange(dataset_config["classes"])
    for split in ("train", "val"):
        labels = arrays[f"y_{split}"]
        if not np.array_equal(np.unique(labels), expected_labels):
            raise ValueError(f"{split} labels do not contain all 13 classes")
        counts = np.bincount(labels, minlength=dataset_config["classes"])
        expected_count = 200 if split == "train" else 50
        if not np.all(counts == expected_count):
            raise ValueError(f"{split} split is not balanced as frozen")

    info = {
        "dataset_id": dataset_config["dataset_id"],
        "dataset_manifest_sha256": manifest_digest,
        "verified_file_sha256": verified_hashes,
        "train_shape": list(arrays["X_train"].shape),
        "validation_shape": list(arrays["X_val"].shape),
        "train_label_shape": list(arrays["y_train"].shape),
        "validation_label_shape": list(arrays["y_val"].shape),
        "train_class_counts": np.bincount(
            arrays["y_train"], minlength=dataset_config["classes"]
        ).astype(int).tolist(),
        "validation_class_counts": np.bincount(
            arrays["y_val"], minlength=dataset_config["classes"]
        ).astype(int).tolist(),
        "pca_state": pca_info,
        "only_train_validation_loaded": True,
        "test_access": False,
    }
    return arrays, info, manifest_bytes


def _build_local_aer(seed: int):
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2

    backend = AerSimulator(
        method="matrix_product_state",
        seed_simulator=seed,
    )
    return backend, SamplerV2.from_backend(backend)


def _inspect_configuration(qubits: int, protocol: dict) -> dict:
    if qubits not in CONFIGURATIONS:
        raise ValueError("This protocol supports only --qubits 16 or --qubits 20")
    from qiskit import transpile
    from src.infrastructure.qiskit_vqc_trainer import (
        _build_feature_map_and_ansatz,
        _n_readout_params,
        _qubit_readout_features,
        _readout_feature_count,
        _split_readout_mlp,
    )

    configuration = next(
        row for row in protocol["configurations"] if row["n_qubits"] == qubits
    )
    circuit, x_params, ansatz_params = _build_feature_map_and_ansatz(
        qubits,
        n_features=qubits,
        feature_map_reps=FEATURE_MAP_REPS,
        ansatz_reps=ANSATZ_REPS,
        entanglement=ENTANGLEMENT,
    )
    if circuit.num_qubits != qubits or len(x_params) != qubits:
        raise RuntimeError("Feature map dimensions do not match the selected PCA prefix")
    if len(ansatz_params) != configuration["expected_ansatz_parameters"]:
        raise RuntimeError(
            f"Installed Qiskit EfficientSU2 has {len(ansatz_params)} parameters; "
            f"protocol expects {configuration['expected_ansatz_parameters']}"
        )
    expected_by_reps = (ANSATZ_REPS + 1) * 2 * qubits
    if len(ansatz_params) != expected_by_reps:
        raise RuntimeError("EfficientSU2 parameter count differs from the rotation-layer check")

    n_measurements = _readout_feature_count(qubits, READOUT_MODE)
    if n_measurements != 2 * qubits - 1:
        raise RuntimeError("Neighbor readout does not produce 2*q-1 features")
    n_readout_parameters = _n_readout_params(
        n_measurements, N_CLASSES, READOUT_HIDDEN_SIZE
    )
    theta_probe = np.zeros(len(ansatz_params) + n_readout_parameters)
    readout_blocks = _split_readout_mlp(
        theta_probe,
        len(ansatz_params),
        n_measurements,
        N_CLASSES,
        READOUT_HIDDEN_SIZE,
    )
    observed_readout_parameters = sum(block.size for block in readout_blocks)
    if observed_readout_parameters != n_readout_parameters:
        raise RuntimeError("MLP readout parameter split/count verification failed")
    total_parameters = len(ansatz_params) + observed_readout_parameters
    measured_circuit = circuit.copy()
    measured_circuit.measure_all()
    backend, sampler = _build_local_aer(TRAINING_RANDOM_SEED)
    measured_circuit = transpile(
        measured_circuit,
        backend=backend,
        optimization_level=1,
    )
    if measured_circuit.num_qubits != qubits:
        raise RuntimeError("Transpiled circuit changed the logical qubit count")
    if measured_circuit.num_parameters != len(x_params) + len(ansatz_params):
        raise RuntimeError("Transpilation changed the encoded/trainable parameter count")
    measurement_instructions = [
        instruction
        for instruction in measured_circuit.data
        if instruction.operation.name == "measure"
    ]
    measured_qubits = [
        measured_circuit.find_bit(instruction.qubits[0]).index
        for instruction in measurement_instructions
    ]
    measurement_map = [
        {
            "qubit": measured_circuit.find_bit(instruction.qubits[0]).index,
            "classical_bit": measured_circuit.find_bit(instruction.clbits[0]).index,
        }
        for instruction in measurement_instructions
    ]
    if len(measurement_instructions) != qubits or sorted(measured_qubits) != list(
        range(qubits)
    ):
        raise RuntimeError("Quantum circuit does not measure every qubit exactly once")
    probe_features = _qubit_readout_features(
        {"0" * qubits: 1}, qubits, READOUT_MODE
    )
    if probe_features.shape != (n_measurements,):
        raise RuntimeError("Measurement decoder feature dimension is incorrect")

    backend_config = {
        "provider": "Qiskit Aer",
        "backend_name": backend.name,
        "method": backend.options.method,
        "device": backend.options.device,
        "seed_simulator": TRAINING_RANDOM_SEED,
        "sampler": type(sampler).__name__,
        "local": True,
        "hardware_execution": False,
    }
    software_versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "qiskit": importlib.metadata.version("qiskit"),
        "qiskit_aer": importlib.metadata.version("qiskit-aer"),
        "scikit_learn": importlib.metadata.version("scikit-learn"),
    }
    del sampler
    del backend

    rss_now = psutil.Process().memory_info().rss
    statevector_bytes = 16 * (1 << qubits)
    batch_state_bytes = statevector_bytes * BATCH_SIZE * 2
    qnspsa_matrix_bytes = 2 * total_parameters**2 * np.dtype(np.float64).itemsize
    estimated_peak = rss_now + batch_state_bytes + qnspsa_matrix_bytes
    circuit_ops = {
        str(name): int(count)
        for name, count in measured_circuit.count_ops().items()
    }
    return {
        "configuration": configuration,
        "n_qubits": qubits,
        "n_pca_components": qubits,
        "selected_pca_columns": list(range(qubits)),
        "n_feature_map_parameters": len(x_params),
        "ansatz_parameter_count": len(ansatz_params),
        "ansatz_parameter_formula_check": {
            "expected": expected_by_reps,
            "observed": len(ansatz_params),
            "matches": len(ansatz_params) == expected_by_reps,
        },
        "classical_readout_parameter_count": observed_readout_parameters,
        "classical_readout_parameter_count_formula": (
            f"({n_measurements}*{READOUT_HIDDEN_SIZE}+{READOUT_HIDDEN_SIZE})"
            f"+({READOUT_HIDDEN_SIZE}*{N_CLASSES}+{N_CLASSES})"
        ),
        "total_trainable_parameter_count": total_parameters,
        "measurement_features": {
            "expected": 2 * qubits - 1,
            "observed_decoder_shape": list(probe_features.shape),
            "observables": (
                [f"P(q{i}=0)" for i in range(qubits)]
                + [f"<Z{i} Z{i + 1}>" for i in range(qubits - 1)]
            ),
            "ordering": "all P(q_i=0) in ascending qubit order, then adjacent ZZ in ascending pair order",
            "bitstring_convention": (
                "Qiskit little-endian: bitstring[-1] is qubit 0; "
                "marginal is P(bit=0)"
            ),
            "measured_qubits": len(measurement_instructions),
            "qubit_to_classical_bit": measurement_map,
            "feature_producer": (
                "_measure_qubit_marginals_via_circuit -> "
                "_qubit_readout_features -> classical MLP"
            ),
        },
        "circuit": {
            "feature_map": "ZZFeatureMap",
            "feature_map_reps": FEATURE_MAP_REPS,
            "feature_map_entanglement": ENTANGLEMENT,
            "ansatz": "EfficientSU2",
            "ansatz_reps": ANSATZ_REPS,
            "ansatz_entanglement": ENTANGLEMENT,
            "ansatz_rotation_gates": ["ry", "rz"],
            "num_qubits": measured_circuit.num_qubits,
            "num_clbits": measured_circuit.num_clbits,
            "num_parameters_including_data": measured_circuit.num_parameters,
            "depth_with_measurements": measured_circuit.depth(),
            "operation_counts_with_measurements": circuit_ops,
            "measurement_register": "meas",
        },
        "readout": {
            "mode": READOUT_MODE,
            "hidden_size": READOUT_HIDDEN_SIZE,
            "output_classes": N_CLASSES,
            "activation": "ReLU hidden layer; softmax output",
        },
        "shot_configuration": {
            "training": SHOTS,
            "reference": SHOTS,
            "warm_start_and_refit": SHOTS,
            "diagnostic": SHOTS,
            "final_evaluation": SHOTS,
        },
        "simulator": backend_config,
        "software_versions": software_versions,
        "resource_estimate": {
            "current_process_rss_bytes": int(rss_now),
            "dense_statevector_bytes_per_circuit": int(statevector_bytes),
            "batch_size": BATCH_SIZE,
            "batch_statevector_envelope_bytes": int(batch_state_bytes),
            "qnspsa_dense_matrix_envelope_bytes": int(qnspsa_matrix_bytes),
            "estimated_peak_rss_bytes": int(estimated_peak),
            "estimated_peak_rss_mib": float(estimated_peak / 2**20),
            "estimator_note": (
                "Heuristic upper envelope using two dense-equivalent simulator buffers "
                "per batch circuit plus two dense QNSPSA parameter matrices; MPS use "
                "depends on entanglement and Aer scheduling, so actual peak can differ."
            ),
        },
    }


def _new_execution_manifest(protocol_hash: str, dataset_info: dict) -> dict:
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "prepared",
        "protocol_sha256": protocol_hash,
        "protocol_file": "protocol.json",
        "dataset_id": dataset_info["dataset_id"],
        "dataset_manifest_sha256": dataset_info["dataset_manifest_sha256"],
        "dataset_manifest_file": "dataset_manifest.json",
        "dataset_integrity": dataset_info,
        "attempt_policy": ATTEMPT_POLICY,
        "checkpoint_directories": {
            "16": str(Path("checkpoints") / "q16"),
            "20": str(Path("checkpoints") / "q20"),
        },
        "configurations": {
            str(qubits): {
                "status": "not_started",
                "attempts": [],
                "result_directory": CHECKPOINT_NAMES[qubits],
            }
            for qubits in CONFIGURATIONS
        },
        "test_access": False,
        "updated_at_utc": _utc_now(),
    }


def _prepare_output(
    output_dir: Path,
    protocol_content: bytes,
    protocol_hash: str,
    dataset_info: dict,
    manifest_content: bytes,
) -> dict:
    protocol_copy = output_dir / "protocol.json"
    manifest_copy = output_dir / "dataset_manifest.json"
    execution_path = output_dir / "execution_manifest.json"

    if execution_path.exists():
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        if (
            execution.get("experiment_id") != EXPERIMENT_ID
            or execution.get("protocol_sha256") != protocol_hash
            or execution.get("dataset_manifest_sha256")
            != dataset_info["dataset_manifest_sha256"]
            or execution.get("dataset_integrity") != dataset_info
        ):
            raise ValueError("Existing execution manifest does not match this run")
        execution["attempt_policy"] = ATTEMPT_POLICY
        for qubits in CONFIGURATIONS:
            config_row = execution.get("configurations", {}).get(str(qubits))
            if not isinstance(config_row, dict):
                raise ValueError("Execution manifest is missing a configuration")
            config_root = output_dir / CHECKPOINT_NAMES[qubits]
            result_path = config_root / "result.json"
            markdown_path = config_root / "result.md"
            if config_row.get("status") == "completed":
                expected_hash = config_row.get("result_sha256")
                stored_path = config_row.get("result_file")
                if not expected_hash or not stored_path:
                    raise ValueError(
                        f"Completed q{qubits} manifest lacks result integrity metadata"
                    )
                verified_path = _output_file(output_dir, stored_path)
                if (
                    verified_path != result_path.resolve()
                    or not verified_path.is_file()
                    or _sha256_file(verified_path) != expected_hash
                ):
                    raise ValueError(
                        f"Completed q{qubits} result hash verification failed"
                    )
                markdown_hash = config_row.get("result_markdown_sha256")
                if (
                    not markdown_hash
                    or not markdown_path.is_file()
                    or _sha256_file(markdown_path) != markdown_hash
                ):
                    raise ValueError(
                        f"Completed q{qubits} Markdown result hash verification failed"
                    )
                completed_attempts = [
                    row
                    for row in config_row.get("attempts", [])
                    if row.get("status") == "completed"
                ]
                if len(completed_attempts) != 1:
                    raise ValueError(
                        f"Completed q{qubits} must have exactly one completed attempt"
                    )
                attempt = completed_attempts[0]
                attempt_path = _output_file(
                    output_dir, attempt.get("result_file", "")
                )
                attempt_markdown_path = _output_file(
                    output_dir, attempt.get("result_markdown_file", "")
                )
                if (
                    not attempt.get("result_sha256")
                    or not attempt_path.is_file()
                    or _sha256_file(attempt_path) != attempt["result_sha256"]
                    or not attempt.get("result_markdown_sha256")
                    or not attempt_markdown_path.is_file()
                    or _sha256_file(attempt_markdown_path)
                    != attempt["result_markdown_sha256"]
                ):
                    raise ValueError(
                        f"Completed q{qubits} attempt result hash verification failed"
                    )
            elif result_path.exists() or markdown_path.exists():
                raise FileExistsError(
                    f"q{qubits} has result files but is not marked completed"
                )

        for qubits in CONFIGURATIONS:
            config_row = execution["configurations"][str(qubits)]
            expected_attempts = {
                row["directory"]
                for row in config_row.get("attempts", [])
                if row.get("directory")
            }
            checkpoint_root = output_dir / "checkpoints" / CHECKPOINT_NAMES[qubits]
            actual_attempts = {
                str(path.relative_to(output_dir))
                for path in checkpoint_root.glob("attempt_*")
            } if checkpoint_root.exists() else set()
            if actual_attempts != expected_attempts:
                raise FileExistsError(
                    f"q{qubits} checkpoint directories do not match the execution manifest"
                )
    else:
        orphan_paths = [
            output_dir / CHECKPOINT_NAMES[qubits] / "result.json"
            for qubits in CONFIGURATIONS
        ] + [
            output_dir / CHECKPOINT_NAMES[qubits] / "result.md"
            for qubits in CONFIGURATIONS
        ]
        orphan_paths.extend(
            path
            for qubits in CONFIGURATIONS
            for path in (
                output_dir / "checkpoints" / CHECKPOINT_NAMES[qubits]
            ).glob("attempt_*")
        )
        if any(path.exists() for path in orphan_paths):
            raise FileExistsError(
                "Output directory contains orphan results or checkpoint attempts "
                "without an execution manifest"
            )
        execution = _new_execution_manifest(protocol_hash, dataset_info)

    if protocol_copy.exists():
        if _sha256_file(protocol_copy) != protocol_hash:
            raise ValueError("Experiment output contains a different frozen protocol")
    if manifest_copy.exists():
        if _sha256_file(manifest_copy) != dataset_info["dataset_manifest_sha256"]:
            raise ValueError("Experiment output contains a different dataset manifest")

    output_dir.mkdir(parents=True, exist_ok=True)
    if not protocol_copy.exists():
        _atomic_bytes(protocol_copy, protocol_content)
    if not manifest_copy.exists():
        _atomic_bytes(manifest_copy, manifest_content)
    return execution


def _diagnostic_indices(
    labels: np.ndarray, per_class: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = np.concatenate(
        [
            rng.choice(
                np.flatnonzero(labels == label),
                size=per_class,
                replace=False,
            )
            for label in np.unique(labels)
        ]
    )
    selected.sort()
    return selected


def _classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        log_loss,
        recall_score,
    )

    predictions = np.argmax(probabilities, axis=1)
    class_ids = np.arange(N_CLASSES)
    return {
        "global_accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=class_ids,
                average="macro",
                zero_division=0,
            )
        ),
        "cross_entropy": float(
            log_loss(
                labels,
                np.clip(probabilities, 1e-12, 1.0),
                labels=class_ids,
            )
        ),
        "per_class_recall": recall_score(
            labels,
            predictions,
            labels=class_ids,
            average=None,
            zero_division=0,
        ).astype(float).tolist(),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=class_ids
        ).astype(int).tolist(),
    }


def _result_markdown(result: dict) -> str:
    required_paths = (
        "status",
        "configuration.n_qubits",
        "configuration.n_pca_components",
        "input.train_examples",
        "input.validation_examples",
        "parameters.quantum_ansatz_count",
        "parameters.classical_readout_count",
        "parameters.total_count",
        "training_protocol.iterations",
        "training_protocol.shots",
        "training_protocol.reference_shots",
        "optimization.termination_reason",
        "optimizer_telemetry.accepted_updates",
        "optimizer_telemetry.rejected_updates",
        "execution_accounting.n_objective_function_calls",
        "execution_accounting.n_optimizer_objective_function_calls",
        "execution_accounting.n_circuit_executions",
        "execution_accounting.n_shots_total",
        "execution_accounting.n_training_examples_evaluated",
        "execution_accounting.n_readout_refits",
        "metrics.training.balanced_accuracy",
        "metrics.training.global_accuracy",
        "metrics.training.macro_f1",
        "metrics.training.cross_entropy",
        "metrics.validation.balanced_accuracy",
        "metrics.validation.global_accuracy",
        "metrics.validation.macro_f1",
        "metrics.validation.cross_entropy",
        "training_trace",
        "optimizer_telemetry.steps",
    )
    for path in required_paths:
        value = result
        for key in path.split("."):
            if not isinstance(value, dict) or key not in value:
                raise ValueError(
                    f"Completed result is missing required field '{path}'"
                )
            value = value[key]

    if result["status"] != "completed":
        raise ValueError("Result Markdown requires status='completed'")
    iterations = result["training_protocol"]["iterations"]
    training_trace = result["training_trace"]
    optimizer_steps = result["optimizer_telemetry"]["steps"]
    if not isinstance(training_trace, list) or len(training_trace) != iterations:
        raise ValueError(
            "Completed result field 'training_trace' must contain exactly "
            f"{iterations} optimizer iterations"
        )
    if not isinstance(optimizer_steps, list) or len(optimizer_steps) != iterations:
        raise ValueError(
            "Completed result field 'optimizer_telemetry.steps' must contain "
            f"exactly {iterations} optimizer iterations"
        )
    for row in training_trace:
        iteration = row.get("iteration")
        for path in (
            "readout_refit.scheduled",
            "pre_refit.minibatch_training_loss",
            "pre_refit.diagnostic_training_metrics.balanced_accuracy",
            "pre_refit.diagnostic_validation_metrics.balanced_accuracy",
        ):
            value = row
            for key in path.split("."):
                if not isinstance(value, dict) or key not in value:
                    raise ValueError(
                        f"Completed result is missing required field "
                        f"'training_trace[{iteration}].{path}'"
                    )
                value = value[key]
        if row["readout_refit"]["scheduled"]:
            for path in (
                "post_refit.minibatch_training_loss",
                "post_refit.diagnostic_training_metrics.balanced_accuracy",
                "post_refit.diagnostic_validation_metrics.balanced_accuracy",
            ):
                value = row
                for key in path.split("."):
                    if not isinstance(value, dict) or key not in value:
                        raise ValueError(
                            f"Completed result is missing required field "
                            f"'training_trace[{iteration}].{path}'"
                        )
                    value = value[key]

    training = result["metrics"]["training"]
    validation = result["metrics"]["validation"]
    configuration = result["configuration"]
    accounting = result["execution_accounting"]
    accepted = result["optimizer_telemetry"]["accepted_updates"]
    rejected = result["optimizer_telemetry"]["rejected_updates"]
    refit_rows = []
    for row in training_trace:
        if not row["readout_refit"]["scheduled"]:
            continue
        post = row["post_refit"]
        post_loss = f"{post['minibatch_training_loss']:.6f}"
        post_train = (
            f"{post['diagnostic_training_metrics']['balanced_accuracy']:.4f}"
        )
        post_val = (
            f"{post['diagnostic_validation_metrics']['balanced_accuracy']:.4f}"
        )
        refit_rows.append(
            "| {iteration} | {pre_loss:.6f} | {post_loss} | "
            "{pre_train:.4f} | {post_train} | {pre_val:.4f} | {post_val} |".format(
                iteration=row["iteration"],
                pre_loss=row["pre_refit"]["minibatch_training_loss"],
                post_loss=post_loss,
                pre_train=row["pre_refit"]["diagnostic_training_metrics"][
                    "balanced_accuracy"
                ],
                post_train=post_train,
                pre_val=row["pre_refit"]["diagnostic_validation_metrics"][
                    "balanced_accuracy"
                ],
                post_val=post_val,
            )
        )
    refit_table = "\n".join(refit_rows) or "| — | — | — | — | — | — | — |"
    return f"""# QNIM quantum PCA scaling — q{configuration['n_qubits']}

- Status: **{result['status']}**
- PCA components / qubits: {configuration['n_pca_components']} / {configuration['n_qubits']}
- Train / validation examples: {result['input']['train_examples']} / {result['input']['validation_examples']}
- Trainable parameters: quantum {result['parameters']['quantum_ansatz_count']}, classical readout {result['parameters']['classical_readout_count']}, total {result['parameters']['total_count']}.
- Simulator: local Aer `matrix_product_state`; shots per circuit: {result['training_protocol']['shots']} (reference evaluations explicitly use {result['training_protocol']['reference_shots']}).
- Iterations / termination: {iterations} / `{result['optimization']['termination_reason']}`.
- Accepted / rejected QNSPSA updates: {accepted} / {rejected}.
- Objective-function calls: {accounting['n_objective_function_calls']} total; {accounting['n_optimizer_objective_function_calls']} from QNSPSA.
- Circuit executions / shots: {accounting['n_circuit_executions']} / {accounting['n_shots_total']} (measured from completed sampler PUB results).
- Training examples evaluated, including repeats: {accounting['n_training_examples_evaluated']}; periodic readout refits: {accounting['n_readout_refits']}.

## Final metrics

| Split | Balanced accuracy | Global accuracy | Macro-F1 | Cross-entropy |
|---|---:|---:|---:|---:|
| Train | {training['balanced_accuracy']:.4f} | {training['global_accuracy']:.4f} | {training['macro_f1']:.4f} | {training['cross_entropy']:.4f} |
| Validation | {validation['balanced_accuracy']:.4f} | {validation['global_accuracy']:.4f} | {validation['macro_f1']:.4f} | {validation['cross_entropy']:.4f} |

Per-iteration loss and fixed diagnostic-subset metrics, per-class recall, confusion matrices, and full optimizer telemetry are in this configuration's `result.json`.

Validation measurements were diagnostic only and were never used in loss evaluation, gradients, optimizer updates, or parameter fitting. No test split was accessed.

## Readout-refit state consistency

Each scheduled refit records pre-refit and post-refit loss and metrics separately. Both stages use the same optimizer training minibatch and the same fixed diagnostic training/validation indices, listed in `result.json`; the post-refit loss is freshly evaluated. A `not recorded` post-refit row indicates an interrupted or failed run, not an accepted completed result.

| Iteration | Pre loss | Post loss | Pre train balanced accuracy | Post train balanced accuracy | Pre validation balanced accuracy | Post validation balanced accuracy |
|---:|---:|---:|---:|---:|---:|---:|
{refit_table}

The legacy `optimization.n_circuit_evaluations` field is retained for old consumers; it counts optimizer objective calls, not circuit executions. Use `execution_accounting` for the separated measured counters.
"""


def _save_progress(
    checkpoint_path: Path,
    execution_path: Path,
    execution: dict,
    qubits: int,
    attempt_record: dict,
    progress: dict,
) -> None:
    attempt_record["latest_iteration"] = progress["iteration"]
    execution["updated_at_utc"] = _utc_now()
    execution["configurations"][str(qubits)]["latest_iteration"] = progress[
        "iteration"
    ]
    _atomic_json(checkpoint_path, progress)
    _atomic_json(execution_path, execution)


def _train_configuration(
    qubits: int,
    seed: int,
    dataset: dict,
    config_info: dict,
    progress_callback: Callable[[dict], None],
) -> dict:
    from qiskit import transpile
    from src.infrastructure.qiskit_vqc_trainer import (
        QiskitVQCTrainer,
        _build_feature_map_and_ansatz,
        _measure_qubit_marginals_via_circuit,
        _n_readout_params,
        _readout_feature_count,
        _readout_mlp_probs,
        _split_readout_mlp,
    )

    configuration = config_info["configuration"]
    X_train = dataset["X_train"][:, :qubits]
    X_val = dataset["X_val"][:, :qubits]
    y_train = dataset["y_train"]
    y_val = dataset["y_val"]
    n_observables = _readout_feature_count(qubits, READOUT_MODE)
    n_quantum = config_info["ansatz_parameter_count"]
    n_classical = _n_readout_params(
        n_observables, N_CLASSES, READOUT_HIDDEN_SIZE
    )
    training_backend, training_sampler = _build_local_aer(
        TRAINING_RANDOM_SEED
    )
    diagnostic_backend, diagnostic_sampler = _build_local_aer(
        DIAGNOSTIC_SIMULATOR_SEED
    )

    metric_circuit, x_params, ansatz_params = _build_feature_map_and_ansatz(
        qubits,
        n_features=qubits,
        feature_map_reps=FEATURE_MAP_REPS,
        ansatz_reps=ANSATZ_REPS,
        entanglement=ENTANGLEMENT,
    )
    metric_circuit.measure_all()
    metric_circuit = transpile(
        metric_circuit,
        backend=diagnostic_backend,
        optimization_level=1,
    )
    if len(ansatz_params) != n_quantum:
        raise RuntimeError("Training circuit ansatz count differs from dry-run")

    train_indices = _diagnostic_indices(
        y_train, DIAGNOSTIC_SAMPLES_PER_CLASS, seed
    )
    validation_indices = _diagnostic_indices(
        y_val, DIAGNOSTIC_SAMPLES_PER_CLASS, seed + 1
    )
    training_trace = []
    evaluation_accounting = {
        "n_circuit_executions": 0,
        "n_shots_total": 0,
        "n_training_examples_evaluated": 0,
        "n_validation_examples_evaluated": 0,
    }

    def evaluate_split(
        features: np.ndarray,
        labels: np.ndarray,
        theta: np.ndarray,
        split: str,
    ):
        quantum_features = _measure_qubit_marginals_via_circuit(
            metric_circuit,
            x_params,
            ansatz_params,
            theta[:n_quantum],
            features,
            qubits,
            shots=SHOTS,
            batch_size=BATCH_SIZE,
            sampler=diagnostic_sampler,
            simulator_seed=DIAGNOSTIC_SIMULATOR_SEED,
            readout_mode=READOUT_MODE,
        )
        if quantum_features.shape != (len(labels), n_observables):
            raise RuntimeError("Measured quantum readout shape differs from protocol")
        n_examples = len(labels)
        evaluation_accounting["n_circuit_executions"] += n_examples
        evaluation_accounting["n_shots_total"] += n_examples * SHOTS
        split_counts = {
            "train": "n_training_examples_evaluated",
            "validation": "n_validation_examples_evaluated",
        }
        if split not in split_counts:
            raise ValueError(f"Unknown evaluation split: {split}")
        split_count = split_counts[split]
        evaluation_accounting[split_count] += n_examples
        W1, b1, W2, b2 = _split_readout_mlp(
            theta,
            n_quantum,
            n_observables,
            N_CLASSES,
            READOUT_HIDDEN_SIZE,
        )
        probabilities = np.asarray(
            [
                _readout_mlp_probs(row, W1, b1, W2, b2)
                for row in quantum_features
            ]
        )
        return _classification_metrics(labels, probabilities), quantum_features

    def diagnostic_metrics(theta: np.ndarray) -> tuple[dict, dict]:
        train_metrics, _ = evaluate_split(
            X_train[train_indices], y_train[train_indices], theta, "train"
        )
        validation_metrics, _ = evaluate_split(
            X_val[validation_indices],
            y_val[validation_indices],
            theta,
            "validation",
        )
        return train_metrics, validation_metrics

    def diagnostic_callback(iteration: int, theta: np.ndarray, loss: float):
        train_metrics, validation_metrics = diagnostic_metrics(theta)
        refit_scheduled = iteration % 10 == 0
        trace_entry = {
            "iteration": int(iteration),
            "diagnostic_data": {
                "training_indices": train_indices.tolist(),
                "validation_indices": validation_indices.tolist(),
                "shots_per_circuit": SHOTS,
                "optimizer_loss_data": "current fixed 32-example training minibatch",
            },
            "pre_refit": {
                "minibatch_training_loss": float(loss),
                "diagnostic_training_metrics": train_metrics,
                "diagnostic_validation_metrics": validation_metrics,
                "parameter_sha256": _sha256_array(theta),
                "parameters": np.asarray(theta, dtype=np.float64).tolist(),
            },
            "post_refit": None,
            "readout_refit": {
                "scheduled": refit_scheduled,
                "status": "pending" if refit_scheduled else "not_scheduled",
            },
            "validation_role": "metrics_only_no_gradient_or_fitting",
        }
        training_trace.append(trace_entry)
        progress_callback(
            {
                "status": "running",
                "iteration": int(iteration),
                "trace": training_trace,
                "latest_parameters": trace_entry["pre_refit"]["parameters"],
                "validation_role": "metrics_only_no_gradient_or_fitting",
            }
        )

    def post_refit_callback(iteration: int, theta: np.ndarray, loss: float):
        if not training_trace or training_trace[-1]["iteration"] != iteration:
            raise RuntimeError("Post-refit metrics have no matching pre-refit state")
        train_metrics, validation_metrics = diagnostic_metrics(theta)
        trace_entry = training_trace[-1]
        trace_entry["post_refit"] = {
            "minibatch_training_loss": float(loss),
            "diagnostic_training_metrics": train_metrics,
            "diagnostic_validation_metrics": validation_metrics,
            "parameter_sha256": _sha256_array(theta),
            "parameters": np.asarray(theta, dtype=np.float64).tolist(),
        }
        trace_entry["readout_refit"]["status"] = "completed"
        progress_callback(
            {
                "status": "running",
                "iteration": int(iteration),
                "trace": training_trace,
                "latest_parameters": trace_entry["post_refit"]["parameters"],
                "validation_role": "metrics_only_no_gradient_or_fitting",
            }
        )

    trainer = QiskitVQCTrainer(
        mode="sim",
        train_on_hardware=False,
        noise_aware_training=False,
        training_batch_size=BATCH_SIZE,
        random_seed=TRAINING_RANDOM_SEED,
        patience=20,
        feature_map_reps=FEATURE_MAP_REPS,
        ansatz_reps=ANSATZ_REPS,
        entanglement=ENTANGLEMENT,
        learning_rate=LEARNING_RATE,
        readout_refit_every=10,
        readout_hidden_size=READOUT_HIDDEN_SIZE,
        readout_mode=READOUT_MODE,
        reference_shots=SHOTS,
    )
    started = time.perf_counter()
    training = trainer.train_vqc(
        X_train=X_train,
        y_train=y_train,
        num_qubits=qubits,
        n_features=qubits,
        max_iterations=MAX_ITERATIONS,
        shots=SHOTS,
        initialization_seed=seed,
        diagnostic_callback=diagnostic_callback,
        post_refit_callback=post_refit_callback,
        sampler_override=training_sampler,
        input_preprocessing="identity",
        sampler_backend_override=training_backend,
    )
    trainer_seconds = time.perf_counter() - started

    batch_history = training["optimizer_batch_indices"]
    for trace_entry in training_trace:
        batch_index = trace_entry["iteration"] + 1
        if batch_index >= len(batch_history):
            raise RuntimeError("Optimizer minibatch trace is incomplete")
        trace_entry["diagnostic_data"][
            "optimizer_loss_training_indices"
        ] = batch_history[batch_index]
    refit_by_iteration = {
        row["iteration"]: row
        for row in training["readout_refit_history"]
    }
    for trace_entry in training_trace:
        if trace_entry["readout_refit"]["scheduled"]:
            refit = refit_by_iteration.get(trace_entry["iteration"])
            if refit is None:
                raise RuntimeError("Scheduled readout refit is missing from telemetry")
            trace_entry["readout_refit"].update(refit)

    weights = np.asarray(training["weights"], dtype=np.float64)
    expected_total = n_quantum + n_classical
    if weights.shape != (expected_total,):
        raise RuntimeError("Trained vector differs from verified parameter counts")
    for step, trace_entry in zip(
        training.get("optimizer_step_history", []), training_trace
    ):
        trace_entry["optimizer_telemetry"] = step

    initial_quantum = np.random.default_rng(seed).normal(
        0.0, 0.01, n_quantum
    )
    all_train_metrics, all_train_features = evaluate_split(
        X_train, y_train, weights, "train"
    )
    all_validation_metrics, all_validation_features = evaluate_split(
        X_val, y_val, weights, "validation"
    )
    if (
        all_train_features.shape != (len(y_train), 2 * qubits - 1)
        or all_validation_features.shape != (len(y_val), 2 * qubits - 1)
    ):
        raise RuntimeError("Final classifier features were not quantum measurements")

    optimizer_steps = training.get("optimizer_step_history", [])
    accepted = sum(bool(row["update_accepted"]) for row in optimizer_steps)
    rejected = len(optimizer_steps) - accepted
    termination = training["termination_reason"]
    if not termination:
        raise RuntimeError("Optimizer did not report a termination reason")
    final_quantum = weights[:n_quantum]
    final_classical = weights[n_quantum:]
    return {
        "status": "completed",
        "experiment_id": EXPERIMENT_ID,
        "configuration": {
            "name": configuration["name"],
            "n_qubits": qubits,
            "n_pca_components": qubits,
            "initialization_seed": seed,
            "training_random_seed": TRAINING_RANDOM_SEED,
        },
        "fit_scope": "train_only",
        "input": {
            "source": "X_train_scaled64.npy/X_val_scaled64.npy",
            "selected_columns_zero_based": list(range(qubits)),
            "preprocessing": "identity; existing PCA MinMax scaling used unchanged",
            "train_examples": int(len(y_train)),
            "validation_examples": int(len(y_val)),
            "measurement_feature_shapes": {
                "train": list(all_train_features.shape),
                "validation": list(all_validation_features.shape),
            },
        },
        "circuit": config_info["circuit"],
        "readout": config_info["readout"],
        "backend": config_info["simulator"],
        "parameters": {
            "quantum_ansatz_count": n_quantum,
            "classical_readout_count": n_classical,
            "total_count": expected_total,
            "feature_map_parameters_are_data_encoded_not_trainable": True,
            "quantum_initialization": initial_quantum.tolist(),
            "quantum_initialization_sha256": _sha256_array(initial_quantum),
            "quantum_final": final_quantum.tolist(),
            "quantum_final_sha256": _sha256_array(final_quantum),
            "classical_readout_final": final_classical.tolist(),
            "classical_readout_final_sha256": _sha256_array(final_classical),
        },
        "training_protocol": {
            "optimizer": "QNSPSA-EML-Feynman",
            "max_iterations": MAX_ITERATIONS,
            "iterations": int(training["iterations"]),
            "batch_size": BATCH_SIZE,
            "shots": SHOTS,
            "reference_shots": SHOTS,
            "learning_rate": LEARNING_RATE,
            "initialization_seed": seed,
            "training_random_seed": TRAINING_RANDOM_SEED,
            "feature_map_reps": FEATURE_MAP_REPS,
            "ansatz_reps": ANSATZ_REPS,
            "readout_hidden_size": READOUT_HIDDEN_SIZE,
            "readout_mode": READOUT_MODE,
            "input_preprocessing": "identity",
        },
        "metrics": {
            "training": all_train_metrics,
            "validation": all_validation_metrics,
            "diagnostic_subset_samples_per_class": DIAGNOSTIC_SAMPLES_PER_CLASS,
            "diagnostic_validation_used_for_updates": False,
        },
        "training_trace": training_trace,
        "optimization": {
            "loss_history": [float(value) for value in training["loss_history"]],
            "final_loss": float(training["training_loss"]),
            "converged": bool(training["converged"]),
            "termination_reason": termination,
            "n_circuit_evaluations": int(
                training["n_circuit_evaluations"]
            ),
            "n_optimizer_objective_function_calls": int(
                training["n_optimizer_objective_function_calls"]
            ),
            "legacy_n_circuit_evaluations_semantics": (
                "Backward-compatible field; this value counts optimizer "
                "objective-function calls, not physical circuit executions."
            ),
            "gradient_variance_history": [
                float(value)
                for value in training["gradient_variance_history"]
            ],
        },
        "optimizer_telemetry": {
            "accepted_updates": int(accepted),
            "rejected_updates": int(rejected),
            "steps": optimizer_steps,
        },
        "execution_accounting": {
            "status": "measured",
            "method": (
                "Count completed SamplerV2 PUB results; each submitted PUB "
                "contains one bound input circuit. Shot total sums the "
                "explicitly requested shots for each completed PUB."
            ),
            "n_objective_function_calls": int(
                training["n_objective_function_calls"]
            ),
            "n_optimizer_objective_function_calls": int(
                training["n_optimizer_objective_function_calls"]
            ),
            "n_circuit_executions": int(
                training["n_circuit_executions"]
                + evaluation_accounting["n_circuit_executions"]
            ),
            "n_shots_total": int(
                training["n_shots_total"]
                + evaluation_accounting["n_shots_total"]
            ),
            "n_training_examples_evaluated": int(
                training["n_training_examples_evaluated"]
                + evaluation_accounting["n_training_examples_evaluated"]
            ),
            "n_validation_examples_evaluated": int(
                evaluation_accounting["n_validation_examples_evaluated"]
            ),
            "n_readout_refits": int(training["n_readout_refits"]),
            "n_readout_initializations": int(
                training["n_readout_initializations"]
            ),
            "estimated": False,
            "example_count_semantics": (
                "Circuit-evaluated examples, including repeated examples "
                "across optimizer calls, refits, diagnostics, and final metrics."
            ),
        },
        "timing": {
            "trainer_seconds_including_diagnostic_metrics": float(trainer_seconds),
            "total_fit_and_full_evaluation_seconds": float(
                time.perf_counter() - started
            ),
        },
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "qiskit": importlib.metadata.version("qiskit"),
            "qiskit_aer": importlib.metadata.version("qiskit-aer"),
        },
        "test_access": False,
    }


def _refresh_manifest_status(manifest: dict) -> None:
    states = [
        row["status"] for row in manifest["configurations"].values()
    ]
    if all(state == "completed" for state in states):
        manifest["status"] = "completed"
    elif any(state == "running" for state in states):
        manifest["status"] = "running"
    elif any(state in {"failed", "interrupted"} for state in states):
        manifest["status"] = "partial"
    else:
        manifest["status"] = "prepared"
    manifest["updated_at_utc"] = _utc_now()


def _run_single_configuration_locked(
    *,
    qubits: int,
    seed: int = SEED,
    dry_run: bool = False,
    new_attempt: bool = False,
    output_dir: Path | None = None,
) -> dict:
    if qubits not in CONFIGURATIONS:
        raise ValueError("--qubits must be 16 or 20")
    if seed != SEED:
        raise ValueError("QNIM_QUANTUM_PCA_SCALING_V1 freezes --seed at 42")
    if dry_run and new_attempt:
        raise ValueError("--dry-run and --new-attempt are mutually exclusive")
    output_dir = output_dir or EXPERIMENT_DIR
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    protocol, protocol_content, protocol_hash = _read_protocol()
    dataset, dataset_info, manifest_content = _load_dataset(protocol)
    config_info = _inspect_configuration(qubits, protocol)
    execution = _prepare_output(
        output_dir,
        protocol_content,
        protocol_hash,
        dataset_info,
        manifest_content,
    )
    execution_path = output_dir / "execution_manifest.json"
    config_row = execution["configurations"][str(qubits)]

    if dry_run:
        config_row["last_dry_run_utc"] = _utc_now()
        config_row["dry_run_configuration"] = config_info
        _refresh_manifest_status(execution)
        _atomic_json(execution_path, execution)
        return {
            "status": "dry_run",
            "experiment_id": EXPERIMENT_ID,
            "configuration": config_info,
            "dataset_integrity": dataset_info,
            "training_started": False,
            "test_access": False,
        }

    if config_row["status"] == "completed":
        if new_attempt:
            raise FileExistsError(
                f"q{qubits} is completed; create a new output directory for another run"
            )
        return {
            "status": "already_completed",
            "configuration": str(qubits),
            "training_started": False,
            "test_access": False,
        }
    if config_row["status"] == "running":
        raise FileExistsError(
            "An attempt is still marked running; verify the process stopped and "
            "mark it interrupted before starting a new attempt."
        )
    if config_row["attempts"] and not new_attempt:
        raise FileExistsError(
            f"q{qubits} has an incomplete attempt; --new-attempt starts a fresh "
            "optimization from seed 42 and does not restore optimizer state"
        )

    config_dir = output_dir / "checkpoints" / CHECKPOINT_NAMES[qubits]
    attempt_number = len(config_row["attempts"]) + 1
    attempt_id = uuid.uuid4().hex
    attempt_dir = config_dir / f"attempt_{attempt_number:02d}_{attempt_id}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = attempt_dir / "checkpoint.json"
    attempt_record = {
        "attempt": attempt_number,
        "attempt_id": attempt_id,
        "status": "running",
        "started_at_utc": _utc_now(),
        "directory": str(attempt_dir.relative_to(output_dir)),
    }
    config_row["attempts"].append(attempt_record)
    config_row["status"] = "running"
    config_row["active_attempt"] = attempt_number
    _refresh_manifest_status(execution)
    _atomic_json(execution_path, execution)
    _atomic_json(
        checkpoint_path,
        {
            "status": "running",
            "experiment_id": EXPERIMENT_ID,
            "qubits": qubits,
            "seed": seed,
            "protocol_sha256": protocol_hash,
            "dataset_manifest_sha256": dataset_info[
                "dataset_manifest_sha256"
            ],
            "iteration": 0,
            "trace": [],
            "validation_role": "metrics_only_no_gradient_or_fitting",
        },
    )

    def progress_callback(progress: dict) -> None:
        _save_progress(
            checkpoint_path,
            execution_path,
            execution,
            qubits,
            attempt_record,
            progress,
        )

    try:
        result = _train_configuration(
            qubits,
            seed,
            dataset,
            config_info,
            progress_callback,
        )
        if result.get("status") != "completed":
            raise RuntimeError(
                "Training function returned without a completed status"
            )
        result["protocol_sha256"] = protocol_hash
        result["dataset_manifest_sha256"] = dataset_info[
            "dataset_manifest_sha256"
        ]
        result["dataset_file_sha256"] = dataset_info["verified_file_sha256"]
        result["integrity"] = dataset_info
        _atomic_json(attempt_dir / "result.json", result)
        attempt_markdown_path = attempt_dir / "result.md"
        _atomic_bytes(
            attempt_markdown_path,
            _result_markdown(result).encode("utf-8"),
        )
        config_root = output_dir / CHECKPOINT_NAMES[qubits]
        config_root.mkdir(parents=True, exist_ok=True)
        _atomic_json(config_root / "result.json", result)
        config_markdown_path = config_root / "result.md"
        _atomic_bytes(
            config_markdown_path,
            _result_markdown(result).encode("utf-8"),
        )
        attempt_result_path = attempt_dir / "result.json"
        config_result_path = config_root / "result.json"
        _atomic_json(
            checkpoint_path,
            {
                "status": "completed",
                "experiment_id": EXPERIMENT_ID,
                "qubits": qubits,
                "seed": seed,
                "iteration": result["training_protocol"]["iterations"],
                "trace": result["training_trace"],
                "result_sha256": _sha256_file(attempt_result_path),
                "result_markdown_sha256": _sha256_file(
                    attempt_markdown_path
                ),
                "validation_role": "metrics_only_no_gradient_or_fitting",
            },
        )
        attempt_record.update(
            {
                "status": "completed",
                "ended_at_utc": _utc_now(),
                "result_file": str(
                    attempt_result_path.relative_to(output_dir)
                ),
                "result_sha256": _sha256_file(attempt_result_path),
                "result_markdown_file": str(
                    attempt_markdown_path.relative_to(output_dir)
                ),
                "result_markdown_sha256": _sha256_file(
                    attempt_markdown_path
                ),
            }
        )
        config_row["status"] = "completed"
        config_row.pop("active_attempt", None)
        config_row["result_file"] = str(
            config_result_path.relative_to(output_dir)
        )
        config_row["result_sha256"] = _sha256_file(config_result_path)
        config_row["result_markdown_sha256"] = _sha256_file(
            config_markdown_path
        )
        _refresh_manifest_status(execution)
        _atomic_json(execution_path, execution)
        return {
            "status": "completed",
            "configuration": str(qubits),
            "result": result,
            "training_started": True,
            "test_access": False,
        }
    except KeyboardInterrupt:
        attempt_record.update(
            {"status": "interrupted", "ended_at_utc": _utc_now()}
        )
        config_row["status"] = "interrupted"
        config_row.pop("active_attempt", None)
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        checkpoint["status"] = "interrupted"
        _atomic_json(checkpoint_path, checkpoint)
        _refresh_manifest_status(execution)
        _atomic_json(execution_path, execution)
        return {
            "status": "interrupted",
            "configuration": str(qubits),
            "training_started": True,
            "test_access": False,
        }
    except Exception as exc:
        attempt_record.update(
            {
                "status": "failed",
                "ended_at_utc": _utc_now(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        config_row["status"] = "failed"
        config_row.pop("active_attempt", None)
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        checkpoint.update(
            {
                "status": "failed",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        _atomic_json(checkpoint_path, checkpoint)
        _refresh_manifest_status(execution)
        _atomic_json(execution_path, execution)
        raise


def run_single_configuration(
    *,
    qubits: int,
    seed: int = SEED,
    dry_run: bool = False,
    new_attempt: bool = False,
    output_dir: Path | None = None,
) -> dict:
    if qubits not in CONFIGURATIONS:
        raise ValueError("--qubits must be 16 or 20")
    if seed != SEED:
        raise ValueError("QNIM_QUANTUM_PCA_SCALING_V1 freezes --seed at 42")
    if dry_run and new_attempt:
        raise ValueError("--dry-run and --new-attempt are mutually exclusive")
    output_dir = output_dir or EXPERIMENT_DIR
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    _read_protocol()
    with _output_lock(output_dir):
        return _run_single_configuration_locked(
            qubits=qubits,
            seed=seed,
            dry_run=dry_run,
            new_attempt=new_attempt,
            output_dir=output_dir,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument(
        "--new-attempt",
        action="store_true",
        help=(
            "Start a uniquely checkpointed fresh optimization after an interrupted "
            "or failed attempt; optimizer state is not restored."
        ),
    )
    parser.add_argument("--qubits", type=int, required=True, choices=CONFIGURATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    result = run_single_configuration(
        qubits=args.qubits,
        seed=args.seed,
        dry_run=args.dry_run,
        new_attempt=args.new_attempt,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 130 if result["status"] == "interrupted" else 0


if __name__ == "__main__":
    raise SystemExit(main())
