"""Execute the approved QNIM neighbor-readout pilot, without accessing test data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PROTOCOL_PATH = ROOT / "results" / "scaling_study" / "qnim_neighbor_readout_pilot_v1_protocol.json"
DATASET_MANIFEST_PATH = (
    ROOT / "results" / "scaling_study" / "qnim_neighbor_readout_pilot_v1_dataset_manifest.json"
)
DATASET_DIR = ROOT / "results" / "scaling_study" / "qnim_q1_6_dataset_v1"
OUTPUT_DIR = ROOT / "results" / "scaling_study" / "qnim_neighbor_readout_pilot_v1"
SMOKE_OUTPUT_DIR = ROOT / "results" / "scaling_study" / "qnim_neighbor_readout_pilot_v1_smoke"

APPROVED_PROTOCOL_SHA256 = "4e6b8915a61bc0b92fa4a6940bd9214b1be99dce761366d1bc56f9eb8fd2e7ea"
APPROVED_DATASET_MANIFEST_SHA256 = "b70eafec3a32acc56c15af9cad0c51998cb93f733d70d307f806ab57f1e87ed9"
FEATURE_MAP_REPS = 2
ANSATZ_REPS = 2
ENTANGLEMENT = "linear"
READOUT_MODE = "marginals_zz_neighbors"
READOUT_HIDDEN_SIZE = 16
N_CLASSES = 13
MAX_ITERATIONS = 20
BATCH_SIZE = 32
SHOTS = 512
TRAINING_RANDOM_SEED = 42
INITIALIZATION_SEEDS = (42, 43, 44)
PATIENCE = 20
LEARNING_RATE = 0.03

QNIM_MATRIX = (
    {"model_family": "qnim", "arm": "full_system", "n_qubits": 4, "pca_components": 4},
    {"model_family": "qnim", "arm": "full_system", "n_qubits": 8, "pca_components": 8},
    {"model_family": "qnim", "arm": "full_system", "n_qubits": 12, "pca_components": 12},
    {"model_family": "qnim", "arm": "fixed_input_dimension", "n_qubits": 8, "pca_components": 4},
    {"model_family": "qnim", "arm": "fixed_input_dimension", "n_qubits": 12, "pca_components": 4},
)
SVM_DIMENSIONS = (4, 8, 12)


class PeakMemoryMonitor:
    def __init__(self, interval_seconds: float = 0.1):
        self.interval_seconds = interval_seconds
        self.process = psutil.Process()
        self.initial_bytes = 0
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> PeakMemoryMonitor:
        self.initial_bytes = self.process.memory_info().rss
        self.peak_bytes = self.initial_bytes
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)

    def as_dict(self) -> dict[str, int | float]:
        return {
            "rss_start_bytes": self.initial_bytes,
            "rss_peak_bytes": self.peak_bytes,
            "peak_rss_delta_bytes": max(0, self.peak_bytes - self.initial_bytes),
            "sampling_interval_seconds": self.interval_seconds,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
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
    _atomic_bytes(path, _json_bytes(payload))


def split_sha256(features: np.ndarray, labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (features, labels):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _verify_protocol(protocol_path: Path) -> tuple[dict, bytes, str]:
    content = protocol_path.read_bytes()
    protocol_hash = _sha256_bytes(content)
    if protocol_hash != APPROVED_PROTOCOL_SHA256:
        raise ValueError(
            "Approved protocol SHA-256 changed; refusing to run the experiment"
        )
    protocol = json.loads(content)
    if protocol.get("experiment_id") != "QNIM_SCALING_NEIGHBOR_PILOT_V1":
        raise ValueError("Unexpected experiment_id in approved protocol")
    if protocol.get("architecture", {}).get("readout_mode") != READOUT_MODE:
        raise ValueError("Protocol readout mode does not match the implementation")
    if protocol.get("configurations") != [
        {key: row[key] for key in ("arm", "n_qubits", "pca_components")}
        for row in QNIM_MATRIX
    ]:
        raise ValueError("Protocol configuration matrix differs from the runner")
    training = protocol.get("training", {})
    expected_training = {
        "max_iterations": MAX_ITERATIONS,
        "batch_size": BATCH_SIZE,
        "shots": SHOTS,
        "initialization_seeds": list(INITIALIZATION_SEEDS),
        "training_random_seed": TRAINING_RANDOM_SEED,
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
    }
    if any(training.get(key) != value for key, value in expected_training.items()):
        raise ValueError("Training budget differs from the approved protocol")
    return protocol, content, protocol_hash


def _read_dataset_manifest(
    manifest_path: Path, protocol_hash: str,
) -> tuple[dict, bytes, str]:
    content = manifest_path.read_bytes()
    manifest_hash = _sha256_bytes(content)
    if manifest_hash != APPROVED_DATASET_MANIFEST_SHA256:
        raise ValueError("Dataset manifest SHA-256 changed; refusing to run")
    manifest = json.loads(content)
    if (
        manifest.get("experiment_id") != "QNIM_SCALING_NEIGHBOR_PILOT_V1"
        or manifest.get("status") != "approved_immutable"
        or manifest.get("approved_protocol_sha256") != protocol_hash
    ):
        raise ValueError("Dataset manifest does not belong to the approved protocol")
    if set(manifest.get("train_validation_hashes", {})) != {
        "train_sha256", "validation_sha256"
    }:
        raise ValueError("Dataset manifest is missing train/validation hashes")
    if set(manifest.get("historical_q1_first_12_component_hashes", {})) != {
        "train_sha256", "validation_sha256"
    }:
        raise ValueError("Dataset manifest is missing the historical PCA-prefix hashes")
    if manifest.get("pca_prefix_check", {}).get("components") != 12:
        raise ValueError("Dataset manifest must validate the first 12 PCA components")
    return manifest, content, manifest_hash


def build_execution_plan(
    *, arm: str | None = None, n_qubits: int | None = None,
    seed: int | None = None,
) -> list[dict]:
    selection_values = (arm, n_qubits, seed)
    if any(value is not None for value in selection_values) and not all(
        value is not None for value in selection_values
    ):
        raise ValueError("--arm, --qubits, and --seed must be supplied together")
    if arm is not None:
        if arm not in {"full_system", "fixed_input_dimension"}:
            raise ValueError("Invalid QNIM arm")
        if seed not in INITIALIZATION_SEEDS:
            raise ValueError("Seed must be one of 42, 43, or 44")

    selected_seed = seed
    plan: list[dict] = []
    for initialization_seed in INITIALIZATION_SEEDS:
        for config in QNIM_MATRIX:
            plan.append({**config, "initialization_seed": initialization_seed})
        for dimensions in SVM_DIMENSIONS:
            plan.append({
                "model_family": "pca_svm_rbf",
                "arm": "full_system",
                "n_qubits": dimensions,
                "pca_components": dimensions,
                "initialization_seed": initialization_seed,
            })
    if arm is not None:
        plan = [
            row for row in plan
            if row["model_family"] == "qnim"
            and row["arm"] == arm
            and row["n_qubits"] == n_qubits
            and row["initialization_seed"] == selected_seed
        ]
        if len(plan) != 1:
            if arm == "fixed_input_dimension" and n_qubits == 4:
                raise ValueError(
                    "fixed_input_dimension q=4 is the same fit as full_system q=4; select full_system"
                )
            raise ValueError("No unique QNIM fit matches the requested arm, qubits, and seed")
    keys = [run_key(row) for row in plan]
    if len(keys) != len(set(keys)):
        raise ValueError("Execution plan contains duplicate configurations or seeds")
    if arm is None and sum(row["model_family"] == "qnim" for row in plan) != 15:
        raise ValueError("Execution plan must contain exactly 15 QNIM fits")
    if arm is None and sum(row["model_family"] == "pca_svm_rbf" for row in plan) != 9:
        raise ValueError("Execution plan must contain exactly 9 PCA+SVM fits")
    return plan


def run_key(config: dict) -> str:
    return (
        f"{config['model_family']}|{config['arm']}|q{config['n_qubits']}"
        f"|pca{config['pca_components']}|seed{config['initialization_seed']}"
    )


def _record_hash(record: dict) -> str:
    content = {key: value for key, value in record.items() if key != "record_sha256"}
    return _sha256_bytes(json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _refresh_record_hash(record: dict) -> None:
    record.pop("record_sha256", None)
    record["record_sha256"] = _record_hash(record)


def _record_memory(monitor: PeakMemoryMonitor) -> dict:
    return monitor.as_dict()


def load_cached_dataset(
    dataset_dir: Path, dataset_manifest: dict, manifest_sha256: str,
) -> tuple[dict[str, np.ndarray], dict]:
    arrays: dict[str, np.ndarray] = {}
    for name in ("X_train", "y_train", "X_val", "y_val"):
        path = dataset_dir / f"{name}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Required cached train/validation file is missing: {path}")
        arrays[name] = np.load(path, allow_pickle=False)

    metadata_path = dataset_dir / "dataset_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Cached dataset metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    expected_shapes = {
        name: tuple(shape)
        for name, shape in dataset_manifest["expected_shapes"].items()
    }
    if set(expected_shapes) != set(arrays):
        raise ValueError("Dataset manifest dimensions do not cover the allowed cached splits")
    for name, expected_shape in expected_shapes.items():
        if arrays[name].shape != expected_shape:
            raise ValueError(f"{name} has shape {arrays[name].shape}, expected {expected_shape}")
    if not np.isfinite(arrays["X_train"]).all() or not np.isfinite(arrays["X_val"]).all():
        raise ValueError("Train/validation features contain non-finite values")
    class_names = dataset_manifest["class_names"]
    n_classes = len(class_names)
    if not np.array_equal(np.unique(arrays["y_train"]), np.arange(n_classes)):
        raise ValueError(f"Train labels do not contain the expected {n_classes} classes")
    if not np.array_equal(np.unique(arrays["y_val"]), np.arange(n_classes)):
        raise ValueError(f"Validation labels do not contain the expected {n_classes} classes")

    actual_hashes = {
        "train_sha256": split_sha256(arrays["X_train"], arrays["y_train"]),
        "validation_sha256": split_sha256(arrays["X_val"], arrays["y_val"]),
    }
    expected_hashes = dataset_manifest["train_validation_hashes"]
    if actual_hashes != expected_hashes:
        raise ValueError("Cached train/validation hashes differ from the immutable dataset manifest")
    if metadata.get("dataset_hashes") != expected_hashes:
        raise ValueError("Cached dataset metadata hashes differ from the immutable dataset manifest")

    prefix_size = dataset_manifest["pca_prefix_check"]["components"]
    prefix_hashes = {
        "train_sha256": split_sha256(arrays["X_train"][:, :prefix_size], arrays["y_train"]),
        "validation_sha256": split_sha256(arrays["X_val"][:, :prefix_size], arrays["y_val"]),
    }
    if prefix_hashes != dataset_manifest["historical_q1_first_12_component_hashes"]:
        raise ValueError("PCA prefix hashes differ from the immutable dataset manifest")
    if metadata.get("class_names") != class_names:
        raise ValueError("Cached dataset class order differs from the immutable dataset manifest")
    return arrays, {
        "full_split_hashes": actual_hashes,
        "first_12_component_hashes": prefix_hashes,
        "dataset_manifest_sha256": manifest_sha256,
        "dataset_manifest": "qnim_neighbor_readout_pilot_v1_dataset_manifest.json",
        "class_names": class_names,
        "shapes": {name: list(array.shape) for name, array in arrays.items()},
    }


def _metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        log_loss,
        recall_score,
    )

    labels = np.arange(N_CLASSES)
    predicted = np.argmax(probabilities, axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, labels=labels, average="macro", zero_division=0)),
        "cross_entropy": float(log_loss(y_true, np.clip(probabilities, 1e-12, 1.0), labels=labels)),
        "recall_per_class": recall_score(y_true, predicted, labels=labels, average=None, zero_division=0).astype(float).tolist(),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=labels).astype(int).tolist(),
    }


def _measurement_seed(config: dict, split: int) -> int:
    arm_code = 0 if config["arm"] == "full_system" else 1
    return int(np.random.SeedSequence([
        TRAINING_RANDOM_SEED,
        config["initialization_seed"],
        config["n_qubits"],
        config["pca_components"],
        arm_code,
        split,
    ]).generate_state(1)[0])


def _build_trainer():
    from src.infrastructure.qiskit_vqc_trainer import QiskitVQCTrainer

    return QiskitVQCTrainer(
        mode="sim",
        feature_map_reps=FEATURE_MAP_REPS,
        ansatz_reps=ANSATZ_REPS,
        entanglement=ENTANGLEMENT,
        patience=PATIENCE,
        learning_rate=LEARNING_RATE,
        training_batch_size=BATCH_SIZE,
        random_seed=TRAINING_RANDOM_SEED,
        readout_hidden_size=READOUT_HIDDEN_SIZE,
        readout_mode=READOUT_MODE,
    )


def _fit_qnim(dataset: dict[str, np.ndarray], config: dict) -> dict:
    from src.infrastructure.qiskit_vqc_trainer import (
        _build_feature_map_and_ansatz,
        _make_vqc_loss_fn,
        _n_readout_params,
        _readout_feature_count,
        _readout_mlp_probs,
        _split_readout_mlp,
    )

    n_qubits = config["n_qubits"]
    n_features = config["pca_components"]
    X_train = dataset["X_train"][:, :n_features]
    X_val = dataset["X_val"][:, :n_features]
    y_train, y_val = dataset["y_train"], dataset["y_val"]
    circuit, x_params, ansatz_params = _build_feature_map_and_ansatz(
        n_qubits,
        n_features=n_features,
        feature_map_reps=FEATURE_MAP_REPS,
        ansatz_reps=ANSATZ_REPS,
        entanglement=ENTANGLEMENT,
    )
    if circuit.num_qubits != n_qubits or len(x_params) != n_features:
        raise ValueError("Constructed circuit dimensions differ from the run configuration")
    n_observables = _readout_feature_count(n_qubits, READOUT_MODE)
    if n_observables != 2 * n_qubits - 1:
        raise ValueError("Neighbor readout must have exactly 2*q-1 features")
    n_quantum_params = len(ansatz_params)
    initial_ansatz = np.random.default_rng(config["initialization_seed"]).normal(
        0.0, 0.01, n_quantum_params
    )
    n_classical_params = _n_readout_params(
        n_observables, N_CLASSES, READOUT_HIDDEN_SIZE
    )
    trainer = _build_trainer()
    if trainer.training_batch_size != BATCH_SIZE:
        raise RuntimeError("Batch size was not propagated to QiskitVQCTrainer")

    trainer_started = time.perf_counter()
    training = trainer.train_vqc(
        X_train=X_train,
        y_train=y_train,
        num_qubits=n_qubits,
        n_features=n_features,
        max_iterations=MAX_ITERATIONS,
        shots=SHOTS,
        initialization_seed=config["initialization_seed"],
    )
    trainer_elapsed = time.perf_counter() - trainer_started
    weights = np.asarray(training["weights"], dtype=float)
    expected_weight_count = n_quantum_params + n_classical_params
    if weights.shape != (expected_weight_count,):
        raise ValueError("Trained parameter count does not match circuit and readout")

    loss_fn = _make_vqc_loss_fn(
        X_train=X_train,
        y_train=y_train,
        n_qubits=n_qubits,
        n_features=n_features,
        shots=SHOTS,
        mode="sim",
        batch_size_local=BATCH_SIZE,
        feature_map_reps=FEATURE_MAP_REPS,
        ansatz_reps=ANSATZ_REPS,
        entanglement=ENTANGLEMENT,
        readout_hidden_size=READOUT_HIDDEN_SIZE,
        random_seed=TRAINING_RANDOM_SEED,
        readout_mode=READOUT_MODE,
    )
    ansatz_weights = weights[:n_quantum_params]
    W1, b1, W2, b2 = _split_readout_mlp(
        weights, n_quantum_params, n_observables, N_CLASSES, READOUT_HIDDEN_SIZE
    )
    train_features = loss_fn.measure_quantum_features(
        X_train,
        ansatz_weights,
        shots_override=SHOTS,
        simulator_seed=_measurement_seed(config, 0),
    )
    validation_features = loss_fn.measure_quantum_features(
        X_val,
        ansatz_weights,
        shots_override=SHOTS,
        simulator_seed=_measurement_seed(config, 1),
    )
    if train_features.shape != (len(y_train), n_observables):
        raise ValueError("Train readout dimensions do not match the configured observables")
    if validation_features.shape != (len(y_val), n_observables):
        raise ValueError("Validation readout dimensions do not match the configured observables")
    train_probabilities = np.asarray([
        _readout_mlp_probs(features, W1, b1, W2, b2) for features in train_features
    ])
    validation_probabilities = np.asarray([
        _readout_mlp_probs(features, W1, b1, W2, b2) for features in validation_features
    ])
    train_metrics = _metrics(y_train, train_probabilities)
    validation_metrics = _metrics(y_val, validation_probabilities)
    return {
        "status": "completed",
        "model_family": "qnim",
        "fit_scope": "train_only",
        "metrics": {
            "training": train_metrics,
            "validation": validation_metrics,
            "accuracy_gap_train_minus_validation": train_metrics["accuracy"] - validation_metrics["accuracy"],
            "balanced_accuracy_gap_train_minus_validation": train_metrics["balanced_accuracy"] - validation_metrics["balanced_accuracy"],
        },
        "parameters": {
            "quantum_trainable_count": n_quantum_params,
            "classical_trainable_count": n_classical_params,
            "total_trainable_count": expected_weight_count,
            "quantum_initialization": initial_ansatz.tolist(),
            "quantum_initialization_sha256": _sha256_bytes(
                np.asarray(initial_ansatz, dtype=np.float64).tobytes()
            ),
            "quantum_final": ansatz_weights.tolist(),
            "quantum_final_sha256": _sha256_bytes(
                np.asarray(ansatz_weights, dtype=np.float64).tobytes()
            ),
            "classical_readout_final": weights[n_quantum_params:].tolist(),
            "classical_readout_final_sha256": _sha256_bytes(
                np.asarray(weights[n_quantum_params:], dtype=np.float64).tobytes()
            ),
        },
        "circuit": {
            "feature_map": "ZZFeatureMap",
            "feature_map_reps": FEATURE_MAP_REPS,
            "ansatz": "EfficientSU2",
            "ansatz_reps": ANSATZ_REPS,
            "entanglement": ENTANGLEMENT,
            "readout_mode": READOUT_MODE,
            "n_features_encoded": n_features,
            "n_observables": n_observables,
            "observable_order": "P0(0)..P0(q-1), then ZZ(0,1), ZZ(1,2), ..., ZZ(q-2,q-1)",
            "readout_hidden_size": READOUT_HIDDEN_SIZE,
        },
        "optimization": {
            "max_iterations": MAX_ITERATIONS,
            "batch_size": BATCH_SIZE,
            "iterations": int(training["iterations"]),
            "converged": bool(training["converged"]),
            "termination_reason": training["termination_reason"],
            "final_training_loss": float(training["training_loss"]),
            "loss_history": [float(value) for value in training["loss_history"]],
            "n_circuit_evaluations": int(training["n_circuit_evaluations"]),
            "gradient_variance_history": [
                float(value) for value in training["gradient_variance_history"]
            ],
            "optimizer_step_history": training.get("optimizer_step_history", []),
            "patience": PATIENCE,
            "patience_note": "patience equals max_iterations; it is not expected to substantially reduce the fixed training budget.",
        },
        "timing": {
            "trainer_elapsed_seconds": trainer_elapsed,
        },
        "measurement_seeds": {
            "training_metrics": _measurement_seed(config, 0),
            "validation_metrics": _measurement_seed(config, 1),
        },
    }


def _fit_svm(dataset: dict[str, np.ndarray], config: dict) -> dict:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    n_features = config["pca_components"]
    X_train = dataset["X_train"][:, :n_features]
    X_val = dataset["X_val"][:, :n_features]
    y_train, y_val = dataset["y_train"], dataset["y_val"]
    model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=config["initialization_seed"],
        ),
    )
    classifier_started = time.perf_counter()
    model.fit(X_train, y_train)
    validation_probabilities = model.predict_proba(X_val)
    classifier_elapsed = time.perf_counter() - classifier_started
    estimator = model[-1]
    if not np.array_equal(estimator.classes_, np.arange(N_CLASSES)):
        raise ValueError("PCA+SVM did not fit all 13 training classes")
    validation_metrics = _metrics(y_val, validation_probabilities)
    return {
        "status": "completed",
        "model_family": "pca_svm_rbf",
        "fit_scope": "train_only",
        "metrics": {
            "training": None,
            "validation": validation_metrics,
            "accuracy_gap_train_minus_validation": None,
            "balanced_accuracy_gap_train_minus_validation": None,
            "training_evaluation_note": "Not computed for the classical control; evaluation is validation-only.",
        },
        "classifier": {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "class_weight": "balanced",
            "probability": True,
            "random_state": config["initialization_seed"],
            "scaler": "StandardScaler fit on train only",
            "scaler_fit_sample_count": len(y_train),
            "scaler_mean": model[0].mean_.astype(float).tolist(),
            "support_vectors_per_class": estimator.n_support_.astype(int).tolist(),
            "support_vector_count": int(estimator.n_support_.sum()),
            "stored_numeric_parameter_count": int(
                estimator.support_vectors_.size
                + estimator.dual_coef_.size
                + estimator.intercept_.size
            ),
        },
        "timing": {
            "classifier_elapsed_seconds": classifier_elapsed,
        },
    }


def _fit_one(dataset: dict[str, np.ndarray], config: dict) -> dict:
    wall_started = time.perf_counter()
    monitor = PeakMemoryMonitor()
    try:
        with monitor:
            result = (
                _fit_qnim(dataset, config)
                if config["model_family"] == "qnim"
                else _fit_svm(dataset, config)
            )
        result["timing"]["wall_clock_elapsed_seconds"] = time.perf_counter() - wall_started
        result["memory"] = _record_memory(monitor)
        return result
    except KeyboardInterrupt:
        return {
            **config,
            "status": "interrupted",
            "timing": {"wall_clock_elapsed_seconds": time.perf_counter() - wall_started},
            "memory": _record_memory(monitor),
            "error": {"type": "KeyboardInterrupt", "message": "Run interrupted; completed runs remain checkpointed."},
        }
    except Exception as exc:
        return {
            **config,
            "status": "failed",
            "timing": {"wall_clock_elapsed_seconds": time.perf_counter() - wall_started},
            "memory": _record_memory(monitor),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }


def _new_results(
    plan: list[dict], protocol_hash: str, dataset_manifest_hash: str,
    dataset_info: dict, run_filter: dict | None,
) -> dict:
    runs = []
    for config in plan:
        record = {"run_key": run_key(config), **config, "status": "pending", "attempts": []}
        _refresh_record_hash(record)
        runs.append(record)
    return {
        "schema_version": 1,
        "experiment_id": "QNIM_SCALING_NEIGHBOR_PILOT_V1",
        "protocol_sha256": protocol_hash,
        "dataset_manifest_sha256": dataset_manifest_hash,
        "run_filter": run_filter,
        "dataset_integrity": dataset_info,
        "status": "prepared",
        "runs": runs,
    }


def _validate_results(
    results: dict, plan: list[dict], protocol_hash: str,
    dataset_manifest_hash: str, dataset_info: dict, run_filter: dict | None,
) -> None:
    if results.get("schema_version") != 1 or results.get("experiment_id") != "QNIM_SCALING_NEIGHBOR_PILOT_V1":
        raise ValueError("Existing results have an unsupported schema or experiment_id")
    if results.get("status") not in {"prepared", "running", "partial", "completed", "failed"}:
        raise ValueError("Existing results have an invalid execution status")
    if results.get("protocol_sha256") != protocol_hash:
        raise ValueError("Existing results use a different approved protocol")
    if results.get("dataset_manifest_sha256") != dataset_manifest_hash:
        raise ValueError("Existing results use a different immutable dataset manifest")
    if results.get("run_filter") != run_filter:
        raise ValueError("Existing results use a different configuration filter")
    if results.get("dataset_integrity") != dataset_info:
        raise ValueError("Existing results use different train/validation splits")
    expected = {run_key(config): config for config in plan}
    actual: set[str] = set()
    for row in results.get("runs", []):
        key = row.get("run_key")
        if key in actual:
            raise ValueError(f"Duplicate saved run configuration: {key}")
        actual.add(key)
        if key not in expected or any(
            row.get(name) != expected[key][name]
            for name in (
                "model_family", "arm", "n_qubits", "pca_components",
                "initialization_seed",
            )
        ):
            raise ValueError(f"Saved run configuration differs from the plan: {key}")
        if row.get("status") not in {
            "pending", "running", "interrupted", "completed", "failed"
        }:
            raise ValueError(f"Saved run has an invalid status: {key}")
        if row.get("record_sha256") != _record_hash(row):
            raise ValueError(f"Saved result hash is invalid for run {key}")
    if actual != set(expected):
        raise ValueError("Saved result configurations do not match the approved execution matrix")


def _save_state(output_dir: Path, results: dict, manifest: dict) -> None:
    results["status"] = manifest["status"]
    results_path = output_dir / "results.json"
    _atomic_json(results_path, results)
    manifest["results_file_sha256"] = _sha256_bytes(results_path.read_bytes())
    manifest["updated_at_utc"] = _utc_now()
    _atomic_json(output_dir / "execution_manifest.json", manifest)


def _init_state(
    output_dir: Path, plan: list[dict], protocol_content: bytes,
    protocol_hash: str, dataset_manifest_content: bytes,
    dataset_manifest_hash: str, dataset_info: dict, run_filter: dict | None,
) -> tuple[dict, dict]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Output directory is not empty; use --resume or preserve it")
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(output_dir / "protocol.json", protocol_content)
    _atomic_bytes(output_dir / "dataset_manifest.json", dataset_manifest_content)
    results = _new_results(
        plan, protocol_hash, dataset_manifest_hash, dataset_info, run_filter
    )
    manifest = {
        "schema_version": 1,
        "experiment_id": "QNIM_SCALING_NEIGHBOR_PILOT_V1",
        "status": "prepared",
        "protocol_sha256": protocol_hash,
        "dataset_manifest_sha256": dataset_manifest_hash,
        "dataset_manifest_file": "dataset_manifest.json",
        "dataset_integrity": dataset_info,
        "run_filter": run_filter,
        "active_run": None,
        "patience_note": "patience=20 equals max_iterations=20; do not treat it as a substantial runtime reduction.",
        "created_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
    }
    _save_state(output_dir, results, manifest)
    return results, manifest


def _resume_state(
    output_dir: Path, plan: list[dict], protocol_hash: str,
    dataset_manifest_hash: str, dataset_info: dict, run_filter: dict | None,
) -> tuple[dict, dict]:
    protocol_path = output_dir / "protocol.json"
    dataset_manifest_path = output_dir / "dataset_manifest.json"
    results_path = output_dir / "results.json"
    manifest_path = output_dir / "execution_manifest.json"
    if not all(path.is_file() for path in (
        protocol_path, dataset_manifest_path, results_path, manifest_path
    )):
        raise FileNotFoundError(
            "--resume requires protocol.json, dataset_manifest.json, results.json, and execution_manifest.json"
        )
    if _sha256_bytes(protocol_path.read_bytes()) != protocol_hash:
        raise ValueError("Saved protocol copy hash differs from the approved protocol")
    if _sha256_bytes(dataset_manifest_path.read_bytes()) != dataset_manifest_hash:
        raise ValueError("Saved dataset manifest hash differs from the approved manifest")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_results(
        results, plan, protocol_hash, dataset_manifest_hash, dataset_info, run_filter
    )
    if manifest.get("status") not in {"prepared", "running", "partial", "completed", "failed"}:
        raise ValueError("Execution manifest has an invalid status")
    if manifest.get("protocol_sha256") != protocol_hash:
        raise ValueError("Execution manifest protocol hash differs from the approved protocol")
    if manifest.get("dataset_manifest_sha256") != dataset_manifest_hash:
        raise ValueError("Execution manifest dataset manifest hash differs")
    if manifest.get("run_filter") != run_filter:
        raise ValueError("Execution manifest has a different configuration filter")
    if manifest.get("dataset_integrity") != dataset_info:
        raise ValueError("Execution manifest train/validation hashes differ")

    current_results_hash = _sha256_bytes(results_path.read_bytes())
    saved_results_hash = manifest.get("results_file_sha256")
    if saved_results_hash != current_results_hash:
        manifest["integrity_reconciliation"] = {
            "previous_manifest_results_sha256": saved_results_hash,
            "validated_results_sha256": current_results_hash,
            "reason": "The results checkpoint was atomically committed before an interruption updating the manifest; all per-run hashes validated.",
        }
    for row in results["runs"]:
        if row["status"] == "running":
            if not row["attempts"]:
                raise ValueError(f"Running checkpoint has no attempt record: {row['run_key']}")
            row["status"] = "interrupted"
            row["attempts"][-1].update({
                "status": "interrupted",
                "ended_at_utc": _utc_now(),
                "note": "The previous process ended before committing this run.",
            })
            _refresh_record_hash(row)
    manifest["status"] = "partial" if any(
        row["status"] != "completed" for row in results["runs"]
    ) else "completed"
    manifest["active_run"] = None
    return results, manifest


def _update_run(results: dict, row: dict, fit_result: dict, attempt: dict) -> None:
    attempts = row["attempts"]
    key = row["run_key"]
    config = {name: row[name] for name in (
        "model_family", "arm", "n_qubits", "pca_components", "initialization_seed"
    )}
    row.clear()
    row.update({"run_key": key, **config, **fit_result, "attempts": attempts})
    attempts[-1].update({
        "status": fit_result["status"],
        "ended_at_utc": _utc_now(),
        "timing": fit_result.get("timing"),
        "memory": fit_result.get("memory"),
    })
    _refresh_record_hash(row)


def _print_dry_run(plan: list[dict], dataset_info: dict, max_runs: int | None) -> None:
    selected_qnim = sum(row["model_family"] == "qnim" for row in plan)
    selected_svm = sum(row["model_family"] == "pca_svm_rbf" for row in plan)
    print(json.dumps({
        "status": "dry_run",
        "experiment_id": "QNIM_SCALING_NEIGHBOR_PILOT_V1",
        "execution_matrix": [
            {"run_key": run_key(row), **row} for row in plan
        ],
        "expected_fits": {
            "selected": {
                "qnim": selected_qnim,
                "pca_svm_rbf": selected_svm,
                "total": len(plan),
            },
            "complete_pilot": {"qnim": 15, "pca_svm_rbf": 9, "total": 24},
        },
        "resource_configuration": {
            "max_iterations": MAX_ITERATIONS,
            "batch_size": BATCH_SIZE,
            "shots": SHOTS,
            "learning_rate": LEARNING_RATE,
            "training_random_seed": TRAINING_RANDOM_SEED,
            "initialization_seeds": list(INITIALIZATION_SEEDS),
            "patience": PATIENCE,
            "patience_note": "patience=20 equals max_iterations=20; it is not expected to substantially reduce runtime.",
            "readout_mode": READOUT_MODE,
            "readout_features": "2*q-1",
        },
        "dataset_integrity": dataset_info,
        "max_runs": max_runs,
        "training_started": False,
    }, indent=2))


def run_pilot(
    *,
    dry_run: bool = False,
    max_runs: int | None = None,
    resume: bool = False,
    arm: str | None = None,
    n_qubits: int | None = None,
    seed: int | None = None,
    protocol_path: Path = PROTOCOL_PATH,
    dataset_manifest_path: Path = DATASET_MANIFEST_PATH,
    dataset_dir: Path = DATASET_DIR,
    output_dir: Path | None = None,
) -> int:
    if max_runs is not None and max_runs <= 0:
        raise ValueError("--max-runs must be a positive integer")
    _, protocol_content, protocol_hash = _verify_protocol(protocol_path)
    dataset_manifest, dataset_manifest_content, dataset_manifest_hash = (
        _read_dataset_manifest(dataset_manifest_path, protocol_hash)
    )
    run_filter = (
        None if arm is None else {
            "arm": arm, "n_qubits": n_qubits, "initialization_seed": seed,
        }
    )
    plan = build_execution_plan(arm=arm, n_qubits=n_qubits, seed=seed)
    if output_dir is None:
        output_dir = SMOKE_OUTPUT_DIR if run_filter is not None else OUTPUT_DIR
    elif not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    dataset, dataset_info = load_cached_dataset(
        dataset_dir, dataset_manifest, dataset_manifest_hash
    )
    if dry_run:
        _print_dry_run(plan, dataset_info, max_runs)
        return 0

    if resume:
        results, manifest = _resume_state(
            output_dir, plan, protocol_hash, dataset_manifest_hash,
            dataset_info, run_filter,
        )
        _save_state(output_dir, results, manifest)
    else:
        results, manifest = _init_state(
            output_dir, plan, protocol_content, protocol_hash,
            dataset_manifest_content, dataset_manifest_hash,
            dataset_info, run_filter,
        )

    row_by_key = {row["run_key"]: row for row in results["runs"]}
    attempted = 0
    for config in plan:
        row = row_by_key[run_key(config)]
        if row["status"] == "completed":
            continue
        if max_runs is not None and attempted >= max_runs:
            break
        attempt = {"started_at_utc": _utc_now(), "status": "running"}
        row["status"] = "running"
        row["attempts"].append(attempt)
        _refresh_record_hash(row)
        manifest["status"] = "running"
        manifest["active_run"] = row["run_key"]
        _save_state(output_dir, results, manifest)

        fit_result = _fit_one(dataset, config)
        _update_run(results, row, fit_result, attempt)
        attempted += 1
        manifest["active_run"] = None
        if fit_result["status"] == "interrupted":
            manifest["status"] = "partial"
            _save_state(output_dir, results, manifest)
            return 130
        if fit_result["status"] == "failed":
            manifest["status"] = "failed"
            _save_state(output_dir, results, manifest)
            return 1
        manifest["status"] = "running"
        _save_state(output_dir, results, manifest)

    remaining = [row for row in results["runs"] if row["status"] != "completed"]
    manifest["status"] = "partial" if remaining else "completed"
    manifest["active_run"] = None
    _save_state(output_dir, results, manifest)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", help="Validate integrity and print the plan without training.")
    group.add_argument("--resume", action="store_true", help="Validate checkpoints and continue incomplete runs.")
    parser.add_argument("--max-runs", type=int, default=None, help="Maximum new fit attempts in this invocation.")
    parser.add_argument("--arm", choices=("full_system", "fixed_input_dimension"))
    parser.add_argument("--qubits", type=int)
    parser.add_argument("--seed", type=int, help="Initialization seed (42, 43, or 44).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to the full-pilot directory or an isolated smoke directory for a filtered run.",
    )
    args = parser.parse_args(argv)
    return run_pilot(
        dry_run=args.dry_run,
        max_runs=args.max_runs,
        resume=args.resume,
        arm=args.arm,
        n_qubits=args.qubits,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
