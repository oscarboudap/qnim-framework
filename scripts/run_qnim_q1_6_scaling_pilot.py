"""Run the validation-only Q1.6 controlled qubit-scaling pilot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import psutil
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DEFAULT_OUTPUT = (
    ROOT / "results" / "scaling_study" / "qnim_q1_6_scaling_pilot_v1.json"
)
DEFAULT_DATASET_DIR = (
    ROOT / "results" / "scaling_study" / "qnim_q1_6_dataset_v1"
)
Q1_2_REPORT = (
    ROOT
    / "results"
    / "scaling_study"
    / "qnim_q1_2_q1_3_parameters_v1.json"
)
Q1_4_REPORT = ROOT / "results" / "scaling_study" / "qnim_q1_4_v1.json"
CLASSICAL_MANIFEST = (
    ROOT / "results" / "scaling_study" / "classical_reference_v1.json"
)
QUIBIT_COUNTS = (4, 8, 12, 16)
MODEL_SEEDS = (42, 43, 44)
DATASET_SEED = 42
FIXED_INPUT_DIMENSION = 4
N_CLASSES = 13
FEATURE_MAP_REPS = 2
ANSATZ_REPS = 2
ENTANGLEMENT = "linear"
READOUT_MODE = "marginals_zz"
READOUT_HIDDEN_SIZE = 16
TRAINING_SEED = 42
TRAINING_BATCH_SIZE = 64
MAX_ITERATIONS = 60
SHOTS = 512
PATIENCE = 20
LEARNING_RATE = 0.03
SCHEMA_VERSION = 1
CLASSICAL_METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "cross_entropy",
)


class PeakMemoryMonitor:
    """Sample process RSS during one run to estimate its peak memory use."""

    def __init__(self, interval_seconds: float = 0.1):
        self.interval_seconds = interval_seconds
        self.process = psutil.Process()
        self.initial_bytes = 0
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self._stop.is_set():
            self.peak_bytes = max(
                self.peak_bytes, self.process.memory_info().rss
            )
            self._stop.wait(self.interval_seconds)

    def __enter__(self):
        self.initial_bytes = self.process.memory_info().rss
        self.peak_bytes = self.initial_bytes
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        self._thread.join()
        self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _split_sha256(features: np.ndarray, labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (features, labels):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, indent=2, allow_nan=False)
            temporary.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _read_frozen_sources() -> dict:
    q1_2_bytes = Q1_2_REPORT.read_bytes()
    q1_4_bytes = Q1_4_REPORT.read_bytes()
    q1_2 = json.loads(q1_2_bytes)
    q1_4 = json.loads(q1_4_bytes)
    if q1_2.get("status") != "completed" or q1_4.get("status") != "completed":
        raise ValueError("Q1.6 requires completed Q1.2 and Q1.4 reports")
    if q1_2.get("experiment_id") != "QNIM_Q1_2_V1":
        raise ValueError("The Q1.2 report has an unexpected experiment id")
    if q1_4.get("experiment_id") != "QNIM_Q1_4_V1":
        raise ValueError("The Q1.4 report has an unexpected experiment id")
    if _sha256_bytes(q1_2_bytes) != q1_4.get("q1_2_report_sha256"):
        raise ValueError("Q1.4 does not reference the supplied Q1.2 report")
    for name, report in (("Q1.2", q1_2), ("Q1.4", q1_4)):
        final_test = report.get("final_test", {})
        if final_test.get("evaluated") is not False:
            raise ValueError(f"{name} final test is not locked")
        if final_test.get("status") != "locked_not_evaluated":
            raise ValueError(f"{name} final test is not locked")
    if q1_2.get("dataset_hashes") != q1_4.get("dataset_hashes"):
        raise ValueError("Q1.2 and Q1.4 do not share the same train/validation hashes")
    if q1_2.get("model_initialization_seeds") != list(MODEL_SEEDS):
        raise ValueError("Q1.6 requires Q1 initialization seeds 42, 43, and 44")

    manifest = json.loads(CLASSICAL_MANIFEST.read_text(encoding="utf-8"))
    source_path = CLASSICAL_MANIFEST.parent / manifest["source_report"]
    source_bytes = source_path.read_bytes()
    source_hash = _sha256_bytes(source_bytes)
    if source_hash != manifest.get("source_sha256"):
        raise ValueError("The frozen classical reference report hash is invalid")
    classical = json.loads(source_bytes)
    config = classical.get("dataset_configuration", {})
    if (
        classical.get("strict_physics") is not True
        or classical.get("fallback_allowed") is not False
        or classical.get("waveform_source") != "PyCBC"
        or classical.get("approximant") != "IMRPhenomPv2"
        or classical.get("n_classes") != N_CLASSES
        or config.get("train_per_class") != 200
        or config.get("validation_per_class") != 50
        or config.get("test_per_class") != 100
        or config.get("pca_fit_components") != 64
    ):
        raise ValueError("The frozen PyCBC dataset protocol differs from Q1")

    return {
        "q1_2": q1_2,
        "q1_4": q1_4,
        "dataset_configuration": config,
        "approximant": classical["approximant"],
        "class_names": q1_2["class_names"],
        "source_hashes": {
            "q1_2_report_sha256": _sha256_bytes(q1_2_bytes),
            "q1_4_report_sha256": _sha256_bytes(q1_4_bytes),
            "classical_reference_report_sha256": source_hash,
        },
    }


def _generate_train_validation(
    adapter,
    seed: int,
    train_per_class: int,
    validation_per_class: int,
    discarded_draws_per_class: int,
    n_components: int,
    n_pca_fit: int,
) -> dict:
    if adapter.strict_physics is not True:
        raise ValueError("Q1.6 requires strict PyCBC generation")
    if not 1 <= n_components <= n_pca_fit:
        raise ValueError("Invalid PCA dimensions")

    rng = np.random.default_rng(seed)
    train_features = []
    train_labels = []
    validation_features = []
    validation_labels = []
    total_per_class = (
        train_per_class + validation_per_class + discarded_draws_per_class
    )
    for class_index in range(adapter.n_classes):
        print(
            f"Generating train/validation waveforms for class "
            f"{class_index}: {adapter.theory_names[class_index]}"
        )
        for event_index in range(total_per_class):
            waveform = adapter._generate_one(class_index, rng)
            if event_index < train_per_class:
                train_features.append(waveform)
                train_labels.append(class_index)
            elif event_index < train_per_class + validation_per_class:
                validation_features.append(waveform)
                validation_labels.append(class_index)

    X_train_raw = np.stack(train_features)
    X_val_raw = np.stack(validation_features)
    y_train = np.asarray(train_labels, dtype=int)
    y_val = np.asarray(validation_labels, dtype=int)
    n_fit = min(n_pca_fit, X_train_raw.shape[0] - 1, X_train_raw.shape[1])
    pca = PCA(n_components=n_fit, random_state=seed)
    pca.fit(X_train_raw)
    X_train_projected = pca.transform(X_train_raw)[:, :n_components]
    X_val_projected = pca.transform(X_val_raw)[:, :n_components]
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    X_train = np.clip(scaler.fit_transform(X_train_projected), -1.0, 1.0)
    X_val = np.clip(scaler.transform(X_val_projected), -1.0, 1.0)
    if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
        raise ValueError("Generated train/validation PCA features are non-finite")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "class_names": list(adapter.theory_names),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_[
            :n_components
        ].tolist(),
        "generation": {
            "dataset_seed": seed,
            "train_per_class": train_per_class,
            "validation_per_class": validation_per_class,
            "discarded_post_validation_draws_per_class": discarded_draws_per_class,
            "n_components": n_components,
            "n_pca_fit": n_fit,
            "pca_fit_split": "train_only",
            "strict_physics": True,
            "waveform_source": "PyCBC",
        },
    }


def _generate_dataset(sources: dict) -> dict:
    from src.infrastructure.ligo_pca_adapter import LIGOPCAAdapter

    adapter = LIGOPCAAdapter(
        n_components=max(QUIBIT_COUNTS),
        n_pca_fit=sources["dataset_configuration"]["pca_fit_components"],
        approximant=sources["approximant"],
        max_classes=N_CLASSES,
        strict_physics=True,
    )
    config = sources["dataset_configuration"]
    dataset = _generate_train_validation(
        adapter=adapter,
        seed=DATASET_SEED,
        train_per_class=config["train_per_class"],
        validation_per_class=config["validation_per_class"],
        discarded_draws_per_class=config["test_per_class"],
        n_components=max(QUIBIT_COUNTS),
        n_pca_fit=config["pca_fit_components"],
    )
    if dataset["class_names"] != sources["class_names"]:
        raise ValueError("Generated class order differs from frozen Q1")

    expected_hashes = sources["q1_2"]["dataset_hashes"]
    actual_prefix_hashes = {
        "train_sha256": _split_sha256(
            dataset["X_train"][:, :12], dataset["y_train"]
        ),
        "validation_sha256": _split_sha256(
            dataset["X_val"][:, :12], dataset["y_val"]
        ),
    }
    if actual_prefix_hashes != expected_hashes:
        raise ValueError(
            "The 12-component prefix does not reproduce the frozen Q1 "
            "train/validation split hashes"
        )
    return dataset


def _save_dataset(dataset_dir: Path, dataset: dict) -> dict:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for name in ("X_train", "y_train", "X_val", "y_val"):
        _atomic_npy(dataset_dir / f"{name}.npy", dataset[name])
    metadata = {
        "class_names": dataset["class_names"],
        "generation": dataset["generation"],
        "pca_explained_variance_ratio": dataset[
            "pca_explained_variance_ratio"
        ],
        "dataset_hashes": {
            "train_sha256": _split_sha256(
                dataset["X_train"], dataset["y_train"]
            ),
            "validation_sha256": _split_sha256(
                dataset["X_val"], dataset["y_val"]
            ),
        },
        "final_test": {
            "opened": False,
            "projected": False,
            "persisted": False,
            "evaluated": False,
            "note": (
                "Post-validation waveform draws were discarded to preserve "
                "the historical generator RNG stream."
            ),
        },
    }
    _atomic_json(dataset_dir / "dataset_metadata.json", metadata)
    return metadata


def _load_dataset(dataset_dir: Path, expected_class_names: list) -> dict:
    arrays = {}
    for name in ("X_train", "y_train", "X_val", "y_val"):
        path = dataset_dir / f"{name}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Missing Q1.6 train/validation file: {path}")
        arrays[name] = np.load(path, allow_pickle=False)
    metadata_path = dataset_dir / "dataset_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing Q1.6 dataset metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("class_names") != expected_class_names:
        raise ValueError("Q1.6 dataset class order differs from the frozen reference")
    if metadata.get("final_test", {}).get("evaluated") is not False:
        raise ValueError("Q1.6 dataset metadata indicates final-test evaluation")

    X_train, y_train = arrays["X_train"], arrays["y_train"]
    X_val, y_val = arrays["X_val"], arrays["y_val"]
    if (
        X_train.shape != (2600, 16)
        or X_val.shape != (650, 16)
        or y_train.shape != (2600,)
        or y_val.shape != (650,)
    ):
        raise ValueError("Unexpected Q1.6 train/validation dimensions")
    if not np.array_equal(np.bincount(y_train), np.full(N_CLASSES, 200)):
        raise ValueError("Q1.6 train split is not balanced as specified")
    if not np.array_equal(np.bincount(y_val), np.full(N_CLASSES, 50)):
        raise ValueError("Q1.6 validation split is not balanced as specified")
    if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
        raise ValueError("Q1.6 dataset contains non-finite values")
    hashes = {
        "train_sha256": _split_sha256(X_train, y_train),
        "validation_sha256": _split_sha256(X_val, y_val),
    }
    if hashes != metadata.get("dataset_hashes"):
        raise ValueError("Q1.6 train/validation hashes do not match dataset metadata")
    return {**arrays, "class_names": expected_class_names, "metadata": metadata}


def _make_svm(seed: int):
    return make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=seed,
        ),
    )


def _classification_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, class_names: list[str]
) -> dict:
    labels = np.arange(len(class_names))
    predictions = np.argmax(probabilities, axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "cross_entropy": float(
            log_loss(
                y_true,
                np.clip(probabilities, 1e-12, 1.0),
                labels=labels,
            )
        ),
        "class_recall": {
            name: float(value)
            for name, value in zip(
                class_names,
                recall_score(
                    y_true,
                    predictions,
                    labels=labels,
                    average=None,
                    zero_division=0,
                ),
            )
        },
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=labels
        ).tolist(),
    }


def _prediction_sha256(predictions: np.ndarray) -> str:
    return _sha256_bytes(
        np.asarray(predictions, dtype=np.int64).tobytes()
    )


def _run_svm(
    dataset: dict,
    n_features: int,
    seed: int,
    n_qubits: int,
    class_names: list[str],
) -> dict:
    X_train = dataset["X_train"][:, :n_features]
    X_val = dataset["X_val"][:, :n_features]
    y_train, y_val = dataset["y_train"], dataset["y_val"]
    model = _make_svm(seed)
    estimator = model[-1]
    started = time.perf_counter()
    with PeakMemoryMonitor() as memory:
        model.fit(X_train, y_train)
        train_probabilities = model.predict_proba(X_train)
        validation_probabilities = model.predict_proba(X_val)
    elapsed = time.perf_counter() - started
    if not np.array_equal(estimator.classes_, np.arange(N_CLASSES)):
        raise ValueError("The PCA+SVM control did not fit all 13 classes")
    train_metrics = _classification_metrics(
        y_train, train_probabilities, class_names
    )
    validation_metrics = _classification_metrics(
        y_val, validation_probabilities, class_names
    )
    support_count = int(estimator.n_support_.sum())
    return {
        "status": "completed",
        "model_family": "pca_svm_rbf",
        "arm": "full_system",
        "n_qubits": n_qubits,
        "n_features": n_features,
        "initialization_seed": seed,
        "fit_scope": "train_only",
        "metrics": {
            "training": train_metrics,
            "validation": validation_metrics,
            "accuracy_gap_train_minus_validation": (
                train_metrics["accuracy"] - validation_metrics["accuracy"]
            ),
            "balanced_accuracy_gap_train_minus_validation": (
                train_metrics["balanced_accuracy"]
                - validation_metrics["balanced_accuracy"]
            ),
        },
        "classifier": {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "class_weight": "balanced",
            "probability": True,
            "random_state": seed,
            "scaler": "StandardScaler, fit on train only",
            "support_vectors_per_class": estimator.n_support_.astype(int).tolist(),
            "support_vector_count": support_count,
            "dual_coefficient_count": int(estimator.dual_coef_.size),
            "intercept_count": int(estimator.intercept_.size),
            "stored_numeric_parameter_count": int(
                estimator.support_vectors_.size
                + estimator.dual_coef_.size
                + estimator.intercept_.size
            ),
        },
        "timing": {
            "fit_and_evaluation_seconds": elapsed,
        },
        "memory": {
            "rss_start_bytes": memory.initial_bytes,
            "rss_peak_bytes": memory.peak_bytes,
            "peak_rss_delta_bytes": max(
                0, memory.peak_bytes - memory.initial_bytes
            ),
            "sampling_interval_seconds": memory.interval_seconds,
        },
        "validation_prediction_sha256": _prediction_sha256(
            np.argmax(validation_probabilities, axis=1)
        ),
        "validation_predictions": np.argmax(
            validation_probabilities, axis=1
        ).astype(int).tolist(),
    }


def _derived_measurement_seed(seed: int, n_qubits: int, arm: str, split: int) -> int:
    arm_number = 0 if arm == "full_system" else 1
    return int(
        np.random.SeedSequence(
            [DATASET_SEED, seed, n_qubits, arm_number, split, 1616]
        ).generate_state(1)[0]
    )


def _run_qnim(
    dataset: dict,
    arm: str,
    n_qubits: int,
    n_features: int,
    seed: int,
    class_names: list[str],
) -> dict:
    from src.infrastructure.qiskit_vqc_trainer import (
        QiskitVQCTrainer,
        _build_feature_map_and_ansatz,
        _make_vqc_loss_fn,
        _n_readout_params,
        _readout_feature_count,
        _readout_mlp_probs,
        _split_readout_mlp,
    )

    X_train = dataset["X_train"][:, :n_features]
    X_val = dataset["X_val"][:, :n_features]
    y_train, y_val = dataset["y_train"], dataset["y_val"]
    circuit, feature_parameters, ansatz_parameters = (
        _build_feature_map_and_ansatz(
            n_qubits,
            n_features=n_features,
            feature_map_reps=FEATURE_MAP_REPS,
            ansatz_reps=ANSATZ_REPS,
            entanglement=ENTANGLEMENT,
        )
    )
    if circuit.num_qubits != n_qubits or len(feature_parameters) != n_features:
        raise ValueError("Constructed QNIM circuit dimensions do not match its config")
    n_ansatz_parameters = len(ansatz_parameters)
    n_observables = _readout_feature_count(n_qubits, READOUT_MODE)
    n_classical_parameters = _n_readout_params(
        n_observables, N_CLASSES, READOUT_HIDDEN_SIZE
    )
    trainer = QiskitVQCTrainer(
        mode="sim",
        feature_map_reps=FEATURE_MAP_REPS,
        ansatz_reps=ANSATZ_REPS,
        entanglement=ENTANGLEMENT,
        patience=PATIENCE,
        learning_rate=LEARNING_RATE,
        training_batch_size=TRAINING_BATCH_SIZE,
        random_seed=TRAINING_SEED,
        readout_hidden_size=READOUT_HIDDEN_SIZE,
        readout_mode=READOUT_MODE,
    )
    started = time.perf_counter()
    with PeakMemoryMonitor() as memory:
        training = trainer.train_vqc(
            X_train=X_train,
            y_train=y_train,
            num_qubits=n_qubits,
            n_features=n_features,
            max_iterations=MAX_ITERATIONS,
            shots=SHOTS,
            initialization_seed=seed,
        )
        weights = np.asarray(training["weights"], dtype=float)
        expected_parameter_count = n_ansatz_parameters + n_classical_parameters
        if weights.shape != (expected_parameter_count,):
            raise ValueError(
                "Trained parameter vector does not match the circuit/readout dimensions"
            )
        loss_fn = _make_vqc_loss_fn(
            X_train=X_train,
            y_train=y_train,
            n_qubits=n_qubits,
            n_features=n_features,
            shots=SHOTS,
            mode="sim",
            batch_size_local=TRAINING_BATCH_SIZE,
            feature_map_reps=FEATURE_MAP_REPS,
            ansatz_reps=ANSATZ_REPS,
            entanglement=ENTANGLEMENT,
            readout_hidden_size=READOUT_HIDDEN_SIZE,
            random_seed=TRAINING_SEED,
            readout_mode=READOUT_MODE,
        )
        ansatz_weights = weights[:n_ansatz_parameters]
        W1, b1, W2, b2 = _split_readout_mlp(
            weights,
            n_ansatz_parameters,
            n_observables,
            N_CLASSES,
            READOUT_HIDDEN_SIZE,
        )
        train_features = loss_fn.measure_quantum_features(
            X_train,
            ansatz_weights,
            shots_override=SHOTS,
            simulator_seed=_derived_measurement_seed(seed, n_qubits, arm, 0),
        )
        validation_features = loss_fn.measure_quantum_features(
            X_val,
            ansatz_weights,
            shots_override=SHOTS,
            simulator_seed=_derived_measurement_seed(seed, n_qubits, arm, 1),
        )
        train_probabilities = np.asarray([
            _readout_mlp_probs(features, W1, b1, W2, b2)
            for features in train_features
        ])
        validation_probabilities = np.asarray([
            _readout_mlp_probs(features, W1, b1, W2, b2)
            for features in validation_features
        ])
    elapsed = time.perf_counter() - started
    train_metrics = _classification_metrics(
        y_train, train_probabilities, class_names
    )
    validation_metrics = _classification_metrics(
        y_val, validation_probabilities, class_names
    )
    initial_ansatz = np.random.default_rng(seed).normal(
        0.0, 0.01, n_ansatz_parameters
    )
    predictions = np.argmax(validation_probabilities, axis=1)
    return {
        "status": "completed",
        "model_family": "qnim",
        "arm": arm,
        "n_qubits": n_qubits,
        "n_features": n_features,
        "readout_mode": READOUT_MODE,
        "initialization_seed": seed,
        "training_random_seed": TRAINING_SEED,
        "fit_scope": "train_only",
        "metrics": {
            "training": train_metrics,
            "validation": validation_metrics,
            "accuracy_gap_train_minus_validation": (
                train_metrics["accuracy"] - validation_metrics["accuracy"]
            ),
            "balanced_accuracy_gap_train_minus_validation": (
                train_metrics["balanced_accuracy"]
                - validation_metrics["balanced_accuracy"]
            ),
        },
        "parameters": {
            "quantum_trainable_count": n_ansatz_parameters,
            "classical_trainable_count": n_classical_parameters,
            "total_trainable_count": expected_parameter_count,
            "quantum_initialization": initial_ansatz.tolist(),
            "quantum_initialization_sha256": _sha256_bytes(
                np.asarray(initial_ansatz, dtype=np.float64).tobytes()
            ),
            "quantum_final": ansatz_weights.tolist(),
            "quantum_final_sha256": _sha256_bytes(
                np.asarray(ansatz_weights, dtype=np.float64).tobytes()
            ),
            "classical_readout_final": weights[
                n_ansatz_parameters:
            ].tolist(),
            "classical_readout_final_sha256": _sha256_bytes(
                np.asarray(
                    weights[n_ansatz_parameters:], dtype=np.float64
                ).tobytes()
            ),
            "readout_warm_start": (
                "MLP fit on a fixed train-only subset of up to 200 examples"
            ),
        },
        "circuit": {
            "feature_map": "ZZFeatureMap",
            "feature_map_reps": FEATURE_MAP_REPS,
            "ansatz": "EfficientSU2",
            "ansatz_reps": ANSATZ_REPS,
            "entanglement": ENTANGLEMENT,
            "n_features_encoded": n_features,
            "n_observables": n_observables,
            "n_zz_correlators": n_qubits * (n_qubits - 1) // 2,
            "readout_hidden_size": READOUT_HIDDEN_SIZE,
            "shots": SHOTS,
            "n_feynman_parameters": min(4, n_qubits),
        },
        "optimization": {
            "max_iterations": MAX_ITERATIONS,
            "iterations": int(training["iterations"]),
            "converged": bool(training["converged"]),
            "termination_reason": training["termination_reason"],
            "final_training_loss": float(training["training_loss"]),
            "loss_history": [
                float(value) for value in training["loss_history"]
            ],
            "n_circuit_evaluations": int(
                training["n_circuit_evaluations"]
            ),
            "gradient_variance_history": [
                float(value)
                for value in training["gradient_variance_history"]
            ],
            "optimizer_step_history": training.get(
                "optimizer_step_history", []
            ),
        },
        "timing": {
            "trainer_seconds": float(training["execution_time_seconds"]),
            "training_and_evaluation_seconds": elapsed,
        },
        "memory": {
            "rss_start_bytes": memory.initial_bytes,
            "rss_peak_bytes": memory.peak_bytes,
            "peak_rss_delta_bytes": max(
                0, memory.peak_bytes - memory.initial_bytes
            ),
            "sampling_interval_seconds": memory.interval_seconds,
        },
        "measurement_seeds": {
            "training_metrics": _derived_measurement_seed(
                seed, n_qubits, arm, 0
            ),
            "validation_metrics": _derived_measurement_seed(
                seed, n_qubits, arm, 1
            ),
        },
        "validation_prediction_sha256": _prediction_sha256(predictions),
        "validation_predictions": predictions.astype(int).tolist(),
    }


def _record_key(record: dict) -> tuple:
    return (
        record.get("model_family"),
        record.get("arm"),
        record.get("n_qubits"),
        record.get("initialization_seed"),
    )


def _expected_keys() -> set[tuple]:
    keys = set()
    for seed in MODEL_SEEDS:
        for n_qubits in QUIBIT_COUNTS:
            keys.add(("pca_svm_rbf", "full_system", n_qubits, seed))
            keys.add(("qnim", "full_system", n_qubits, seed))
            if n_qubits > FIXED_INPUT_DIMENSION:
                keys.add(("qnim", "fixed_input_dimension", n_qubits, seed))
    return keys


def _aggregate(records: list[dict]) -> dict:
    completed = [
        record for record in records if record.get("status") == "completed"
    ]
    summary = {}
    arms = (
        ("qnim", "full_system"),
        ("qnim", "fixed_input_dimension"),
        ("pca_svm_rbf", "full_system"),
    )
    for model_family, arm in arms:
        per_qubit = {}
        for n_qubits in QUIBIT_COUNTS:
            rows = [
                row
                for row in completed
                if row["model_family"] == model_family
                and row["arm"] == arm
                and row["n_qubits"] == n_qubits
            ]
            if arm == "fixed_input_dimension" and n_qubits == 4 and not rows:
                rows = [
                    row
                    for row in completed
                    if row["model_family"] == "qnim"
                    and row["arm"] == "full_system"
                    and row["n_qubits"] == 4
                ]
            if not rows:
                continue
            metrics = {}
            for metric in CLASSICAL_METRIC_NAMES:
                values = [
                    row["metrics"]["validation"][metric] for row in rows
                ]
                metrics[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "per_seed": values,
                }
            per_qubit[str(n_qubits)] = {
                "n_initializations": len(rows),
                "metrics": metrics,
                "mean_training_seconds": float(np.mean([
                    row["timing"].get(
                        "training_and_evaluation_seconds",
                        row["timing"].get("fit_and_evaluation_seconds", 0.0),
                    )
                    for row in rows
                ])),
                "mean_peak_rss_bytes": float(np.mean([
                    row["memory"]["rss_peak_bytes"] for row in rows
                ])),
                "quantum_trainable_parameters": rows[0].get(
                    "parameters", {}
                ).get("quantum_trainable_count"),
                "classical_trainable_parameters": rows[0].get(
                    "parameters", {}
                ).get("classical_trainable_count"),
                "svm_stored_numeric_parameter_count": rows[0].get(
                    "classifier", {}
                ).get("stored_numeric_parameter_count"),
                "svm_support_vector_count_mean": (
                    float(np.mean([
                        row["classifier"]["support_vector_count"]
                        for row in rows
                    ]))
                    if model_family == "pca_svm_rbf"
                    else None
                ),
                "svm_stored_parameter_count_mean": (
                    float(np.mean([
                        row["classifier"]["stored_numeric_parameter_count"]
                        for row in rows
                    ]))
                    if model_family == "pca_svm_rbf"
                    else None
                ),
                "n_observables": rows[0].get("circuit", {}).get(
                    "n_observables"
                ),
            }
        summary[f"{model_family}:{arm}"] = per_qubit

    full_rows = summary.get("qnim:full_system", {})
    fixed_rows = summary.get("qnim:fixed_input_dimension", {})
    svm_rows = summary.get("pca_svm_rbf:full_system", {})
    deltas = {"qnim_full_minus_fixed": {}, "qnim_full_minus_svm": {}}
    for n_qubits in QUIBIT_COUNTS:
        key = str(n_qubits)
        if key in full_rows and key in fixed_rows:
            deltas["qnim_full_minus_fixed"][key] = {
                "balanced_accuracy_per_seed": [
                    full_rows[key]["metrics"]["balanced_accuracy"]["per_seed"][i]
                    - fixed_rows[key]["metrics"]["balanced_accuracy"]["per_seed"][i]
                    for i in range(min(
                        len(full_rows[key]["metrics"]["balanced_accuracy"]["per_seed"]),
                        len(fixed_rows[key]["metrics"]["balanced_accuracy"]["per_seed"]),
                    ))
                ],
                "note": (
                    "At 4 qubits both arms are identical and share one fit."
                    if n_qubits == FIXED_INPUT_DIMENSION
                    else None
                ),
            }
        if key in full_rows and key in svm_rows:
            qnim_values = full_rows[key]["metrics"]["balanced_accuracy"]["per_seed"]
            svm_values = svm_rows[key]["metrics"]["balanced_accuracy"]["per_seed"]
            deltas["qnim_full_minus_svm"][key] = [
                qnim_values[i] - svm_values[i]
                for i in range(min(len(qnim_values), len(svm_values)))
            ]

    slopes = {}
    for name in (
        "qnim:full_system",
        "qnim:fixed_input_dimension",
        "pca_svm_rbf:full_system",
    ):
        rows = summary.get(name, {})
        usable_q = [
            q for q in QUIBIT_COUNTS
            if str(q) in rows
            and "balanced_accuracy" in rows[str(q)]["metrics"]
        ]
        if len(usable_q) >= 2:
            vals = [
                rows[str(q)]["metrics"]["balanced_accuracy"]["mean"]
                for q in usable_q
            ]
            accs = [
                rows[str(q)]["metrics"]["accuracy"]["mean"]
                for q in usable_q
            ]
            slopes[name] = {
                "balanced_accuracy_per_qubit": float(
                    np.polyfit(usable_q, vals, 1)[0]
                ),
                "accuracy_per_qubit": float(
                    np.polyfit(usable_q, accs, 1)[0]
                ),
                "fit_qubit_counts": usable_q,
                "interpretation": "descriptive_pilot_only",
            }
    return {
        "by_model_and_arm": summary,
        "paired_seed_differences": deltas,
        "descriptive_linear_slopes": slopes,
        "statistical_caution": (
            "The three repetitions vary model initialization only and share "
            "one dataset seed; they are not independent dataset replicates."
        ),
    }


def _configuration() -> dict:
    return {
        "qubit_counts": list(QUIBIT_COUNTS),
        "arms": {
            "full_system": {
                "n_features": "n_qubits",
                "pca_components": "first n_qubits components",
                "readout": READOUT_MODE,
            },
            "fixed_input_dimension": {
                "n_features": FIXED_INPUT_DIMENSION,
                "pca_components": "first 4 components for every n_qubits",
                "encoding": (
                    "ZZFeatureMap encodes the four inputs on qubits 0-3. "
                    "Remaining qubits start in |0> and are active workspace "
                    "qubits: EfficientSU2 applies trainable rotations and "
                    "linear nearest-neighbor entanglement across the full "
                    "register, allowing data-bearing and added qubits to "
                    "interact. No extra classical inputs are added."
                ),
                "q4_alias": (
                    "At q=4 this arm is identical to full_system and reuses "
                    "the same fit."
                ),
            },
        },
        "circuit": {
            "feature_map": "ZZFeatureMap",
            "feature_map_reps": FEATURE_MAP_REPS,
            "ansatz": "EfficientSU2",
            "ansatz_reps": ANSATZ_REPS,
            "entanglement": ENTANGLEMENT,
            "readout_mode": READOUT_MODE,
            "readout_hidden_size": READOUT_HIDDEN_SIZE,
            "shots": SHOTS,
        },
        "training": {
            "max_iterations": MAX_ITERATIONS,
            "patience": PATIENCE,
            "learning_rate": LEARNING_RATE,
            "batch_size": TRAINING_BATCH_SIZE,
            "training_random_seed": TRAINING_SEED,
            "initialization_seeds": list(MODEL_SEEDS),
            "readout_warm_start_samples": 200,
            "budgets_identical_across_configurations": True,
        },
        "classical_control": {
            "model": "PCA + StandardScaler + RBF SVC",
            "fit_split": "train_only",
            "C": 1.0,
            "gamma": "scale",
            "class_weight": "balanced",
            "probability": True,
        },
        "metrics_split": "validation_only",
        "final_test": "locked_not_opened_or_evaluated",
    }


def _report_template(sources: dict, dataset_metadata: dict | None = None) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "QNIM_Q1_6_SCALING_PILOT_V1",
        "status": "in_progress",
        "configuration": _configuration(),
        "dataset": {
            "dataset_seed": DATASET_SEED,
            "class_names": sources["class_names"],
            "source_hashes": sources["source_hashes"],
            "dataset_hashes": (
                dataset_metadata.get("dataset_hashes")
                if dataset_metadata
                else None
            ),
            "generation": (
                dataset_metadata.get("generation")
                if dataset_metadata
                else None
            ),
            "pca_explained_variance_ratio": (
                dataset_metadata.get("pca_explained_variance_ratio")
                if dataset_metadata
                else None
            ),
            "first_12_component_hashes_match_q1": (
                dataset_metadata is not None
            ),
            "test_access": {
                "files_loaded": False,
                "projected": False,
                "evaluated": False,
                "persisted": False,
            },
        },
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "qiskit": importlib.metadata.version("qiskit"),
            "qiskit_aer": importlib.metadata.version("qiskit-aer"),
            "pycbc": importlib.metadata.version("PyCBC"),
        },
        "runs": [],
        "errors": [],
        "aggregates": None,
        "analysis": None,
        "final_test": {
            "opened": False,
            "evaluated": False,
            "status": "locked_not_opened",
        },
    }


def _validate_resume(report: dict, sources: dict) -> None:
    if report.get("experiment_id") != "QNIM_Q1_6_SCALING_PILOT_V1":
        raise ValueError("Cannot resume a report from another experiment")
    if report.get("configuration") != _configuration():
        raise ValueError("Cannot resume Q1.6 with changed configuration")
    if report.get("dataset", {}).get("source_hashes") != sources["source_hashes"]:
        raise ValueError("Cannot resume Q1.6 with changed Q1 reference reports")
    if report.get("final_test", {}).get("evaluated") is not False:
        raise ValueError("Cannot resume Q1.6 after final-test evaluation")


def _fit_final_reports(report: dict, output_path: Path) -> None:
    report["aggregates"] = _aggregate(report["runs"])
    expected = _expected_keys()
    completed = {
        _record_key(row)
        for row in report["runs"]
        if row.get("status") == "completed"
    }
    missing = expected - completed
    report["status"] = (
        "partial"
        if missing
        else "completed_with_errors"
        if report["errors"]
        else "completed"
    )
    report["missing_runs"] = [
        list(key) for key in sorted(missing, key=str)
    ]
    _atomic_json(output_path, report)
    _write_markdown_report(report, output_path.with_suffix(".md"))
    _write_scaling_figure(report, output_path.with_suffix(".png"))


def _write_markdown_report(report: dict, path: Path) -> None:
    config = report["configuration"]
    aggregate = report.get("aggregates") or {}
    rows = aggregate.get("by_model_and_arm", {})
    lines = [
        "# Q1.6 — Controlled QNIM Scaling Pilot",
        "",
        f"**Estado:** `{report['status']}`  ",
        "**Evaluación:** validation only; final test remains locked.",
        "",
        "## Diseño y límites",
        "",
        f"- QuBits: {config['qubit_counts']}; inicializaciones: {config['training']['initialization_seeds']}.",
        "- Dataset seed 42 y las mismas muestras train/validation para todas las condiciones.",
        "- Los hashes de las primeras 12 componentes reproducen Q1.2/Q1.4.",
        "- Los valores 42/43/44 son inicializaciones del modelo, no tres datasets independientes.",
        "- En la línea fija se codifican las primeras cuatro PCA en qubits 0–3; qubits adicionales forman parte del workspace entrenable de EfficientSU2 y se conectan por el entrelazamiento lineal.",
        "- En q=4 ambas líneas coinciden y comparten ajuste.",
        "- No se cargaron, proyectaron, persistieron ni evaluaron arrays del test final.",
        "- Para conservar exactamente el flujo RNG histórico, se descartaron los draws posteriores a validation; no se proyectaron ni almacenaron.",
        "",
        "## Métricas QNIM de validation",
        "",
        "| Línea | Qubits | n_features | Balanced accuracy (media ± SD) | Accuracy | Macro-F1 | Cross-entropy | Parámetros cuánticos | Parámetros clásicos | Observables | Tiempo medio (s) | RSS pico medio (MB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, key in (
        ("Full-system", "qnim:full_system"),
        ("Fixed-input", "qnim:fixed_input_dimension"),
    ):
        for n_qubits in QUIBIT_COUNTS:
            record = rows.get(key, {}).get(str(n_qubits))
            if not record:
                continue
            metrics = record["metrics"]
            ba = metrics["balanced_accuracy"]
            lines.append(
                f"| {arm} | {n_qubits} | "
                f"{n_qubits if arm == 'Full-system' else FIXED_INPUT_DIMENSION} | "
                f"{ba['mean']:.4f} ± {ba['std']:.4f} | "
                f"{metrics['accuracy']['mean']:.4f} | "
                f"{metrics['macro_f1']['mean']:.4f} | "
                f"{metrics['cross_entropy']['mean']:.4f} | "
                f"{record['quantum_trainable_parameters']} | "
                f"{record['classical_trainable_parameters']} | "
                f"{record['n_observables']} | "
                f"{record['mean_training_seconds']:.1f} | "
                f"{record['mean_peak_rss_bytes'] / (1024 ** 2):.1f} |"
            )
    lines.extend([
        "",
        "## Control PCA + SVM RBF",
        "",
        "| QuBits | PCA features | Balanced accuracy (media ± SD) | Accuracy | Macro-F1 | Cross-entropy | Vectores soporte (media) | Parámetros numéricos almacenados (media) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for n_qubits in QUIBIT_COUNTS:
        record = rows.get("pca_svm_rbf:full_system", {}).get(str(n_qubits))
        if not record:
            continue
        metrics = record["metrics"]
        ba = metrics["balanced_accuracy"]
        lines.append(
            f"| {n_qubits} | {n_qubits} | {ba['mean']:.4f} ± {ba['std']:.4f} | "
            f"{metrics['accuracy']['mean']:.4f} | "
            f"{metrics['macro_f1']['mean']:.4f} | "
            f"{metrics['cross_entropy']['mean']:.4f} | "
            f"{record['svm_support_vector_count_mean']:.1f} | "
            f"{record['svm_stored_parameter_count_mean']:.1f} |"
        )
    analysis = aggregate.get("descriptive_linear_slopes", {})
    lines.extend([
        "",
        "## Lectura del piloto",
        "",
        "Las pendientes lineales se reportan como descripciones de estas cuatro configuraciones; con tres inicializaciones y un único dataset no constituyen evidencia estadística confirmatoria.",
        "",
    ])
    for name, result in analysis.items():
        lines.append(
            f"- `{name}`: pendiente balanced accuracy = "
            f"{result['balanced_accuracy_per_qubit']:.6f} por qubit; "
            f"accuracy = {result['accuracy_per_qubit']:.6f} por qubit."
        )
    lines.extend([
        "",
        "La tabla de diferencias pareadas entre líneas y frente al SVM está en `aggregates.paired_seed_differences` del JSON. El detalle por inicialización, curvas de pérdida, matrices de confusión, recalls por clase y parámetros finales están en `runs`.",
        "",
        "![Curvas de escalabilidad Q1.6](qnim_q1_6_scaling_pilot_v1.png)",
        "",
        "## Errores y limitaciones",
        "",
    ])
    if report["errors"]:
        lines.extend(f"- {item['message']}" for item in report["errors"])
    else:
        lines.append("- No se registraron errores de ejecución.")
    lines.extend([
        "- Las tres inicializaciones usan el mismo dataset seed 42; no son réplicas independientes de generación física.",
        "- En full-system crecen simultáneamente los componentes PCA, qubits y observables/readout; SVM controla el valor predictivo de las entradas, pero no aísla por sí solo cada fuente de capacidad.",
        "- En fixed-input, los qubits adicionales son workspace paramétrico que interactúa mediante EfficientSU2; aumentan también los parámetros cuánticos y clásicos, por lo que un cambio no se atribuye únicamente al número de qubits.",
        "- El RSS pico muestreado es la memoria del proceso completo, incluidos dataset y cachés de Python/Qiskit; no es una atribución exacta al circuito aislado.",
        "- No se ejecutó hardware IBM, no se accedió al test final y no se inicia Q1.7 automáticamente.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_scaling_figure(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = report.get("aggregates", {}).get("by_model_and_arm", {})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metric_specs = (
        ("balanced_accuracy", "Balanced accuracy"),
        ("accuracy", "Accuracy"),
    )
    series = (
        ("qnim:full_system", "QNIM full-system", "o"),
        ("qnim:fixed_input_dimension", "QNIM fixed 4 PCA", "s"),
        ("pca_svm_rbf:full_system", "PCA + SVM RBF", "^"),
    )
    for axis, (metric, title) in zip(axes[0], metric_specs):
        for key, label, marker in series:
            rows = summary.get(key, {})
            x = [q for q in QUIBIT_COUNTS if str(q) in rows]
            y = [
                rows[str(q)]["metrics"][metric]["mean"]
                for q in x
            ]
            error = [
                rows[str(q)]["metrics"][metric]["std"]
                for q in x
            ]
            if x:
                axis.errorbar(x, y, yerr=error, marker=marker, capsize=3, label=label)
        axis.set_title(title)
        axis.set_xlabel("Qubits")
        axis.set_ylabel("Score")
        axis.set_xticks(QUIBIT_COUNTS)
        axis.grid(True, alpha=0.3)
        axis.legend()

    full = summary.get("qnim:full_system", {})
    xs = [q for q in QUIBIT_COUNTS if str(q) in full]
    if xs:
        axes[1, 0].plot(
            xs,
            [
                full[str(q)]["quantum_trainable_parameters"]
                for q in xs
            ],
            marker="o",
            label="Quantum trainable",
        )
        axes[1, 0].plot(
            xs,
            [
                full[str(q)]["classical_trainable_parameters"]
                for q in xs
            ],
            marker="s",
            label="Classical readout trainable",
        )
        axes[1, 0].plot(
            xs,
            [
                full[str(q)]["n_observables"]
                for q in xs
            ],
            marker="^",
            label="Readout observables",
        )
    axes[1, 0].set_title("Trainable parameters and observables")
    axes[1, 0].set_xlabel("Qubits")
    axes[1, 0].set_xticks(QUIBIT_COUNTS)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    memory_axis = axes[1, 1].twinx()
    for key, label, marker in series[:2]:
        rows = summary.get(key, {})
        x = [q for q in QUIBIT_COUNTS if str(q) in rows]
        if x:
            axes[1, 1].plot(
                x,
                [
                    rows[str(q)]["mean_training_seconds"]
                    for q in x
                ],
                marker=marker,
                label=f"{label} time",
            )
            memory_axis.plot(
                x,
                [
                    rows[str(q)]["mean_peak_rss_bytes"] / (1024 ** 2)
                    for q in x
                ],
                marker=marker,
                linestyle="--",
                label=f"{label} peak RSS (MB)",
            )
    axes[1, 1].set_title("Computation cost")
    axes[1, 1].set_xlabel("Qubits")
    axes[1, 1].set_ylabel("Seconds")
    memory_axis.set_ylabel("Process RSS (MB)")
    axes[1, 1].set_xticks(QUIBIT_COUNTS)
    axes[1, 1].grid(True, alpha=0.3)
    cost_handles, cost_labels = axes[1, 1].get_legend_handles_labels()
    memory_handles, memory_labels = memory_axis.get_legend_handles_labels()
    axes[1, 1].legend(
        cost_handles + memory_handles,
        cost_labels + memory_labels,
        fontsize="small",
    )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_q1_6(
    output_path: Path = DEFAULT_OUTPUT,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    resume: bool = False,
    prepare_only: bool = False,
) -> dict:
    output_path = Path(output_path)
    dataset_dir = Path(dataset_dir)
    if output_path.exists() and not resume:
        raise FileExistsError(
            f"Refusing to overwrite Q1.6 report: {output_path}"
        )
    if resume and not output_path.is_file():
        raise FileNotFoundError(f"Q1.6 checkpoint does not exist: {output_path}")
    sources = _read_frozen_sources()
    if resume:
        report = json.loads(output_path.read_text(encoding="utf-8"))
        _validate_resume(report, sources)
    else:
        report = _report_template(sources)
        _atomic_json(output_path, report)

    dataset_paths = [
        dataset_dir / f"{name}.npy"
        for name in ("X_train", "y_train", "X_val", "y_val")
    ]
    present_dataset_files = [path.is_file() for path in dataset_paths]
    if any(present_dataset_files) and not all(present_dataset_files):
        raise FileNotFoundError("Q1.6 train/validation dataset cache is incomplete")
    dataset_files_exist = all(present_dataset_files)
    if dataset_files_exist:
        dataset = _load_dataset(dataset_dir, sources["class_names"])
        metadata = dataset["metadata"]
        prefix_hashes = {
            "train_sha256": _split_sha256(
                dataset["X_train"][:, :12], dataset["y_train"]
            ),
            "validation_sha256": _split_sha256(
                dataset["X_val"][:, :12], dataset["y_val"]
            ),
        }
        if prefix_hashes != sources["q1_2"]["dataset_hashes"]:
            raise ValueError("Saved Q1.6 dataset does not reproduce Q1 split hashes")
        dataset_metadata = {
            "dataset_hashes": metadata["dataset_hashes"],
            "generation": metadata["generation"],
            "pca_explained_variance_ratio": metadata[
                "pca_explained_variance_ratio"
            ],
        }
        report["dataset"].update(dataset_metadata)
        report["dataset"]["first_12_component_hashes_match_q1"] = True
    else:
        try:
            generated = _generate_dataset(sources)
            metadata = _save_dataset(dataset_dir, generated)
        except Exception as exc:
            report["status"] = "failed"
            report["errors"].append({
                "stage": "train_validation_generation",
                "type": type(exc).__name__,
                "message": str(exc),
            })
            _atomic_json(output_path, report)
            raise
        dataset = _load_dataset(dataset_dir, sources["class_names"])
        report["dataset"].update({
            "dataset_hashes": metadata["dataset_hashes"],
            "generation": metadata["generation"],
            "pca_explained_variance_ratio": metadata[
                "pca_explained_variance_ratio"
            ],
            "first_12_component_hashes_match_q1": True,
        })
    report["dataset"]["cache_directory"] = str(dataset_dir.resolve())
    if prepare_only:
        report["status"] = "prepared"
        _atomic_json(output_path, report)
        return report
    _atomic_json(output_path, report)

    completed_keys = {
        _record_key(row)
        for row in report["runs"]
        if row.get("status") == "completed"
    }
    for seed in MODEL_SEEDS:
        for n_qubits in QUIBIT_COUNTS:
            for model_family, arm, n_features in (
                ("pca_svm_rbf", "full_system", n_qubits),
                ("qnim", "full_system", n_qubits),
                ("qnim", "fixed_input_dimension", FIXED_INPUT_DIMENSION),
            ):
                if arm == "fixed_input_dimension" and (
                    n_qubits == FIXED_INPUT_DIMENSION
                ):
                    continue
                key = (model_family, arm, n_qubits, seed)
                if key in completed_keys:
                    continue
                print(
                    f"Q1.6 running {model_family} {arm} q={n_qubits} "
                    f"features={n_features} seed={seed}"
                )
                try:
                    if model_family == "pca_svm_rbf":
                        record = _run_svm(
                            dataset,
                            n_features=n_features,
                            seed=seed,
                            n_qubits=n_qubits,
                            class_names=sources["class_names"],
                        )
                    else:
                        record = _run_qnim(
                            dataset,
                            arm=arm,
                            n_qubits=n_qubits,
                            n_features=n_features,
                            seed=seed,
                            class_names=sources["class_names"],
                        )
                    record["initialization_seed"] = seed
                    record["dataset_seed"] = DATASET_SEED
                    record["train_sha256"] = report["dataset"]["dataset_hashes"][
                        "train_sha256"
                    ]
                    record["validation_sha256"] = report["dataset"][
                        "dataset_hashes"
                    ]["validation_sha256"]
                    report["runs"].append(record)
                    completed_keys.add(key)
                except Exception as exc:
                    report["errors"].append({
                        "model_family": model_family,
                        "arm": arm,
                        "n_qubits": n_qubits,
                        "n_features": n_features,
                        "initialization_seed": seed,
                        "type": type(exc).__name__,
                        "message": str(exc),
                    })
                _atomic_json(output_path, report)

    _fit_final_reports(report, output_path)
    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validation-only controlled QNIM qubit scaling pilot"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Generate/hash-check train and validation, without starting model runs",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    report = run_q1_6(
        output_path=args.output,
        dataset_dir=args.dataset_dir,
        resume=args.resume,
        prepare_only=args.prepare_only,
    )
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output.resolve()),
        "runs": len(report["runs"]),
        "errors": len(report["errors"]),
        "final_test": report["final_test"]["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
