"""Run the frozen three-seed Q1 simulator experiment on the LIGO-PCA split."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DEFAULT_REFERENCE = ROOT / "results" / "scaling_study" / "classical_reference_v1.json"
DEFAULT_OUTPUT = ROOT / "results" / "scaling_study" / "qnim_q1_v1.json"
REFERENCE_ID = "CLASSICAL_REFERENCE_V1"
CHANCE = 1.0 / 13
N_QUIBITS = 12
FEATURE_MAP_REPS = 2
ANSATZ_REPS = 2
ENTANGLEMENT = "linear"
READOUT_HIDDEN_SIZE = 16
TRAINING_BATCH_SIZE = 64
PATIENCE = 20
LEARNING_RATE = 0.01
READOUT_REFIT_EVERY = 10
N_FEYNMAN_PARAMS = 4


def _read_reference(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("reference_id") != REFERENCE_ID:
        raise ValueError(f"Expected reference {REFERENCE_ID}")

    report_path = manifest_path.parent / manifest["source_report"]
    report_bytes = report_path.read_bytes()
    report_sha256 = hashlib.sha256(report_bytes).hexdigest().upper()
    if report_sha256 != manifest.get("source_sha256"):
        raise ValueError("Classical reference report SHA-256 does not match its manifest")
    report = json.loads(report_bytes)

    if report.get("decision", {}).get("eligible_for_qnim_training") is not True:
        raise ValueError("The frozen classical reference does not permit QNIM training")
    protocol = report.get("pre_registered_protocol", {})
    if protocol.get("eligibility_split") != "validation_only":
        raise ValueError("Classical eligibility was not determined on validation")
    if protocol.get("test_evaluation_requested") is not False:
        raise ValueError("The classical reference must keep final test locked")
    if report.get("final_test", {}).get("status") != "locked_not_evaluated":
        raise ValueError("The classical reference final test is not locked")
    if report.get("n_classes") != 13:
        raise ValueError("Q1 requires the frozen 13-class reference")
    if protocol.get("seeds") != [42, 43, 44]:
        raise ValueError("Q1 requires the frozen seeds 42, 43, and 44")
    if (
        report.get("waveform_source") != "PyCBC"
        or report.get("strict_physics") is not True
        or report.get("fallback_allowed") is not False
        or report.get("approximant") != "IMRPhenomPv2"
    ):
        raise ValueError("The frozen reference is not strict PyCBC IMRPhenomPv2 data")
    if report.get("dataset_configuration") != {
        "train_per_class": 200,
        "validation_per_class": 50,
        "test_per_class": 100,
        "components": N_QUIBITS,
        "pca_fit_components": 64,
    }:
        raise ValueError("The frozen reference dataset configuration changed")

    return report, report_sha256


def _validate_dataset(dataset, reference_dataset, config):
    n_classes = 13
    expected_train = config["train_per_class"] * n_classes
    expected_validation = config["validation_per_class"] * n_classes
    if dataset.n_classes != n_classes:
        raise ValueError("Generated dataset does not contain 13 classes")
    if list(dataset.theory_names) != reference_dataset["class_names"]:
        raise ValueError("Generated class order differs from the frozen reference")

    X_train = np.asarray(dataset.X_train)
    y_train = np.asarray(dataset.y_train)
    X_val = np.asarray(dataset.X_val)
    y_val = np.asarray(dataset.y_val)
    if X_train.shape != (expected_train, N_QUIBITS):
        raise ValueError(f"Unexpected X_train shape: {X_train.shape}")
    if X_val.shape != (expected_validation, N_QUIBITS):
        raise ValueError(f"Unexpected X_val shape: {X_val.shape}")
    if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
        raise ValueError("Train or validation features contain non-finite values")
    for split, labels, expected_count in (
        ("train", y_train, config["train_per_class"]),
        ("validation", y_val, config["validation_per_class"]),
    ):
        if labels.ndim != 1 or not np.array_equal(labels, labels.astype(int)):
            raise ValueError(f"{split} labels must be a one-dimensional integer array")
        if not np.array_equal(
            np.bincount(labels.astype(int), minlength=n_classes),
            np.full(n_classes, expected_count),
        ):
            raise ValueError(f"{split} does not have the frozen balanced class counts")

    from scripts.diagnose_qnim_classical_baselines import _split_sha256

    actual_train_hash = _split_sha256(X_train, y_train)
    actual_validation_hash = _split_sha256(X_val, y_val)
    if actual_train_hash != reference_dataset["train_sha256"]:
        raise ValueError(f"Seed {reference_dataset['seed']} train split hash mismatch")
    if actual_validation_hash != reference_dataset["validation_sha256"]:
        raise ValueError(
            f"Seed {reference_dataset['seed']} validation split hash mismatch"
        )
    return X_train, y_train, X_val, y_val


def _architecture_metadata():
    from src.infrastructure.qiskit_vqc_trainer import (
        _build_feature_map_and_ansatz,
        _n_readout_params,
    )

    circuit, feature_params, ansatz_params = _build_feature_map_and_ansatz(
        N_QUIBITS,
        n_features=N_QUIBITS,
        feature_map_reps=FEATURE_MAP_REPS,
        ansatz_reps=ANSATZ_REPS,
        entanglement=ENTANGLEMENT,
    )
    readout_params = _n_readout_params(
        N_QUIBITS, 13, READOUT_HIDDEN_SIZE
    )
    return {
        "n_qubits": circuit.num_qubits,
        "n_features": N_QUIBITS,
        "feature_map": "ZZFeatureMap",
        "feature_map_reps": FEATURE_MAP_REPS,
        "feature_map_parameters": len(feature_params),
        "ansatz": "EfficientSU2",
        "ansatz_reps": ANSATZ_REPS,
        "ansatz_parameters": len(ansatz_params),
        "entanglement": ENTANGLEMENT,
        "readout": "12 qubit marginals -> one-hidden-layer MLP -> 13 classes",
        "readout_hidden_size": READOUT_HIDDEN_SIZE,
        "readout_parameters": readout_params,
        "total_trainable_parameters": len(ansatz_params) + readout_params,
    }


def _train_and_evaluate_seed(
    seed, X_train, y_train, X_val, y_val, max_iterations, shots,
    trainer_factory=None,
):
    if trainer_factory is None:
        from src.infrastructure.qiskit_vqc_trainer import QiskitVQCTrainer

        trainer_factory = QiskitVQCTrainer

    trainer = trainer_factory(
        mode="sim",
        random_seed=seed,
        feature_map_reps=FEATURE_MAP_REPS,
        ansatz_reps=ANSATZ_REPS,
        entanglement=ENTANGLEMENT,
        readout_hidden_size=READOUT_HIDDEN_SIZE,
        training_batch_size=TRAINING_BATCH_SIZE,
        patience=PATIENCE,
        learning_rate=LEARNING_RATE,
        readout_refit_every=READOUT_REFIT_EVERY,
    )
    started = time.perf_counter()
    training = trainer.train_vqc(
        X_train=X_train,
        y_train=y_train,
        num_qubits=N_QUIBITS,
        n_features=N_QUIBITS,
        max_iterations=max_iterations,
        shots=shots,
        n_feynman_params=N_FEYNMAN_PARAMS,
    )
    _, _, predictions = trainer._evaluate_real_predictions(
        weights=training["weights"],
        X_train=X_train,
        X_val=X_val,
        y_val=y_val,
        n_qubits=N_QUIBITS,
        n_features=N_QUIBITS,
        n_classes=13,
        shots=shots,
    )
    return {
        "seed": seed,
        "training_loss": float(training["training_loss"]),
        "iterations": int(training["iterations"]),
        "circuit_evaluations": int(training["n_circuit_evaluations"]),
        "training_time_seconds": float(training["execution_time_seconds"]),
        "validation_accuracy": float(accuracy_score(y_val, predictions)),
        "validation_balanced_accuracy": float(
            balanced_accuracy_score(y_val, predictions)
        ),
        "validation_confusion_matrix": confusion_matrix(
            y_val, predictions, labels=np.arange(13)
        ).tolist(),
        "validation_predictions": np.asarray(predictions, dtype=int).tolist(),
        "elapsed_seconds_including_validation": time.perf_counter() - started,
    }


def _write_json_atomically(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(report, temporary_file, indent=2, allow_nan=False)
            temporary_file.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def run_q1(
    reference_manifest=DEFAULT_REFERENCE,
    output_path=DEFAULT_OUTPUT,
    max_iterations=100,
    shots=512,
    adapter_factory=None,
    trainer_factory=None,
):
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite Q1 results: {output_path}")
    if max_iterations < 1 or shots < 1:
        raise ValueError("max_iterations and shots must be positive")

    reference, reference_sha256 = _read_reference(Path(reference_manifest))
    if adapter_factory is None:
        from src.infrastructure.ligo_pca_adapter import LIGOPCAAdapter

        adapter_factory = LIGOPCAAdapter

    config = reference["dataset_configuration"]
    reference_datasets = {
        item["seed"]: item for item in reference["datasets"]
    }
    if sorted(reference_datasets) != [42, 43, 44]:
        raise ValueError("Classical reference datasets do not match the fixed seeds")

    report = {
        "experiment_id": "QNIM_Q1_V1",
        "status": "in_progress",
        "classical_reference_id": REFERENCE_ID,
        "classical_reference_sha256": reference_sha256,
        "dataset_source": "strict PyCBC LIGOPCAAdapter; hashes must match the frozen report",
        "dataset_configuration": config,
        "architecture": _architecture_metadata(),
        "training_configuration": {
            "mode": "sim",
            "shots": shots,
            "max_iterations": max_iterations,
            "training_batch_size": TRAINING_BATCH_SIZE,
            "patience": PATIENCE,
            "learning_rate": LEARNING_RATE,
            "readout_refit_every": READOUT_REFIT_EVERY,
            "n_feynman_params": N_FEYNMAN_PARAMS,
        },
        "randomness": "Dataset and trainer random seeds are 42, 43, and 44 respectively.",
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "PyCBC": importlib.metadata.version("PyCBC"),
            "LALSuite": importlib.metadata.version("lalsuite"),
            "qiskit": importlib.metadata.version("qiskit"),
            "qiskit_aer": importlib.metadata.version("qiskit-aer"),
        },
        "per_seed": [],
        "completed_seeds": [],
        "final_test": {
            "evaluated": False,
            "status": "locked_not_evaluated",
            "models": {},
        },
    }
    _write_json_atomically(output_path, report)

    for seed in [42, 43, 44]:
        adapter = adapter_factory(
            n_components=N_QUIBITS,
            n_pca_fit=config["pca_fit_components"],
            approximant=reference["approximant"],
            max_classes=13,
            strict_physics=True,
        )
        dataset = adapter.generate_balanced_dataset(
            n_per_class=config["train_per_class"],
            n_val_per_class=config["validation_per_class"],
            n_test_per_class=config["test_per_class"],
            seed=seed,
            verbose=True,
        )
        split_data = _validate_dataset(dataset, reference_datasets[seed], config)
        result = _train_and_evaluate_seed(
            seed,
            *split_data,
            max_iterations=max_iterations,
            shots=shots,
            trainer_factory=trainer_factory,
        )
        result["train_sha256"] = reference_datasets[seed]["train_sha256"]
        result["validation_sha256"] = reference_datasets[seed]["validation_sha256"]
        report["per_seed"].append(result)
        report["completed_seeds"].append(seed)
        _write_json_atomically(output_path, report)

    scores = [
        item["validation_balanced_accuracy"] for item in report["per_seed"]
    ]
    report["validation_summary"] = {
        "statistic": "Mean of the three per-seed validation balanced accuracies.",
        "mean_balanced_accuracy": float(np.mean(scores)),
        "chance_balanced_accuracy": CHANCE,
        "every_seed_above_chance": bool(all(score > CHANCE for score in scores)),
        "qn_fits_classes_above_chance": bool(
            np.mean(scores) > CHANCE and all(score > CHANCE for score in scores)
        ),
    }
    report["status"] = "completed"
    _write_json_atomically(output_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Run frozen QNIM Q1 on the classical reference train/validation splits."
    )
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run_q1(
        max_iterations=args.max_iterations,
        shots=args.shots,
        output_path=args.output,
    )
    print(json.dumps(report["validation_summary"], indent=2))
    print(f"Q1 results written to {args.output}")


if __name__ == "__main__":
    main()
