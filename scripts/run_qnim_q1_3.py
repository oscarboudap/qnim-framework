"""Probe frozen QNIM circuit representations with classical classifiers."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.diagnose_qnim_classical_baselines import _make_model
from scripts.run_qnim_q1 import (
    DEFAULT_REFERENCE,
    N_QUIBITS,
    _architecture_metadata,
    _write_json_atomically,
)
from scripts.run_qnim_q1_1 import (
    DATASET_SEED,
    MODEL_SEEDS,
    _classification_metrics,
)
from scripts.run_qnim_q1_2 import (
    DEFAULT_SMOKE_REFERENCE,
    _build_loss,
    _read_dataset,
)

DEFAULT_Q1_2_REPORT = (
    ROOT / "results" / "scaling_study" / "qnim_q1_2_q1_3_parameters_v1.json"
)
DEFAULT_OUTPUT = ROOT / "results" / "scaling_study" / "qnim_q1_3_v2.json"
DEFAULT_SMOKE_OUTPUT = ROOT / "results" / "scaling_study" / "qnim_q1_3_smoke_v2.json"
SCHEMA_VERSION = 1
PROBE_MODELS = ("logistic_regression", "svm_rbf", "mlp")
N_CLASSES = 13
MLP_MAX_ITER = 2000


def _sha256(array):
    return hashlib.sha256(
        np.asarray(array, dtype=np.float64).tobytes()
    ).hexdigest().upper()


def _read_q1_2_report(path):
    content = Path(path).read_bytes()
    report = json.loads(content)
    if report.get("experiment_id") != "QNIM_Q1_2_V1":
        raise ValueError("Q1.3 requires the completed full Q1.2 report")
    if report.get("status") != "completed":
        raise ValueError("Q1.2 must be completed before Q1.3")
    if report.get("final_test", {}).get("status") != "locked_not_evaluated":
        raise ValueError("Q1.3 requires the Q1.2 final test to remain locked")
    if report.get("model_initialization_seeds") != list(MODEL_SEEDS):
        raise ValueError("Q1.2 model initialization seeds differ from Q1.3")
    if report.get("architecture") != _architecture_metadata():
        raise ValueError("Q1.2 architecture does not match the Q1.3 circuit")
    if report.get("validation_policy", {}).get("final_test") != "locked_not_evaluated":
        raise ValueError("Q1.2 validation policy does not keep the final test locked")
    return report, hashlib.sha256(content).hexdigest().upper()


def _circuit_parameter_sets(q1_2_report, seeds=MODEL_SEEDS):
    runs = q1_2_report["experiments"]["Q1_2_A"]["trainable_ansatz"]
    by_seed = {run["initialization_seed"]: run for run in runs}
    if len(by_seed) != len(runs):
        raise ValueError("Q1.2 contains duplicate trainable ansatz seeds")

    n_params = q1_2_report["architecture"]["ansatz_parameters"]
    parameter_sets = {}
    for seed in seeds:
        run = by_seed.get(seed)
        if run is None or run.get("status") != "completed":
            raise ValueError(f"Q1.2 has no completed trainable run for seed {seed}")

        initial = np.random.default_rng(seed).normal(0.0, 0.01, n_params)
        if _sha256(initial) != run.get("initial_ansatz_sha256"):
            raise ValueError(f"Q1.2 initial ansatz hash mismatch for seed {seed}")

        raw_trained = run.get("final_ansatz_parameters")
        if raw_trained is None:
            raise ValueError(
                f"Q1.2 does not persist the exact final ansatz vector for seed {seed}; "
                "rerun Q1.2 with final parameter export enabled"
            )
        trained = np.asarray(raw_trained, dtype=float)
        if trained.shape != (n_params,) or not np.isfinite(trained).all():
            raise ValueError(f"Q1.2 final ansatz parameters are invalid for seed {seed}")
        if _sha256(trained) != run.get("final_ansatz_sha256"):
            raise ValueError(f"Q1.2 final ansatz hash mismatch for seed {seed}")
        parameter_sets[seed] = {"initial": initial, "trained": trained}
    return parameter_sets


def _fit_probe_models(X_train, y_train, X_val, y_val, seed, model_names=PROBE_MODELS):
    results = {}
    for model_name in model_names:
        model = _make_model(
            model_name,
            seed,
            probability=True,
            mlp_max_iter=MLP_MAX_ITER,
        )
        with warnings.catch_warnings(record=True) as fit_warnings:
            warnings.simplefilter("always")
            model.fit(X_train, y_train)

        train_probabilities = model.predict_proba(X_train)
        validation_probabilities = model.predict_proba(X_val)
        final_estimator = model.steps[-1][1] if hasattr(model, "steps") else model
        classes = np.asarray(final_estimator.classes_)
        if not np.array_equal(classes, np.arange(N_CLASSES)):
            raise ValueError(f"{model_name} did not fit all 13 classes")
        results[model_name] = {
            "training": _classification_metrics(y_train, train_probabilities),
            "validation": _classification_metrics(y_val, validation_probabilities),
            "fit_warnings": [
                f"{item.category.__name__}: {item.message}" for item in fit_warnings
            ],
            "iterations": (
                int(final_estimator.n_iter_)
                if model_name == "mlp" and hasattr(final_estimator, "n_iter_")
                else None
            ),
        }
    return results


def _extract_and_probe(
    loss_fn, X_train, y_train, X_val, y_val, seed, condition, parameters
):
    started = time.perf_counter()
    train_features = loss_fn.measure_quantum_features_exact(X_train, parameters)
    validation_features = loss_fn.measure_quantum_features_exact(X_val, parameters)
    extraction_seconds = time.perf_counter() - started
    expected_shape = (len(X_train), N_QUIBITS)
    if train_features.shape != expected_shape or validation_features.shape != (
        len(X_val),
        N_QUIBITS,
    ):
        raise ValueError("Exact Q1.3 marginals have unexpected dimensions")
    if not np.isfinite(train_features).all() or not np.isfinite(validation_features).all():
        raise ValueError("Exact Q1.3 marginals contain non-finite values")

    return {
        "status": "completed",
        "initialization_seed": seed,
        "circuit_condition": condition,
        "ansatz_parameters_sha256": _sha256(parameters),
        "train_feature_sha256": _sha256(train_features),
        "validation_feature_sha256": _sha256(validation_features),
        "feature_shapes": {
            "train": list(train_features.shape),
            "validation": list(validation_features.shape),
        },
        "feature_extraction_seconds": extraction_seconds,
        "classifiers": _fit_probe_models(
            train_features, y_train, validation_features, y_val, seed
        ),
    }


def _validate_checkpoint(report, expected):
    if report.get("status") != "in_progress":
        raise ValueError("Only an in-progress Q1.3 report can be resumed")
    for key in (
        "checkpoint_schema_version",
        "experiment_id",
        "q1_2_report_sha256",
        "classical_reference_id",
        "classical_reference_sha256",
        "dataset_seed",
        "model_initialization_seeds",
        "dataset_hashes",
        "class_names",
        "architecture",
        "configuration",
        "software_versions",
        "validation_policy",
        "final_test",
    ):
        if report.get(key) != expected.get(key):
            raise ValueError(f"Cannot resume Q1.3: {key} differs from checkpoint")
    runs = report.get("experiments", {}).get("frozen_circuit_probes")
    if not isinstance(runs, list):
        raise ValueError("Cannot resume Q1.3: invalid probe checkpoints")
    completed = set()
    valid_conditions = {"initial", "trained"}
    for run in runs:
        key = (run.get("initialization_seed"), run.get("circuit_condition"))
        if (
            key in completed
            or key[0] not in expected["model_initialization_seeds"]
            or key[1] not in valid_conditions
            or run.get("status") != "completed"
        ):
            raise ValueError("Cannot resume Q1.3: invalid completed probe")
        completed.add(key)


def run_q1_3(
    q1_2_report_path=DEFAULT_Q1_2_REPORT,
    reference_manifest=DEFAULT_REFERENCE,
    output_path=DEFAULT_OUTPUT,
    adapter_factory=None,
    resume=False,
    smoke=False,
    smoke_reference_path=DEFAULT_SMOKE_REFERENCE,
):
    q1_2_report_path = Path(q1_2_report_path)
    output_path = Path(output_path)
    if smoke and output_path == DEFAULT_OUTPUT:
        output_path = DEFAULT_SMOKE_OUTPUT
    if output_path.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite Q1.3 results: {output_path}")
    if resume and not output_path.exists():
        raise FileNotFoundError(f"Q1.3 checkpoint does not exist: {output_path}")

    q1_2_report, q1_2_sha256 = _read_q1_2_report(q1_2_report_path)
    parameter_sets = _circuit_parameter_sets(
        q1_2_report, (DATASET_SEED,) if smoke else MODEL_SEEDS
    )
    _, reference_sha256, reference_id, expected_dataset, split, seeds = (
        _read_dataset(
            smoke,
            reference_manifest,
            smoke_reference_path,
            adapter_factory,
        )
    )
    if not smoke and reference_sha256 != q1_2_report["classical_reference_sha256"]:
        raise ValueError("Q1.2 and Q1.3 classical reference hashes differ")
    X_train, y_train, X_val, y_val = split
    if list(expected_dataset["class_names"]) != q1_2_report["class_names"]:
        raise ValueError("Q1.2 and Q1.3 class labels differ")
    loss_fn = _build_loss(X_train, y_train, shots=512)
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "qiskit": importlib.metadata.version("qiskit"),
        "qiskit_aer": importlib.metadata.version("qiskit-aer"),
    }
    configuration = {
        "smoke_mode": smoke,
        "measurement": "exact statevector single-qubit marginals",
        "n_qubits": N_QUIBITS,
        "n_features": N_QUIBITS,
        "feature_map": "ZZFeatureMap",
        "feature_map_reps": 2,
        "ansatz": "EfficientSU2",
        "ansatz_reps": 2,
        "entanglement": "linear",
        "circuit_conditions": ["initial", "trained"],
        "classifier_models": list(PROBE_MODELS),
        "classifier_probability_estimates": True,
        "classifier_configuration": {
            "logistic_regression": {
                "standard_scaler": True,
                "max_iter": 2000,
                "class_weight": "balanced",
            },
            "svm_rbf": {
                "standard_scaler": True,
                "kernel": "rbf",
                "class_weight": "balanced",
                "probability": True,
                "random_state": "initialization seed",
            },
            "mlp": {
                "standard_scaler": True,
                "hidden_layer_sizes": [64],
                "max_iter": MLP_MAX_ITER,
                "early_stopping": False,
            },
        },
        "validation_used_for_fitting": False,
    }
    report = {
        "checkpoint_schema_version": SCHEMA_VERSION,
        "experiment_id": "QNIM_Q1_3_SMOKE_V2" if smoke else "QNIM_Q1_3_V2",
        "experiment_scope": (
            "integration smoke only; not scientific evidence"
            if smoke else "frozen-circuit quantum feature representation probe"
        ),
        "status": "in_progress",
        "active_run": None,
        "q1_2_report_sha256": q1_2_sha256,
        "classical_reference_id": reference_id,
        "classical_reference_sha256": reference_sha256,
        "dataset_seed": DATASET_SEED,
        "model_initialization_seeds": list(seeds),
        "dataset_hashes": {
            "train_sha256": expected_dataset["train_sha256"],
            "validation_sha256": expected_dataset["validation_sha256"],
        },
        "class_names": list(expected_dataset["class_names"]),
        "architecture": _architecture_metadata(),
        "configuration": configuration,
        "software_versions": versions,
        "validation_policy": {
            "validation_used_for_fitting": False,
            "validation_used_for_metrics_only": True,
            "final_test": "locked_not_evaluated",
        },
        "experiments": {"frozen_circuit_probes": []},
        "final_test": {
            "evaluated": False,
            "status": "locked_not_evaluated",
            "models": {},
        },
    }

    if resume:
        expected_report = report
        report = json.loads(output_path.read_text(encoding="utf-8"))
        _validate_checkpoint(report, expected_report)
        report["active_run"] = None
        _write_json_atomically(output_path, report)
    else:
        _write_json_atomically(output_path, report)

    completed = {
        (run["initialization_seed"], run["circuit_condition"])
        for run in report["experiments"]["frozen_circuit_probes"]
    }
    for seed in seeds:
        for condition in ("initial", "trained"):
            if (seed, condition) in completed:
                continue
            report["active_run"] = {
                "initialization_seed": seed,
                "circuit_condition": condition,
            }
            _write_json_atomically(output_path, report)
            run = _extract_and_probe(
                loss_fn,
                X_train,
                y_train,
                X_val,
                y_val,
                seed,
                condition,
                parameter_sets[seed][condition],
            )
            report["experiments"]["frozen_circuit_probes"].append(run)
            report["active_run"] = None
            _write_json_atomically(output_path, report)
    report["status"] = "completed"
    _write_json_atomically(output_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Probe frozen QNIM circuit features with classical classifiers."
    )
    parser.add_argument("--q1-2-report", type=Path, default=DEFAULT_Q1_2_REPORT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-reference", type=Path, default=DEFAULT_SMOKE_REFERENCE)
    args = parser.parse_args()
    output = args.output or (DEFAULT_SMOKE_OUTPUT if args.smoke else DEFAULT_OUTPUT)
    report = run_q1_3(
        q1_2_report_path=args.q1_2_report,
        reference_manifest=args.reference,
        output_path=output,
        resume=args.resume,
        smoke=args.smoke,
        smoke_reference_path=args.smoke_reference,
    )
    print(json.dumps({
        "status": report["status"],
        "output": str(output),
        "final_test": report["final_test"]["status"],
        "probe_runs": len(report["experiments"]["frozen_circuit_probes"]),
    }, indent=2))


if __name__ == "__main__":
    main()
