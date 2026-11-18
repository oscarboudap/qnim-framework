"""Diagnose QNIM optimization and frozen-ansatz readout on the Q1 data."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.diagnose_qnim_classical_baselines import _make_model
from scripts.run_qnim_q1 import (
    DEFAULT_REFERENCE,
    N_QUIBITS,
    READOUT_HIDDEN_SIZE,
    TRAINING_BATCH_SIZE,
    _architecture_metadata,
    _read_reference,
    _validate_dataset,
    _write_json_atomically,
)

DEFAULT_OUTPUT = ROOT / "results" / "scaling_study" / "qnim_q1_1_v1.json"
DEFAULT_SMOKE_REFERENCE = (
    ROOT / "results" / "scaling_study" / "qnim_classical_diagnostic_smoke_v2.json"
)
DEFAULT_SMOKE_OUTPUT = ROOT / "results" / "scaling_study" / "qnim_q1_1_smoke_v1.json"
DEFAULT_BENCHMARK_OUTPUT = (
    ROOT / "results" / "scaling_study" / "qnim_q1_1_diagnostic_cost_v1.json"
)
CHECKPOINT_SCHEMA_VERSION = 1
DATASET_SEED = 42
TRAINING_SEED = 42
MODEL_SEEDS = (42, 43, 44)
FEATURE_MAP_REPS = 2
ANSATZ_REPS = 2
ENTANGLEMENT = "linear"
PATIENCE = 20
LEARNING_RATE = 0.01
READOUT_REFIT_EVERY = 10
N_FEYNMAN_PARAMS = 4


def _read_smoke_reference(path):
    content = Path(path).read_bytes()
    report = json.loads(content)
    if (
        report.get("waveform_source") != "PyCBC"
        or report.get("strict_physics") is not True
        or report.get("fallback_allowed") is not False
        or report.get("approximant") != "IMRPhenomPv2"
        or report.get("n_classes") != 13
        or report.get("final_test", {}).get("status") != "locked_not_evaluated"
        or report.get("pre_registered_protocol", {}).get("test_evaluation_requested")
        is not False
    ):
        raise ValueError("The smoke reference is not strict, locked 13-class PyCBC data")
    if report.get("dataset_configuration") != {
        "train_per_class": 6,
        "validation_per_class": 2,
        "test_per_class": 2,
        "components": N_QUIBITS,
        "pca_fit_components": 64,
    }:
        raise ValueError("The smoke reference dataset configuration changed")
    datasets = {item["seed"]: item for item in report.get("datasets", [])}
    if sorted(datasets) != [42, 43, 44]:
        raise ValueError("The smoke reference must contain seeds 42, 43, and 44")
    return report, hashlib.sha256(content).hexdigest().upper(), datasets


def _validate_resume_report(report, expected_report):
    if report.get("status") != "in_progress":
        raise ValueError("Only an in-progress Q1.1 report can be resumed")
    immutable_keys = (
        "checkpoint_schema_version",
        "experiment_id",
        "experiment_scope",
        "classical_reference_id",
        "classical_reference_sha256",
        "dataset_seed",
        "model_initialization_seeds",
        "training_random_seed",
        "randomness",
        "dataset_hashes",
        "architecture",
        "training_configuration",
        "validation_policy",
        "software_versions",
        "final_test",
    )
    for key in immutable_keys:
        if report.get(key) != expected_report.get(key):
            raise ValueError(f"Cannot resume Q1.1: {key} differs from checkpoint")

    experiments = report.get("experiments", {})
    expected_experiments = expected_report["experiments"]
    for condition in ("C_frozen_ansatz_trainable_readout", "D_trainable_ansatz"):
        runs = experiments.get(condition, {}).get("runs")
        if not isinstance(runs, list):
            raise ValueError(f"Cannot resume Q1.1: invalid {condition} checkpoints")
        seeds = [run.get("initialization_seed") for run in runs]
        if (
            len(seeds) != len(set(seeds))
            or any(seed not in expected_report["model_initialization_seeds"] for seed in seeds)
            or any(run.get("status") != "completed" for run in runs)
            or any(run.get("dataset_seed") != expected_report["dataset_seed"] for run in runs)
            or any(
                run.get("training_random_seed")
                != expected_report["training_random_seed"]
                for run in runs
            )
        ):
            raise ValueError(f"Cannot resume Q1.1: invalid {condition} run state")
    baseline = experiments.get("B_classical_mlp", {})
    if baseline.get("status") not in {"pending", "completed"}:
        raise ValueError("Cannot resume Q1.1: invalid classical baseline state")
    if experiments.get("A_current_qnim_instrumented") != expected_experiments[
        "A_current_qnim_instrumented"
    ]:
        raise ValueError("Cannot resume Q1.1: the A/D alias changed")
    if report.get("validation_policy") != expected_report.get("validation_policy"):
        raise ValueError("Cannot resume Q1.1: validation policy changed")


def _seed_for_diagnostic(dataset_seed, training_seed, iteration, split_index):
    seed = np.random.SeedSequence(
        [dataset_seed, training_seed, iteration, split_index]
    ).generate_state(1)[0]
    return int(seed)


def _classification_metrics(y_true, probabilities):
    raw_labels = np.asarray(y_true)
    if raw_labels.ndim != 1 or not np.array_equal(raw_labels, raw_labels.astype(int)):
        raise ValueError("Labels must be a one-dimensional integer array")
    y_true = raw_labels.astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    n_classes = 13
    if probabilities.shape != (len(y_true), n_classes):
        raise ValueError("Class probabilities do not match the 13-class labels")
    if not np.isfinite(probabilities).all():
        raise ValueError("Class probabilities contain non-finite values")
    if np.any(y_true < 0) or np.any(y_true >= n_classes):
        raise ValueError("Labels must be in the range [0, 12]")
    if np.any(probabilities < 0.0) or not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=1e-7, atol=1e-9
    ):
        raise ValueError("Class probabilities must be normalized")

    predictions = np.argmax(probabilities, axis=1)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(confusion, (y_true, predictions), 1)
    support = confusion.sum(axis=1)
    if np.any(support == 0):
        raise ValueError("Every diagnostic split must contain every class")
    recalls = np.diag(confusion) / support
    selected_probabilities = np.clip(
        probabilities[np.arange(len(y_true)), y_true], 1e-12, 1.0
    )
    return {
        "cross_entropy": float(-np.log(selected_probabilities).mean()),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "class_recall": recalls.tolist(),
        "prediction_counts": np.bincount(
            predictions, minlength=n_classes
        ).tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def _batch_mlp_probabilities(features, W1, b1, W2, b2):
    hidden = np.maximum(0.0, features @ W1.T + b1)
    logits = hidden @ W2.T + b2
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _make_trainable_observer(
    X_train, y_train, X_val, y_val, dataset_seed, shots
):
    from src.infrastructure.qiskit_vqc_trainer import (
        _build_feature_map_and_ansatz,
        _predict_via_circuit,
        _split_readout_mlp,
        chebyshev_preprocess,
        compute_chebyshev_stats,
    )

    combined, x_params, ansatz_params = _build_feature_map_and_ansatz(
        N_QUIBITS,
        n_features=N_QUIBITS,
        feature_map_reps=FEATURE_MAP_REPS,
        ansatz_reps=ANSATZ_REPS,
        entanglement=ENTANGLEMENT,
    )
    combined_run = combined.copy()
    combined_run.measure_all()
    n_ansatz_params = len(ansatz_params)
    train_stats = compute_chebyshev_stats(X_train)
    X_train_cheb = chebyshev_preprocess(X_train, stats=train_stats)
    X_val_cheb = chebyshev_preprocess(X_val, stats=train_stats)
    history = []

    def evaluate(theta, iteration):
        W1, b1, W2, b2 = _split_readout_mlp(
            theta,
            n_ansatz_params,
            N_QUIBITS,
            13,
            READOUT_HIDDEN_SIZE,
        )
        results = {}
        for split_index, (name, features, labels) in enumerate((
            ("train", X_train_cheb, y_train),
            ("validation", X_val_cheb, y_val),
        )):
            probabilities = _predict_via_circuit(
                combined_run,
                x_params,
                ansatz_params,
                theta,
                W1,
                b1,
                W2,
                b2,
                features,
                N_QUIBITS,
                shots=shots,
                batch_size=TRAINING_BATCH_SIZE,
                simulator_seed=_seed_for_diagnostic(
                    dataset_seed, TRAINING_SEED, iteration, split_index
                ),
                return_probabilities=True,
            )
            results[name] = _classification_metrics(labels, probabilities)
        return results

    def observe(iteration, theta, optimizer_batch_loss):
        started = time.perf_counter()
        results = evaluate(theta, iteration)
        results.update({
            "iteration": int(iteration),
            "optimizer_batch_training_loss": float(optimizer_batch_loss),
            "ansatz_parameters": theta[:n_ansatz_params].tolist(),
            "readout_parameters": theta[n_ansatz_params:].tolist(),
            "diagnostic_evaluation_seconds": time.perf_counter() - started,
        })
        history.append(results)

    return observe, evaluate, history


def _run_frozen_ansatz(
    X_train,
    y_train,
    X_val,
    y_val,
    dataset_seed,
    model_seed,
    shots,
    readout_iterations,
):
    from src.infrastructure.qiskit_vqc_trainer import (
        _fit_readout_mlp_classically,
        _flatten_readout_mlp,
        _make_vqc_loss_fn,
    )

    run_started = time.perf_counter()
    loss_fn = _make_vqc_loss_fn(
        X_train=X_train,
        y_train=y_train,
        n_qubits=N_QUIBITS,
        n_features=N_QUIBITS,
        shots=shots,
        mode="sim",
        batch_size_local=TRAINING_BATCH_SIZE,
        feature_map_reps=FEATURE_MAP_REPS,
        ansatz_reps=ANSATZ_REPS,
        entanglement=ENTANGLEMENT,
        readout_hidden_size=READOUT_HIDDEN_SIZE,
        random_seed=TRAINING_SEED,
    )
    ansatz_init = np.random.default_rng(model_seed).normal(
        0.0, 0.01, loss_fn.n_ansatz_params
    )
    ansatz_init.setflags(write=False)
    ansatz_sha256 = hashlib.sha256(ansatz_init.tobytes()).hexdigest().upper()
    feature_extraction_started = time.perf_counter()
    train_features = loss_fn.measure_quantum_features(
        X_train,
        ansatz_init,
        shots_override=shots,
        simulator_seed=_seed_for_diagnostic(
            dataset_seed, TRAINING_SEED, 0, 0
        ),
    )
    validation_features = loss_fn.measure_quantum_features(
        X_val,
        ansatz_init,
        shots_override=shots,
        simulator_seed=_seed_for_diagnostic(
            dataset_seed, TRAINING_SEED, 0, 1
        ),
    )
    feature_extraction_seconds = time.perf_counter() - feature_extraction_started
    history = []

    def observe(iteration, W1, b1, W2, b2):
        train_probabilities = _batch_mlp_probabilities(
            train_features, W1, b1, W2, b2
        )
        validation_probabilities = _batch_mlp_probabilities(
            validation_features, W1, b1, W2, b2
        )
        history.append({
            "iteration": int(iteration),
            "training": _classification_metrics(y_train, train_probabilities),
            "validation": _classification_metrics(y_val, validation_probabilities),
            "readout_parameters": _flatten_readout_mlp(
                W1, b1, W2, b2
            ).tolist(),
        })

    started = time.perf_counter()
    W1, b1, W2, b2 = _fit_readout_mlp_classically(
        train_features,
        y_train,
        13,
        hidden_size=READOUT_HIDDEN_SIZE,
        n_iters=readout_iterations,
        seed=model_seed,
        diagnostic_callback=observe,
    )
    if hashlib.sha256(ansatz_init.tobytes()).hexdigest().upper() != ansatz_sha256:
        raise RuntimeError("Frozen ansatz parameters changed during readout training")
    return {
        "status": "completed",
        "dataset_seed": dataset_seed,
        "initialization_seed": model_seed,
        "training_random_seed": TRAINING_SEED,
        "termination_reason": "readout_iterations_reached",
        "iterations": readout_iterations,
        "feature_extraction_seconds": feature_extraction_seconds,
        "training_time_seconds": time.perf_counter() - started,
        "elapsed_seconds_including_feature_extraction": (
            time.perf_counter() - run_started
        ),
        "ansatz_frozen": True,
        "ansatz_initial_sha256": ansatz_sha256,
        "readout_fit_samples": len(y_train),
        "validation_used_for_readout_fit": False,
        "frozen_ansatz_parameters": ansatz_init.tolist(),
        "final_readout_parameters": _flatten_readout_mlp(
            W1, b1, W2, b2
        ).tolist(),
        "iteration_history": history,
    }


def _run_trainable_ansatz(
    X_train,
    y_train,
    X_val,
    y_val,
    dataset_seed,
    model_seed,
    max_iterations,
    shots,
):
    from src.infrastructure.qiskit_vqc_trainer import QiskitVQCTrainer

    trainer = QiskitVQCTrainer(
        mode="sim",
        random_seed=TRAINING_SEED,
        feature_map_reps=FEATURE_MAP_REPS,
        ansatz_reps=ANSATZ_REPS,
        entanglement=ENTANGLEMENT,
        readout_hidden_size=READOUT_HIDDEN_SIZE,
        training_batch_size=TRAINING_BATCH_SIZE,
        patience=PATIENCE,
        learning_rate=LEARNING_RATE,
        readout_refit_every=READOUT_REFIT_EVERY,
    )
    observer, evaluate, history = _make_trainable_observer(
        X_train,
        y_train,
        X_val,
        y_val,
        dataset_seed,
        shots,
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
        diagnostic_callback=observer,
        initialization_seed=model_seed,
    )
    final_metrics = evaluate(
        training["weights"], int(training["iterations"]) + 1
    )
    return {
        "status": "completed",
        "dataset_seed": dataset_seed,
        "initialization_seed": model_seed,
        "training_random_seed": TRAINING_SEED,
        "termination_reason": training["termination_reason"],
        "iterations": int(training["iterations"]),
        "validation_used_for_optimizer_updates": False,
        "circuit_evaluations": int(training["n_circuit_evaluations"]),
        "training_time_seconds": float(training["execution_time_seconds"]),
        "elapsed_seconds_including_diagnostics": time.perf_counter() - started,
        "final_training_loss": float(training["training_loss"]),
        "final_metrics": final_metrics,
        "iteration_history": history,
    }


def _run_classical_baseline(
    X_train, y_train, X_val, y_val, reference, seed
):
    model = _make_model("mlp", seed)
    model.fit(X_train, y_train)
    train_probabilities = model.predict_proba(X_train)
    validation_probabilities = model.predict_proba(X_val)
    train_metrics = _classification_metrics(y_train, train_probabilities)
    validation_metrics = _classification_metrics(y_val, validation_probabilities)
    if reference is None:
        reproduced = None
    else:
        expected = next(
            item for item in reference["validation"]["models"]["mlp"]["per_seed"]
            if item["seed"] == seed
        )
        reproduced = bool(
            np.isclose(
                validation_metrics["balanced_accuracy"],
                expected["balanced_accuracy"],
                rtol=0.0,
                atol=1e-12,
            )
        )
        if not reproduced:
            raise RuntimeError(
                "The classical MLP did not reproduce its frozen reference score"
            )
    return {
        "status": "completed",
        "dataset_seed": seed,
        "initialization_seed": seed,
        "training": train_metrics,
        "validation": validation_metrics,
        "training_loss_curve": list(model.named_steps["mlpclassifier"].loss_curve_),
        "matches_frozen_reference": reproduced,
    }


def run_q1_1(
    reference_manifest=DEFAULT_REFERENCE,
    output_path=DEFAULT_OUTPUT,
    max_iterations=None,
    shots=None,
    readout_iterations=None,
    adapter_factory=None,
    resume=False,
    smoke=False,
    smoke_reference_path=DEFAULT_SMOKE_REFERENCE,
):
    output_path = Path(output_path)
    if smoke and output_path == DEFAULT_OUTPUT:
        output_path = DEFAULT_SMOKE_OUTPUT
    max_iterations = max_iterations if max_iterations is not None else (
        1 if smoke else 100
    )
    shots = shots if shots is not None else (8 if smoke else 512)
    readout_iterations = readout_iterations if readout_iterations is not None else (
        2 if smoke else 1500
    )
    if output_path.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite Q1.1 results: {output_path}")
    if resume and not output_path.exists():
        raise FileNotFoundError(f"Q1.1 checkpoint does not exist: {output_path}")
    if min(max_iterations, shots, readout_iterations) < 1:
        raise ValueError("Iteration and shot counts must be positive")

    if smoke:
        reference, reference_sha256, reference_datasets = _read_smoke_reference(
            smoke_reference_path
        )
        reference_id = "CLASSICAL_DIAGNOSTIC_SMOKE_V2"
        model_seeds = (42,)
    else:
        reference, reference_sha256 = _read_reference(Path(reference_manifest))
        reference_datasets = {
            item["seed"]: item for item in reference["datasets"]
        }
        if sorted(reference_datasets) != [42, 43, 44]:
            raise ValueError(
                "Classical reference datasets do not match the fixed seeds"
            )
        reference_id = "CLASSICAL_REFERENCE_V1"
        model_seeds = MODEL_SEEDS
    expected_dataset = reference_datasets[DATASET_SEED]
    config = reference["dataset_configuration"]
    if adapter_factory is None:
        from src.infrastructure.ligo_pca_adapter import LIGOPCAAdapter

        adapter_factory = LIGOPCAAdapter
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
        seed=DATASET_SEED,
        verbose=True,
    )
    X_train, y_train, X_val, y_val = _validate_dataset(
        dataset, expected_dataset, config
    )
    architecture = _architecture_metadata()
    architecture["diagnostic_evaluation"] = (
        "Full train and validation partitions at every optimizer iteration; "
        "validation metrics are observational only."
    )

    report = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "experiment_id": "QNIM_Q1_1_SMOKE_V1" if smoke else "QNIM_Q1_1_V1",
        "experiment_scope": (
            "integration smoke only; not scientific evidence"
            if smoke else "frozen-reference diagnostic"
        ),
        "status": "in_progress",
        "active_run": None,
        "classical_reference_id": reference_id,
        "classical_reference_sha256": reference_sha256,
        "dataset_seed": DATASET_SEED,
        "model_initialization_seeds": list(model_seeds),
        "training_random_seed": TRAINING_SEED,
        "randomness": (
            f"The dataset seed is 42. Initialization seeds are {list(model_seeds)}; "
            "training randomness (optimizer, batches, and training sampler) is "
            "fixed at 42. Diagnostic simulator seeds are derived from the "
            "dataset seed, fixed training seed, iteration, and split."
        ),
        "dataset_hashes": {
            "train_sha256": expected_dataset["train_sha256"],
            "validation_sha256": expected_dataset["validation_sha256"],
        },
        "architecture": architecture,
        "training_configuration": {
            "mode": "sim",
            "smoke_mode": smoke,
            "shots": shots,
            "max_iterations": max_iterations,
            "patience": PATIENCE,
            "training_batch_size": TRAINING_BATCH_SIZE,
            "readout_iterations": readout_iterations,
            "readout_refit_every": READOUT_REFIT_EVERY,
        },
        "validation_policy": {
            "used_for_parameter_updates": False,
            "metrics_recorded_after_optimizer_steps": True,
            "final_test": "locked_not_evaluated",
        },
        "experiments": {
            "A_current_qnim_instrumented": {
                "same_run_as": "D_trainable_ansatz.initialization_seed_42",
                "note": "A and D are the same trainable architecture; no duplicate fit is run."
            },
            "B_classical_mlp": {
                "status": "pending",
                "source": (
                    "Smoke split; no frozen-score comparison."
                    if smoke else
                    "Re-fit on the same seed-42 split and compare with CLASSICAL_REFERENCE_V1."
                ),
            },
            "C_frozen_ansatz_trainable_readout": {
                "runs": [],
                "readout_training": (
                    "Full train partition; the circuit ansatz is initialized by model seed "
                    "and held fixed; validation is diagnostic only."
                ),
            },
            "D_trainable_ansatz": {
                "runs": [],
                "readout_training": (
                    "Existing QNIM warm-start and periodic refit; validation is "
                    "passed only to the non-mutating diagnostic observer."
                ),
            },
        },
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "PyCBC": importlib.metadata.version("PyCBC"),
            "LALSuite": importlib.metadata.version("lalsuite"),
            "qiskit": importlib.metadata.version("qiskit"),
            "qiskit_aer": importlib.metadata.version("qiskit-aer"),
        },
        "final_test": {
            "evaluated": False,
            "status": "locked_not_evaluated",
            "models": {},
        },
    }

    if resume:
        expected_report = report
        report = json.loads(output_path.read_text(encoding="utf-8"))
        _validate_resume_report(report, expected_report)
        report["active_run"] = None
        _write_json_atomically(output_path, report)
    else:
        _write_json_atomically(output_path, report)

    if report["experiments"]["B_classical_mlp"].get("status") != "completed":
        report["active_run"] = {
            "condition": "B_classical_mlp",
            "initialization_seed": DATASET_SEED,
        }
        _write_json_atomically(output_path, report)
        report["experiments"]["B_classical_mlp"] = _run_classical_baseline(
            X_train,
            y_train,
            X_val,
            y_val,
            None if smoke else reference,
            DATASET_SEED,
        )
        report["active_run"] = None
        _write_json_atomically(output_path, report)

    for model_seed in model_seeds:
        completed = {
            run["initialization_seed"]
            for run in report["experiments"]["D_trainable_ansatz"]["runs"]
        }
        if model_seed in completed:
            continue
        report["active_run"] = {
            "condition": "D_trainable_ansatz",
            "initialization_seed": model_seed,
        }
        _write_json_atomically(output_path, report)
        run = _run_trainable_ansatz(
            X_train,
            y_train,
            X_val,
            y_val,
            DATASET_SEED,
            model_seed,
            max_iterations,
            shots,
        )
        report["experiments"]["D_trainable_ansatz"]["runs"].append(run)
        report["active_run"] = None
        _write_json_atomically(output_path, report)

    for model_seed in model_seeds:
        completed = {
            run["initialization_seed"]
            for run in report["experiments"][
                "C_frozen_ansatz_trainable_readout"
            ]["runs"]
        }
        if model_seed in completed:
            continue
        report["active_run"] = {
            "condition": "C_frozen_ansatz_trainable_readout",
            "initialization_seed": model_seed,
        }
        _write_json_atomically(output_path, report)
        run = _run_frozen_ansatz(
            X_train,
            y_train,
            X_val,
            y_val,
            DATASET_SEED,
            model_seed,
            shots,
            readout_iterations,
        )
        report["experiments"][
            "C_frozen_ansatz_trainable_readout"
        ]["runs"].append(run)
        report["active_run"] = None
        _write_json_atomically(output_path, report)

    report["status"] = "completed"
    _write_json_atomically(output_path, report)
    return report


def benchmark_diagnostic_iteration(
    reference_manifest=DEFAULT_REFERENCE,
    output_path=DEFAULT_BENCHMARK_OUTPUT,
    shots=512,
    adapter_factory=None,
):
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite diagnostic benchmark: {output_path}"
        )
    if shots < 1:
        raise ValueError("shots must be positive")

    reference, reference_sha256 = _read_reference(Path(reference_manifest))
    expected_dataset = next(
        item for item in reference["datasets"] if item["seed"] == DATASET_SEED
    )
    config = reference["dataset_configuration"]
    if adapter_factory is None:
        from src.infrastructure.ligo_pca_adapter import LIGOPCAAdapter

        adapter_factory = LIGOPCAAdapter
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
        seed=DATASET_SEED,
        verbose=True,
    )
    X_train, y_train, X_val, y_val = _validate_dataset(
        dataset, expected_dataset, config
    )
    observer, _, history = _make_trainable_observer(
        X_train, y_train, X_val, y_val, DATASET_SEED, shots
    )
    architecture = _architecture_metadata()
    from src.infrastructure.qiskit_vqc_trainer import _build_feature_map_and_ansatz

    _, _, ansatz_params = _build_feature_map_and_ansatz(
        N_QUIBITS,
        n_features=N_QUIBITS,
        feature_map_reps=FEATURE_MAP_REPS,
        ansatz_reps=ANSATZ_REPS,
        entanglement=ENTANGLEMENT,
    )
    theta = np.zeros(architecture["total_trainable_parameters"], dtype=float)
    theta[:len(ansatz_params)] = np.random.default_rng(DATASET_SEED).normal(
        0.0, 0.01, len(ansatz_params)
    )

    benchmark_started = time.perf_counter()
    observer(1, theta, 0.0)
    elapsed = time.perf_counter() - benchmark_started
    report = {
        "experiment_id": "QNIM_Q1_1_DIAGNOSTIC_COST_V1",
        "status": "completed",
        "classical_reference_id": "CLASSICAL_REFERENCE_V1",
        "classical_reference_sha256": reference_sha256,
        "dataset_seed": DATASET_SEED,
        "dataset_hashes": {
            "train_sha256": expected_dataset["train_sha256"],
            "validation_sha256": expected_dataset["validation_sha256"],
        },
        "n_train_examples": len(y_train),
        "n_validation_examples": len(y_val),
        "n_diagnostic_examples": len(y_train) + len(y_val),
        "shots": shots,
        "batch_size": TRAINING_BATCH_SIZE,
        "architecture": architecture,
        "diagnostic_evaluation_seconds": history[0][
            "diagnostic_evaluation_seconds"
        ],
        "elapsed_seconds": elapsed,
        "metric_snapshot": {
            split: {
                "cross_entropy": history[0][split]["cross_entropy"],
                "balanced_accuracy": history[0][split]["balanced_accuracy"],
            }
            for split in ("train", "validation")
        },
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "qiskit": importlib.metadata.version("qiskit"),
            "qiskit_aer": importlib.metadata.version("qiskit-aer"),
        },
        "final_test": {
            "evaluated": False,
            "status": "locked_not_evaluated",
            "models": {},
        },
    }
    _write_json_atomically(output_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Instrument the frozen QNIM architecture and run Q1.1 ablations."
    )
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--shots", type=int)
    parser.add_argument("--readout-iterations", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--benchmark-diagnostic-iteration", action="store_true")
    parser.add_argument(
        "--benchmark-output", type=Path, default=DEFAULT_BENCHMARK_OUTPUT
    )
    parser.add_argument(
        "--smoke-reference", type=Path, default=DEFAULT_SMOKE_REFERENCE
    )
    args = parser.parse_args()
    if args.benchmark_diagnostic_iteration:
        if args.smoke or args.resume:
            parser.error(
                "--benchmark-diagnostic-iteration cannot be combined with --smoke or --resume"
            )
        report = benchmark_diagnostic_iteration(
            shots=args.shots if args.shots is not None else 512,
            output_path=args.benchmark_output,
        )
        print(json.dumps({
            "status": report["status"],
            "output": str(args.benchmark_output),
            "diagnostic_evaluation_seconds": report[
                "diagnostic_evaluation_seconds"
            ],
            "n_diagnostic_examples": report["n_diagnostic_examples"],
        }, indent=2))
        return
    if args.output is None:
        args.output = DEFAULT_SMOKE_OUTPUT if args.smoke else DEFAULT_OUTPUT
    report = run_q1_1(
        max_iterations=args.max_iterations,
        shots=args.shots,
        readout_iterations=args.readout_iterations,
        output_path=args.output,
        resume=args.resume,
        smoke=args.smoke,
        smoke_reference_path=args.smoke_reference,
    )
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "final_test": report["final_test"]["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
