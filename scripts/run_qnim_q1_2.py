"""Run matched Q1.2 ablations on the frozen seed-42 train/validation split."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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
from scripts.run_qnim_q1_1 import (
    ANSATZ_REPS,
    DATASET_SEED,
    DEFAULT_SMOKE_REFERENCE,
    ENTANGLEMENT,
    FEATURE_MAP_REPS,
    LEARNING_RATE,
    MODEL_SEEDS,
    N_FEYNMAN_PARAMS,
    READOUT_REFIT_EVERY,
    TRAINING_SEED,
    _batch_mlp_probabilities,
    _classification_metrics,
    _read_smoke_reference,
    _seed_for_diagnostic,
    _make_trainable_observer,
)

DEFAULT_OUTPUT = ROOT / "results" / "scaling_study" / "qnim_q1_2_v1.json"
DEFAULT_SMOKE_OUTPUT = ROOT / "results" / "scaling_study" / "qnim_q1_2_smoke_v1.json"
SCHEMA_VERSION = 1
MIN_PATIENCE_IMPROVEMENT = 5e-4
PATIENCE = 20


def _read_dataset(smoke, reference_manifest, smoke_reference_path, adapter_factory):
    if smoke:
        reference, reference_sha256, datasets = _read_smoke_reference(
            smoke_reference_path
        )
        reference_id = "CLASSICAL_DIAGNOSTIC_SMOKE_V2"
        seeds = (42,)
    else:
        reference, reference_sha256 = _read_reference(Path(reference_manifest))
        datasets = {item["seed"]: item for item in reference["datasets"]}
        if sorted(datasets) != [42, 43, 44]:
            raise ValueError("Classical reference datasets do not match seeds 42, 43, 44")
        reference_id = "CLASSICAL_REFERENCE_V1"
        seeds = MODEL_SEEDS

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
    split = _validate_dataset(dataset, datasets[DATASET_SEED], config)
    return reference, reference_sha256, reference_id, datasets[DATASET_SEED], split, seeds


def _build_loss(X_train, y_train, shots):
    from src.infrastructure.qiskit_vqc_trainer import _make_vqc_loss_fn

    return _make_vqc_loss_fn(
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


def _initial_parameters(X_train, y_train, model_seed, shots):
    from src.infrastructure.qiskit_vqc_trainer import _flatten_readout_mlp

    loss_fn = _build_loss(X_train, y_train, shots)
    ansatz = np.random.default_rng(model_seed).normal(
        0.0, 0.01, loss_fn.n_ansatz_params
    )
    readout = loss_fn.warm_start_readout(ansatz)
    parameters = np.concatenate([ansatz, _flatten_readout_mlp(*readout)])
    if not np.isfinite(parameters).all():
        raise ValueError("Common Q1.2 initialization contains non-finite parameters")
    return parameters, loss_fn.n_ansatz_params


def _sha256(array):
    return hashlib.sha256(np.asarray(array, dtype=np.float64).tobytes()).hexdigest().upper()


def _run_exact_vs_shots(
    X_train, y_train, X_val, y_val, model_seed, shots, readout_iterations
):
    from src.infrastructure.qiskit_vqc_trainer import (
        _fit_readout_mlp_classically,
    )

    started = time.perf_counter()
    loss_fn = _build_loss(X_train, y_train, shots)
    ansatz = np.random.default_rng(model_seed).normal(
        0.0, 0.01, loss_fn.n_ansatz_params
    )
    features = {}
    timings = {}
    for mode in ("exact", "shots"):
        mode_started = time.perf_counter()
        if mode == "exact":
            measure = loss_fn.measure_quantum_features_exact
            train_features = measure(X_train, ansatz)
            validation_features = measure(X_val, ansatz)
        else:
            train_features = loss_fn.measure_quantum_features(
                X_train,
                ansatz,
                shots_override=shots,
                simulator_seed=_seed_for_diagnostic(
                    DATASET_SEED, TRAINING_SEED, 1, 0
                ),
            )
            validation_features = loss_fn.measure_quantum_features(
                X_val,
                ansatz,
                shots_override=shots,
                simulator_seed=_seed_for_diagnostic(
                    DATASET_SEED, TRAINING_SEED, 1, 1
                ),
            )
        features[mode] = (train_features, validation_features)
        timings[mode] = time.perf_counter() - mode_started

    results = {}
    for mode in ("exact", "shots"):
        train_features, validation_features = features[mode]
        fit_started = time.perf_counter()
        W1, b1, W2, b2 = _fit_readout_mlp_classically(
            train_features,
            y_train,
            n_classes=13,
            hidden_size=READOUT_HIDDEN_SIZE,
            n_iters=readout_iterations,
            seed=model_seed,
        )
        train_probs = _batch_mlp_probabilities(train_features, W1, b1, W2, b2)
        validation_probs = _batch_mlp_probabilities(
            validation_features, W1, b1, W2, b2
        )
        results[mode] = {
            "training": _classification_metrics(y_train, train_probs),
            "validation": _classification_metrics(y_val, validation_probs),
            "readout_fit_seconds": time.perf_counter() - fit_started,
            "readout_fit_samples": len(y_train),
            "validation_used_for_readout_fit": False,
        }

    exact_train, exact_val = features["exact"]
    shots_train, shots_val = features["shots"]
    results["feature_comparison"] = {
        "train_rmse": float(np.sqrt(np.mean((exact_train - shots_train) ** 2))),
        "validation_rmse": float(np.sqrt(np.mean((exact_val - shots_val) ** 2))),
        "train_max_absolute_error": float(np.max(np.abs(exact_train - shots_train))),
        "validation_max_absolute_error": float(
            np.max(np.abs(exact_val - shots_val))
        ),
    }
    return {
        "status": "completed",
        "dataset_seed": DATASET_SEED,
        "initialization_seed": model_seed,
        "training_random_seed": TRAINING_SEED,
        "ansatz_sha256": _sha256(ansatz),
        "shots": shots,
        "feature_extraction_seconds": timings,
        "elapsed_seconds": time.perf_counter() - started,
        "results": results,
    }


def _would_stop_by_patience(reference_history):
    best_loss = float("inf")
    patience_count = 0
    for index, row in enumerate(reference_history, start=1):
        loss = row["training_reference_loss"]
        if loss < best_loss - MIN_PATIENCE_IMPROVEMENT:
            best_loss = loss
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= PATIENCE:
            return index
    return None


def _run_matched_ablation(
    X_train,
    y_train,
    X_val,
    y_val,
    model_seed,
    initial_parameters,
    n_ansatz_params,
    updates,
    shots,
    trainable,
):
    from src.infrastructure.qiskit_vqc_trainer import QiskitVQCTrainer

    theta0 = np.asarray(initial_parameters, dtype=float).copy()
    initial_hash = _sha256(theta0)
    ansatz_initial_hash = _sha256(theta0[:n_ansatz_params])
    readout_initial_hash = _sha256(theta0[n_ansatz_params:])
    snapshots = []
    previous = theta0.copy()

    def observe(iteration, theta, loss):
        nonlocal previous
        current = np.asarray(theta, dtype=float)
        step = current - previous
        snapshots.append({
            "iteration": int(iteration),
            "optimizer_batch_loss": float(loss),
            "callback_parameter_delta_norm": float(np.linalg.norm(step)),
            "callback_ansatz_delta_norm": float(
                np.linalg.norm(step[:n_ansatz_params])
            ),
            "callback_readout_delta_norm": float(
                np.linalg.norm(step[n_ansatz_params:])
            ),
            "ansatz_parameters": current[:n_ansatz_params].tolist(),
            "readout_parameter_norm": float(
                np.linalg.norm(current[n_ansatz_params:])
            ),
        })
        previous = current.copy()

    trainer = QiskitVQCTrainer(
        mode="sim",
        random_seed=TRAINING_SEED,
        feature_map_reps=FEATURE_MAP_REPS,
        ansatz_reps=ANSATZ_REPS,
        entanglement=ENTANGLEMENT,
        readout_hidden_size=READOUT_HIDDEN_SIZE,
        training_batch_size=TRAINING_BATCH_SIZE,
        patience=updates + 1,
        learning_rate=LEARNING_RATE,
        readout_refit_every=READOUT_REFIT_EVERY,
    )
    training = trainer.train_vqc(
        X_train=X_train,
        y_train=y_train,
        num_qubits=N_QUIBITS,
        n_features=N_QUIBITS,
        max_iterations=updates,
        shots=shots,
        n_feynman_params=N_FEYNMAN_PARAMS,
        initialization_seed=model_seed,
        initial_parameters=theta0,
        freeze_ansatz=not trainable,
        diagnostic_callback=observe,
    )
    final_weights = np.asarray(training["weights"], dtype=float)
    if not trainable and not np.array_equal(
        final_weights[:n_ansatz_params], theta0[:n_ansatz_params]
    ):
        raise RuntimeError("The frozen-ansatz optimization changed ansatz parameters")

    _, evaluate, _ = _make_trainable_observer(
        X_train, y_train, X_val, y_val, DATASET_SEED, shots
    )
    final_metrics = evaluate(final_weights, updates + 1)
    optimizer_history = training["optimizer_step_history"]
    observed_updates = int(training["iterations"])
    if observed_updates != updates:
        raise RuntimeError(
            f"Matched run performed {observed_updates} updates; expected {updates}"
        )
    return {
        "status": "completed",
        "dataset_seed": DATASET_SEED,
        "initialization_seed": model_seed,
        "training_random_seed": TRAINING_SEED,
        "ansatz_trainable": trainable,
        "updates": observed_updates,
        "termination_reason": training["termination_reason"],
        "initial_parameters_sha256": initial_hash,
        "initial_ansatz_sha256": ansatz_initial_hash,
        "initial_readout_sha256": readout_initial_hash,
        "final_ansatz_sha256": _sha256(final_weights[:n_ansatz_params]),
        "final_ansatz_parameters": final_weights[:n_ansatz_params].tolist(),
        "frozen_ansatz_verified": not trainable and np.array_equal(
            final_weights[:n_ansatz_params], theta0[:n_ansatz_params]
        ),
        "ansatz_total_change_norm": float(
            np.linalg.norm(
                final_weights[:n_ansatz_params] - theta0[:n_ansatz_params]
            )
        ),
        "final_metrics": final_metrics,
        "execution_time_seconds": float(training["execution_time_seconds"]),
        "circuit_evaluations": int(training["n_circuit_evaluations"]),
        "optimizer_step_history": optimizer_history,
        "parameter_snapshots": snapshots,
        "would_stop_by_original_patience_at": _would_stop_by_patience(
            optimizer_history
        ),
    }


def _validate_checkpoint(report, expected):
    if report.get("status") != "in_progress":
        raise ValueError("Only an in-progress Q1.2 report can be resumed")
    for key in (
        "checkpoint_schema_version",
        "experiment_id",
        "classical_reference_id",
        "classical_reference_sha256",
        "dataset_seed",
        "model_initialization_seeds",
        "training_random_seed",
        "dataset_hashes",
        "class_names",
        "architecture",
        "configuration",
        "software_versions",
        "final_test",
    ):
        if report.get(key) != expected.get(key):
            raise ValueError(f"Cannot resume Q1.2: {key} differs from checkpoint")
    if report.get("experiments", {}).get("Q1_2_A", {}).get(
        "optimizer_controls"
    ) != expected.get("experiments", {}).get("Q1_2_A", {}).get(
        "optimizer_controls"
    ):
        raise ValueError("Cannot resume Q1.2: optimizer controls changed")
    for condition in ("frozen_ansatz", "trainable_ansatz"):
        runs = report.get("experiments", {}).get("Q1_2_A", {}).get(condition)
        if not isinstance(runs, list):
            raise ValueError(f"Cannot resume Q1.2: invalid {condition} checkpoints")
        seeds = [run.get("initialization_seed") for run in runs]
        if (
            len(seeds) != len(set(seeds))
            or any(seed not in expected["model_initialization_seeds"] for seed in seeds)
            or any(run.get("status") != "completed" for run in runs)
        ):
            raise ValueError(f"Cannot resume Q1.2: invalid {condition} state")
    measurement = report.get("experiments", {}).get("Q1_2_B", {})
    if measurement.get("status") not in {"pending", "completed"}:
        raise ValueError("Cannot resume Q1.2: invalid exact/shots state")


def run_q1_2(
    reference_manifest=DEFAULT_REFERENCE,
    output_path=DEFAULT_OUTPUT,
    updates=None,
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
    updates = updates if updates is not None else (2 if smoke else 60)
    shots = shots if shots is not None else (8 if smoke else 512)
    readout_iterations = (
        readout_iterations if readout_iterations is not None else
        (30 if smoke else 1500)
    )
    if min(updates, shots, readout_iterations) < 1:
        raise ValueError("updates, shots, and readout_iterations must be positive")
    if output_path.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite Q1.2 results: {output_path}")
    if resume and not output_path.exists():
        raise FileNotFoundError(f"Q1.2 checkpoint does not exist: {output_path}")

    reference, reference_sha256, reference_id, expected_dataset, split, seeds = (
        _read_dataset(
            smoke, reference_manifest, smoke_reference_path, adapter_factory
        )
    )
    X_train, y_train, X_val, y_val = split
    architecture = _architecture_metadata()
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "PyCBC": importlib.metadata.version("PyCBC"),
        "LALSuite": importlib.metadata.version("lalsuite"),
        "qiskit": importlib.metadata.version("qiskit"),
        "qiskit_aer": importlib.metadata.version("qiskit-aer"),
    }
    configuration = {
        "smoke_mode": smoke,
        "updates": updates,
        "shots": shots,
        "readout_iterations": readout_iterations,
        "training_batch_size": TRAINING_BATCH_SIZE,
        "readout_refit_every": READOUT_REFIT_EVERY,
        "learning_rate": LEARNING_RATE,
        "patience_during_matched_ablation": updates + 1,
        "original_patience_for_replay": PATIENCE,
        "minimum_improvement_for_patience_replay": MIN_PATIENCE_IMPROVEMENT,
        "ansatz_trainable_feynman_parameters": N_FEYNMAN_PARAMS,
    }
    report = {
        "checkpoint_schema_version": SCHEMA_VERSION,
        "experiment_id": "QNIM_Q1_2_SMOKE_V1" if smoke else "QNIM_Q1_2_V1",
        "experiment_scope": (
            "integration smoke only; not scientific evidence"
            if smoke else "matched diagnostic on frozen synthetic data"
        ),
        "status": "in_progress",
        "active_run": None,
        "classical_reference_id": reference_id,
        "classical_reference_sha256": reference_sha256,
        "dataset_seed": DATASET_SEED,
        "model_initialization_seeds": list(seeds),
        "training_random_seed": TRAINING_SEED,
        "dataset_hashes": {
            "train_sha256": expected_dataset["train_sha256"],
            "validation_sha256": expected_dataset["validation_sha256"],
        },
        "class_names": list(expected_dataset["class_names"]),
        "architecture": architecture,
        "configuration": configuration,
        "software_versions": versions,
        "validation_policy": {
            "used_for_training_or_early_stopping": False,
            "used_for_final_metrics_only": True,
            "final_test": "locked_not_evaluated",
        },
        "experiments": {
            "Q1_2_A": {
                "description": (
                    "Paired trainable vs frozen ansatz, with identical full "
                    "initial vectors, training data seed, optimizer seed, "
                    "minibatch sequence, update count, readout warm-start, and "
                    "readout refit schedule."
                ),
                "optimizer_controls": {
                    "algorithm": "QNSPSA-EML-Feynman",
                    "optimizer_seed": TRAINING_SEED,
                    "fixed_minibatch_sequence": True,
                    "fixed_update_count": updates,
                    "frozen_run_active_parameters": "classical readout only",
                    "trainable_run_feynman_parameters": N_FEYNMAN_PARAMS,
                },
                "frozen_ansatz": [],
                "trainable_ansatz": [],
            },
            "Q1_2_B": {
                "status": "pending",
                "description": (
                    "Exact statevector and finite-shot marginals for the same "
                    "seed-42 ansatz and train/validation examples."
                ),
            },
            "Q1_2_C": {
                "source": "Optimizer telemetry from the matched Q1_2_A trainable runs.",
                "description": (
                    "Record minibatch and fixed-train-reference losses, update "
                    "norms, gradient variance, accepted steps, and replay the "
                    "original patience rule without affecting matched updates."
                ),
            },
        },
        "final_test": {
            "evaluated": False,
            "status": "locked_not_evaluated",
            "models": {},
        },
    }

    if resume:
        expected = report
        report = json.loads(output_path.read_text(encoding="utf-8"))
        _validate_checkpoint(report, expected)
        report["active_run"] = None
        _write_json_atomically(output_path, report)
    else:
        _write_json_atomically(output_path, report)

    if report["experiments"]["Q1_2_B"].get("status") != "completed":
        report["active_run"] = {"condition": "Q1_2_B", "seed": DATASET_SEED}
        _write_json_atomically(output_path, report)
        report["experiments"]["Q1_2_B"] = _run_exact_vs_shots(
            X_train,
            y_train,
            X_val,
            y_val,
            DATASET_SEED,
            shots,
            readout_iterations,
        )
        report["active_run"] = None
        _write_json_atomically(output_path, report)

    for model_seed in seeds:
        existing_frozen = {
            run["initialization_seed"]
            for run in report["experiments"]["Q1_2_A"]["frozen_ansatz"]
        }
        existing_trainable = {
            run["initialization_seed"]
            for run in report["experiments"]["Q1_2_A"]["trainable_ansatz"]
        }
        if model_seed in existing_frozen and model_seed in existing_trainable:
            continue

        initial, n_ansatz_params = _initial_parameters(
            X_train, y_train, model_seed, shots
        )
        for trainable, condition in (
            (False, "frozen_ansatz"),
            (True, "trainable_ansatz"),
        ):
            completed = {
                run["initialization_seed"]
                for run in report["experiments"]["Q1_2_A"][condition]
            }
            if model_seed in completed:
                continue
            report["active_run"] = {
                "condition": condition,
                "seed": model_seed,
            }
            _write_json_atomically(output_path, report)
            run = _run_matched_ablation(
                X_train,
                y_train,
                X_val,
                y_val,
                model_seed,
                initial,
                n_ansatz_params,
                updates,
                shots,
                trainable,
            )
            report["experiments"]["Q1_2_A"][condition].append(run)
            report["active_run"] = None
            _write_json_atomically(output_path, report)

    frozen = {
        run["initialization_seed"]: run
        for run in report["experiments"]["Q1_2_A"]["frozen_ansatz"]
    }
    trainable = {
        run["initialization_seed"]: run
        for run in report["experiments"]["Q1_2_A"]["trainable_ansatz"]
    }
    report["experiments"]["Q1_2_A"]["paired_summary"] = [
        {
            "initialization_seed": seed,
            "initial_vectors_match": (
                frozen[seed]["initial_parameters_sha256"]
                == trainable[seed]["initial_parameters_sha256"]
            ),
            "frozen_validation_balanced_accuracy": (
                frozen[seed]["final_metrics"]["validation"]["balanced_accuracy"]
            ),
            "trainable_validation_balanced_accuracy": (
                trainable[seed]["final_metrics"]["validation"]["balanced_accuracy"]
            ),
            "trainable_minus_frozen_balanced_accuracy": (
                trainable[seed]["final_metrics"]["validation"]["balanced_accuracy"]
                - frozen[seed]["final_metrics"]["validation"]["balanced_accuracy"]
            ),
            "would_stop_by_original_patience_at": (
                trainable[seed]["would_stop_by_original_patience_at"]
            ),
        }
        for seed in seeds
    ]
    if any(not row["initial_vectors_match"] for row in
           report["experiments"]["Q1_2_A"]["paired_summary"]):
        raise RuntimeError("Matched frozen/trainable initial parameter vectors differ")
    report["status"] = "completed"
    _write_json_atomically(output_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Run matched Q1.2 ansatz, measurement, and optimizer diagnostics."
    )
    parser.add_argument("--updates", type=int)
    parser.add_argument("--shots", type=int)
    parser.add_argument("--readout-iterations", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-reference", type=Path, default=DEFAULT_SMOKE_REFERENCE)
    args = parser.parse_args()
    output = args.output or (DEFAULT_SMOKE_OUTPUT if args.smoke else DEFAULT_OUTPUT)
    report = run_q1_2(
        output_path=output,
        updates=args.updates,
        shots=args.shots,
        readout_iterations=args.readout_iterations,
        resume=args.resume,
        smoke=args.smoke,
        smoke_reference_path=args.smoke_reference,
    )
    print(json.dumps({
        "status": report["status"],
        "output": str(output),
        "final_test": report["final_test"]["status"],
        "paired_runs": len(report["experiments"]["Q1_2_A"]["paired_summary"]),
    }, indent=2))


if __name__ == "__main__":
    main()
