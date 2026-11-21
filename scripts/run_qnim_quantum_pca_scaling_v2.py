"""Independent QNIM q16/PCA16 V2 runner; dry-run never builds or runs a circuit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.infrastructure.qiskit_vqc_trainer import _make_vqc_loss_fn
from src.infrastructure.qnim_v2_support import (
    atomic_save_npz,
    atomic_write_json,
    candidate_statistics,
    compare_candidates,
    indices_sha256,
    measurement_artifact_arrays,
    measurement_moments,
    mlp_loss_and_feature_gradient,
    parameters_sha256,
    probability_artifact_arrays,
    probability_diagnostics,
    stratified_disjoint_roles,
    validate_disjoint_roles,
)
from src.infrastructure.qnspsa_eml_feynman import QNSPSAConfig, QNSPSAEMLFeynman

PROTOCOL_PATH = (
    ROOT
    / "results"
    / "scaling_study"
    / "qnim_quantum_pca_scaling_v2"
    / "protocol.json"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_verified_inputs() -> tuple[dict, dict[str, np.ndarray], dict[str, Any]]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol_hash = _canonical_hash(protocol)
    dataset = protocol["dataset"]
    dataset_dir = (PROTOCOL_PATH.parent / dataset["relative_path"]).resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    expected_hashes = dict(dataset["files"])
    expected_hashes["dataset_manifest.json"] = dataset["dataset_manifest_sha256"]
    for filename, expected in expected_hashes.items():
        actual = _sha256_file(dataset_dir / filename)
        if actual != expected:
            raise ValueError(f"Dataset integrity check failed for {filename}")

    arrays = {}
    for filename, key in (
        ("X_train_scaled64.npy", "X_train"),
        ("X_val_scaled64.npy", "X_val"),
        ("y_train.npy", "y_train"),
        ("y_val.npy", "y_val"),
    ):
        arrays[key] = np.load(dataset_dir / filename, allow_pickle=False)

    if arrays["X_train"].shape != (2600, 64):
        raise ValueError("V2 requires the preserved 2600 x 64 training matrix")
    if arrays["X_val"].shape != (650, 64):
        raise ValueError("V2 requires the preserved 650 x 64 validation matrix")
    if arrays["y_train"].shape != (2600,) or arrays["y_val"].shape != (650,):
        raise ValueError("Dataset label arrays have unexpected shapes")
    if not np.array_equal(np.unique(arrays["y_train"]), np.arange(13)):
        raise ValueError("Training labels must be ordered integer IDs 0 through 12")
    if not np.array_equal(np.unique(arrays["y_val"]), np.arange(13)):
        raise ValueError("Validation labels must be ordered integer IDs 0 through 12")
    if not np.all(np.bincount(arrays["y_train"], minlength=13) == 200):
        raise ValueError("Training split must retain 200 examples per class")
    if not np.all(np.bincount(arrays["y_val"], minlength=13) == 50):
        raise ValueError("Validation split must retain 50 examples per class")

    roles = stratified_disjoint_roles(arrays["y_train"], seed=42)
    validate_disjoint_roles(arrays["y_train"], roles)
    train_diagnostics = roles["reference"].copy()
    rng = np.random.default_rng(42042)
    validation_diagnostics = np.sort(
        np.concatenate(
            [
                rng.permutation(np.flatnonzero(arrays["y_val"] == class_id))[:5]
                for class_id in range(13)
            ]
        )
    ).astype(np.int64)
    data_hashes = {
        filename: expected_hashes[filename]
        for filename in expected_hashes
    }
    return protocol, arrays, {
        "protocol_sha256": protocol_hash,
        "dataset_hashes": data_hashes,
        "roles": roles,
        "validation_diagnostic_indices": validation_diagnostics,
        "training_diagnostic_indices": train_diagnostics,
    }


def _new_attempt_dir(base: Path, label: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = base / "attempts" / f"{label}_{timestamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_dry_run(
    protocol: dict,
    arrays: dict[str, np.ndarray],
    verified: dict[str, Any],
) -> Path:
    output_dir = _new_attempt_dir(PROTOCOL_PATH.parent, "dry_run")
    roles = verified["roles"]
    atomic_save_npz(
        output_dir / "training_roles.npz",
        {name: indices for name, indices in roles.items()},
    )
    atomic_save_npz(
        output_dir / "validation_diagnostic_indices.npz",
        {"indices": verified["validation_diagnostic_indices"]},
    )
    manifest = {
        "status": "dry_run_passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": verified["protocol_sha256"],
        "dataset_hashes": verified["dataset_hashes"],
        "training_role_sizes": {name: len(indices) for name, indices in roles.items()},
        "training_role_sha256": {
            name: indices_sha256(indices) for name, indices in roles.items()
        },
        "validation_diagnostic_indices_sha256": indices_sha256(
            verified["validation_diagnostic_indices"]
        ),
        "training_shape": list(arrays["X_train"].shape),
        "validation_shape": list(arrays["X_val"].shape),
        "class_order": list(range(13)),
        "protocol_id": protocol["experiment_id"],
        "dataset_regenerated": False,
        "test_accessed": False,
        "training_executed": False,
        "sampler_jobs_submitted": False,
    }
    atomic_write_json(output_dir / "attempt_manifest.json", manifest)
    return output_dir


def _parameter_hashes(parameters: np.ndarray, n_ansatz: int) -> dict[str, str]:
    return {
        "quantum_parameter_sha256": parameters_sha256(parameters[:n_ansatz]),
        "classical_parameter_sha256": parameters_sha256(parameters[n_ansatz:]),
        "complete_parameter_sha256": parameters_sha256(parameters),
    }


def _measurement_rows(
    count_rows: list[dict[str, int]],
    indices: np.ndarray,
    labels: np.ndarray,
    parameters: np.ndarray,
    *,
    split: str,
    iteration: int,
    phase: str,
    timestamp: str,
    n_ansatz: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    hashes = _parameter_hashes(parameters, n_ansatz)
    rows = []
    probabilities = []
    for counts, example_id in zip(count_rows, indices):
        features, standard_errors, _, shots = measurement_moments(counts, 16)
        if shots != 512:
            raise RuntimeError(f"Expected 512 shots per observation, received {shots}")
        probs, loss, _ = mlp_loss_and_feature_gradient(
            features,
            parameters,
            int(labels[example_id]),
            n_ansatz_parameters=n_ansatz,
            n_features=31,
            n_classes=13,
            hidden_size=16,
        )
        probabilities.append(probs)
        rows.append({
            "example_id": int(example_id),
            "split": split,
            "class_id": int(labels[example_id]),
            "iteration": int(iteration),
            "phase": phase,
            **hashes,
            "actual_shots": shots,
            "features": features,
            "standard_errors": standard_errors,
            "counts": counts,
            "timestamp_utc": timestamp,
            "backend": "Qiskit StatevectorSampler",
            "negative_log_likelihood": loss,
        })
    return rows, np.asarray(probabilities, dtype=np.float64)


def _probability_rows(
    probabilities: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    *,
    split: str,
    iteration: int,
    phase: str,
    hashes: dict[str, str],
) -> list[dict[str, Any]]:
    diagnostics = probability_diagnostics(probabilities, labels[indices], n_classes=13)
    return [
        {
            "example_id": int(index),
            "split": split,
            "true_label": int(labels[index]),
            "predicted_label": int(diagnostics["predicted_labels"][row]),
            "iteration": iteration,
            "phase": phase,
            **hashes,
            "probabilities": probabilities[row],
            "negative_log_likelihood": float(
                diagnostics["negative_log_likelihood"][row]
            ),
            "top_two_margin": float(diagnostics["top_two_margin"][row]),
            "predictive_entropy": float(diagnostics["predictive_entropy"][row]),
        }
        for row, index in enumerate(indices)
    ]


def _validation_metrics(
    probabilities: np.ndarray, labels: np.ndarray
) -> dict[str, Any]:
    diagnostics = probability_diagnostics(probabilities, labels, n_classes=13)
    predicted = diagnostics["predicted_labels"]
    confusion = np.zeros((13, 13), dtype=np.int64)
    np.add.at(confusion, (labels, predicted), 1)
    support = confusion.sum(axis=1)
    true_positive = np.diag(confusion)
    recall = np.divide(true_positive, support, out=np.zeros(13), where=support > 0)
    precision = np.divide(
        true_positive,
        confusion.sum(axis=0),
        out=np.zeros(13),
        where=confusion.sum(axis=0) > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(13),
        where=(precision + recall) > 0,
    )
    return {
        "accuracy": float(np.mean(predicted == labels)),
        "balanced_accuracy": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "cross_entropy": float(np.mean(diagnostics["negative_log_likelihood"])),
        "multiclass_brier_score": diagnostics["multiclass_brier_score"],
        "top_label_ece": diagnostics["top_label_ece"],
        "calibration_bins": diagnostics["calibration_bins"],
        "confusion_matrix_true_rows_predicted_columns": confusion.tolist(),
    }


def _run_training(
    protocol: dict,
    arrays: dict[str, np.ndarray],
    verified: dict[str, Any],
) -> Path:
    started_clock = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    output_dir = _new_attempt_dir(PROTOCOL_PATH.parent, "q16")
    atomic_save_npz(
        output_dir / "data_role_indices.npz",
        {
            **verified["roles"],
            "validation_diagnostic": verified["validation_diagnostic_indices"],
        },
    )
    atomic_write_json(
        output_dir / "attempt_manifest.json",
        {
            "status": "running",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": verified["protocol_sha256"],
            "dataset_hashes": verified["dataset_hashes"],
            "training_role_sha256": {
                name: indices_sha256(indices)
                for name, indices in verified["roles"].items()
            },
            "validation_diagnostic_indices_sha256": indices_sha256(
                verified["validation_diagnostic_indices"]
            ),
            "training_executed": False,
            "test_accessed": False,
            "sampler_jobs_submitted": False,
        },
    )
    X_train = arrays["X_train"][:, :16]
    X_val = arrays["X_val"][:, :16]
    y_train = arrays["y_train"].astype(np.int64, copy=False)
    y_val = arrays["y_val"].astype(np.int64, copy=False)
    roles = verified["roles"]

    loss_fn = None
    try:
        loss_fn = _make_vqc_loss_fn(
            X_train=X_train,
            y_train=y_train,
            n_qubits=16,
            n_features=16,
            shots=512,
            mode="sim",
            batch_size_local=32,
            feature_map_reps=2,
            ansatz_reps=2,
            entanglement="linear",
            readout_hidden_size=16,
            random_seed=42,
            readout_mode="marginals_zz_neighbors",
            input_preprocessing="identity",
            gradient_indices=roles["gradient"],
            readout_indices=roles["readout"],
            checkpoint_reference_indices=roles["reference"],
            readout_fit_seed=42,
        )
        rng = np.random.default_rng(42)
        ansatz_initial = rng.normal(0.0, 0.01, loss_fn.n_ansatz_params)
        readout_initial = loss_fn.warm_start_readout(ansatz_initial)
        initial_parameters = np.concatenate(
            (
                ansatz_initial,
                np.concatenate(
                    [
                        block.reshape(-1) if block.ndim == 2 else block
                        for block in readout_initial
                    ]
                ),
            )
        )
        expected_total = loss_fn.n_ansatz_params + 733
        if initial_parameters.shape != (expected_total,):
            raise ValueError(
                "Initialized parameter count differs from the frozen protocol"
            )
        loss_fn.execution_telemetry["n_readout_initializations"] += 1
    except Exception as exc:
        atomic_write_json(
            output_dir / "attempt_manifest.json",
            {
                "status": "failed_during_initialization",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "protocol_sha256": verified["protocol_sha256"],
                "dataset_hashes": verified["dataset_hashes"],
                "training_optimization_started": False,
                "test_accessed": False,
                "sampler_jobs_submitted": bool(
                    getattr(loss_fn, "execution_telemetry", {}).get(
                        "n_sampler_jobs_submitted", 0
                    )
                ),
                "sampler_run_calls": getattr(
                    loss_fn, "execution_telemetry", {}
                ).get("n_sampler_run_calls", 0),
                "sampler_jobs_submitted_count": getattr(
                    loss_fn, "execution_telemetry", {}
                ).get("n_sampler_jobs_submitted", 0),
                "failure": f"{type(exc).__name__}: {exc}",
            },
        )
        raise

    candidate_records: list[dict[str, Any]] = []
    candidate_vectors: list[np.ndarray] = []
    candidate_stats: list[dict[str, Any]] = []
    measurement_metadata: list[dict[str, Any]] = []
    measurement_shards: list[str] = []
    probability_rows: list[dict[str, Any]] = []
    incumbent_stats = None
    selected_candidate_index: int | None = None
    comparison_number = 0
    readout_refit_history = []
    observations_dir = output_dir / "quantum_measurements"
    observations_dir.mkdir()
    observation_metadata_path = output_dir / "quantum_measurements.json"
    candidates_path = output_dir / "checkpoint_candidates.npz"
    candidate_metadata_path = output_dir / "checkpoint_candidates.json"

    def persist_measurements(rows: list[dict[str, Any]]) -> None:
        shard_name = f"observations_{len(measurement_shards):04d}.npz"
        atomic_save_npz(
            observations_dir / shard_name,
            measurement_artifact_arrays(rows),
        )
        measurement_shards.append(shard_name)
        measurement_metadata.extend(
            {
                "example_id": row["example_id"],
                "split": row["split"],
                "class_id": row["class_id"],
                "iteration": row["iteration"],
                "phase": row["phase"],
                "quantum_parameter_sha256": row["quantum_parameter_sha256"],
                "classical_parameter_sha256": row["classical_parameter_sha256"],
                "complete_parameter_sha256": row["complete_parameter_sha256"],
                "actual_shots": row["actual_shots"],
                "timestamp_utc": row["timestamp_utc"],
                "backend": row["backend"],
            }
            for row in rows
        )
        atomic_write_json(
            observation_metadata_path,
            {
                "backend": "Qiskit StatevectorSampler",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "qubits": 16,
                "protocol_sha256": verified["protocol_sha256"],
                "shards": measurement_shards,
                "observation_count": len(measurement_metadata),
                "observations": measurement_metadata,
            },
        )

    def persist_candidates() -> None:
        atomic_save_npz(
            candidates_path,
            {
                "parameters": np.stack(candidate_vectors),
                "reference_losses": np.stack(
                    [item["per_example_losses"] for item in candidate_stats]
                ),
                "reference_loss_variances": np.stack(
                    [item["per_example_loss_variances"] for item in candidate_stats]
                ),
            },
        )
        atomic_write_json(candidate_metadata_path, {"candidates": candidate_records})

    def checkpoint_candidate(iteration, phase, parameters, context):
        nonlocal incumbent_stats, comparison_number, selected_candidate_index
        timestamp = datetime.now(timezone.utc).isoformat()
        count_rows = loss_fn.measure_counts_by_indices(
            parameters, roles["reference"], 512
        )
        stats = candidate_statistics(
            count_rows,
            y_train,
            roles["reference"],
            parameters,
            n_qubits=16,
            n_ansatz_parameters=loss_fn.n_ansatz_params,
            n_classes=13,
            hidden_size=16,
        )
        if not np.all(stats["actual_shots"] == 512):
            raise RuntimeError("Checkpoint candidates must use exactly 512 shots")
        if phase == "initial":
            replaces = True
            comparison = None
            incumbent_stats = stats
        else:
            comparison_number += 1
            comparison = compare_candidates(
                stats,
                incumbent_stats,
                comparison_number=comparison_number,
            )
            replaces = bool(comparison["replaces_incumbent"])
            if replaces:
                incumbent_stats = stats

        hashes = _parameter_hashes(parameters, loss_fn.n_ansatz_params)
        record = {
            "candidate_index": len(candidate_records),
            "iteration": int(iteration),
            "state_phase": phase,
            **hashes,
            "reference_objective": stats["objective"],
            "reference_objective_standard_error": stats[
                "objective_standard_error"
            ],
            "reference_examples_sha256": stats["reference_indices_sha256"],
            "reference_examples": len(roles["reference"]),
            "actual_shots_per_example": stats["actual_shots"].tolist(),
            "shots_requested": 512,
            "evaluation_seed_policy": (
                "single StatevectorSampler(seed=42), sequential random stream"
            ),
            "uncertainty_method": stats["uncertainty_method"],
            "comparison": comparison,
            "replaces_incumbent": replaces,
            "context": context,
            "timestamp_utc": timestamp,
        }
        candidate_records.append(record)
        candidate_vectors.append(parameters.copy())
        candidate_stats.append(stats)
        if replaces:
            selected_candidate_index = record["candidate_index"]

        rows, probabilities = _measurement_rows(
            count_rows,
            roles["reference"],
            y_train,
            parameters,
            split="train",
            iteration=iteration,
            phase=phase,
            timestamp=timestamp,
            n_ansatz=loss_fn.n_ansatz_params,
        )
        new_measurement_rows = rows.copy()
        if phase == "initial":
            probability_rows.extend(
                _probability_rows(
                    probabilities,
                    y_train,
                    roles["reference"],
                    split="train",
                    iteration=iteration,
                    phase=phase,
                    hashes=hashes,
                )
            )
            validation_indices = verified["validation_diagnostic_indices"]
            val_counts = loss_fn.measure_counts_external(
                parameters, X_val[validation_indices], 512
            )
            val_rows, val_probabilities = _measurement_rows(
                val_counts,
                validation_indices,
                y_val,
                parameters,
                split="validation",
                iteration=iteration,
                phase=phase,
                timestamp=datetime.now(timezone.utc).isoformat(),
                n_ansatz=loss_fn.n_ansatz_params,
            )
            new_measurement_rows.extend(val_rows)
            probability_rows.extend(
                _probability_rows(
                    val_probabilities,
                    y_val,
                    validation_indices,
                    split="validation",
                    iteration=iteration,
                    phase=phase,
                    hashes=hashes,
                )
            )

        persist_candidates()
        if replaces:
            atomic_save_npz(
                output_dir / "selected_checkpoint.npz",
                {
                    "parameters": parameters,
                    "quantum_parameters": parameters[: loss_fn.n_ansatz_params],
                    "classical_parameters": parameters[loss_fn.n_ansatz_params :],
                },
            )
            atomic_write_json(
                output_dir / "selected_checkpoint.json",
                {
                    "candidate_index": record["candidate_index"],
                    "iteration": record["iteration"],
                    "state_phase": record["state_phase"],
                    **hashes,
                    "reference_objective": record["reference_objective"],
                    "reference_objective_standard_error": record[
                        "reference_objective_standard_error"
                    ],
                    "reference_examples_sha256": record[
                        "reference_examples_sha256"
                    ],
                    "shots_requested": 512,
                    "evaluation_seed_policy": record["evaluation_seed_policy"],
                    "comparison": record["comparison"],
                    "uncertainty_method": record["uncertainty_method"],
                },
            )
        persist_measurements(new_measurement_rows)
        return {
            "objective": stats["objective"],
            "replaces_incumbent": replaces,
            "objective_function_calls": 1,
        }

    def optimizer_callback(iteration, parameters, loss):
        if iteration % 10 != 0:
            return False
        ansatz_parameters = parameters[: loss_fn.n_ansatz_params]
        refit = loss_fn.warm_start_readout(ansatz_parameters)
        cursor = loss_fn.n_ansatz_params
        for block in refit:
            flat = block.reshape(-1) if block.ndim == 2 else block
            parameters[cursor : cursor + len(flat)] = flat
            cursor += len(flat)
        loss_fn.execution_telemetry["n_readout_refits"] += 1
        readout_refit_history.append({
            "iteration": int(iteration),
            "training_indices": roles["readout"].tolist(),
            "training_indices_sha256": indices_sha256(roles["readout"]),
            "n_training_examples": len(roles["readout"]),
        })
        return True

    try:
        optimizer = QNSPSAEMLFeynman(
            QNSPSAConfig(
                maxiter=20,
                lr=0.03,
                perturbation=0.05,
                lambda_eml=0.01,
                hessian_regularization=1e-4,
                blocking_delta=1e-3,
                n_feynman_params=4,
                n_gl_points=8,
                patience=20,
                min_improvement=5e-4,
                seed=42,
            )
        )
        result = optimizer.minimize(
            loss_fn,
            initial_parameters,
            callback=optimizer_callback,
            checkpoint_candidate_callback=checkpoint_candidate,
            parameter_block_sizes=(loss_fn.n_ansatz_params, 733),
        )

        selected_parameters = result.optimal_params.copy()
        selected_hashes = _parameter_hashes(
            selected_parameters, loss_fn.n_ansatz_params
        )
        if selected_candidate_index is None:
            raise RuntimeError("No checkpoint candidate was selected")
        selected_candidate = candidate_records[selected_candidate_index]
        if (
            selected_candidate["complete_parameter_sha256"]
            != selected_hashes["complete_parameter_sha256"]
        ):
            raise RuntimeError("Optimizer and runner selected different checkpoints")
        selected_stats = candidate_stats[selected_candidate["candidate_index"]]
        selected_indices = verified["validation_diagnostic_indices"]
        selected_diag_counts = loss_fn.measure_counts_external(
            selected_parameters, X_val[selected_indices], 512
        )
        selected_diag_rows, selected_diag_probabilities = _measurement_rows(
            selected_diag_counts,
            selected_indices,
            y_val,
            selected_parameters,
            split="validation",
            iteration=selected_candidate["iteration"],
            phase="selected",
            timestamp=datetime.now(timezone.utc).isoformat(),
            n_ansatz=loss_fn.n_ansatz_params,
        )
        persist_measurements(selected_diag_rows)
        probability_rows.extend(
            _probability_rows(
                selected_stats["probabilities"],
                y_train,
                roles["reference"],
                split="train",
                iteration=selected_candidate["iteration"],
                phase="selected",
                hashes=selected_hashes,
            )
        )
        probability_rows.extend(
            _probability_rows(
                selected_diag_probabilities,
                y_val,
                selected_indices,
                split="validation",
                iteration=selected_candidate["iteration"],
                phase="selected",
                hashes=selected_hashes,
            )
        )

        full_validation_counts = loss_fn.measure_counts_external(
            selected_parameters, X_val, 512
        )
        full_probabilities = np.empty((len(y_val), 13), dtype=np.float64)
        for row, counts in enumerate(full_validation_counts):
            features, _, _, actual_shots = measurement_moments(counts, 16)
            if actual_shots != 512:
                raise RuntimeError(
                    f"Expected 512 validation shots, received {actual_shots}"
                )
            full_probabilities[row], _, _ = mlp_loss_and_feature_gradient(
                features,
                selected_parameters,
                int(y_val[row]),
                n_ansatz_parameters=loss_fn.n_ansatz_params,
                n_features=31,
                n_classes=13,
                hidden_size=16,
            )
        final_prob_rows = _probability_rows(
            full_probabilities,
            y_val,
            np.arange(len(y_val)),
            split="validation",
            iteration=selected_candidate["iteration"],
            phase="final_full_validation",
            hashes=selected_hashes,
        )
        probability_rows.extend(final_prob_rows)

        atomic_save_npz(
            output_dir / "probabilities.npz",
            probability_artifact_arrays(probability_rows),
        )
        atomic_write_json(
            output_dir / "metrics.json",
            {
                **_validation_metrics(full_probabilities, y_val),
                "selected_candidate_index": selected_candidate["candidate_index"],
                "selected_checkpoint": selected_candidate,
                "initial_parameter_hashes": _parameter_hashes(
                    initial_parameters, loss_fn.n_ansatz_params
                ),
            },
        )
        atomic_save_npz(
            output_dir / "selected_checkpoint.npz",
            {
                "parameters": selected_parameters,
                "quantum_parameters": selected_parameters[: loss_fn.n_ansatz_params],
                "classical_parameters": selected_parameters[loss_fn.n_ansatz_params :],
            },
        )
        persist_candidates()
        attempt_manifest = {
            "status": "completed",
            "started_at_utc": started_at,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_clock_seconds": time.perf_counter() - started_clock,
            "protocol_sha256": verified["protocol_sha256"],
            "dataset_hashes": verified["dataset_hashes"],
            "training_role_sha256": {
                name: indices_sha256(indices) for name, indices in roles.items()
            },
            "training_executed": True,
            "test_accessed": False,
            "sampler_backend": "Qiskit StatevectorSampler",
            "sampler_jobs_submitted": True,
            "sampler_run_calls": loss_fn.execution_telemetry[
                "n_sampler_run_calls"
            ],
            "sampler_jobs_submitted_count": loss_fn.execution_telemetry[
                "n_sampler_jobs_submitted"
            ],
            "sampler_jobs_completed": loss_fn.execution_telemetry[
                "n_sampler_jobs_completed"
            ],
            "circuit_executions": loss_fn.execution_telemetry[
                "n_circuit_executions"
            ],
            "actual_shots_total": loss_fn.execution_telemetry["n_shots_total"],
            "checkpoint_reference_examples_evaluated": loss_fn.execution_telemetry[
                "n_checkpoint_reference_examples_evaluated"
            ],
            "validation_examples_evaluated": loss_fn.execution_telemetry[
                "n_validation_examples_evaluated"
            ],
            "readout_initializations": loss_fn.execution_telemetry[
                "n_readout_initializations"
            ],
            "readout_refits": readout_refit_history,
            "optimizer_iterations": result.n_iter,
            "optimizer_objective_function_calls": result.n_objective_function_calls,
            "optimizer_step_history": result.optimizer_step_history,
            "selected_parameter_hashes": selected_hashes,
            "selected_candidate_index": selected_candidate["candidate_index"],
            "failure": None,
        }
        atomic_write_json(output_dir / "attempt_manifest.json", attempt_manifest)
        return output_dir
    except Exception as exc:
        atomic_write_json(
            output_dir / "attempt_manifest.json",
            {
                "status": "failed",
                "started_at_utc": started_at,
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "wall_clock_seconds": time.perf_counter() - started_clock,
                "protocol_sha256": verified["protocol_sha256"],
                "dataset_hashes": verified["dataset_hashes"],
                "training_executed": True,
                "test_accessed": False,
                "sampler_run_calls": loss_fn.execution_telemetry[
                    "n_sampler_run_calls"
                ],
                "sampler_jobs_submitted_count": loss_fn.execution_telemetry[
                    "n_sampler_jobs_submitted"
                ],
                "sampler_jobs_completed": loss_fn.execution_telemetry[
                    "n_sampler_jobs_completed"
                ],
                "failure": f"{type(exc).__name__}: {exc}",
                "candidate_count": len(candidate_records),
                "last_atomic_candidate_metadata": candidate_metadata_path.name,
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    protocol, arrays, verified = _load_verified_inputs()
    if args.dry_run:
        output_dir = _write_dry_run(protocol, arrays, verified)
        print(json.dumps({
            "status": "dry_run_passed",
            "output_dir": str(output_dir),
            "protocol_sha256": verified["protocol_sha256"],
            "training_roles": {
                name: {
                    "size": len(indices),
                    "sha256": indices_sha256(indices),
                }
                for name, indices in verified["roles"].items()
            },
            "test_accessed": False,
            "training_executed": False,
            "sampler_jobs_submitted": False,
        }, indent=2))
        return 0

    output_dir = _run_training(protocol, arrays, verified)
    print(f"QNIM V2 completed: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
