"""Pure data, measurement, and selection helpers for QNIM V2."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from statistics import NormalDist
from pathlib import Path
from typing import Any

import numpy as np


def stratified_disjoint_roles(
    labels: np.ndarray,
    *,
    per_class: tuple[int, int, int] = (175, 15, 10),
    seed: int = 42,
) -> dict[str, np.ndarray]:
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional array")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must contain integer class IDs")
    if len(per_class) != 3 or any(size < 0 for size in per_class):
        raise ValueError("per_class must contain three non-negative sizes")

    classes, class_counts = np.unique(labels, return_counts=True)
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("class IDs must be contiguous and start at zero")
    if np.any(class_counts != sum(per_class)):
        raise ValueError("each class must have enough examples for the declared roles")

    rng = np.random.default_rng(seed)
    roles: dict[str, list[np.ndarray]] = {
        "gradient": [],
        "readout": [],
        "reference": [],
    }
    for class_id in classes:
        indices = np.flatnonzero(labels == class_id)
        indices = rng.permutation(indices)
        first = per_class[0]
        second = first + per_class[1]
        roles["gradient"].append(indices[:first])
        roles["readout"].append(indices[first:second])
        roles["reference"].append(indices[second:])

    result = {
        name: np.sort(np.concatenate(groups)).astype(np.int64)
        for name, groups in roles.items()
    }
    validate_disjoint_roles(labels, result, per_class=per_class)
    return result


def validate_disjoint_roles(
    labels: np.ndarray,
    roles: dict[str, np.ndarray],
    *,
    per_class: tuple[int, int, int] = (175, 15, 10),
) -> None:
    labels = np.asarray(labels)
    names = ("gradient", "readout", "reference")
    if set(roles) != set(names):
        raise ValueError("roles must contain gradient, readout, and reference")
    normalized = {}
    n_classes = len(np.unique(labels))
    for name in names:
        values = np.asarray(roles[name], dtype=np.int64)
        if values.ndim != 1 or np.any(values < 0) or np.any(values >= len(labels)):
            raise ValueError(f"{name} indices must be valid one-dimensional indices")
        if len(np.unique(values)) != len(values):
            raise ValueError(f"{name} indices contain duplicates")
        normalized[name] = values

    if any(
        np.intersect1d(normalized[left], normalized[right]).size
        for i, left in enumerate(names)
        for right in names[i + 1 :]
    ):
        raise ValueError("training roles must be pairwise disjoint")
    combined = np.concatenate([normalized[name] for name in names])
    if len(combined) != len(labels) or not np.array_equal(
        np.sort(combined), np.arange(len(labels))
    ):
        raise ValueError("training roles must partition the complete training split")
    for role_index, name in enumerate(names):
        counts = np.bincount(
            labels[normalized[name]], minlength=n_classes
        )
        if not np.array_equal(counts, np.full(n_classes, per_class[role_index])):
            raise ValueError(f"{name} role is not stratified as declared")


def indices_sha256(indices: np.ndarray) -> str:
    canonical = np.ascontiguousarray(indices, dtype="<i8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def parameters_sha256(parameters: np.ndarray) -> str:
    canonical = np.ascontiguousarray(parameters, dtype="<f8")
    if not np.isfinite(canonical).all():
        raise ValueError("parameters must be finite")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def bitstring_features(bitstring: str, n_qubits: int) -> np.ndarray:
    bits = bitstring.replace(" ", "")
    if len(bits) != n_qubits or any(bit not in "01" for bit in bits):
        raise ValueError("measurement bitstring is invalid")
    z_values = np.fromiter(
        (1.0 - 2.0 * int(bits[-(qubit + 1)]) for qubit in range(n_qubits)),
        dtype=np.float64,
        count=n_qubits,
    )
    marginals = (z_values + 1.0) / 2.0
    adjacent_zz = z_values[:-1] * z_values[1:]
    return np.concatenate((marginals, adjacent_zz))


def measurement_moments(
    counts: dict[str, int],
    n_qubits: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if not counts or not 1 <= n_qubits <= 16:
        raise ValueError("non-empty counts and 1 <= n_qubits <= 16 are required")
    outcomes = []
    frequencies = []
    for bitstring, count in counts.items():
        if not isinstance(count, (int, np.integer)) or count <= 0:
            raise ValueError("measurement counts must be positive integers")
        outcomes.append(bitstring_features(str(bitstring), n_qubits))
        frequencies.append(int(count))

    values = np.asarray(outcomes, dtype=np.float64)
    weights = np.asarray(frequencies, dtype=np.float64)
    shots = int(weights.sum())
    if shots < 2:
        raise ValueError("at least two shots are required to estimate uncertainty")
    mean = weights @ values / shots

    centered = values - mean
    sample_covariance = (centered.T * weights) @ centered / (shots - 1)
    covariance_of_mean = sample_covariance / shots
    standard_errors = np.sqrt(np.maximum(np.diag(covariance_of_mean), 0.0))
    return mean, standard_errors, covariance_of_mean, shots


def mlp_loss_and_feature_gradient(
    features: np.ndarray,
    parameters: np.ndarray,
    label: int,
    *,
    n_ansatz_parameters: int,
    n_features: int,
    n_classes: int,
    hidden_size: int,
    probability_floor: float = 1e-10,
) -> tuple[np.ndarray, float, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    parameters = np.asarray(parameters, dtype=np.float64)
    if features.shape != (n_features,) or not np.isfinite(features).all():
        raise ValueError("features must be a finite vector of the declared size")
    expected_tail = hidden_size * n_features + hidden_size
    expected_tail += n_classes * hidden_size + n_classes
    if parameters.ndim != 1 or parameters.size != n_ansatz_parameters + expected_tail:
        raise ValueError("parameter vector does not match the declared MLP shape")
    if not 0 <= label < n_classes:
        raise ValueError("label is outside the declared class range")

    tail = parameters[n_ansatz_parameters:]
    cursor = hidden_size * n_features
    W1 = tail[:cursor].reshape(hidden_size, n_features)
    b1 = tail[cursor : cursor + hidden_size]
    cursor += hidden_size
    W2 = tail[cursor : cursor + n_classes * hidden_size].reshape(
        n_classes, hidden_size
    )
    cursor += n_classes * hidden_size
    b2 = tail[cursor : cursor + n_classes]

    hidden_pre_activation = W1 @ features + b1
    hidden = np.maximum(hidden_pre_activation, 0.0)
    logits = W2 @ hidden + b2
    logits -= np.max(logits)
    exp_logits = np.exp(logits)
    probabilities = exp_logits / np.sum(exp_logits)
    true_probability = float(np.clip(probabilities[label], probability_floor, 1.0))
    loss = -math.log(true_probability)

    if probabilities[label] <= probability_floor:
        return probabilities, loss, np.zeros(n_features, dtype=np.float64)
    d_logits = probabilities.copy()
    d_logits[label] -= 1.0
    d_hidden = W2.T @ d_logits
    d_hidden_pre_activation = d_hidden * (hidden_pre_activation > 0.0)
    d_features = W1.T @ d_hidden_pre_activation
    return probabilities, loss, d_features


def candidate_statistics(
    count_rows: list[dict[str, int]],
    labels: np.ndarray,
    reference_indices: np.ndarray,
    parameters: np.ndarray,
    *,
    n_qubits: int = 16,
    n_ansatz_parameters: int = 96,
    n_classes: int = 13,
    hidden_size: int = 16,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    reference_indices = np.asarray(reference_indices, dtype=np.int64)
    if len(count_rows) == 0 or len(count_rows) != len(reference_indices):
        raise ValueError("count rows must align with reference indices")
    losses = np.empty(len(count_rows), dtype=np.float64)
    variances = np.empty(len(count_rows), dtype=np.float64)
    probabilities = np.empty((len(count_rows), n_classes), dtype=np.float64)
    features = np.empty((len(count_rows), 2 * n_qubits - 1), dtype=np.float64)
    standard_errors = np.empty_like(features)
    actual_shots = np.empty(len(count_rows), dtype=np.int64)

    for row, (counts, example_index) in enumerate(zip(count_rows, reference_indices)):
        mean, stderr, covariance, shots = measurement_moments(counts, n_qubits)
        probs, loss, gradient = mlp_loss_and_feature_gradient(
            mean,
            parameters,
            int(labels[example_index]),
            n_ansatz_parameters=n_ansatz_parameters,
            n_features=2 * n_qubits - 1,
            n_classes=n_classes,
            hidden_size=hidden_size,
        )
        probabilities[row] = probs
        losses[row] = loss
        features[row] = mean
        standard_errors[row] = stderr
        actual_shots[row] = shots
        variances[row] = max(float(gradient @ covariance @ gradient), 0.0)

    objective = float(np.mean(losses))
    objective_se = float(np.sqrt(np.sum(variances)) / len(losses))
    return {
        "objective": objective,
        "objective_standard_error": objective_se,
        "per_example_losses": losses,
        "per_example_loss_variances": variances,
        "probabilities": probabilities,
        "features": features,
        "feature_standard_errors": standard_errors,
        "actual_shots": actual_shots,
        "reference_indices": reference_indices.copy(),
        "reference_indices_sha256": indices_sha256(reference_indices),
        "uncertainty_method": (
            "first_order_delta_method_using_full_bitstring_feature_covariance"
        ),
    }


def compare_candidates(
    candidate: dict[str, Any],
    incumbent: dict[str, Any],
    *,
    comparison_number: int,
    maximum_comparisons: int = 22,
    familywise_alpha: float = 0.05,
    minimum_improvement: float = 5e-4,
) -> dict[str, float | bool | str]:
    candidate_indices = np.asarray(candidate["reference_indices"], dtype=np.int64)
    incumbent_indices = np.asarray(incumbent["reference_indices"], dtype=np.int64)
    if not np.array_equal(candidate_indices, incumbent_indices):
        raise ValueError("candidate and incumbent must use identical reference examples")
    if candidate["reference_indices_sha256"] != incumbent["reference_indices_sha256"]:
        raise ValueError("candidate and incumbent reference-index hashes differ")
    if not 1 <= comparison_number <= maximum_comparisons:
        raise ValueError("comparison_number exceeds the declared comparison budget")

    candidate_losses = np.asarray(candidate["per_example_losses"], dtype=np.float64)
    incumbent_losses = np.asarray(incumbent["per_example_losses"], dtype=np.float64)
    candidate_variances = np.asarray(
        candidate["per_example_loss_variances"], dtype=np.float64
    )
    incumbent_variances = np.asarray(
        incumbent["per_example_loss_variances"], dtype=np.float64
    )
    n_examples = len(candidate_indices)
    if any(
        values.shape != (n_examples,)
        for values in (
            candidate_losses,
            incumbent_losses,
            candidate_variances,
            incumbent_variances,
        )
    ):
        raise ValueError("candidate loss and uncertainty arrays must match references")

    paired_differences = candidate_losses - incumbent_losses
    delta = float(np.mean(paired_differences))
    standard_error = float(
        np.sqrt(np.sum(candidate_variances + incumbent_variances)) / n_examples
    )
    adjusted_alpha = familywise_alpha / maximum_comparisons
    critical_value = NormalDist().inv_cdf(1.0 - adjusted_alpha)
    upper_bound = delta + critical_value * standard_error
    replaces_incumbent = upper_bound < -minimum_improvement
    return {
        "paired_objective_delta": delta,
        "paired_standard_error": standard_error,
        "one_sided_upper_bound": upper_bound,
        "critical_value": critical_value,
        "minimum_improvement": minimum_improvement,
        "replaces_incumbent": replaces_incumbent,
        "uncertainty_method": (
            "paired_fixed-example_delta_method; familywise_alpha="
            f"{familywise_alpha}; comparisons={maximum_comparisons}"
        ),
    }


def probability_diagnostics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    n_classes: int = 13,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.ndim != 2 or probabilities.shape != (
        len(labels),
        n_classes,
    ):
        raise ValueError("probabilities must have shape (n_examples, n_classes)")
    if len(labels) == 0 or not np.isfinite(probabilities).all():
        raise ValueError("probabilities and labels must be non-empty and finite")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("probabilities must lie in [0,1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8, rtol=1e-8):
        raise ValueError("probability rows must sum to one")
    if np.any(labels < 0) or np.any(labels >= n_classes):
        raise ValueError("labels are outside the declared class range")

    predictions = np.argmax(probabilities, axis=1)
    rows = np.arange(len(labels))
    nll = -np.log(np.clip(probabilities[rows, labels], 1e-10, 1.0))
    one_hot = np.eye(n_classes, dtype=np.float64)[labels]
    brier = np.sum((probabilities - one_hot) ** 2, axis=1)
    sorted_probs = np.sort(probabilities, axis=1)
    confidence = sorted_probs[:, -1]
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    entropy = -np.sum(
        probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)),
        axis=1,
    )

    edges = np.linspace(0.0, 1.0, calibration_bins + 1)
    ece = 0.0
    bins = []
    correct = predictions == labels
    for bin_index in range(calibration_bins):
        if bin_index == calibration_bins - 1:
            mask = (confidence >= edges[bin_index]) & (confidence <= edges[bin_index + 1])
        else:
            mask = (confidence >= edges[bin_index]) & (confidence < edges[bin_index + 1])
        count = int(mask.sum())
        mean_confidence = float(np.mean(confidence[mask])) if count else None
        accuracy = float(np.mean(correct[mask])) if count else None
        if count:
            ece += count / len(labels) * abs(mean_confidence - accuracy)
        bins.append(
            {
                "lower": float(edges[bin_index]),
                "upper": float(edges[bin_index + 1]),
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )

    return {
        "predicted_labels": predictions,
        "negative_log_likelihood": nll,
        "top_two_margin": margins,
        "predictive_entropy": entropy,
        "multiclass_brier_score": float(np.mean(brier)),
        "top_label_ece": float(ece),
        "calibration_bins": bins,
        "calibration_bin_count": calibration_bins,
    }


def measurement_artifact_arrays(
    observations: list[dict[str, Any]],
    *,
    n_qubits: int = 16,
) -> dict[str, np.ndarray]:
    if not observations or not 1 <= n_qubits <= 16:
        raise ValueError("measurement artifact requires at least one observation")
    offsets = [0]
    outcome_values: list[int] = []
    outcome_counts: list[int] = []
    rows = []
    for observation in observations:
        counts = observation["counts"]
        row_shots = 0
        for bitstring in sorted(counts):
            bits = str(bitstring).replace(" ", "")
            if len(bits) != n_qubits or any(bit not in "01" for bit in bits):
                raise ValueError("observation contains an invalid bitstring")
            count = int(counts[bitstring])
            if count <= 0 or count > np.iinfo(np.uint32).max:
                raise ValueError("outcome count is outside uint32 range")
            outcome_values.append(int(bits, 2))
            outcome_counts.append(count)
            row_shots += count
        if row_shots != int(observation["actual_shots"]):
            raise ValueError("stored outcome counts differ from actual_shots")
        offsets.append(len(outcome_values))
        rows.append(observation)

    def array(name: str, dtype: str) -> np.ndarray:
        return np.asarray([row[name] for row in rows], dtype=dtype)

    features = np.asarray([row["features"] for row in rows], dtype=np.float32)
    standard_errors = np.asarray(
        [row["standard_errors"] for row in rows], dtype=np.float32
    )
    expected_features = 2 * n_qubits - 1
    if features.shape != (len(rows), expected_features):
        raise ValueError("measurement feature shape differs from n_qubits")
    if standard_errors.shape != features.shape:
        raise ValueError("measurement standard errors do not match feature shape")
    if not np.isfinite(features).all() or not np.isfinite(standard_errors).all():
        raise ValueError("measurement features and standard errors must be finite")
    marginals = features[:, :n_qubits]
    adjacent_zz = features[:, n_qubits:]
    if np.any(marginals < 0.0) or np.any(marginals > 1.0):
        raise ValueError("marginal measurements must lie in [0,1]")
    if np.any(adjacent_zz < -1.0) or np.any(adjacent_zz > 1.0):
        raise ValueError("adjacent ZZ measurements must lie in [-1,1]")

    return {
        "example_id": array("example_id", "int32"),
        "split": array("split", "U10"),
        "class_id": array("class_id", "uint8"),
        "iteration": array("iteration", "int16"),
        "phase": array("phase", "U16"),
        "quantum_parameter_sha256": array("quantum_parameter_sha256", "U64"),
        "classical_parameter_sha256": array("classical_parameter_sha256", "U64"),
        "complete_parameter_sha256": array("complete_parameter_sha256", "U64"),
        "actual_shots": array("actual_shots", "uint32"),
        "features": features,
        "standard_errors": standard_errors,
        "marginals": marginals,
        "adjacent_zz_correlations": adjacent_zz,
        "marginal_standard_errors": standard_errors[:, :n_qubits],
        "adjacent_zz_standard_errors": standard_errors[:, n_qubits:],
        "outcome_offsets": np.asarray(offsets, dtype=np.int64),
        "outcome_bitstrings": np.asarray(outcome_values, dtype=np.uint16),
        "outcome_counts": np.asarray(outcome_counts, dtype=np.uint32),
    }


def probability_artifact_arrays(
    rows: list[dict[str, Any]],
    *,
    n_classes: int = 13,
) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError("probability artifact requires at least one row")
    probabilities = np.asarray(
        [row["probabilities"] for row in rows], dtype=np.float32
    )
    if probabilities.shape != (len(rows), n_classes):
        raise ValueError("probability rows have an unexpected class dimension")
    if not np.isfinite(probabilities).all():
        raise ValueError("probability artifact contains non-finite values")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("probability artifact values must lie in [0,1]")
    if not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6
    ):
        raise ValueError("probability artifact rows must sum to one")
    result = {
        "example_id": np.asarray([r["example_id"] for r in rows], dtype=np.int32),
        "split": np.asarray([r["split"] for r in rows], dtype="U10"),
        "true_label": np.asarray([r["true_label"] for r in rows], dtype=np.uint8),
        "predicted_label": np.asarray(
            [r["predicted_label"] for r in rows], dtype=np.uint8
        ),
        "iteration": np.asarray([r["iteration"] for r in rows], dtype=np.int16),
        "phase": np.asarray([r["phase"] for r in rows], dtype="U16"),
        "quantum_parameter_sha256": np.asarray(
            [r["quantum_parameter_sha256"] for r in rows], dtype="U64"
        ),
        "classical_parameter_sha256": np.asarray(
            [r["classical_parameter_sha256"] for r in rows], dtype="U64"
        ),
        "complete_parameter_sha256": np.asarray(
            [r["complete_parameter_sha256"] for r in rows], dtype="U64"
        ),
        "probabilities": probabilities,
        "negative_log_likelihood": np.asarray(
            [r["negative_log_likelihood"] for r in rows], dtype=np.float32
        ),
        "top_two_margin": np.asarray(
            [r["top_two_margin"] for r in rows], dtype=np.float32
        ),
        "predictive_entropy": np.asarray(
            [r["predictive_entropy"] for r in rows], dtype=np.float32
        ),
    }
    if np.any(result["true_label"] >= n_classes) or np.any(
        result["predicted_label"] >= n_classes
    ):
        raise ValueError("probability artifact labels are outside the class range")
    for key in ("negative_log_likelihood", "top_two_margin", "predictive_entropy"):
        if not np.isfinite(result[key]).all():
            raise ValueError(f"{key} contains non-finite values")
    return result


def atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
