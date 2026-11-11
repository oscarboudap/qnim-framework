"""Statistical metrics for controlled QNIM scaling experiments."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score

RANDOM_BASELINE = 1.0 / 13.0


def _validate_predictions(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true e y_pred deben ser vectores 1D")
    if y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("y_true e y_pred deben tener la misma longitud no vacía")
    return y_true, y_pred


def stratified_bootstrap(
    y_true,
    y_pred,
    n_iterations: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Estimate a balanced-accuracy confidence interval by stratified resampling."""
    y_true, y_pred = _validate_predictions(y_true, y_pred)
    if n_iterations < 1:
        raise ValueError("n_iterations debe ser positivo")

    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(y_true == label) for label in np.unique(y_true)]
    scores = np.empty(n_iterations, dtype=float)

    for iteration in range(n_iterations):
        selected = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True)
            for indices in class_indices
        ])
        scores[iteration] = balanced_accuracy_score(
            y_true[selected], y_pred[selected]
        )

    return {
        "mean": float(np.mean(scores)),
        "lower_95": float(np.quantile(scores, 0.025)),
        "upper_95": float(np.quantile(scores, 0.975)),
        "one_sided_lower_95": float(np.quantile(scores, 0.05)),
    }


def permutation_test(
    y_true,
    y_pred,
    n_permutations: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Test label/prediction association while preserving label frequencies."""
    y_true, y_pred = _validate_predictions(y_true, y_pred)
    if n_permutations < 1:
        raise ValueError("n_permutations debe ser positivo")

    rng = np.random.default_rng(seed)
    observed = balanced_accuracy_score(y_true, y_pred)
    null_scores = np.empty(n_permutations, dtype=float)
    for iteration in range(n_permutations):
        null_scores[iteration] = balanced_accuracy_score(
            rng.permutation(y_true), y_pred
        )

    p_value = (1 + np.count_nonzero(null_scores >= observed)) / (n_permutations + 1)
    return {"observed": float(observed), "p_value": float(p_value)}


def paired_bootstrap_difference(
    y_true,
    pred_baseline,
    pred_extended,
    n_iterations: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Compare two classifiers on the same stratified bootstrap event indices."""
    y_true, pred_baseline = _validate_predictions(y_true, pred_baseline)
    y_true_extended, pred_extended = _validate_predictions(y_true, pred_extended)
    if not np.array_equal(y_true, y_true_extended):
        raise ValueError("Las comparaciones emparejadas requieren el mismo y_true")
    if n_iterations < 1:
        raise ValueError("n_iterations debe ser positivo")

    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(y_true == label) for label in np.unique(y_true)]
    differences = np.empty(n_iterations, dtype=float)
    for iteration in range(n_iterations):
        selected = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True)
            for indices in class_indices
        ])
        score_baseline = balanced_accuracy_score(
            y_true[selected], pred_baseline[selected]
        )
        score_extended = balanced_accuracy_score(
            y_true[selected], pred_extended[selected]
        )
        differences[iteration] = score_extended - score_baseline

    return {
        "mean_difference": float(np.mean(differences)),
        "lower_95": float(np.quantile(differences, 0.025)),
        "upper_95": float(np.quantile(differences, 0.975)),
    }


def holm_bonferroni(p_values) -> np.ndarray:
    """Return Holm-adjusted p-values in their original order."""
    p_values = np.asarray(p_values, dtype=float)
    if p_values.ndim != 1 or p_values.size == 0:
        raise ValueError("p_values debe ser un vector 1D no vacío")
    if not np.isfinite(p_values).all() or np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("Los p-values deben estar entre 0 y 1")

    order = np.argsort(p_values)
    factors = p_values.size - np.arange(p_values.size)
    adjusted_sorted = np.maximum.accumulate(factors * p_values[order])
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted