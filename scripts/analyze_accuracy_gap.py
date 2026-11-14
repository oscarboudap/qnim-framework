"""Diagnose chance-level accuracy in the paired-injection scaling dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binom
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _models(seed: int) -> dict:
    return {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
        ),
        "svc_rbf": make_pipeline(StandardScaler(), SVC(class_weight="balanced")),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }


def _load_split(dataset_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    X_path = dataset_dir / f"X_{split}.npy"
    y_path = dataset_dir / f"y_{split}.npy"
    if not X_path.is_file() or not y_path.is_file():
        raise FileNotFoundError(f"Faltan los arrays {X_path} o {y_path}")
    X = np.load(X_path, allow_pickle=False)
    y = np.load(y_path, allow_pickle=False)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(y) == 0:
        raise ValueError(f"El split {split} debe tener X 2D e y 1D alineados")
    if not np.isfinite(X).all():
        raise ValueError(f"X_{split} contiene NaN o infinitos")
    return X, y


def _validate_pair_layout(y: np.ndarray, n_classes: int, split: str) -> int:
    if len(y) % n_classes:
        raise ValueError(f"El split {split} no contiene pares completos de {n_classes} clases")
    grouped = y.reshape(-1, n_classes)
    expected = np.tile(np.arange(n_classes), (len(grouped), 1))
    if not np.array_equal(grouped, expected):
        raise ValueError(
            f"El split {split} no sigue el orden emparejado clase 0..{n_classes - 1}"
        )
    return len(grouped)


def _block_bootstrap(
    correct_by_pair: np.ndarray,
    iterations: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    scores = np.empty(iterations, dtype=float)
    n_pairs = len(correct_by_pair)
    for index in range(iterations):
        selected = rng.integers(0, n_pairs, size=n_pairs)
        scores[index] = correct_by_pair[selected].mean()
    return np.quantile(scores, [0.025, 0.975]).astype(float).tolist()


def _within_pair_permutation_p(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    iterations: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    n_classes = len(np.unique(y_true))
    truth_by_pair = y_true.reshape(-1, n_classes)
    pred_by_pair = y_pred.reshape(-1, n_classes)
    observed_correct = int(np.count_nonzero(truth_by_pair == pred_by_pair))
    null_correct = np.empty(iterations, dtype=np.int64)
    for index in range(iterations):
        shuffled = np.stack([rng.permutation(labels) for labels in truth_by_pair])
        null_correct[index] = np.count_nonzero(shuffled == pred_by_pair)
    return float((1 + np.count_nonzero(null_correct >= observed_correct)) / (iterations + 1))


def _resource_projection(n_classes: int) -> dict:
    chance = 1.0 / n_classes
    alpha = 0.05
    targets = (0.10, 0.15, 0.20)
    rows = []
    for per_class in (10, 30, 50, 100, 300, 900):
        n = n_classes * per_class
        critical = int(binom.ppf(1 - alpha, n, chance)) + 1
        rows.append({
            "n_test_per_class": per_class,
            "n_total": n,
            "critical_correct_for_one_sided_alpha_0_05": critical,
            "critical_accuracy": critical / n,
            "power": {
                f"true_accuracy_{target:.2f}": float(binom.sf(critical - 1, n, target))
                for target in targets
            },
        })

    minimum_sample_sizes = {}
    for target in targets:
        low, high = 1, 10000
        while low < high:
            middle = (low + high) // 2
            n = n_classes * middle
            critical = int(binom.ppf(1 - alpha, n, chance)) + 1
            if binom.sf(critical - 1, n, target) >= 0.80:
                high = middle
            else:
                low = middle + 1
        minimum_sample_sizes[f"true_accuracy_{target:.2f}"] = low

    shots = (64, 512, 1024, 4096, 16384)
    return {
        "assumption": (
            f"Independent per-event correctness with a balanced {n_classes}-class test set; "
            "optimistic when events share a noise segment."
        ),
        "alpha_one_sided": alpha,
        "power_table": rows,
        "minimum_test_events_per_class_for_80_percent_power": minimum_sample_sizes,
        "worst_case_shot_probability_standard_error": {
            str(count): float(0.5 / np.sqrt(count)) for count in shots
        },
    }


def _evaluate_feature_source(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    train_pairs: int,
    val_pairs: int,
    bootstrap_iterations: int,
    permutation_iterations: int,
    seed: int,
) -> dict:
    train_groups = np.repeat(np.arange(train_pairs), n_classes)
    val_groups = np.repeat(np.arange(val_pairs), n_classes)
    cv = GroupKFold(n_splits=5)
    cv_results = {}
    validation_results = {}
    correct_by_model = {}

    for name, model in _models(seed).items():
        cv_pred = cross_val_predict(
            model, X_train, y_train, groups=train_groups, cv=cv, n_jobs=1
        )
        cv_results[name] = float(balanced_accuracy_score(y_train, cv_pred))

        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        score = float(balanced_accuracy_score(y_val, predictions))
        correct = (predictions.reshape(val_pairs, n_classes)
                   == y_val.reshape(val_pairs, n_classes)).astype(float)
        correct_by_model[name] = correct
        validation_results[name] = {
            "balanced_accuracy": score,
            "group_bootstrap_95_ci": _block_bootstrap(
                correct, bootstrap_iterations, seed
            ),
            "within_pair_permutation_p_unadjusted": _within_pair_permutation_p(
                y_val, predictions, permutation_iterations, seed
            ),
        }

    return {
        "grouped_5_fold_cv_balanced_accuracy": cv_results,
        "validation": validation_results,
        "best_model_by_training_cv": max(cv_results, key=cv_results.get),
    }


def analyze(
    dataset_dir: Path,
    bootstrap_iterations: int,
    permutation_iterations: int,
    seed: int,
) -> dict:
    X_train, y_train = _load_split(dataset_dir, "train")
    X_val, y_val = _load_split(dataset_dir, "val")
    classes = np.unique(y_train)
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("Las etiquetas de train deben ser consecutivas desde cero")
    n_classes = len(classes)
    if not np.array_equal(np.unique(y_val), classes):
        raise ValueError("Train y validation deben contener las mismas clases")
    train_pairs = _validate_pair_layout(y_train, n_classes, "train")
    val_pairs = _validate_pair_layout(y_val, n_classes, "val")

    digest = hashlib.sha256()
    for name, array in (
        ("X_train", X_train), ("y_train", y_train),
        ("X_val", X_val), ("y_val", y_val),
    ):
        digest.update(name.encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())

    feature_sources = {"pca": (X_train, X_val)}
    physical_paths = (
        dataset_dir / "X_physical_train.npy",
        dataset_dir / "X_physical_val.npy",
    )
    if all(path.is_file() for path in physical_paths):
        feature_sources["physical"] = (
            np.load(physical_paths[0], allow_pickle=False),
            np.load(physical_paths[1], allow_pickle=False),
        )
    global_template_paths = (
        dataset_dir / "X_global_template_train.npy",
        dataset_dir / "X_global_template_val.npy",
    )
    if all(path.is_file() for path in global_template_paths):
        feature_sources["global_template"] = (
            np.load(global_template_paths[0], allow_pickle=False),
            np.load(global_template_paths[1], allow_pickle=False),
        )

    results = {}
    for source, (Xtr, Xv) in feature_sources.items():
        if Xtr.shape[1] != Xv.shape[1]:
            raise ValueError(f"Train y validation tienen distintas features para {source}")
        if len(Xtr) != len(y_train) or len(Xv) != len(y_val):
            raise ValueError(f"Features y etiquetas no están alineadas para {source}")
        paired_train = Xtr.reshape(train_pairs, n_classes, -1)
        paired_val = Xv.reshape(val_pairs, n_classes, -1)
        delta_train = (paired_train - paired_train[:, :1, :]).reshape(len(Xtr), -1)
        delta_val = (paired_val - paired_val[:, :1, :]).reshape(len(Xv), -1)
        results[source] = {
            "single_event_features": _evaluate_feature_source(
                Xtr, y_train, Xv, y_val, n_classes, train_pairs, val_pairs,
                bootstrap_iterations, permutation_iterations, seed,
            ),
            "paired_control_delta_diagnostic": _evaluate_feature_source(
                delta_train, y_train, delta_val, y_val, n_classes,
                train_pairs, val_pairs, bootstrap_iterations,
                permutation_iterations, seed,
            ),
        }

    metadata_path = dataset_dir / "pca_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    matched_filter_result = None
    matched_filter_path = dataset_dir / "X_matched_filter_val.npy"
    if matched_filter_path.is_file():
        matched_scores = np.load(matched_filter_path, allow_pickle=False)
        if (
            matched_scores.ndim != 2
            or matched_scores.shape != (len(y_val), n_classes)
            or not np.isfinite(matched_scores).all()
        ):
            raise ValueError(
                "X_matched_filter_val debe ser una matriz finita (n_validation, n_classes)"
            )
        matched_predictions = matched_scores.argmax(axis=1)
        matched_correct = (
            matched_predictions.reshape(val_pairs, n_classes)
            == y_val.reshape(val_pairs, n_classes)
        ).astype(float)
        matched_filter_result = {
            "validation": {
                "accuracy": float(np.mean(matched_predictions == y_val)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y_val, matched_predictions)
                ),
                "group_bootstrap_95_ci": _block_bootstrap(
                    matched_correct, bootstrap_iterations, seed
                ),
                "within_pair_permutation_p_unadjusted": _within_pair_permutation_p(
                    y_val,
                    matched_predictions,
                    permutation_iterations,
                    seed,
                ),
                "confusion_matrix": confusion_matrix(
                    y_val, matched_predictions
                ).tolist(),
            },
            "selection": metadata.get("matched_filter_bank"),
        }
    return {
        "dataset_dir": str(dataset_dir),
        "dataset_sha256_train_val": digest.hexdigest(),
        "n_classes": n_classes,
        "random_baseline": 1.0 / n_classes,
        "n_train_pairs": train_pairs,
        "n_validation_pairs": val_pairs,
        "n_test_per_class_from_metadata": metadata.get("n_test_per_class"),
        "test_arrays_read": False,
        "snr_normalization": metadata.get("paired_injections", {}).get(
            "snr_normalization", "reference_gr"
        ),
        "paired_reference_is_oracle": metadata.get("physical_features", {}).get(
            "paired_reference_is_oracle"
        ),
        "bootstrap_iterations": bootstrap_iterations,
        "permutation_iterations": permutation_iterations,
        "seed": seed,
        "feature_results": results,
        "matched_filter_bank": matched_filter_result,
        "resource_projection_binomial_approximation": _resource_projection(n_classes),
        "interpretation": (
            "The paired-control delta subtracts each theory feature vector from "
            "the GR feature vector for the same source/noise pair. This oracle "
            "control is a diagnostic only, not a deployable single-event input. "
            "The global_template source uses one GR mean template fit on train "
            "and does not use a paired reference. "
            "Validation statistics are exploratory and unadjusted."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=20000)
    parser.add_argument("--permutation-iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()
    if args.bootstrap_iterations < 1 or args.permutation_iterations < 1:
        raise ValueError("Las iteraciones de bootstrap/permutación deben ser positivas")

    result = analyze(
        args.dataset_dir,
        args.bootstrap_iterations,
        args.permutation_iterations,
        args.seed,
    )
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output is None:
        print(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"Análisis guardado en {args.output.resolve()}")


if __name__ == "__main__":
    main()
