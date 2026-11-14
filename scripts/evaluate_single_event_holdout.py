"""Evaluate one fixed classical baseline on an untouched paired-injection test split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score


def _validate_pair_layout(y: np.ndarray, n_classes: int, split: str) -> int:
    if y.ndim != 1 or len(y) == 0 or len(y) % n_classes:
        raise ValueError(f"El split {split} no contiene pares completos")
    grouped = y.reshape(-1, n_classes)
    if not np.array_equal(grouped, np.tile(np.arange(n_classes), (len(grouped), 1))):
        raise ValueError(f"Las etiquetas de {split} no están ordenadas por pareja")
    return len(grouped)


def _paired_permutation_p(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    iterations: int,
    seed: int,
) -> float:
    truth = y_true.reshape(-1, n_classes)
    predictions = y_pred.reshape(-1, n_classes)
    observed_correct = int(np.count_nonzero(truth == predictions))
    rng = np.random.default_rng(seed)
    null_correct = np.empty(iterations, dtype=np.int64)
    for index in range(iterations):
        shuffled = np.stack([rng.permutation(row) for row in truth])
        null_correct[index] = np.count_nonzero(shuffled == predictions)
    return float((1 + np.count_nonzero(null_correct >= observed_correct)) / (iterations + 1))


def evaluate(
    dataset_dir: Path,
    iterations: int,
    seed: int,
    feature_source: str = "pca",
) -> dict:
    feature_files = {
        "pca": ("X_train.npy", "X_test.npy"),
        "global_template": (
            "X_global_template_train.npy",
            "X_global_template_test.npy",
        ),
    }
    if feature_source not in feature_files:
        raise ValueError(f"Fuente de features desconocida: {feature_source}")

    metadata_path = dataset_dir / "pca_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    split_segments = metadata.get("noise_segments")
    if not isinstance(split_segments, dict) or not {"train", "val", "test"} <= set(split_segments):
        raise ValueError("Se requiere metadata de segmentos train, val y test")
    segment_sets = {split: set(split_segments[split]) for split in ("train", "val", "test")}
    if (
        segment_sets["train"] & segment_sets["val"]
        or segment_sets["train"] & segment_sets["test"]
        or segment_sets["val"] & segment_sets["test"]
    ):
        raise ValueError("Los segmentos de ruido deben ser disjuntos entre splits")

    train_file, test_file = feature_files[feature_source]
    X_train = np.load(dataset_dir / train_file, allow_pickle=False)
    y_train = np.load(dataset_dir / "y_train.npy", allow_pickle=False)
    if X_train.ndim != 2 or y_train.ndim != 1 or len(X_train) != len(y_train):
        raise ValueError("Train requiere X 2D e y 1D alineados")
    classes = np.unique(y_train)
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("Las etiquetas de train deben ser consecutivas desde cero")
    n_classes = len(classes)
    train_pairs = _validate_pair_layout(y_train, n_classes, "train")

    model = ExtraTreesClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    X_test = np.load(dataset_dir / test_file, allow_pickle=False)
    y_test = np.load(dataset_dir / "y_test.npy", allow_pickle=False)
    if X_test.ndim != 2 or X_test.shape[1] != X_train.shape[1]:
        raise ValueError("X_test debe tener las mismas features que X_train")
    if y_test.ndim != 1 or len(X_test) != len(y_test):
        raise ValueError("Test requiere X 2D e y 1D alineados")
    if not np.array_equal(np.unique(y_test), classes):
        raise ValueError("Test debe contener todas las clases de train")
    test_pairs = _validate_pair_layout(y_test, n_classes, "test")
    test_counts = np.bincount(y_test, minlength=n_classes)
    if not np.all(test_counts == test_counts[0]):
        raise ValueError("El test debe estar balanceado por clase")

    predictions = model.predict(X_test)
    truth_by_pair = y_test.reshape(test_pairs, n_classes)
    predictions_by_pair = predictions.reshape(test_pairs, n_classes)
    correct_by_pair = (truth_by_pair == predictions_by_pair).astype(float)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(iterations, dtype=float)
    for index in range(iterations):
        selected = rng.integers(0, test_pairs, size=test_pairs)
        bootstrap[index] = correct_by_pair[selected].mean()

    digest = hashlib.sha256()
    for name, array in (("X_test", X_test), ("y_test", y_test)):
        digest.update(name.encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())

    return {
        "dataset_dir": str(dataset_dir),
        "dataset_test_sha256": digest.hexdigest(),
        "feature_source": feature_source,
        "model": "ExtraTreesClassifier(n_estimators=300,class_weight=balanced,random_state=42)",
        "n_train": int(len(y_train)),
        "n_train_pairs": train_pairs,
        "n_test": int(len(y_test)),
        "n_test_pairs": test_pairs,
        "n_classes": n_classes,
        "test_per_class": test_counts.tolist(),
        "noise_segments_disjoint": True,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "chance": 1.0 / n_classes,
        "group_bootstrap_95_ci": np.quantile(bootstrap, [0.025, 0.975]).astype(float).tolist(),
        "within_pair_permutation_p_unadjusted": _paired_permutation_p(
            y_test, predictions, n_classes, iterations, seed
        ),
        "permutation_iterations": iterations,
        "predicted_class_counts": np.bincount(predictions, minlength=n_classes).tolist(),
        "test_used_for_training_or_tuning": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--feature-source",
        choices=("pca", "global_template"),
        default="pca",
    )
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations debe ser positivo")
    result = evaluate(
        args.dataset_dir,
        args.iterations,
        args.seed,
        feature_source=args.feature_source,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Test guardado en {args.output.resolve()}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
