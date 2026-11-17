"""Screen LIGO-PCA data with classical baselines before QNIM training."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_NAMES = (
    "logistic_regression",
    "random_forest",
    "svm_rbf",
    "mlp",
)


def _make_model(name, seed):
    if name == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=seed,
            ),
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "svm_rbf":
        return make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", class_weight="balanced"),
        )
    if name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64,),
                max_iter=500,
                random_state=seed,
            ),
        )
    raise ValueError(f"Unknown baseline: {name}")


def _validate_dataset(dataset, n_classes):
    for split in ("train", "val", "test"):
        features = np.asarray(getattr(dataset, f"X_{split}"))
        labels = np.asarray(getattr(dataset, f"y_{split}"))
        if features.ndim != 2 or labels.ndim != 1:
            raise ValueError(f"{split} must contain a 2D feature matrix and 1D labels")
        if len(features) != len(labels) or not np.isfinite(features).all():
            raise ValueError(f"{split} has mismatched rows or non-finite features")
        integer_labels = labels.astype(int)
        if not np.array_equal(labels, integer_labels):
            raise ValueError(f"{split} labels must be integers")
        if np.any(integer_labels < 0) or np.any(integer_labels >= n_classes):
            raise ValueError(f"{split} contains an out-of-range class label")
        counts = np.bincount(integer_labels, minlength=n_classes)
        if len(counts) != n_classes or np.any(counts == 0):
            raise ValueError(f"{split} must contain every class")


def _stratified_bootstrap_ci(y_true, prediction, n_classes, iterations, seed):
    rng = np.random.default_rng(seed)
    class_indices = [
        np.flatnonzero(y_true == label) for label in range(n_classes)
    ]
    scores = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        indices = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True)
            for indices in class_indices
        ])
        scores[iteration] = balanced_accuracy_score(
            y_true[indices], prediction[indices]
        )
    return [
        float(np.quantile(scores, 0.025)),
        float(np.quantile(scores, 0.975)),
    ]


def evaluate_datasets(datasets, seeds, n_permutations=199, n_bootstrap=2000):
    """Fit fixed baselines on train, report validation, and test once."""
    if not datasets or len(datasets) != len(seeds):
        raise ValueError("Provide one generated dataset for each seed")
    if n_permutations < 1 or n_bootstrap < 1:
        raise ValueError("Permutation and bootstrap counts must be positive")

    n_classes = datasets[0].n_classes
    for dataset in datasets:
        if dataset.n_classes != n_classes:
            raise ValueError("All generated datasets must have the same classes")
        _validate_dataset(dataset, n_classes)

    test_labels = np.concatenate([dataset.y_test for dataset in datasets])
    val_labels = np.concatenate([dataset.y_val for dataset in datasets])
    labels = np.arange(n_classes)
    results = {}

    for model_index, model_name in enumerate(MODEL_NAMES):
        test_predictions = []
        val_predictions = []
        per_seed = []
        for dataset, seed in zip(datasets, seeds):
            model = _make_model(model_name, seed)
            model.fit(dataset.X_train, dataset.y_train)
            test_prediction = model.predict(dataset.X_test)
            val_prediction = model.predict(dataset.X_val)
            test_predictions.append(test_prediction)
            val_predictions.append(val_prediction)
            per_seed.append({
                "seed": int(seed),
                "test_balanced_accuracy": float(
                    balanced_accuracy_score(dataset.y_test, test_prediction)
                ),
                "validation_balanced_accuracy": float(
                    balanced_accuracy_score(dataset.y_val, val_prediction)
                ),
            })

        pooled_test_prediction = np.concatenate(test_predictions)
        pooled_val_prediction = np.concatenate(val_predictions)
        observed_score = float(
            balanced_accuracy_score(test_labels, pooled_test_prediction)
        )

        rng = np.random.default_rng(7919 + model_index)
        null_scores = np.empty(n_permutations, dtype=float)
        for iteration in range(n_permutations):
            iteration_predictions = []
            for dataset, seed in zip(datasets, seeds):
                permuted_model = _make_model(model_name, seed)
                permuted_model.fit(
                    dataset.X_train,
                    rng.permutation(dataset.y_train),
                )
                iteration_predictions.append(
                    permuted_model.predict(dataset.X_test)
                )
            null_scores[iteration] = balanced_accuracy_score(
                test_labels, np.concatenate(iteration_predictions)
            )

        permutation_p = float(
            (1 + np.count_nonzero(null_scores >= observed_score))
            / (n_permutations + 1)
        )
        results[model_name] = {
            "test_accuracy": float(accuracy_score(test_labels, pooled_test_prediction)),
            "test_balanced_accuracy": observed_score,
            "test_balanced_accuracy_ci95": _stratified_bootstrap_ci(
                test_labels,
                pooled_test_prediction,
                n_classes,
                n_bootstrap,
                seed=104729 + model_index,
            ),
            "validation_balanced_accuracy": float(
                balanced_accuracy_score(val_labels, pooled_val_prediction)
            ),
            "test_confusion_matrix": confusion_matrix(
                test_labels, pooled_test_prediction, labels=labels
            ).tolist(),
            "per_seed": per_seed,
            "training_label_permutation": {
                "iterations": n_permutations,
                "null_balanced_accuracy_mean": float(np.mean(null_scores)),
                "p_value": permutation_p,
                "bonferroni_p_value_over_four_baselines": min(
                    1.0, permutation_p * len(MODEL_NAMES)
                ),
            },
        }

    chance = 1.0 / n_classes
    eligible_models = [
        name for name, result in results.items()
        if (
            result["test_balanced_accuracy_ci95"][0] > chance
            and result["training_label_permutation"][
                "bonferroni_p_value_over_four_baselines"
            ] <= 0.05
        )
    ]
    return {
        "n_classes": n_classes,
        "chance_balanced_accuracy": chance,
        "seeds": [int(seed) for seed in seeds],
        "permutation_test": (
            "Train labels shuffled and each model refit; test labels remain unchanged."
        ),
        "multiple_comparison_correction": "Bonferroni over four baselines",
        "models": results,
        "decision": {
            "eligible_for_qnim_training": bool(eligible_models),
            "qualifying_baselines": eligible_models,
            "criterion": (
                "At least one baseline has a stratified-bootstrap 95% CI lower "
                "bound above chance and a Bonferroni-corrected permutation p-value <= 0.05."
            ),
        },
    }


def _parse_seeds(value):
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Seeds must be comma-separated integers") from exc
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise argparse.ArgumentTypeError("Provide unique, non-negative seeds")
    return seeds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run strict PyCBC generation and classical LIGO-PCA baselines."
    )
    parser.add_argument("--train-per-class", type=int, default=20)
    parser.add_argument("--validation-per-class", type=int, default=5)
    parser.add_argument("--test-per-class", type=int, default=10)
    parser.add_argument("--components", type=int, default=12)
    parser.add_argument("--pca-fit-components", type=int, default=64)
    parser.add_argument("--approximant", default="IMRPhenomPv2")
    parser.add_argument("--seeds", type=_parse_seeds, default=[42, 43, 44])
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument(
        "--output",
        default="results/scaling_study/qnim_classical_diagnostic.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if min(
        args.train_per_class,
        args.validation_per_class,
        args.test_per_class,
        args.components,
        args.pca_fit_components,
        args.permutations,
        args.bootstrap_iterations,
    ) < 1:
        raise ValueError("Dataset sizes, PCA dimensions, and iteration counts must be positive")
    if args.pca_fit_components < args.components:
        raise ValueError("PCA fit components must be >= retained components")

    from src.infrastructure.ligo_pca_adapter import LIGOPCAAdapter

    datasets = []
    dataset_summaries = []
    total_events = (
        args.train_per_class
        + args.validation_per_class
        + args.test_per_class
    ) * 13 * len(args.seeds)
    print(f"Strict PyCBC diagnostic: generating {total_events} events.")
    for seed in args.seeds:
        adapter = LIGOPCAAdapter(
            n_components=args.components,
            n_pca_fit=args.pca_fit_components,
            approximant=args.approximant,
            max_classes=13,
            strict_physics=True,
        )
        dataset = adapter.generate_balanced_dataset(
            n_per_class=args.train_per_class,
            n_val_per_class=args.validation_per_class,
            n_test_per_class=args.test_per_class,
            seed=seed,
            verbose=True,
        )
        datasets.append(dataset)
        dataset_summaries.append({
            "seed": seed,
            "class_names": dataset.theory_names,
            "train_shape": list(dataset.X_train.shape),
            "validation_shape": list(dataset.X_val.shape),
            "test_shape": list(dataset.X_test.shape),
            "train_class_counts": np.bincount(
                dataset.y_train, minlength=dataset.n_classes
            ).tolist(),
            "validation_class_counts": np.bincount(
                dataset.y_val, minlength=dataset.n_classes
            ).tolist(),
            "test_class_counts": np.bincount(
                dataset.y_test, minlength=dataset.n_classes
            ).tolist(),
            "pca_fit_split": "train_only",
            "scaler_fit_split": "train_only",
            "pca_explained_variance_retained": dataset.metadata["var_explained"],
        })

    report = {
        "task": "13-class LIGO-PCA classical baseline screen",
        "waveform_source": "PyCBC",
        "strict_physics": True,
        "fallback_allowed": False,
        "approximant": args.approximant,
        "detector_noise_included": False,
        "snr": None,
        "scientific_scope_warning": (
            "This generator uses whitened PyCBC waveforms but does not inject "
            "detector noise; a positive result is a screening signal, not evidence "
            "of separability in real detector data."
        ),
        "dataset_configuration": {
            "train_per_class": args.train_per_class,
            "validation_per_class": args.validation_per_class,
            "test_per_class": args.test_per_class,
            "components": args.components,
            "pca_fit_components": args.pca_fit_components,
        },
        "datasets": dataset_summaries,
        **evaluate_datasets(
            datasets,
            args.seeds,
            n_permutations=args.permutations,
            n_bootstrap=args.bootstrap_iterations,
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["decision"], indent=2))
    print(f"Full diagnostic report written to {output}")


if __name__ == "__main__":
    main()
