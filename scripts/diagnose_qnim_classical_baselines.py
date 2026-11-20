"""Screen LIGO-PCA data with classical baselines before QNIM training."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
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


def _make_model(name, seed, probability=False, mlp_max_iter=500):
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
            SVC(
                kernel="rbf",
                class_weight="balanced",
                probability=probability,
                random_state=seed,
            ),
        )
    if name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64,),
                max_iter=mlp_max_iter,
                random_state=seed,
            ),
        )
    raise ValueError(f"Unknown baseline: {name}")


def _validate_split(dataset, split, n_classes):
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


def _permuted_training_labels(labels, seed, model_index, iteration):
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), model_index, iteration])
    )
    return rng.permutation(labels)


def _mean_seed_balanced_accuracy(labels_by_seed, predictions_by_seed):
    if not labels_by_seed or len(labels_by_seed) != len(predictions_by_seed):
        raise ValueError("Provide matching labels and predictions for every seed")
    return float(np.mean([
        balanced_accuracy_score(labels, prediction)
        for labels, prediction in zip(labels_by_seed, predictions_by_seed)
    ]))


def _split_sha256(features, labels):
    digest = hashlib.sha256()
    for array in (features, labels):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _validation_decision(validation_results, chance):
    qualifying_models = [
        name for name, result in validation_results.items()
        if (
            result["mean_balanced_accuracy"] > chance
            and all(
                seed_result["balanced_accuracy"] > chance
                for seed_result in result["per_seed"]
            )
            and result["training_label_permutation"][
                "bonferroni_adjusted_p_value"
            ] <= 0.05
        )
    ]
    return {
        "eligible_for_qnim_training": bool(qualifying_models),
        "qualifying_baselines": qualifying_models,
        "criterion": (
            "Mean validation balanced accuracy is above chance, every seed's "
            "validation balanced accuracy is above chance, and the training-label "
            "permutation p-value is <= 0.05 after Bonferroni correction over "
            "the four baselines."
        ),
        "uses_final_test": False,
    }


def evaluate_datasets(datasets, seeds, n_permutations=999, evaluate_final_test=False):
    """Use validation for the advancement decision; evaluate final test only on request."""
    if not datasets or len(datasets) != len(seeds):
        raise ValueError("Provide one generated dataset for each seed")
    if n_permutations < 1:
        raise ValueError("Permutation count must be positive")
    if len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("Seeds must be unique and non-negative")

    n_classes = datasets[0].n_classes
    for dataset in datasets:
        if dataset.n_classes != n_classes:
            raise ValueError("All generated datasets must have the same classes")
        _validate_split(dataset, "train", n_classes)
        _validate_split(dataset, "val", n_classes)
        if evaluate_final_test:
            _validate_split(dataset, "test", n_classes)

    validation_results = {}
    final_test_results = {}

    for model_index, model_name in enumerate(MODEL_NAMES):
        val_scores = []
        val_labels_by_seed = []
        val_predictions_by_seed = []
        for dataset, seed in zip(datasets, seeds):
            model = _make_model(model_name, seed)
            model.fit(dataset.X_train, dataset.y_train)
            val_prediction = model.predict(dataset.X_val)
            val_labels_by_seed.append(dataset.y_val)
            val_predictions_by_seed.append(val_prediction)
            val_scores.append(
                float(balanced_accuracy_score(dataset.y_val, val_prediction))
            )
        observed_score = _mean_seed_balanced_accuracy(
            val_labels_by_seed, val_predictions_by_seed
        )

        null_scores = np.empty(n_permutations, dtype=float)
        null_scores_by_seed = []
        for iteration in range(n_permutations):
            iteration_predictions = []
            for dataset, seed in zip(datasets, seeds):
                permuted_model = _make_model(model_name, seed)
                permuted_model.fit(
                    dataset.X_train,
                    _permuted_training_labels(
                        dataset.y_train, seed, model_index, iteration
                    ),
                )
                prediction = permuted_model.predict(dataset.X_val)
                iteration_predictions.append(prediction)
            iteration_scores = [
                float(balanced_accuracy_score(dataset.y_val, prediction))
                for dataset, prediction in zip(datasets, iteration_predictions)
            ]
            null_scores_by_seed.append(iteration_scores)
            null_scores[iteration] = _mean_seed_balanced_accuracy(
                [dataset.y_val for dataset in datasets],
                iteration_predictions,
            )

        exceedances = int(np.count_nonzero(null_scores >= observed_score))
        permutation_p = float(
            (1 + exceedances) / (n_permutations + 1)
        )
        validation_results[model_name] = {
            "mean_balanced_accuracy": observed_score,
            "mean_accuracy": float(np.mean([
                accuracy_score(labels, prediction)
                for labels, prediction in zip(
                    val_labels_by_seed, val_predictions_by_seed
                )
            ])),
            "per_seed": [
                {
                    "seed": int(seed),
                    "balanced_accuracy": score,
                    "accuracy": float(accuracy_score(labels, prediction)),
                }
                for seed, score, labels, prediction in zip(
                    seeds, val_scores, val_labels_by_seed, val_predictions_by_seed
                )
            ],
            "pooled_confusion_matrix": confusion_matrix(
                np.concatenate(val_labels_by_seed),
                np.concatenate(val_predictions_by_seed),
                labels=np.arange(n_classes),
            ).tolist(),
            "training_label_permutation": {
                "iterations": n_permutations,
                "observed_mean_balanced_accuracy": observed_score,
                "null_mean_balanced_accuracy": float(np.mean(null_scores)),
                "exceedances_at_or_above_observed": exceedances,
                "p_value": permutation_p,
                "bonferroni_adjusted_p_value": min(
                    1.0, permutation_p * len(MODEL_NAMES)
                ),
                "permutation_statistics": [
                    {
                        "iteration": iteration,
                        "balanced_accuracy_by_seed": scores,
                        "mean_balanced_accuracy": float(null_scores[iteration]),
                    }
                    for iteration, scores in enumerate(null_scores_by_seed)
                ],
            },
        }

    chance = 1.0 / n_classes
    decision = _validation_decision(validation_results, chance)

    if evaluate_final_test:
        for model_name in MODEL_NAMES:
            test_labels = []
            test_predictions = []
            per_seed = []
            for dataset, seed in zip(datasets, seeds):
                model = _make_model(model_name, seed)
                model.fit(dataset.X_train, dataset.y_train)
                prediction = model.predict(dataset.X_test)
                test_labels.append(dataset.y_test)
                test_predictions.append(prediction)
                per_seed.append({
                    "seed": int(seed),
                    "accuracy": float(accuracy_score(dataset.y_test, prediction)),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(dataset.y_test, prediction)
                    ),
                })
            pooled_labels = np.concatenate(test_labels)
            pooled_predictions = np.concatenate(test_predictions)
            final_test_results[model_name] = {
                "mean_balanced_accuracy": float(np.mean([
                    item["balanced_accuracy"] for item in per_seed
                ])),
                "mean_accuracy": float(np.mean([
                    item["accuracy"] for item in per_seed
                ])),
                "per_seed": per_seed,
                "pooled_confusion_matrix": confusion_matrix(
                    pooled_labels, pooled_predictions, labels=np.arange(n_classes)
                ).tolist(),
            }

    return {
        "n_classes": n_classes,
        "chance_balanced_accuracy": chance,
        "seeds": [int(seed) for seed in seeds],
        "validation": {
            "statistic": "Mean of per-seed validation balanced accuracies.",
            "permutation_test": (
                "Independently permute each seed's training labels, refit, and "
                "evaluate on unchanged validation labels; aggregate with the "
                "same mean-per-seed balanced accuracy statistic."
            ),
            "multiple_comparison_correction": (
                f"Bonferroni over {len(MODEL_NAMES)} baselines; family-wise alpha=0.05."
            ),
            "models": validation_results,
        },
        "final_test": {
            "evaluated": evaluate_final_test,
            "status": (
                "evaluated_after_freezing_protocol"
                if evaluate_final_test else "locked_not_evaluated"
            ),
            "models": final_test_results,
        },
        "decision": decision,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run strict PyCBC generation and classical LIGO-PCA baselines."
    )
    parser.add_argument("--train-per-class", type=int, default=200)
    parser.add_argument("--validation-per-class", type=int, default=50)
    parser.add_argument("--test-per-class", type=int, default=100)
    parser.add_argument("--components", type=int, default=12)
    parser.add_argument("--pca-fit-components", type=int, default=64)
    parser.add_argument("--approximant", default="IMRPhenomPv2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument(
        "--evaluate-final-test",
        action="store_true",
        help="Evaluate the locked final test after freezing all model and protocol choices.",
    )
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
    ) < 1:
        raise ValueError("Dataset sizes, PCA dimensions, and iteration counts must be positive")
    if args.pca_fit_components < args.components:
        raise ValueError("PCA fit components must be >= retained components")
    if len(args.seeds) != 3:
        raise ValueError("The pre-registered diagnostic protocol requires exactly three seeds")
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        raise ValueError("Seeds must be unique and non-negative")

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
            "train_sha256": _split_sha256(dataset.X_train, dataset.y_train),
            "validation_sha256": _split_sha256(dataset.X_val, dataset.y_val),
            "train_class_counts": np.bincount(
                dataset.y_train, minlength=dataset.n_classes
            ).tolist(),
            "validation_class_counts": np.bincount(
                dataset.y_val, minlength=dataset.n_classes
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
        "pre_registered_protocol": {
            "seeds": args.seeds,
            "permutations": args.permutations,
            "default_permutations": 999,
            "permutation_model_refits": args.permutations * len(args.seeds),
            "permutation_model_refits_total": (
                args.permutations * len(args.seeds) * len(MODEL_NAMES)
            ),
            "eligibility_split": "validation_only",
            "final_test_requires_explicit_flag": True,
            "seed_consistency": "Every seed's validation balanced accuracy must exceed chance.",
            "significance_threshold": 0.05,
            "multiple_comparison_correction": "Bonferroni over four baselines",
            "test_evaluation_requested": args.evaluate_final_test,
        },
        "randomness": {
            "data_generation_and_pca": "NumPy default_rng(seed); PCA random_state=seed.",
            "models": "Each model receives its dataset seed as random_state where supported.",
            "permutations": (
                "Each seed/model/iteration uses an independent NumPy "
                "SeedSequence([dataset_seed, model_index, permutation_iteration])."
            ),
            "model_index_order": list(MODEL_NAMES),
        },
        "software_versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "PyCBC": importlib.metadata.version("PyCBC"),
            "LALSuite": importlib.metadata.version("lalsuite"),
        },
        "datasets": dataset_summaries,
        **evaluate_datasets(
            datasets,
            args.seeds,
            n_permutations=args.permutations,
            evaluate_final_test=args.evaluate_final_test,
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["decision"], indent=2))
    print(f"Full diagnostic report written to {output}")


if __name__ == "__main__":
    main()
