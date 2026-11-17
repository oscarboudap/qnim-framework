import argparse
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.diagnose_qnim_classical_baselines import (
    MODEL_NAMES,
    _parse_seeds,
    evaluate_datasets,
)


def _dataset(seed):
    rng = np.random.default_rng(seed)
    class_centers = np.array([[-1.0, -1.0], [1.0, 1.0]])

    def split(count_per_class):
        features = np.concatenate([
            rng.normal(center, 0.25, size=(count_per_class, 2))
            for center in class_centers
        ])
        labels = np.repeat(np.arange(2), count_per_class)
        return features, labels

    x_train, y_train = split(8)
    x_val, y_val = split(3)
    x_test, y_test = split(4)
    return SimpleNamespace(
        n_classes=2,
        X_train=x_train,
        y_train=y_train,
        X_val=x_val,
        y_val=y_val,
        X_test=x_test,
        y_test=y_test,
    )


def test_seed_parser_rejects_duplicates_and_negative_values():
    assert _parse_seeds("4, 9") == [4, 9]

    for invalid in ("", "4,4", "-1,2", "not-a-seed"):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_seeds(invalid)


def test_evaluator_fits_four_baselines_and_runs_training_label_permutations():
    report = evaluate_datasets(
        [_dataset(1), _dataset(2)],
        seeds=[1, 2],
        n_permutations=2,
        n_bootstrap=10,
    )

    assert set(report["models"]) == set(MODEL_NAMES)
    assert report["n_classes"] == 2
    for model_report in report["models"].values():
        assert model_report["test_confusion_matrix"] and len(
            model_report["test_confusion_matrix"]
        ) == 2
        assert model_report["training_label_permutation"]["iterations"] == 2
        assert len(model_report["test_balanced_accuracy_ci95"]) == 2
