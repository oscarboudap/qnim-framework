import sys
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score

from scripts.diagnose_qnim_classical_baselines import (
    MODEL_NAMES,
    _mean_seed_balanced_accuracy,
    _permuted_training_labels,
    _split_sha256,
    _validation_decision,
    evaluate_datasets,
    parse_args,
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


def test_default_protocol_is_predeclared(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["diagnose_qnim_classical_baselines.py"])

    args = parse_args()

    assert args.train_per_class == 200
    assert args.validation_per_class == 50
    assert args.test_per_class == 100
    assert args.seeds == [42, 43, 44]
    assert args.permutations == 999
    assert args.evaluate_final_test is False


def test_seed_cli_accepts_space_separated_seed_values(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["diagnose_qnim_classical_baselines.py", "--seeds", "42", "43", "44"],
    )

    assert parse_args().seeds == [42, 43, 44]


def test_permutation_seed_is_reproducible_and_independent_by_dataset_seed():
    labels = np.tile(np.arange(4), 16)

    first = _permuted_training_labels(labels, seed=42, model_index=1, iteration=7)
    repeat = _permuted_training_labels(labels, seed=42, model_index=1, iteration=7)
    other_seed = _permuted_training_labels(labels, seed=43, model_index=1, iteration=7)

    np.testing.assert_array_equal(first, repeat)
    assert not np.array_equal(first, other_seed)
    np.testing.assert_array_equal(np.bincount(first), np.bincount(labels))


def test_permutation_statistic_is_mean_of_per_seed_balanced_accuracies():
    labels_by_seed = [
        np.array([0] * 9 + [1]),
        np.array([0] + [1] * 9),
    ]
    predictions_by_seed = [
        np.zeros(10, dtype=int),
        np.ones(10, dtype=int),
    ]

    observed = _mean_seed_balanced_accuracy(labels_by_seed, predictions_by_seed)

    assert observed == pytest.approx(0.5)
    assert observed != pytest.approx(
        balanced_accuracy_score(
            np.concatenate(labels_by_seed),
            np.concatenate(predictions_by_seed),
        )
    )


def test_split_fingerprint_is_stable_and_sensitive_to_data():
    dataset = _dataset(8)

    fingerprint = _split_sha256(dataset.X_train, dataset.y_train)

    assert fingerprint == _split_sha256(dataset.X_train.copy(), dataset.y_train.copy())
    assert fingerprint != _split_sha256(dataset.X_val, dataset.y_val)


def test_evaluator_fits_four_baselines_and_runs_training_label_permutations():
    report = evaluate_datasets(
        [_dataset(1), _dataset(2)],
        seeds=[1, 2],
        n_permutations=2,
    )

    assert set(report["validation"]["models"]) == set(MODEL_NAMES)
    assert report["n_classes"] == 2
    assert report["final_test"] == {
        "evaluated": False,
        "status": "locked_not_evaluated",
        "models": {},
    }
    assert report["decision"]["uses_final_test"] is False
    for model_report in report["validation"]["models"].values():
        assert len(model_report["per_seed"]) == 2
        assert model_report["mean_balanced_accuracy"] == pytest.approx(
            np.mean([
                item["balanced_accuracy"] for item in model_report["per_seed"]
            ])
        )
        assert model_report["training_label_permutation"]["iterations"] == 2
        assert model_report["training_label_permutation"]["p_value"] >= 1 / 3
        assert len(model_report["pooled_confusion_matrix"]) == 2
        permutation_result = model_report["training_label_permutation"]
        statistics = permutation_result["permutation_statistics"]
        assert len(statistics) == 2
        for statistic in statistics:
            assert statistic["mean_balanced_accuracy"] == pytest.approx(
                np.mean(statistic["balanced_accuracy_by_seed"])
            )
        assert permutation_result["p_value"] == pytest.approx(
            (1 + permutation_result["exceedances_at_or_above_observed"]) / 3
        )


def test_validation_decision_requires_every_seed_to_beat_chance():
    model_result = {
        "mean_balanced_accuracy": 0.6,
        "per_seed": [
            {"balanced_accuracy": 0.7},
            {"balanced_accuracy": 0.6},
            {"balanced_accuracy": 0.49},
        ],
        "training_label_permutation": {
            "bonferroni_adjusted_p_value": 0.04,
        },
    }

    decision = _validation_decision({"baseline": model_result}, chance=0.5)
    assert decision["eligible_for_qnim_training"] is False
    assert decision["uses_final_test"] is False

    model_result["per_seed"][-1]["balanced_accuracy"] = 0.51
    decision = _validation_decision({"baseline": model_result}, chance=0.5)
    assert decision["eligible_for_qnim_training"] is True


def test_validation_gate_never_reads_final_test():
    class LockedTestDataset:
        def __init__(self, dataset):
            self.dataset = dataset

        def __getattr__(self, name):
            if name in {"X_test", "y_test"}:
                raise AssertionError("Validation decision accessed the locked test")
            return getattr(self.dataset, name)

    report = evaluate_datasets(
        [LockedTestDataset(_dataset(3))],
        seeds=[3],
        n_permutations=1,
    )

    assert report["final_test"]["status"] == "locked_not_evaluated"
    assert report["decision"]["uses_final_test"] is False


def test_evaluator_is_reproducible_for_fixed_data_seeds():
    dataset = _dataset(6)

    first = evaluate_datasets([dataset], seeds=[6], n_permutations=2)
    second = evaluate_datasets([dataset], seeds=[6], n_permutations=2)

    assert first["validation"] == second["validation"]


def test_final_test_metrics_are_separate_and_opt_in():
    report = evaluate_datasets(
        [_dataset(4)],
        seeds=[4],
        n_permutations=1,
        evaluate_final_test=True,
    )

    assert report["final_test"]["evaluated"] is True
    assert report["final_test"]["status"] == "evaluated_after_freezing_protocol"
    assert set(report["final_test"]["models"]) == set(MODEL_NAMES)
    assert report["decision"]["uses_final_test"] is False
