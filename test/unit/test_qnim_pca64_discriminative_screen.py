import numpy as np

from scripts.run_qnim_pca64_discriminative_screen import (
    _angle_summary,
    _classification_metrics,
    _paired_changes,
)


def test_classification_metrics_keep_fixed_class_order():
    y_true = np.array([0, 0, 1, 1])
    predictions = np.array([0, 1, 1, 1])
    probabilities = np.array(
        [[0.8, 0.2], [0.4, 0.6], [0.1, 0.9], [0.2, 0.8]]
    )

    metrics = _classification_metrics(
        y_true, predictions, probabilities, ["A", "B"]
    )

    assert metrics["balanced_accuracy"] == 0.75
    assert metrics["global_accuracy"] == 0.75
    assert metrics["per_class_recall"] == {"A": 0.5, "B": 1.0}
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]


def test_angle_summary_quantiles_cover_pooled_principal_angles():
    summary = _angle_summary([[1.0, 2.0], [3.0, 4.0]])

    assert summary["pooled_principal_angle_percentiles_degrees"]["median"] == 2.5
    assert summary["pooled_principal_angle_percentiles_degrees"]["maximum"] == 4.0
    assert summary["maximum_principal_angle_per_resample_degrees"] == [2.0, 4.0]


def test_paired_changes_compare_each_dimension_to_pca16():
    def result(balanced_accuracy, recall_a, matrix):
        return {
            "metrics": {
                "balanced_accuracy": balanced_accuracy,
                "global_accuracy": balanced_accuracy,
                "macro_f1": balanced_accuracy,
                "cross_entropy": 1.0 - balanced_accuracy,
                "per_class_recall": {"A": recall_a},
                "confusion_matrix": matrix,
            }
        }

    results = {
        "16": result(0.5, 0.4, [[1, 0], [0, 1]]),
        "20": result(0.6, 0.6, [[2, 0], [0, 1]]),
    }

    changes = _paired_changes(results)

    assert np.isclose(
        changes["20"]["paired_metric_differences_vs_pca16"][
            "balanced_accuracy"
        ],
        0.1,
    )
    assert np.isclose(
        changes["20"]["per_class_recall_difference_vs_pca16"]["A"], 0.2
    )
    assert changes["20"]["confusion_matrix_count_difference_vs_pca16"] == [
        [1, 0],
        [0, 0],
    ]
