import numpy as np
import pytest

from scripts.analyze_accuracy_gap import (
    _resource_projection,
    _validate_pair_layout,
    _within_pair_permutation_p,
    analyze,
)


def test_within_pair_permutation_counts_exact_chance_without_float_tie_bias():
    n_classes = 13
    y_true = np.tile(np.arange(n_classes), 100)
    y_pred = np.full_like(y_true, 4)

    p_value = _within_pair_permutation_p(
        y_true,
        y_pred,
        iterations=1000,
        seed=42,
    )

    assert p_value == 1.0


def test_validate_pair_layout_rejects_misaligned_labels():
    labels = np.tile(np.arange(13), 2)
    labels[[0, 1]] = labels[[1, 0]]

    with pytest.raises(ValueError, match="orden emparejado"):
        _validate_pair_layout(labels, n_classes=13, split="test")


def test_resource_projection_uses_requested_class_count():
    projection = _resource_projection(10)

    assert "balanced 10-class test set" in projection["assumption"]
    assert projection["power_table"][3]["n_total"] == 1000
    assert projection["power_table"][3]["critical_correct_for_one_sided_alpha_0_05"] == 117


def test_analyze_includes_global_template_features_when_available(tmp_path):
    rng = np.random.default_rng(42)
    y_train = np.tile(np.arange(3), 15)
    y_val = np.tile(np.arange(3), 5)
    X_train = np.column_stack(
        (y_train + rng.normal(0, 0.1, len(y_train)), rng.normal(size=len(y_train)))
    )
    X_val = np.column_stack(
        (y_val + rng.normal(0, 0.1, len(y_val)), rng.normal(size=len(y_val)))
    )

    for split, features, labels in (("train", X_train, y_train), ("val", X_val, y_val)):
        np.save(tmp_path / f"X_{split}.npy", features)
        np.save(tmp_path / f"y_{split}.npy", labels)
        np.save(tmp_path / f"X_global_template_{split}.npy", features + 0.01)
    matched_scores = np.zeros((len(y_val), 3))
    matched_scores[np.arange(len(y_val)), y_val] = 1.0
    np.save(tmp_path / "X_matched_filter_val.npy", matched_scores)

    result = analyze(
        tmp_path,
        bootstrap_iterations=10,
        permutation_iterations=10,
        seed=42,
    )

    assert "global_template" in result["feature_results"]
    assert result["matched_filter_bank"]["validation"]["balanced_accuracy"] == 1.0
    assert result["test_arrays_read"] is False
