import numpy as np
import pytest

from src.application.scaling_metrics import (
    RANDOM_BASELINE,
    holm_bonferroni,
    paired_bootstrap_difference,
    permutation_test,
    stratified_bootstrap,
)


def test_stratified_bootstrap_returns_reproducible_confidence_interval():
    y_true = np.repeat(np.arange(3), 8)
    result = stratified_bootstrap(y_true, y_true, n_iterations=100, seed=7)

    assert result["mean"] == 1.0
    assert result["lower_95"] == 1.0
    assert result["upper_95"] == 1.0
    assert stratified_bootstrap(y_true, y_true, n_iterations=100, seed=7) == result


def test_permutation_test_detects_perfect_predictions():
    y_true = np.repeat(np.arange(4), 10)
    result = permutation_test(y_true, y_true, n_permutations=500, seed=3)

    assert result["observed"] == 1.0
    assert result["p_value"] < 0.05


def test_paired_bootstrap_uses_same_events_and_detects_improvement():
    y_true = np.repeat(np.arange(3), 8)
    pred_baseline = np.zeros_like(y_true)
    result = paired_bootstrap_difference(
        y_true, pred_baseline, y_true, n_iterations=100, seed=11
    )

    assert result["mean_difference"] == pytest.approx(2 / 3)
    assert result["lower_95"] > 0


def test_holm_bonferroni_adjusts_p_values_in_input_order():
    adjusted = holm_bonferroni([0.01, 0.04, 0.03])

    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])


def test_random_baseline_is_thirteen_class_chance_level():
    assert RANDOM_BASELINE == pytest.approx(1 / 13)


@pytest.mark.parametrize("function", [stratified_bootstrap, permutation_test])
def test_metrics_reject_empty_inputs(function):
    with pytest.raises(ValueError):
        function([], [], n_iterations=1) if function is stratified_bootstrap else function(
            [], [], n_permutations=1
        )