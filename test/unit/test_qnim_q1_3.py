import numpy as np
import pytest

from scripts.diagnose_qnim_classical_baselines import _make_model
from scripts import run_qnim_q1_3


def test_q1_2_initial_and_final_ansatz_parameters_are_hash_validated():
    initial = np.random.default_rng(42).normal(0.0, 0.01, 3)
    trained = np.array([0.1, -0.2, 0.3])
    report = {
        "architecture": {"ansatz_parameters": 3},
        "experiments": {
            "Q1_2_A": {
                "trainable_ansatz": [{
                    "status": "completed",
                    "initialization_seed": 42,
                    "initial_ansatz_sha256": run_qnim_q1_3._sha256(initial),
                    "final_ansatz_sha256": run_qnim_q1_3._sha256(trained),
                    "final_ansatz_parameters": trained.tolist(),
                    "updates": 60,
                }],
            },
        },
    }

    parameters = run_qnim_q1_3._circuit_parameter_sets(report, seeds=(42,))

    np.testing.assert_array_equal(parameters[42]["initial"], initial)
    np.testing.assert_array_equal(parameters[42]["trained"], trained)

    report["experiments"]["Q1_2_A"]["trainable_ansatz"][0][
        "final_ansatz_sha256"
    ] = "invalid"
    with pytest.raises(ValueError, match="final ansatz hash mismatch"):
        run_qnim_q1_3._circuit_parameter_sets(report, seeds=(42,))


def test_q1_3_refuses_q1_2_reports_without_exact_final_parameters():
    report = {
        "architecture": {"ansatz_parameters": 3},
        "experiments": {
            "Q1_2_A": {
                "trainable_ansatz": [{
                    "status": "completed",
                    "initialization_seed": 42,
                    "initial_ansatz_sha256": run_qnim_q1_3._sha256(
                        np.random.default_rng(42).normal(0.0, 0.01, 3)
                    ),
                    "final_ansatz_sha256": "hash-only",
                }],
            },
        },
    }

    with pytest.raises(ValueError, match="does not persist the exact final ansatz"):
        run_qnim_q1_3._circuit_parameter_sets(report, seeds=(42,))


def test_classical_probes_fit_train_only(monkeypatch):
    fit_sizes = []

    class Probe:
        classes_ = np.arange(13)

        def fit(self, features, labels):
            fit_sizes.append((len(features), len(labels)))
            return self

        def predict_proba(self, features):
            return np.full((len(features), 13), 1 / 13)

    factory_calls = []

    def make_probe(name, seed, probability, mlp_max_iter):
        factory_calls.append((name, seed, probability, mlp_max_iter))
        return Probe()

    monkeypatch.setattr(run_qnim_q1_3, "_make_model", make_probe)
    X_train = np.zeros((26, 12))
    y_train = np.tile(np.arange(13), 2)
    X_val = np.zeros((13, 12))
    y_val = np.arange(13)

    metrics = run_qnim_q1_3._fit_probe_models(
        X_train, y_train, X_val, y_val, seed=42
    )

    assert set(metrics) == set(run_qnim_q1_3.PROBE_MODELS)
    assert factory_calls == [
        (name, 42, True, run_qnim_q1_3.MLP_MAX_ITER)
        for name in run_qnim_q1_3.PROBE_MODELS
    ]
    assert fit_sizes == [(26, 26)] * len(run_qnim_q1_3.PROBE_MODELS)
    assert all(
        result["validation"]["balanced_accuracy"] == pytest.approx(1 / 13)
        for result in metrics.values()
    )


def test_svm_probability_estimates_are_opt_in():
    assert _make_model("svm_rbf", 42)[-1].probability is False
    probability_svm = _make_model("svm_rbf", 42, probability=True)[-1]
    assert probability_svm.probability is True
    assert probability_svm.random_state == 42


def test_mlp_iteration_limit_is_configurable_without_changing_baseline_default():
    assert _make_model("mlp", 42)[-1].max_iter == 500
    assert _make_model("mlp", 42, mlp_max_iter=2000)[-1].max_iter == 2000
