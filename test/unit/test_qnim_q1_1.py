import numpy as np
import pytest
from types import SimpleNamespace

from scripts.run_qnim_q1_1 import (
    _classification_metrics,
    _seed_for_diagnostic,
    _validate_resume_report,
)
import src.infrastructure.qiskit_vqc_trainer as qiskit_vqc_trainer
from src.infrastructure.qiskit_vqc_trainer import _fit_readout_mlp_classically


def test_classification_metrics_report_per_class_and_prediction_counts():
    labels = np.arange(13)
    probabilities = np.eye(13)

    metrics = _classification_metrics(labels, probabilities)

    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["class_recall"] == [1.0] * 13
    assert metrics["prediction_counts"] == [1] * 13
    assert np.array_equal(metrics["confusion_matrix"], np.eye(13, dtype=int))
    assert metrics["cross_entropy"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "labels, probabilities",
    [
        (np.array([13]), np.full((1, 13), 1 / 13)),
        (np.array([0]), np.zeros((1, 13))),
        (np.array([0]), np.full((1, 12), 1 / 12)),
    ],
)
def test_classification_metrics_reject_invalid_inputs(labels, probabilities):
    with pytest.raises(ValueError):
        _classification_metrics(labels, probabilities)


def test_diagnostic_seeds_are_repeatable_and_split_specific():
    train_seed = _seed_for_diagnostic(42, 43, 7, 0)
    assert train_seed == _seed_for_diagnostic(42, 43, 7, 0)
    assert train_seed != _seed_for_diagnostic(42, 43, 7, 1)


def test_readout_diagnostic_callback_receives_copies():
    rng = np.random.default_rng(42)
    features = rng.uniform(0.0, 1.0, size=(26, 12))
    labels = np.repeat(np.arange(13), 2)
    observed = []

    def callback(iteration, W1, b1, W2, b2):
        observed.append((iteration, W1.copy()))
        if iteration == 1:
            W1[:] = 0.0

    weights = _fit_readout_mlp_classically(
        features,
        labels,
        n_classes=13,
        n_iters=2,
        seed=42,
        diagnostic_callback=callback,
    )
    reference_weights = _fit_readout_mlp_classically(
        features, labels, n_classes=13, n_iters=2, seed=42
    )

    assert [iteration for iteration, _ in observed] == [1, 2]
    for actual, expected in zip(weights, reference_weights):
        np.testing.assert_array_equal(actual, expected)


def test_vqc_initialization_seed_is_separate_from_training_seed(monkeypatch):
    captured = []
    loss_seeds = []

    def fake_loss_fn(**kwargs):
        loss_seeds.append(kwargs["random_seed"])
        return lambda theta: 1.0

    class FakeOptimizer:
        def __init__(self, config):
            assert config["seed"] == 42

        def minimize(self, loss_fn, x0, callback=None):
            captured.append(x0.copy())
            if callback is not None:
                callback(1, x0, 1.0)
            return SimpleNamespace(
                optimal_params=x0.copy(),
                final_loss=1.0,
                n_iter=1,
                n_evals=1,
                speedup_vs_spsa=1.0,
                converged=False,
                loss_history=[1.0],
                gradient_variance_history=[],
            )

    monkeypatch.setattr(qiskit_vqc_trainer, "_make_vqc_loss_fn", fake_loss_fn)
    monkeypatch.setattr(
        qiskit_vqc_trainer, "QNSPSAConfig", lambda **kwargs: kwargs
    )
    monkeypatch.setattr(
        qiskit_vqc_trainer, "QNSPSAEMLFeynman", FakeOptimizer
    )
    trainer = qiskit_vqc_trainer.QiskitVQCTrainer(
        mode="sim", random_seed=42
    )
    features = np.zeros((13, 12))
    labels = np.arange(13)
    diagnostics = []

    for initialization_seed in (42, 43, None):
        trainer.train_vqc(
            X_train=features,
            y_train=labels,
            num_qubits=12,
            n_features=12,
            max_iterations=1,
            shots=8,
            initialization_seed=initialization_seed,
            diagnostic_callback=lambda iteration, theta, loss: diagnostics.append(
                (iteration, theta.copy(), loss)
            ),
        )

    assert loss_seeds == [42, 42, 42]
    np.testing.assert_array_equal(captured[0], captured[2])
    assert not np.array_equal(captured[0], captured[1])
    assert [item[0] for item in diagnostics] == [1, 1, 1]


def test_resume_accepts_completed_seed_checkpoints_and_rejects_protocol_changes():
    alias = {"same_run_as": "D_trainable_ansatz.initialization_seed_42"}
    expected = {
        "checkpoint_schema_version": 1,
        "experiment_id": "QNIM_Q1_1_V1",
        "experiment_scope": "frozen-reference diagnostic",
        "classical_reference_id": "CLASSICAL_REFERENCE_V1",
        "classical_reference_sha256": "ref",
        "dataset_seed": 42,
        "model_initialization_seeds": [42, 43],
        "training_random_seed": 42,
        "randomness": "fixed",
        "dataset_hashes": {"train": "train", "validation": "validation"},
        "architecture": {"n_qubits": 12},
        "training_configuration": {"max_iterations": 1},
        "validation_policy": {"used_for_parameter_updates": False},
        "software_versions": {"python": "3.12"},
        "final_test": {"status": "locked_not_evaluated"},
        "experiments": {
            "A_current_qnim_instrumented": alias,
            "B_classical_mlp": {"status": "pending"},
            "C_frozen_ansatz_trainable_readout": {
                "runs": [{
                    "status": "completed",
                    "initialization_seed": 42,
                    "dataset_seed": 42,
                    "training_random_seed": 42,
                }],
            },
            "D_trainable_ansatz": {"runs": []},
        },
    }
    checkpoint = {
        **expected,
        "status": "in_progress",
        "active_run": {"condition": "D_trainable_ansatz", "initialization_seed": 43},
    }

    _validate_resume_report(checkpoint, expected)

    changed = {**expected, "training_configuration": {"max_iterations": 2}}
    with pytest.raises(ValueError, match="training_configuration"):
        _validate_resume_report(checkpoint, changed)
