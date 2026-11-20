import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_qnim_quantum_pca_scaling as scaling
from src.infrastructure.qiskit_vqc_trainer import _transform_feature_matrix


def _fake_inputs():
    dataset = {
        "X_train": np.zeros((2600, 64)),
        "X_val": np.zeros((650, 64)),
        "y_train": np.zeros(2600, dtype=int),
        "y_val": np.zeros(650, dtype=int),
    }
    dataset_info = {
        "dataset_id": "test-pca64",
        "dataset_manifest_sha256": "a" * 64,
        "verified_file_sha256": {},
        "train_shape": [2600, 64],
        "validation_shape": [650, 64],
        "train_label_shape": [2600],
        "validation_label_shape": [650],
        "train_class_counts": [200] * 13,
        "validation_class_counts": [50] * 13,
        "pca_state": {},
        "only_train_validation_loaded": True,
        "test_access": False,
    }
    config_info = {
        "configuration": {"name": "reference", "n_qubits": 16},
        "n_qubits": 16,
        "n_pca_components": 16,
    }
    return dataset, dataset_info, config_info


def _install_fake_inputs(monkeypatch):
    dataset, dataset_info, config_info = _fake_inputs()
    manifest_content = b"immutable test manifest"
    dataset_info["dataset_manifest_sha256"] = hashlib.sha256(
        manifest_content
    ).hexdigest()
    monkeypatch.setattr(
        scaling,
        "_load_dataset",
        lambda _protocol: (dataset, dataset_info, manifest_content),
    )
    monkeypatch.setattr(
        scaling,
        "_inspect_configuration",
        lambda _qubits, _protocol: config_info,
    )


def _fake_result(qubits=16):
    metrics = {
        "global_accuracy": 0.1,
        "balanced_accuracy": 0.1,
        "macro_f1": 0.1,
        "cross_entropy": 2.5,
    }
    ansatz_count = {16: 96, 20: 120}[qubits]
    readout_count = {16: 733, 20: 861}[qubits]
    total_count = ansatz_count + readout_count
    return {
        "status": "completed",
        "configuration": {
            "name": "reference" if qubits == 16 else "expanded",
            "n_qubits": qubits,
            "n_pca_components": qubits,
        },
        "data": {"train_examples": 2600, "validation_examples": 650},
        "metrics": {"training": metrics, "validation": metrics},
        "optimizer_telemetry": {
            "accepted_updates": 1,
            "rejected_updates": 0,
            "steps": [],
        },
        "parameters": {
            "quantum_ansatz_count": ansatz_count,
            "classical_readout_count": readout_count,
            "total_count": total_count,
        },
        "optimization": {
            "iterations": 1,
            "termination_reason": "max_iterations_reached",
        },
        "training_protocol": {"shots": 512, "reference_shots": 512},
        "training_trace": [],
        "execution_accounting": {
            "status": "measured",
            "n_objective_function_calls": 1,
            "n_optimizer_objective_function_calls": 1,
            "n_circuit_executions": 4,
            "n_shots_total": 2048,
            "n_training_examples_evaluated": 4,
            "n_validation_examples_evaluated": 0,
            "n_readout_refits": 0,
            "n_readout_initializations": 1,
        },
    }


def test_dry_run_never_calls_training_and_keeps_configuration_pending(
    tmp_path, monkeypatch
):
    _install_fake_inputs(monkeypatch)

    def forbidden_training(*_args, **_kwargs):
        raise AssertionError("dry-run must not train")

    monkeypatch.setattr(scaling, "_train_configuration", forbidden_training)
    result = scaling.run_single_configuration(
        qubits=16, dry_run=True, output_dir=tmp_path
    )

    manifest = json.loads(
        (tmp_path / "execution_manifest.json").read_text(encoding="utf-8")
    )
    assert result["training_started"] is False
    assert result["test_access"] is False
    assert manifest["configurations"]["16"]["status"] == "not_started"
    assert manifest["configurations"]["16"]["dry_run_configuration"]
    assert not (tmp_path / "checkpoints").exists()
    assert not (tmp_path / ".runner.lock").exists()


def test_mocked_training_records_checkpoint_and_completed_result_is_immutable(
    tmp_path, monkeypatch
):
    _install_fake_inputs(monkeypatch)
    calls = []

    def fake_training(_qubits, _seed, _dataset, _config, progress_callback):
        calls.append("called")
        progress_callback(
            {
                "status": "running",
                "iteration": 1,
                "trace": [{"iteration": 1}],
                "latest_parameters": [0.0],
                "validation_role": "metrics_only_no_gradient_or_fitting",
            }
        )
        return _fake_result()

    monkeypatch.setattr(scaling, "_train_configuration", fake_training)
    first = scaling.run_single_configuration(qubits=16, output_dir=tmp_path)
    result_path = tmp_path / "q16" / "result.json"
    original_hash = hashlib.sha256(result_path.read_bytes()).hexdigest()

    second = scaling.run_single_configuration(qubits=16, output_dir=tmp_path)
    manifest = json.loads(
        (tmp_path / "execution_manifest.json").read_text(encoding="utf-8")
    )
    checkpoint = json.loads(
        (
            tmp_path
            / "checkpoints"
            / "q16"
            / next((tmp_path / "checkpoints" / "q16").glob("attempt_*")).name
            / "checkpoint.json"
        ).read_text(encoding="utf-8")
    )

    assert first["status"] == "completed"
    assert second["status"] == "already_completed"
    assert calls == ["called"]
    assert manifest["configurations"]["16"]["status"] == "completed"
    assert checkpoint["status"] == "completed"
    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == original_hash
    with pytest.raises(FileExistsError, match="create a new output directory"):
        scaling.run_single_configuration(
            qubits=16, new_attempt=True, output_dir=tmp_path
        )


def test_failed_mocked_training_is_not_marked_completed(tmp_path, monkeypatch):
    _install_fake_inputs(monkeypatch)

    def failing_training(*_args, **_kwargs):
        raise RuntimeError("mock failure")

    monkeypatch.setattr(scaling, "_train_configuration", failing_training)
    with pytest.raises(RuntimeError, match="mock failure"):
        scaling.run_single_configuration(qubits=16, output_dir=tmp_path)

    manifest = json.loads(
        (tmp_path / "execution_manifest.json").read_text(encoding="utf-8")
    )
    checkpoint = json.loads(
        (
            tmp_path
            / "checkpoints"
            / "q16"
            / next((tmp_path / "checkpoints" / "q16").glob("attempt_*")).name
            / "checkpoint.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["configurations"]["16"]["status"] == "failed"
    assert checkpoint["status"] == "failed"
    assert not (tmp_path / "q16" / "result.json").exists()


def test_interrupted_attempt_requires_a_fresh_unique_attempt(
    tmp_path, monkeypatch
):
    _install_fake_inputs(monkeypatch)
    calls = []

    def interrupted_then_completed(_qubits, _seed, _dataset, _config, progress):
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            progress(
                {
                    "status": "running",
                    "iteration": 1,
                    "trace": [{"iteration": 1}],
                    "latest_parameters": [0.0],
                    "validation_role": "metrics_only_no_gradient_or_fitting",
                }
            )
            raise KeyboardInterrupt
        return _fake_result()

    monkeypatch.setattr(
        scaling, "_train_configuration", interrupted_then_completed
    )
    interrupted = scaling.run_single_configuration(
        qubits=16, output_dir=tmp_path
    )
    assert interrupted["status"] == "interrupted"
    with pytest.raises(FileExistsError, match="--new-attempt"):
        scaling.run_single_configuration(qubits=16, output_dir=tmp_path)

    completed = scaling.run_single_configuration(
        qubits=16, new_attempt=True, output_dir=tmp_path
    )
    manifest = json.loads(
        (tmp_path / "execution_manifest.json").read_text(encoding="utf-8")
    )
    attempts = manifest["configurations"]["16"]["attempts"]
    assert completed["status"] == "completed"
    assert len(attempts) == 2
    assert attempts[0]["status"] == "interrupted"
    assert attempts[1]["status"] == "completed"
    assert attempts[0]["attempt_id"] != attempts[1]["attempt_id"]
    assert len(set(calls)) == 2


@pytest.mark.parametrize("orphan", ["result", "checkpoint"])
def test_output_without_manifest_rejects_orphan_artifacts(
    tmp_path, monkeypatch, orphan
):
    _install_fake_inputs(monkeypatch)
    if orphan == "result":
        orphan_path = tmp_path / "q16" / "result.json"
        orphan_path.parent.mkdir(parents=True)
        orphan_path.write_text('{"untouched":true}', encoding="utf-8")
    else:
        orphan_path = (
            tmp_path / "checkpoints" / "q16" / "attempt_01_orphan"
        )
        orphan_path.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="orphan"):
        scaling.run_single_configuration(
            qubits=16, dry_run=True, output_dir=tmp_path
        )
    assert orphan_path.exists()
    assert not (tmp_path / "execution_manifest.json").exists()


def test_locked_output_directory_rejects_duplicate_execution(
    tmp_path, monkeypatch
):
    _install_fake_inputs(monkeypatch)
    lock_path = tmp_path / ".runner.lock"
    lock_path.write_text("12345\n", encoding="ascii")

    with pytest.raises(FileExistsError, match="locked"):
        scaling.run_single_configuration(
            qubits=16, dry_run=True, output_dir=tmp_path
        )

    assert lock_path.read_text(encoding="ascii") == "12345\n"
    assert not (tmp_path / "execution_manifest.json").exists()


def test_atomic_file_write_preserves_previous_content_on_replace_failure(
    tmp_path, monkeypatch
):
    target = tmp_path / "result.md"
    target.write_text("previous\n", encoding="utf-8")

    def fail_replace(_source, _destination):
        raise OSError("simulated replace interruption")

    monkeypatch.setattr(scaling.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace interruption"):
        scaling._atomic_bytes(target, b"new result\n")

    assert target.read_text(encoding="utf-8") == "previous\n"
    assert not list(tmp_path.glob(".result.md.*.tmp"))


@pytest.mark.parametrize(
    "target",
    [
        "configuration-json",
        "attempt-json",
        "configuration-markdown",
        "attempt-markdown",
    ],
)
def test_completed_result_hash_is_verified_before_acceptance(
    tmp_path, monkeypatch, target
):
    _install_fake_inputs(monkeypatch)
    monkeypatch.setattr(
        scaling,
        "_train_configuration",
        lambda *_args, **_kwargs: _fake_result(),
    )
    scaling.run_single_configuration(qubits=16, output_dir=tmp_path)
    if target == "configuration-json":
        result_path = tmp_path / "q16" / "result.json"
    elif target == "attempt-json":
        attempt_dir = next(
            (tmp_path / "checkpoints" / "q16").glob("attempt_*")
        )
        result_path = attempt_dir / "result.json"
    elif target == "configuration-markdown":
        result_path = tmp_path / "q16" / "result.md"
    else:
        attempt_dir = next(
            (tmp_path / "checkpoints" / "q16").glob("attempt_*")
        )
        result_path = attempt_dir / "result.md"
    result_path.write_text('{"corrupted":true}', encoding="utf-8")

    with pytest.raises(ValueError, match="result hash verification failed"):
        scaling.run_single_configuration(qubits=16, output_dir=tmp_path)


def test_identity_preprocessing_preserves_stored_pca_values():
    features = np.array([[-0.7, 0.1], [0.4, 0.9]])
    result = _transform_feature_matrix(features, 2, "identity")
    np.testing.assert_array_equal(result, features)
    with pytest.raises(ValueError, match="train-only statistics"):
        _transform_feature_matrix(features, 2, "chebyshev")


@pytest.mark.parametrize("qubits", [16, 20])
def test_aer_measurements_produce_the_configured_quantum_readout(qubits):
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2
    from src.infrastructure.qiskit_vqc_trainer import (
        _build_feature_map_and_ansatz,
        _measure_qubit_marginals_via_circuit,
    )

    circuit, x_params, ansatz_params = _build_feature_map_and_ansatz(
        qubits,
        n_features=qubits,
        feature_map_reps=2,
        ansatz_reps=2,
        entanglement="linear",
    )
    circuit.measure_all()
    backend = AerSimulator(
        method="matrix_product_state",
        seed_simulator=42,
    )
    circuit = transpile(circuit, backend=backend, optimization_level=1)
    sampler = SamplerV2.from_backend(backend)

    features = _measure_qubit_marginals_via_circuit(
        circuit,
        x_params,
        ansatz_params,
        np.zeros(len(ansatz_params)),
        np.zeros((1, qubits)),
        qubits,
        shots=16,
        batch_size=1,
        sampler=sampler,
        simulator_seed=42,
        readout_mode="marginals_zz_neighbors",
    )
    assert features.shape == (1, 2 * qubits - 1)
    assert np.isfinite(features).all()
    assert np.all(np.abs(features) <= 1.0)


def test_injected_aer_backend_compiles_and_evaluates_a_vqc_loss():
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2
    from src.infrastructure.qiskit_vqc_trainer import (
        _make_vqc_loss_fn,
        _n_readout_params,
    )

    backend = AerSimulator(
        method="matrix_product_state",
        seed_simulator=42,
    )
    loss_fn = _make_vqc_loss_fn(
        X_train=np.zeros((1, 16)),
        y_train=np.array([0]),
        n_qubits=16,
        n_features=16,
        shots=16,
        backend_sampler=SamplerV2.from_backend(backend),
        batch_size_local=1,
        feature_map_reps=2,
        ansatz_reps=2,
        readout_mode="marginals_zz_neighbors",
        input_preprocessing="identity",
        sampler_backend=backend,
    )
    n_readout = _n_readout_params(
        loss_fn.readout_feature_count, 1, loss_fn.hidden_size
    )
    value = loss_fn(np.zeros(loss_fn.n_ansatz_params + n_readout))
    assert np.isfinite(value)


def test_reference_and_all_loss_evaluation_paths_use_protocol_shots(monkeypatch):
    from src.infrastructure import qiskit_vqc_trainer as trainer

    class FakeJob:
        def __init__(self, count):
            self.count = count

        def result(self):
            return [
                SimpleNamespace(
                    data=SimpleNamespace(
                        meas=SimpleNamespace(
                            get_counts=lambda: {"00": 512}
                        )
                    )
                )
                for _ in range(self.count)
            ]

    class RecordingSampler:
        def __init__(self):
            self.shot_counts = []

        def run(self, pubs, shots):
            self.shot_counts.append(shots)
            return FakeJob(len(pubs))

    sampler = RecordingSampler()
    monkeypatch.setattr(
        trainer,
        "_fit_readout_mlp_classically",
        lambda X, y, n_classes, hidden_size=16: (
            np.zeros((X.shape[1], hidden_size)),
            np.zeros(hidden_size),
            np.zeros((hidden_size, n_classes)),
            np.zeros(n_classes),
        ),
    )
    loss_fn = trainer._make_vqc_loss_fn(
        X_train=np.zeros((2, 2)),
        y_train=np.array([0, 1]),
        n_qubits=2,
        n_features=2,
        shots=512,
        reference_shots=512,
        backend_sampler=sampler,
        batch_size_local=2,
        readout_hidden_size=2,
        readout_mode="marginals_zz_neighbors",
        input_preprocessing="identity",
    )
    n_readout = trainer._n_readout_params(
        loss_fn.readout_feature_count, 2, loss_fn.hidden_size
    )
    theta = np.zeros(loss_fn.n_ansatz_params + n_readout)

    loss_fn(theta)
    loss_fn(theta + 0.01)
    loss_fn(theta - 0.01)
    loss_fn(theta + np.pi / 4)
    loss_fn(theta)
    loss_fn.evaluate_reference(theta)
    loss_fn.warm_start_readout(theta[: loss_fn.n_ansatz_params])

    assert sampler.shot_counts
    assert set(sampler.shot_counts) == {512}
    assert loss_fn.execution_telemetry["n_circuit_executions"] == 14
    assert loss_fn.execution_telemetry["n_shots_total"] == 14 * 512
    assert loss_fn.execution_telemetry["n_training_examples_evaluated"] == 14


def test_optimizer_separates_objective_calls_from_measured_circuits():
    from src.infrastructure.qnspsa_eml_feynman import (
        QNSPSAConfig,
        QNSPSAEMLFeynman,
    )

    telemetry = {
        "n_circuit_executions": 0,
        "n_shots_total": 0,
        "n_training_examples_evaluated": 0,
    }

    def loss(theta):
        telemetry["n_circuit_executions"] += 1
        telemetry["n_shots_total"] += 512
        telemetry["n_training_examples_evaluated"] += 1
        return float(np.dot(theta, theta))

    loss.refresh_batch = lambda: None
    loss.evaluate_reference = loss
    loss.execution_telemetry = telemetry
    result = QNSPSAEMLFeynman(
        QNSPSAConfig(
            maxiter=1,
            n_feynman_params=1,
            n_gl_points=2,
            patience=5,
            seed=42,
        )
    ).minimize(loss, np.array([0.1, -0.2]))

    assert result.n_objective_function_calls == result.n_evals
    assert result.n_circuit_executions == result.n_evals
    assert result.n_shots_total == result.n_evals * 512
    assert result.n_training_examples_evaluated == result.n_evals
    assert result.execution_accounting_measured is True


def test_trainer_rechecks_loss_after_readout_refit(monkeypatch):
    from src.infrastructure import qiskit_vqc_trainer as trainer

    events = []
    batch_indices = np.array([4, 7])
    telemetry = {
        "n_objective_function_calls": 0,
        "n_circuit_executions": 0,
        "n_shots_total": 0,
        "n_training_examples_evaluated": 0,
        "n_readout_refits": 0,
        "n_readout_initializations": 0,
    }

    class FakeLoss:
        n_ansatz_params = 2
        readout_feature_count = 3
        hidden_size = 2
        execution_telemetry = telemetry
        batch_history = [batch_indices]
        last_readout_indices = np.array([], dtype=int)

        def __init__(self):
            self.warm_starts = 0

        def __call__(self, theta):
            self.execution_telemetry["n_objective_function_calls"] += 1
            events.append(("loss", theta.copy(), batch_indices.copy()))
            return float(theta[-1])

        def refresh_batch(self):
            return None

        def evaluate_reference(self, theta):
            return self(theta)

        def current_batch_indices(self):
            return batch_indices.copy()

        def warm_start_readout(self, _ansatz):
            self.warm_starts += 1
            self.last_readout_indices = np.array([0, 1])
            value = self.warm_starts / 10
            return (
                np.full((3, 2), value),
                np.full(2, value),
                np.full((2, 2), value),
                np.full(2, value),
            )

    loss_fn = FakeLoss()

    class FakeOptimizer:
        def __init__(self, config):
            self.config = config

        def minimize(self, loss, initial, callback=None, **_kwargs):
            theta = initial.copy()
            initial_loss = loss(theta)
            callback(10, theta, initial_loss)
            return SimpleNamespace(
                optimal_params=theta.copy(),
                final_loss=initial_loss,
                n_iter=10,
                n_evals=1,
                n_objective_function_calls=1,
                n_circuit_executions=0,
                n_shots_total=0,
                n_training_examples_evaluated=0,
                execution_accounting_measured=True,
                time_s=0.0,
                speedup_vs_spsa=1.0,
                converged=False,
                loss_history=[initial_loss],
                gradient_variance_history=[],
                optimizer_step_history=[],
            )

    monkeypatch.setattr(
        trainer, "_make_vqc_loss_fn", lambda **_kwargs: loss_fn
    )
    monkeypatch.setattr(
        trainer,
        "_flatten_readout_mlp",
        lambda W1, _b1, _W2, _b2: np.full(14, W1[0, 0]),
    )
    monkeypatch.setattr(trainer, "QNSPSAEMLFeynman", FakeOptimizer)
    diagnostic_records = []
    post_refit_records = []

    result = trainer.QiskitVQCTrainer(
        mode="sim",
        training_batch_size=2,
        readout_refit_every=10,
    ).train_vqc(
        X_train=np.zeros((2, 2)),
        y_train=np.array([0, 1]),
        num_qubits=2,
        n_features=2,
        max_iterations=10,
        shots=512,
        diagnostic_callback=lambda iteration, theta, loss: diagnostic_records.append(
            (iteration, theta, loss)
        ),
        post_refit_callback=lambda iteration, theta, loss: post_refit_records.append(
            (iteration, theta, loss)
        ),
    )

    assert len(diagnostic_records) == 1
    assert len(post_refit_records) == 1
    pre = diagnostic_records[0]
    post = post_refit_records[0]
    assert pre[0] == post[0] == 10
    assert pre[2] == pytest.approx(0.1)
    assert post[2] == pytest.approx(0.2)
    np.testing.assert_array_equal(pre[1][-14:], np.full(14, 0.1))
    np.testing.assert_array_equal(post[1][-14:], np.full(14, 0.2))
    np.testing.assert_array_equal(events[0][2], events[1][2])
    assert result["n_readout_refits"] == 1
    assert result["n_readout_initializations"] == 1


def test_runner_explicitly_sets_reference_shots_to_protocol_value(monkeypatch):
    import qiskit
    from src.infrastructure import qiskit_vqc_trainer as trainer

    class FakeCircuit:
        def measure_all(self):
            return None

    class StopAfterConfiguration(Exception):
        pass

    captured = {}

    class CapturingTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def train_vqc(self, **_kwargs):
            raise StopAfterConfiguration

    monkeypatch.setattr(trainer, "QiskitVQCTrainer", CapturingTrainer)
    monkeypatch.setattr(
        trainer,
        "_build_feature_map_and_ansatz",
        lambda qubits, **_kwargs: (
            FakeCircuit(),
            list(range(qubits)),
            list(range(96)),
        ),
    )
    monkeypatch.setattr(
        qiskit, "transpile", lambda circuit, **_kwargs: circuit
    )
    monkeypatch.setattr(
        scaling, "_build_local_aer", lambda _seed: (object(), object())
    )
    dataset, _, _ = _fake_inputs()
    dataset["y_train"] = np.repeat(np.arange(13), 200)
    dataset["y_val"] = np.repeat(np.arange(13), 50)
    config_info = {
        "configuration": {"name": "reference"},
        "ansatz_parameter_count": 96,
    }

    with pytest.raises(StopAfterConfiguration):
        scaling._train_configuration(
            16, 42, dataset, config_info, lambda _progress: None
        )

    assert captured["reference_shots"] == 512
    assert captured["learning_rate"] == 0.03
    assert captured["training_batch_size"] == 32


def test_refit_trace_records_pre_and_post_states_on_identical_data(
    monkeypatch,
):
    import qiskit
    from src.infrastructure import qiskit_vqc_trainer as trainer

    class FakeCircuit:
        def measure_all(self):
            return None

    class FakeTrainer:
        def __init__(self, **_kwargs):
            pass

        def train_vqc(self, **kwargs):
            pre = np.zeros(829)
            post = pre.copy()
            post[-1] = 0.25
            kwargs["diagnostic_callback"](10, pre, 1.0)
            kwargs["post_refit_callback"](10, post, 0.8)
            return {
                "weights": pre,
                "training_loss": 0.8,
                "iterations": 10,
                "converged": False,
                "termination_reason": "max_iterations_reached",
                "n_circuit_evaluations": 1,
                "n_optimizer_objective_function_calls": 1,
                "n_objective_function_calls": 2,
                "n_circuit_executions": 4,
                "n_shots_total": 2048,
                "n_training_examples_evaluated": 4,
                "n_readout_refits": 1,
                "n_readout_initializations": 1,
                "optimizer_batch_indices": [list(range(32)) for _ in range(12)],
                "readout_refit_history": [
                    {
                        "iteration": 10,
                        "training_indices": list(range(200)),
                        "n_training_examples": 200,
                    }
                ],
                "optimizer_step_history": [
                    {"iteration": 10, "update_accepted": True}
                ],
                "loss_history": [1.0],
                "gradient_variance_history": [0.0],
                "execution_accounting_measured": True,
            }

    measurement_sizes = []

    def fake_measure(
        _circuit, _x, _a, _theta, features, qubits, **_kwargs
    ):
        measurement_sizes.append(len(features))
        return np.zeros((len(features), 2 * qubits - 1))

    monkeypatch.setattr(trainer, "QiskitVQCTrainer", FakeTrainer)
    monkeypatch.setattr(
        trainer,
        "_build_feature_map_and_ansatz",
        lambda qubits, **_kwargs: (
            FakeCircuit(),
            list(range(qubits)),
            list(range(96)),
        ),
    )
    monkeypatch.setattr(
        trainer,
        "_measure_qubit_marginals_via_circuit",
        fake_measure,
    )
    monkeypatch.setattr(
        trainer,
        "_readout_mlp_probs",
        lambda *_args: np.full(13, 1.0 / 13),
    )
    monkeypatch.setattr(
        qiskit, "transpile", lambda circuit, **_kwargs: circuit
    )
    monkeypatch.setattr(
        scaling, "_build_local_aer", lambda _seed: (object(), object())
    )
    monkeypatch.setattr(
        scaling,
        "_classification_metrics",
        lambda *_args: {
            "balanced_accuracy": 1.0 / 13,
            "global_accuracy": 1.0 / 13,
            "macro_f1": 1.0 / 13,
            "cross_entropy": 2.5,
        },
    )
    dataset, _, _ = _fake_inputs()
    dataset["y_train"] = np.repeat(np.arange(13), 200)
    dataset["y_val"] = np.repeat(np.arange(13), 50)
    config_info = {
        "configuration": {"name": "reference"},
        "ansatz_parameter_count": 96,
        "circuit": {},
        "readout": {},
        "simulator": {},
    }
    result = scaling._train_configuration(
        16, 42, dataset, config_info, lambda _progress: None
    )

    trace = result["training_trace"][0]
    assert trace["pre_refit"]["minibatch_training_loss"] == 1.0
    assert trace["post_refit"]["minibatch_training_loss"] == 0.8
    assert (
        trace["pre_refit"]["parameter_sha256"]
        != trace["post_refit"]["parameter_sha256"]
    )
    assert (
        trace["diagnostic_data"]["training_indices"]
        == scaling._diagnostic_indices(
            dataset["y_train"], 5, 42
        ).tolist()
    )
    assert (
        trace["diagnostic_data"]["validation_indices"]
        == scaling._diagnostic_indices(
            dataset["y_val"], 5, 43
        ).tolist()
    )
    assert measurement_sizes == [65, 65, 65, 65, 2600, 650]
    assert trace["diagnostic_data"]["optimizer_loss_training_indices"] == list(
        range(32)
    )
    assert trace["readout_refit"]["status"] == "completed"
    assert trace["readout_refit"]["n_training_examples"] == 200
