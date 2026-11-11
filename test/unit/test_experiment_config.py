import pytest

from src.application.use_cases.generate_experiment_results_use_case import (
    ExperimentConfig,
    GenerateExperimentResultsUseCase,
)


def test_experiment_config_supports_27q_hardware_profile():
    cfg = ExperimentConfig(
        n_qubits=27,
        pca_components=12,
        feature_map_reps=1,
        ansatz_reps=2,
        entanglement="linear",
    )

    assert cfg.n_qubits == 27
    assert cfg.pca_components == 12
    assert cfg.feature_map_reps == 1
    assert cfg.ansatz_reps == 2
    assert cfg.entanglement == "linear"


def test_experiment_config_allows_independent_pca_and_qubit_counts():
    cfg = ExperimentConfig(n_qubits=16, pca_components=12)
    assert cfg.n_qubits == 16
    assert cfg.pca_components == 12


def test_experiment_config_rejects_pca_larger_than_qubits():
    with pytest.raises(ValueError, match="pca_components"):
        ExperimentConfig(n_qubits=12, pca_components=16)


def test_experiment_config_requires_positive_hardware_finetune_budget():
    with pytest.raises(ValueError, match="hardware_finetune_steps"):
        ExperimentConfig(mode="ibm", train_on_hardware=True, hardware_finetune_steps=0)


def test_execute_passes_configured_feature_count_to_trainer():
    class FakeDataset:
        X_train = type("Features", (), {"shape": (4, 8)})()

    class FakeTrainer:
        def train_and_evaluate(self, dataset, **kwargs):
            self.kwargs = kwargs
            return type("Result", (), {
                "accuracy_sim": 0.75,
                "accuracy_real_no_zne": None,
                "accuracy_real_zne": None,
                "loss_history": [0.6],
                "n_epochs": 1,
                "confusion_matrix": None,
            })()

    config = ExperimentConfig(n_qubits=16, pca_components=8, n_pca_fit=32)
    use_case = GenerateExperimentResultsUseCase(config)
    trainer = FakeTrainer()
    use_case._build_dataset = lambda: FakeDataset()
    use_case._build_vqc_trainer = lambda: trainer

    use_case.execute()

    assert trainer.kwargs["n_features"] == 8
    assert trainer.kwargs["use_real_hardware"] is False


def test_execute_uses_trainer_ibm_measurement_without_revalidating():
    class FakeDataset:
        X_train = type("Features", (), {"shape": (4, 8)})()

    class FakeTrainer:
        def train_and_evaluate(self, dataset, **kwargs):
            self.kwargs = kwargs
            return type("Result", (), {
                "accuracy_sim": 0.75,
                "accuracy_real_no_zne": 0.68,
                "accuracy_real_zne": 0.72,
                "loss_history": [0.6],
                "n_epochs": 1,
                "confusion_matrix": None,
            })()

    config = ExperimentConfig(
        mode="ibm", n_qubits=16, pca_components=8, n_pca_fit=32, use_zne=True
    )
    use_case = GenerateExperimentResultsUseCase(config)
    trainer = FakeTrainer()
    use_case._build_dataset = lambda: FakeDataset()
    use_case._build_vqc_trainer = lambda: trainer

    result = use_case.execute()

    assert trainer.kwargs["n_features"] == 8
    assert trainer.kwargs["use_real_hardware"] is True
    assert result.accuracy_ibm == 0.72


def test_execute_passes_bounded_hardware_finetune_budget():
    class FakeDataset:
        X_train = type("Features", (), {"shape": (4, 8)})()

    class FakeTrainer:
        def train_and_evaluate(self, dataset, **kwargs):
            self.kwargs = kwargs
            return type("Result", (), {
                "accuracy_sim": 0.75,
                "accuracy_real_no_zne": 0.68,
                "accuracy_real_zne": 0.72,
                "loss_history": [0.6],
                "n_epochs": 1,
                "confusion_matrix": None,
            })()

    config = ExperimentConfig(
        mode="ibm",
        n_qubits=16,
        pca_components=8,
        n_pca_fit=32,
        train_on_hardware=True,
        hardware_finetune_steps=7,
    )
    use_case = GenerateExperimentResultsUseCase(config)
    trainer = FakeTrainer()
    use_case._build_dataset = lambda: FakeDataset()
    use_case._build_vqc_trainer = lambda: trainer

    use_case.execute()

    assert trainer.kwargs["use_real_hardware"] is True
    assert trainer.kwargs["hardware_finetune_steps"] == 7


def test_ibm_trainer_requires_token_before_importing_quantum_dependencies(monkeypatch):
    import src.application.use_cases.generate_experiment_results_use_case as use_case_module

    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    monkeypatch.setattr(use_case_module, "get_ibm_quantum_token", lambda: "")
    use_case = GenerateExperimentResultsUseCase(ExperimentConfig(mode="ibm"))

    with pytest.raises(RuntimeError, match="IBM_QUANTUM_TOKEN"):
        use_case._build_vqc_trainer()


def _load_vqc_trainer_module(monkeypatch):
    import importlib
    import sys
    from pathlib import Path
    from types import ModuleType

    infrastructure_package = ModuleType("src.infrastructure")
    infrastructure_package.__path__ = [
        str(Path(__file__).resolve().parents[2] / "src" / "infrastructure")
    ]
    monkeypatch.setitem(sys.modules, "src.infrastructure", infrastructure_package)
    module_name = "src.infrastructure.qiskit_vqc_trainer"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    return importlib.import_module(module_name)


@pytest.mark.parametrize(
    ("n_qubits", "n_features"),
    [(12, 12), (16, 12), (16, 16), (20, 12), (20, 20), (32, 12), (32, 32)],
)
def test_feature_map_and_ansatz_support_independent_dimensions(
    monkeypatch, n_qubits, n_features
):
    trainer_module = _load_vqc_trainer_module(monkeypatch)
    _build_feature_map_and_ansatz = trainer_module._build_feature_map_and_ansatz

    circuit, feature_params, ansatz_params = _build_feature_map_and_ansatz(
        n_qubits=n_qubits,
        n_features=n_features,
        feature_map_reps=1,
        ansatz_reps=1,
    )

    assert circuit.num_qubits == n_qubits
    assert len(feature_params) == n_features
    assert len(ansatz_params) > 0


def test_feature_map_rejects_more_features_than_qubits(monkeypatch):
    trainer_module = _load_vqc_trainer_module(monkeypatch)

    with pytest.raises(ValueError, match="n_features <= n_qubits"):
        trainer_module._build_feature_map_and_ansatz(n_qubits=12, n_features=16)


def test_hardware_finetuning_uses_remote_sampler_with_mocked_runtime(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    import numpy as np

    trainer_module = _load_vqc_trainer_module(monkeypatch)
    backend = SimpleNamespace(num_qubits=4)
    captured = {}

    class FakeService:
        def __init__(self, channel, token):
            assert token == "test-token"

        def backend(self, name):
            assert name == "mock-backend"
            return backend

    class FakeSampler:
        def __init__(self, mode):
            self.mode = mode

    runtime_module = ModuleType("qiskit_ibm_runtime")
    runtime_module.QiskitRuntimeService = FakeService
    runtime_module.SamplerV2 = FakeSampler
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", runtime_module)

    class FakeLoss:
        def __init__(self):
            self.last_execution = {"last": {}}
            self.calls = 0

        def evaluate_batch(self, parameters, indices, shots):
            self.calls += 1
            self.last_execution["last"] = {
                "job_ids": [f"job-{self.calls}"],
                "shots": len(indices) * shots,
                "pubs": len(indices),
                "qpu_seconds": None,
            }
            return float(np.sum(parameters**2))

    def fake_loss(**kwargs):
        captured.update(kwargs)
        return FakeLoss()

    monkeypatch.setattr(trainer_module, "_make_vqc_loss_fn", fake_loss)
    trainer = trainer_module.QiskitVQCTrainer(
        mode="ibm",
        token="test-token",
        backend_name="mock-backend",
        train_on_hardware=True,
    )

    result = trainer.fine_tune_on_hardware(
        initial_weights=np.ones(3),
        X_train=np.zeros((6, 2)),
        y_train=np.array([0, 1, 0, 1, 0, 1]),
        n_qubits=2,
        n_features=2,
        steps=1,
        batch_size=2,
        shots=10,
        reference_every=0,
    )

    assert isinstance(captured["backend_sampler"], FakeSampler)
    assert captured["backend_sampler"].mode is backend
    assert captured["ibm_backend"] is backend
    assert captured["remote_hardware"] is True
    assert result.steps_completed == 1
    assert result.jobs == 2
    assert result.shots == 40
    assert result.qpu_seconds is None


def test_train_and_evaluate_runs_local_pretraining_then_bounded_finetuning(monkeypatch):
    import numpy as np
    from types import SimpleNamespace

    trainer_module = _load_vqc_trainer_module(monkeypatch)
    trainer = trainer_module.QiskitVQCTrainer(
        mode="ibm", token="test-token", train_on_hardware=True
    )
    dataset = SimpleNamespace(
        X_train=np.zeros((4, 2)),
        y_train=np.array([0, 1, 0, 1]),
        X_val=np.zeros((2, 2)),
        y_val=np.array([0, 1]),
        n_classes=2,
    )
    local_weights = np.array([1.0, 1.0])
    adapted_weights = np.array([2.0, 2.0])
    calls = []

    def fake_train_vqc(**kwargs):
        calls.append("local")
        return {
            "weights": local_weights,
            "loss_history": [0.8],
            "iterations": 1,
            "speedup_vs_spsa": 1.0,
            "n_circuit_evaluations": 2,
            "execution_time_seconds": 1.0,
            "converged": True,
            "gradient_variance_history": [],
        }

    def fake_finetune(**kwargs):
        calls.append("hardware_spsa")
        assert np.array_equal(kwargs["initial_weights"], local_weights)
        assert kwargs["steps"] == 3
        return SimpleNamespace(
            parameters=adapted_weights,
            steps_completed=3,
            jobs=6,
            pubs=12,
            shots=384,
            qpu_seconds=None,
            job_ids=["mock-job-1"],
            reference_history=[],
            loss_history=[0.7, 0.6, 0.5],
        )

    def fake_evaluate(**kwargs):
        calls.append("validation")
        assert np.array_equal(kwargs["weights"], adapted_weights)
        return 0.75, [[1.0, 0.0], [0.0, 1.0]], np.array([0, 1])

    monkeypatch.setattr(trainer, "train_vqc", fake_train_vqc)
    monkeypatch.setattr(trainer, "fine_tune_on_hardware", fake_finetune)
    monkeypatch.setattr(trainer, "_evaluate_real_predictions", fake_evaluate)
    monkeypatch.setattr(trainer, "estimate_accuracy_vs_snr", lambda **kwargs: {})

    result = trainer.train_and_evaluate(
        dataset,
        n_qubits=2,
        n_features=2,
        max_iterations=1,
        hardware_finetune_steps=3,
    )

    assert calls == ["local", "hardware_spsa", "validation"]
    assert result.n_epochs == 4
    assert result.n_circuit_evaluations == 14
    assert result.hardware_spsa["qpu_seconds"] is None
    assert result.accuracy_val_history == []


def test_vqc_loss_propagates_sampler_failures(monkeypatch):
    import numpy as np

    trainer_module = _load_vqc_trainer_module(monkeypatch)

    class FailingSampler:
        def run(self, pubs, shots):
            raise RuntimeError("sampler unavailable")

    loss_fn = trainer_module._make_vqc_loss_fn(
        X_train=np.zeros((2, 2)),
        y_train=np.array([0, 1]),
        n_qubits=2,
        n_features=2,
        backend_sampler=FailingSampler(),
        batch_size_local=2,
        feature_map_reps=1,
        ansatz_reps=1,
        readout_hidden_size=4,
    )
    theta = np.zeros(
        loss_fn.n_ansatz_params
        + trainer_module._n_readout_params(2, 2, loss_fn.hidden_size)
    )

    with pytest.raises(RuntimeError, match="Fallo de ejecución del circuito"):
        loss_fn(theta)


def test_real_prediction_failure_is_not_replaced_with_estimated_accuracy(monkeypatch):
    import numpy as np

    trainer_module = _load_vqc_trainer_module(monkeypatch)
    trainer = trainer_module.QiskitVQCTrainer()

    def fail_circuit_build(*args, **kwargs):
        raise RuntimeError("simulator unavailable")

    monkeypatch.setattr(trainer_module, "_build_feature_map_and_ansatz", fail_circuit_build)

    with pytest.raises(RuntimeError, match="Fallo de evaluación del circuito"):
        trainer._evaluate_real_predictions(
            weights=np.zeros(1),
            X_train=np.zeros((2, 2)),
            X_val=np.zeros((2, 2)),
            y_val=np.array([0, 1]),
            n_qubits=2,
            n_features=2,
            n_classes=2,
        )


def test_train_vqc_does_not_label_loss_estimate_as_validation_accuracy(monkeypatch):
    import numpy as np
    from types import SimpleNamespace

    trainer_module = _load_vqc_trainer_module(monkeypatch)
    optimizer_configs = []

    class FakeOptimizer:
        def __init__(self, config):
            optimizer_configs.append(config)

        def minimize(self, loss_fn, x0, callback):
            return SimpleNamespace(
                optimal_params=x0,
                final_loss=0.5,
                n_iter=1,
                n_evals=1,
                speedup_vs_spsa=1.0,
                converged=True,
                loss_history=[0.5],
                gradient_variance_history=[],
            )

    monkeypatch.setattr(trainer_module, "_make_vqc_loss_fn", lambda **kwargs: lambda theta: 0.5)
    monkeypatch.setattr(trainer_module, "QNSPSAEMLFeynman", FakeOptimizer)
    trainer = trainer_module.QiskitVQCTrainer(mode="sim", random_seed=17)

    result = trainer.train_vqc(
        X_train=np.zeros((2, 2)),
        y_train=np.array([0, 1]),
        num_qubits=2,
        n_features=2,
        max_iterations=1,
    )

    assert result["validation_accuracy"] is None
    assert result["estimated_accuracy"] is not None
    assert optimizer_configs[0].seed == 17


def test_stratified_indices_use_equal_per_class_counts(monkeypatch):
    import numpy as np

    trainer_module = _load_vqc_trainer_module(monkeypatch)
    labels = np.repeat(np.arange(3), 10)

    indices = trainer_module._stratified_indices(labels, max_samples=8, seed=5)

    assert len(indices) == 6
    np.testing.assert_array_equal(np.bincount(labels[indices]), [2, 2, 2])
