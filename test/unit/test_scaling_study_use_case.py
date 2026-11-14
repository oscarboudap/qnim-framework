import json
import sys
from types import SimpleNamespace
from types import ModuleType

import numpy as np
import pytest

from src.application.use_cases.run_scaling_study_use_case import RunScalingStudyUseCase


class FakeTrainer:
    def __init__(self, predicted_labels):
        self.predicted_labels = predicted_labels

    def train_and_evaluate(self, dataset, **kwargs):
        assert kwargs["use_real_hardware"] is False
        assert kwargs["compute_accuracy_vs_snr"] is False
        self.training_features = dataset.X_train.copy()
        self.feature_count = kwargs["n_features"]
        return SimpleNamespace(accuracy_sim=0.5, final_weights=np.array([1.0]))

    def _evaluate_real_predictions(self, **kwargs):
        labels = kwargs["y_val"]
        predictions = self.predicted_labels
        matrix = np.eye(kwargs["n_classes"]).tolist()
        return float(np.mean(labels == predictions)), matrix, predictions.copy()


def _write_dataset(path):
    path.mkdir()
    np.save(path / "X_train.npy", np.arange(24 * 12, dtype=float).reshape(24, 12))
    np.save(path / "y_train.npy", np.repeat(np.arange(3), 8))
    np.save(path / "X_val.npy", np.arange(9 * 12, dtype=float).reshape(9, 12))
    np.save(path / "y_val.npy", np.repeat(np.arange(3), 3))
    np.save(path / "X_test.npy", np.arange(12 * 12, dtype=float).reshape(12, 12))
    np.save(path / "y_test.npy", np.repeat(np.arange(3), 4))


def test_scaling_study_uses_validation_without_reading_test_and_saves_results(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir)
    (dataset_dir / "X_test.npy").unlink()
    (dataset_dir / "y_test.npy").unlink()
    experiments = (
        {"id": "n12_d12", "n_qubits": 12, "pca_components": 12},
        {"id": "n16_d12", "n_qubits": 16, "pca_components": 12},
    )
    study = RunScalingStudyUseCase(
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "results",
        bootstrap_iterations=100,
        permutation_iterations=100,
        experiments=experiments,
    )
    seen_train_rows = []

    def build_trainer(experiment):
        seen_train_rows.append(experiment["id"])
        predictions = (
            np.zeros(9, dtype=int)
            if experiment["id"] == "n12_d12"
            else np.repeat(np.arange(3), 3)
        )
        return FakeTrainer(predictions)

    monkeypatch.setattr(study, "_build_trainer", build_trainer)
    monkeypatch.setattr(study, "_commit_hash", lambda: None)

    summary = study.execute()

    assert seen_train_rows == ["n12_d12", "n16_d12"]
    assert summary["records"][1]["paired_difference_vs_reference"]["lower_95"] > 0
    assert summary["records"][0]["dataset_sha256"] == summary["records"][1]["dataset_sha256"]
    assert summary["records"][1]["y_true"] == summary["records"][0]["y_true"]
    assert summary["records"][0]["evaluation_split"] == "validation"
    assert len(summary["records"][0]["y_true"]) == 9
    assert (tmp_path / "results/summary/scaling_metrics.csv").is_file()
    saved = json.loads(
        (tmp_path / "results/experiments/n16_d12_seed42/result.json").read_text()
    )
    assert saved["backend"] == "local_simulator"
    assert saved["y_pred"] == np.repeat(np.arange(3), 3).tolist()


def test_scaling_study_requires_independent_test_files(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    np.save(dataset_dir / "X_train.npy", np.ones((3, 12)))
    np.save(dataset_dir / "y_train.npy", np.array([0, 1, 2]))

    study = RunScalingStudyUseCase(dataset_dir=dataset_dir)

    with pytest.raises(FileNotFoundError, match="X_val.npy"):
        study._load_dataset()


def test_scaling_study_passes_training_seed_to_trainer(tmp_path, monkeypatch):
    import importlib
    from pathlib import Path

    package = ModuleType("src.infrastructure")
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "src" / "infrastructure")]
    trainer_module = ModuleType("src.infrastructure.qiskit_vqc_trainer")
    captured = {}

    class SeedCapturingTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    trainer_module.QiskitVQCTrainer = SeedCapturingTrainer
    monkeypatch.setitem(sys.modules, "src.infrastructure", package)
    monkeypatch.setitem(sys.modules, "src.infrastructure.qiskit_vqc_trainer", trainer_module)

    study = RunScalingStudyUseCase(dataset_dir=tmp_path, seed=19)
    study._build_trainer({"id": "n12_d12"})

    assert captured["random_seed"] == 19