import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.diagnose_qnim_classical_baselines import _split_sha256
from scripts.run_qnim_q1 import REFERENCE_ID, _read_reference, run_q1


def _reference_files(tmp_path, datasets, config):
    report = {
        "decision": {"eligible_for_qnim_training": True},
        "pre_registered_protocol": {
            "eligibility_split": "validation_only",
            "test_evaluation_requested": False,
            "seeds": [42, 43, 44],
        },
        "final_test": {"status": "locked_not_evaluated"},
        "n_classes": 13,
        "waveform_source": "PyCBC",
        "strict_physics": True,
        "fallback_allowed": False,
        "approximant": "IMRPhenomPv2",
        "dataset_configuration": config,
        "datasets": datasets,
    }
    report_path = tmp_path / "pilot.json"
    report_bytes = json.dumps(report).encode("utf-8")
    report_path.write_bytes(report_bytes)
    manifest_path = tmp_path / "reference.json"
    manifest_path.write_text(
        json.dumps({
            "reference_id": REFERENCE_ID,
            "source_report": report_path.name,
            "source_sha256": hashlib.sha256(report_bytes).hexdigest().upper(),
        }),
        encoding="utf-8",
    )
    return manifest_path, report_path


def _fake_dataset(seed):
    rng = np.random.default_rng(seed)
    y_train = np.repeat(np.arange(13), 200)
    y_val = np.repeat(np.arange(13), 50)
    X_train = rng.normal(size=(len(y_train), 12))
    X_val = rng.normal(size=(len(y_val), 12))
    return SimpleNamespace(
        n_classes=13,
        theory_names=[f"class-{index}" for index in range(13)],
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )


def test_reference_manifest_rejects_changed_report(tmp_path):
    manifest_path, report_path = _reference_files(
        tmp_path, [], {"train_per_class": 2, "validation_per_class": 1,
                       "test_per_class": 1, "components": 12,
                       "pca_fit_components": 12}
    )

    report_path.write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        _read_reference(manifest_path)


def test_q1_reuses_hashed_train_validation_and_never_reads_test(tmp_path, monkeypatch):
    config = {
        "train_per_class": 200,
        "validation_per_class": 50,
        "test_per_class": 100,
        "components": 12,
        "pca_fit_components": 64,
    }
    datasets = []
    generated = {}
    for seed in [42, 43, 44]:
        dataset = _fake_dataset(seed)
        generated[seed] = dataset
        datasets.append({
            "seed": seed,
            "class_names": dataset.theory_names,
            "train_sha256": _split_sha256(dataset.X_train, dataset.y_train),
            "validation_sha256": _split_sha256(dataset.X_val, dataset.y_val),
        })
    manifest_path, _ = _reference_files(tmp_path, datasets, config)

    class LockedTestDataset:
        def __init__(self, dataset):
            self.dataset = dataset
            self.n_classes = dataset.n_classes
            self.theory_names = dataset.theory_names
            self.X_train = dataset.X_train
            self.y_train = dataset.y_train
            self.X_val = dataset.X_val
            self.y_val = dataset.y_val

        @property
        def X_test(self):
            raise AssertionError("QNIM Q1 read X_test")

        @property
        def y_test(self):
            raise AssertionError("QNIM Q1 read y_test")

    class FakeAdapter:
        def __init__(self, **kwargs):
            assert kwargs["strict_physics"] is True
            self.seed = None

        def generate_balanced_dataset(self, seed, **kwargs):
            return LockedTestDataset(generated[seed])

    class FakeTrainer:
        def __init__(self, **kwargs):
            assert kwargs["mode"] == "sim"
            self.seed = kwargs["random_seed"]

        def train_vqc(self, **kwargs):
            assert kwargs["num_qubits"] == 12
            assert kwargs["n_features"] == 12
            return {
                "weights": np.zeros(1),
                "training_loss": 1.0,
                "iterations": 1,
                "n_circuit_evaluations": 2,
                "execution_time_seconds": 0.1,
            }

        def _evaluate_real_predictions(self, **kwargs):
            np.testing.assert_array_equal(kwargs["X_train"], generated[self.seed].X_train)
            np.testing.assert_array_equal(kwargs["X_val"], generated[self.seed].X_val)
            return 1.0 / 13, None, np.zeros(len(kwargs["y_val"]), dtype=int)

    monkeypatch.setattr(
        "scripts.run_qnim_q1._architecture_metadata",
        lambda: {"n_qubits": 12},
    )
    output = tmp_path / "q1.json"
    result = run_q1(
        reference_manifest=manifest_path,
        output_path=output,
        max_iterations=1,
        shots=8,
        adapter_factory=FakeAdapter,
        trainer_factory=FakeTrainer,
    )

    assert result["status"] == "completed"
    assert result["completed_seeds"] == [42, 43, 44]
    assert result["final_test"] == {
        "evaluated": False,
        "status": "locked_not_evaluated",
        "models": {},
    }
    assert output.exists()
