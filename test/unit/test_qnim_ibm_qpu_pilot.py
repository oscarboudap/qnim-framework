import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.infrastructure.hardware_spsa_optimizer import HardwareSPSAOptimizer
from src.infrastructure.qnim_ibm_qpu_pilot import (
    DurableJobRecordStore,
    DurableSampler,
    PILOT_EXPERIMENT_ID,
    PilotBudgetError,
    PilotPreflightError,
    estimate_resources,
    load_protocol,
    require_submission_enabled,
    save_parameter_checkpoint,
    validate_connected_layout,
    verify_parameter_checkpoint,
    verify_dataset_provenance,
    validate_pilot_job_context,
)


ROOT = Path(__file__).resolve().parents[2]


class TestSplitSentinel:
    X_train = np.zeros((4, 2))
    y_train = np.array([0, 1, 0, 1])
    X_val = np.ones((2, 2))
    y_val = np.array([0, 1])
    n_classes = 2

    @property
    def X_test(self):
        raise AssertionError("X_test must remain sealed")

    @property
    def y_test(self):
        raise AssertionError("y_test must remain sealed")


def _trainer_module(monkeypatch):
    import importlib
    import sys
    from types import ModuleType

    infrastructure_package = ModuleType("src.infrastructure")
    infrastructure_package.__path__ = [
        str(ROOT / "src" / "infrastructure")
    ]
    monkeypatch.setitem(sys.modules, "src.infrastructure", infrastructure_package)
    name = "src.infrastructure.qiskit_vqc_trainer"
    monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module(name)


def test_local_training_hardware_orchestration_and_validation_never_read_test(monkeypatch):
    module = _trainer_module(monkeypatch)
    trainer = module.QiskitVQCTrainer(
        mode="ibm",
        token="mock-token",
        train_on_hardware=True,
        random_seed=2,
    )
    trainer.train_vqc = lambda **kwargs: {
        "weights": np.array([1.0, 2.0]),
        "loss_history": [0.8],
        "iterations": 1,
        "speedup_vs_spsa": 1.0,
        "n_circuit_evaluations": 2,
        "execution_time_seconds": 0.1,
        "converged": True,
        "gradient_variance_history": [],
    }
    trainer.fine_tune_on_hardware = lambda **kwargs: SimpleNamespace(
        parameters=np.array([2.0, 3.0]),
        steps_completed=1,
        jobs=2,
        pubs=4,
        shots=40,
        qpu_seconds=None,
        job_ids=["mock-job"],
        reference_history=[],
        loss_history=[0.6],
    )

    def validation(weights, dataset, n_qubits, n_features, use_zne, **kwargs):
        assert np.array_equal(dataset.X_val, TestSplitSentinel.X_val)
        assert np.array_equal(dataset.y_val, TestSplitSentinel.y_val)
        return 0.75, 0.75

    trainer._validate_on_ibm = validation
    trainer._evaluate_real_predictions = lambda **kwargs: (
        0.5,
        [[0.5, 0.5], [0.5, 0.5]],
        np.array([0, 0]),
    )
    trainer.estimate_accuracy_vs_snr = lambda **kwargs: {}

    result = trainer.train_and_evaluate(
        TestSplitSentinel(),
        n_qubits=2,
        n_features=2,
        max_iterations=1,
        use_real_hardware=True,
        hardware_finetune_steps=1,
        compute_accuracy_vs_snr=False,
    )
    assert result.accuracy_real_no_zne == 0.75


def test_validation_route_selects_validation_arrays_without_test_access(monkeypatch):
    module = _trainer_module(monkeypatch)
    trainer = module.QiskitVQCTrainer(mode="sim")
    captured = {}

    def evaluate_split(**kwargs):
        captured.update(kwargs)
        return 0.5, 0.5

    trainer._evaluate_on_ibm_split = evaluate_split
    trainer._validate_on_ibm(
        np.zeros(2),
        TestSplitSentinel(),
        n_qubits=2,
        n_features=2,
        use_zne=False,
    )
    assert captured["split_name"] == "validation"
    assert captured["X_eval"] is TestSplitSentinel.X_val
    assert captured["y_eval"] is TestSplitSentinel.y_val


def test_validation_metric_generation_uses_the_supplied_validation_labels(monkeypatch):
    module = _trainer_module(monkeypatch)
    metrics = module._validation_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 1]),
        n_classes=2,
    )
    assert metrics["accuracy"] == 0.75
    assert metrics["balanced_accuracy"] == 0.75
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]


def test_final_test_evaluation_requires_explicit_gate_before_access(monkeypatch):
    module = _trainer_module(monkeypatch)
    trainer = module.QiskitVQCTrainer(mode="sim")
    with pytest.raises(PermissionError, match="authorize_test_access"):
        trainer.evaluate_test_on_ibm(
            np.zeros(2),
            TestSplitSentinel(),
            n_qubits=2,
            n_features=2,
        )


def test_provenance_verification_rejects_unpinned_dataset(tmp_path):
    protocol = load_protocol()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "dataset_manifest.json").write_text("{}", encoding="utf-8")
    (dataset_dir / "dataset_metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(PilotPreflightError, match="manifest hash"):
        verify_dataset_provenance(dataset_dir, protocol, ROOT)


def test_resource_estimate_matches_declared_source_accounting():
    estimate = estimate_resources(load_protocol())
    assert estimate["local_qnspsa_objective_calls"] == 401
    assert estimate["local_sampler_runs"] == 403
    assert estimate["local_pubs_and_circuits"] == 3264
    assert estimate["local_shots"] == 916_992
    assert estimate["hardware_qpu_jobs"] == 11
    assert estimate["validation_qpu_jobs"] == 1
    assert estimate["total_qpu_jobs"] == 12
    assert estimate["total_qpu_pubs_and_circuits"] == 121
    assert estimate["total_qpu_shots"] == 46_592
    assert estimate["qpu_seconds"] == -1
    assert estimate["resource_units"] == -1


def test_resource_estimate_rejects_over_budget_configuration():
    protocol = load_protocol()
    protocol["resource_budget"]["max_total_qpu_shots"] = 100
    with pytest.raises(PilotBudgetError, match="max_total_qpu_shots"):
        estimate_resources(protocol)


def test_submission_is_disabled_by_default():
    protocol = load_protocol()
    assert protocol["submission_enabled"] is False
    with pytest.raises(PilotPreflightError, match="disabled"):
        require_submission_enabled(protocol, submit_requested=True)


def test_direct_pilot_execution_context_respects_disabled_submission():
    with pytest.raises(PilotPreflightError, match="disabled"):
        validate_pilot_job_context(
            None,
            {"experiment_id": PILOT_EXPERIMENT_ID},
        )


def test_optimizer_identity_remains_hardware_spsa():
    protocol = load_protocol()
    assert protocol["algorithm_identity"]["pretraining"] == "local QNSPSA-EML-Feynman"
    assert protocol["algorithm_identity"]["hardware_update"] == (
        "IBM HardwareSPSAOptimizer (SPSA fine-tuning)"
    )
    assert protocol["algorithm_identity"]["remote_qnspsa_eml_feynman"] is False
    assert HardwareSPSAOptimizer.__name__ == "HardwareSPSAOptimizer"


def test_layout_must_be_unique_in_range_and_connected():
    assert validate_connected_layout(
        (1, 2, 3, 4), {(1, 2), (2, 3), (3, 4)}, logical_qubits=4, backend_qubits=5
    ) == (1, 2, 3, 4)
    with pytest.raises(ValueError, match="connected path"):
        validate_connected_layout(
            (0, 2, 3, 4), {(0, 1), (1, 2), (2, 3), (3, 4)},
            logical_qubits=4, backend_qubits=5,
        )
    with pytest.raises(ValueError, match="unique"):
        validate_connected_layout(
            (1, 2, 2, 3), {(1, 2), (2, 3)}, logical_qubits=4, backend_qubits=5
        )


def test_durable_sampler_persists_intent_and_job_before_result(tmp_path):
    record_dir = tmp_path / "records"
    store = DurableJobRecordStore(record_dir)
    metadata = {
        "run_id": "pilot-run",
        "experiment_id": "TEST_DURABLE_SAMPLER",
        "protocol_hash": "protocol",
        "dataset_hashes": {"train": "train-hash"},
        "dataset_provenance": {"strict_physics": True},
        "pca_hash": "pca-hash",
        "parameter_hash": "parameters",
        "logical_qubits": 8,
        "backend": "mock-backend",
        "physical_layout": list(range(8)),
        "transpiler_configuration": {"optimization_level": 1},
        "optimizer_state": {"step": 1},
        "random_seed": 42,
        "sampling_policy": {"shots": 8},
        "calibration_metadata": {"captured": True},
    }

    class FakeJob:
        def job_id(self):
            return "job-123"

        def metrics(self):
            return {"usage": {"quantum_seconds": 1.25, "resource_units": 2}}

        def result(self):
            return ["counts"]

    class FakeSampler:
        def run(self, pubs, shots):
            records = list(record_dir.glob("*.json"))
            assert len(records) == 1
            assert json.loads(records[0].read_text())["status"] == "submission_intent"
            assert shots == 8
            return FakeJob()

    sampler = DurableSampler(
        FakeSampler(),
        store,
        metadata,
        circuit_hasher=lambda pubs: "circuit-hash",
    )
    sampler.set_execution_context(
        {
            "parameter_hash": "parameters-step-1",
            "optimizer_state": {"step": 1},
            "sampling_policy": {"shots": 8},
        }
    )
    job = sampler.run([object()], shots=8)
    assert job.job_id() == "job-123"
    assert job.result() == ["counts"]
    record = json.loads(next(record_dir.glob("*.json")).read_text())
    assert record["status"] == "completed"
    assert record["job_id"] == "job-123"
    assert record["result_sha256"]
    assert record["qpu_seconds"] == 1.25
    assert record["resource_units"] == 2


def test_unresolved_submission_record_blocks_automatic_resubmission(tmp_path):
    store = DurableJobRecordStore(tmp_path)
    metadata = {
        "run_id": "pilot-run",
        "experiment_id": "TEST_UNRESOLVED_RECORD",
        "protocol_hash": "protocol",
        "dataset_hashes": {"train": "train-hash"},
        "dataset_provenance": {"strict_physics": True},
        "pca_hash": "pca-hash",
        "parameter_hash": "parameters",
        "logical_qubits": 8,
        "backend": "mock-backend",
        "physical_layout": list(range(8)),
        "transpiler_configuration": {},
        "optimizer_state": {},
        "random_seed": 42,
        "sampling_policy": {},
        "calibration_metadata": {},
    }
    store.create_intent(
        "first-job", metadata, pubs=1, shots_per_pub=8, circuit_sha256="circuit"
    )
    with pytest.raises(PilotPreflightError, match="unresolved"):
        store.create_intent(
            "retry-job", metadata, pubs=1, shots_per_pub=8, circuit_sha256="circuit"
        )


def test_pilot_record_blocks_retry_until_evaluation_is_persisted(monkeypatch, tmp_path):
    import src.infrastructure.qnim_ibm_qpu_pilot as pilot_module

    monkeypatch.setattr(
        pilot_module, "validate_pilot_job_context", lambda *args, **kwargs: None
    )
    store = DurableJobRecordStore(tmp_path, max_jobs=2, max_shots=4096)
    metadata = {
        "run_id": "pilot-run",
        "experiment_id": PILOT_EXPERIMENT_ID,
        "protocol_hash": "protocol",
        "dataset_hashes": {"train": "train"},
        "dataset_provenance": {"strict_physics": True},
        "pca_hash": "pca",
        "parameter_hash": "parameters",
        "logical_qubits": 8,
        "backend": "mock-backend",
        "physical_layout": list(range(8)),
        "transpiler_configuration": {},
        "optimizer_state": {},
        "random_seed": 42,
        "sampling_policy": {},
        "calibration_metadata": {},
    }

    class FakeJob:
        def job_id(self):
            return "job-1"

        def metrics(self):
            return {}

        def result(self):
            return ["counts"]

    class FakeSampler:
        def run(self, pubs, shots):
            return FakeJob()

    sampler = DurableSampler(
        FakeSampler(),
        store,
        metadata,
        circuit_hasher=lambda pubs: "circuit",
    )
    sampler.set_execution_context(
        {
            "optimizer_state": {
                "step": 1,
                "evaluation_role": "spsa_plus",
            },
            "sampling_policy": {
                "sample_indices": [0, 1, 2, 3],
                "seed": 42,
            },
        }
    )
    job = sampler.run([object()] * 4, shots=128)
    job.result()
    record_path = next(tmp_path.glob("*.json"))
    assert json.loads(record_path.read_text())["status"] == "evaluation_incomplete"
    with pytest.raises(PilotPreflightError, match="unresolved"):
        store.create_intent(
            "retry",
            metadata,
            pubs=1,
            shots_per_pub=8,
            circuit_sha256="circuit",
        )

    sampler.update_last_record({"evaluation_loss": 0.5})
    assert json.loads(record_path.read_text())["status"] == "completed"


def test_job_record_store_enforces_cumulative_job_and_shot_budgets(tmp_path):
    store = DurableJobRecordStore(
        tmp_path,
        max_jobs=1,
        max_shots=8,
    )
    metadata = {
        "run_id": "pilot-run",
        "experiment_id": "TEST_JOB_BUDGET",
        "protocol_hash": "protocol",
        "dataset_hashes": {"train": "train-hash"},
        "dataset_provenance": {"strict_physics": True},
        "pca_hash": "pca-hash",
        "parameter_hash": "parameters",
        "logical_qubits": 8,
        "backend": "mock-backend",
        "physical_layout": list(range(8)),
        "transpiler_configuration": {},
        "optimizer_state": {},
        "random_seed": 42,
        "sampling_policy": {},
        "calibration_metadata": {},
    }
    record = store.create_intent(
        "first-job", metadata, pubs=1, shots_per_pub=8, circuit_sha256="circuit"
    )
    store.update(record, {"status": "completed"})
    with pytest.raises(PilotBudgetError, match="job budget"):
        store.create_intent(
            "second-job", metadata, pubs=1, shots_per_pub=1, circuit_sha256="circuit"
        )
    shot_limited_store = DurableJobRecordStore(
        tmp_path / "shot-limit",
        max_jobs=2,
        max_shots=8,
    )
    first = shot_limited_store.create_intent(
        "first-job", metadata, pubs=1, shots_per_pub=8, circuit_sha256="circuit"
    )
    shot_limited_store.update(first, {"status": "completed"})
    with pytest.raises(PilotBudgetError, match="shot budget"):
        shot_limited_store.create_intent(
            "second-job", metadata, pubs=1, shots_per_pub=1, circuit_sha256="circuit"
        )


def test_pilot_ledger_rejects_indirect_step_and_validation_budget_overruns(tmp_path):
    base_metadata = {
        "run_id": "pilot-run",
        "experiment_id": PILOT_EXPERIMENT_ID,
        "protocol_hash": "protocol",
        "dataset_hashes": {"train": "train"},
        "dataset_provenance": {"strict_physics": True},
        "pca_hash": "pca",
        "parameter_hash": "parameters",
        "logical_qubits": 8,
        "backend": "mock-backend",
        "physical_layout": list(range(8)),
        "transpiler_configuration": {},
        "random_seed": 42,
        "calibration_metadata": {},
    }

    step_store = DurableJobRecordStore(tmp_path / "steps", max_jobs=16, max_shots=60000)
    with pytest.raises(PilotBudgetError, match="step"):
        step_store.create_intent(
            "step-six",
            {
                **base_metadata,
                "optimizer_state": {
                    "step": 6,
                    "evaluation_role": "spsa_plus",
                },
                "sampling_policy": {"sample_indices": [0, 1, 2, 3]},
            },
            pubs=4,
            shots_per_pub=128,
            circuit_sha256="circuit",
        )

    validation_store = DurableJobRecordStore(
        tmp_path / "validation", max_jobs=16, max_shots=60000
    )
    with pytest.raises(PilotBudgetError, match="validation example"):
        validation_store.create_intent(
            "validation-overflow",
            {
                **base_metadata,
                "optimizer_state": {"phase": "validation"},
                "sampling_policy": {"split": "validation"},
                "validation_examples": 66,
            },
            pubs=66,
            shots_per_pub=512,
            circuit_sha256="circuit",
        )


def test_checkpoint_integrity_detects_modified_array(tmp_path):
    values = np.array([0.25, -0.5])
    save_parameter_checkpoint(
        tmp_path,
        "weights",
        values,
        {"experiment_id": "QNIM_IBM_QPU_PILOT_V1"},
    )
    assert verify_parameter_checkpoint(tmp_path, "weights")["parameter_sha256"]
    with (tmp_path / "weights.npz").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(PilotPreflightError, match="array hash"):
        verify_parameter_checkpoint(tmp_path, "weights")
