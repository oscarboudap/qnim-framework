import importlib
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def _load_optimizer(monkeypatch):
    package = ModuleType("src.infrastructure")
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "src" / "infrastructure")]
    monkeypatch.setitem(sys.modules, "src.infrastructure", package)
    module_name = "src.infrastructure.hardware_spsa_optimizer"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    return importlib.import_module(module_name)


def test_spsa_uses_two_evaluations_with_the_same_batch_and_audits_usage(monkeypatch):
    module = _load_optimizer(monkeypatch)
    seen_batches = []

    def evaluator(parameters, batch_indices, shots):
        seen_batches.append(batch_indices.copy())
        return module.HardwareEvaluation(
            loss=float(np.sum(parameters**2)),
            job_ids=(f"job-{len(seen_batches)}",),
            shots=shots * len(batch_indices),
            pubs=len(batch_indices),
            qpu_seconds=0.25,
        )

    optimizer = module.HardwareSPSAOptimizer(module.HardwareSPSAConfig(steps=2, seed=5))
    result = optimizer.optimize(
        initial_parameters=np.array([1.0, -0.5]),
        n_samples=20,
        batch_size=4,
        shots=64,
        evaluator=evaluator,
    )

    np.testing.assert_array_equal(seen_batches[0], seen_batches[1])
    np.testing.assert_array_equal(seen_batches[2], seen_batches[3])
    assert not np.array_equal(seen_batches[0], seen_batches[2])
    assert result.steps_completed == 2
    assert result.jobs == 4
    assert result.pubs == 16
    assert result.shots == 1024
    assert result.qpu_seconds == 1.0
    assert len(result.job_ids) == 4


def test_reference_evaluation_is_periodic_and_uses_configured_shots(monkeypatch):
    module = _load_optimizer(monkeypatch)
    reference_calls = []

    def evaluator(parameters, batch_indices, shots):
        return module.HardwareEvaluation(float(np.sum(parameters**2)), (), shots, len(batch_indices), 0.0)

    def reference_evaluator(parameters, batch_indices, shots):
        reference_calls.append((batch_indices.copy(), shots))
        return module.HardwareEvaluation(float(np.sum(parameters**2)), (), shots, len(batch_indices), 0.0)

    result = module.HardwareSPSAOptimizer(
        module.HardwareSPSAConfig(steps=3, reference_every=2, reference_shots=17)
    ).optimize(
        initial_parameters=np.ones(2),
        n_samples=10,
        batch_size=3,
        shots=8,
        evaluator=evaluator,
        reference_evaluator=reference_evaluator,
        reference_indices=np.array([0, 1, 2]),
    )

    assert len(reference_calls) == 1
    assert reference_calls[0][1] == 17
    assert result.reference_history[0]["step"] == 2


def test_failed_second_evaluation_does_not_update_parameters(monkeypatch):
    module = _load_optimizer(monkeypatch)
    initial = np.array([0.75, -0.25])
    calls = 0

    def evaluator(parameters, batch_indices, shots):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("job failed")
        return module.HardwareEvaluation(1.0, ("job-1",), shots, len(batch_indices), 0.1)

    optimizer = module.HardwareSPSAOptimizer(module.HardwareSPSAConfig(steps=1, seed=2))

    with pytest.raises(module.HardwareSPSAExecutionError, match="job failed") as exc_info:
        optimizer.optimize(initial, 10, 4, 32, evaluator)

    np.testing.assert_array_equal(exc_info.value.parameters, initial)
    assert exc_info.value.audit.steps_completed == 0
    assert exc_info.value.audit.jobs == 1


def test_hardware_evaluation_rejects_nonfinite_loss(monkeypatch):
    module = _load_optimizer(monkeypatch)

    def evaluator(parameters, batch_indices, shots):
        return module.HardwareEvaluation(float("nan"), (), shots, len(batch_indices), 0.0)

    with pytest.raises(module.HardwareSPSAExecutionError, match="finita"):
        module.HardwareSPSAOptimizer(module.HardwareSPSAConfig(steps=1)).optimize(
            np.array([1.0]), 4, 2, 8, evaluator
        )
