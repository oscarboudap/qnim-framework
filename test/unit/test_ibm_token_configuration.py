import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_collector(monkeypatch):
    package = ModuleType("src.infrastructure")
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "src" / "infrastructure")]
    monkeypatch.setitem(sys.modules, "src.infrastructure", package)
    module_name = "src.infrastructure.ibm_quantum_results_collector"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    return importlib.import_module(module_name)


def test_collector_reads_token_from_environment(monkeypatch):
    module = _load_collector(monkeypatch)
    monkeypatch.setattr(module, "_QISKIT_AVAILABLE", True)
    monkeypatch.setenv("IBM_QUANTUM_TOKEN", "unit-test-token")

    collector = module.IBMQuantumResultsCollector(use_real_hardware=True)

    assert collector.token == "unit-test-token"


def test_explicit_token_takes_precedence_over_environment(monkeypatch):
    module = _load_collector(monkeypatch)
    monkeypatch.setattr(module, "_QISKIT_AVAILABLE", True)
    monkeypatch.setenv("IBM_QUANTUM_TOKEN", "environment-test-token")

    collector = module.IBMQuantumResultsCollector(
        token="explicit-test-token",
        use_real_hardware=True,
    )

    assert collector.token == "explicit-test-token"


def test_collector_requires_token_for_hardware_mode(monkeypatch):
    module = _load_collector(monkeypatch)
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    monkeypatch.setattr(module, "get_ibm_quantum_token", lambda token=None: "")

    with pytest.raises(RuntimeError, match="IBM_QUANTUM_TOKEN"):
        module.IBMQuantumResultsCollector(use_real_hardware=True)


def test_shared_resolver_reads_local_env_file(monkeypatch, tmp_path):
    from src import ibm_credentials

    env_file = tmp_path / ".env"
    env_file.write_text("IBM_QUANTUM_TOKEN=local-test-token\n", encoding="utf-8")
    monkeypatch.setattr(ibm_credentials, "_ENV_PATH", env_file)
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)

    assert ibm_credentials.get_ibm_quantum_token() == "local-test-token"


def test_legacy_collector_refuses_to_report_unmeasured_hardware_results(monkeypatch):
    module = _load_collector(monkeypatch)
    collector = object.__new__(module.IBMQuantumResultsCollector)
    collector.use_real_hardware = True

    with pytest.raises(NotImplementedError, match="no ejecuta predicciones en QPU"):
        collector.run_full_experiment(
            X_train=None,
            y_train=None,
            X_val=None,
            y_val=None,
        )


@pytest.mark.parametrize(
    ("method_name", "args", "match"),
    [
        ("_validate_on_real_hardware", (None, None, None), "no envía los circuitos"),
        ("_compute_qfi_vs_cfi", (None, None, None), "no se generan valores de referencia sintéticos"),
        ("_reanalyze_gw150914", (None,), "no se devolverán parámetros sintéticos"),
    ],
)
def test_legacy_collector_rejects_unimplemented_physics_claims(
    monkeypatch, method_name, args, match
):
    module = _load_collector(monkeypatch)
    method = getattr(module.IBMQuantumResultsCollector, method_name)

    with pytest.raises(NotImplementedError, match=match):
        method(object.__new__(module.IBMQuantumResultsCollector), *args)
