"""
Infrastructure Layer
====================

Hexagonal Architecture: Adapters implementing Abstract Ports
-----------------------------------------------------------

Outside the Boundary: Infrastructure adapters and concrete implementations.

Port Implementations (Application ← Infrastructure):
    - IQuantumOptimizerPort ← NealSimulatedAnnealerAdapter
    - IQuantumMLTrainerPort ← QiskitVQCTrainer
    - IPreprocessingPort ← SklearnPreprocessor
    - IMetricsReporterPort ← MatplotlibMetricsReporter
    - IDataLoaderPort ← QuantumDatasetLoader
    - IStoragePort ← HDF5Exporter
    - ISyntheticDataGeneratorPort ← SSTGAdapter

Domain Interfaces (Infrastructure ← Domain):
    - IGateBasedQuantumComputer ← IBMQuantumAdapter
    - IQuantumAnnealer ← NealSimulatedAnnealerAdapter

Exception Framework:
    - InfrastructureException (base)
    - DataLoaderException
    - StorageException
    - QuantumComputeException
    - TrainingException
    - PreprocessingException
    - ReportingException
"""

from importlib import import_module


_EXPORTS = {
    "InfrastructureException": (".exceptions", "InfrastructureException"),
    "DataLoaderException": (".exceptions", "DataLoaderException"),
    "StorageException": (".exceptions", "StorageException"),
    "QuantumComputeException": (".exceptions", "QuantumComputeException"),
    "TrainingException": (".exceptions", "TrainingException"),
    "PreprocessingException": (".exceptions", "PreprocessingException"),
    "ReportingException": (".exceptions", "ReportingException"),
    "StorageHDF5Exporter": (".storage.hdf5_exporter", "HDF5Exporter"),
    "QuantumDatasetLoader": (".storage.quantum_dataloader", "QuantumDatasetLoader"),
    "MassiveDatasetLoader": (".storage.massive_loader", "MassiveDatasetLoader"),
    "NealSimulatedAnnealerAdapter": (".neal_annealer_adapter", "NealSimulatedAnnealerAdapter"),
    "IBMQuantumAdapter": (".ibm_quantum_adapter", "IBMQuantumAdapter"),
    "QiskitVQCTrainer": (".qiskit_vqc_trainer", "QiskitVQCTrainer"),
    "SklearnPreprocessor": (".sklearn_preprocessing_adapter", "SklearnPreprocessor"),
    "MatplotlibMetricsReporter": (".matplotlib_metrics_reporter", "MatplotlibMetricsReporter"),
    "SSTGAdapter": (".sstg_adapter", "SSTGAdapter"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value

