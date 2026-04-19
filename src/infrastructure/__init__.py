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

# ============================================================================
# STORAGE ADAPTERS
# ============================================================================

from src.infrastructure.storage.hdf5_exporter import (
    HDF5Exporter as StorageHDF5Exporter,
)

from src.infrastructure.storage.quantum_dataloader import (
    QuantumDatasetLoader,
)

from src.infrastructure.storage.massive_loader import (
    MassiveDatasetLoader,
)

# ============================================================================
# QUANTUM COMPUTING ADAPTERS
# ============================================================================

from src.infrastructure.neal_annealer_adapter import (
    NealSimulatedAnnealerAdapter,
)

from src.infrastructure.ibm_quantum_adapter import (
    IBMQuantumAdapter,
)

# ============================================================================
# APPLICATION PORT IMPLEMENTATIONS (NEW ADAPTERS)
# ============================================================================

from src.infrastructure.qiskit_vqc_trainer import (
    QiskitVQCTrainer,
)

from src.infrastructure.sklearn_preprocessing_adapter import (
    SklearnPreprocessor,
)

from src.infrastructure.matplotlib_metrics_reporter import (
    MatplotlibMetricsReporter,
)

from src.infrastructure.sstg_adapter import (
    SSTGAdapter,
)

# ============================================================================
# EXCEPTION FRAMEWORK
# ============================================================================

from src.infrastructure.exceptions import (
    InfrastructureException,
    DataLoaderException,
    StorageException,
    QuantumComputeException,
    TrainingException,
    PreprocessingException,
    ReportingException,
)

# ============================================================================
# EXPORTS: All public API
# ============================================================================

__all__ = [
    # Exceptions
    "InfrastructureException",
    "DataLoaderException",
    "StorageException",
    "QuantumComputeException",
    "TrainingException",
    "PreprocessingException",
    "ReportingException",
    
    # Storage Adapters (IStoragePort, IDataLoaderPort)
    "StorageHDF5Exporter",
    "QuantumDatasetLoader",
    "MassiveDatasetLoader",
    
    # Quantum Computing Adapters (IQuantumOptimizerPort, IGateBasedQuantumComputer)
    "NealSimulatedAnnealerAdapter",
    "IBMQuantumAdapter",
    
    # Application Port Implementations
    "QiskitVQCTrainer",              # Implements IQuantumMLTrainerPort
    "SklearnPreprocessor",           # Implements IPreprocessingPort
    "MatplotlibMetricsReporter",     # Implements IMetricsReporterPort
    "SSTGAdapter",                   # Implements ISyntheticDataGeneratorPort
]

