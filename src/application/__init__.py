"""
Application Layer: Use Cases & Orchestration
=============================================

Patrones:
- Clean Architecture: Application encapsula Business Rules
- Hexagonal Architecture: Puertos abstractos, adaptadores en infraestructura
- Data Transfer Objects: DTOs para entrada/salida (type safety)
- Use Cases: Cada clase = un caso de uso (Single Responsibility)

Public API:
"""

# ============================================================================
# USE CASES (Aplicación)
# ============================================================================

from src.application.hybrid_orchestrator import HybridInferenceOrchestrator
from src.application.model_training_service import ModelTrainingUseCase
from src.application.process_event_use_case import DecodeGravitationalWaveUseCase
from src.application.validation_service import ModelValidationUseCase
from src.application.sstg_service import SyntheticDataGenerationUseCase

# ============================================================================
# DTOs & Value Objects
# ============================================================================

from src.application.dto import (
    # Inference pipeline
    ClassicParametersExtracted,
    BeyondGRSignature,
    QuantumTopologySignature,
    DeepQuantumManifoldSignature,
    ClassificationResult,
    InferenceResult,
    # Training & Validation
    TrainingMetrics,
    ConfusionMatrixData,
    SyntheticDatasetInfo,
    # Configuration
    ClassificationThresholds,
    # Exceptions
    ApplicationException,
    PortNotAvailableException,
    InvalidInputException,
    InferenceFailedException,
    TrainingFailedException,
)

# ============================================================================
# PORT INTERFACES (Hexagonal Architecture)
# ============================================================================

from src.application.ports import (
    # Data Access
    IDataLoaderPort,
    IStoragePort,
    # Quantum Optimization
    IQuantumOptimizerPort,
    # ML Training
    IQuantumMLTrainerPort,
    IPreprocessingPort,
    # Reporting
    IMetricsReporterPort,
    # Data Synthesis
    ISyntheticDataGeneratorPort,
    # Repository
    IQuantumDecodedEventRepository,
    # Observables
    ITrainingProgressObserver,
)

__all__ = [
    # ====== USE CASES ======
    "HybridInferenceOrchestrator",
    "ModelTrainingUseCase",
    "DecodeGravitationalWaveUseCase",
    "ModelValidationUseCase",
    "SyntheticDataGenerationUseCase",
    
    # ====== DTOs ======
    "ClassicParametersExtracted",
    "BeyondGRSignature",
    "QuantumTopologySignature",
    "DeepQuantumManifoldSignature",
    "ClassificationResult",
    "InferenceResult",
    "TrainingMetrics",
    "ConfusionMatrixData",
    "SyntheticDatasetInfo",
    "ClassificationThresholds",
    
    # ====== EXCEPTIONS ======
    "ApplicationException",
    "PortNotAvailableException",
    "InvalidInputException",
    "InferenceFailedException",
    "TrainingFailedException",
    
    # ====== PORTS ======
    "IDataLoaderPort",
    "IStoragePort",
    "IQuantumOptimizerPort",
    "IQuantumMLTrainerPort",
    "IPreprocessingPort",
    "IMetricsReporterPort",
    "ISyntheticDataGeneratorPort",
    "IQuantumDecodedEventRepository",
    "ITrainingProgressObserver",
]

__version__ = "2.0.0"  # After DDD refactor

