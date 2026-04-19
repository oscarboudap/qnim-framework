"""
Script Layer Module

Provides infrastructure for scripts and command-line entry points to invoke
the application layer according to hexagonal/clean architecture principles.

Modules:
  - script_config: Centralized configuration management with VOs
  - script_exceptions: Typed exception hierarchy
  - script_container: Dependency injection container
  - script_logging: Structured logging utilities

Example (train.py):
  from src.scripts import ScriptConfig, ScriptContainer, get_script_logger
  
  def main() -> None:
      logger = get_script_logger("train")
      try:
          config = ScriptConfig.from_env()
          container = ScriptContainer(config)
          use_case = container.get_training_use_case()
          logger.info("Starting training", extra={
              "qubits": config.training.vqc.num_qubits,
              "max_iterations": config.training.vqc.max_iterations
          })
          use_case.execute(config.training)
          logger.info("Training completed successfully")
      except ScriptException as e:
          logger.error(f"Script failed: {e}", exc_info=True)
          raise SystemExit(1)

Architecture Principles:
  1. Scripts ARE NOT infrastructure or domain code
  2. Scripts DO invoke application layer use cases
  3. Scripts DO use dependency injection
  4. Scripts DO validate all inputs
  5. Scripts DO handle all errors gracefully
  6. Scripts DO provide structured logging/monitoring
  7. Scripts DO centralize configuration

Compliance Matrix:
  ✅ DDD: Value objects for configuration (ScriptConfig)
  ✅ Hexagonal: Application layer via container dependency injection
  ✅ Clean: Presentation layer (scripts) doesn't know about domain/infra
  ✅ Type Safety: 100% type hints on all exported functions
  ✅ Error Handling: Custom exception types for each failure mode
  ✅ Configuration: Centralized in frozen dataclasses
  ✅ Testing: All components mockable and testable
  ✅ Observability: Structured logging with progress tracking
"""

# ============================================================================
# CONFIGURATION EXPORTS
# ============================================================================

from .script_config import (
    QuantumVQCConfig,
    PreprocessingConfig,
    PathsConfig,
    TrainingConfig,
    SweepExperimentConfig,
    InferenceConfig,
    PlottingConfig,
    ScriptConfig,
)

# ============================================================================
# EXCEPTION EXPORTS
# ============================================================================

from .script_exceptions import (
    ScriptException,
    ScriptConfigurationException,
    ScriptExecutionException,
    ScriptDataException,
    ScriptComputationException,
    ScriptIOException,
)

# ============================================================================
# CONTAINER & DI EXPORTS
# ============================================================================

from .script_container import ScriptContainer

# ============================================================================
# LOGGING EXPORTS
# ============================================================================

from .script_logging import (
    get_script_logger,
    create_execution_context,
    ProgressTracker,
)

# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Configuration
    "QuantumVQCConfig",
    "PreprocessingConfig",
    "PathsConfig",
    "TrainingConfig",
    "SweepExperimentConfig",
    "InferenceConfig",
    "PlottingConfig",
    "ScriptConfig",
    
    # Exceptions
    "ScriptException",
    "ScriptConfigurationException",
    "ScriptExecutionException",
    "ScriptDataException",
    "ScriptComputationException",
    "ScriptIOException",
    
    # Container
    "ScriptContainer",
    
    # Logging
    "get_script_logger",
    "create_execution_context",
    "ProgressTracker",
]

__version__ = "1.0.0"
