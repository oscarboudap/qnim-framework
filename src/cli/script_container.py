"""
Script Layer Dependency Injection Container

Manages instantiation and lifecycle of all script dependencies,
implementing Inversion of Control pattern.

Architecture:
  - Single point of dependency creation (factory pattern)
  - Validates all dependencies available before use
  - Enables testing with mock implementations
  - Centralizes error handling for missing dependencies

Pattern:
  container = ScriptContainer(config)
  use_case = container.get_training_use_case()
  use_case.execute(params)
"""

from typing import Optional
import logging
import os

from .script_config import ScriptConfig, TrainingConfig
from .script_exceptions import ScriptConfigurationException


class ScriptContainer:
    """
    Dependency Injection container for script layer.
    
    Responsible for:
      - Creating infrastructure adapters
      - Injecting dependencies into use cases
      - Managing configuration
      - Validating all dependencies available
    """
    
    def __init__(self, config: ScriptConfig) -> None:
        """
        Initialize DI container with configuration.
        
        Args:
          config: ScriptConfig with all parameters
        
        Raises:
          ScriptConfigurationException: If configuration invalid
        """
        self._config = config
        self._logger = logging.getLogger(self.__class__.__name__)
        self._validate_config()
        self._logger.info("ScriptContainer initialized successfully")
    
    def _validate_config(self) -> None:
        """
        Validate that configuration is safe.
        
        Raises:
          ScriptConfigurationException: If any issue found
        """
        # Check all required paths exist and are writable
        required_dirs = [
            str(self._config.training.paths.get_data_synthetic_path()),
            str(self._config.training.paths.get_data_raw_path()),
            self._config.training.paths.models_dir,
            self._config.training.paths.reports_dir,
        ]
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                self._logger.warning(f"Creating missing directory: {dir_path}")
                try:
                    os.makedirs(dir_path, exist_ok=True)
                except OSError as e:
                    raise ScriptConfigurationException(
                        f"Cannot create required directory",
                        context={"path": dir_path, "error": str(e)}
                    )
    
    def get_training_use_case(self):
        """
        Get the training use case with all dependencies injected.
        
        Returns:
          ModelTrainingUseCase: Ready for execution
        
        Raises:
          ScriptConfigurationException: If any dependency unavailable
        """
        from src.application import ModelTrainingUseCase
        from src.infrastructure.qiskit_vqc_trainer import QiskitVQCTrainer
        from src.infrastructure.sklearn_preprocessing_adapter import SklearnPreprocessor
        
        try:
            # Create infrastructure adapters
            quantum_trainer = QiskitVQCTrainer(
                num_qubits=self._config.training.vqc.num_qubits,
                num_features=self._config.training.vqc.num_features,
                ansatz_reps=self._config.training.vqc.ansatz_reps,
                max_iterations=self._config.training.vqc.max_iterations,
            )
            
            preprocessor = SklearnPreprocessor(
                n_components=self._config.training.preprocessing.pca_components,
                target_samples=self._config.training.preprocessing.target_samples,
            )
            
            # Create use case with injected dependencies
            use_case = ModelTrainingUseCase(
                quantum_trainer_port=quantum_trainer,
                preprocessing_port=preprocessor,
            )
            
            self._logger.info("Training use case created with dependencies injected")
            return use_case
        
        except Exception as e:
            raise ScriptConfigurationException(
                f"Failed to create training use case",
                context={"error": str(e), "type": type(e).__name__}
            )
    
    def get_inference_use_case(self):
        """
        Get the inference use case with all dependencies injected.
        
        Returns:
          DecodeGravitationalWaveUseCase: Ready for execution
        """
        from src.application import DecodeGravitationalWaveUseCase
        from src.application.hybrid_orchestrator import HybridInferenceOrchestrator
        from src.infrastructure.ibm_quantum_adapter import IBMQuantumAdapter
        from src.infrastructure.neal_annealer_adapter import NealSimulatedAnnealerAdapter
        from src.infrastructure.sklearn_preprocessing_adapter import SklearnPreprocessor
        
        try:
            # Create quantum infrastructure
            ibm_adapter = IBMQuantumAdapter(
                weights_path=str(self._config.training.paths.get_vqc_weights_path()),
            )
            dwave_adapter = NealSimulatedAnnealerAdapter()
            
            # Create orchestrator
            orchestrator = HybridInferenceOrchestrator(
                quantum_computer=ibm_adapter,
                quantum_optimizer=dwave_adapter,
            )
            
            # Create preprocessing
            preprocessor = SklearnPreprocessor(
                n_components=self._config.training.preprocessing.pca_components,
                target_samples=self._config.training.preprocessing.target_samples,
            )
            
            # Create use case
            use_case = DecodeGravitationalWaveUseCase(
                orchestrator=orchestrator,
                preprocessor=preprocessor,
            )
            
            self._logger.info("Inference use case created with dependencies injected")
            return use_case
        
        except Exception as e:
            raise ScriptConfigurationException(
                f"Failed to create inference use case",
                context={"error": str(e), "type": type(e).__name__}
            )
    
    def get_data_loader(self):
        """
        Get the data loader adapter.
        
        Returns:
          QuantumDatasetLoader: Configured for target sample size
        """
        from src.infrastructure.storage.quantum_dataloader import QuantumDatasetLoader
        
        try:
            loader = QuantumDatasetLoader(
                target_samples=self._config.training.preprocessing.target_samples
            )
            self._logger.info(f"Data loader created: {self._config.training.preprocessing.target_samples} samples")
            return loader
        except Exception as e:
            raise ScriptConfigurationException(
                f"Failed to create data loader",
                context={"error": str(e)}
            )
    
    def get_metrics_reporter(self):
        """
        Get the metrics reporting adapter.
        
        Returns:
          MatplotlibMetricsReporter: Configured with output directory
        """
        from src.infrastructure.matplotlib_metrics_reporter import MatplotlibMetricsReporter
        
        try:
            reporter = MatplotlibMetricsReporter(
                output_dir=self._config.training.paths.reports_dir
            )
            self._logger.info(f"Metrics reporter created: {self._config.training.paths.reports_dir}")
            return reporter
        except Exception as e:
            raise ScriptConfigurationException(
                f"Failed to create metrics reporter",
                context={"error": str(e)}
            )
    
    @property
    def config(self) -> ScriptConfig:
        """Get the configuration object."""
        return self._config
