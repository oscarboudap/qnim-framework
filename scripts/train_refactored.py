"""
Training Script - Refactored with DDD/Clean Architecture

This script demonstrates the correct pattern for entry points:
  1. Load centralized configuration (ScriptConfig)
  2. Create DI container with configuration
  3. Get use case from container (all deps injected)
  4. Execute use case with typed inputs
  5. Handle errors gracefully with structured logging

Old Pattern (❌ WRONG):
  - Direct infrastructure instantiation
  - Hardcoded magic numbers
  - No error handling
  - No validation
  - Skipped application layer

New Pattern (✅ CORRECT):
  - Centralized configuration in ScriptConfig VO
  - Dependency injection via ScriptContainer
  - Application layer use case invocation
  - Full type hints and validation
  - Structured error handling and logging
"""

from typing import Optional
import os
import sys
from pathlib import Path

# Add parent directory to path so imports work from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

# Scripts layer (new)
from src.scripts import (
    ScriptConfig,
    ScriptContainer,
    get_script_logger,
    create_execution_context,
    ScriptException,
    ScriptConfigurationException,
    ScriptExecutionException,
)


def main() -> None:
    """
    Main entry point for training script.
    
    Flow:
      1. Initialize logger with context
      2. Load configuration from environment
      3. Create dependency injection container
      4. Retrieve training use case
      5. Execute training
      6. Log completion or errors
    
    Raises:
      SystemExit: On any error (non-zero exit code)
    """
    # Initialize logger
    logger = get_script_logger("train")
    context = create_execution_context("train.py")
    
    try:
        # ====================================================================
        # STEP 1: Load centralized configuration
        # ====================================================================
        logger.info("Loading script configuration", extra=context)
        config = ScriptConfig.from_env()
        
        # Log configuration for audit trail
        logger.configuration(
            "VQC",
            num_qubits=config.training.vqc.num_qubits,
            num_features=config.training.vqc.num_features,
            optimizer=config.training.vqc.optimizer_name,
            max_iterations=config.training.vqc.max_iterations,
        )
        logger.configuration(
            "Preprocessing",
            target_samples=config.training.preprocessing.target_samples,
            pca_components=config.training.preprocessing.pca_components,
        )
        logger.configuration(
            "Paths",
            models_dir=config.training.paths.models_dir,
            data_dir=config.training.paths.data_dir,
        )
        
        # ====================================================================
        # STEP 2: Create dependency injection container
        # ====================================================================
        logger.step("INIT", "Creating dependency injection container")
        container = ScriptContainer(config)
        
        # ====================================================================
        # STEP 3: Create required directories
        # ====================================================================
        logger.step("IO", "Ensuring required directories exist")
        os.makedirs(config.training.paths.models_dir, exist_ok=True)
        
        # ====================================================================
        # STEP 4: Retrieve training use case from container
        # ====================================================================
        logger.step("DI", "Retrieving training use case with injected dependencies")
        use_case = container.get_training_use_case()
        
        # ====================================================================
        # STEP 5: Execute training
        # ====================================================================
        logger.step(
            "TRAINING",
            "Starting VQC training",
            qubits=config.training.vqc.num_qubits,
            iterations=config.training.vqc.max_iterations,
            batch_size=config.training.batch_size,
        )
        
        # Create training input DTO (typed)
        from src.application.dto import TrainingInputDTO
        
        training_input = TrainingInputDTO(
            num_qubits=config.training.vqc.num_qubits,
            num_features=config.training.vqc.num_features,
            max_iterations=config.training.vqc.max_iterations,
            batch_size=config.training.batch_size,
            learning_rate=config.training.vqc.learning_rate,
            optimizer_name=config.training.vqc.optimizer_name,
        )
        
        # Execute training use case
        training_result = use_case.execute(training_input)
        
        # Log metrics
        logger.metric("final_accuracy", training_result.accuracy, unit="ratio")
        logger.metric(
            "final_loss",
            training_result.loss,
            unit="cross_entropy",
            iteration=training_result.num_iterations,
        )
        
        # ====================================================================
        # STEP 6: Save weights and pipeline
        # ====================================================================
        logger.step("STORAGE", "Saving trained model weights and pipeline")
        
        weights_path = str(config.training.paths.get_vqc_weights_path())
        try:
            import numpy as np
            np.save(weights_path, training_result.weights)
            logger.info(f"Weights saved successfully: {weights_path}")
        except Exception as e:
            raise ScriptExecutionException(
                "Failed to save weights",
                context={"path": weights_path, "error": str(e)}
            )
        
        # ====================================================================
        # STEP 7: Summary and completion
        # ====================================================================
        logger.step(
            "COMPLETE",
            "Training completed successfully",
            total_iterations=training_result.num_iterations,
            final_accuracy=f"{training_result.accuracy:.4f}",
            weights_saved=weights_path,
        )
        
        logger.info("✅ Training script completed successfully")
        return None
    
    except ScriptConfigurationException as e:
        """Handle configuration errors."""
        logger.error(
            f"❌ Configuration error: {e}",
            extra={**context, "error_type": "ScriptConfigurationException"}
        )
        raise SystemExit(1)
    
    except ScriptExecutionException as e:
        """Handle execution errors."""
        logger.error(
            f"❌ Execution error: {e}",
            extra={**context, "error_type": "ScriptExecutionException"}
        )
        raise SystemExit(1)
    
    except ScriptException as e:
        """Handle generic script errors."""
        logger.error(
            f"❌ Script error: {e}",
            extra={**context, "error_type": type(e).__name__}
        )
        raise SystemExit(1)
    
    except Exception as e:
        """Handle unexpected errors."""
        logger.error(
            f"❌ Unexpected error: {e}",
            extra={**context, "error_type": type(e).__name__},
            exc_info=True  # Include stack trace
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
