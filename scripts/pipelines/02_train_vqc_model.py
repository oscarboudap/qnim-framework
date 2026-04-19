#!/usr/bin/env python3
"""
Block 2: Train Quantum VQC Model

Trains a Variational Quantum Classifier on synthetic dataset:
  - Loads preprocessed GW strain data
  - Feature engineering: PCA dimensionality reduction
  - Quantum feature mapping: ZZFeatureMap + RealAmplitudes
  - Optimization: SPSA (Simultaneous Perturbation Stochastic Approximation)
  - Saves trained weights for inference
  
Produces:
  - Trained VQC weights
  - Training curves (loss, accuracy)
  - Validation metrics
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import numpy as np
from typing import Dict, Any, Optional
import joblib

# CLI Framework
from src.cli import (
    ScriptConfig,
    ScriptContainer,
    get_script_logger,
    create_execution_context,
    ScriptException,
    ScriptExecutionException,
)

# Infrastructure & Application
from src.infrastructure.qiskit_vqc_trainer import QiskitVQCTrainer
from src.infrastructure.sklearn_preprocessing_adapter import SklearnPreprocessor
from src.infrastructure.storage.quantum_dataloader import QuantumDatasetLoader


def main(synthetic_data_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Train Quantum VQC model on synthetic GW dataset.
    
    Args:
        synthetic_data_path: Path to HDF5 file from Block 1
    
    Returns:
        Dict with:
            - "weights_path": Path to saved weights
            - "training_accuracy": Final training accuracy
            - "validation_accuracy": Validation accuracy
            - "learning_curve": History of metrics
            - "model": Trained model object (optional)
    """
    logger = get_script_logger("block_2_training")
    context = create_execution_context("02_train_vqc_model.py")
    
    try:
        # ====================================================================
        # INITIALIZATION
        # ====================================================================
        logger.step("INIT", "Initializing Block 2: VQC Training")
        
        config = ScriptConfig.from_env()
        container = ScriptContainer(config)
        
        logger.configuration(
            "VQCConfig",
            num_qubits=config.training.vqc.num_qubits,
            num_features=config.training.vqc.num_features,
            optimizer=config.training.vqc.optimizer_name,
            max_iterations=config.training.vqc.max_iterations,
            learning_rate=config.training.vqc.learning_rate,
        )
        
        # ====================================================================
        # DATA LOADING & PREPROCESSING
        # ====================================================================
        logger.step("DATA", "Loading and preprocessing synthetic data")
        
        data_loader = container.get_data_loader()
        
        # Load data (from Block 1 output or default location)
        if synthetic_data_path:
            logger.info(f"Loading from Block 1 output: {synthetic_data_path}")
            # TODO: Implement custom loader for HDF5 from Block 1
            X, y, scaler = data_loader.load_and_preprocess(
                n_components=config.training.vqc.num_features
            )
        else:
            logger.info("Loading from default synthetic data location")
            X, y, scaler = data_loader.load_and_preprocess(
                n_components=config.training.vqc.num_features
            )
        
        # Save preprocessor for inference later
        preprocessor_path = config.training.paths.get_preprocessing_pipeline_path()
        joblib.dump(scaler, preprocessor_path)
        logger.info(f"Preprocessor saved: {preprocessor_path}")
        
        # Log data statistics
        logger.metric("num_samples", X.shape[0], unit="samples")
        logger.metric("feature_dimension", X.shape[1], unit="dimensions")
        logger.metric("num_classes", len(np.unique(y)), unit="classes")
        
        # ====================================================================
        # MODEL TRAINING
        # ====================================================================
        logger.step("TRAINING", "Training Quantum VQC")
        
        trainer = QiskitVQCTrainer()
        
        training_result = trainer.train_vqc(
            X_train=X,
            y_train=y,
            num_qubits=config.training.vqc.num_qubits,
            max_iterations=config.training.vqc.max_iterations,
            optimizer_name=config.training.vqc.optimizer_name,
        )
        
        # Extract results
        weights = training_result.get("weights", np.array([]))
        training_accuracy = training_result.get("training_accuracy", 0.0)
        validation_accuracy = training_result.get("validation_accuracy", 0.0)
        final_loss = training_result.get("final_loss", np.inf)
        iterations_completed = training_result.get("iterations", 0)
        
        # ====================================================================
        # SAVE WEIGHTS
        # ====================================================================
        logger.step("STORAGE", "Saving trained weights")
        
        weights_path = config.training.paths.get_vqc_weights_path()
        np.save(weights_path, weights)
        
        logger.metric("training_accuracy", training_accuracy, unit="ratio")
        logger.metric("validation_accuracy", validation_accuracy, unit="ratio")
        logger.metric("final_loss", final_loss, unit="cross_entropy")
        logger.metric("iterations_completed", iterations_completed, unit="iterations")
        
        logger.info(f"✅ Weights saved: {weights_path}")
        
        # ====================================================================
        # REPORTING
        # ====================================================================
        logger.step("REPORT", "Block 2 Training Complete")
        logger.info(f"✅ Training Accuracy: {training_accuracy:.4f}")
        logger.info(f"✅ Validation Accuracy: {validation_accuracy:.4f}")
        
        return {
            "weights_path": str(weights_path),
            "training_accuracy": float(training_accuracy),
            "validation_accuracy": float(validation_accuracy),
            "final_loss": float(final_loss),
            "iterations": int(iterations_completed),
            "learning_curve": training_result.get("history", {}),
        }
    
    except ScriptExecutionException as e:
        logger.error(f"❌ Block 2 failed: {e}", extra=context)
        raise SystemExit(1)
    
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", extra=context, exc_info=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
