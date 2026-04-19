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
import h5py
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
        
        # Find Block 1 output file
        if synthetic_data_path:
            data_file = Path(synthetic_data_path)
            logger.info(f"Loading from Block 1 output: {data_file}")
        else:
            # Find most recent synthetic_gw_*.h5 file
            synthetic_dir = Path(config.training.paths.get_data_synthetic_path())
            synthetic_files = sorted(synthetic_dir.glob("synthetic_gw_*.h5"), reverse=True)
            if not synthetic_files:
                raise ScriptExecutionException(
                    "No synthetic data files found",
                    context={"directory": str(synthetic_dir)}
                )
            data_file = synthetic_files[0]
            logger.info(f"Using latest Block 1 output: {data_file}")
        
        # Load strain data and labels from Block 1 HDF5
        try:
            logger.debug(f"  → Reading HDF5: {data_file}")
            with h5py.File(data_file, 'r') as h5f:
                strain_plus = h5f['strain_plus'][:]  # Shape: (n_events, n_samples)
                theory_labels = h5f['theory_labels'][:]  # Shape: (n_events,)
                
                logger.debug(f"    Loaded strain_plus: shape={strain_plus.shape}")
                logger.debug(f"    Loaded theory_labels: shape={theory_labels.shape}")
        except Exception as e:
            raise ScriptExecutionException(
                f"Failed to load Block 1 data",
                context={"file": str(data_file), "error": str(e)}
            )
        
        # Encode theory labels as integers
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(theory_labels)
        
        logger.metric("num_samples", strain_plus.shape[0], unit="samples")
        logger.metric("sample_length", strain_plus.shape[1], unit="samples")
        logger.metric("num_classes", len(np.unique(y)), unit="classes")
        for theory, count in zip(label_encoder.classes_, np.bincount(y)):
            logger.metric(f"class_{theory}", count, unit="events")
        
        # Feature preprocessing: PCA
        try:
            logger.debug(f"  → Applying PCA ({config.training.vqc.num_features} features)...")
            from sklearn.decomposition import PCA
            
            pca = PCA(n_components=config.training.vqc.num_features)
            X = pca.fit_transform(strain_plus)
            
            logger.debug(f"    PCA shape: {X.shape}")
            logger.metric("pca_explained_variance", float(np.sum(pca.explained_variance_ratio_)), unit="ratio")
        except Exception as e:
            raise ScriptExecutionException(
                f"PCA preprocessing failed",
                context={"n_components": config.training.vqc.num_features, "error": str(e)}
            )
        
        # Train/test split
        try:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            logger.metric("train_samples", X_train.shape[0], unit="samples")
            logger.metric("test_samples", X_test.shape[0], unit="samples")
        except Exception as e:
            raise ScriptExecutionException(
                f"Train/test split failed",
                context={"error": str(e)}
            )
        
        # Save preprocessor for inference
        try:
            import joblib
            preprocessor_path = config.training.paths.get_preprocessing_pipeline_path()
            joblib.dump((pca, label_encoder), preprocessor_path)
            logger.info(f"✓ Preprocessor saved: {preprocessor_path}")
        except Exception as e:
            logger.warning(f"Could not save preprocessor: {e}")
        
        # ====================================================================
        # MODEL TRAINING
        # ====================================================================
        logger.step("TRAINING", "Training Quantum VQC")
        
        trainer = QiskitVQCTrainer()
        
        training_result = trainer.train_vqc(
            X_train=X_train,
            y_train=y_train,
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
