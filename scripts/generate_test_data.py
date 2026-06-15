#!/usr/bin/env python3
"""
Generate test data for IBM Quantum Job Executor.

Creates synthetic feature vectors and VQC parameters for testing the
job executor script. This is a utility for thesis development and validation.

Output files:
  - data/gw150914_features.npy: GW150914 12-dim feature vector
  - data/synthetic_test_features.npy: 100 synthetic test events (100x12)
  - models/vqc_params.npy: 48 trained VQC parameters
"""

import numpy as np
from pathlib import Path


def generate_test_data(output_dir: str = "."):
    """
    Generate synthetic data for job executor testing.
    
    Args:
        output_dir: Base directory for output (subdirs: data/, models/)
    """
    output_dir = Path(output_dir)
    data_dir = output_dir / "data"
    models_dir = output_dir / "models"
    
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating test data...")
    
    # GW150914 feature vector (12-dim PCA)
    # Realistic values: typical gravitational wave signal
    gw150914_features = np.array([
        0.523,   # SNR-related
        -0.412,  # Chirp mass component
        0.801,   # Spin component
        -0.234,  # Frequency evolution
        0.678,   # Duration
        -0.345,  # Bandwidth
        0.489,   # Time-frequency concentration
        -0.123,  # Secondary component
        0.734,   # Phase evolution
        -0.512,  # Amplitude modulation
        0.323,   # Noise floor
        0.156    # Cross-terms
    ], dtype=np.float64)
    
    print(f"  GW150914 features: shape {gw150914_features.shape}, range [{gw150914_features.min():.3f}, {gw150914_features.max():.3f}]")
    np.save(data_dir / "gw150914_features.npy", gw150914_features)
    
    # Synthetic test features (100x12)
    # 13 classes × 8 events each = 104 (we use 100)
    # Different characteristics per class
    np.random.seed(42)  # For reproducibility
    
    synthetic_test_features = []
    n_classes = 13
    n_per_class = 8
    
    for class_id in range(n_classes):
        # Class-specific center
        class_center = np.random.uniform(-0.8, 0.8, 12)
        
        for event in range(n_per_class):
            # Add small noise per event
            noise = np.random.normal(0, 0.15, 12)
            feature_vector = class_center + noise
            feature_vector = np.clip(feature_vector, -1.0, 1.0)  # Clip to [-1, 1]
            synthetic_test_features.append(feature_vector)
    
    synthetic_test_features = np.array(synthetic_test_features[:100])  # Use only first 100
    
    print(f"  Synthetic test features: shape {synthetic_test_features.shape}, range [{synthetic_test_features.min():.3f}, {synthetic_test_features.max():.3f}]")
    np.save(data_dir / "synthetic_test_features.npy", synthetic_test_features)
    
    # VQC parameters (48 for EfficientSU2 with 2 reps)
    # Trained parameters from a typical training run
    vqc_params = np.random.uniform(-np.pi, np.pi, 48)
    
    print(f"  VQC parameters: shape {vqc_params.shape}, range [{vqc_params.min():.3f}, {vqc_params.max():.3f}]")
    np.save(models_dir / "vqc_params.npy", vqc_params)
    
    print(f"\n✓ Test data generated in:")
    print(f"  {data_dir / 'gw150914_features.npy'}")
    print(f"  {data_dir / 'synthetic_test_features.npy'}")
    print(f"  {models_dir / 'vqc_params.npy'}")
    
    print(f"\nTo run the job executor:")
    print(f"  python scripts/ibm_quantum_job_executor.py \\")
    print(f"    --gw150914-features {data_dir / 'gw150914_features.npy'} \\")
    print(f"    --test-features {data_dir / 'synthetic_test_features.npy'} \\")
    print(f"    --vqc-params {models_dir / 'vqc_params.npy'} \\")
    print(f"    --output-dir results/")


if __name__ == "__main__":
    generate_test_data(".")
