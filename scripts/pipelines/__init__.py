"""
Main Processing Pipelines for QNIM

This package contains the three main pipeline blocks of the TFM:

1. **Block 1**: Generate Synthetic Gravitational Waves (01_generate_synthetic_gw.py)
   - Solves all physics equations (Post-Newtonian + EOB)
   - Injects anomalies from quantum gravity layers (4, 5, 6, 7)
   - Produces ground truth datasets with known theory signatures

2. **Block 2**: Train Quantum VQC Model (02_train_vqc_model.py)
   - Trains Variational Quantum Classifier on synthetic data
   - Uses Qiskit infrastructure with SPSA optimizer
   - Saves trained weights for inference

3. **Block 3**: Exhaustive Statistical Validation (03_validate_exhaustive.py)
   - Runs Monte Carlo sweeps at multiple SNR levels
   - Performs significance tests (χ², Bayesian model comparison)
   - Compares against classical MCMC benchmarks
   - Produces 5σ confidence intervals

**Orchestrator**: 00_full_pipeline.py
   - Sequences Blocks 1→2→3
   - Manages dependencies and data flow
   - Unified reporting

Usage:
    # Run full pipeline
    python scripts/pipelines/00_full_pipeline.py
    
    # Run individual blocks
    python scripts/pipelines/01_generate_synthetic_gw.py
    python scripts/pipelines/02_train_vqc_model.py
    python scripts/pipelines/03_validate_exhaustive.py
"""

__all__ = [
    "full_pipeline",
    "generate_synthetic_gw",
    "train_vqc_model",
    "validate_exhaustive",
]
