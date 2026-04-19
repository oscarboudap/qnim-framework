# PRACTICAL IMPLEMENTATION GUIDE: QNIM Reproducible Execution

**Technical Reference for Immediate Deployment**

---

## 1. System Requirements and Setup

### 1.1 Hardware Prerequisites

```
CPU:          Intel i7+ or AMD Ryzen 5+
              (8 cores recommended for simulator)
              
RAM:          16 GB minimum
              24+ GB recommended
              
Storage:      100 GB SSD (for models + data)
              
Internet:     Required (cloud quantum access)
```

### 1.2 Software Stack

#### 1.2.1 OS Compatibility

```
✅ Linux (Ubuntu 20.04+)     — PRIMARY
✅ macOS (12.0+)             — SECONDARY
✅ Windows 10/11             — SUPPORTED (WSL2 recommended)
```

#### 1.2.2 Python Environment

```bash
# Recommended: Python 3.9+
python --version  # 3.9.7 or newer

# Create isolated virtual environment
python -m venv qnim_env
source qnim_env/bin/activate          # Linux/macOS
qnim_env\Scripts\activate             # Windows CMD
```

### 1.3 Dependency Installation

#### 1.3.1 Clone Repository

```bash
git clone https://github.com/your-org/qnim.git
cd qnim
```

#### 1.3.2 Install Requirements

```bash
pip install --upgrade pip wheel setuptools

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import qiskit; print(qiskit.__version__)"
python -c "import dwave; print(dwave.__version__)"
python -c "import numpy; print(numpy.__version__)"
```

#### 1.3.3 Expected Output

```
Latest qiskit version installed: 0.43.0 ✓
D-Wave Neal installed: 0.6.0 ✓
NumPy installed: 1.24.3 ✓
Ready for QNIM execution.
```

### 1.4 Credentials Configuration

#### 1.4.1 IBM Quantum Access

```bash
# Create .env file
cat > .env << 'EOF'
IBM_QUANTUM_TOKEN=YOUR_TOKEN_HERE
IBM_QUANTUM_CHANNEL=ibm_quantum
IBM_QUANTUM_PROVIDER=ibm-q

DWAVE_API_TOKEN=YOUR_TOKEN_HERE
DWAVE_SOLVER=DW_2000Q_6
EOF

# Protect credentials
chmod 600 .env
```

**How to obtain tokens:**
1. IBM Quantum: https://quantum.ibm.com (create free account, generate token)
2. D-Wave: https://www.dwavesys.com/leap/ (free tier available)

#### 1.4.2 Load Credentials in Python

```python
import os
from dotenv import load_dotenv

load_dotenv('.env')
ibm_token = os.getenv('IBM_QUANTUM_TOKEN')
dwave_token = os.getenv('DWAVE_API_TOKEN')

# Test connection
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService(channel='ibm_quantum', token=ibm_token)
print(f"Connected. Available backends: {service.backends()}")
```

---

## 2. Quick Start (5 Minutes)

### 2.1 The 5-Minute Demo

```bash
# Step 1: Navigate to scripts
cd scripts

# Step 2: Run simulator (no quantum hardware needed)
python run_qnim_simulator.py \
  --mode quick \
  --num_events 10 \
  --output_dir ../reports/quick_demo

# Expected output:
# ✓ Generated 10 synthetic GW events
# ✓ Processed 10 events in classical simulator
# ✓ Report saved: reports/quick_demo/summary_report_20240419.csv
```

### 2.2 Expected Output Format

```csv
event_id,m1_solar_mass,m2_solar_mass,snr,beyond_gr_probability,theory
GW_001,35.2,30.1,8.5,0.15,"GR"
GW_002,42.1,37.3,12.1,0.82,"UniversalTheory"
GW_003,15.4,14.8,6.2,0.28,"GR"

Total Events Analyzed: 10
Average Processing Time: 45 sec/event
Average Confidence: 0.91
```

### 2.3 Visualize Results

```bash
# Generate plots
python plot_results.py \
  --input reports/quick_demo/summary_report_20240419.csv \
  --output reports/quick_demo/figures

# Generated:
# - mass_distribution.png
# - confidence_histogram.pdf
# - parameter_corner_plot.pdf
```

---

## 3. Complete Pipeline: A-F Phases

### 3.1 PHASE A: Synthetic Data Generation

**Objective:** Create realistic gravitational wave signals with known ground truth

```bash
cd scripts/pipelines

# Command
python 01_generate_synthetic_gw.py \
  --count 1000 \
  --output_format hdf5 \
  --inject_anomalies false

# Configuration (modify in pipeline):
GENERATION_PARAMS = {
    'num_events': 1000,
    'mass_range_m1': (5, 80),        # Solar masses
    'mass_range_m2': (5, 80),
    'spin_range': (-0.99, 0.99),     # Dimensionless spin
    'snr_range': (8, 25),            # Signal-to-noise ratio
    'duration': 4.0,                 # seconds
    'sampling_rate': 4096.0,         # Hz (LIGO standard)
    'noise_model': 'O3_aLIGO_PSD'   # Real LIGO noise
}

# Output: data/synthetic/synthetic_gw_YYYYMMDD_HHMMSS.h5
# Size: ~500 MB for 1000 events
```

**Validation:**
```python
# Verify generated data
import h5py
with h5py.File('data/synthetic/synthetic_gw_*.h5', 'r') as f:
    print(f"Events: {len(f['strain_h1'])}")
    print(f"Sampling rate: {f.attrs['sampling_rate']} Hz")
    print(f"Duration: {f.attrs['duration']} sec")
```

### 3.2 PHASE B: Feature Extraction and Preprocessing

**Objective:** Convert time-domain signals to manageable feature vectors

```bash
python 02_preprocess_features.py \
  --input_file data/synthetic/synthetic_gw_*.h5 \
  --output_format numpy \
  --apply_pca true \
  --n_components 12

# Processing steps (automated):
# 1. Load strain timeseries from HDF5
# 2. Bandpass filter (35-250 Hz for LIGO)
# 3. Compute FFT (power spectral density)
# 4. Extract spectral moments (mean, variance, skewness, kurtosis)
# 5. Apply PCA: 512 dimensions → 12 principal components
# 6. Normalize (z-score: mean=0, std=1)
# 7. Save as .npy for fast loading

# Detailed mapping:
"""
Input: strain(t) [length=16384 samples, 4 sec at 4096 Hz]
  ↓
FFT → |X(f)| [frequency domain, 0-2048 Hz]
  ↓
Extract features:
  - Centroid frequency
  - Bandwidth
  - Spectral moments (1st, 2nd, 3rd, 4th)
  - Peak value
  - Energy (integral)
  - Skewness, Kurtosis
  - Entropy
  [Total: ~100 features]
  ↓
PCA (sklearn) → 12 components [95% variance retained]
  ↓
Standardization → μ=0, σ=1
  ↓
Output: X.npy [shape=(1000, 12)]
         y.npy [shape=(1000,), labels]
         pca_model.pkl [for inference]
"""

# Output:
# - X_train.npy (800 events, 12 features)
# - X_test.npy (200 events, 12 features)
# - pca_model.pkl (sklearn PCA transformer)
# - feature_stats.json (means, stds for later inference)
```

### 3.3 PHASE C: VQC Model Training

**Objective:** Optimize 12-qubit quantum circuit weights for beyond-GR classification

```bash
python 02_train_vqc_model.py \
  --training_data X_train.npy \
  --labels y_train.npy \
  --num_epochs 50 \
  --backend simulator \
  --loss_function binary_crossentropy

# Training procedure (COBYLA optimization):
"""
EPOCH 1/50: Initialize random circuit weights θ ∈ [0, 2π]^12
  ↓
Forward: ψ(x, θ) → expectation values → logits → probabilities
  ↓
Loss: L(θ) = -1/(800) Σ [y·log(p) + (1-y)·log(1-p)]
  ↓
COBYLA: Optimize θ to minimize L [gradient-free, robust]
  ↓
Validation on X_test every 10 epochs
  ↓
EPOCH 50: Final θ saved

Loss curve (typical):
Epoch 1:    Loss = 0.89
Epoch 10:   Loss = 0.45
Epoch 30:   Loss = 0.23
Epoch 50:   Loss = 0.18  ← Converged

Training time: ~2-4 hours (depending on backend)
"""

# Output:
# - qnim_vqc_weights.npy [shape=(42,) = 12 qubits × 3 params + 6 bias]
# - training_history.json [loss per epoch]
# - test_accuracy.txt ["91% on validation set"]
```

### 3.4 PHASE D: Exhaustive Validation

**Objective:** Cross-validate hyperparameters, verify reproducibility

```bash
python 03_validate_exhaustive.py \
  --model_weights qnim_vqc_weights.npy \
  --num_folds 5 \
  --test_anomalies true

# Validation procedure (5-fold cross-validation):
"""
Partition dataset into 5 folds:
  Fold 1: Train [4/5 data], Validate [1/5 data] → Accuracy₁
  Fold 2: Train [different 4/5], Validate [remaining 1/5] → Accuracy₂
  ...
  Fold 5: → Accuracy₅

Final Accuracy = (Acc₁ + Acc₂ + ... + Acc₅) / 5

Confidence interval: σ = √(Σ(Accᵢ - mean)² / 4)
"""

# Output:
validation_report.json:
{
    "average_accuracy": 0.91,
    "std_dev": 0.032,
    "confidence_interval_95%": [0.88, 0.94],
    "sensitivity": 0.87,
    "specificity": 0.94,
    "roc_auc": 0.92,
    "reproducibility_factor": 0.998
}
```

### 3.5 PHASE E: Inference on New Events

**Objective:** Classify novel gravitational wave events

```bash
python run_qnim_inference.py \
  --event_hdf5 data/raw/H-H1_LOSC_4_V2-1126259446-32.hdf5 \
  --model_weights models/qnim_vqc_weights.npy \
  --pca_model models/pca_model.pkl \
  --backend simulator

# Inference procedure (new event):
"""
INPUT: strain_h1(t) [real LIGO data, 4 seconds]
  ↓
Preprocessing:
  1. Load from HDF5
  2. Bandpass filter (35-250 Hz)
  3. FFT → spectral features
  4. PCA transform (using stored model)
  5. Normalize
  → Feature vector [12,]
  ↓
D-Wave Branch (parallel):
  6a. Map features → ising/QUBO
  6b. Submit to D-Wave cloud
  6c. Wait 2 minutes
  6d. Extract best solution
  6e. Unmap → physical parameters (m₁, m₂, χ_eff)
  ↓
IBM Branch (parallel):
  7a. Load trained VQC circuit
  7b. Feed 12 features
  7c. Execute 100 shots
  7d. Measure top qubit → binary classification
  7e. Consensus → P(beyond-GR)
  ↓
Aggregation:
  8. Combine D-Wave + IBM results
  9. Compute confidence margins
  10. Generate Planck reliability sheet
  ↓
OUTPUT: JSON result
"""

# Example output JSON:
{
    "event_id": "GW150914_reanalyzed",
    "timestamp_utc": "2015-09-14T09:50:45",
    "d_wave_parameters": {
        "m1_solar_masses": 35.2,
        "m2_solar_masses": 30.1,
        "effective_spin": 0.07,
        "confidence": 0.86
    },
    "ibm_classification": {
        "theory_beyond_gr": false,
        "p_beyond_gr": 0.18,
        "p_gr": 0.82,
        "significance_sigma": 0.9
    },
    "execution_time_seconds": 185,
    "backend_used": "simulator"
}
```

### 3.6 PHASE F: Report Generation

**Objective:** Produce publication-ready summary statistics

```bash
python generate_reports.py \
  --results_json reports/inference_results_*.json \
  --output_format csv,pdf,txt

# Generated files:
# 1. summary_statistics.csv
summary_statistics.csv:
"""
event_name,snr,m1,m2,chi,p_anomaly,theory_prediction
GW150914,24.5,35.2,30.1,0.07,0.18,GR
GW170814,18.3,30.5,25.3,0.03,0.31,GR
GW170121,13.4,42.1,37.3,0.12,0.89,Beyond-GR
...
Aggregate Statistics:
  Mean anomaly probability: 0.35
  Median SNR: 15.2
  Theory predictions (GR): 87%, (Beyond-GR): 13%
"""

# 2. performance_metrics.txt
"""
=== QNIM PERFORMANCE REPORT ===

Total Events Analyzed:        N=347
Successful Inferences:        346 (99.7%)
Failed Events:                1 (0.3%)

ACCURACY METRICS:
- Average parameter error (m1):    ±1.1%
- Average parameter error (m2):    ±0.9%
- Anomaly detection sensitivity:   87%
- False positive rate:             2.1%

COMPUTATIONAL EFFICIENCY:
- Time per event:             188 sec (3.1 min)
- Total compute time:         18.2 hours
- Average CPU utilization:    47%
- Peak memory usage:          8.3 GB

HYPOTHESIS TESTS:
- Events consistent with GR:     306/346 (88%)
- Marginal anomalies (1-3σ):     28/346 (8%)
- Strong anomalies (3σ+):        12/346 (3%)

RECOMMENDATIONS:
1. Follow up 12 strong anomaly events for LIGO publication
2. Joint analysis of 28 marginal events ongoing
3. No immediate evidence of Beyond-GR signatures
"""

# 3. Planck_Reliability_Report.csv (Bayesian evidence)
"""
Event,LogBE_GR,LogBE_SUGRA,LogBE_String,LogBE_LQG,Jeffreys_Scale
GW150914,1.2,0.4,-0.8,-1.2,"Inconclusive"
GW170814,2.1,-0.3,-1.5,-2.0,"Moderate vs Beyond"
GW170121,-0.5,1.8,1.2,0.3,"Moderate for Beyond"
...
"""
```

---

## 4. Troubleshooting Guide

### 4.1 Import Errors

#### Problem: `ModuleNotFoundError: No module named 'qiskit'`

**Solution:**
```bash
# Check pip cache
pip cache purge

# Reinstall with verbose output
pip install --upgrade --force-reinstall qiskit

# Verify
python -c "import qiskit; print(qiskit.__version__)"
```

#### Problem: `ImportError: cannot import name 'QiskitRuntimeService'`

**Solution:**
```bash
# Ensure qiskit_ibm_runtime is installed (not just qiskit)
pip install qiskit-ibm-runtime>=0.14.0

# Verify both are present
pip list | grep qiskit
```

### 4.2 IBM Quantum Connection Issues

#### Problem: `QiskitRuntimeInvalidInputsError: Invalid token`

**Solution:**
```bash
# Verify token in .env file
cat .env | grep IBM_QUANTUM_TOKEN

# Pre-test connection
python << 'EOF'
import os
from dotenv import load_dotenv
from qiskit_ibm_runtime import QiskitRuntimeService

load_dotenv('.env')
token = os.getenv('IBM_QUANTUM_TOKEN')

if not token:
    print("❌ Token not found in .env")
    exit(1)

try:
    service = QiskitRuntimeService(channel='ibm_quantum', token=token)
    backends = service.backends()
    print(f"✅ Connected. Available backends: {len(backends)}")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    exit(1)
EOF
```

#### Problem: `Backend kingston is not available`

**Solution:**
```bash
# Fetch available backends
python << 'EOF'
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService()
for backend in service.backends():
    print(f"- {backend.name}: {backend.num_qubits} qubits")
EOF

# Update config to use available backend
# In config/universe_params.yaml:
# ibm_backend: "ibmq_manhattan"  # or another available QPU
```

### 4.3 D-Wave Connection Issues

#### Problem: `SolverNotFoundError: Solver not found`

**Solution:**
```bash
# Verify D-Wave solver availability
python << 'EOF'
from dwave.cloud import Client

client = Client()
available_solvers = client.get_solvers()
print(f"Available solvers: {[s.name for s in available_solvers]}")

# For free tier:
# 'Advantage_system1' or 'Advantage2_prototype1' available
EOF

# Update config:
# In config/universe_params.yaml:
# dwave_solver: "Advantage_system1"
```

### 4.4 Memory Errors

#### Problem: `MemoryError: Unable to allocate X GB`

**Solution:**
```bash
# Reduce batch size in training
# In scripts/02_train_vqc_model.py:
BATCH_SIZE = 64  # Reduce from 256

# Or enable memory mapping for large datasets
import numpy as np
X_train = np.load('X_train.npy', mmap_mode='r')  # Read-only, no copy
```

#### Problem: `CUDA out of memory`

**Solution:**
```bash
# Use CPU instead of GPU
export CUDA_VISIBLE_DEVICES=""

python run_qnim_inference.py --backend cpu
```

---

## 5. Running on Real Hardware

### 5.1 IBM Quantum (Real Quantum Computer)

**Configuration:**
```python
# In src/infrastructure/ibm_quantum_adapter.py:

from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()
backend = service.backend("ibmq_manhattan")  # Real QPU

# Cost estimation
pricing = {
    'per_job': 0.005,      # $0.005 per job submission
    'per_transaction_set': 0.035,  # Additional $0.035 per 5 minutes
}

# For 100 events at 3 min each = 300 minutes = 60 transaction sets
cost = 100 * 0.005 + 60 * 0.035 = $2.60
```

**Execution:**
```bash
python run_qnim_inference.py \
  --backend real_ibm \
  --num_shots 1024 \
  --optimization_level 3

# Output:
# ✓ Job submitted to IBM Quantum
# ✓ Job ID: 123e4567-e89b-12d3-a456-426614174000
# ⏳ Estimated completion: 30 minutes
```

### 5.2 D-Wave Advantage (Real Quantum Annealer)

**Configuration:**
```python
from dwave.system import DWaveSampler, EmbeddingComposite

sampler = EmbeddingComposite(
    DWaveSampler(endpoint='https://cloud.dwavesys.com/sapi')
)

# Free tier provides 1000 problem submissions/month
# Each submission: cost = 1 DQM unit = free
```

**Execution:**
```bash
python run_qnim_inference.py \
  --backend real_dwave \
  --num_reads 5000 \
  --annealing_time 2000

# Output:
# ✓ Problem embedded on D-Wave Advantage
# ✓ Execution time: 1.2 seconds
# ✓ Solution energy: -524.3
```

---

## 6. Performance Optimization

### 6.1 Profiling Code

```bash
# Run with profiling
python -m cProfile -s cumulative run_qnim_inference.py > profile.txt

# Analyze top functions
head -20 profile.txt
"""
Output example:
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
   10      4.2      0.42     18.5     1.85     ibm_quantum_adapter.py:47(execute_circuit)
   10      2.1      0.21     2.1      0.21     {dwave_solver}
   50      0.8      0.016    0.8      0.016    {h5py.get_dataset}
   ...

Most time spent in IBM backend (81% of runtime).
"""
```

### 6.2 Parallelization

```bash
# Process multiple events in parallel
python -m scripts/run_qnim_inference.py \
  --parallel true \
  --num_workers 4 \
  --input_files event_1.h5 event_2.h5 event_3.h5 event_4.h5

# Speed improvement: ~3.8x (4 workers)
# Total time: 24 events × 185s ÷ 4 = ~1.1 hours
```

### 6.3 Caching

```python
# Enable result caching
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_pca_features(strain_hash: str):
    # Hash remains same for identical strain → cached result
    return pca_transform(...)

# Improvement: ~2x speedup for repeated analyses
```

### 6.4 GPU Acceleration

```bash
# Enable GPU for PCA and preprocessing
export CUPY_CUDA_PER_THREAD_DEFAULT_STREAM=1

python run_qnim_inference.py \
  --use_gpu true \
  --gpu_device 0

# Expected speedup: 10-15x for FFT/PCA stages
```

---

## 7. Reproducibility Audit

### 7.1 Checklist

```
□ Python version: 3.9+
□ All packages from requirements.txt installed
□ Credentials configured (.env file)
□ Random seed fixed:
  
  import numpy as np
  import random
  np.random.seed(42)
  random.seed(42)
  
□ Data checksums verified:
  
  md5sum data/raw/*.h5
  # Compare with documented hashes
  
□ Model weights version documented:
  
  ls -la models/qnim_vqc_weights*.npy
  # Latest weights from version control
```

### 7.2 Automated Audit Script

```bash
python scripts/validate_reproducibility.py \
  --reference_results reports/reference_run_*.json \
  --new_results reports/current_run_*.json

# Output:
"""
=== REPRODUCIBILITY AUDIT ===

✓ Python versions match: 3.9.7
✓ Package versions match: numpy 1.24.3, qiskit 0.43.0
✓ Random seeds identical: seed=42
✓ Model checksums identical: md5=3f4e...
✓ Data integrity: All files valid
✓ Results correlation:
    - m1 parameter:     r=0.998 (excellent)
    - m2 parameter:     r=0.997 (excellent)
    - anomaly prob:     r=0.994 (excellent)
    
Overall Reproducibility Score: 99.6% ✓

CONCLUSION: Execution is fully reproducible.
            Results differ by <0.4% from reference.
"""
```

---

## 8. Continuous Integration / Automated Testing

### 8.1 Local Testing Before Commit

```bash
# Run test suite
pytest test/ -v

# Expected:
"""
test_domain_tests_kerr_vacuum_provider.py::test_schwarzschild_metric PASSED
test_application_tests_hybrid_orchestrator.py::test_dwave_branch PASSED
test_infrastructure_tests_ibm_adapter.py::test_circuit_generation PASSED

======================== 42 passed in 3.21s ========================
"""
```

### 8.2 GitHub Actions Configuration

```yaml
# .github/workflows/test.yml
name: QNIM Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest test/ -v
      - name: Check code style
        run: flake8 src/ --max-line-length=100
```

---

**END OF PRACTICAL IMPLEMENTATION GUIDE**

*This document enables rapid prototyping, deployment, troubleshooting, and reproducible execution of QNIM across diverse computational environments.*
