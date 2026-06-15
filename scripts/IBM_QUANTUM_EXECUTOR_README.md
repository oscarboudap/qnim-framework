# IBM Quantum Job Executor for QNIM Master's Thesis

## Overview

This script executes three sequential IBM Quantum jobs with complete metadata capture for reproducibility and verifiability in astrophysics applications. Designed for master's thesis research on Quantum Neuro-Inspired Manifolds (QNIM) applied to gravitational wave signal classification.

### Key Features

✅ **Full Metadata Capture**
- Job IDs and timestamps (ISO 8601)
- Backend properties (T1, T2, 2-qubit gate errors)
- VQC architecture details (88 CNOT gates, 48 parameters)
- Input features and raw expectation values
- Class probability distributions and predictions

✅ **Three-Job Sequential Execution**
- Job 1: GW150914 single event with Zero-Noise Extrapolation (resilience_level=2)
- Job 2: 100 synthetic events with ZNE (resilience_level=2)
- Job 3: 100 synthetic events without ZNE (resilience_level=0, baseline)

✅ **Robust Error Handling**
- Try/except blocks around all job submissions and result retrievals
- Partial results saved even if one job fails
- Detailed error logging with job recovery info

✅ **Comprehensive Metrics**
- Balanced accuracy (for imbalanced multi-class scenarios)
- Per-class precision, recall, F1 scores
- Comparison: ZNE vs non-ZNE performance

✅ **Thesis-Ready Output**
- JSON files per job (named after job_id)
- Summary JSON with all job IDs for citation
- Human-readable appendix format for direct thesis inclusion

---

## Architecture

### VQC Circuit (12 Qubits, 88 CNOTs)

```
Input: 12-dimensional PCA feature vector
       ↓
   ┌─────────────────────────┐
   │   ZZFeatureMap          │
   │   - 2 repetitions       │
   │   - Cyclic entanglement │
   │   - Features: x₀...x₁₁  │
   └─────────────────────────┘
       ↓
   ┌─────────────────────────┐
   │  EfficientSU2 Ansatz    │
   │  - 2 repetitions        │
   │  - Linear entanglement  │
   │  - 48 parameters: θ₀...θ₄₇
   │  - 88 CNOT gates total  │
   └─────────────────────────┘
       ↓
    Measure Z₀
       ↓
   Output: E[Z] ∈ [-1, 1]
```

### Job Sequence

**Job 1: GW150914 with ZNE**
- Input: 1 event (GW150914 signal)
- Backend: ibm_fez (156 qubits)
- Resilience Level: 2 (ZNE enabled)
- Noise Amplification: [1, 2, 3] via gate folding
- Shots: 8192
- Output: 1 prediction + metadata

**Job 2: 100 Synthetic Events with ZNE**
- Input: 100 synthetic test events (13 classes × 8 events)
- Backend: ibm_fez
- Resilience Level: 2 (ZNE enabled)
- Same noise amplification
- Shots: 8192
- Output: 100 predictions + balanced accuracy

**Job 3: 100 Synthetic Events without ZNE (Baseline)**
- Input: Same 100 synthetic events as Job 2
- Backend: ibm_fez
- Resilience Level: 0 (ZNE disabled)
- Shots: 8192
- Output: 100 predictions + balanced accuracy

---

## Prerequisites

### Python Packages
```bash
pip install qiskit-ibm-runtime qiskit numpy scipy scikit-learn tqdm
```

### IBM Quantum Access
- Valid IBM Quantum account with access to ibm_fez backend
- IBM_QUANTUM_TOKEN environment variable set or pass via `--token`

### Input Data
Three numpy files required:

1. **GW150914 Features** (`gw150914_features.npy`)
   - Shape: (12,)
   - Format: 12-dimensional PCA feature vector
   - Range: typically [-1, 1]

2. **Synthetic Test Features** (`synthetic_test_features.npy`)
   - Shape: (100, 12)
   - Format: 100 events × 12 PCA dimensions
   - Classes: 13 theory classes × 8 events each
   - Range: [-1, 1]

3. **VQC Parameters** (`vqc_params.npy`)
   - Shape: (48,)
   - Format: Trained EfficientSU2 ansatz parameters
   - Range: typically [-π, π]

---

## Usage

### Quick Start with Test Data

```bash
# Generate synthetic test data
python scripts/generate_test_data.py

# Run full execution pipeline
python scripts/ibm_quantum_job_executor.py \
    --gw150914-features data/gw150914_features.npy \
    --test-features data/synthetic_test_features.npy \
    --vqc-params models/vqc_params.npy \
    --output-dir results/
```

### Using Real Pre-computed Features

```bash
python scripts/ibm_quantum_job_executor.py \
    --gw150914-features path/to/gw150914_extracted.npy \
    --test-features path/to/test_events.npy \
    --vqc-params path/to/trained_params.npy \
    --output-dir results/ \
    --token "your_ibm_quantum_token"
```

### Selective Job Execution

```bash
# Skip Job 2 and 3, only run GW150914
python scripts/ibm_quantum_job_executor.py \
    --gw150914-features data/gw150914_features.npy \
    --test-features data/synthetic_test_features.npy \
    --vqc-params models/vqc_params.npy \
    --skip-job2 --skip-job3
```

### Command-Line Arguments

```
--gw150914-features PATH     Path to GW150914 feature vector (.npy), shape (12,) [REQUIRED]
--test-features PATH         Path to synthetic test features (.npy), shape (100, 12) [REQUIRED]
--vqc-params PATH            Path to trained VQC parameters (.npy), shape (48,) [REQUIRED]
--output-dir PATH            Output directory for job results (default: results/)
--token TOKEN                IBM Quantum token (uses IBM_QUANTUM_TOKEN env var if omitted)
--skip-job1                  Skip Job 1 (GW150914 with ZNE)
--skip-job2                  Skip Job 2 (100 events with ZNE)
--skip-job3                  Skip Job 3 (100 events without ZNE)
```

---

## Output Files

### Per-Job JSON Files
Location: `results/job_<job_id>.json`

```json
{
  "job_name": "Job 1: GW150914 + ZNE",
  "job_ids": ["cxyz123..."],
  "backend_name": "ibm_fez",
  "creation_timestamp": "2025-05-19T18:08:12.123456+00:00",
  "use_zne": true,
  "shots": 8192,
  "n_events": 1,
  "vqc_config": {
    "n_qubits": 12,
    "feature_map_reps": 2,
    "ansatz_reps": 2,
    "n_ansatz_params": 48,
    "total_cnot_gates": 88
  },
  "backend_properties": {
    "backend_name": "ibm_fez",
    "backend_version": "1.0.0",
    "n_qubits": 156,
    "T1": {
      "q0": 1.23e-04,
      "q1": 1.45e-04,
      ...
    },
    "T2": {
      "q0": 2.34e-04,
      "q1": 2.56e-04,
      ...
    },
    "2qubit_gate_errors": {
      "cx_q0_q1": 0.00234,
      "cx_q1_q2": 0.00245,
      ...
    }
  },
  "input_features": [
    [0.523, -0.412, 0.801, ...],
    ...
  ],
  "raw_expectation_values": [
    0.234,
    ...
  ],
  "class_probabilities": [
    [0.234, 0.766],
    ...
  ],
  "predicted_labels": [1, 0, 1, ...],
  "true_labels": [1, 0, 1, ...],
  "metrics": {
    "balanced_accuracy": 0.8234,
    "precision": 0.8156,
    "recall": 0.8312,
    "f1_score": 0.8233
  }
}
```

### Summary JSON
Location: `results/summary.json`

```json
{
  "experiment_timestamp": "2025-05-19T18:40:03.456789+00:00",
  "backend_name": "ibm_fez",
  "total_jobs": 3,
  "job_ids": ["cxyz123...", "cxyz456...", "cxyz789..."],
  "job_summaries": [
    {
      "job_name": "Job 1: GW150914 + ZNE",
      "job_ids": ["cxyz123..."],
      "use_zne": true,
      "n_events": 1,
      "metrics": {}
    },
    {
      "job_name": "Job 2: 100 Events + ZNE",
      "job_ids": ["cxyz456..."],
      "use_zne": true,
      "n_events": 100,
      "metrics": {
        "balanced_accuracy": 0.8234,
        "precision": 0.8156,
        "recall": 0.8312,
        "f1_score": 0.8233
      }
    },
    {
      "job_name": "Job 3: 100 Events - No ZNE",
      "job_ids": ["cxyz789..."],
      "use_zne": false,
      "n_events": 100,
      "metrics": {
        "balanced_accuracy": 0.7845,
        "precision": 0.7722,
        "recall": 0.7968,
        "f1_score": 0.7844
      }
    }
  ]
}
```

### Console Output (Thesis Appendix Format)

```
==========================================================================================
IBM QUANTUM JOB EXECUTION SUMMARY — For Thesis Appendix
==========================================================================================
Backend: ibm_fez
Execution Date: 2025-05-19T18:40:03.456789+00:00
VQC Config: 12 qubits, 88 CNOT gates, 48 parameters
Shots per circuit: 8192
------------------------------------------------------------------------------------------

JOB 1: JOB 1: GW150914 + ZNE
  Job IDs: cxyz123...
  Events: 1
  Zero-Noise Extrapolation: Yes
  T1 (μs): 1.34e-04 ± 2.34e-05
  T2 (μs): 2.45e-04 ± 3.45e-05
  2-Qubit Gate Error: 2.34e-03 ± 1.23e-04

  RESULTS:
    Balanced Accuracy: N/A

JOB 2: JOB 2: 100 EVENTS + ZNE
  Job IDs: cxyz456...
  Events: 100
  Zero-Noise Extrapolation: Yes
  T1 (μs): 1.34e-04 ± 2.34e-05
  T2 (μs): 2.45e-04 ± 3.45e-05
  2-Qubit Gate Error: 2.34e-03 ± 1.23e-04

  RESULTS:
    Balanced Accuracy: 0.8234
    Precision:        0.8156
    Recall:           0.8312
    F1 Score:         0.8233

JOB 3: JOB 3: 100 EVENTS - NO ZNE
  Job IDs: cxyz789...
  Events: 100
  Zero-Noise Extrapolation: No
  T1 (μs): 1.34e-04 ± 2.34e-05
  T2 (μs): 2.45e-04 ± 3.45e-05
  2-Qubit Gate Error: 2.34e-03 ± 1.23e-04

  RESULTS:
    Balanced Accuracy: 0.7845
    Precision:        0.7722
    Recall:           0.7968
    F1 Score:         0.7844

==========================================================================================
```

---

## Reproducibility & Citation

### For Thesis Inclusion

```bibtex
@data{qnim_ibm_jobs_2025,
  title={QNIM VQC Inference on IBM Quantum -- Gravitational Wave Classification},
  author={Your Name},
  year={2025},
  month={May},
  publisher={IBM Quantum},
  note={Job IDs: cxyz123..., cxyz456..., cxyz789..., 
        Backend: ibm_fez (156 qubits), 
        Retrieved from results/summary.json}
}
```

### Reproducibility Checklist

- [x] Backend properties snapshot (T1, T2, gate errors) captured
- [x] Job IDs logged for IBM Quantum job tracking
- [x] ISO 8601 timestamps for all events
- [x] Input feature vectors stored in JSON
- [x] Raw expectation values (pre-processing) saved
- [x] VQC circuit architecture documented (12q, 88 CNOTs, 48 params)
- [x] Resilience level and ZNE settings logged
- [x] Full probability distributions (softmax decoded)
- [x] Predictions traceable to raw measurements

---

## Error Handling & Recovery

### Job Submission Failures
If a job fails to submit (e.g., network error), the script:
1. Logs the error with full traceback
2. Continues with next job
3. Saves partial results to JSON with "error" field
4. Returns non-zero exit code only if NO jobs completed

### Result Retrieval Failures
If job results can't be retrieved:
1. Waits with exponential backoff (default: 3 retries)
2. Saves job ID for manual retrieval
3. Logs error with IBM Quantum job link

### Example Recovery
```bash
# After network interruption, view partial results:
cat results/job_cxyz123.json | grep error

# Re-run only Jobs 2 & 3:
python scripts/ibm_quantum_job_executor.py \
    --gw150914-features ... \
    --test-features ... \
    --vqc-params ... \
    --skip-job1
```

---

## Performance Notes

### Execution Time Estimates (ibm_fez open plan)

| Job | Events | Mode | Est. Time |
|-----|--------|------|-----------|
| Job 1 | 1 | ZNE (3 noise factors) | ~5-10 min |
| Job 2 | 100 | ZNE (3 noise factors) | ~1-2 hours* |
| Job 3 | 100 | No ZNE | ~20-30 min* |

*Actual time depends on queue depth and IBM Quantum availability

### Memory Usage
- Typical per-job: ~50 MB (1 event) to ~500 MB (100 events)
- Output JSON per job: ~5-10 MB per 100 events

---

## Troubleshooting

### Error: "IBM_QUANTUM_TOKEN not found"
```bash
export IBM_QUANTUM_TOKEN="your_token_here"
# or
python ... --token "your_token_here"
```

### Error: "Backend properties not available"
This is non-fatal. Backend will use default noise model.
Job continues and completes normally.

### Error: "Failed to load feature vectors"
```bash
# Verify file exists and shape:
python -c "import numpy as np; x = np.load('path.npy'); print(x.shape)"

# Expected shapes:
# GW150914: (12,)
# Synthetic: (100, 12)
# VQC params: (48,)
```

### Slow Job Completion
Jobs queue on ibm_fez during peak hours. Check status:
```bash
# View job in IBM Quantum console:
# https://quantum.ibm.com/jobs

# Or programmatically (advanced):
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService()
job = service.job("JOB_ID")
print(job.status())
```

---

## Advanced Usage

### Custom Noise Model (For Simulation)
To test with local simulator before submitting to hardware:

```bash
# Modify script: change BACKEND_NAME to "simulator_statevector"
# Set USE_REAL_HARDWARE=False
# Re-run with same arguments
```

### Batch Processing Multiple Experiments
```bash
#!/bin/bash
for seed in {1..5}; do
    python scripts/ibm_quantum_job_executor.py \
        --gw150914-features data/gw150914_features_seed${seed}.npy \
        --test-features data/synthetic_seed${seed}.npy \
        --vqc-params models/vqc_params_seed${seed}.npy \
        --output-dir results/seed${seed}/
done
```

---

## References

- Qiskit Runtime: https://docs.quantum.ibm.com/
- IBM Quantum: https://quantum.ibm.com/
- Zero-Noise Extrapolation: https://arxiv.org/abs/2005.10921
- QNIM Framework: [Insert thesis reference]

---

## Support & Contact

For issues or questions:
1. Check troubleshooting section above
2. Review IBM Quantum documentation
3. Check job logs in `results/` directory
4. Verify credentials and data files

---

*Last Updated: 2025-05-19*
*Version: 1.0 (Master's Thesis Release)*
