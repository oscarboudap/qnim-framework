# IBM Quantum Job Executor - Quick Reference Guide

## What You Have

Three production-ready Python scripts for executing and analyzing IBM Quantum VQC inference jobs for your astrophysics thesis:

### 1. **ibm_quantum_job_executor.py** (Main Script)
Executes three sequential jobs with complete metadata capture:
- Job 1: GW150914 with Zero-Noise Extrapolation
- Job 2: 100 synthetic events with ZNE
- Job 3: 100 synthetic events without ZNE (baseline)

**Output**: 
- 3 JSON files with raw data + metrics
- 1 summary JSON with all job IDs for citation
- Console output formatted for thesis appendix

### 2. **generate_test_data.py** (Utility)
Creates synthetic test data for validation and development:
- Generates GW150914 feature vector (12-dim)
- Generates 100 synthetic test events (100×12)
- Generates random VQC parameters (48-dim)

**Use when**: You want to test the script without real data first

### 3. **analyze_quantum_results.py** (Analysis Utility)
Post-execution analysis and reporting:
- Loads and summarizes job results
- Compares ZNE vs baseline performance
- Generates LaTeX tables for thesis

**Use after**: Execution completes to analyze results

### 4. **IBM_QUANTUM_EXECUTOR_README.md** (Documentation)
Complete documentation with:
- Architecture diagrams
- Usage examples
- Output format specifications
- Troubleshooting guide
- Performance estimates

---

## Quick Start (5 Minutes)

### Step 1: Install Dependencies
```bash
cd /c/Users/oscar/Desktop/TFM/qnim/qnim
pip install qiskit-ibm-runtime qiskit numpy scipy scikit-learn tqdm tabulate
```

### Step 2: Generate Test Data (Optional)
```bash
python scripts/generate_test_data.py
```

Output: `data/gw150914_features.npy`, `data/synthetic_test_features.npy`, `models/vqc_params.npy`

### Step 3: Set IBM Token
```bash
export IBM_QUANTUM_TOKEN="your_ibm_quantum_token_here"
```

Or pass directly: `python ... --token "your_token"`

### Step 4: Run Executor
```bash
python scripts/ibm_quantum_job_executor.py \
    --gw150914-features data/gw150914_features.npy \
    --test-features data/synthetic_test_features.npy \
    --vqc-params models/vqc_params.npy \
    --output-dir results/
```

### Step 5: View Results
```bash
# Summary
python scripts/analyze_quantum_results.py --summary-file results/summary.json

# Generate thesis table
python scripts/analyze_quantum_results.py --thesis-table results/ --output thesis_table.tex
```

---

## Key Specifications

| Parameter | Value |
|-----------|-------|
| Backend | ibm_fez (156 qubits) |
| VQC Qubits | 12 |
| CNOT Gates | 88 |
| VQC Parameters | 48 (EfficientSU2, 2 reps) |
| Feature Dimension | 12 (PCA) |
| Shots | 8192 |
| Job 1 Events | 1 (GW150914) |
| Job 2 Events | 100 (synthetic) |
| Job 3 Events | 100 (synthetic) |
| **Job 1 & 2** | ZNE enabled (resilience_level=2) |
| **Job 3** | ZNE disabled (resilience_level=0) |
| Noise Factors | [1, 2, 3] via gate folding |

---

## Input Data Format

### GW150914 Features File
- **Filename**: Any `.npy` file, e.g., `gw150914_features.npy`
- **Shape**: (12,)
- **Data Type**: numpy float64
- **Typical Range**: [-1, 1]
- **Description**: 12-dimensional PCA feature vector of GW150914 gravitational wave signal

### Synthetic Test Features File
- **Filename**: Any `.npy` file, e.g., `synthetic_test_features.npy`
- **Shape**: (100, 12)
- **Data Type**: numpy float64
- **Structure**: 13 theory classes × 8 events = 100 total
- **Typical Range**: [-1, 1]

### VQC Parameters File
- **Filename**: Any `.npy` file, e.g., `vqc_params.npy`
- **Shape**: (48,)
- **Data Type**: numpy float64
- **Typical Range**: [-π, π]
- **Description**: Pre-trained EfficientSU2(2 reps) parameters

**Generate test files with**: `python scripts/generate_test_data.py`

---

## Output Files Explained

### Per-Job Files: `results/job_<JOB_ID>.json`
Contains:
- `job_ids`: IBM Quantum job ID strings
- `creation_timestamp`: ISO 8601 format
- `backend_properties`: T1, T2, gate errors snapshot
- `input_features`: Input feature vectors as JSON array
- `raw_expectation_values`: Raw Z expectation values [-1, 1]
- `class_probabilities`: Softmax-decoded probabilities
- `predicted_labels`: Predicted class per event
- `true_labels`: True class labels (if provided)
- `metrics`: Balanced accuracy, precision, recall, F1

**Example**:
```json
{
  "job_name": "Job 2: 100 Events + ZNE",
  "job_ids": ["cxyz123..."],
  "backend_name": "ibm_fez",
  "creation_timestamp": "2025-05-19T18:08:12+00:00",
  "use_zne": true,
  "n_events": 100,
  "metrics": {
    "balanced_accuracy": 0.8234,
    "precision": 0.8156,
    "recall": 0.8312,
    "f1_score": 0.8233
  }
}
```

### Summary File: `results/summary.json`
Contains:
- `experiment_timestamp`: When experiment ran
- `total_jobs`: Number of jobs executed
- `job_ids`: All job IDs (for citation)
- `job_summaries`: Summary of each job with metrics

**Example**:
```json
{
  "experiment_timestamp": "2025-05-19T18:40:03+00:00",
  "backend_name": "ibm_fez",
  "total_jobs": 3,
  "job_ids": ["cxyz123...", "cxyz456...", "cxyz789..."],
  "job_summaries": [
    {
      "job_name": "Job 1: GW150914 + ZNE",
      "job_ids": ["cxyz123..."],
      "use_zne": true,
      "metrics": {}
    },
    ...
  ]
}
```

### Console Output (Thesis Appendix Format)
Automatically printed at end of execution:
```
==========================================================================================
IBM QUANTUM JOB EXECUTION SUMMARY — For Thesis Appendix
==========================================================================================

JOB 1: GW150914 + ZNE
  Job IDs: cxyz123...
  Events: 1
  Zero-Noise Extrapolation: Yes
  T1 (μs): 1.34e-04 ± 2.34e-05
  T2 (μs): 2.45e-04 ± 3.45e-05
  2-Qubit Gate Error: 2.34e-03 ± 1.23e-04

JOB 2: 100 EVENTS + ZNE
  Balanced Accuracy: 0.8234
  Precision: 0.8156
  Recall: 0.8312
  F1 Score: 0.8233

JOB 3: 100 EVENTS - NO ZNE
  Balanced Accuracy: 0.7845
  Precision: 0.7722
  Recall: 0.7968
  F1 Score: 0.7844
```

---

## Command-Line Options

```bash
python scripts/ibm_quantum_job_executor.py \
    --gw150914-features PATH      # [REQUIRED] GW150914 vector file
    --test-features PATH           # [REQUIRED] 100 synthetic events file
    --vqc-params PATH              # [REQUIRED] VQC parameters file
    --output-dir PATH              # Output directory (default: results/)
    --token TOKEN                  # IBM token (default: IBM_QUANTUM_TOKEN env var)
    --skip-job1                    # Skip Job 1
    --skip-job2                    # Skip Job 2
    --skip-job3                    # Skip Job 3
```

---

## Common Workflows

### Test with Synthetic Data First
```bash
python scripts/generate_test_data.py
python scripts/ibm_quantum_job_executor.py \
    --gw150914-features data/gw150914_features.npy \
    --test-features data/synthetic_test_features.npy \
    --vqc-params models/vqc_params.npy \
    --skip-job3  # Skip baseline to save time
```

### Run Only Job 1 (GW150914)
```bash
python scripts/ibm_quantum_job_executor.py \
    --gw150914-features data/gw150914_features.npy \
    --test-features data/synthetic_test_features.npy \
    --vqc-params models/vqc_params.npy \
    --skip-job2 --skip-job3
```

### Compare ZNE Performance
```bash
# After execution, compare Job 2 (ZNE) vs Job 3 (baseline)
python scripts/analyze_quantum_results.py \
    --compare-jobs results/job_<JOB2_ID>.json results/job_<JOB3_ID>.json
```

### Generate Thesis Table
```bash
python scripts/analyze_quantum_results.py \
    --thesis-table results/ \
    --output my_thesis_results.tex

# Then include in LaTeX:
# \input{my_thesis_results.tex}
```

---

## Execution Time Estimates

| Job | Events | Mode | Est. Time |
|-----|--------|------|-----------|
| Job 1 | 1 | ZNE | 5–10 min |
| Job 2 | 100 | ZNE | 1–2 hours |
| Job 3 | 100 | No ZNE | 20–30 min |
| **Total** | — | — | **1.5–2.5 hours** |

*Times vary with IBM Quantum queue depth. Test with synthetic data first.*

---

## Error Handling

The script automatically:
- ✅ Catches network errors during job submission
- ✅ Retries failed result retrievals
- ✅ Saves partial results if jobs fail
- ✅ Continues with next job if current fails
- ✅ Logs all errors with full details
- ✅ Provides job IDs for manual tracking

**If a job fails**: Check `results/job_<ID>.json` for error details, or re-run with `--skip-job<N>` to continue from next job.

---

## Reproducibility Checklist

For thesis citation, ensure you capture:

- [ ] Job IDs (saved in `results/summary.json`)
- [ ] Execution date/time (ISO 8601, saved in metadata)
- [ ] Backend name (ibm_fez)
- [ ] Backend properties (T1, T2 in JSON)
- [ ] VQC architecture (12q, 88 CNOTs, 48 params)
- [ ] Input file paths (saved as metadata)
- [ ] Raw measurements (in per-job JSON)
- [ ] Balanced accuracy + metrics (in JSON)

**All captured automatically** → Include in thesis as:

```
This section reports results from IBM Quantum jobs executed on the ibm_fez backend
(156 qubits). Job execution metadata including job IDs, backend calibration data,
and detailed results are provided in Appendix X. The following job IDs enable
reproducibility and verification:

Job IDs: cxyz123..., cxyz456..., cxyz789...
Backend: ibm_fez
Execution Date: 2025-05-19
```

---

## Next Steps

1. **Prepare data**: Format your GW150914 and synthetic test events as `.npy` files
2. **Set token**: `export IBM_QUANTUM_TOKEN="..."`
3. **Run executor**: `python scripts/ibm_quantum_job_executor.py --gw150914-features ... --test-features ... --vqc-params ...`
4. **Analyze results**: `python scripts/analyze_quantum_results.py --summary-file results/summary.json`
5. **Generate table**: `python scripts/analyze_quantum_results.py --thesis-table results/ --output thesis_table.tex`
6. **Cite in thesis**: Include job IDs and backend metadata from `results/summary.json`

---

## Support & Documentation

- **Full README**: `scripts/IBM_QUANTUM_EXECUTOR_README.md`
- **IBM Qiskit Runtime Docs**: https://docs.quantum.ibm.com/
- **IBM Quantum**: https://quantum.ibm.com/

---

*Created: 2025-05-19 | Version: 1.0 | Master's Thesis Edition*
