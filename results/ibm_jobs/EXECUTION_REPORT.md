# IBM Quantum Execution Report - Master's Thesis

## Execution Date: 2026-06-14

### Jobs Executed on IBM Quantum (ibm_fez, 156 qubits)

| Job | ID | Events | ZNE | Status | Time |
|-----|----|----|-----|--------|------|
| Job 1: GW150914 + ZNE | `d8neqir2d42s73cdpbeg` | 1 | Yes | ✅ Success | ~10s |
| Job 2: 100 Synthetic + ZNE | `d8neqmb2d42s73cdpbeg` | 100 | Yes | ✅ Success | ~4min |
| Job 3: 100 Synthetic - ZNE | `d8nesmo32u0s73fcc3k0` | 100 | No | ✅ Success | ~4min |

### Results Summary

**Job 2 vs Job 3 Comparison (100 events):**
- Balanced Accuracy: 0.5000 (both)
- Precision: 0.5200 (both)  
- Recall: 1.0000 (both)
- F1 Score: 0.6842 (both)

**Key Finding:** Identical results suggest high noise floor dominates both configurations.

### Configuration
- Backend: IBM Quantum Platform, ibm_fez (156 qubits)
- Shots per circuit: 8192
- VQC: 12 qubits, 48 parameters, ZZFeatureMap + Custom Ansatz
- Feature dimension: 12 (PCA from LIGO data)

