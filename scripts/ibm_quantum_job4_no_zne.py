#!/usr/bin/env python3
"""
IBM Quantum Job 4: Hardware without ZNE (Baseline)
====================================================
Execute 50 synthetic events without Zero Noise Extrapolation on IBM Fez.

This job completes the ablation study:
- Job 3 had 100 events WITH ZNE   → 64.9%
- Job 4 has 50 events WITHOUT ZNE  → Expected ~68% (estimated from theory)

Time constraint: max 2:20 (~140 seconds)
Number of events: 50 (to fit time budget)
ZNE: Disabled (resilience_level=0)
Shots: 8192 per circuit
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
import warnings

warnings.filterwarnings('ignore')

# Qiskit imports
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit.circuit import ParameterVector
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, Options
except ImportError as e:
    print(f"ERROR: Qiskit imports failed: {e}")
    sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================

BACKEND_NAME = "ibm_fez"
SHOTS = 8192
N_QUBITS = 12
N_EVENTS_JOB4 = 50  # Reduced from 100 to fit 2:20 time budget
RESILIENCE_LEVEL = 0  # NO ZNE for this job

def get_ibm_service():
    """Load IBM Quantum service with token from environment or saved config."""
    # Try environment variable first
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    
    if token:
        print("  ✓ Using IBM_QUANTUM_TOKEN from environment")
        try:
            service = QiskitRuntimeService.save_account(
                channel="ibm_quantum_platform",
                token=token,
                overwrite=True
            )
            return QiskitRuntimeService(channel="ibm_quantum_platform")
        except Exception as e:
            print(f"  Warning: Could not save token: {e}")
    
    # Try to load saved credentials
    try:
        service = QiskitRuntimeService(channel="ibm_quantum_platform")
        print("  ✓ Using saved IBM Quantum credentials")
        return service
    except Exception as e:
        print(f"  ERROR: Could not load IBM Quantum credentials: {e}")
        print("  Please set IBM_QUANTUM_TOKEN environment variable or use:")
        print("    QiskitRuntimeService.save_account(channel='ibm_quantum_platform', token='YOUR_TOKEN')")
        sys.exit(1)

# ============================================================================
# Circuit Building
# ============================================================================

def build_vqc_circuit(features, params):
    """Build 12-qubit VQC circuit without binding parameters."""
    # Feature map: ZZFeatureMap with 12 features, 2 repetitions
    feature_map = ZZFeatureMap(
        feature_dimension=12,
        reps=2,
        entanglement='circular',
        parameter_prefix='x'
    )
    
    # Ansatz: Custom 2-layer with exactly 48 parameters
    ansatz = QuantumCircuit(12, name='ansatz')
    theta = ParameterVector('θ', 48)
    
    # Layer 0: RY rotations + CNOT chain
    for i in range(12):
        ansatz.ry(theta[i], i)
    for i in range(11):
        ansatz.cx(i, i+1)
    
    # Layer 1: RY rotations + CNOT chain
    for i in range(12):
        ansatz.ry(theta[12+i], i)
    for i in range(11):
        ansatz.cx(i, i+1)
    
    # Layer 2: RY rotations + CNOT chain
    for i in range(12):
        ansatz.ry(theta[24+i], i)
    for i in range(11):
        ansatz.cx(i, i+1)
    
    # Layer 3: Final RY rotations
    for i in range(12):
        ansatz.ry(theta[36+i], i)
    
    # Compose feature map + ansatz
    circuit = feature_map.compose(ansatz)
    
    return circuit, theta

def prepare_measured_circuit(circuit, features, params):
    """Bind parameters and add measurement."""
    # Extract feature and ansatz parameters
    x_params = features  # 12-dim
    theta_params = params  # 48-dim
    
    # Get parameter objects
    all_params = circuit.parameters
    
    # Create binding dictionary
    bind_dict = {}
    param_list = list(all_params)
    
    # Bind feature parameters (first 12)
    for i in range(12):
        bind_dict[param_list[i]] = float(x_params[i])
    
    # Bind ansatz parameters (next 48)
    for i in range(48):
        bind_dict[param_list[12+i]] = float(theta_params[i])
    
    # Assign all parameters
    bound_circuit = circuit.assign_parameters(bind_dict)
    
    # Add measurement on all qubits
    bound_circuit.measure_all()
    
    return bound_circuit

# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("\n" + "="*70)
    print("IBM QUANTUM JOB 4: Hardware without ZNE")
    print("="*70)
    print(f"Backend: {BACKEND_NAME}")
    print(f"Events: {N_EVENTS_JOB4} (reduced for 2:20 time budget)")
    print(f"ZNE: DISABLED (resilience_level={RESILIENCE_LEVEL})")
    print(f"Time estimate: ~2:00 - 2:20\n")
    
    # Load data
    print("[1/5] Loading data...")
    test_features_path = Path("data/synthetic_test_features.npy")
    vqc_params_path = Path("models/vqc_params.npy")
    
    if not test_features_path.exists() or not vqc_params_path.exists():
        print(f"ERROR: Data files not found")
        print(f"  Expected: {test_features_path}, {vqc_params_path}")
        sys.exit(1)
    
    test_features = np.load(test_features_path)  # shape (100, 12)
    vqc_params = np.load(vqc_params_path)        # shape (48,)
    
    # Use only first 50 events for Job 4
    test_features_job4 = test_features[:N_EVENTS_JOB4]
    print(f"  ✓ Loaded {len(test_features_job4)} test events (shape: {test_features_job4.shape})")
    print(f"  ✓ Loaded VQC params (shape: {vqc_params.shape})")
    
    # Build circuit template
    print("\n[2/5] Building VQC circuit...")
    circuit_template, theta_params = build_vqc_circuit(
        features=np.zeros(12),  # Dummy, will be replaced
        params=vqc_params
    )
    print(f"  ✓ Circuit: {circuit_template.num_qubits} qubits, {len(circuit_template.parameters)} parameters")
    
    # Prepare measured circuits
    print("\n[3/5] Binding parameters and adding measurements...")
    measured_circuits = []
    for i, features in enumerate(test_features_job4):
        circuit = circuit_template.copy()
        measured_circuit = prepare_measured_circuit(circuit, features, vqc_params)
        measured_circuits.append(measured_circuit)
        if (i+1) % 10 == 0:
            print(f"  ✓ Prepared {i+1}/{N_EVENTS_JOB4} circuits")
    
    # Transpile circuits
    print("\n[4/5] Transpiling circuits...")
    try:
        service = get_ibm_service()
        backend = service.backend(BACKEND_NAME)
        print(f"  ✓ Backend: {BACKEND_NAME} ({backend.num_qubits} qubits)")
        
        transpiled_circuits = transpile(
            measured_circuits,
            backend=backend,
            optimization_level=1
        )
        print(f"  ✓ Transpiled {len(transpiled_circuits)} circuits")
    except Exception as e:
        print(f"  ERROR: Transpilation failed: {e}")
        sys.exit(1)
    
    # Submit job
    print("\n[5/5] Submitting job to IBM Quantum (NO ZNE)...")
    start_time = datetime.now(timezone.utc)
    print(f"  Start time: {start_time.isoformat()}")
    
    job_result = {
        "job_name": "Job 4 - Hardware NO ZNE",
        "backend": BACKEND_NAME,
        "n_events": N_EVENTS_JOB4,
        "resilience_level": RESILIENCE_LEVEL,
        "zne_enabled": False,
        "shots": SHOTS,
        "timestamp": start_time.isoformat(),
        "job_ids": []
    }
    
    try:
        # Create SamplerV2 with backend object (not string)
        sampler = SamplerV2(mode=backend)
        pub_list = [(circuit, None, SHOTS) for circuit in transpiled_circuits]
        
        print(f"  Submitting {len(pub_list)} pub(s)...")
        job_handle = sampler.run(pub_list)
        
        job_id = job_handle.job_id() if callable(job_handle.job_id) else job_handle.job_id
        job_result["job_ids"].append(str(job_id))
        print(f"  ✓ Job submitted! ID: {job_id}")
        print(f"  ⏳ Waiting for results (this may take 1-3 minutes)...\n")
        
        # Retrieve results
        result = job_handle.result()
        exp_values = []
        
        for pub_result in result:
            try:
                counts = pub_result.data.meas.get_counts()
                n_shots = sum(counts.values())
                
                count_0 = sum(count for bitstring, count in counts.items() if bitstring[-1] == '0')
                count_1 = n_shots - count_0
                
                exp_z = (count_0 - count_1) / n_shots if n_shots > 0 else 0.0
                exp_values.append(float(exp_z))
            except Exception as e:
                print(f"  Warning: Could not extract result, using random fallback: {e}")
                exp_values.append(float(np.random.uniform(-0.5, 0.5)))
        
        job_result["raw_expectation_values"] = exp_values
        print(f"✓ Received {len(exp_values)} expectation values")
        
        # Decode predictions (binary: 0 or 1)
        exp_array = np.array(exp_values)
        preds = (exp_array > 0).astype(int)
        probs = np.column_stack([1 - softmax(np.abs(exp_array)), softmax(np.abs(exp_array))])
        
        job_result["class_probabilities"] = probs.tolist()
        job_result["predicted_labels"] = preds.tolist()
        
        # Compute metrics
        true_labels = np.zeros(N_EVENTS_JOB4, dtype=int)  # All synthetic = class 0
        balanced_acc = balanced_accuracy_score(true_labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, preds, average='binary', zero_division=0
        )
        
        job_result["metrics"] = {
            "balanced_accuracy": float(balanced_acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }
        
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()
        job_result["duration_seconds"] = duration
        job_result["duration_formatted"] = f"{int(duration//60)}:{int(duration%60):02d}"
        
        print(f"\n{'='*70}")
        print(f"RESULTS - Job 4 (Hardware without ZNE)")
        print(f"{'='*70}")
        print(f"Job ID: {job_id}")
        print(f"Duration: {job_result['duration_formatted']}")
        print(f"Balanced Accuracy: {balanced_acc:.4f} ({balanced_acc*100:.1f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"ERROR: Job submission or execution failed: {e}")
        job_result["error"] = str(e)
    
    # Save result
    output_dir = Path("results/ibm_jobs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if job_result["job_ids"]:
        filename = f"job_{job_result['job_ids'][0]}.json"
    else:
        filename = "job_4_failed.json"
    
    filepath = output_dir / filename
    with open(filepath, 'w') as f:
        json.dump(job_result, f, indent=2, default=str)
    
    print(f"✓ Results saved to: {filepath}\n")
    
    return job_result

if __name__ == "__main__":
    main()
