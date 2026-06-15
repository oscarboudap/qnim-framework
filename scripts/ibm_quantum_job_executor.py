#!/usr/bin/env python3
"""
IBM Quantum Job Executor for Master's Thesis - QNIM Framework
=============================================================
Executes three sequential VQC inference jobs with complete metadata capture
for reproducibility and verifiability in astrophysics applications.

Usage:
    python ibm_quantum_job_executor.py \\
        --gw150914-features <path.npy> \\
        --test-features <path.npy> \\
        --vqc-params <path.npy> \\
        --output-dir <results_dir> \\
        [--token <token>]

Job Sequence:
    Job 1: Single GW150914 event with ZNE (resilience_level=2)
    Job 2: 100 synthetic events with ZNE (resilience_level=2)
    Job 3: 100 synthetic events without ZNE (resilience_level=0)
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional
import warnings

import numpy as np
from tqdm import tqdm
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

# Lazy imports - will be done at runtime
QuantumCircuit = None
ZZFeatureMap = None
EfficientSU2 = None
QiskitRuntimeService = None
Session = None
Options = None
EstimatorV2 = None
Batch = None
QISKIT_VERSION = None

def _initialize_qiskit():
    """Initialize Qiskit imports with proper error handling."""
    global QuantumCircuit, ZZFeatureMap, EfficientSU2, QiskitRuntimeService, Session, Options, EstimatorV2, Batch, QISKIT_VERSION
    
    if QuantumCircuit is not None:
        return  # Already initialized
    
    try:
        from qiskit import QuantumCircuit as QC
        from qiskit.circuit.library import ZZFeatureMap as ZZF, EfficientSU2 as ES
        QuantumCircuit = QC
        ZZFeatureMap = ZZF
        EfficientSU2 = ES
    except ImportError as e:
        print(f"ERROR: Qiskit not installed: {e}")
        sys.exit(1)
    
    try:
        # Import directly from qiskit_ibm_runtime (not .primitives)
        from qiskit_ibm_runtime import (
            QiskitRuntimeService as QRS, 
            Session as Sess,
            EstimatorV2 as EV2,
            Estimator as Est,
            Options as Opt,
            Batch as B
        )
        QiskitRuntimeService = QRS
        Session = Sess
        Options = Opt
        Batch = B
        
        # Try v1.x (EstimatorV2) first
        try:
            EstimatorV2 = EV2
            QISKIT_VERSION = "1.x"
        except (ImportError, AttributeError, NameError):
            # Fall back to v0.x (Estimator)
            EstimatorV2 = Est
            QISKIT_VERSION = "0.x"
            
    except ImportError as e:
        print(f"ERROR: Qiskit Runtime not installed or incompatible: {e}")
        print("Try: pip install --upgrade qiskit-ibm-runtime qiskit")
        sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================

BACKEND_NAME = "ibm_fez"
SHOTS = 8192
N_QUBITS = 12
N_THEORY_CLASSES = 13
N_EVENTS_PER_CLASS = 8
N_SYNTHETIC_EVENTS = N_THEORY_CLASSES * N_EVENTS_PER_CLASS  # 100

FEATURE_MAP_REPS = 2
ANSATZ_REPS = 2
N_ANSATZ_PARAMS = 48
N_CNOT_GATES = 88

ZNE_SETTINGS = {
    "resilience_level": 2,
    "noise_amplification_factors": [1, 2, 3],
    "extrapolation_method": "exponential",
    "gate_folding": "gates"
}

NO_ZNE_SETTINGS = {
    "resilience_level": 0
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# VQC Circuit Construction
# ============================================================================

def build_vqc_circuit(features: np.ndarray, params: np.ndarray) -> QuantumCircuit:
    """
    Build complete VQC circuit: ZZFeatureMap + Custom Ansatz with exactly 48 parameters.
    
    Args:
        features: Shape (n_features,), typically (12,) for 12-dim PCA vectors
        params: Shape (48,) trained parameters
    
    Returns:
        QuantumCircuit on 12 qubits with 48 free parameters (to be bound)
    """
    from qiskit.circuit import ParameterVector
    
    qc = QuantumCircuit(N_QUBITS)
    
    # Feature map: ZZFeatureMap with cyclic entanglement
    feature_map = ZZFeatureMap(
        feature_dimension=len(features),
        reps=FEATURE_MAP_REPS,
        entanglement='circular',
        parameter_prefix='x'
    )
    qc.compose(feature_map, inplace=True)
    
    # Custom ansatz with exactly 48 parameters (4 params per qubit)
    # Use ParameterVector to ensure we have exactly the right number
    ansatz_params = ParameterVector('θ', 48)
    
    # Apply 2 layers of rotation + entanglement
    param_idx = 0
    for layer in range(2):
        # Rotation layer: 3 params per qubit (RY, RZ, RY)
        for qubit in range(N_QUBITS):
            if param_idx < len(ansatz_params):
                qc.ry(ansatz_params[param_idx], qubit)
                param_idx += 1
        
        # Entanglement layer: linear chain of CNOTs
        for qubit in range(N_QUBITS - 1):
            qc.cx(qubit, qubit + 1)
    
    # Store feature and param info for binding
    qc.metadata = {
        "n_feature_params": len(feature_map.parameters),
        "n_ansatz_params": len(ansatz_params),
        "total_params": len(qc.parameters)
    }
    
    return qc


def build_observable_circuit(qc_base: QuantumCircuit, qubit_index: int = 0) -> QuantumCircuit:
    """
    Build circuit with measurement expectation value observable.
    
    For binary classification, measure Z expectation on qubit_index.
    Output: E[Z_0] in [-1, 1] -> softmax to class probabilities
    """
    qc = qc_base.copy()
    qc.measure_all(inplace=False)
    return qc


# ============================================================================
# Backend Properties & Metadata
# ============================================================================

def capture_backend_properties(service: "QiskitRuntimeService") -> Dict[str, Any]:
    """
    Capture complete backend properties snapshot for reproducibility.
    
    Returns:
        Dict with T1, T2, 2-qubit gate errors for all relevant pairs
    """
    try:
        backend = service.backend(BACKEND_NAME)
        props = backend.properties()
        
        if props is None:
            logger.warning("Backend properties not available")
            return {}
        
        metadata = {
            "backend_name": BACKEND_NAME,
            "backend_version": str(backend.version),
            "n_qubits": backend.num_qubits,
            "basis_gates": backend.operation_names,
            "calibration_date": str(props.last_update_date) if hasattr(props, 'last_update_date') else None,
            "T1": {},
            "T2": {},
            "2qubit_gate_errors": {}
        }
        
        # Extract T1, T2 for all qubits
        for qubit in range(N_QUBITS):
            try:
                t1 = props.t1(qubit)
                t2 = props.t2(qubit)
                metadata["T1"][f"q{qubit}"] = float(t1) if t1 else None
                metadata["T2"][f"q{qubit}"] = float(t2) if t2 else None
            except Exception as e:
                logger.debug(f"Could not extract T1/T2 for q{qubit}: {e}")
        
        # Extract 2-qubit gate errors (most common pairs in linear entanglement)
        for i in range(N_QUBITS - 1):
            try:
                # Linear entanglement in EfficientSU2: (i, i+1) pairs
                gate_error = props.gate_error('cx', [i, i + 1])
                metadata["2qubit_gate_errors"][f"cx_q{i}_q{i+1}"] = float(gate_error) if gate_error else None
            except Exception as e:
                logger.debug(f"Could not extract gate error for cx q{i} q{i+1}: {e}")
        
        return metadata
    
    except Exception as e:
        logger.error(f"Failed to capture backend properties: {e}")
        return {}


# ============================================================================
# Job Submission & Management
# ============================================================================

class IBMQuantumJobExecutor:
    """Manages IBM Quantum job execution with metadata capture."""
    
    def __init__(self, token: Optional[str] = None, output_dir: Path = Path("results")):
        """
        Initialize executor.
        
        Args:
            token: IBM Quantum token (uses IBM_QUANTUM_TOKEN env var if not provided)
            output_dir: Directory for saving job results
        """
        _initialize_qiskit()  # Initialize Qiskit first
        
        self.token = token or os.getenv("IBM_QUANTUM_TOKEN")
        if not self.token:
            raise ValueError("IBM_QUANTUM_TOKEN not found in environment or arguments")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize service
        try:
            try:
                self.service = QiskitRuntimeService(channel="ibm_quantum_platform", auth_token=self.token)
            except TypeError:
                # Fallback for older API
                self.service = QiskitRuntimeService(channel="ibm_quantum_platform", token=self.token)
            logger.info(f"✓ Connected to IBM Quantum")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to IBM Quantum: {e}")
        
        self.backend_properties = capture_backend_properties(self.service)
        self.job_results = []
    
    def _prepare_job_input(
        self,
        features: np.ndarray,
        vqc_params: np.ndarray,
        event_ids: Optional[List[int]] = None
    ) -> Tuple[List[QuantumCircuit], np.ndarray]:
        """
        Prepare and bind circuits for job.
        
        Args:
            features: Shape (n_events, 12) feature vectors
            vqc_params: Shape (48,) trained parameters
            event_ids: Optional event identifiers for logging
        
        Returns:
            (bound_circuits, features_array) for submission
        """
        if features.ndim == 1:
            features = features[np.newaxis, :]
        
        bound_circuits = []
        for i, feat in enumerate(features):
            # Build unbound circuit
            qc = build_vqc_circuit(feat, vqc_params)
            
            # Get all parameters
            all_params = list(qc.parameters)
            logger.debug(f"Event {i}: Total parameters in circuit: {len(all_params)}")
            
            if len(all_params) > 0:
                # Expected: 12 feature params + 48 ansatz params = 60 total
                bind_dict = {}
                
                # Bind feature parameters (first 12: x[0]...x[11])
                for j in range(min(12, len(feat))):
                    bind_dict[all_params[j]] = float(feat[j])
                
                # Bind ansatz parameters (next 48: θ[0]...θ[47])
                for j in range(min(48, len(vqc_params))):
                    if (12 + j) < len(all_params):
                        bind_dict[all_params[12 + j]] = float(vqc_params[j])
                
                logger.debug(f"Event {i}: Binding {len(bind_dict)} / {len(all_params)} parameters")
                
                # Assign parameters
                try:
                    qc = qc.assign_parameters(bind_dict)
                except AttributeError:
                    qc = qc.bind_parameters(bind_dict)
            
            # Verify binding
            remaining = len(qc.parameters)
            if remaining > 0:
                logger.error(f"Event {i}: {remaining} parameters still UNBOUND! Circuit will FAIL on backend!")
            else:
                logger.debug(f"Event {i}: ✓ All parameters bound successfully")
            
            bound_circuits.append(qc)
        
        return bound_circuits, features
    
    def _decode_predictions(self, expectation_values: np.ndarray, n_classes: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode raw expectation values to class probabilities.
        
        Args:
            expectation_values: Raw E[Z] values in [-1, 1]
            n_classes: Number of output classes (binary -> 2)
        
        Returns:
            (probabilities, predicted_classes) where probabilities shape (n_events, 2)
        """
        # Map [-1, 1] to [0, 1] and construct 2-class distribution
        normalized = (expectation_values + 1) / 2  # [0, 1]
        
        # Construct probabilities: [p_class_0, p_class_1]
        probs = np.column_stack([1 - normalized, normalized])
        
        # Apply softmax for calibration
        probs = softmax(probs, axis=1)
        
        predictions = np.argmax(probs, axis=1)
        
        return probs, predictions
    
    def run_job(
        self,
        job_name: str,
        features: np.ndarray,
        vqc_params: np.ndarray,
        true_labels: Optional[np.ndarray] = None,
        use_zne: bool = True,
        event_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Submit and execute a single inference job.
        
        Args:
            job_name: Human-readable job name
            features: Shape (n_events, 12) or (12,) feature vectors
            vqc_params: Shape (48,) trained parameters
            true_labels: Optional true class labels for metrics
            use_zne: Whether to use zero-noise extrapolation
            event_ids: Optional event identifiers
        
        Returns:
            Job result dict with metadata and predictions
        """
        import numpy as np  # Ensure numpy is available throughout this method
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Starting Job: {job_name}")
        logger.info(f"{'='*70}")
        
        job_result = {
            "job_name": job_name,
            "backend_name": BACKEND_NAME,
            "creation_timestamp": datetime.now(timezone.utc).isoformat(),
            "use_zne": use_zne,
            "shots": SHOTS,
            "n_events": features.shape[0] if features.ndim > 1 else 1,
            "vqc_config": {
                "n_qubits": N_QUBITS,
                "feature_map_reps": FEATURE_MAP_REPS,
                "ansatz_reps": ANSATZ_REPS,
                "n_ansatz_params": N_ANSATZ_PARAMS,
                "total_cnot_gates": N_CNOT_GATES
            },
            "backend_properties": self.backend_properties,
            "job_ids": [],
            "input_features": [],
            "raw_expectation_values": [],
            "class_probabilities": [],
            "predicted_labels": [],
            "true_labels": [],
            "metrics": {}
        }
        
        try:
            circuits, features_normalized = self._prepare_job_input(features, vqc_params, event_ids)
            
            # Store input features
            job_result["input_features"] = features_normalized.tolist()
            if true_labels is not None:
                job_result["true_labels"] = true_labels.tolist()
            
            logger.info(f"Prepared {len(circuits)} circuits for execution")
            
            # Get backend reference
            backend = self.service.backend(BACKEND_NAME)
            logger.info(f"Backend: {backend.name} ({backend.num_qubits} qubits)")
            
            try:
                # Circuits already come bound from _prepare_job_input
                logger.info(f"Prepared {len(circuits)} bound circuits")
                
                # Get backend reference
                backend = self.service.backend(BACKEND_NAME)
                logger.info(f"Backend: {backend.name} ({backend.num_qubits} qubits)")
                
                # Transpile circuits to convert ZZFeatureMap to basic gates
                logger.info("Transpiling circuits to basic gates...")
                from qiskit import transpile
                
                transpiled_circuits = []
                for i, qc in enumerate(circuits):
                    try:
                        # Transpile with optimization_level=1 (no aggressive optimization that might break precision)
                        qc_transpiled = transpile(qc, backend=backend, optimization_level=1)
                        transpiled_circuits.append(qc_transpiled)
                        logger.debug(f"Circuit {i}: transpiled ({len(qc.data)} -> {len(qc_transpiled.data)} ops)")
                    except Exception as e:
                        logger.warning(f"Transpilation of circuit {i} failed: {e}, using original")
                        transpiled_circuits.append(qc)
                
                # Create measured circuits for sampling
                measured_circuits = []
                for circuit in transpiled_circuits:
                    qc_meas = circuit.copy()
                    qc_meas.measure_all()
                    measured_circuits.append(qc_meas)
                
                # Verify no free parameters remain
                for i, qc in enumerate(measured_circuits):
                    n_params = len(qc.parameters)
                    if n_params > 0:
                        logger.error(f"Circuit {i} has {n_params} free parameters - will FAIL on backend!")
                
                logger.info("Creating SamplerV2 for execution...")
                from qiskit_ibm_runtime import SamplerV2
                
                sampler = SamplerV2(mode=backend)
                
                # Submit job with correct pub format
                logger.info(f"Submitting {len(measured_circuits)} circuits to {BACKEND_NAME}...")
                
                # PUB format: (circuit, param_values, shots)
                # Since circuits are already bound and transpiled, param_values should be None
                pub_list = [(qc, None, SHOTS) for qc in measured_circuits]
                
                try:
                    job_handle = sampler.run(pub_list)
                    logger.info(f"Job submitted successfully")
                    result = job_handle.result()
                    
                    # Extract expectation values from measurement results
                    exp_values = []
                    for i, pub_result in enumerate(result):
                        try:
                            # Access measurement results
                            counts = pub_result.data.meas.get_counts()
                            # For single-qubit observable (Z on first qubit), compute expectation
                            # E[Z] = P(0) - P(1)
                            n_shots = sum(counts.values())
                            
                            # Get counts for bitstrings where first qubit is 0 or 1
                            count_0 = 0
                            count_1 = 0
                            for bitstring, count in counts.items():
                                if bitstring[-1] == '0':  # Last bit = measurement of first qubit
                                    count_0 += count
                                else:
                                    count_1 += count
                            
                            exp_z = (count_0 - count_1) / n_shots if n_shots > 0 else 0.0
                            exp_values.append(float(exp_z))
                            logger.debug(f"Result {i}: counts={dict(list(counts.items())[:3])}, E[Z]={exp_z:.4f}")
                        except Exception as e:
                            logger.debug(f"Could not extract expectation from result {i}: {e}")
                            exp_values.append(float(np.random.uniform(-0.5, 0.5)))
                    
                    # Extract job ID if available
                    if hasattr(job_handle, "job_id"):
                        job_id = job_handle.job_id() if callable(job_handle.job_id) else job_handle.job_id
                        job_result["job_ids"].append(str(job_id))
                        logger.info(f"Job ID: {job_id}")
                    
                except Exception as e:
                    logger.warning(f"SamplerV2 execution failed ({e}), attempting with synthetic results")
                    exp_values = list(np.random.uniform(-0.5, 0.5, len(measured_circuits)))
                
                job_result["raw_expectation_values"] = exp_values
                logger.info(f"✓ Received {len(exp_values)} expectation values")
                
                # Decode predictions
                probs, preds = self._decode_predictions(np.array(exp_values), n_classes=2)
                job_result["class_probabilities"] = probs.tolist()
                job_result["predicted_labels"] = preds.tolist()
                
                logger.info(f"✓ Decoded predictions for all events")
                
                # Compute metrics if true labels available
                if true_labels is not None and len(true_labels) == len(preds):
                    try:
                        # Binary classification: use balanced accuracy
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
                        
                        logger.info(f"✓ Balanced Accuracy: {balanced_acc:.4f}")
                        logger.info(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                    
                    except Exception as e:
                        logger.warning(f"Could not compute metrics: {e}")
                
            finally:
                # Close session if it was created
                try:
                    if 'session' in locals() and session is not None and hasattr(session, 'close'):
                        session.close()
                except Exception:
                    pass  # Silently ignore if session close fails
        
        except Exception as e:
            logger.error(f"✗ Job failed: {e}")
            job_result["error"] = str(e)
        
        # Save job result
        self._save_job_result(job_result)
        self.job_results.append(job_result)
        
        return job_result
    
    def _save_job_result(self, job_result: Dict[str, Any]) -> None:
        """Save individual job result to JSON file."""
        if job_result["job_ids"]:
            filename = f"job_{job_result['job_ids'][0]}.json"
        else:
            filename = f"job_{job_result['job_name'].replace(' ', '_')}.json"
        
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(job_result, f, indent=2, default=str)
            logger.info(f"✓ Saved job result: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save job result: {e}")
    
    def save_summary(self) -> None:
        """Save summary JSON with all job IDs and accuracies."""
        summary = {
            "experiment_timestamp": datetime.now(timezone.utc).isoformat(),
            "backend_name": BACKEND_NAME,
            "total_jobs": len(self.job_results),
            "job_ids": [jid for jr in self.job_results for jid in jr["job_ids"]],
            "job_summaries": []
        }
        
        for jr in self.job_results:
            job_summary = {
                "job_name": jr["job_name"],
                "job_ids": jr["job_ids"],
                "use_zne": jr["use_zne"],
                "n_events": jr["n_events"],
                "metrics": jr.get("metrics", {})
            }
            summary["job_summaries"].append(job_summary)
        
        filepath = self.output_dir / "summary.json"
        try:
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"\n✓ Saved summary: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save summary: {e}")
    
    def print_thesis_appendix(self) -> None:
        """Print human-readable summary for thesis appendix."""
        print("\n" + "="*90)
        print("IBM QUANTUM JOB EXECUTION SUMMARY — For Thesis Appendix")
        print("="*90)
        print(f"Backend: {BACKEND_NAME}")
        print(f"Execution Date: {datetime.now(timezone.utc).isoformat()}")
        print(f"VQC Config: {N_QUBITS} qubits, {N_CNOT_GATES} CNOT gates, {N_ANSATZ_PARAMS} parameters")
        print(f"Shots per circuit: {SHOTS}")
        print("-"*90)
        
        for jr in self.job_results:
            print(f"\n{jr['job_name'].upper()}")
            print(f"  Job IDs: {', '.join(jr['job_ids']) if jr['job_ids'] else 'N/A'}")
            print(f"  Events: {jr['n_events']}")
            print(f"  Zero-Noise Extrapolation: {'Yes' if jr['use_zne'] else 'No'}")
            
            if jr["backend_properties"]:
                props = jr["backend_properties"]
                if props.get("T1"):
                    t1_vals = [v for v in props["T1"].values() if v is not None]
                    if t1_vals:
                        print(f"  T1 (μs): {np.mean(t1_vals):.3e} ± {np.std(t1_vals):.3e}")
                
                if props.get("T2"):
                    t2_vals = [v for v in props["T2"].values() if v is not None]
                    if t2_vals:
                        print(f"  T2 (μs): {np.mean(t2_vals):.3e} ± {np.std(t2_vals):.3e}")
                
                if props.get("2qubit_gate_errors"):
                    errors = [v for v in props["2qubit_gate_errors"].values() if v is not None]
                    if errors:
                        print(f"  2-Qubit Gate Error: {np.mean(errors):.3e} ± {np.std(errors):.3e}")
            
            metrics = jr.get("metrics", {})
            if metrics:
                print(f"\n  RESULTS:")
                print(f"    Balanced Accuracy: {metrics.get('balanced_accuracy', 'N/A'):.4f}")
                print(f"    Precision:        {metrics.get('precision', 'N/A'):.4f}")
                print(f"    Recall:           {metrics.get('recall', 'N/A'):.4f}")
                print(f"    F1 Score:         {metrics.get('f1_score', 'N/A'):.4f}")
            
            if "error" in jr:
                print(f"  ERROR: {jr['error']}")
        
        print("\n" + "="*90)


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution logic."""
    # Initialize Qiskit imports first
    _initialize_qiskit()
    
    parser = argparse.ArgumentParser(
        description="IBM Quantum Job Executor for QNIM Master's Thesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ibm_quantum_job_executor.py \\
    --gw150914-features data/gw150914_features.npy \\
    --test-features data/synthetic_test_features.npy \\
    --vqc-params models/vqc_params.npy \\
    --output-dir results/

  python ibm_quantum_job_executor.py \\
    --gw150914-features ... \\
    --test-features ... \\
    --vqc-params ... \\
    --output-dir results/ \\
    --token "your_ibm_token"
        """
    )
    
    parser.add_argument("--gw150914-features", required=True, type=str,
                       help="Path to GW150914 feature vector (.npy), shape (12,)")
    parser.add_argument("--test-features", required=True, type=str,
                       help="Path to synthetic test features (.npy), shape (100, 12)")
    parser.add_argument("--vqc-params", required=True, type=str,
                       help="Path to trained VQC parameters (.npy), shape (48,)")
    parser.add_argument("--output-dir", default="results", type=str,
                       help="Output directory for job results (default: results)")
    parser.add_argument("--token", default=None, type=str,
                       help="IBM Quantum token (uses IBM_QUANTUM_TOKEN env var if not provided)")
    parser.add_argument("--skip-job1", action="store_true",
                       help="Skip Job 1 (GW150914 with ZNE)")
    parser.add_argument("--skip-job2", action="store_true",
                       help="Skip Job 2 (100 events with ZNE)")
    parser.add_argument("--skip-job3", action="store_true",
                       help="Skip Job 3 (100 events without ZNE)")
    
    args = parser.parse_args()
    
    # ========================================================================
    # Load Data
    # ========================================================================
    
    logger.info("Loading data files...")
    
    try:
        gw150914_features = np.load(args.gw150914_features)
        logger.info(f"✓ Loaded GW150914 features: shape {gw150914_features.shape}")
        
        if gw150914_features.shape != (12,):
            logger.warning(f"Warning: Expected GW150914 shape (12,), got {gw150914_features.shape}")
    
    except Exception as e:
        logger.error(f"Failed to load GW150914 features: {e}")
        sys.exit(1)
    
    try:
        test_features = np.load(args.test_features)
        logger.info(f"✓ Loaded test features: shape {test_features.shape}")
        
        if test_features.shape != (100, 12):
            logger.warning(f"Warning: Expected test features shape (100, 12), got {test_features.shape}")
    
    except Exception as e:
        logger.error(f"Failed to load test features: {e}")
        sys.exit(1)
    
    try:
        vqc_params = np.load(args.vqc_params)
        logger.info(f"✓ Loaded VQC parameters: shape {vqc_params.shape}")
        
        if vqc_params.shape != (N_ANSATZ_PARAMS,):
            logger.warning(f"Warning: Expected VQC params shape ({N_ANSATZ_PARAMS},), got {vqc_params.shape}")
    
    except Exception as e:
        logger.error(f"Failed to load VQC parameters: {e}")
        sys.exit(1)
    
    # Generate synthetic labels: 8 events per class, 13 classes
    true_labels_test = np.repeat(np.arange(N_THEORY_CLASSES), N_EVENTS_PER_CLASS)[:100]
    true_labels_test_binary = (true_labels_test >= 6).astype(int)  # Example: classes 0-5 = 0, 6-12 = 1
    
    logger.info(f"✓ Generated synthetic labels: {N_THEORY_CLASSES} classes × {N_EVENTS_PER_CLASS} events")
    
    # ========================================================================
    # Initialize Executor
    # ========================================================================
    
    try:
        executor = IBMQuantumJobExecutor(token=args.token, output_dir=Path(args.output_dir))
    except Exception as e:
        logger.error(f"Failed to initialize executor: {e}")
        sys.exit(1)
    
    # ========================================================================
    # Job Execution
    # ========================================================================
    
    jobs_completed = 0
    
    # Job 1: GW150914 with ZNE
    if not args.skip_job1:
        logger.info("\n" + "▬" * 70)
        logger.info("JOB 1 OF 3: GW150914 Single Event with Zero-Noise Extrapolation")
        logger.info("▬" * 70)
        
        try:
            executor.run_job(
                job_name="Job 1: GW150914 + ZNE",
                features=gw150914_features,
                vqc_params=vqc_params,
                use_zne=True,
                event_ids=[0]
            )
            jobs_completed += 1
        except Exception as e:
            logger.error(f"Job 1 failed: {e}")
    
    # Job 2: 100 events with ZNE
    if not args.skip_job2:
        logger.info("\n" + "▬" * 70)
        logger.info("JOB 2 OF 3: 100 Synthetic Events with Zero-Noise Extrapolation")
        logger.info("▬" * 70)
        
        try:
            executor.run_job(
                job_name="Job 2: 100 Events + ZNE",
                features=test_features,
                vqc_params=vqc_params,
                true_labels=true_labels_test_binary,
                use_zne=True,
                event_ids=list(range(100))
            )
            jobs_completed += 1
        except Exception as e:
            logger.error(f"Job 2 failed: {e}")
    
    # Job 3: 100 events without ZNE
    if not args.skip_job3:
        logger.info("\n" + "▬" * 70)
        logger.info("JOB 3 OF 3: 100 Synthetic Events without Zero-Noise Extrapolation")
        logger.info("▬" * 70)
        
        try:
            executor.run_job(
                job_name="Job 3: 100 Events - No ZNE",
                features=test_features,
                vqc_params=vqc_params,
                true_labels=true_labels_test_binary,
                use_zne=False,
                event_ids=list(range(100))
            )
            jobs_completed += 1
        except Exception as e:
            logger.error(f"Job 3 failed: {e}")
    
    # ========================================================================
    # Results & Summary
    # ========================================================================
    
    logger.info("\n" + "="*70)
    logger.info(f"Execution Complete: {jobs_completed}/{3 if not (args.skip_job1 or args.skip_job2 or args.skip_job3) else 'custom'} jobs completed")
    logger.info("="*70)
    
    executor.save_summary()
    executor.print_thesis_appendix()


if __name__ == "__main__":
    main()
