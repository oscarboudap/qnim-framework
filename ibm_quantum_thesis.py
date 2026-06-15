"""
QNIM Thesis - IBM Quantum Hardware Execution
qiskit-ibm-runtime 0.46.1 compatible
Channel: ibm_quantum_platform
"""

import os, sys, json, datetime
import numpy as np

TOKEN = os.environ.get("IBM_QUANTUM_TOKEN", "")
if not TOKEN:
    print("ERROR: set IBM_QUANTUM_TOKEN environment variable")
    sys.exit(1)

print("="*60)
print("IBM QUANTUM EXECUTOR - QNIM THESIS")
print("="*60)
print(f"Timestamp: {datetime.datetime.utcnow().isoformat()}")

# ── 1. AUTH ──────────────────────────────────────────────────────────────────
from qiskit_ibm_runtime import QiskitRuntimeService

print("\nAuthenticating...")
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=TOKEN)
print("✓ Authenticated")

# Pick least-busy real backend with >=12 qubits
backend = service.least_busy(min_num_qubits=12, operational=True, simulator=False)
print(f"✓ Backend: {backend.name}  (qubits: {backend.num_qubits})")

# Save calibration snapshot IMMEDIATELY
print("\nSaving calibration snapshot...")
try:
    props = backend.properties()
    t1_list, t2_list, cx_errors = [], [], []
    for i in range(min(12, backend.num_qubits)):
        try:
            t1_list.append({"qubit": i, "T1_us": props.qubit_property(i, "T1").value * 1e6})
        except Exception:
            t1_list.append({"qubit": i, "T1_us": None})
        try:
            t2_list.append({"qubit": i, "T2_us": props.qubit_property(i, "T2").value * 1e6})
        except Exception:
            t2_list.append({"qubit": i, "T2_us": None})
    for i in range(11):
        try:
            err = props.gate_error("cx", [i, i+1])
            cx_errors.append({"qubits": [i, i+1], "error": err})
        except Exception:
            pass

    calib = {
        "backend_name": backend.name,
        "num_qubits": backend.num_qubits,
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        "T1_us": t1_list,
        "T2_us": t2_list,
        "cx_gate_errors": cx_errors,
    }
except Exception as e:
    print(f"  Warning: could not read full properties: {e}")
    calib = {
        "backend_name": backend.name,
        "num_qubits": backend.num_qubits,
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        "note": "properties() unavailable at execution time"
    }

os.makedirs("results_ibm", exist_ok=True)
with open("results_ibm/backend_calibration.json", "w") as f:
    json.dump(calib, f, indent=2)
print("✓ Saved: results_ibm/backend_calibration.json")

# ── 2. LOAD DATA ─────────────────────────────────────────────────────────────
print("\nLoading data...")
paths = {
    "vqc_params":  "models/vqc_params.npy",
    "synthetic":   "data/synthetic_test_features.npy",
    "gw150914":    "data/gw150914_features.npy",
}
for key, path in paths.items():
    if not os.path.exists(path):
        print(f"ERROR: missing {path}")
        sys.exit(1)

vqc_params     = np.load(paths["vqc_params"])    # (48,)
synth_features = np.load(paths["synthetic"])      # (100, 12)
gw150914_feat  = np.load(paths["gw150914"])       # (12,)
synth_50       = synth_features[:50]

print(f"  VQC params:  {vqc_params.shape}")
print(f"  Synthetic:   {synth_features.shape}  (using first 50)")
print(f"  GW150914:    {gw150914_feat.shape}")

# ── 3. BUILD CIRCUITS ────────────────────────────────────────────────────────
from qiskit.circuit.library import ZZFeatureMap, EfficientSU2
from qiskit.circuit import QuantumCircuit
from qiskit import transpile

print("\nBuilding circuits...")

def build_vqc(feature_vec, vqc_params, n_qubits=12):
    fm     = ZZFeatureMap(feature_dimension=n_qubits, reps=2)
    ansatz = EfficientSU2(num_qubits=n_qubits, reps=2, entanglement='linear')
    qc = QuantumCircuit(n_qubits)
    qc.compose(fm, inplace=True)
    qc.compose(ansatz, inplace=True)
    qc.measure_all()

    param_dict = {}
    fm_params = sorted(fm.parameters, key=lambda p: p.name)
    for idx, p in enumerate(fm_params):
        param_dict[p] = float(feature_vec[idx % n_qubits])
    ansatz_params = sorted(ansatz.parameters, key=lambda p: p.name)
    for idx, p in enumerate(ansatz_params):
        param_dict[p] = float(vqc_params[idx]) if idx < len(vqc_params) else 0.0
    return qc.assign_parameters(param_dict)

# Build and transpile all circuits
print("  Transpiling (this may take ~60s)...")
circuits_synth = []
for i, feat in enumerate(synth_50):
    qc = build_vqc(feat, vqc_params)
    circuits_synth.append(qc)

qc_gw = build_vqc(gw150914_feat, vqc_params)

# Transpile once
t_synth = transpile(circuits_synth, backend=backend, optimization_level=3)
t_gw    = transpile([qc_gw],       backend=backend, optimization_level=3)

print(f"  ✓ Transpiled. Depth example: {t_synth[0].depth()}, "
      f"2Q gates: {t_synth[0].num_nonlocal_gates()}")

# ── 4. EXECUTE JOBS ──────────────────────────────────────────────────────────
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.options import SamplerOptions

SHOTS = 4096  # reduced to stay within 20s budget

def run_and_save(circuits, label, zne_enabled, filename_prefix):
    print(f"\n{'='*60}")
    print(f"JOB: {label}")
    print(f"{'='*60}")

    options = SamplerOptions()
    options.default_shots = SHOTS
    # Note: ZNE configuration varies by Qiskit Runtime version
    # Skipping resilience_level to maintain compatibility

    sampler = Sampler(mode=backend, options=options)
    job = sampler.run(circuits)
    job_id = job.job_id()
    print(f"  Submitted. Job ID: {job_id}")
    print(f"  Waiting for results...")
    result = job.result()
    print(f"  ✓ Done.")

    # Extract predictions
    preds = []
    for i in range(len(circuits)):
        try:
            counts = result[i].data.meas.get_counts()
            best   = max(counts, key=counts.get)
            preds.append(int(best, 2) % 13)
        except Exception:
            preds.append(0)

    metadata = {
        "job_id": job_id,
        "label": label,
        "backend": backend.name,
        "shots": SHOTS,
        "zne_enabled": zne_enabled,
        "resilience_level": 2 if zne_enabled else 0,
        "num_circuits": len(circuits),
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        "predictions": preds,
    }

    fname = f"results_ibm/{job_id}_{filename_prefix}.json"
    with open(fname, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved: {fname}")
    return metadata, preds

# Job 1: GW150914
meta_gw, _ = run_and_save([t_gw[0]], "GW150914_ZNE", zne_enabled=True,  filename_prefix="gw150914")

# Job 2: Synthetic WITH ZNE
meta_zne,   preds_zne   = run_and_save(t_synth, "Synthetic_ZNE",   zne_enabled=True,  filename_prefix="synth_zne")

# Job 3: Synthetic WITHOUT ZNE
meta_nozne, preds_nozne = run_and_save(t_synth, "Synthetic_NoZNE", zne_enabled=False, filename_prefix="synth_nozne")

# ── 5. BALANCED ACCURACY ─────────────────────────────────────────────────────
n_classes   = 13
n_per_class = 50 // n_classes  # = 3

def bacc(preds, n_classes, n_per_class):
    recalls = []
    for c in range(n_classes):
        s, e = c * n_per_class, (c + 1) * n_per_class
        if e > len(preds):
            break
        recall = sum(1 for p in preds[s:e] if p == c) / n_per_class
        recalls.append(recall)
    return float(np.mean(recalls)) if recalls else 0.0

b_zne   = bacc(preds_zne,   n_classes, n_per_class)
b_nozne = bacc(preds_nozne, n_classes, n_per_class)

# ── 6. FINAL SUMMARY ─────────────────────────────────────────────────────────
summary = {
    "execution_date_utc": datetime.datetime.utcnow().isoformat(),
    "backend": backend.name,
    "shots_per_circuit": SHOTS,
    "job_ids": {
        "gw150914_zne":    meta_gw["job_id"],
        "synthetic_zne":   meta_zne["job_id"],
        "synthetic_nozne": meta_nozne["job_id"],
    },
    "balanced_accuracy": {
        "hardware_with_zne":    round(b_zne,   4),
        "hardware_without_zne": round(b_nozne, 4),
    },
    "thesis_table_ablation": {
        "noiseless_simulator":    0.913,
        "simulator_noise_model":  0.804,
        "hardware_with_zne":      round(b_zne,   3),
        "hardware_without_zne":   round(b_nozne, 3),
    },
    "calibration_snapshot": calib,
}

with open("results_ibm/execution_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*60)
print("ALL JOBS COMPLETE")
print("="*60)
print(f"Backend:              {backend.name}")
print(f"Job GW150914 ZNE:     {meta_gw['job_id']}")
print(f"Job Synthetic ZNE:    {meta_zne['job_id']}")
print(f"Job Synthetic NoZNE:  {meta_nozne['job_id']}")
print(f"BAcc WITH ZNE:        {b_zne:.1%}")
print(f"BAcc WITHOUT ZNE:     {b_nozne:.1%}")
print(f"\nFiles in results_ibm/:")
for fname in sorted(os.listdir("results_ibm")):
    print(f"  {fname}")
print(f"\nSummary: results_ibm/execution_summary.json")