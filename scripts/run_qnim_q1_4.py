"""Compare exact QNIM marginal, correlation, local-Pauli, and PCA probes."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_qnim_q1 import (
    DEFAULT_REFERENCE,
    N_QUIBITS,
    _architecture_metadata,
    _write_json_atomically,
)
from scripts.run_qnim_q1_1 import DATASET_SEED, MODEL_SEEDS
from scripts.run_qnim_q1_2 import (
    DEFAULT_SMOKE_REFERENCE,
    _build_loss,
    _read_dataset,
)
from scripts.run_qnim_q1_3 import (
    DEFAULT_Q1_2_REPORT,
    DEFAULT_OUTPUT as DEFAULT_Q1_3_REPORT,
    _circuit_parameter_sets,
    _fit_probe_models,
    _read_q1_2_report,
)

DEFAULT_OUTPUT = ROOT / "results" / "scaling_study" / "qnim_q1_4_v1.json"
DEFAULT_SMOKE_OUTPUT = ROOT / "results" / "scaling_study" / "qnim_q1_4_smoke_v1.json"
CLASSIFIERS = ("logistic_regression", "svm_rbf")
VARIANTS = (
    "A_marginals",
    "B_adjacent_zz",
    "C_all_zz",
    "D_local_xyz",
)
N_CLASSES = 13
SCHEMA_VERSION = 1


def _sha256(array):
    return hashlib.sha256(
        np.asarray(array, dtype=np.float64).tobytes()
    ).hexdigest().upper()


def _read_q1_3_report(path, q1_2_sha256):
    content = Path(path).read_bytes()
    report = json.loads(content)
    if report.get("experiment_id") != "QNIM_Q1_3_V2":
        raise ValueError("Q1.4 requires the completed Q1.3 v2 report")
    if report.get("status") != "completed":
        raise ValueError("Q1.3 must be completed before Q1.4")
    if report.get("q1_2_report_sha256") != q1_2_sha256:
        raise ValueError("Q1.3 was not produced from the supplied Q1.2 report")
    if report.get("final_test", {}).get("status") != "locked_not_evaluated":
        raise ValueError("Q1.4 requires the Q1.3 final test to remain locked")
    return report, hashlib.sha256(content).hexdigest().upper()


def _build_representations(observables):
    p0 = np.asarray(observables["p0"], dtype=float)
    zz = np.asarray(observables["zz"], dtype=float)
    xyz = np.asarray(observables["xyz"], dtype=float)
    pairs = list(combinations(range(N_QUIBITS), 2))
    expected = {
        "p0": (len(p0), N_QUIBITS),
        "zz": (len(p0), len(pairs)),
        "xyz": (len(p0), 3 * N_QUIBITS),
    }
    if p0.shape != expected["p0"] or zz.shape != expected["zz"] or xyz.shape != expected["xyz"]:
        raise ValueError("Exact Q1.4 observables have unexpected dimensions")
    if not all(np.isfinite(values).all() for values in (p0, zz, xyz)):
        raise ValueError("Exact Q1.4 observables contain non-finite values")

    pair_indices = {pair: index for index, pair in enumerate(pairs)}
    adjacent_indices = [
        pair_indices[(qubit, qubit + 1)]
        for qubit in range(N_QUIBITS - 1)
    ]
    representations = {
        "A_marginals": p0,
        "B_adjacent_zz": np.column_stack((p0, zz[:, adjacent_indices])),
        "C_all_zz": np.column_stack((p0, zz)),
        "D_local_xyz": xyz,
    }
    expected_dimensions = {
        "A_marginals": 12,
        "B_adjacent_zz": 23,
        "C_all_zz": 78,
        "D_local_xyz": 36,
    }
    for name, features in representations.items():
        if features.shape[1] != expected_dimensions[name]:
            raise RuntimeError(f"{name} feature count differs from the protocol")
    return representations


def _extract_representations(loss_fn, X_train, X_val, parameters):
    started = time.perf_counter()
    train_observables = loss_fn.measure_quantum_observables_exact(
        X_train, parameters
    )
    validation_observables = loss_fn.measure_quantum_observables_exact(
        X_val, parameters
    )
    elapsed = time.perf_counter() - started
    train = _build_representations(train_observables)
    validation = _build_representations(validation_observables)
    return {
        name: {"train": train[name], "validation": validation[name]}
        for name in VARIANTS
    }, elapsed


def _validate_q1_3_marginals(q1_3_report, seed, condition, representations):
    run = next(
        (
            item
            for item in q1_3_report["experiments"]["frozen_circuit_probes"]
            if item["initialization_seed"] == seed
            and item["circuit_condition"] == condition
        ),
        None,
    )
    if run is None:
        raise ValueError(
            f"Q1.3 has no marginal reference for seed {seed}, {condition}"
        )
    if (
        _sha256(representations["A_marginals"]["train"])
        != run["train_feature_sha256"]
        or _sha256(representations["A_marginals"]["validation"])
        != run["validation_feature_sha256"]
    ):
        raise ValueError(
            f"Q1.4 marginal control does not reproduce Q1.3 for seed {seed}, {condition}"
        )


def _validate_checkpoint(report, expected):
    if report.get("status") != "in_progress":
        raise ValueError("Only an in-progress Q1.4 report can be resumed")
    for key in (
        "checkpoint_schema_version",
        "experiment_id",
        "q1_2_report_sha256",
        "q1_3_report_sha256",
        "classical_reference_id",
        "classical_reference_sha256",
        "dataset_seed",
        "model_initialization_seeds",
        "dataset_hashes",
        "class_names",
        "architecture",
        "configuration",
        "software_versions",
        "validation_policy",
        "final_test",
    ):
        if report.get(key) != expected.get(key):
            raise ValueError(f"Cannot resume Q1.4: {key} differs from checkpoint")
    controls = report.get("experiments", {}).get("raw_pca_control")
    probes = report.get("experiments", {}).get("circuit_probes")
    if not isinstance(controls, list) or not isinstance(probes, list):
        raise ValueError("Cannot resume Q1.4: invalid checkpoints")
    control_seeds = [run.get("initialization_seed") for run in controls]
    if (
        len(control_seeds) != len(set(control_seeds))
        or any(seed not in expected["model_initialization_seeds"] for seed in control_seeds)
        or any(run.get("status") != "completed" for run in controls)
    ):
        raise ValueError("Cannot resume Q1.4: invalid raw PCA control state")
    probe_keys = [
        (run.get("initialization_seed"), run.get("circuit_condition"))
        for run in probes
    ]
    if (
        len(probe_keys) != len(set(probe_keys))
        or any(seed not in expected["model_initialization_seeds"] for seed, _ in probe_keys)
        or any(condition not in {"initial", "trained"} for _, condition in probe_keys)
        or any(run.get("status") != "completed" for run in probes)
        or any(set(run.get("variants", {})) != set(VARIANTS) for run in probes)
    ):
        raise ValueError("Cannot resume Q1.4: invalid circuit probe state")


def run_q1_4(
    q1_2_report_path=DEFAULT_Q1_2_REPORT,
    q1_3_report_path=DEFAULT_Q1_3_REPORT,
    reference_manifest=DEFAULT_REFERENCE,
    output_path=DEFAULT_OUTPUT,
    adapter_factory=None,
    resume=False,
    smoke=False,
    smoke_reference_path=DEFAULT_SMOKE_REFERENCE,
):
    q1_2_report_path = Path(q1_2_report_path)
    q1_3_report_path = Path(q1_3_report_path)
    output_path = Path(output_path)
    if smoke and output_path == DEFAULT_OUTPUT:
        output_path = DEFAULT_SMOKE_OUTPUT
    if output_path.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite Q1.4 results: {output_path}")
    if resume and not output_path.exists():
        raise FileNotFoundError(f"Q1.4 checkpoint does not exist: {output_path}")

    q1_2_report, q1_2_sha256 = _read_q1_2_report(q1_2_report_path)
    q1_3_report, q1_3_sha256 = _read_q1_3_report(
        q1_3_report_path, q1_2_sha256
    )
    seeds = (DATASET_SEED,) if smoke else MODEL_SEEDS
    parameter_sets = _circuit_parameter_sets(q1_2_report, seeds)
    _, reference_sha256, reference_id, expected_dataset, split, dataset_seeds = (
        _read_dataset(
            smoke,
            reference_manifest,
            smoke_reference_path,
            adapter_factory,
        )
    )
    if not smoke:
        if reference_sha256 != q1_2_report["classical_reference_sha256"]:
            raise ValueError("Q1.2 and Q1.4 classical reference hashes differ")
        if (
            q1_2_report["dataset_hashes"] != q1_3_report["dataset_hashes"]
            or q1_2_report["dataset_hashes"]["train_sha256"]
            != expected_dataset["train_sha256"]
            or q1_2_report["dataset_hashes"]["validation_sha256"]
            != expected_dataset["validation_sha256"]
        ):
            raise ValueError("Q1.2, Q1.3, and Q1.4 split hashes differ")
        if reference_sha256 != q1_3_report["classical_reference_sha256"]:
            raise ValueError("Q1.3 and Q1.4 classical reference hashes differ")
    if dataset_seeds != seeds:
        raise ValueError("Q1.4 model seeds do not match the dataset mode")
    if list(expected_dataset["class_names"]) != q1_2_report["class_names"]:
        raise ValueError("Q1.2 and Q1.4 class labels differ")

    X_train, y_train, X_val, y_val = split
    loss_fn = _build_loss(X_train, y_train, shots=512)
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "qiskit": importlib.metadata.version("qiskit"),
        "qiskit_aer": importlib.metadata.version("qiskit-aer"),
    }
    configuration = {
        "smoke_mode": smoke,
        "measurement": "exact statevector observables; no shots",
        "n_qubits": N_QUIBITS,
        "feature_map": "ZZFeatureMap",
        "feature_map_reps": 2,
        "ansatz": "EfficientSU2",
        "ansatz_reps": 2,
        "entanglement": "linear",
        "circuit_conditions": ["initial", "trained"],
        "variants": {
            "A_marginals": {
                "features": "P(q_i=0), qubits 0..11",
                "count": 12,
            },
            "B_adjacent_zz": {
                "features": "P(q_i=0) plus <Z_i Z_(i+1)>",
                "pairs": [[i, i + 1] for i in range(N_QUIBITS - 1)],
                "count": 23,
            },
            "C_all_zz": {
                "features": "P(q_i=0) plus all <Z_i Z_j>, i<j",
                "pairs": [list(pair) for pair in combinations(range(N_QUIBITS), 2)],
                "count": 78,
            },
            "D_local_xyz": {
                "features": "<X_i>, <Y_i>, <Z_i> interleaved by qubit",
                "count": 36,
            },
        },
        "classifiers": {
            "logistic_regression": {
                "standard_scaler": True,
                "max_iter": 2000,
                "class_weight": "balanced",
            },
            "svm_rbf": {
                "standard_scaler": True,
                "kernel": "rbf",
                "class_weight": "balanced",
                "probability": True,
                "random_state": "initialization seed",
            },
        },
        "raw_pca_control": {
            "features": "the same 12 train-fitted PCA components",
            "validation_used_for_fitting": False,
        },
    }
    report = {
        "checkpoint_schema_version": SCHEMA_VERSION,
        "experiment_id": "QNIM_Q1_4_SMOKE_V1" if smoke else "QNIM_Q1_4_V1",
        "experiment_scope": (
            "integration smoke only; not scientific evidence"
            if smoke else "predefined exact quantum observable readout comparison"
        ),
        "status": "in_progress",
        "active_run": None,
        "q1_2_report_sha256": q1_2_sha256,
        "q1_3_report_sha256": q1_3_sha256,
        "classical_reference_id": reference_id,
        "classical_reference_sha256": reference_sha256,
        "dataset_seed": DATASET_SEED,
        "model_initialization_seeds": list(seeds),
        "dataset_hashes": {
            "train_sha256": expected_dataset["train_sha256"],
            "validation_sha256": expected_dataset["validation_sha256"],
        },
        "class_names": list(expected_dataset["class_names"]),
        "architecture": _architecture_metadata(),
        "configuration": configuration,
        "software_versions": versions,
        "validation_policy": {
            "validation_used_for_fitting": False,
            "validation_used_for_metrics_only": True,
            "final_test": "locked_not_evaluated",
        },
        "experiments": {
            "raw_pca_control": [],
            "circuit_probes": [],
        },
        "final_test": {
            "evaluated": False,
            "status": "locked_not_evaluated",
            "models": {},
        },
    }

    if resume:
        expected = report
        report = json.loads(output_path.read_text(encoding="utf-8"))
        _validate_checkpoint(report, expected)
        report["active_run"] = None
        _write_json_atomically(output_path, report)
    else:
        _write_json_atomically(output_path, report)

    completed_controls = {
        run["initialization_seed"]
        for run in report["experiments"]["raw_pca_control"]
    }
    for seed in seeds:
        if seed in completed_controls:
            continue
        report["active_run"] = {"condition": "raw_pca_control", "seed": seed}
        _write_json_atomically(output_path, report)
        report["experiments"]["raw_pca_control"].append({
            "status": "completed",
            "initialization_seed": seed,
            "train_features_sha256": _sha256(X_train),
            "validation_features_sha256": _sha256(X_val),
            "classifiers": _fit_probe_models(
                X_train,
                y_train,
                X_val,
                y_val,
                seed,
                model_names=CLASSIFIERS,
            ),
        })
        report["active_run"] = None
        _write_json_atomically(output_path, report)

    completed_probes = {
        (run["initialization_seed"], run["circuit_condition"])
        for run in report["experiments"]["circuit_probes"]
    }
    for seed in seeds:
        for condition in ("initial", "trained"):
            key = (seed, condition)
            if key in completed_probes:
                continue
            report["active_run"] = {
                "initialization_seed": seed,
                "circuit_condition": condition,
            }
            _write_json_atomically(output_path, report)
            started = time.perf_counter()
            representations, extraction_seconds = _extract_representations(
                loss_fn,
                X_train,
                X_val,
                parameter_sets[seed][condition],
            )
            if not smoke:
                _validate_q1_3_marginals(
                    q1_3_report, seed, condition, representations
                )
            variants = {}
            for name in VARIANTS:
                train_features = representations[name]["train"]
                validation_features = representations[name]["validation"]
                variants[name] = {
                    "feature_count": train_features.shape[1],
                    "train_features_sha256": _sha256(train_features),
                    "validation_features_sha256": _sha256(validation_features),
                    "classifiers": _fit_probe_models(
                        train_features,
                        y_train,
                        validation_features,
                        y_val,
                        seed,
                        model_names=CLASSIFIERS,
                    ),
                }
            report["experiments"]["circuit_probes"].append({
                "status": "completed",
                "initialization_seed": seed,
                "circuit_condition": condition,
                "ansatz_parameters_sha256": _sha256(
                    parameter_sets[seed][condition]
                ),
                "feature_extraction_seconds": extraction_seconds,
                "elapsed_seconds": time.perf_counter() - started,
                "variants": variants,
            })
            report["active_run"] = None
            _write_json_atomically(output_path, report)
    report["status"] = "completed"
    _write_json_atomically(output_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Compare exact QNIM marginal, correlation, XYZ, and PCA probes."
    )
    parser.add_argument("--q1-2-report", type=Path, default=DEFAULT_Q1_2_REPORT)
    parser.add_argument("--q1-3-report", type=Path, default=DEFAULT_Q1_3_REPORT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-reference", type=Path, default=DEFAULT_SMOKE_REFERENCE)
    args = parser.parse_args()
    output = args.output or (DEFAULT_SMOKE_OUTPUT if args.smoke else DEFAULT_OUTPUT)
    report = run_q1_4(
        q1_2_report_path=args.q1_2_report,
        q1_3_report_path=args.q1_3_report,
        reference_manifest=args.reference,
        output_path=output,
        resume=args.resume,
        smoke=args.smoke,
        smoke_reference_path=args.smoke_reference,
    )
    print(json.dumps({
        "status": report["status"],
        "output": str(output),
        "final_test": report["final_test"]["status"],
        "raw_pca_runs": len(report["experiments"]["raw_pca_control"]),
        "circuit_runs": len(report["experiments"]["circuit_probes"]),
    }, indent=2))


if __name__ == "__main__":
    main()
