"""Preflight and explicitly gated runner for QNIM_IBM_QPU_PILOT_V1."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.infrastructure.qnim_ibm_qpu_pilot import (
    PILOT_PROTOCOL_PATH,
    DurableJobRecordStore,
    PilotPreflightError,
    estimate_resources,
    find_connected_linear_layouts,
    load_protocol,
    load_train_validation_splits,
    local_preflight,
    parameter_sha256,
    query_ibm_preflight,
    require_submission_enabled,
    safe_instance_metadata,
    save_parameter_checkpoint,
    sha256_file,
    validate_connected_layout,
    verify_parameter_checkpoint,
    verify_dataset_provenance,
    write_json_metadata,
)


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _parse_layout(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Layout must be comma-separated physical-qubit indices"
        ) from exc


def _repository_dataset_path(protocol: dict[str, Any]) -> Path:
    return REPOSITORY_ROOT / protocol["dataset"]["relative_path"]


def _preflight_report(protocol: dict[str, Any]) -> dict[str, Any]:
    report = local_preflight(protocol)
    report["protocol_path"] = str(PILOT_PROTOCOL_PATH.relative_to(REPOSITORY_ROOT))
    report["protocol_sha256"] = sha256_file(PILOT_PROTOCOL_PATH)
    report["submission_enabled"] = protocol["submission_enabled"]
    return report


def _backend_calibration_metadata(backend: Any) -> dict[str, Any]:
    status_obj = backend.status()
    report: dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "backend_name": str(backend.name),
        "backend_version": getattr(backend, "backend_version", None),
        "operational": getattr(status_obj, "operational", None),
        "pending_jobs": getattr(status_obj, "pending_jobs", None),
        "status_message": getattr(status_obj, "status_msg", None),
    }
    properties = backend.properties()
    if properties is None:
        raise PilotPreflightError("Backend calibration properties are unavailable")
    report["last_update_date"] = str(properties.last_update_date)
    report["qubits"] = [
        [
            {
                "name": item.name,
                "value": item.value,
                "unit": item.unit,
                "date": str(item.date),
            }
            for item in qubit
        ]
        for qubit in properties.qubits
    ]
    report["gates"] = [
        {
            "name": gate.gate,
            "qubits": list(gate.qubits),
            "parameters": [
                {
                    "name": item.name,
                    "value": item.value,
                    "unit": item.unit,
                    "date": str(item.date),
                }
                for item in gate.parameters
            ],
        }
        for gate in properties.gates
    ]
    return report


def _execute_pilot(
    protocol: dict[str, Any],
    *,
    backend_name: str,
    physical_layout: tuple[int, ...],
) -> dict[str, Any]:
    require_submission_enabled(protocol, submit_requested=True)
    preflight = local_preflight(protocol)
    if preflight["installed_versions"]["qiskit"] != protocol["software"]["qiskit"]:
        raise PilotPreflightError("Installed Qiskit does not match the protocol pin")
    if (
        preflight["installed_versions"]["qiskit-ibm-runtime"]
        != protocol["software"]["qiskit-ibm-runtime"]
    ):
        raise PilotPreflightError("Installed IBM Runtime does not match the protocol pin")

    from src.ibm_credentials import get_ibm_quantum_token
    from qiskit_ibm_runtime import QiskitRuntimeService
    from src.infrastructure.qiskit_vqc_trainer import QiskitVQCTrainer

    token = get_ibm_quantum_token()
    if not token:
        raise PilotPreflightError("IBM_QUANTUM_TOKEN is unavailable; no login was attempted")
    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=token,
    )
    backend = service.backend(backend_name)
    account_scope = safe_instance_metadata(service)
    if int(backend.num_qubits) < int(protocol["model"]["logical_qubits"]):
        raise PilotPreflightError("Selected backend has fewer than eight qubits")
    coupling = getattr(backend, "coupling_map", None)
    get_edges = getattr(coupling, "get_edges", None)
    if not callable(get_edges):
        raise PilotPreflightError("Selected backend does not expose a coupling map")
    layout = validate_connected_layout(
        physical_layout,
        get_edges(),
        logical_qubits=int(protocol["model"]["logical_qubits"]),
        backend_qubits=int(backend.num_qubits),
    )
    calibration = _backend_calibration_metadata(backend)
    resources = estimate_resources(protocol)
    dataset_dir = _repository_dataset_path(protocol)
    splits, provenance = load_train_validation_splits(
        dataset_dir, protocol, REPOSITORY_ROOT
    )

    run_id = f"qnim-ibm-pilot-{uuid.uuid4().hex[:12]}"
    store = DurableJobRecordStore(
        REPOSITORY_ROOT / protocol["job_recording"]["directory"],
        max_jobs=int(protocol["resource_budget"]["max_qpu_jobs"]),
        max_shots=int(protocol["resource_budget"]["max_total_qpu_shots"]),
    )
    store.check_budget(
        resources["total_qpu_jobs"],
        resources["total_qpu_shots"],
    )
    run_dir = (
        REPOSITORY_ROOT
        / "results"
        / "scaling_study"
        / "qnim_ibm_qpu_pilot_v1"
        / "runs"
        / run_id
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    protocol_hash = sha256_file(PILOT_PROTOCOL_PATH)
    run_metadata = {
        "experiment_id": protocol["experiment_id"],
        "run_id": run_id,
        "protocol_sha256": protocol_hash,
        "dataset_provenance": provenance,
        "algorithm_identity": protocol["algorithm_identity"],
        "model": protocol["model"],
        "local_pretraining": protocol["local_pretraining"],
        "hardware_spsa": protocol["hardware_spsa"],
        "validation": protocol["validation"],
        "resource_estimate": resources,
        "backend": backend_name,
        "instance": account_scope["instance"],
        "region": account_scope["region"],
        "physical_layout": list(layout),
        "calibration_metadata": calibration,
        "test_accessed": False,
    }
    write_json_metadata(run_dir / "run_manifest.json", run_metadata)
    trainer = QiskitVQCTrainer(
        mode="ibm",
        token=token,
        backend_name=backend_name,
        training_batch_size=int(protocol["local_pretraining"]["batch_size"]),
        hardware_shots=int(protocol["hardware_spsa"]["shots"]),
        hardware_batch_size=int(protocol["hardware_spsa"]["batch_size"]),
        hardware_finetune_steps=int(protocol["hardware_spsa"]["steps"]),
        n_hw_validation=int(protocol["validation"]["examples"]),
        shots_validation=int(protocol["validation"]["shots"]),
        random_seed=int(protocol["local_pretraining"]["seed"]),
        feature_map_reps=int(protocol["model"]["feature_map_reps"]),
        ansatz_reps=int(protocol["model"]["ansatz_reps"]),
        entanglement=str(protocol["model"]["entanglement"]),
        learning_rate=float(protocol["hardware_spsa"]["learning_rate"]),
        reference_shots=int(protocol["hardware_spsa"]["reference_shots"]),
        readout_refit_every=int(protocol["local_pretraining"]["readout_refit_every"]),
        readout_hidden_size=int(protocol["model"]["readout_hidden_size"]),
        readout_mode=str(protocol["model"]["readout_mode"]),
    )
    local_result = trainer.train_vqc(
        X_train=splits.X_train,
        y_train=splits.y_train,
        num_qubits=int(protocol["model"]["logical_qubits"]),
        n_features=int(protocol["model"]["input_dimension"]),
        max_iterations=int(protocol["local_pretraining"]["max_iterations"]),
        shots=int(protocol["local_pretraining"]["shots"]),
        n_feynman_params=int(protocol["local_pretraining"]["n_feynman_params"]),
        initialization_seed=int(protocol["local_pretraining"]["seed"]),
    )
    weights = local_result["weights"]
    checkpoint_metadata = {
        "experiment_id": protocol["experiment_id"],
        "protocol_sha256": protocol_hash,
        "dataset_id": protocol["dataset"]["dataset_id"],
        "dataset_hashes": provenance["file_sha256"],
        "dataset_provenance": provenance,
        "pca_hash": provenance["pca_sha256"],
        "logical_qubits": int(protocol["model"]["logical_qubits"]),
        "optimizer": protocol["algorithm_identity"]["pretraining"],
        "random_seed": int(protocol["local_pretraining"]["seed"]),
    }
    save_parameter_checkpoint(run_dir, "local_pretraining", weights, checkpoint_metadata)

    job_record_metadata = {
        "run_id": run_id,
        "experiment_id": protocol["experiment_id"],
        "protocol_hash": protocol_hash,
        "dataset_hashes": provenance["file_sha256"],
        "dataset_provenance": provenance,
        "pca_hash": provenance["pca_sha256"],
        "parameter_hash": parameter_sha256(weights),
        "logical_qubits": int(protocol["model"]["logical_qubits"]),
        "backend": backend_name,
        "instance": account_scope["instance"],
        "region": account_scope["region"],
        "physical_layout": list(layout),
        "transpiler_configuration": {
            "optimization_level": 1,
            "initial_layout": list(layout),
        },
        "optimizer_state": {
            "optimizer": "HardwareSPSAOptimizer",
            "steps_completed": 0,
        },
        "random_seed": int(protocol["hardware_spsa"]["seed"]),
        "sampling_policy": {
            "batch_size": int(protocol["hardware_spsa"]["batch_size"]),
            "shots": int(protocol["hardware_spsa"]["shots"]),
            "reference_every": int(protocol["hardware_spsa"]["reference_every"]),
        },
        "calibration_metadata": calibration,
    }
    adapted = trainer.fine_tune_on_hardware(
        initial_weights=weights,
        X_train=splits.X_train,
        y_train=splits.y_train,
        n_qubits=int(protocol["model"]["logical_qubits"]),
        n_features=int(protocol["model"]["input_dimension"]),
        steps=int(protocol["hardware_spsa"]["steps"]),
        batch_size=int(protocol["hardware_spsa"]["batch_size"]),
        shots=int(protocol["hardware_spsa"]["shots"]),
        reference_every=int(protocol["hardware_spsa"]["reference_every"]),
        seed=int(protocol["hardware_spsa"]["seed"]),
        physical_layout=layout,
        job_record_store=store,
        job_record_metadata=job_record_metadata,
    )
    save_parameter_checkpoint(
        run_dir,
        "hardware_spsa",
        adapted.parameters,
        {
            **checkpoint_metadata,
            "optimizer": protocol["algorithm_identity"]["hardware_update"],
            "job_ids": adapted.job_ids,
            "steps_completed": adapted.steps_completed,
        },
    )
    validation_metadata = {
        **job_record_metadata,
        "run_id": f"{run_id}-validation",
        "parameter_hash": parameter_sha256(adapted.parameters),
        "optimizer_state": {
            "optimizer": protocol["algorithm_identity"]["hardware_update"],
            "steps_completed": adapted.steps_completed,
        },
    }
    validation_result = trainer._validate_on_ibm(
        adapted.parameters,
        splits,
        int(protocol["model"]["logical_qubits"]),
        int(protocol["model"]["input_dimension"]),
        use_zne=bool(protocol["validation"]["use_zne"]),
        physical_layout=layout,
        job_record_store=store,
        job_record_metadata=validation_metadata,
    )
    verify_parameter_checkpoint(run_dir, "local_pretraining")
    verify_parameter_checkpoint(run_dir, "hardware_spsa")
    return {
        "run_id": run_id,
        "algorithm_identity": protocol["algorithm_identity"],
        "backend": backend_name,
        "physical_layout": list(layout),
        "resources_estimated": resources,
        "local_pretraining": {
            "iterations": local_result["iterations"],
            "objective_calls": local_result["n_objective_function_calls"],
            "weights_sha256": parameter_sha256(weights),
        },
        "hardware_spsa": {
            "steps_completed": adapted.steps_completed,
            "jobs": adapted.jobs,
            "pubs": adapted.pubs,
            "shots": adapted.shots,
            "job_ids": adapted.job_ids,
        },
        "validation": {
            "split": "validation",
            "balanced_accuracy": validation_result[0],
            "balanced_accuracy_zne": validation_result[1],
            "n_examples": int(protocol["validation"]["examples"]),
        },
        "test_accessed": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="Local package and protocol check; no account access")
    parser.add_argument("--query-ibm", action="store_true", help="Explicitly authenticate and query visible backend metadata; submits no job")
    parser.add_argument("--verify-dataset", action="store_true", help="Verify manifest/source/input hashes without loading arrays")
    parser.add_argument("--estimate", action="store_true", help="Print source-derived local and QPU resource estimates")
    parser.add_argument("--execute-pilot", action="store_true", help="Run local pretraining, hardware SPSA, then validation-only evaluation")
    parser.add_argument("--submit", action="store_true", help="Explicit submission authorization; also requires submission_enabled in protocol")
    parser.add_argument("--backend", help="Backend name returned by the explicit IBM preflight")
    parser.add_argument("--layout", type=_parse_layout, help="Ordered connected layout, e.g. 0,1,2,3,4,5,6,7")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.submit and not args.execute_pilot:
        raise PilotPreflightError("--submit is only valid with --execute-pilot")
    protocol = load_protocol()
    reports = []
    if args.preflight:
        reports.append({"local_preflight": _preflight_report(protocol)})
    if args.query_ibm:
        from src.ibm_credentials import get_ibm_quantum_token

        reports.append({"ibm_preflight": query_ibm_preflight(get_ibm_quantum_token())})
    if args.verify_dataset:
        reports.append(
            {
                "dataset_provenance": verify_dataset_provenance(
                    _repository_dataset_path(protocol),
                    protocol,
                    REPOSITORY_ROOT,
                )
            }
        )
    if args.estimate:
        reports.append({"resource_estimate": estimate_resources(protocol)})
    if args.execute_pilot:
        require_submission_enabled(protocol, args.submit)
        if not args.backend or args.layout is None:
            raise PilotPreflightError(
                "Pilot execution requires --backend and an explicitly selected --layout"
            )
        reports.append(
            {
                "pilot_result": _execute_pilot(
                    protocol,
                    backend_name=args.backend,
                    physical_layout=args.layout,
                )
            }
        )
    if not reports:
        _build_parser().print_help()
        return 0
    _print_json(reports)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PilotPreflightError as exc:
        print(f"PRE-FLIGHT FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
