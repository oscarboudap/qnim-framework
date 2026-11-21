# QNIM IBM QPU Pilot V1 — Implementation Report

**Status:** implementation and safe checks complete; hardware submission remains disabled.
**Experiment identity:** `QNIM_IBM_QPU_PILOT_V1`

## 1. Files changed

- Added [qnim_ibm_qpu_pilot.py](../../../src/infrastructure/qnim_ibm_qpu_pilot.py): protocol and provenance verification, resource accounting, connected-layout checks, durable job records, checkpoint integrity, and explicit IBM preflight.
- Added [run_qnim_ibm_qpu_pilot.py](../../../scripts/run_qnim_ibm_qpu_pilot.py): no-account preflight, opt-in account metadata query, provenance/resource checks, and an explicitly gated execution path.
- Added [protocol.json](./protocol.json): isolated q8 pilot identity, data hashes, settings, and hard budgets. `submission_enabled` is `false`.
- Updated [qiskit_vqc_trainer.py](../../../src/infrastructure/qiskit_vqc_trainer.py): ordinary IBM evaluation now consumes validation arrays only; final test evaluation is a separate `authorize_test_access=True` operation. Hardware SPSA and hardware evaluation require a connected layout and durable job-record context.
- Added [test_qnim_ibm_qpu_pilot.py](../../../test/unit/test_qnim_ibm_qpu_pilot.py): isolated sentinel, provenance, metric, budget, identity, layout, persistence, and checkpoint tests.
- Updated [test_experiment_config.py](../../../test/unit/test_experiment_config.py): the existing mocked hardware-SPSA integration test supplies the newly required layout and durable record context.
- Added this report. No V1, V2, V3, Q1.6 dataset, or historical result artifacts were modified.

## 2. Test-set leakage guard evidence

- `train_and_evaluate` trains from `X_train/y_train` and its normal validation path obtains only `X_val/y_val`.
- `_validate_on_ibm` delegates to the validation split. It does not probe for `X_test` or `y_test` (including with `hasattr`, which could invoke sentinel properties).
- The pilot data container, `PilotSplits`, has only train and validation fields and no test attributes.
- `evaluate_test_on_ibm` is a separate method and raises `PermissionError` unless `authorize_test_access=True`; the guard runs before reading either test property.
- Sentinel tests verify local training, hardware fine-tuning orchestration, and validation selection do not access test properties. An additional test checks the explicit final-test gate.

## 3. Data provenance verification

The runner verifies the existing Q1.6 V2 manifest, each train/validation/PCA artifact hash, and all recorded generator-source hashes before loading arrays. The no-account preflight verified:

- Dataset: `QNIM_Q1_6_DATASET_V2_PCA64`.
- Manifest SHA-256: `53a153278ce83722a61a02cd8bff6749c599209fc11202b5e64ed5ee23c05bfd`.
- `strict_physics=true`, `fallback_allowed=false`, waveform source `PyCBC`, approximant `IMRPhenomPv2`.
- The recorded class injectors/parameter ranges, class labels, analytic aLIGO O3 whitening configuration, PyCBC 2.9.0 version, and generator source hashes are retained in the pilot run manifest and job records.
- PCA and scaler fit split are both recorded as training-only; test data was neither generated nor loaded.

**Scientific limitation:** this is a strict PyCBC simulated-injection dataset, not observed detector data, and it has no detector noise. Its metadata records class-level injection definitions/ranges but not a per-event injection-parameter ledger. The pilot therefore cannot be certified as evidence of astrophysical discrimination; the allowed scope is engineering and QPU-feedback validation on simulated injections.

## 4. QPU pilot execution architecture

1. Default preflight reads the protocol, installed package metadata, manifests, and hashes only. It does not authenticate, load the feature arrays, construct circuits, or submit jobs.
2. `--query-ibm` is the explicit account-query operation. It reports account-visible backend names/status, qubit count, target instructions, coupling edges, connected linear 8-qubit layouts, queue information, and detectable Runtime client symbols. It submits no sampler job and does not print credentials.
3. Before execution, the runner requires an enabled protocol, `--submit`, a backend, a supplied ordered layout, installed pinned package versions, verified provenance, and a passing resource estimate.
4. The selected layout is checked against the selected backend's live coupling map and is passed as `initial_layout` for transpilation. No layout is fabricated.
5. Local pretraining runs first. The hardware phase then uses the existing `HardwareSPSAOptimizer`, and validation uses only the configured validation subset.
6. Each sampler call first atomically persists a submission-intent record. The returned job ID is persisted before awaiting results. Ambiguous submissions/completions block automatic retry. Records capture protocol/data/PCA/parameter/circuit hashes, layout, transpiler settings, optimizer/sampling state, calibration data, timestamps, PUBs, shots, result hash, and IBM resource metrics when returned.
7. Checkpoints are written atomically and re-verified by both file hash and parameter hash.

## 5. Explicit optimizer identity

The pilot is **local QNSPSA-EML-Feynman pretraining + IBM `HardwareSPSAOptimizer` SPSA fine-tuning**. The hardware optimizer remains a distinct experimental arm. This implementation does **not** perform remote QNSPSA-EML-Feynman training.

## 6. IBM preflight results

The safe local preflight reported:

| Item | Result |
|---|---|
| Installed Qiskit | 2.3.1; matches declared pin |
| Declared IBM Runtime | 0.46.1 |
| Installed IBM Runtime | Not installed; version/API compatibility unverified |
| Installed Qiskit Aer | 0.17.2; not used or run |
| Installed Qiskit Machine Learning | Not installed |
| Account query | Not performed |
| Backend, instance, plan, quotas, queue, layouts | Unknown; account access was not requested |
| Sampler/QPU jobs submitted | None |

No packages were installed or upgraded. The repository's existing `.venv312` was left unchanged.

## 7. Backend and layout findings

There are no account-verified backend or layout findings. No credentials were queried. A future account preflight must identify an operational backend and return a connected eight-qubit layout; the operator must select and pass one explicitly. Plan capabilities and quotas remain unknown unless IBM exposes them through account-visible metadata.

## 8. Resource and run/job cost tables

Estimates are upper bounds for the configured maximum iterations/steps, derived from the current `QNSPSA-EML-Feynman`, readout-refit, `HardwareSPSAOptimizer`, and validation paths. Early stopping can reduce local pretraining use. ZNE is disabled.

| Phase | Objective/evaluation calls | Sampler runs/jobs | PUBs/circuits | Shots |
|---|---:|---:|---:|---:|
| Local QNSPSA-EML-Feynman objective | 401 | — | — | — |
| Local batches, references, warm start/refit | — | 403 local sampler runs | 3,264 | 916,992 |
| Hardware SPSA plus/minus | 10 | 10 QPU jobs | 40 | 5,120 |
| Hardware reference evaluation | 1 | 1 QPU job | 16 | 8,192 |
| Validation-only evaluation | 1 | 1 QPU job | 65 | 33,280 |
| **Total QPU estimate** | **12 evaluations/jobs as shown** | **12 QPU jobs** | **121** | **46,592** |

The submission-path caps are **16 QPU jobs**, **60,000 total QPU shots**, **5 optimization steps**, and **65 validation examples**. The job store enforces cumulative job/shot limits before each submission; the estimator also rejects over-budget configurations. QPU seconds and resource units are deliberately unestimated (`unknown`) pending actual backend/job evidence.

## 9. Independence engineering and PRNG assumptions

The local training path uses the existing seeded `StatevectorSampler` instance and deterministic pseudorandom replay. It does not claim mathematical independence. Runtime sampling is treated as Runtime-managed; the actual hardware randomness model has not been established. A seed is recorded for the optimizer and the selected indices are recorded, but neither a seed nor a separate seed per call proves independence. No hardware-independence or stationarity claim is made.

## 10. Unresolved decisions

- IBM account authentication, instance/region, backend availability, backend status, queue, and an actual connected layout.
- Installation and compatibility validation of the declared IBM Runtime 0.46.1 in an isolated environment.
- IBM plan capabilities and job/shot/resource quotas.
- Research-owner authorization of a QPU submission and acceptance of the proposed 16-job/60,000-shot ceiling.
- A documented randomness model for the planned hardware measurements.
- Whether a per-event injection ledger and detector-noise/observational dataset are required before any scientific claim. The current Q1.6 set does not establish real astrophysical discrimination.

## 11. Future implementation and execution requirements

The safe command exercised here is:

```powershell
.\.venv312\Scripts\python.exe scripts\run_qnim_ibm_qpu_pilot.py --preflight --verify-dataset --estimate
```

The explicit account query, when authorized, is:

```powershell
.\.venv312\Scripts\python.exe scripts\run_qnim_ibm_qpu_pilot.py --query-ibm
```

The future execution command shape is:

```powershell
.\.venv-qnim-ibm-pilot\Scripts\python.exe scripts\run_qnim_ibm_qpu_pilot.py --execute-pilot --submit --backend <backend_name> --layout <connected-8-qubit-layout>
```

That execution command currently fails closed because `submission_enabled` is false. Do not enable it until the research owner approves the scientific scope, randomness model, backend/layout, and resource cap. To test the declared dependency without changing `.venv312`, create a separate environment:

```powershell
py -3.12 -m venv .venv-qnim-ibm-pilot
.\.venv-qnim-ibm-pilot\Scripts\python.exe -m pip install --upgrade pip
.\.venv-qnim-ibm-pilot\Scripts\python.exe -m pip install -r requirements.txt
```

Before any submission, re-run the safe preflight in that environment, explicitly query IBM account/backend metadata, verify a selected connected layout and its calibration snapshot, confirm available plan/quota, and obtain owner authorization. No test split is needed or loaded by the pilot.

## 12. Final GO/NO-GO

- **Mathematical/implementation design:** the q8 hybrid workflow, split guard, provenance verification, resource accounting, layout checks, persistence, and disabled-by-default submission gate are implemented and safely tested.
- **Primary proposed method:** local QNSPSA-EML-Feynman pretraining followed by QPU `HardwareSPSAOptimizer` SPSA updates, then validation-only measurement.
- **Fallback:** local pretraining plus validation-only QPU inference without hardware fine-tuning, if the research owner authorizes that narrower operation. No remote QNSPSA claim is permitted.
- **Unverified assumptions:** IBM backend availability/layout, account plan and quotas, Runtime compatibility, and the hardware randomness model.
- **Why the pilot is not authorized:** QPU resource availability and randomness are unknown, the installed Runtime is absent, and the owner has not resolved/approved the resource ceiling and scientific scope. Q1.6 cannot support an observational astrophysical claim.
- **Decision:** **NO-GO for QPU execution now; GO for isolated software readiness and no-job preflight only.** `submission_enabled` remains `false`.

Safe checks completed: 14 new pilot tests passed, the focused existing mocked SPSA integration test passed, static compilation passed, and the no-account provenance/resource preflight passed. No training, circuit construction, Aer/StatevectorSampler execution, account query, or QPU job was run.
