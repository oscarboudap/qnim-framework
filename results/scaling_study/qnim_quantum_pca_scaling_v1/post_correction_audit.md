# QNIM_QUANTUM_PCA_SCALING_V1 — Post-correction audit

**Date:** 2026-11-20
**Recommendation: GO for an explicitly authorized exploratory run only.** No training was started. This recommendation does not validate the Feynman quadrature for this architecture, establish quantum natural gradient or barren-plateau mitigation, or support a quantum-advantage claim.

## 1. Issues fixed

- The runner now passes `reference_shots=512` explicitly to `QiskitVQCTrainer`. The protocol remains frozen; its SHA-256 is unchanged.
- Optimizer objective-function calls, completed circuit PUB results, requested shots, evaluated training/validation examples, and periodic readout refits are separate counters. Historical `n_evals` and `optimization.n_circuit_evaluations` remain available with their old objective-call meaning.
- The scaling runner now records pre-refit and post-refit states separately. It re-evaluates the post-refit loss on the same optimizer minibatch and recalculates both diagnostic splits using the same persisted sample indices. Readout-refit failures now fail the run rather than being silently accepted.
- A methodology note documents the optimizer's actual calculations and their interpretation limits.
- The misleading `--resume` option was removed. `--new-attempt` creates a uniquely named attempt from seed 42; it does not restore optimizer state. An exclusive output lock prevents simultaneous writes. Orphan artifacts and corrupted completed results are rejected, and JSON/Markdown writes are atomic.

## 2. Frozen protocol and dataset integrity

The protocol file was not modified. Its SHA-256 remains:

`87581f0b3d0a435ecfd44be59145a13bf824977c732f03d401f30520eb335d8e`

The PCA64 dataset manifest SHA-256 remains:

`53a153278ce83722a61a02cd8bff6749c599209fc11202b5e64ed5ee23c05bfd`

All allowed train/validation artifacts match the frozen hashes:

| File | SHA-256 |
|---|---|
| `X_train_scaled64.npy` | `6fb936d7836789e9ca59941e47a2cdff701acd5d057f4258db82b134237e5dad` |
| `X_val_scaled64.npy` | `f717042628bd35f7a8dad9457996f0d9b3d9d22a08803319c874390714af2783` |
| `y_train.npy` | `9e4cc1a458f2f7825f69718bdd4a78bc5e56acc21d6396b14a2e302a477f01d6` |
| `y_val.npy` | `81c6b2e663a9d65c96c699cd88253efb5003b9893ae19d21c7fa2b5d12719ead` |
| `pca_state.npz` | `183108741f6b991242621a1d8c05ca481c851da459645c7a99a69433ab7e99ff` |

Only the allowlisted train, validation, PCA-state, and dataset-manifest files were checked. No test split was loaded or inspected; both dry-run responses and the execution manifest report `test_access=false`. The PCA64 dataset and historical reports were not modified.

The frozen protocol still contains its original `checkpoint_policy.resume` text so its hash is preserved. Operationally, the runner now uses the safer, explicitly documented `--new-attempt` policy; this is a checkpoint-safety correction, not a change to model, data, or training settings.

## 3. Effective optimizer behavior

`QiskitVQCTrainer.train_vqc` instantiates `QNSPSAEMLFeynman`. Its SPSA estimates use two independent Rademacher directions and four perturbed loss calls. `F_hat` is the mean of the two outer products of stochastic loss gradients, not a physical QFI/QGT. The preconditioner is a dense, regularized `H_inv` updated by the implemented Sherman–Morrison-style operation.

The EML term uses the minimum eigenvalue of the regularized `F_hat` proxy and a capped multiple of the first gradient estimate. The eight-point Gauss–Legendre calculation applies to the first four ansatz coordinates. EML is not ZNE; no noise scaling or extrapolation occurs. The exact implemented calculations and limitations are in [optimizer_methodology_note.md](./optimizer_methodology_note.md).

**Unresolved scientific limitation:** the historical Feynman rationale is stated for a Chebyshev feature map, but this protocol uses ZZFeatureMap inputs and applies the quadrature to EfficientSU2 ansatz angles. The specific integral identity and eight-point accuracy have not been established for this architecture and nonlinear readout. Treat that component as an unvalidated heuristic; do not claim exact Feynman gradients, physical QFI, quantum natural gradient, or proven barren-plateau mitigation.

## 4. Effective shot counts

The runner's dry-run reports and trainer construction now agree on all protocol shots:

| Evaluation path | Shots |
|---|---:|
| Training/current loss | 512 |
| SPSA perturbations | 512 |
| Feynman quadrature | 512 |
| Blocking | 512 |
| Training-reference loss | **512** |
| Readout warm start and refits | 512 |
| Train/validation diagnostics | 512 |
| Final train/validation evaluation | 512 |

The trainer uses the explicit `reference_shots=512`; the prior local default of `shots * 2` is not used by this runner. A recording-sampler regression test exercises ordinary, perturbed, quadrature-style, blocking-style, reference, and readout-fit loss paths and fails if any submitted call differs from 512.

## 5. Circuit-accounting semantics

`n_objective_function_calls` is the number of loss-function evaluations; `n_optimizer_objective_function_calls` isolates calls made by QNSPSA. `n_circuit_executions` counts completed SamplerV2 PUB results, with one bound input circuit per PUB. `n_shots_total` is the sum of the explicitly requested shots for those completed PUB results. `n_training_examples_evaluated` counts circuit-evaluated training rows, including repeated rows; validation rows are reported separately. `n_readout_refits` counts successful periodic refits, excluding the initial warm start.

These are execution counters, not estimates. The trainer validates returned PUB counts before adding them. The runner's diagnostic/final measurement helper contributes one completed circuit per returned feature row. **No training counters exist yet for q16 or q20 because no training was run.** The legacy `n_circuit_evaluations` field is retained as an objective-call alias for existing consumers.

## 6. Trace-state consistency

Every iteration records its pre-refit parameter vector, hash, minibatch loss, and diagnostic metrics. The trace persists the exact optimizer minibatch indices and fixed train/validation diagnostic indices. At iterations 10 and 20, it additionally records the completed readout refit's training indices, followed by a fresh loss evaluation and separately hashed post-refit metrics on those same identified datasets. JSON contains the full metrics and parameter snapshots; result Markdown presents pre/post losses and balanced accuracies separately.

The new callback API is opt-in. Existing trainer callers without a `post_refit_callback` retain their previous single-callback behavior; the scaling runner supplies both callbacks and uses the corrected pre/post trace.

## 7. Checkpoint and output integrity

- Every fresh attempt uses `attempt_NN_<uuid>`; no attempt directory is reused.
- The runner has no `--resume`. `--new-attempt` is only a fresh seeded restart after a failed/interrupted attempt; it never restores QNSPSA state.
- Concurrent runs using the same output directory are blocked by `.runner.lock`. A stale lock after forced process termination must be removed only after confirming no run is active.
- Missing-manifest orphan results/checkpoints, result files not marked completed, and completed JSON/Markdown files whose stored hashes do not match are rejected before acceptance or overwrite.
- JSON and Markdown use same-directory temporary files and atomic replacement. Completed output is immutable; another run requires a new output directory.
- Regression tests cover graceful interruption, fresh unique attempts, lock collisions, orphan artifacts, duplicate completed runs, corrupted JSON/Markdown hashes, and atomic replacement failure.

## 8. Regression tests

Command:

```powershell
& .\.venv312\Scripts\python.exe -m pytest test\unit\test_experiment_config.py test\unit\test_qnim_q1_1.py test\unit\test_qnim_q1_2.py test\unit\test_qnim_q1_4.py test\unit\test_qnim_q1_6_scaling_pilot.py test\unit\test_qnim_neighbor_readout_pilot.py test\unit\test_qnim_quantum_pca_scaling.py -q
```

**Result: 90 passed, 0 failed, 0 skipped in 3.57 seconds.** This includes the 76 previously passing tests and 14 additional corrective cases. Existing small Aer smoke tests ran as part of the suite; no q16/q20 training or expensive simulation was run.

## 9. Final q16/q20 dry-runs

Both commands completed with `status=dry_run`, `training_started=false`, and `test_access=false`. They verify the protocol/dataset, construct and transpile circuits, and calculate resource envelopes; they do not submit circuit measurements.

| Configuration | PCA / qubits | Ansatz parameters | Readout parameters | Total trainable | Measurements | Every shot path | Estimated peak memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| q16 | 16 / 16 | 96 | 733 | 829 | 31 | 512 | ~203.08 MiB |
| q20 | 20 / 20 | 120 | 861 | 981 | 39 | 512 | ~1,167.47 MiB |

Memory values are heuristic dry-run estimates, not measured training peaks. The execution manifest remains `prepared`; q16 and q20 are both `not_started`.

## 10. Files and whitespace status

Files changed or created by this corrective phase:

- Modified: `scripts/run_qnim_quantum_pca_scaling.py`, `src/infrastructure/qiskit_vqc_trainer.py`, `src/infrastructure/qnspsa_eml_feynman.py`, `test/unit/test_qnim_quantum_pca_scaling.py`.
- Created: this report and `optimizer_methodology_note.md`.
- Updated by the required dry-runs: `execution_manifest.json` only; it records both dry-runs and still marks both configurations `not_started`.
- Unchanged: frozen `protocol.json`, `pre_execution_audit.md`, PCA64 dataset, and other historical reports.

The existing worktree remains broadly dirty; unrelated files were not reset, cleaned, or formatted. Default `git diff --check` returns exit code 2 with **819 warnings / 1,638 output lines**, grouped exactly as follows:

| Path | Warnings |
|---|---:|
| `README.md` | 163 |
| `scripts/run_scaling_study.py` | 10 |
| `src/application/use_cases/run_scaling_study_use_case.py` | 13 |
| `src/infrastructure/qiskit_vqc_trainer.py` | 513 |
| `src/infrastructure/qnspsa_eml_feynman.py` | 100 |
| `test/unit/test_run_scaling_study_cli.py` | 5 |
| `test/unit/test_scaling_study_use_case.py` | 15 |

`core.autocrlf=true` causes the default check to flag CRLF endings. The CRLF-aware command `git -c core.whitespace=cr-at-eol diff --check` returns no output; the experiment's added lines and new runner/test/report files have no trailing spaces. No unrelated whitespace was changed.

## Final recommendation

**GO for an explicitly authorized exploratory q16/q20 run under the unchanged model/data/training protocol.** Keep conclusions limited to this implementation and combined system scaling. The unvalidated Feynman component, gradient-proxy `F_hat`, and simultaneous change in qubits/input dimension/parameter counts prevent stronger optimizer or causal-scaling claims. No training was started; explicit authorization is still required.
