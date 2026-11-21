# QNIM q16/PCA16 — Failed-attempt recovery audit

**Recovery date:** 2026-11-20
**Scope:** Read-only forensic recovery of the single authorized q16 attempt, followed by a report-schema fix. No QNIM training or expensive simulation was run during recovery.

## 1. Original attempt and preservation

The execution manifest identifies exactly one q16 attempt:

`results/scaling_study/qnim_quantum_pca_scaling_v1/checkpoints/q16/attempt_01_803f64cdd3424d1e960c646ef7e53084/`

Its attempt ID is `803f64cdd3424d1e960c646ef7e53084`. The manifest records start `2026-11-20T15:49:41.904337+00:00`, end `2026-11-20T16:24:30.238955+00:00`, latest iteration 20, and `KeyError: 'data'`. q20 remains `not_started`; no other attempt exists.

Before any recovery changes, I recorded:

| Artifact | Size | Last modified (UTC) | SHA-256 |
|---|---:|---|---|
| `execution_manifest.json` | 15,957 B | `2026-11-20T16:24:30.3041405Z` | `dde551781263d934794c2ad7c8467f0aca9dfce54f9116fdca5af0cf1a3b316f` |
| `protocol.json` | 5,096 B | `2026-11-20T14:52:17.4284394Z` | `87581f0b3d0a435ecfd44be59145a13bf824977c732f03d401f30520eb335d8e` |
| `checkpoint.json` | 805,869 B | `2026-11-20T16:24:30.2947113Z` | `08b26743930a6101c03582ac3328896fc7dd9e65d09bad3cbf40a364b7858896` |
| `result.json` | 856,259 B | `2026-11-20T16:24:30.2252912Z` | `127111b4dc6218cfbe3ee3904f6a4eb50770a317e959a4904f6dd310e935c8e8` |

Both attempt JSON files parse successfully. The checkpoint and manifest have no stored result hash (`result_sha256` is absent/null); the result hash above was computed directly. `result.md` and atomic-write temporary files were absent. The attempt directory contained only `checkpoint.json` and `result.json` at inspection. The original result and checkpoint were left unchanged. A separate [result.recovered.md](./checkpoints/q16/attempt_01_803f64cdd3424d1e960c646ef7e53084/result.recovered.md) was created; its SHA-256 is `f812ec0eec8088abc9db8c84418706cb45a4689ae4c76d09d703e6491d927562`.

## 2. Training completion and recovered evidence

Training **completed all 20 optimizer iterations**. Evidence is consistent across the completed `result.json` (`status=completed`, `training_protocol.iterations=20`, 20 trace entries), the checkpoint trace (iteration 20, 20 entries), and the manifest's `latest_iteration=20`. The optimizer did not stop early: `converged=false`, `termination_reason=max_iterations_reached`. The runner then completed the final train/validation evaluation and assembled the result before failing in report generation. Thus the manifest's `failed` status describes output finalization, not a training exception.

The final parameter vectors were persisted in `result.json`: 96 quantum ansatz values and 733 classical readout values (829 total). Their stored SHA-256 values are:

- Quantum initialization: `e812a67f32d48fb25a6f5a37baf184cfef3354854e37438e1b0347ddb6b58485`
- Final quantum parameters: `ebd5d407584b9cc8398cfab4b05def4312e1323254d16933766e9cff8de159da`
- Final classical readout: `902b0265b837e3d5abd6425e64adac9ad0bc127fc78de5fcd81463596eb0ea51`

Pre-refit and post-refit measurements were saved at iterations 10 and 20, with separate parameter hashes. The final full-split metrics and fixed-subset validation diagnostics are present. Validation was metrics-only (`diagnostic_validation_used_for_updates=false`). No test split was accessed; both the result and execution manifest record `test_access=false`.

### Final metrics from the saved model

| Split | Examples | Accuracy | Balanced accuracy | Macro-F1 | Cross-entropy |
|---|---:|---:|---:|---:|---:|
| Train | 2,600 | 0.1062 | 0.1062 | 0.0921 | 3.3994 |
| Validation | 650 | 0.1077 | 0.1077 | 0.0956 | 3.3959 |

The full per-class precision/recall/F1 tables, confusion matrices, and all 20 iteration records are in [result.recovered.md](./checkpoints/q16/attempt_01_803f64cdd3424d1e960c646ef7e53084/result.recovered.md). Precision and F1 are explicitly marked there as derived from the saved confusion matrices; the original JSON stores per-class recall and confusion matrices.

### Optimizer and execution telemetry

- Accepted/rejected updates: **9 / 11**.
- Objective calls: **803 total**, including **801 QNSPSA** calls.
- Measured circuit executions: **34,366**; requested shots: **17,595,392**.
- Training/validation circuit-evaluated examples, including repeats: **32,286 / 2,080**.
- Periodic readout refits: **2**; readout initializations: **1**.
- Runtime: **1,905.905 s** trainer time; **2,088.202 s** total fit and full evaluation. Manifest start/end interval: **2,088.335 s**.
- Recorded gradient information is variance per step, not gradient L2 norm. Step and proposed-step norms are saved. Peak memory was not recorded.
- The result has no warnings list and all serialized numeric values are finite. Gradient variance is high on some iterations (maximum about 217.45); 11 proposed updates were rejected by blocking. These are observations, not proof of numerical instability.
- Backend/software: local CPU Qiskit Aer `matrix_product_state`, Qiskit 2.3.1, Aer 0.17.2, Python 3.12.0, NumPy 2.2.6. No hardware run occurred.

## 3. Integrity and dataset checks

The protocol hash remains `87581f0b3d0a435ecfd44be59145a13bf824977c732f03d401f30520eb335d8e`. The result records the same protocol hash. Its dataset manifest hash is `53a153278ce83722a61a02cd8bff6749c599209fc11202b5e64ed5ee23c05bfd`.

I re-hashed only the allowlisted train/validation and PCA-state inputs (not test data); all match the frozen protocol:

| File | SHA-256 |
|---|---|
| `X_train_scaled64.npy` | `6fb936d7836789e9ca59941e47a2cdff701acd5d057f4258db82b134237e5dad` |
| `X_val_scaled64.npy` | `f717042628bd35f7a8dad9457996f0d9b3d9d22a08803319c874390714af2783` |
| `y_train.npy` | `9e4cc1a458f2f7825f69718bdd4a78bc5e56acc21d6396b14a2e302a477f01d6` |
| `y_val.npy` | `81c6b2e663a9d65c96c699cd88253efb5003b9893ae19d21c7fa2b5d12719ead` |
| `pca_state.npz` | `183108741f6b991242621a1d8c05ca481c851da459645c7a99a69433ab7e99ff` |
| `dataset_manifest.json` | `53a153278ce83722a61a02cd8bff6749c599209fc11202b5e64ed5ee23c05bfd` |

The dataset and frozen protocol were not modified. The manifest remains a forensic record of the report-finalization failure; it was not rewritten. No q20 attempt was created.

## 4. Reporting bug: cause and correction

`_train_configuration()` emits sample counts under `result["input"]`, along with feature shapes and selected columns. `_result_markdown()` instead dereferenced `result["data"]["train_examples"]`, causing the reported `KeyError: 'data'`. A second stale dereference would also have failed later: the renderer and checkpoint finalizer expected `optimization.iterations`, but the producer stores the authoritative count at `training_protocol.iterations`.

The producer's authoritative locations are:

| Information | Result field | Saved value |
|---|---|---:|
| Train / validation examples | `input.train_examples` / `input.validation_examples` | 2,600 / 650 |
| PCA dimension / qubits | `configuration.n_pca_components` / `configuration.n_qubits` | 16 / 16 |
| Quantum measurement feature dimension | `input.measurement_feature_shapes.train[1]` | 31 |
| Number of classes | `readout.output_classes` | 13 |
| Optimizer iteration count | `training_protocol.iterations` | 20 |
| Termination and final loss | `optimization.termination_reason` / `optimization.final_loss` | `max_iterations_reached` / 3.396796 |

The runner now:

1. Reads sample counts from `input` and iteration count from `training_protocol`.
2. Validates required fields before rendering and raises a descriptive `ValueError` for missing training data.
3. Checks that trace and optimizer-step counts match the protocol iteration count, and that scheduled refits have post-refit metrics.
4. Uses the same authoritative iteration count for the final checkpoint.

The result schema and saved attempt were not rewritten; atomic writes and hash protections remain in use.

## 5. Code changes and validation

Files changed for the reporting correction:

- Modified [run_qnim_quantum_pca_scaling.py](../../../scripts/run_qnim_quantum_pca_scaling.py).
- Modified [test_qnim_quantum_pca_scaling.py](../../../test/unit/test_qnim_quantum_pca_scaling.py).
- Created this audit and the separate recovered report linked above.

The fake-result fixture now uses the producer's actual schema. New synthetic-only tests reproduce the missing `input.train_examples` case and verify the Markdown iteration count and metrics; the mocked-run integration test also checks that checkpoint and Markdown iteration counts match the saved JSON.

Validation:

- Focused synthetic reporting/checkpoint tests: **3 passed**.
- Targeted QNIM regression suite: **97 passed** in 2.56 s.
- No q16/q20 training or expensive simulation was performed by these tests.
- An intermediate run exposed the second stale `optimization.iterations` checkpoint lookup; it was corrected before the final passing suite.

## 6. Scientific interpretation and next steps

The validation accuracy (0.1077) is descriptively above uniform 13-class chance (0.0769), but this is one exploratory seed, with no repeated-seed or independent-test evaluation. Macro-F1 is 0.0956; classes 11 and 12 were never predicted in the full validation confusion matrix. This run is useful as a preserved exploratory observation and debugging record, **not** as statistically supported evidence of reliable class discrimination, physical-theory separation, or quantum advantage.

No additional execution is required to recover this completed attempt. Any new seed, q20 run, or test evaluation requires separate authorization.
