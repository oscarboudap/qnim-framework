# QNIM V2 implementation audit

## Status

QNIM_QUANTUM_PCA_SCALING_V2 is implemented as an independent q16/PCA16 experiment. Its protocol is separate from the frozen V1 protocol. This audit authorizes implementation review only; it does not authorize training, Aer measurement jobs, QPU jobs, q20, additional seeds, or test-split access.

## Implemented components

- [`protocol.json`](./protocol.json) freezes the q16 architecture, 2,600/650 train-validation data, disjoint training roles, optimizer settings, checkpoint comparison rule, telemetry, measurement artifacts, and execution limits.
- [`run_qnim_quantum_pca_scaling_v2.py`](../../../scripts/run_qnim_quantum_pca_scaling_v2.py) is an independent runner. `--dry-run` verifies source hashes and roles without constructing a circuit or submitting sampler jobs. `--execute` is available for a separately authorized future run; it was not invoked.
- [`qnim_v2_support.py`](../../../src/infrastructure/qnim_v2_support.py) implements deterministic stratified partitions, canonical hashes, little-endian marginal/adjacent-ZZ decoding, full joint-bitstring covariance, first-order delta-method loss uncertainty, paired candidate comparison, calibration summaries, and atomic compressed NPZ/JSON writes.
- [`qnspsa_eml_feynman.py`](../../../src/infrastructure/qnspsa_eml_feynman.py) adds optional initial/pre-refit/post-refit checkpoint hooks and blockwise telemetry. With those hooks omitted, the legacy optimizer selection and update path is retained. The QNSPSA update equations are unchanged.
- [`qiskit_vqc_trainer.py`](../../../src/infrastructure/qiskit_vqc_trainer.py) accepts optional training-role indices while retaining legacy defaults. V2 samples optimizer batches only from A, fits/refits the readout only on B, and measures checkpoint candidates only on C.
- [`test_qnim_quantum_pca_scaling_v2.py`](../../../test/unit/test_qnim_quantum_pca_scaling_v2.py) adds synthetic coverage for the new support and selection behavior.

## Protocol and data integrity dry-run

The final dry-run reported:

- Protocol SHA-256: `3ab6351ba932957a055eb401f4b597dfd88d5366c30722fd116fb983f4111ed6`
- Source dataset manifest hash: `53a153278ce83722a61a02cd8bff6749c599209fc11202b5e64ed5ee23c05bfd`
- Training roles, seed 42:

| Role | Size | SHA-256 |
|---|---:|---|
| Gradient optimization A | 2,275 (175/class) | `e3578cb9b97d90b0e2392f724cb21930b3f0f686de318b3f70de06a87456d686` |
| Readout fitting B | 195 (15/class) | `c4825749d1ed7ba9c0d3ce143305def5aaacf987c5ea615a07623695b23007a6` |
| Checkpoint reference C | 130 (10/class) | `503e476866030059a9e78f82f620fe17a0e5292a123a26558f147a8635ea9c36` |

The dry-run verified the stored train, validation, label, PCA-state, and dataset-manifest hashes; the original train and validation shapes; and that A/B/C are stratified, disjoint, and cover the 2,600-example training split. It also wrote the exact role and validation-diagnostic indices to a V2-only attempt folder. No test file was loaded.

## Selection, telemetry, and artifacts

The optimizer callback scores the initial post-warm-start vector, every post-blocking pre-refit vector, and each scheduled post-refit vector on the same 130 training-reference examples at 512 shots. The scored vector and hashes cover the complete `[96 ansatz | 733 MLP]` state. A candidate replaces the incumbent only when the one-sided paired upper bound is below `-0.0005`, using a Bonferroni-adjusted familywise alpha of 0.05 across at most 22 non-initial comparisons.

Per-example count dictionaries retain joint bitstring outcomes. Marginal/adjacent-ZZ covariance is calculated from those counts and propagated through the MLP loss with a first-order delta method. The method is explicitly approximate and conditional on the fixed examples; it does not include population sampling, optimizer/run variation, backend drift, or model misspecification, and is not described as statistical significance.

Candidate vectors and metadata are atomically persisted; the current selected checkpoint is updated atomically after each accepted candidate. Quantum observations are separate compressed NPZ shards with a JSON sidecar and include the 16 marginals, 15 neighbor-ZZ correlations, standard errors, counts, hashes, split/class, phase, timestamp, backend, and actual shots. Probability/calibration outputs are stored separately. Full validation probabilities and aggregate metrics are reporting-only.

Optimizer telemetry records each SPSA estimate's blockwise norm, the empirical dispersion of the two SPSA probes, Feynman/EML/combined/preconditioned norms, proposed/applied steps, parameter norms, relative movement from initialization, and acceptance/rejection. No gradient norm is inferred from the prior combined gradient-variance field; no QFI, physical quantum natural gradient, or barren-plateau mitigation claim is made.

## Validation results

- V2 synthetic tests plus the existing quantum-PCA-scaling regressions: **34 passed**.
- Broader targeted QNIM suite excluding the one PyCBC-dependent test: **97 passed**.
- The complete targeted pattern collected 98 tests; one pre-existing Q1 test failed because the environment lacks PyCBC distribution metadata (`PackageNotFoundError`). No package was installed.
- Python compilation checks passed for all changed Python sources.
- Final `--dry-run` passed with `training_executed=false`, `sampler_jobs_submitted=false`, and `test_accessed=false`.
- No training, circuit measurement, Aer job, QPU job, q20 run, or test-split access occurred.

Pylance signature-compatibility requests timed out. Existing regression tests covered the shared trainer/optimizer compatibility paths, but this is not a substitute for a live sampler integration run.

## Isolation and remaining risks

The V2 runner writes runtime outputs only under `results/scaling_study/qnim_quantum_pca_scaling_v2/`. The existing worktree already contained V1-related modifications and preserved artifacts from earlier work; this implementation left them untouched and did not edit the V1 protocol, result/checkpoint, or recovered reports.

The real sampler integration, count extraction on a completed circuit job, full V2 runtime, and resource totals remain unverified because measurement jobs were prohibited. A process interruption preserves candidate vectors and the latest selected state, but optimizer RNG/preconditioner state is not resumed; a restart must be a new fixed-seed attempt. Local sampler job IDs may not be available, so aggregate circuit counts, shots, and wall time are the primary resource telemetry.

## Recommendation

**NO-GO for a future exploratory training run at this point.** The protocol, synthetic logic, dataset integrity, partitions, and dry-run are verified, but end-to-end sampler behavior has not been exercised. Reassess after separate explicit execution authorization and a permitted non-training sampler integration check; do not infer scientific performance from this implementation audit.
