# QNIM_QUANTUM_PCA_SCALING_V1 — Pre-execution audit

**Recommendation: NO-GO. Do not authorize q16 or q20 yet.**

The circuit and dataset checks pass, but the effective shot count differs from the frozen protocol, optimizer evaluation telemetry is mislabeled, and progress metrics can pair a pre-refit loss with post-refit parameters. These should be resolved or explicitly accepted and documented before training.

## 1. Optimizer verification

Execution path: `scripts/run_qnim_quantum_pca_scaling.py::_train_configuration` calls `QiskitVQCTrainer.train_vqc`; that constructs `QNSPSAConfig`, instantiates `QNSPSAEMLFeynman`, and calls its `minimize` method. The configured class is therefore actually used; this is not a fallback or a second optimizer.

| Mechanism | Effective behavior |
|---|---|
| QNSPSA-EML-Feynman | Active. |
| Natural-gradient update | Active as a QNG-like preconditioner: `H_inv @ g_total`. |
| True quantum Fisher information (QFI) | **Not computed.** `F_hat` is formed from two outer products of SPSA loss-gradient estimates; no state-derivative, overlap, or QFI circuit calculation is performed. It is a loss-gradient proxy, not a physical QFI/QGT. |
| EML anti-plateau term | Active: `lambda_eml=0.01`, capped by the implementation's `eml_max_scale=10`. This is not zero-noise extrapolation. |
| Zero-noise extrapolation / noise-aware training | Disabled: `noise_aware_training=False`, local Aer sampler, no ZNE path. |
| Feynman Gauss-Legendre term | Active for the first four entries of the optimizer vector, which are **ansatz** parameters, with 8 quadrature points. The method docstring motivates this quadrature for a Chebyshev feature map; this run instead uses ZZFeatureMap inputs and quadratures ansatz parameters. Its accuracy for this architecture is not established by this audit. |
| Curvature regularization | The optimizer adds `hessian_regularization=1e-4` to its inverse-preconditioner matrix. There is no separate Feynman-curvature regularizer; the Feynman component is a gradient quadrature. |

Other effective QNSPSA settings are `lr=0.03`, perturbation `0.05`, blocking delta `0.001`, `n_feynman_params=4`, patience `20`, minimum improvement `0.0005`, and seed `42`. They match the frozen protocol and current QNIM trainer's implementation.

## 2. Effective training configuration

| Setting | q16 | q20 |
|---|---:|---:|
| Training rows / validation rows | 2600 / 650 | 2600 / 650 |
| PCA columns | 0–15 | 0–19 |
| Classes | 13 | 13 |
| Maximum iterations | 20 | 20 |
| Training batch | 32 | 32 |
| Optimizer seed / initialization seed | 42 / 42 | 42 / 42 |
| Learning rate | 0.03 | 0.03 |
| Ordinary training, warm-start, diagnostic, and final-evaluation shots | 512 | 512 |
| Fixed training-reference evaluation shots | **1024** | **1024** |

Both runs load the same hash-pinned train/validation arrays and labels, then take a prefix of each feature row. Only `X_train`/`y_train` are passed to `train_vqc`. Validation is measured by a diagnostics callback and for final reporting; it is not passed to the optimizer, reference-loss selector, gradient, or readout fitting.

**Protocol deviation:** `QiskitVQCTrainer` leaves `reference_shots=None`; the loss function then uses `shots * 2` for local reference-loss evaluations. With 512 configured shots, those evaluations use 1024 shots. The protocol does not state this exception. Resolve whether the intended protocol is 512 for every measurement or explicitly authorizes 1024-shot reference measurements before execution.

For each optimizer iteration, the implementation:

1. Resamples one fixed 32-example train batch and evaluates its current loss.
2. Uses two SPSA estimates, requiring four batch-loss evaluations.
3. Estimates the first four gradient coordinates using one baseline plus `4 × 8` Gauss-Legendre evaluations.
4. Evaluates the proposed step for blocking.
5. Evaluates the fixed 130-example training-reference set for best-parameter selection.

That is **40 loss-function calls per iteration**: 39 calls over 32 examples and one reference call over 130 examples. It corresponds to **1,378 circuit executions per optimizer iteration**, not 40 circuits. The 130-example reference size is `max(128, 10 × 13)`; the local reference shot count is 1024.

At 20 iterations, the expected accounting from the actual code is:

- Optimizer: 27,592 circuits (including the initial 32-example loss).
- Readout warm starts/refits: up to 600 circuits (200 initially, then 200 at iterations 10 and 20).
- Per-iteration train/validation diagnostics: 2,600 circuits (65 per split per iteration).
- Final train/validation evaluation: 3,250 circuits.
- **Total: 34,042 circuit executions**, assuming all 20 callbacks and refits run.
- **Total shots: 18,760,704**, counting reference circuits at 1024 shots and all other circuits at 512.

The trainer reports `n_circuit_evaluations=result.n_evals`. That counter is **801 objective-function calls** for a 20-iteration run, not 801 circuit executions. The runner carries this value into its result JSON under the same misleading name.

At iterations 10 and 20, the trainer refits the readout before calling the runner's diagnostic callback. The callback's parameter metrics therefore use post-refit parameters, while its `loss` argument was calculated before that refit. Those trace rows are not directly comparable as currently labeled.

## 3. Quantum architecture verification

The prior q16/q20 dry-runs built and transpiled both circuits for local Aer; the runner's regression tests executed each architecture on Aer and checked the decoded output shape. No training run was made.

| Configuration | Feature map | Ansatz trainable parameters | Readout measurements | Classical readout parameters | Total |
|---|---|---:|---:|---:|---:|
| q16 / PCA16 | ZZFeatureMap, reps 2, linear | 96 | 31 | 733 | 829 |
| q20 / PCA20 | ZZFeatureMap, reps 2, linear | 120 | 39 | 861 | 981 |

Both use EfficientSU2 with reps 2 and linear entanglement. The measurement vector is ordered as `P(q_i=0)` for every qubit, followed by adjacent `<Z_i Z_(i+1)>` values. The code uses Qiskit's little-endian bitstring convention. In both training loss and runner evaluation, Aer counts are decoded by `_qubit_readout_features`, and those quantum features—not PCA columns—are passed to the 16-hidden-unit, 13-output MLP. PCA values enter only the feature map.

The actual circuit checks confirmed 16/20 measured qubits and 31/39 decoded features. The focused tests exercised one 16-shot Aer circuit for each dimension and one single-batch q16 loss evaluation; these were plumbing tests, not training.

## 4. Regression tests

Command run:

```powershell
& .\.venv312\Scripts\python.exe -m pytest test\unit\test_experiment_config.py test\unit\test_qnim_q1_1.py test\unit\test_qnim_q1_2.py test\unit\test_qnim_q1_4.py test\unit\test_qnim_q1_6_scaling_pilot.py test\unit\test_qnim_neighbor_readout_pilot.py test\unit\test_qnim_quantum_pca_scaling.py -q
```

**Result: 76 passed in 3.09 seconds; 0 failed, 0 skipped.** This covers trainer/configuration compatibility, QNSPSA behavior, the neighbor-readout pilot, and the new runner.

The complete 40-file unit suite was not run: it includes unrelated waveform/data-pipeline tests outside this audit, and the explicit restriction against accessing test data takes precedence. No q16/q20 training was invoked.

## 5. Code review and preservation

Experiment files in the worktree:

- New: `scripts/run_qnim_quantum_pca_scaling.py`, `test/unit/test_qnim_quantum_pca_scaling.py`, `results/scaling_study/qnim_quantum_pca_scaling_v1/protocol.json`, and `execution_manifest.json`.
- Modified: `src/infrastructure/qiskit_vqc_trainer.py`. Its new sampler/backend and preprocessing arguments are optional and appended with defaults; existing caller compatibility tests pass. The runner relies on this private trainer API and several private measurement/readout helpers, so future internal refactors must keep them synchronized.
- `src/infrastructure/qnspsa_eml_feynman.py` and `README.md` are also dirty in the worktree from preceding work; they were not edited during this audit.

The seed is fixed at 42 and other seeds are rejected. The protocol SHA-256 is pinned. Resume preserves attempt traces but **restarts an incomplete fit from the fixed seed**; it does not restore QNSPSA's matrix, RNG, or batch state. A completed configuration is not rerun through the normal manifest path.

Atomic JSON writes and separate attempt directories reduce partial-write risk. However, Markdown result files use non-atomic `write_text`; a process interruption can leave a truncated report. A newly initialized output directory also does not reject orphaned `q16/result.*` or `q20/result.*` files when its execution manifest is missing, and the completed-resume path does not verify the stored result hash. These are result-integrity risks to address before relying on resume or custom output directories.

`git status --short` showed 55 entries before this report was added, including unrelated prior experiment artifacts. `git diff --check` **failed**: it reported trailing whitespace in the modified README and trainer additions. No unrelated cleanup was made.

## 6. Resource assessment

The current dry-run's memory numbers are heuristic, not measured training peaks:

| Configuration | Dry-run peak estimate | Dense-equivalent batch envelope |
|---|---:|---:|
| q16 | ~202 MiB | 64 MiB |
| q20 | ~1,167 MiB | 1,024 MiB |

The estimate assumes two dense-equivalent state buffers per 32-circuit batch plus two dense QNSPSA parameter matrices. The configured Aer backend is matrix-product-state, whose actual memory depends on entanglement and simulator scheduling; these estimates are not a measured MPS peak. QNSPSA's dense parameter matrices are about 10.5 MiB at q16 and 14.7 MiB at q20, before other arrays and simulator workspace.

Historical train/validation-only pilot measurements provide context, not a q16/q20 runtime prediction: q4 runs took 201.6–208.4 seconds with 187.8–190.0 MiB peak RSS; a q8/PCA4 run took 298.9 seconds with 212.1 MiB peak RSS, each at 20 iterations and 801 objective calls. The configurations and hardware/software path differ. **No runtime estimate is extrapolated** for q16 or q20.

The main cost is repeated 512-shot batch circuits (plus 1024-shot reference circuits), especially the 32 Feynman quadrature evaluations per iteration, followed by diagnostics and full-split measurements.

## 7. Scientific limitation

q16/PCA16 versus q20/PCA20 simultaneously changes qubit count, input dimensionality, ansatz parameters (96→120), and classical readout parameters (733→861). It measures combined full-system scaling and cannot isolate a causal effect of adding qubits. A future control should hold PCA input dimensionality fixed (for example PCA16) while changing qubit count, with a predeclared circuit/input-encoding policy. No such control was implemented or run here.

## Scope note

No final-test dataset file was loaded by the runner, dry-runs, or targeted tests. During a historical-resource query, an archived Q1.6 JSON report whose schema contains a `final_test` field was deserialized; that field was not displayed, inspected, or used, and that report was excluded from the resource evidence above. The historical timings above come only from the neighbor-readout train/validation pilot reports.

## Final recommendation

**NO-GO pending resolution** of the 1024-shot reference-loss deviation and correction or explicit acceptance of the circuit-count and refit-trace reporting issues. Also harden report/checkpoint integrity before using custom output directories or resuming partial executions. No frozen protocol was modified and no training was started.
