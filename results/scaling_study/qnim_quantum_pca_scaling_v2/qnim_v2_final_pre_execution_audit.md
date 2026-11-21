# QNIM V2 final pre-execution audit

## 1. Executive summary

**Overall recommendation: NO-GO for QNIM V2 training.** The V2 runner's role isolation, complete-state checkpoint bookkeeping, synthetic uncertainty tests, and mocked sampler contract are substantially implemented. The current evidence does not establish a calibrated checkpoint confidence bound or verify the runner's default q16 sampler against a real result. The audit also encountered an execution-restriction breach described below; this report must not be treated as a fully compliant pre-execution clearance.

**Restriction breach — disclosure:** a combined pytest command used two `-k` selectors; the final selector superseded the intended `not aer` exclusion. It therefore ran three local Aer sampler test cases: q16 and q20 parameterized measurements at 16 shots, and one q16 injected-Aer loss evaluation at 16 shots. These were isolated test calls, not optimizer training; there was no QPU submission or test-split access. They were nevertheless real Aer measurement executions and violated the requested prohibition. The source tests are [test_qnim_quantum_pca_scaling.py](../../../test/unit/test_qnim_quantum_pca_scaling.py#L407) and [test_qnim_quantum_pca_scaling.py](../../../test/unit/test_qnim_quantum_pca_scaling.py#L449). No further sampler or training execution was performed after that command.

The frozen protocol remains unchanged. Its canonical SHA-256 is `3ab6351ba932957a055eb401f4b597dfd88d5366c30722fd116fb983f4111ed6`. It specifies q16/PCA16 and `StatevectorSampler(seed=42)`; the q16 runner selects `StatevectorSampler`, while the trainer's Aer branch is selected for `n_qubits >= 20` or by explicit sampler injection. Thus an Aer smoke would test an injected alternate backend, not the frozen q16 default. The protocol is [protocol.json](./protocol.json); default selection is in [`_make_vqc_loss_fn`](../../../src/infrastructure/qiskit_vqc_trainer.py#L892).

Status by authorization boundary:

| Area | Audit status |
|---|---|
| Implementation for synthetic/mocked validation | **Conditional** — the targeted V2 and regression checks passed, but live q16 `StatevectorSampler` compatibility remains unverified. |
| A new local Aer smoke | **NO-GO now** — the unapproved Aer test executions have already occurred, and Aer is not the frozen q16 sampler. Any further execution requires separate explicit authorization and a clearly scoped protocol-compatible test. |
| Full q16 V2 training | **NO-GO** — no training is authorized; the sampler path and checkpoint uncertainty limitations remain. |

## 2. Checkpoint uncertainty: implemented equations and validity

The implementation turns each measured bitstring into 31 observables. For Qiskit bitstrings, `bits[-(i+1)]` is qubit `i`; the feature vector is the 16 marginals \(m_i=(1+z_i)/2\), followed by 15 adjacent products \(z_i z_{i+1}\), with \(z_i=1-2b_i\). The bit-order implementation is [`bitstring_features`](../../../src/infrastructure/qnim_v2_support.py#L111), and the count-moment estimator is [`measurement_moments`](../../../src/infrastructure/qnim_v2_support.py#L126).

For example \(i\), with \(S_i\) shots and count \(N_{ib}\) for bitstring \(b\):

\[
\hat\mu_i=\frac{1}{S_i}\sum_b N_{ib}f(b), \qquad
\hat\Sigma_i=\frac{1}{S_i-1}\sum_b N_{ib}(f(b)-\hat\mu_i)(f(b)-\hat\mu_i)^T,
\]
\[
\widehat{\mathrm{Cov}}(\hat\mu_i)=\hat\Sigma_i/S_i.
\]

Because covariance is calculated from the joint bitstring outcomes, it includes covariance among all 31 observables measured in the same shots. The MLP applies a ReLU hidden layer and softmax to \(\hat\mu_i\). For loss \(\ell_i=-\log(\max(p_{i,y_i},10^{-10}))\), the code calculates the feature gradient through the fixed MLP and estimates:

\[
\hat v_i=g_i^T\widehat{\mathrm{Cov}}(\hat\mu_i)g_i,\qquad
\widehat{\mathrm{SE}}(\bar\ell)=\frac{\sqrt{\sum_i\hat v_i}}{n}.
\]

The implemented gradient uses the active-ReLU mask. If the true-class probability is at or below the floor, it returns a zero feature gradient. These calculations are in [`mlp_loss_and_feature_gradient`](../../../src/infrastructure/qnim_v2_support.py#L154) and [`candidate_statistics`](../../../src/infrastructure/qnim_v2_support.py#L206).

For a candidate \(c\) and incumbent \(b\), the point estimate is the mean of the same-example differences \(d_i=\ell_{c,i}-\ell_{b,i}\). Its standard error is computed as:

\[
\widehat{\mathrm{SE}}(\bar d)=\frac{\sqrt{\sum_i(\hat v_{c,i}+\hat v_{b,i})}}{n}.
\]

This pairs examples and requires identical ordered reference indices, but assumes independent shot noise between candidate and incumbent evaluations: it adds their variances and does not estimate cross-candidate covariance. With 22 planned non-initial comparisons, the rule uses \(\alpha/22\), a one-sided normal critical value (about 2.84), and replaces only if the upper bound is below \(-0.0005\). The code is [`compare_candidates`](../../../src/infrastructure/qnim_v2_support.py#L265); the protocol's alpha and threshold are unchanged.

**Interpretation:** this is an uncertainty-aware heuristic, not a guaranteed or calibrated confidence bound. It is a first-order plug-in delta approximation conditional on the fixed 130 examples. It does not account for population sampling, optimizer variability, backend drift, random-seed variation, or model misspecification. The normal approximation can be poor near a ReLU kink or the probability floor; the zero-gradient branch can understate variation when finite-shot features cross such a boundary. The 512-shot covariance estimate is finite-sample and the objective is nonlinear. Bonferroni controls familywise error only if the component comparison tests are valid; approximate normal limits do not establish that condition. The adaptive incumbent reuses an earlier noisy estimate, and the code does not perform an independent confirmation measurement or a sequentially calibrated confidence procedure.

Synthetic evidence is limited but useful: the smooth 512-shot Monte Carlo test found the delta variance close to empirical variance (about 1.07 ratio) and nominal 95% interval coverage around 94%; the nearly tied case did not replace. In 1,000 synthetic 22-comparison null families, 22 families replaced an incumbent (2.2%). These checks exercise generated distributions, not real circuit outcomes. The suite also checks low-probability clipping and uniform/high-entropy behavior. Tests are in [test_qnim_quantum_pca_scaling_v2.py](../../../test/unit/test_qnim_quantum_pca_scaling_v2.py#L158), [test_qnim_quantum_pca_scaling_v2.py](../../../test/unit/test_qnim_quantum_pca_scaling_v2.py#L219), and [test_qnim_quantum_pca_scaling_v2.py](../../../test/unit/test_qnim_quantum_pca_scaling_v2.py#L268).

## 3. Circuit, sampler, and result contract

The builder composes a 16-input `ZZFeatureMap(reps=2, entanglement="linear")` with `EfficientSU2(reps=2, entanglement="linear")`, then the V2 runner measures all 16 qubits. The model has 96 ansatz parameters and 31 readout features. The circuit builder and feature binding are [`_build_feature_map_and_ansatz`](../../../src/infrastructure/qiskit_vqc_trainer.py#L203) and [`_bind_sample`](../../../src/infrastructure/qiskit_vqc_trainer.py#L330).

At q16, `_make_vqc_loss_fn` chooses `StatevectorSampler(seed=42)` unless a sampler is explicitly injected. Each PUB is `(bound_circuit, [])`; the result is read in PUB order from `result[i].data.meas.get_counts()`. The V2 loss, observations, and warm-start paths validate result cardinality, available bit/shots metadata, bitstring validity and count totals. The q16 count parser is inside [`_make_vqc_loss_fn`](../../../src/infrastructure/qiskit_vqc_trainer.py#L1125); observation and warm-start paths use it at the same factory's `measure_count_rows` and `warm_start_readout` closures. Missing-width bitstrings are rejected even when metadata is absent.

Expected dimensions and conventions are internally consistent:

- 16 qubits and 16 classical measurement bits.
- Exactly 16 marginals followed by 15 adjacent ZZ correlations (31 features).
- Class IDs/output columns 0–12; the MLP output width is 13.
- 512 shots are passed for V2 checkpoint/reference and validation evaluations, and actual count totals are checked.
- The PUB/result order is retained when mapping each result to its input example.

The deterministic FakeSampler/FakeJob tests emulate the Qiskit V2 result shape (`data.meas.get_counts`, `num_bits`, `num_shots`) and exercise bit ordering, PUB association, role indices, parameter hashes, output dimensions, malformed counts, short bitstrings, and shot counts. They passed, but mocks do **not** prove compatibility with a live `StatevectorSampler`. Static inspection found Qiskit 2.3.1's `StatevectorSampler.run(pubs, *, shots=...)` and `BitArray.num_bits`/`num_shots`; Aer 0.17.2 is installed. No real `StatevectorSampler` result was obtained.

The accidental Aer tests described above do not close this gap: one used the 16/20-qubit Aer parameterized test at 16 shots; one used an injected q16 Aer loss at 16 shots. Neither tests the frozen q16 default at 512 shots. The V2 runner's use of `StatevectorSampler` and the Aer injection branch are documented by [`_make_vqc_loss_fn`](../../../src/infrastructure/qiskit_vqc_trainer.py#L892) and the frozen sampler policy in [protocol.json](./protocol.json).

## 4. Training-role isolation and data access

The V2 loader hashes the declared dataset files and loads only `X_train_scaled64.npy`, `X_val_scaled64.npy`, `y_train.npy`, and `y_val.npy`; it does not load an `X_test` or `y_test` file. It forms roles with seed 42 and validation diagnostics with seed 42042 in [`_load_verified_inputs`](../../../scripts/run_qnim_quantum_pca_scaling_v2.py#L61).

I independently loaded the persisted role index arrays from the existing dry-run receipt and checked the actual indices against `y_train`:

| Role | Size | Per-class counts | SHA-256 |
|---|---:|---:|---|
| A — gradient | 2,275 | 175 each | `e3578cb9b97d90b0e2392f724cb21930b3f0f686de318b3f70de06a87456d686` |
| B — readout fit | 195 | 15 each | `c4825749d1ed7ba9c0d3ce143305def5aaacf987c5ea615a07623695b23007a6` |
| C — checkpoint reference | 130 | 10 each | `503e476866030059a9e78f82f620fe17a0e5292a123a26558f147a8635ea9c36` |

The arrays are pairwise disjoint and their sorted union is exactly `0..2599`; they are not merely disjoint by hash or size. The 65 validation diagnostic indices are within `0..649` and contain exactly five examples per class. Validation remains outside all three training roles.

The runner passes A/B/C explicitly to the trainer. The trainer samples optimizer batches only from A, fixes warm-start/refit examples to B, and rejects checkpoint-measurement indices other than C. The V2 call sites are in [`_run_training`](../../../scripts/run_qnim_quantum_pca_scaling_v2.py#L292), trainer role validation/batch selection at [`_make_vqc_loss_fn`](../../../src/infrastructure/qiskit_vqc_trainer.py#L1066), the C-only guard in [`measure_counts_by_indices`](../../../src/infrastructure/qiskit_vqc_trainer.py#L1299), and B selection in [`warm_start_readout`](../../../src/infrastructure/qiskit_vqc_trainer.py#L1417). Legacy defaults exist for callers that omit role indices, but the V2 runner does not use them.

Validation measurements/probabilities are emitted for fixed diagnostic rows and the selected checkpoint's full 650-row validation split, after or alongside training solely for reporting. They are not passed into optimizer updates, refit fitting, or `compare_candidates`. The runner's checkpoint callback and final reporting are in [`checkpoint_candidate`](../../../scripts/run_qnim_quantum_pca_scaling_v2.py#L468) and [`_run_training`](../../../scripts/run_qnim_quantum_pca_scaling_v2.py#L665). The protocol hash and dataset hashes were reverified; no test-split file was accessed. The historical validation accuracy/cross-entropy summary does not contain recoverable per-example probabilities, so none were reconstructed.

## 5. Checkpoint lifecycle, optimizer telemetry, and recovery

`QNSPSAEMLFeynman.minimize` invokes the checkpoint callback on the post-warm-start initial vector before optimization; it requires this initial state to be declared the incumbent. On each iteration, the optimizer computes a proposed vector, applies or rejects it through the blocking test, then evaluates the complete post-blocking vector as `pre_refit`. If the scheduled callback refits the MLP, it evaluates the mutated complete vector as `post_refit`. Every callback receives a copy of the complete parameter vector; an accepted state is copied to `best_theta`, and the returned `optimal_params` is that saved vector. This path is in [`minimize`](../../../src/infrastructure/qnspsa_eml_feynman.py#L189), especially its initial candidate and selection at lines 321–361 and pre/post-refit evaluations at lines 535–645.

The optimizer separates 96 quantum and 733 classical parameters (829 total). Each step records block norms for the two SPSA estimates, Feynman and EML contributions, combined/preconditioned gradients, proposed/applied movement, parameter norms, relative movement from initialization, and acceptance/rejection. The readout refit mutates only the classical slice every 10 iterations; the runner records it from B and requests a post-refit candidate. See the block telemetry and selection logic in [`minimize`](../../../src/infrastructure/qnspsa_eml_feynman.py#L431) and the refit callback in [`_run_training`](../../../scripts/run_qnim_quantum_pca_scaling_v2.py#L619).

Important telemetry limitation: the recorded SPSA block variance is the coordinate-wise sample variance of only two perturbation estimates, averaged within each block. That is a noisy two-probe diagnostic, not a stable estimate of a gradient distribution. Existing combined `gradient_variance_history` is also not a quantum-only gradient variance. The quantum norms are sliced from actual vectors; they are not inferred from that legacy scalar. No QFI, physical quantum natural gradient, or validated barren-plateau mitigation is claimed.

The runner now carries the exact selected candidate index through final serialization and verifies the selected candidate's complete-parameter hash against the optimizer's returned vector. This avoids ambiguity when different candidate evaluations have identical parameter hashes but different finite-shot measurements. Candidates, losses/variance arrays, measurement shards, metadata, and the final checkpoint are written atomically **per file**. NPZ and JSON sidecars are not a cross-file transaction; a crash between replacements can leave an orphan or mismatched pair. See [`atomic_save_npz`](../../../src/infrastructure/qnim_v2_support.py#L529), [`atomic_write_json`](../../../src/infrastructure/qnim_v2_support.py#L552), and selected-state handling in [`_run_training`](../../../scripts/run_qnim_quantum_pca_scaling_v2.py#L665).

Attempts use a new timestamped directory and do not restore QNSPSA's preconditioner, optimizer RNG, sampler RNG, or live minibatch. A new attempt starts from fixed initialization; it is a restart, not a resume. A caught ordinary exception writes a failed manifest, but abrupt process termination or `KeyboardInterrupt` may leave a running manifest and whatever individual atomic files were already replaced. Saved full parameter vectors and measurement counts can reproduce a recorded checkpoint's feature/loss calculation; they do not restore or replay the optimizer trajectory or sampler PRNG state.

## 6. Quantum representation, probabilities, and scientific interpretation

The circuit is structurally the learned representation: PCA16 inputs are encoded, passed through the 96-parameter `EfficientSU2`, and measured into 31 features. The 733-parameter MLP maps those features to 13 probabilities. The MLP is larger than the quantum parameter block and its scheduled refits can cause substantial classical movement, but it does not bypass the quantum feature extraction. Measurement observations and MLP probabilities are written separately. The observation arrays preserve joint sparse bitstring counts, 31 feature means, standard errors, split/example/class/phase, and parameter hashes; probability arrays separately store 13 probabilities and per-example prediction/loss/calibration telemetry. See [`measurement_artifact_arrays`](../../../src/infrastructure/qnim_v2_support.py#L398), [`probability_artifact_arrays`](../../../src/infrastructure/qnim_v2_support.py#L470), and the runner's persistence functions at [`_run_training`](../../../scripts/run_qnim_quantum_pca_scaling_v2.py#L407).

Parameter movement plus the fixed diagnostic quantum features can show whether the circuit changed and whether those changes are observable. They do not by themselves establish that the quantum representation is useful. That requires later controlled comparisons (trainable versus frozen ansatz; entangling versus non-entangling; joint versus alternating/refit schedules), fixed input dimensions, repeated seeds, and analysis of representation sensitivity. No ablations were run here and no quantum-advantage claim follows.

Accuracy counts argmax decisions; cross-entropy penalizes the true-class probability on every example. A few more correct argmaxes than uniform chance can coexist with cross-entropy worse than \(\ln(13)\) if wrong predictions are confident or the true-class probability is very small. The saved probabilities are required to determine which behavior occurred in a particular run; unavailable V1 probabilities were not reconstructed. V2 records per-example NLL, margins, predictive entropy, Brier score, and 10-bin top-label ECE through [`probability_diagnostics`](../../../src/infrastructure/qnim_v2_support.py#L326). These validation outputs are reporting-only.

## 7. Validation performed and worktree isolation

- V2 plus safe V1 QNIM regression command with `not aer`: **39 passed, 3 deselected** (before the final strict-width correction).
- Non-training q1.4 circuit/readout regressions: **8 passed, 1 deselected**.
- After strict-width correction, compilation of modified implementation/test files passed.
- A later combined pytest invocation reported **51 passed, 1 deselected**, including all 20 V2 tests and 23 V1 scaling tests. Because its duplicate `-k` usage failed to exclude Aer tests, this result is contaminated by the three local Aer executions disclosed above and is not clean evidence of the prohibited no-measurement condition.
- Protocol/dataset verification returned the frozen protocol hash and declared train/validation/dataset hashes; shapes were `(2600, 64)` and `(650, 64)`. No training or test file was used.
- No QPU job, optimizer training run, or test-split access occurred.

The worktree was already dirty in V1-related paths and historical/data artifacts. This audit did not edit the V1 protocol, V1 execution manifest/result/checkpoint, or historical reports. Shared trainer/optimizer files have changes, so backward compatibility is covered only by the targeted regression tests above. `git status` showed the pre-existing V1 modifications; they were not reverted or cleaned.

## 8. Exact requirements for any future minimal sampler smoke

No further sampler execution is authorized by this audit. If separately authorized, a compliant minimal integration check should:

1. Be one isolated, logged attempt; no optimizer construction or training loop, no readout fit, no q20, no QPU, and no test data or labels.
2. Use one fixed synthetic 16-feature vector (or a specifically authorized training-only vector), the frozen q16 circuit architecture, and one PUB.
3. Use exactly 512 shots and verify one result for one PUB, 16 measured bits, 512 actual counts, exact 16-bit keys, 16 marginals, 15 adjacent ZZ values, 31 finite features, and a 13-element finite probability vector if exercising the loss adapter.
4. Persist the backend, job identifier if provided, actual shots, result shape, and pass/fail in a dedicated smoke manifest; do not write into V1 or claim a V2 training attempt.
5. Match the intent: to validate the frozen q16 V2 path, use `StatevectorSampler(seed=42)`. If Aer is specifically authorized, inject Aer only for this one integration check and label it an alternate-backend test; it does not validate the frozen q16 default. Any protocol backend change requires a versioned protocol decision, not a silent edit.

## 9. Remaining risks and final recommendation

The principal scientific/statistical risks are the approximate and potentially miscalibrated delta-method comparison; adaptive incumbent reuse; the two-probe gradient-variance telemetry; the large classical head/refits relative to the quantum block; and the absence of real default-sampler compatibility evidence. Operational risks include per-file rather than transactional artifact writes and no optimizer-state resume.

**Final recommendation: NO-GO.** The implementation is conditionally ready for further non-executing synthetic/mocked work, but this audit cannot grant execution readiness. The minimal local Aer smoke is **not ready to authorize as the next step**: an unintended Aer test execution already happened, and Aer is not the frozen q16 default. Full q16 V2 training remains **NO-GO** pending explicit user authorization, a separately scoped protocol-compatible sampler smoke, and an accepted plan for the uncertainty limitations. Stop here; no experiment was started.
