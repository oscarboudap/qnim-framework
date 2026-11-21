# QNIM IBM QPU Scalability Readiness Audit

**Review date:** 2026-11-20
**Repository state reviewed:** branch `experiment/qnim-above-chance`; the worktree already contained unrelated tracked and untracked changes. This audit adds only this report and does not alter those changes.
**Decision:** **NO-GO for QPU scaling runs.** The repository has a narrow QPU-in-the-loop SPSA fine-tuning route, but it does not currently run the QNSPSA-EML-Feynman optimizer remotely as the V1/V2 scaling protocols imply. In addition, the standard `mode="ibm"` evaluation path reads `X_test`/`y_test`, the IBM account/backend and plan were not verified, and the LIGO-PCA generator permits a sinusoidal fallback by default.

This was a static audit. No credentials or account were accessed, no dataset arrays or test examples were loaded, and no circuit, sampler, simulator, training, or QPU job was run. Resource totals below are arithmetic projections from source control flow, not measured QPU usage.

## 1. Executive summary and current recommendation

| Question | Finding |
|---|---|
| Does any code submit real IBM jobs? | Yes. `QiskitVQCTrainer.fine_tune_on_hardware` creates `SamplerV2(mode=backend)` and evaluates SPSA losses on the selected backend. |
| Does that run QNSPSA-EML-Feynman on the QPU? | No. `train_and_evaluate` first runs `train_vqc` locally, then optionally adapts the complete parameter vector with a separate hardware SPSA optimizer. This is QPU-in-the-loop **SPSA fine-tuning**, not remote QNSPSA-EML-Feynman training. |
| Do the current V1/V2 scaling runners use QPUs? | No. V1 explicitly uses local Aer MPS; V2 uses local `StatevectorSampler`. Both set the hardware-training path off. |
| Can the standard IBM mode be used without touching test data? | No. `mode="ibm"` sets `use_real_hardware=True`; the standard `train_and_evaluate` path calls `_validate_on_ibm`, which samples `dataset.X_test` and `dataset.y_test`. This is a test-sealing blocker. |
| Is an IBM backend, plan, or quota confirmed? | No. No account query was performed. Backend names in source are defaults, not evidence of account access or current availability. |
| Can the current data establish real gravitational-theory discrimination? | Not by itself. LIGO-PCA currently generates PyCBC waveforms and beyond-GR injections, whitens using an analytic PSD approximation, and does not consume observed LIGO event data. Its PyCBC fallback is enabled unless `strict_physics=True` is explicitly supplied. |

**Recommendation:** do not submit q16/q20 training or scaling jobs yet. First create a separately versioned hardware protocol; preserve V1/V2 unchanged. Implement a deliberate remote QNSPSA path if that is the claimed algorithm, keep the current hardware SPSA route as a separately named comparator, remove test access from all development/evaluation paths, and establish backend/layout, Runtime-version, strict-data, and budget gates.

## 2. Scope, evidence, and repository state

The audit followed the QNIM trainer and optimizer into the V1/V2 runners, the application entry point, hardware adapters, declared dependencies, and current public IBM documentation. The repository and API references are linked throughout. The V1/V2 protocol files were treated as frozen; neither was edited.

The relevant checked-in configuration is in [`requirements.txt`](../../requirements.txt#L93-L97): Qiskit 2.3.1, Aer 0.17.2, IBM Runtime 0.46.1, and Qiskit Machine Learning 0.9.0 are declared. In the inspected local environments, `.venv312` had Qiskit/Aer but not IBM Runtime or Qiskit Machine Learning; the other inspected Python environments lacked these Qiskit packages. Thus the declared Runtime interface could not be validated locally. Current IBM documentation is newer than the pinned Runtime version; version compatibility must be established before relying on examples from the current docs.

The current branch is `experiment/qnim-above-chance`. Pre-existing modifications and untracked experiment artifacts were present. They were not reverted, cleaned, or edited. The report is the only file created for this audit.

## 3. What the current training path actually does

### 3.1 Local QNSPSA path

`QiskitVQCTrainer.train_vqc` constructs its loss with `mode="sim"` and `remote_hardware=False`. It selects the supplied sampler override if any; otherwise it selects local `StatevectorSampler`, or Aer MPS for circuits with at least 20 qubits. `noise_aware_training` also constructs a **local Aer sampler** from the IBM backend; it is not remote training. See [`_make_vqc_loss_fn`](../../src/infrastructure/qiskit_vqc_trainer.py#L892-L1032) and [`train_vqc`](../../src/infrastructure/qiskit_vqc_trainer.py#L1701-L1810).

The V1 runner explicitly builds local Aer MPS samplers for training and diagnostics, uses q16/q20 with the number of PCA features equal to the number of qubits, 20 iterations, batch size 32, and 512 shots, and sets `train_on_hardware=False`. See [`run_qnim_quantum_pca_scaling.py`](../../scripts/run_qnim_quantum_pca_scaling.py#L30-L51), [`_build_local_aer`](../../scripts/run_qnim_quantum_pca_scaling.py#L249-L257), and its training invocation at [line 1071](../../scripts/run_qnim_quantum_pca_scaling.py#L1058-L1085).

The V2 runner fixes q16/PCA16, trains with local `StatevectorSampler`, and contains no QPU submission path. It verifies and loads only train and validation arrays; its frozen roles are 2,275 optimizer examples, 195 readout examples, and 130 checkpoint-reference examples, while validation has 650 examples and is reporting-only. See [`run_qnim_quantum_pca_scaling_v2.py`](../../scripts/run_qnim_quantum_pca_scaling_v2.py#L35-L125), [`_run_training`](../../scripts/run_qnim_quantum_pca_scaling_v2.py#L295-L362), and the frozen configuration at [lines 640-658](../../scripts/run_qnim_quantum_pca_scaling_v2.py#L640-L658).

### 3.2 The only implemented QPU-feedback training route

`train_and_evaluate` calls local `train_vqc` first. Only if `train_on_hardware=True` does it then call `fine_tune_on_hardware`. That method creates a Runtime service, resolves the backend, checks only `backend.num_qubits`, transpiles for that backend, and sends losses through `SamplerV2(mode=backend)`. It passes those losses to [`HardwareSPSAOptimizer.optimize`](../../src/infrastructure/hardware_spsa_optimizer.py#L97-L179), which performs ordinary two-sided SPSA updates on the full quantum-plus-readout vector. It does **not** call QNSPSA-EML-Feynman for those updates. See [`fine_tune_on_hardware`](../../src/infrastructure/qiskit_vqc_trainer.py#L1589-L1685) and [`train_and_evaluate`](../../src/infrastructure/qiskit_vqc_trainer.py#L2130-L2207).

Therefore:

1. `mode="ibm"` alone does not make QNSPSA training remote.
2. `train_on_hardware=True` means local QNSPSA pretraining followed by QPU SPSA fine-tuning.
3. Any result must identify which optimizer produced the QPU-driven updates. Calling the combined path “QNSPSA training on IBM hardware” would be inaccurate.
4. The generic `sampler_override` is a low-level injection point, not a complete supported remote QNSPSA workflow: normal `train_vqc` still sets `remote_hardware=False`, uses local batch sizing, and does not establish a Runtime session or a remote-run policy.

## 4. QNSPSA circuit/job accounting

The custom optimizer computes, per iteration with a batch-capable loss:

- one current-loss evaluation on the refreshed batch;
- two evaluations for the first SPSA gradient;
- two for the second SPSA gradient used in its rank-2 metric estimate;
- one baseline plus eight Gauss-Legendre evaluations for each of \(m\) Feynman parameters;
- one blocking evaluation;
- one fixed-reference evaluation when there is no checkpoint-candidate callback.

For \(m=4\), that is \(1+2+2+(1+8m)+1+1=40\) objective calls per iteration. With \(T\) full iterations, the base count is \(1+40T\), including the initial loss. The implementation uses fixed GL-8 nodes; changing a config field named `n_gl_points` does not change the actual eight-point loop. See the optimizer loop and fixed quadrature implementation in [`qnspsa_eml_feynman.py`](../../src/infrastructure/qnspsa_eml_feynman.py#L370-L470) and [`_feynman_gradient`](../../src/infrastructure/qnspsa_eml_feynman.py#L703-L750).

Each current objective call submits one PUB per example through one `sampler.run`; parameter binding is done by creating an individually bound circuit for each example. See [`evaluate_batch`](../../src/infrastructure/qiskit_vqc_trainer.py#L1168-L1244). This means a call is one sampler job with many circuits/PUBs, not one QPU job per example.

### Full scheduled local-run totals from the existing runners

The following arithmetic assumes all 20 iterations complete, 13 classes, 2,600 training examples, 650 validation examples, and no early termination.

| Runner | Objective evaluations | Total bound-circuit PUBs | Requested shots | Sampler runs/jobs | QPU jobs |
|---|---:|---:|---:|---:|---:|
| V1 q16 or q20, including warm start/refits, diagnostics, and final train/validation reports | 803 QNSPSA loss calls (801 base + 2 post-refit calls) | 34,366 | 17,595,392 | 1,041 local | 0 |
| V2 q16, including checkpoint calls, readout work, diagnostics, and final validation report | 781 optimizer loss calls + 23 checkpoint measurements | 29,347 | 15,025,664 | 810 local | 0 |

**V1 arithmetic:** the 801 base optimizer calls evaluate 27,592 PUBs (32-example minibatches, plus 130 reference examples on each of 20 iterations). Two post-refit loss evaluations add 64; the 200-example initial readout fit and two 200-example refits add 600; 22 pre/post-refit diagnostic snapshots add 2,860; and final full train/validation metrics add 3,250. These sum to 34,366 PUBs, each at 512 shots. The 1,041 jobs are 803 loss calls, 3 readout calls, 132 diagnostic chunk calls, and 103 final-metric chunk calls. Source constants, diagnostics, callbacks, and final evaluation are in [`run_qnim_quantum_pca_scaling.py`](../../scripts/run_qnim_quantum_pca_scaling.py#L30-L51), [diagnostic accounting](../../scripts/run_qnim_quantum_pca_scaling.py#L923-L1000), and [training/final metrics](../../scripts/run_qnim_quantum_pca_scaling.py#L1058-L1127).

**V2 arithmetic:** \(1+20(1+4+1+8\cdot4+1)=781\) minibatch loss calls, each with 32 PUBs; 23 checkpoint states measured over 130 reference examples; 65 validation diagnostics at initialization and again for the selected checkpoint; 195-example readout initialization plus two refits; and 650 final validation circuits. This totals 29,347 PUBs at 512 shots. See [`checkpoint_candidate`](../../scripts/run_qnim_quantum_pca_scaling_v2.py#L467-L580), [`optimizer_callback`](../../scripts/run_qnim_quantum_pca_scaling_v2.py#L618-L635), and [selected/final validation](../../scripts/run_qnim_quantum_pca_scaling_v2.py#L675-L720).

These are local simulator workloads, not QPU workloads. They do show that directly transplanting the current one-call/one-job objective loop to a QPU would create hundreds of sequential jobs unless objective evaluations were redesigned and batched. The current protocol files remain untouched: [`V1 protocol`](./qnim_quantum_pca_scaling_v1/protocol.json) and [`V2 protocol`](./qnim_quantum_pca_scaling_v2/protocol.json).

## 5. Exact cost of the implemented QPU fine-tuning

The default fine-tuning route uses 20 steps, batch size 4, and 128 shots for each plus/minus SPSA loss. It remeasures a 16-example reference subset every fifth step at the configured reference shot count (the CLI default is 4,096). Each plus, minus, or reference `evaluate_batch` invocation calls `sampler.run` separately. Source: [`HardwareSPSAOptimizer.optimize`](../../src/infrastructure/hardware_spsa_optimizer.py#L97-L179), [`ExperimentConfig`](../../src/application/use_cases/generate_experiment_results_use_case.py#L13-L45), and CLI defaults in [`generate_results.py`](../../scripts/generate_results.py#L25-L82).

| Component | Jobs | PUBs | Shots |
|---|---:|---:|---:|
| 20 SPSA steps: plus and minus, 4 examples × 128 shots each | 40 | 160 | 20,480 |
| Four reference evaluations: 16 examples × 4,096 shots | 4 | 64 | 262,144 |
| **QPU fine-tuning subtotal** | **44** | **224** | **282,624** |

If `reference_shots` is not set and falls back to 128, the same 44 jobs and 224 PUBs total 28,672 shots. The standard CLI sets it to 4,096. Actual QPU seconds and resource-unit cost cannot be inferred from shots alone; they require backend-specific execution/queue details and returned job metrics.

The two SPSA loss jobs at a given step are mathematically ready from the same current parameter vector and perturbation, so a future implementation could submit both in one job or as independent jobs in a Runtime Batch, then wait before updating parameters. Current code does not do this: it blocks on the plus result before submitting minus. Reference evaluations occur after the update and cannot be submitted ahead of that step. This optimization must preserve the existing SPSA equations and be validated separately.

The current standard IBM evaluation path also submits one final job over `n_hw_validation` examples. With the default 1,300 examples and 2,048 shots, that is 1,300 PUBs and 2,662,400 executions. With the repository’s custom three-scale ZNE, it is 3,900 PUBs and 7,987,200 executions. This calculation is not approval to use the test set; see Section 6.

## 6. Test-set leakage is a hard blocker

Despite its name and docstring, `_validate_on_ibm` checks and reads `dataset.X_test`/`dataset.y_test`, selects examples from `y_test`, builds circuits from `X_test`, and computes metrics against `y_test`. `train_and_evaluate` calls this function whenever `use_real_hardware` and a token are present. The application use case sets `use_real_hardware=(mode == "ibm")`, even when `train_on_hardware=False`. See [`train_and_evaluate`](../../src/infrastructure/qiskit_vqc_trainer.py#L2195-L2235), [`_validate_on_ibm`](../../src/infrastructure/qiskit_vqc_trainer.py#L2291-L2350), and [`GenerateExperimentResultsUseCase.execute`](../../src/application/use_cases/generate_experiment_results_use_case.py#L150-L175).

Consequences:

- Running the normal `--mode ibm` workflow on a dataset with a sealed test split consumes that split.
- `train_on_hardware=False` does not prevent this IBM test evaluation.
- `train_on_hardware=True` also proceeds to this evaluation after hardware fine-tuning.
- Repeated experiments, ZNE selection, or layout/model changes after seeing these test results would invalidate the test as an independent final evaluation.

**Required before any QPU run:** make the development-time IBM evaluation use only validation data, and create a separately authorized one-shot final-test path that does not feed back into selection. Keep the existing test split sealed until that final protocol step. This audit only inspected the source reference; it did not read or access test data.

## 7. IBM Runtime execution modes, caps, and API compatibility

The code uses `SamplerV2(mode=backend)`, i.e. individual **job mode**; it does not instantiate a `Session` or `Batch` for training. Runtime’s current documentation recommends Session for iterative workloads because it avoids repeated queueing after the first job, while Batch is for independently executable jobs without conditional dependencies. Open Plan users cannot submit Session jobs. Therefore:

- Sessions are the natural mode for sequential variational optimization if the account plan supports them.
- Batch can help only for independent work (for example, submitting already-known plus/minus candidates together); it cannot run future optimizer steps whose parameters depend on prior results.
- If only job/Batch mode is available, the current 44-job SPSA path may incur repeated queue delays; a remote QNSPSA path with hundreds of sequential objective jobs would be particularly inefficient.

IBM documents an upper limit of 10 million Sampler executions per job, where executions equal broadcast circuit instances times shots. It also documents per-circuit gate limits, including five million two-qubit gates. The calculated single inference and custom-ZNE jobs above are below the 10-million shot-execution limit absent additional broadcast/twirling overhead. Gate counts and actual ISA depth remain unknown until a specific backend target is queried and circuits are transpiled.

Hardware submission requires ISA-compliant circuits for the selected backend’s instruction set and connectivity. The code calls `transpile(..., backend=..., optimization_level=1)`, but does not freeze/log an initial layout, final layout, backend target/calibration snapshot, or post-transpile depth and gate counts as a cross-qubit comparability record. IBM’s transpilation guide describes layout and routing as backend-dependent transformations; see [IBM transpilation documentation](https://quantum.cloud.ibm.com/docs/en/qiskit/transpile).

The repository pins Runtime 0.46.1, while the current IBM pages consulted document newer Runtime examples (0.50.0). Runtime is not installed in the inspected `.venv312`, so `SamplerV2(mode=backend)`, service channel, options, session, and result compatibility with the pinned environment are **unverified**. Do not upgrade dependencies or treat current docs as proof that the pinned installation works; first make a separately versioned environment and check the exact pinned client API.

Official documentation consulted (2026-11-20):

- [Execution modes: job, Session, Batch](https://quantum.cloud.ibm.com/docs/en/ibm-quantum-compute/execution-modes)
- [Job limits](https://quantum.cloud.ibm.com/docs/en/ibm-quantum-compute/job-limits)
- [Backend information, status, target, and account-visible systems](https://quantum.cloud.ibm.com/docs/en/ibm-quantum-compute/qpu-information)
- [Sampler PUB and result model](https://eu-de.quantum.cloud.ibm.com/docs/en/guides/sampler-input-output)
- [Error mitigation and suppression](https://quantum.cloud.ibm.com/docs/en/guides/error-mitigation-and-suppression-techniques)
- [Plans overview](https://quantum.cloud.ibm.com/docs/en/ibm-quantum-platform/plans-overview)

The Open Plan page currently describes up to the resource-unit equivalent of 10 minutes of QPU time per 28-day rolling window; Open users cannot submit Sessions. This is a public-plan description, not evidence of the account owner’s plan, remaining quota, region, or eligibility. No account, token, or backend was queried.

## 8. Backend, layout, and physical feasibility

The trainer defaults to `ibm_fez`, while the legacy IBM adapter/results collector default to `ibm_kingston`. These are defaults only. IBM documentation says the accessible backend list depends on the authenticated instance and region; a backend name does not establish current status or access. No conclusion can be made here about whether q8/q12/q16/q20 fit an available, operational device or which layout has usable connectivity.

`fine_tune_on_hardware` checks only that `backend.num_qubits >= n_qubits`; that is necessary but insufficient. Before a scaling experiment, the owner must establish:

1. An authenticated, account-visible, operational backend and its current Target, coupling map, supported instruction set, plan, region, queue and usage limits.
2. A connected physical layout for each logical size; preferably nested subgraphs \(P_8 \subset P_{12} \subset P_{16} \subset P_{20}\) on the same backend so the comparison does not silently change device or calibration conditions.
3. A frozen transpiler version, optimization settings, layout/routing policy, transpiler seed, initial/final layout, and saved post-transpile circuit hash, depth, and gate counts for every model.
4. Calibration timestamps and relevant readout/two-qubit error data for each run. Pair or block comparisons in a short time window where possible; do not claim hardware stationarity without evidence.
5. A fallback policy if a connected layout or system is unavailable. A different backend or disconnected layout is a new experimental factor, not a transparent substitute.

The legacy [`IBMQuantumAdapter`](../../src/infrastructure/ibm_quantum_adapter.py#L60-L95) is an inference adapter, not a training route. Its `get_backend_properties` returns a fabricated `num_qubits: 127` after any exception, so it must not be used as evidence of device capability. The legacy [`IBMQuantumResultsCollector`](../../src/infrastructure/ibm_quantum_results_collector.py#L332-L344) explicitly rejects real-hardware execution in `run_full_experiment`.

## 9. Controlled qubit-scaling experiment design

The existing V1 q16/PCA16 versus q20/PCA20 comparison changes both the quantum circuit size and the encoded input dimension. It cannot attribute a performance difference to extra qubits alone. V2 is a single q16/PCA16 condition and is not a scaling comparison.

Use the following separately versioned matrix, sharing a single predeclared dataset split, training-only fitted preprocessing, class order, shot policy, optimizer settings, seed list, and evaluation rules:

| Logical qubits \(q\) | PCA/encoded dimensions \(d=8\) | \(d=12\) | \(d=16\) | \(d=20\) |
|---:|:---:|:---:|:---:|:---:|
| 8 | A + B | — | — | — |
| 12 | A | B | — | — |
| 16 | A | A | B | — |
| 20 | A | A | A | B |

- **Experiment A — added trainable qubits at fixed input:** the \(d=8\) column for q8/q12/q16/q20. The feature map receives the same eight PCA inputs; additional qubits participate in the ansatz. This addresses whether extra trainable/entangled qubits help with a fixed input representation.
- **Experiment B — matched feature/qubit scaling:** the diagonal q=d cells (8/8, 12/12, 16/16, 20/20). This is an end-to-end scaling study, but it intentionally combines added features and qubits and cannot isolate either effect alone.
- The full triangular matrix gives the extra comparisons needed to estimate feature-dimension effects at fixed q and qubit effects at fixed d where \(d \le q\). The A and B cells are subsets of this matrix, not independent protocols.

Fit one PCA basis with the training data only, then use nested prefixes of that same basis for d=8/12/16/20. Fit scaling only on training data and reuse its transform for validation/test. Keep the 650-example validation reporting-only and the test split sealed until final evaluation.

Readout capacity also changes with q. For the existing neighbor-marginal-plus-adjacent-ZZ readout, the feature width is \(2q-1\); with hidden width 16 and 13 classes, the MLP parameter count is \(32q+221\), e.g. 733 at q16. Report parameter totals and include a capacity-control arm (a predeclared fixed-dimensional readout or equivalent classical-capacity control); otherwise any performance trend mixes quantum-qubit count with readout capacity.

## 10. Scientific comparability, metrics, and baselines

### 10.1 Data provenance and physical claims

`LIGOPCAAdapter` uses PyCBC `get_td_waveform`, injects beyond-GR modifications, whitens with a repository-defined analytic O3 PSD approximation, and then performs PCA/scaling. It does not ingest measured LIGO O3 strain. It is a simulated/injected waveform study, not yet evidence that QNIM detects differences in observed astrophysical signals. See [`_generate_base_waveform`](../../src/infrastructure/ligo_pca_adapter.py#L133-L160), the analytic PSD/whitening code at [lines 38-101](../../src/infrastructure/ligo_pca_adapter.py#L38-L101), and the split/PCA/scaler pipeline at [lines 325-422](../../src/infrastructure/ligo_pca_adapter.py#L325-L422).

There is a direct provenance risk: `_generate_base_waveform` defaults `strict_physics=False`, and `LIGOPCAAdapter` also defaults it to false. On PyCBC failure it warns and returns a sinusoidal fallback. The use case constructs `LIGOPCAAdapter` without setting strict mode. Thus a future generation through the standard LIGO-PCA application path can silently mix fallback signals into the supposedly physical dataset. Enable strict failure and retain generator/injection provenance before treating results as physical. This report does not determine whether an already-generated dataset contains fallback records.

Before any physical interpretation, independently audit each injection’s parameterization and label mapping, verify that the modification survives whitening/PCA, preserve GR and modified waveforms plus injection metadata, and include realistic noise/backgrounds and held-out physical parameter ranges. Synthetic class separability alone does not establish separability of real theories in detector data.

### 10.2 Comparability controls

- Hold optimizer equations, initialization policy, stopping rule, shot count, data roles, preprocessing, readout training, and model-selection rule fixed across cells.
- Use the same predeclared seeds across conditions and report every attempted run, including failed jobs and early stops.
- Separate ideal local simulation, backend-calibrated noisy simulation, and physical QPU results. Do not call a simulator using backend-derived noise “hardware execution.”
- Report QPU results with and without application-level ZNE as distinct arms. The repository’s custom scale factors [1, 3, 5] triple PUB count; IBM service-side suppression/mitigation settings must also be recorded and held constant or explicitly treated as another factor. “No custom ZNE” is not necessarily “no mitigation.”
- Report balanced accuracy, accuracy, macro-F1, confusion matrix, per-class recall, cross-entropy/calibration, and uncertainty across seeds and runs. Select models and layouts using validation only.

### 10.3 Classical controls

Train logistic regression, Random Forest, RBF SVM, and a conventional MLP on exactly the same train/validation/test partitions and PCA representations. Fit preprocessing and tune only on train/validation. This answers whether the dataset itself is discriminative and whether a QNIM effect exceeds a classical baseline; it does not require QPU resources.

## 11. Reproducibility and execution audit trail

The V1/V2 runners have useful seed, protocol-hash, data-hash, parameter-hash, and PUB/shot telemetry. V2 records a single sequential `StatevectorSampler(seed=42)` stream for checkpoint measurement; this is reproducible local simulation metadata, not independent-sample evidence and not a model for QPU randomness. V1 states that a new attempt restarts from seed 42 and does not restore optimizer state. See [`V1 attempt policy`](../../scripts/run_qnim_quantum_pca_scaling.py#L30-L51) and [`V2 candidate metadata`](../../scripts/run_qnim_quantum_pca_scaling_v2.py#L467-L530).

The hardware SPSA result can accumulate job IDs, PUBs, shots, and QPU seconds when returned by the evaluator. However, the current flow needs a durable per-job record before/at submission, not only an end-of-run aggregate, so interruption recovery can identify already-submitted work. For a future remote optimizer, persist before waiting: run ID, protocol and data hashes, parameter-vector hash, optimizer state/RNG state, sample indices, sampler configuration, backend/instance identifiers (non-secret), job ID, submission/completion timestamps, PUB count, shots, result hash, error/status, and QPU usage.

Seeds are necessary for reproducibility but do not establish mathematical independence of QPU shot outcomes. A hardware protocol must separately model drift, calibration changes, temporal correlations, job ordering, and the source of any randomized compilation/twirling. Do not state that independently seeded host PRNGs prove independent hardware measurements.

## 12. IBM availability, account, and resource uncertainties

No credential value was read, no IBM service object was constructed, and no backend or account was queried. Therefore the following remain unknown:

- token validity, IBM instance/region, plan, roles, remaining RUs, and whether Sessions are available;
- which named backend is available and operational at run time, its qubit count, topology, calibration, queue, and limits;
- whether the pinned Runtime 0.46.1 client can run the exact current trainer’s SamplerV2 interface in the selected Python environment;
- actual QPU seconds/RU consumption for the planned circuits, including routing, scheduling, server-side mitigation, and queue delay.

Do not use hard-coded backend defaults or the adapter’s 127-qubit exception fallback as confirmation. The account owner must verify the live accessible backend list and plan through the IBM-supported account interface when authorized.

## 13. Implementation gaps to close before a pilot

1. **Seal test data:** add a validation-only IBM evaluation path and a guard that fails if a development run requests `X_test`/`y_test`. Reserve the test split for one final, predeclared evaluation.
2. **Make algorithm identity explicit:** either implement QNSPSA-EML-Feynman objective evaluation on Runtime Sampler with correct hardware batching, or label the existing remote optimizer “hardware SPSA fine-tuning.” Do not present one as the other.
3. **Handle Runtime modes deliberately:** support job mode as a baseline and Session only when the account plan/client supports it; batch simultaneous independent evaluations where useful. Record mode and fallback behavior.
4. **Reduce remote round trips:** batch known evaluations within each outer iteration where mathematically possible (e.g. the two SPSA perturbations and predetermined Feynman points) and keep blocking/update-dependent measurements after their prerequisite results. Preserve current optimizer equations and validate batching equivalence locally before any hardware run.
5. **Lock circuit compilation:** use a backend-derived ISA target, explicit layout policy, fixed transpiler version/options/seed, and persist physical layout, operation counts, depth, circuit hash, and backend calibration metadata.
6. **Fail closed on physics data:** set strict PyCBC mode in the dataset generation entry point, record each event’s generating approximant, injected parameters, and fallback status, and block training if any event is not from the intended generator.
7. **Separate cost accounting:** report local optimizer objective calls, local PUBs/shots, QPU PUBs/shots/jobs, validation jobs, ZNE multiplier, and measured QPU seconds separately. A sampler call is not a circuit count.
8. **Version the protocol:** create a new protocol identity (for example `QNIM_IBM_QPU_SCALING_V1`) with immutable hashes and predeclared q/d matrix, seeds, train/validation/test roles, backend/layout, shots, optimizer, mitigation settings, and resource cap. Do not modify the existing V1 or frozen V2 artifacts.

## 14. Staged roadmap and go/no-go gates

| Stage | Work | Gate to proceed |
|---|---|---|
| 0. Freeze design | Approve the new hardware protocol, matrix, seed policy, statistical metrics, test-seal policy, backend/layout strategy, and resource cap. | Protocol review signed; V1/V2 hashes and files remain unchanged. |
| 1. Data integrity | Strict PyCBC generation, injection audit, event-level provenance, train-only PCA/scaler, verified data hashes. | Zero fallback events; labels and physical injection deltas pass a separately reviewed integrity report. |
| 2. Safe local parity | Implement/identify remote-compatible QNSPSA semantics and a separately named SPSA fine-tune arm; add exact call/job accounting and validation-only IBM path. | Local batched/unbatched objective and optimizer trajectories agree within predeclared tolerance; test-sentinel guard proves no test access. |
| 3. Runtime readiness | Verify pinned or newly versioned Runtime client, authorized account, accessible backend, plan, ISA transpilation, session/batch support, and job limits. | A no-training readiness check and approved small-circuit dry-run/estimator (if allowed by the owner) establish concrete backend/resource feasibility. |
| 4. Simulator screening | Run classical baselines and the planned q/d matrix on the simulator; select shortlisted conditions using validation only. | Dataset shows independently validated discriminative signal and costs fit the approved cap. |
| 5. Hardware pilot | Execute a small, predeclared q8/q12 pilot using identical data roles and layout/calibration logging; compare explicit QPU feedback with simulator results. | Job audit complete, no leakage, resource use within cap, and results stable enough to justify scaling. |
| 6. Scaling and final holdout | Extend to q16/q20 and perform the single authorized final test evaluation after all model/layout/mitigation choices are locked. | Report all seeds/jobs/failures and distinguish simulated injected-waveform evidence from any real-observation claim. |

## 15. Final readiness decision

**Current status: NO-GO for QPU scaling or a scientific claim of QNIM QPU training.** There is an operationally plausible real-QPU SPSA fine-tuning path, but it is not the custom QNSPSA-EML-Feynman optimizer used by the local scaling runners. The V1 and V2 scaling runners themselves submit zero QPU jobs. No backend/account/plan or pinned Runtime compatibility was verified. The standard IBM evaluation path consumes the test split, and the default LIGO-PCA generation route allows PyCBC failure fallback.

**GO only for the next design/engineering stage** after preserving V1/V2 and resolving the blockers above. QPU execution should remain unauthorized until the research owner approves a new protocol, validates strict data provenance, seals the test path, verifies an account-visible backend and Runtime client, chooses the actual remote optimizer, and sets a resource cap.

**Unverified risks to carry forward:** IBM account/backend availability and plan; hardware stationarity and shot independence; exact post-transpilation depth and gate cost; Runtime 0.46.1 compatibility; data-event provenance/fallback status for any already-generated dataset; and whether simulated injection discrimination generalizes to observed gravitational-wave data.
