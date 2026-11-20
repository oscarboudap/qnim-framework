# QNIM

Proyecto mínimo para ejecutar la pipeline experimental de QNIM con el entrypoint principal.

## Uso rápido

```bash
python scripts/generate_results.py --mode sim
python scripts/generate_results.py --mode ibm
```

## Perfil 27 qubits

```bash
python scripts/generate_results.py --from-disk --mode sim --profile ibm_27q_proposal --max-iter 1
```

Perfiles disponibles en [config/experiment_profiles.json](config/experiment_profiles.json).

## Classical diagnostic before QNIM

Run a small, strict-PyCBC LIGO-PCA screening experiment before spending time on
quantum training:

```bash
python scripts/diagnose_qnim_classical_baselines.py
```

By default this uses 200/50/100 train/validation/final-test events per class
across the predeclared seeds 42, 43, and 44, runs logistic regression, random
forest, RBF SVM, and an MLP, and performs 999 training-label permutations.
PyCBC failures stop the run instead of falling back to a synthetic chirp. The
report records the resulting number of permutation refits before the experiment
starts; use smaller explicitly chosen sizes only for technical smoke tests.

The `validation` report contains the mean of per-seed balanced accuracies and a
permutation test that independently shuffles each seed's training labels,
refits, and evaluates on unchanged validation labels. Advancement requires a
mean above chance, every seed above chance, and a permutation p-value at most
0.05 after Bonferroni correction across the four baselines. The final test is
locked and not evaluated by default; only run with `--evaluate-final-test` after
all model and protocol choices, including QNIM, are frozen. Its metrics are
written separately under `final_test`. Seeds, dataset sizes, permutation count,
random-number derivation, and Python/scientific dependency versions are recorded
in the JSON report. Each permutation uses an independent NumPy
`SeedSequence([dataset_seed, model_index, permutation_iteration])`.

The current LIGO-PCA generator does not inject detector noise or model SNR.
Consequently, a positive report is only a screening signal for the generated
waveforms, not evidence of separability in real detector data.

## QNIM Q1 simulator experiment

The classical pilot is frozen as `CLASSICAL_REFERENCE_V1`; its manifest pins
the source report by SHA-256. Q1 regenerates each strict-PyCBC dataset and stops
unless its train and validation hashes exactly match that reference. It uses
the same 12 PCA components, three seeds, 12-qubit ZZFeatureMap and
EfficientSU2 (two repetitions, linear entanglement), and the existing 12-qubit
marginal/MLP readout. The current circuit has 72 ansatz parameters and 429
readout parameters. Only train data is passed to the optimizer; validation is
evaluated after training, and the final test is not read or evaluated.

Run Q1 in the configured Python environment:

```bash
python scripts/run_qnim_q1.py
```

Defaults are 100 optimizer iterations and 512 shots. The JSON is written
incrementally, records the three validation results separately, and refuses to
overwrite an existing result file. A completed result is evidence about the
synthetic dataset only, not about real LIGO observations or quantum advantage.

## QNIM Q1.1 diagnostic

Q1.1 keeps the Q1 architecture and dataset fixed, pins the seed-42
train/validation split to `CLASSICAL_REFERENCE_V1`, and varies only parameter
initialization seeds (42, 43, 44) while holding optimizer, batch, and training
sampler randomness fixed at 42. It records train and
validation cross-entropy, balanced accuracy, per-class recall, prediction
counts, confusion matrices, and parameter snapshots. Validation is observed
only after each optimizer step and never updates parameters; the final test
remains locked.

The four comparisons are: A, the instrumented current QNIM run; B, a refit of
the classical MLP checked against the frozen seed-42 score; C, a frozen random
ansatz with a trainable readout; and D, the trainable QNIM ansatz. A and D use
the same execution, avoiding a duplicate fit. Diagnostic measurements use
separate deterministic simulator seeds so they do not consume the optimizer's
sampler sequence.

```bash
python scripts/run_qnim_q1_1.py
```

The default run evaluates all 3,250 train and validation examples after every
QNIM iteration and trains each frozen-ansatz readout for 1,500 iterations;
this is an intentionally expensive diagnostic and is not launched
automatically. The JSON is written atomically after the baseline and each
condition/initialization seed. `--resume` verifies the reference hashes,
protocol, configuration, and dependency versions, then skips completed runs;
an interrupted seed is restarted. Existing output is never overwritten without
`--resume`.

Run the reduced integration smoke (one initialization seed, one QNIM
iteration, eight shots, two readout iterations) against the frozen
`qnim_classical_diagnostic_smoke_v2.json` split:

```bash
python scripts/run_qnim_q1_1.py --smoke
```

This smoke exercises A–D on the small strict-PyCBC split and is not scientific
evidence. Before the full run, time one diagnostic evaluation over the complete
3,250-example seed-42 train/validation split:

```bash
python scripts/run_qnim_q1_1.py --benchmark-diagnostic-iteration
```

The benchmark writes `results/scaling_study/qnim_q1_1_diagnostic_cost_v1.json`
and leaves the final test locked.

## QNIM Q1.2 controlled diagnostics

Q1.2 adds three controlled checks without changing the 12-qubit circuit:

- **A:** trainable and frozen-ansatz runs start from the same complete
  ansatz/readout vector, training seed and minibatch sequence, use the same
  readout-refit schedule, and run a fixed number of updates. The frozen arm
  optimizes only readout parameters.
- **B:** compares exact statevector marginals with finite-shot marginals for
  the same seed-42 circuit and train/validation examples, then fits matching
  readouts on train only.
- **C:** records training minibatch/reference losses, proposed and accepted
  optimizer step norms, blocking decisions, gradient variance, parameter
  snapshots, and the iteration where the original patience rule would have
  stopped. This replay does not alter the matched fixed-update runs.

Run the reduced integration smoke first:

```bash
python scripts/run_qnim_q1_2.py --smoke
```

It uses the frozen small PyCBC split, one initialization, two updates, and
eight shots; its results are not scientific evidence. The full command uses
three initializations, 60 matched updates, and 512 shots:

```bash
python scripts/run_qnim_q1_2.py
```

This full run is intentionally not started automatically. Both modes keep the
final test locked and write atomic checkpoints after each comparison/run;
`--resume` validates the frozen hashes and configuration before continuing.
Outputs are stored in `results/scaling_study/qnim_q1_2_v1.json` or
`results/scaling_study/qnim_q1_2_smoke_v1.json`.

## QNIM Q1.3 frozen representation probes

Q1.3 extracts exact statevector qubit marginals from the seed-specific initial
ansatz and the corresponding final Q1.2 trainable ansatz. Logistic regression,
RBF SVM, and MLP probes fit on train features only; validation is used only for
metrics. The MLP is allowed 2,000 iterations without validation-based early
stopping. The probes report balanced accuracy, cross-entropy, class recall, and
confusion matrices. This measures the diagnostic utility of these frozen
features, not quantum advantage. The final test remains locked.

Run the reduced integration smoke first:

```bash
python scripts/run_qnim_q1_3.py --smoke
```

The full run uses all three Q1.2 initialization seeds and checkpoints each
initial/trained circuit probe atomically. It requires a completed
`qnim_q1_2_q1_3_parameters_v1.json` report with exact final ansatz vectors:

```bash
python scripts/run_qnim_q1_3.py
```

Q1.2 reports created before exact final-vector export are rejected; rerun Q1.2
to a new output path before running Q1.3. This repeats the full, potentially
long Q1.2 diagnostic and does not overwrite the existing report.

Resume an interrupted run with `--resume`. Outputs are stored in
`results/scaling_study/qnim_q1_3_v2.json` or
`results/scaling_study/qnim_q1_3_smoke_v2.json`.

## QNIM Q1.4 correlation readout probes

Q1.4 keeps the Q1.2 initial and trained circuits fixed and compares four exact
statevector representations: 12 P(q_i=0) marginals, those marginals plus 11
adjacent ZZ correlators, those marginals plus all 66 pairwise ZZ correlators,
and 36 local XYZ expectations. Logistic regression and RBF SVM are fit on
train only, alongside a raw 12-component PCA control using the same classifiers.
The A marginal features are hash-checked against Q1.3; validation is used only
for metrics, and the final test remains locked.

Run the reduced smoke first:

```bash
python scripts/run_qnim_q1_4.py --smoke
```

The full run uses all three initialization seeds and requires the completed
Q1.2 report with exact final vectors and the matching Q1.3 v2 report:

```bash
python scripts/run_qnim_q1_4.py
```

Both modes checkpoint each PCA control and circuit probe atomically and support
`--resume`. Outputs are stored in
`results/scaling_study/qnim_q1_4_v1.json` or
`results/scaling_study/qnim_q1_4_smoke_v1.json`.

## QNIM scalable readout

The simulator scaling runner keeps its historical marginal readout by default.
To train the same QNIM pipeline with all pairwise ZZ correlators appended to
the qubit marginals, select `marginals_zz`:

```bash
python scripts/run_scaling_study.py --dataset-dir <dataset> --readout-mode marginals_zz
```

For `q` qubits this readout has `q + q(q-1)/2` inputs. The marginal-only mode
remains available as the control. Marginals and ZZ correlators are estimated
from the same computational-basis samples; the larger readout does increase
the number of trainable classical parameters, so that capacity difference
must be considered when interpreting scaling results. This option does not
change the frozen Q1 reports or evaluate the final test.

For a linearly growing readout, select `marginals_zz_neighbors` to append only
the ZZ correlations between adjacent qubits in the linear register:

```bash
python scripts/run_scaling_study.py --dataset-dir <dataset> --readout-mode marginals_zz_neighbors
```

This readout has `2q - 1` inputs for `q` qubits. Marginals and adjacent-pair
correlators are computed from the same computational-basis samples. The mode is
opt-in; historical experiments continue to use their recorded readout.

## QNIM neighbor-readout pilot

The approved `QNIM_SCALING_NEIGHBOR_PILOT_V1` has a dedicated resumable runner.
It validates the immutable
`results/scaling_study/qnim_neighbor_readout_pilot_v1_dataset_manifest.json`
and its pinned SHA-256, then checks the cached train/validation hashes and
historical 12-component prefix before execution. It never loads test data or
regenerates splits.

Validate the plan without fitting models:

```bash
python scripts/run_qnim_neighbor_readout_pilot.py --dry-run
```

The approved budget is 20 iterations, batch size 32, and 512 shots. Since
patience is also 20, it is not expected to substantially shorten that budget.
To start the experiment later, run:

```bash
python scripts/run_qnim_neighbor_readout_pilot.py
```

For one explicitly selected QNIM fit, specify all of `--arm`, `--qubits`, and
`--seed`. Filtered runs use a separate smoke output directory and do not
contaminate the complete pilot:

```bash
python scripts/run_qnim_neighbor_readout_pilot.py --arm full_system --qubits 4 --seed 42
```

Resume an interrupted run with `--resume`. Checkpoints and the copied protocol
and dataset manifest are stored under
`results/scaling_study/qnim_neighbor_readout_pilot_v1/`.

## Estructura mínima

- [scripts/generate_results.py](scripts/generate_results.py): entrypoint principal.
- [scripts/diagnose_qnim_classical_baselines.py](scripts/diagnose_qnim_classical_baselines.py): strict-PyCBC classical data screen.
- [scripts/run_qnim_neighbor_readout_pilot.py](scripts/run_qnim_neighbor_readout_pilot.py): resumable local-simulator pilot with train/validation integrity checks.
- [scripts/run_qnim_q1.py](scripts/run_qnim_q1.py): frozen-reference QNIM simulator experiment.
- [scripts/run_qnim_q1_1.py](scripts/run_qnim_q1_1.py): instrumented Q1.1 diagnostics and ablations.
- [scripts/run_qnim_q1_2.py](scripts/run_qnim_q1_2.py): matched ansatz, shot-noise, and optimizer diagnostics.
- [scripts/run_qnim_q1_3.py](scripts/run_qnim_q1_3.py): exact frozen-circuit feature probes.
- [scripts/run_qnim_q1_4.py](scripts/run_qnim_q1_4.py): exact correlation and local-Pauli probes.
- [src](src): lógica de aplicación e infraestructura.
- [config](config): configuración y perfiles.
- [requirements.txt](requirements.txt): dependencias mínimas.
