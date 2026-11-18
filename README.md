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

## Estructura mínima

- [scripts/generate_results.py](scripts/generate_results.py): entrypoint principal.
- [scripts/diagnose_qnim_classical_baselines.py](scripts/diagnose_qnim_classical_baselines.py): strict-PyCBC classical data screen.
- [scripts/run_qnim_q1.py](scripts/run_qnim_q1.py): frozen-reference QNIM simulator experiment.
- [scripts/run_qnim_q1_1.py](scripts/run_qnim_q1_1.py): instrumented Q1.1 diagnostics and ablations.
- [src](src): lógica de aplicación e infraestructura.
- [config](config): configuración y perfiles.
- [requirements.txt](requirements.txt): dependencias mínimas.
