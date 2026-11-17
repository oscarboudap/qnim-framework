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

## Estructura mínima

- [scripts/generate_results.py](scripts/generate_results.py): entrypoint principal.
- [scripts/diagnose_qnim_classical_baselines.py](scripts/diagnose_qnim_classical_baselines.py): strict-PyCBC classical data screen.
- [src](src): lógica de aplicación e infraestructura.
- [config](config): configuración y perfiles.
- [requirements.txt](requirements.txt): dependencias mínimas.
