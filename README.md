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

By default this uses 20/5/10 train/validation/test events per class across three
seeds, runs logistic regression, random forest, RBF SVM, and an MLP, and writes
`results/scaling_study/qnim_classical_diagnostic.json`. PyCBC failures stop the
run instead of falling back to a synthetic chirp. The test split is not used
for model selection; permutation tests shuffle training labels and refit each
baseline. Dataset sizes, seeds, permutation count, and output path are
configurable through `--help`.

The current LIGO-PCA generator does not inject detector noise or model SNR.
Consequently, a positive report is only a screening signal for the generated
waveforms, not evidence of separability in real detector data.

## Estructura mínima

- [scripts/generate_results.py](scripts/generate_results.py): entrypoint principal.
- [scripts/diagnose_qnim_classical_baselines.py](scripts/diagnose_qnim_classical_baselines.py): strict-PyCBC classical data screen.
- [src](src): lógica de aplicación e infraestructura.
- [config](config): configuración y perfiles.
- [requirements.txt](requirements.txt): dependencias mínimas.
