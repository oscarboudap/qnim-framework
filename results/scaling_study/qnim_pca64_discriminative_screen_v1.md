# QNIM PCA64 discriminative scaling screen

## Scope and protocol

- Dataset: `qnim_q1_6_dataset_v2_pca64`; only train and validation arrays were loaded.
- No final test data, QNIM training, or IBM hardware was used. Source dataset and prior reports were not modified.
- The same 2,600 training and 650 validation examples and labels were used at every dimension.
- For each PCA prefix, `StandardScaler` was fit on train only, followed by RBF `SVC(C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=42)`. No hyperparameters were tuned on validation.
- `fit_seconds` includes scaler fitting and SVC fitting/calibration. `inference_seconds` includes validation scaling, `predict`, and `predict_proba`.

## Validation results

| PCA dims | Balanced accuracy | Global accuracy | Macro-F1 | Cross-entropy | Δ BA vs 16 | Δ accuracy | Δ macro-F1 | Δ cross-entropy | Fit s | Inference s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.3785 | 0.3785 | 0.3827 | 1.5627 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0.861 | 0.128 |
| 20 | 0.4308 | 0.4308 | 0.4527 | 1.4066 | +0.0523 | +0.0523 | +0.0700 | -0.1561 | 0.877 | 0.129 |
| 24 | 0.4354 | 0.4354 | 0.4595 | 1.4002 | +0.0569 | +0.0569 | +0.0768 | -0.1625 | 0.912 | 0.125 |
| 32 | 0.4462 | 0.4462 | 0.4733 | 1.3851 | +0.0677 | +0.0677 | +0.0907 | -0.1776 | 0.865 | 0.125 |

### Per-class recall

| Class | PCA16 | PCA20 | PCA24 | PCA32 |
|---|---:|---:|---:|---:|
| GR | 0.100 | 0.120 | 0.140 | 0.180 |
| standard-siren | 0.220 | 0.280 | 0.300 | 0.300 |
| qnm-21 | 0.380 | 0.480 | 0.520 | 0.520 |
| qnm-33 | 0.640 | 0.580 | 0.600 | 0.580 |
| pn-deformation | 0.580 | 0.600 | 0.600 | 0.680 |
| extra-dimensions | 0.280 | 0.380 | 0.360 | 0.360 |
| scalar-tensor | 0.160 | 0.200 | 0.180 | 0.180 |
| graviton-mass | 0.080 | 0.300 | 0.300 | 0.260 |
| chern-simons | 0.920 | 0.920 | 0.920 | 0.920 |
| liv-alpha2 | 0.540 | 0.600 | 0.600 | 0.600 |
| liv-alpha4 | 0.320 | 0.400 | 0.460 | 0.480 |
| loop-quantum-gravity | 0.040 | 0.080 | 0.080 | 0.200 |
| gup | 0.660 | 0.660 | 0.600 | 0.540 |

All four 13×13 confusion matrices and their count differences against PCA16 are in the JSON report; rows are true classes and columns are predicted classes, in this order: GR, standard-siren, qnm-21, qnm-33, pn-deformation, extra-dimensions, scalar-tensor, graviton-mass, chern-simons, liv-alpha2, liv-alpha4, loop-quantum-gravity, gup.

The highest observed validation balanced accuracy is PCA32 (difference vs PCA16: +0.0677). This is a descriptive comparison on one fixed validation set, not a significance test or an independent test result.

## PCA subspace stability and eigenvalue gaps

The source analysis used 20 **stratified subsamples**, not bootstrap samples: 160 of 200 training events per class (2,080 total), without replacement. For PC20/PC24, the sample indices were reconstructed from the saved seeds and strategy, then the randomized PCA fits were repeated on the existing raw training matrix; no waveforms were regenerated. Reconstructed PC16/PC32 angles match the saved values within the recorded tolerance.

Percentiles are over all principal angles pooled across the 20 resamples. The eigenvalue gap is `(lambda_k - lambda_(k+1))/lambda_k`; boundary variance ratios are from the full training-only PCA.

| Boundary k | Median angle ° | P90 ° | P95 ° | P99 ° | Maximum ° | Relative eigenvalue gap | Variance ratio at PCk | Variance after k through PC64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.2050 | 3.9476 | 6.1956 | 11.1220 | 14.0929 | 0.194783 | 1.614e-06 | 1.116e-03% |
| 20 | 0.3498 | 5.7191 | 8.8137 | 18.5151 | 22.7020 | 0.18101 | 8.095e-07 | 7.120e-04% |
| 24 | 0.6875 | 7.5651 | 11.2876 | 18.4848 | 30.4975 | 0.109795 | 4.202e-07 | 4.957e-04% |
| 32 | 0.9110 | 9.6430 | 16.8115 | 46.6206 | 81.5564 | 0.046781 | 1.799e-07 | 2.776e-04% |

## Conclusions and limitations

- This screen tests whether fixed SVC validation metrics change with PCA dimension; it does not establish generalization beyond this validation set.
- The JSON records paired metric differences, per-class recall changes, confusion matrices, timings, all angle distributions, and eigenvalue gaps.
- At PC32, components 33–64 account for 0.000277625% of total variance. Compare this low-variance tail with the PC32 angle distribution and eigenvalue gap: instability in these directions is consistent with weakly determined, low-variance PCA directions, not by itself evidence of class-discriminative information.
- Low-variance directions can have unstable bases even when cumulative explained variance is high; PCA subspace stability should be considered alongside SVC performance.
- Probability calibration is internal to SVC and uses training data. The validation set was used only for the requested final comparison.
