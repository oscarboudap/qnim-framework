# Q1.6 — Análisis parcial de escalabilidad

- **Estado:** `partial_analysis`
- **Fuente:** `qnim_q1_6_scaling_pilot_v1_partial_20261120.json`
- **SHA-256 de la fuente:** `074FDA49C5A3D4DB1AAAC8ED5F8D160DC3C7B8A6EDA9481073FA40FAE129CB7A`
- **Ejecuciones:** 10/33 completadas; 23 pendientes.
- **Evaluación:** solo validation; test final bloqueado y no accedido.
- **Semillas disponibles:** [42] (inicialización del modelo; un único dataset seed).

## Métricas disponibles

| Modelo | Línea | Qubits | Features | Seeds | Balanced accuracy | Accuracy | Macro-F1 | Cross-entropy | Tiempo total medio |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| QNIM | full_system | 4 | 4 | [42] | 0.0985 (n=1; SD no estimable) | 0.0985 | 0.0947 | 6.3648 | 00:15:07 (906.6 s) |
| QNIM | full_system | 8 | 8 | [42] | 0.1354 (n=1; SD no estimable) | 0.1354 | 0.1356 | 12.1090 | 00:17:03 (1022.5 s) |
| QNIM | full_system | 12 | 12 | [42] | 0.1600 (n=1; SD no estimable) | 0.1600 | 0.1544 | 9.1851 | 00:58:32 (3511.7 s) |
| QNIM | full_system | 16 | 16 | [42] | 0.2123 (n=1; SD no estimable) | 0.2123 | 0.2159 | 7.3531 | 18:43:41 (67421.1 s) |
| QNIM | fixed_input_dimension | 4 | 4 | [42] | 0.0985 (n=1; SD no estimable) | 0.0985 | 0.0947 | 6.3648 | 00:15:07 (906.6 s) |
| QNIM | fixed_input_dimension | 8 | 4 | [42] | 0.0754 (n=1; SD no estimable) | 0.0754 | 0.0587 | 4.3069 | 00:16:36 (996.2 s) |
| QNIM | fixed_input_dimension | 12 | 4 | [42] | 0.0923 (n=1; SD no estimable) | 0.0923 | 0.0736 | 4.1557 | 00:37:22 (2241.5 s) |
| PCA + SVM RBF | full_system | 4 | 4 | [42] | 0.1200 (n=1; SD no estimable) | 0.1200 | 0.0746 | 2.5424 | 00:00:02 (1.8 s) |
| PCA + SVM RBF | full_system | 8 | 8 | [42] | 0.3215 (n=1; SD no estimable) | 0.3215 | 0.2973 | 1.8678 | 00:00:02 (1.6 s) |
| PCA + SVM RBF | full_system | 12 | 12 | [42] | 0.3800 (n=1; SD no estimable) | 0.3800 | 0.3645 | 1.6174 | 00:00:01 (1.2 s) |
| PCA + SVM RBF | full_system | 16 | 16 | [42] | 0.3969 (n=1; SD no estimable) | 0.3969 | 0.3876 | 1.5611 | 00:00:01 (1.2 s) |

La dimensión fija de 4 features en q=4 es un alias del ajuste full-system de q=4; no representa una segunda ejecución independiente.

## Comparaciones pareadas

| Comparación | Qubits | Inicializaciones coincidentes | Diferencia de balanced accuracy (QNIM full − control) |
|---|---:|---|---:|
| QNIM full − PCA + SVM | 4 | 1 | seed 42: -0.0215 |
| QNIM full − PCA + SVM | 8 | 1 | seed 42: -0.1862 |
| QNIM full − PCA + SVM | 12 | 1 | seed 42: -0.2200 |
| QNIM full − PCA + SVM | 16 | 1 | seed 42: -0.1846 |
| QNIM full − QNIM fixed-input | 4 | 1 | seed 42: +0.0000 (mismo ajuste) |
| QNIM full − QNIM fixed-input | 8 | 1 | seed 42: +0.0600 |
| QNIM full − QNIM fixed-input | 12 | 1 | seed 42: +0.0677 |
| QNIM full − QNIM fixed-input | 16 | 0 | sin comparación disponible |

## Ejecuciones pendientes

La ejecución interrumpida fue QNIM fixed-input, 16 qubits, 4 features, semilla 42; no llegó a registrarse. Las demás claves pendientes son:

- `pca_svm_rbf` / `full_system` / q=4 / seed=43
- `pca_svm_rbf` / `full_system` / q=4 / seed=44
- `pca_svm_rbf` / `full_system` / q=8 / seed=43
- `pca_svm_rbf` / `full_system` / q=8 / seed=44
- `pca_svm_rbf` / `full_system` / q=12 / seed=43
- `pca_svm_rbf` / `full_system` / q=12 / seed=44
- `pca_svm_rbf` / `full_system` / q=16 / seed=43
- `pca_svm_rbf` / `full_system` / q=16 / seed=44
- `qnim` / `fixed_input_dimension` / q=8 / seed=43
- `qnim` / `fixed_input_dimension` / q=8 / seed=44
- `qnim` / `fixed_input_dimension` / q=12 / seed=43
- `qnim` / `fixed_input_dimension` / q=12 / seed=44
- `qnim` / `fixed_input_dimension` / q=16 / seed=42
- `qnim` / `fixed_input_dimension` / q=16 / seed=43
- `qnim` / `fixed_input_dimension` / q=16 / seed=44
- `qnim` / `full_system` / q=4 / seed=43
- `qnim` / `full_system` / q=4 / seed=44
- `qnim` / `full_system` / q=8 / seed=43
- `qnim` / `full_system` / q=8 / seed=44
- `qnim` / `full_system` / q=12 / seed=43
- `qnim` / `full_system` / q=12 / seed=44
- `qnim` / `full_system` / q=16 / seed=43
- `qnim` / `full_system` / q=16 / seed=44

## Lectura y límites

- La balanced accuracy de QNIM full-system aumenta en cada configuración disponible.
- PCA + SVM supera a QNIM full-system en los qubits disponibles: q=4, q=8, q=12, q=16.
- Una sola inicialización y un solo dataset no permiten estimar variabilidad entre semillas ni demostrar escalabilidad reproducible.
- En full-system cambian a la vez el número de qubits y las componentes PCA; el control SVM no aísla completamente esos factores.
- Los tiempos totales externos se usan para comparar configuraciones. Se conservan aparte los tiempos internos del entrenador.

### Diferencias en los tiempos registrados

- `qnim/full_system/q=16/seed=42`: externo 18:43:41 (67421.1 s), interno 22:36:19 (81378.6 s). Se reportan ambos sin intentar reconciliarlos.

### Errores conservados en el informe fuente

- `ModuleNotFoundError` (train_validation_generation): No module named 'src'
- `ValueError` (train_validation_generation): The 12-component prefix does not reproduce the frozen Q1 train/validation split hashes
- Son errores de intentos de preparación previos; la copia registra que el dataset se preparó posteriormente y que los hashes coincidieron.

No se entrenaron modelos, no se regeneró el dataset y no se accedió al test.
