# Diagnóstico científico QNIM q16/PCA16

## Alcance y fuentes

Análisis de solo lectura del intento preservado `803f64cdd3424d1e960c646ef7e53084`. No se entrenó, simuló ni ejecutó ningún circuito; no se cargaron archivos de datos ni se accedió a test. Se usaron el resultado, checkpoint, informe recuperado, metadatos de clases y código fuente existentes. Las métricas por iteración se calcularon sobre los subconjuntos diagnósticos fijos guardados en el resultado: 5 ejemplos por clase (65 train y 65 validation), salvo las métricas finales, que usan 2,600 train y 650 validation.

Fuentes principales: [result.json](./checkpoints/q16/attempt_01_803f64cdd3424d1e960c646ef7e53084/result.json), [checkpoint.json](./checkpoints/q16/attempt_01_803f64cdd3424d1e960c646ef7e53084/checkpoint.json), [result.recovered.md](./checkpoints/q16/attempt_01_803f64cdd3424d1e960c646ef7e53084/result.recovered.md), [runner](../../../scripts/run_qnim_quantum_pca_scaling.py), [trainer Qiskit](../../../src/infrastructure/qiskit_vqc_trainer.py), [optimizador](../../../src/infrastructure/qnspsa_eml_feynman.py), [adaptador LIGO-PCA](../../../src/infrastructure/ligo_pca_adapter.py) y [metadatos PCA64](../qnim_q1_6_dataset_v2_pca64/dataset_metadata.json).

## 1. Trayectoria de optimización

El optimizador completó 20/20 iteraciones, sin declarar convergencia (`max_iterations_reached`). Aceptó 9 actualizaciones y rechazó 11. Los rechazos conservan el paso aplicado en cero, aunque el paso propuesto y la varianza del gradiente SPSA sigan registrados. La varianza guardada va de 0.0456 a 217.4504; **no** es una norma del gradiente y no se infiere una de ella.

La tabla usa las pérdidas de minibatch de 32 ejemplos que registra cada callback. Los diagnósticos de validation son accuracy / balanced accuracy / macro-F1 sobre los mismos 65 ejemplos en todas las iteraciones. No se calcula post-refit cuando no hubo refit. `step/propuesto` son las normas guardadas por el optimizador.

| Iter. | Pre-refit loss | Pre-refit val acc / bal. acc / F1 | Post-refit loss | Post-refit val acc / bal. acc / F1 | Actualización | Step / propuesto | Varianza grad. |
|---:|---:|---|---:|---|---|---:|---:|
| 1 | 3.3349 | 0.0615 / 0.0615 / 0.0564 | — | — | Aceptada | 0.0200 / 0.0200 | 0.6885 |
| 2 | 3.7279 | 0.0615 / 0.0615 / 0.0564 | — | — | Aceptada | 0.0132 / 0.0132 | 0.6423 |
| 3 | 3.0979 | 0.0615 / 0.0615 / 0.0564 | — | — | Rechazada | 0.0000 / 0.0264 | 3.9597 |
| 4 | 3.7692 | 0.0615 / 0.0615 / 0.0564 | — | — | Rechazada | 0.0000 / 0.0552 | 47.5724 |
| 5 | 4.7701 | 0.0615 / 0.0615 / 0.0564 | — | — | Rechazada | 0.0000 / 0.0465 | 26.8020 |
| 6 | 2.9793 | 0.0462 / 0.0462 / 0.0294 | — | — | Aceptada | 0.0354 / 0.0354 | 41.7279 |
| 7 | 3.1415 | 0.0462 / 0.0462 / 0.0294 | — | — | Rechazada | 0.0000 / 0.0221 | 6.1819 |
| 8 | 3.2276 | 0.0462 / 0.0462 / 0.0294 | — | — | Rechazada | 0.0000 / 0.0018 | 0.0456 |
| 9 | 3.6866 | 0.0462 / 0.0462 / 0.0294 | — | — | Rechazada | 0.0000 / 0.0229 | 9.0319 |
| 10 | 3.8643 | 0.0462 / 0.0462 / 0.0294 | 7.7086 | 0.1385 / 0.1385 / 0.1341 | Aceptada | 0.0369 / 0.0369 | 30.8618 |
| 11 | 5.9939 | 0.1385 / 0.1385 / 0.1341 | — | — | Rechazada | 0.0000 / 0.0289 | 217.4504 |
| 12 | 6.2041 | 0.1385 / 0.1385 / 0.1341 | — | — | Rechazada | 0.0000 / 0.0276 | 40.0211 |
| 13 | 6.5248 | 0.1385 / 0.1385 / 0.1341 | — | — | Rechazada | 0.0000 / 0.0381 | 30.2904 |
| 14 | 4.3029 | 0.1231 / 0.1231 / 0.1201 | — | — | Aceptada | 0.0042 / 0.0042 | 0.3511 |
| 15 | 5.4345 | 0.1231 / 0.1231 / 0.1201 | — | — | Rechazada | 0.0000 / 0.0343 | 34.1166 |
| 16 | 7.8873 | 0.1385 / 0.1385 / 0.1458 | — | — | Aceptada | 0.0260 / 0.0260 | 16.5005 |
| 17 | 5.4601 | 0.1385 / 0.1385 / 0.1458 | — | — | Rechazada | 0.0000 / 0.0122 | 215.8163 |
| 18 | 6.0784 | 0.1231 / 0.1231 / 0.1201 | — | — | Aceptada | 0.0034 / 0.0034 | 0.2709 |
| 19 | 5.3464 | 0.1231 / 0.1231 / 0.1293 | — | — | Aceptada | 0.0322 / 0.0322 | 24.8789 |
| 20 | 4.1226 | 0.1231 / 0.1231 / 0.1176 | 4.4046 | 0.1385 / 0.1385 / 0.1049 | Aceptada | 0.0284 / 0.0284 | 20.1947 |

Las pérdidas de minibatch fluctúan; no hay descenso consistente. La pérdida de referencia fija fue mínima en iteración 1 (3.396796) y el valor final guardado sigue siendo ese mínimo. Tras el refit de iteración 10, la pérdida de referencia de la iteración 11 sube a 5.947227 y termina en 5.978092. El aumento de validation diagnóstica después del primer refit no se acompaña de una mejora de pérdida: el CE de validation sobre el subconjunto fijo empeora de 3.1725 a 6.5799.

**Refits (mismo minibatch y mismos subconjuntos diagnósticos antes y después):**

| Iteración / estado | Loss minibatch | Train diagnóstico CE | Validation diagnóstico CE | Train acc / bal. acc / F1 | Validation acc / bal. acc / F1 |
|---|---:|---:|---:|---|---|
| 10 pre | 3.8643 | 3.6663 | 3.1725 | 0.1077 / 0.1077 / 0.0782 | 0.0462 / 0.0462 / 0.0294 |
| 10 post | 7.7086 | 7.4122 | 6.5799 | 0.0769 / 0.0769 / 0.0704 | 0.1385 / 0.1385 / 0.1341 |
| 20 pre | 4.1226 | 7.3474 | 6.6326 | 0.0769 / 0.0769 / 0.0692 | 0.1231 / 0.1231 / 0.1176 |
| 20 post | 4.4046 | 5.5553 | 4.3715 | 0.0769 / 0.0769 / 0.0749 | 0.1385 / 0.1385 / 0.1049 |

Los refits no mejoran de manera uniforme: en iteración 10 empeoran fuertemente ambas pérdidas, aunque suban los indicadores de clasificación de validation; en iteración 20 mejoran el CE diagnóstico pero bajan macro-F1 de validation. El código vuelve a inicializar y ajustar el MLP con 200 ejemplos train en cada llamada; los dos refits guardan los mismos índices train. En cada refit los parámetros cuánticos no cambian, pero la norma L2 del cambio de la cabeza clásica es 45.1463 y 49.0004, respectivamente.

### Estado que produjo el resultado final

El vector final guardado coincide elemento por elemento con el snapshot **pre-refit de la iteración 1**, no con el estado de la iteración 20. El optimizador conserva el `best_theta` según la pérdida de referencia train, actualizada en la iteración 1; los callbacks de refit posteriores mutan el estado vivo, pero no reemplazan ese mejor snapshot en la selección final. Por tanto, las métricas finales sobre los 2,600/650 ejemplos describen el estado seleccionado de iteración 1, no el último estado ni el post-refit de iteración 20.

El bloque cuántico sí se movió desde la inicialización hasta ese snapshot: delta L2 0.006844 (norma inicial 0.074999; máximo cambio por coordenada 0.000893). En las ocho actualizaciones aceptadas de iteraciones 2–20, la suma de longitudes L2 por bloque fue 0.061409 para los 96 parámetros cuánticos y 0.168765 para los 733 clásicos. Estas sumas son longitudes de trayectoria, no desplazamientos netos. El cambio de los refits clásicos, por sí solo, es mucho mayor que el movimiento cuántico observado.

## 2. Cross-entropy y probabilidades

- `_readout_mlp_probs` calcula `h = ReLU(W1 @ features + b1)`, logits `W2 @ h + b2`, resta `max(logits)`, y usa `exp(logits) / sum(exp(logits))`. Restar el máximo es la estabilización numérica estándar del softmax.
- La pérdida de entrenamiento es la media de `-log(p_clase_correcta)`, después de limitar probabilidades a `[1e-10, 1]`. Las métricas finales usan `sklearn.metrics.log_loss` sobre probabilidades limitadas a `[1e-12, 1]`, con etiquetas explícitas `0..12`. Scikit-learn valida que cada fila represente una distribución normalizada; no hay una normalización explícita adicional en `_classification_metrics`.
- La accuracy de validation final es 0.107692 (70/650), algo por encima de `1/13 = 0.076923`, pero el CE es 3.395866 frente al CE uniforme `ln(13) = 2.564949`. Esto no es una contradicción: argmax puede acertar ligeramente más que el azar, mientras la probabilidad asignada a la clase verdadera sea muy baja en muchos ejemplos. El CE equivale a una media geométrica de probabilidad verdadera `exp(-3.395866) ≈ 0.0335`, menor que `1/13`; no permite reconstruir la distribución por ejemplo.
- No se guardaron probabilidades por ejemplo. Por ello no se puede recalcular el CE desde observaciones individuales ni atribuirlo a casos concretos. Los CE diagnósticos de refit, especialmente 6.5799 en validation tras el refit 10, sí son consistentes con probabilidades mal calibradas o confiadas en clases incorrectas, pero no identifican una causa individual.

No se observa una inestabilidad de softmax o un error de clipping que explique por sí solo el CE. El rendimiento y los CE guardados son coherentes con una clasificación pobre y una confianza probabilística deficiente; la falta de probabilidades persistidas limita el diagnóstico más fino.

## 3. Etiquetas, clases y matriz de confusión

El orden de clases del [adaptador](../../../src/infrastructure/ligo_pca_adapter.py) y de los [metadatos del dataset](../qnim_q1_6_dataset_v2_pca64/dataset_metadata.json) es idéntico. El generador asigna `label = cls` mientras recorre ese orden; el runner valida que train y validation contengan las etiquetas enteras `0..12`, con 200 y 50 ejemplos por clase. El trainer construye el one-hot con `y_onehot[row, y]`, y el MLP tiene 13 filas de salida. No hay un paso de decodificación separado que pudiera intercambiar las clases 11 y 12.

| ID | Clase |
---:|---|
| 0 | GR |
| 1 | standard-siren |
| 2 | qnm-21 |
| 3 | qnm-33 |
| 4 | pn-deformation |
| 5 | extra-dimensions |
| 6 | scalar-tensor |
| 7 | graviton-mass |
| 8 | chern-simons |
| 9 | liv-alpha2 |
| 10 | liv-alpha4 |
| 11 | loop-quantum-gravity |
| 12 | gup |

La matriz de `sklearn.confusion_matrix(y_true, y_pred, labels=range(13))` usa filas como clase verdadera y columnas como clase predicha. En la validation completa, las 13 filas tienen soporte 50; los conteos de predicción por columna son `[12, 47, 105, 38, 38, 59, 116, 47, 95, 46, 47, 0, 0]`. Hubo cero predicciones para las clases 11 y 12, que sí están presentes en el soporte; sus recalls son ambos cero. La clase 0 tuvo 12 predicciones y ningún acierto.

La evidencia apunta a colapso/comportamiento del clasificador, no a un error en el mapeo de clases. No demuestra qué logits o señales hicieron que la cabeza dejara de elegir las clases 11 y 12.

## 4. Circuito y readout cuántico

La configuración persistida es `ZZFeatureMap(16, reps=2, entanglement=linear)` seguido de `EfficientSU2(16, reps=2, entanglement=linear)`, medido en 16 qubits. El vector tiene 96 parámetros de ansatz. El mapa enlaza los parámetros de entrada obtenidos de `feature_map.parameters` con el vector de 16 componentes; `_bind_sample` enlaza los primeros 96 parámetros entrenables con `ansatz.parameters`. El dimensionamiento registrado incluye 112 parámetros de circuito, de los que 16 son entradas de datos y 96 son parámetros entrenables.

El modo `marginals_zz_neighbors` forma 31 características en este orden: `P(q_i=0)` para `i=0..15`, seguidas de `⟨Z_i Z_(i+1)⟩` para `i=0..14`. El decodificador respeta la convención little-endian de Qiskit: el último carácter del bitstring corresponde a qubit 0. Las correlaciones adyacentes se calculan multiplicando los autovalores Z de cada par. Las marginales están acotadas por `[0,1]` y las correlaciones por `[-1,1]` cuando se calculan desde conteos válidos. El orden de entradas al MLP es primero las 16 marginales y después las 15 correlaciones.

La cabeza tiene dimensión oculta 16, activación ReLU y salida softmax de 13 clases; sus 733 parámetros se descomponen en `W1(16×31)`, `b1(16)`, `W2(13×16)` y `b2(13)`. Los 829 parámetros optimizados concatenan los 96 del ansatz y esos 733 del MLP. El objetivo completo usa el vector combinado, así que las perturbaciones SPSA pueden afectar ambos bloques.

La inicialización cuántica es normal con desviación 0.01; la cabeza se inicializa mediante un ajuste clásico sobre features medidas de 200 eventos train. Los refits repiten ese ajuste para los mismos 200 índices train y sustituyen la cabeza, no los parámetros cuánticos. Los snapshots/hash y las formas de features están guardados, pero no se conservaron los vectores brutos de medición; por ello no se pueden auditar empíricamente aquí todos sus valores ni medir la separabilidad de los observables antes de la cabeza clásica. El código de binding/decodificación no presenta un defecto evidente que explique el resultado.

## 5. Papel de la cabeza clásica

La cabeza clásica contiene 7.64 veces más parámetros que el ansatz (733 frente a 96). En las actualizaciones aceptadas posteriores a la primera, la longitud acumulada de movimiento del bloque clásico fue aproximadamente 2.75 veces la del bloque cuántico. Además, los refits causaron cambios clásicos L2 de 45.15 y 49.00, sin mover el circuito. Así, la evolución del readout domina el movimiento paramétrico medido en esta ejecución.

Esto no demuestra que las features cuánticas carezcan de información: no se guardó una representación de features con la que evaluarlas separadamente, y no se ajustó ningún clasificador adicional. Sí demuestra que la cabeza y sus refits alteran fuertemente las probabilidades, y que el modelo seleccionado al final no incorpora los dos estados post-refit.

## 6. Defectos identificados y limitaciones

**No se encontró un defecto confirmado** de codificación de etiquetas, dimensiones de salida, cálculo de softmax, orden de bits o extracción de marginales/ZZ. La validación fue leída repetidamente para diagnóstico, pero el resultado registra que no se usó para gradientes, fitting ni actualizaciones; la selección de mejor estado se basa en la pérdida de referencia train.

Hallazgo de comportamiento a resolver antes de escalar: el estado final seleccionado permanece en iteración 1, mientras los refits de iteraciones 10 y 20 alteran mucho la cabeza. La comparación del `best_theta` ocurre antes del callback de refit; el estado post-refit no se compara directamente como candidato a mejor estado en esa iteración. Esto concuerda con la implementación de selección por pérdida de referencia, pero deja abierta la cuestión de si ese criterio y el calendario de refit representan la intención experimental.

Limitaciones de evidencia: no hay probabilidades por ejemplo, features cuánticas brutas, norma L2 del gradiente, peak memory ni lista persistida de warnings. Hay un solo seed y no se ha evaluado test; la accuracy de validation apenas supera el azar y no constituye una conclusión estadística generalizable. No se evaluó si las features medidas contienen separabilidad independiente de la cabeza.

## 7. Recursos e integridad

- Backend: Qiskit Aer local CPU, `matrix_product_state`; Python 3.12.0, NumPy 2.2.6, Qiskit 2.3.1, Aer 0.17.2.
- Runtime medido: 1,905.905 s en el trainer incluyendo diagnósticos; 2,088.202 s para ajuste y evaluación final.
- Contabilidad medida de resultados PUB completados: 34,366 circuitos y 17,595,392 shots solicitados; 803 llamadas a objetivo (801 del optimizador); 32,286 ejemplos train y 2,080 validation evaluados, incluyendo repeticiones; 2 refits. No son estimaciones.
- Memoria máxima y norma del gradiente: no disponibles. La varianza y las normas de paso sí están en la traza.
- Hash de `result.json`: `127111b4dc6218cfbe3ee3904f6a4eb50770a317e959a4904f6dd310e935c8e8`; hash de `checkpoint.json`: `08b26743930a6101c03582ac3328896fc7dd9e65d09bad3cbf40a364b7858896`.
- Hash del protocolo congelado: `87581f0b3d0a435ecfd44be59145a13bf824977c732f03d401f30520eb335d8e`; hash del manifiesto del dataset: `53a153278ce83722a61a02cd8bff6749c599209fc11202b5e64ed5ee23c05bfd`. El informe de recuperación registra que los hashes allowlisted de train/validation y PCA coinciden con el protocolo.
- Hashes de parámetros registrados: inicialización cuántica `e812a67f32d48fb25a6f5a37baf184cfef3354854e37438e1b0347ddb6b58485`; parámetros cuánticos finales `ebd5d407584b9cc8398cfab4b05def4312e1323254d16933766e9cff8de159da`; readout final `902b0265b837e3d5abd6425e64adac9ad0bc127fc78de5fcd81463596eb0ea51`.
- `test_access=false` está registrado en el resultado y checkpoint. Este diagnóstico no abrió ni evaluó el split de test, no cargó matrices de train/validation y no ejecutó circuitos.

## 8. Recomendaciones priorizadas

1. **Antes de escalar a q20, investigar convergencia y selección del estado.** El mejor loss de referencia fue el de la iteración 1; la pérdida empeoró tras el refit 10; 11 de 20 pasos se rechazaron. Aclarar cómo los estados post-refit deben entrar en la comparación de mejor checkpoint y por qué las refits modifican tanto la cabeza.
2. **No interpretar el aumento de accuracy diagnóstica como mejora suficiente.** En iteración 10 el CE de validation se más que duplicó pese al alza de accuracy; en iteración 20 macro-F1 bajó tras el refit. Mantener separados los diagnósticos de loss/calibración y clasificación.
3. **No afirmar que QNIM distingue teorías físicas con este resultado.** El modelo completo apenas supera el azar en una validation, carece de evaluación test y de réplicas; las features cuánticas brutas tampoco están disponibles para un análisis de separabilidad.

## Recomendación final

**C. Investigar convergencia del optimizador y presupuesto de entrenamiento antes de escalar a q20.**

La elección se apoya en el mínimo de pérdida de referencia en la primera iteración, la ausencia de mejora posterior, 11 actualizaciones rechazadas y los grandes cambios del readout en refits que empeoran o mezclan los indicadores. No hay evidencia suficiente para atribuir el resultado a un defecto de mapeo/circuito y recomendar A; tampoco hay base para detener definitivamente toda la configuración (D) sin resolver primero la selección/convergencia observada. Esta recomendación no autoriza ni inicia otro experimento.
