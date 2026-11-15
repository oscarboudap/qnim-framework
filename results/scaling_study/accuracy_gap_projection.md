# Diagnóstico de accuracy y proyección de recursos

**Fecha:** 14-11-2026
**Dataset analizado:** `dataset_merger_centered_300`  
**Clases:** 13; azar uniforme = `1/13 = 7,69 %`

## Conclusión

No hay evidencia de que aumentar qubits, iteraciones o shots, por sí solo,
lleve la accuracy por encima del azar. En los datos actuales, varios modelos
clásicos distintos obtienen exactamente el nivel de azar con las features de
un evento; la información de clase no aparece en la representación que recibe
el clasificador. Más cómputo no recupera información que el preprocesado haya
eliminado o que quede por debajo del ruido.

El TFM ya documenta el mismo límite de hardware: informa 6 % en IBM Fez con
13 clases y 100 muestras, frente a 7,7 % de azar, y advierte que esa ejecución
no valida el clasificador (resumen; sección 7.2.1, p. 67; apéndice N.6,
p. 173). El 83,2 % es resultado de simulador, no de hardware. El TFM también
lo describe como resultado condicionado por una cuota limitada, transferencia
de pesos sin reentrenamiento en hardware y ausencia de mitigación de lectura.

Por tanto, no es correcto prometer ni proyectar una mejora de accuracy en IBM
sin corregir primero el dataset/representación y repetir una prueba independiente.

## Corrección crítica del pipeline (14-11-2026)

La auditoría encontró un error de unidades en
[`_inject_and_whiten`](../../src/infrastructure/ligo_noise_injection_adapter.py):
se calculaba el SNR con `numpy.fft` sin el factor `delta_t**2` requerido al
combinar una FFT discreta con una PSD física en strain²/Hz. A 4096 Hz, las
inyecciones generadas con la fórmula anterior quedaban escaladas para un SNR
físico aproximadamente 4096 veces menor que el objetivo. La fórmula se corrigió
y hay una prueba analítica de regresión.

También se corrigió el generador de ruido del control sintético: normalizaba la
PSD por su mediana y perdía su escala absoluta. En la única muestra O3b local,
el factor de amplitud que correspondía a esa escala era `3,2e-22`, no 1. Los
resultados de los controles ppE anteriores a estas correcciones quedan
**invalidados**. Cualquier dataset generado por este adaptador antes del arreglo
debe regenerarse antes de interpretar su SNR; sus arrays de test siguen cerrados.

## Resultados reproducidos

### Pilotos VQC ya guardados en el repositorio

| Ejecución | Accuracy balanceada | IC bootstrap 95 % | p de permutación | Conclusión |
|---|---:|---:|---:|---|
| `pilot_12q_10iter` | 7,92 % | 6,69–9,23 % | 0,371 | No supera azar de forma significativa |
| `pilot_smoke` | 8,00 % | 6,84–9,28 % | 0,317 | No supera azar de forma significativa |

Ambos tienen 13 clases y usan un conjunto de validación balanceado. Sus
intervalos incluyen 7,69 % y sus pruebas de permutación no son significativas.
Los resultados son pilotos, no estimaciones de una curva de escalado. No se
obtuvo una nueva métrica VQC de la versión local actual: el intento de
entrenamiento con el dataset completo se detuvo tras más de seis minutos sin
devolver resultado. Por tanto, las métricas VQC de la tabla son las que ya
estaban guardadas, no una certificación de los cambios locales no confirmados.

### Features del dataset actual

`X_train` tiene 3.900 ejemplos, `X_val` 1.300, y cada clase está balanceada.
Los modelos Logistic Regression, SVC RBF, Extra Trees y MLP obtuvieron
7,69 % en validación con las features PCA; Logistic Regression, SVC RBF y
Extra Trees también dieron 7,69 % al usar las features físicas sin
transformación emparejada. El análisis Fisher guardado en los metadatos da
p-valores de 1,0 para los primeros componentes. Esto apunta al cuello de
botella en la señal/representación, no a falta de qubits.

Hay un indicio diagnóstico prometedor, pero **no es accuracy desplegable**.
Al restar cada vector de features físicas del control GR generado con la
misma fuente y el mismo segmento de ruido, Extra Trees logra 92,31 % en
validación cruzada agrupada por pareja y 93,15 % en validación. El IC
bootstrap por bloques de parejas es 92,08–94,23 %; una permutación dentro de
pareja da `p = 0,00005` con 20.000 permutaciones. Esta resta usa el control
contrafactual exacto disponible en la simulación (los metadatos lo declaran
como referencia oráculo). En un evento real no se dispone de una segunda
medición del mismo evento, con idéntico ruido, bajo GR. Por ello, el resultado
indica que las firmas inyectadas son distinguibles al cancelar el ruido
emparejado, pero **no demuestra** una clasificación de evento único.

El dataset `dataset_merger_centered_300` declara solo **1 ejemplo de test por
clase** (13 en total), así que no sirve para una evaluación final. Sí hay otro
split guardado, `dataset_physical128`, con 100 ejemplos de test por clase y
segmentos de ruido disjuntos entre train/validation/test. Evalué allí una sola
vez un baseline Extra Trees fijado: accuracy y balanced accuracy = **7,69 %**,
IC bootstrap por parejas **7,69–7,69 %**, permutación por parejas `p = 1,0`.
El clasificador colapsó casi todas sus predicciones en una clase (1.170 de
1.300) y solo predijo tres clases. Es evidencia independiente de que la
representación de evento único no está siendo aprendida por este baseline;
no es una evaluación VQC. Los detalles están en
[`final_test_physical128.json`](./final_test_physical128.json).

### ¿Reducir de 13 a 10 clases supera el azar?

No en la validación disponible. Para no escoger clases mirando el test,
seleccioné las 10 con mejor recall en validación cruzada agrupada de train y
entrené de nuevo en esas clases. En los dos tipos de features (PCA y físicas),
la validación independiente de 49 parejas/490 ejemplos dio **10,00 %** de
accuracy y balanced accuracy: exactamente el azar de 10 clases. El IC
bootstrap por pareja fue también 10,00–10,00 %. La selección excluyó
`liv-alpha4`, `loop-quantum-gravity` y `gup`; aun con esta selección favorable
según train, no se observó mejora.

También probé una fusión semántica de las variantes QNM y LIV, que da 11
clases: balanced accuracy de 9,09 % en ambas representaciones, exactamente el
azar de 11 clases. Una tarea binaria GR/no-GR dio 92,31 % de accuracy bruta,
pero solo 50,00 % de balanced accuracy (azar): el modelo predijo todos los
ejemplos como no-GR. Por eso, subir la accuracy bruta reduciendo o
desequilibrando las clases puede ser engañoso.

Estas pruebas usaron validación, no el test, y no justifican una mejora
confirmada. Para 10 clases y 100 ejemplos por clase, el modelo binomial
optimista requiere al menos 117 aciertos de 1.000 (11,7 %) para rechazar azar
con `alpha = 0,05`; si la accuracy real fuera 15 %, la potencia aproximada
sería 99,9 %. La proyección presupone eventos independientes y sobreestima la
potencia si hay correlación entre eventos.

### Auditoría de generación y SNR

En `dataset_physical128`, los metadatos confirman que train/validation tienen
etiquetas 0–12 en orden dentro de cada pareja, el mismo SNR objetivo y ruido
compartido dentro de cada pareja; los segmentos entre splits son disjuntos.
La evaluación de validación por cuartiles del SNR objetivo (9,24–15,71;
15,71–20,08; 20,08–25,35; 25,35–29,18) quedó en 7,69 % de balanced accuracy
en los cuatro cuartiles, tanto para PCA como para las features físicas
emparejadas. En el rango SNR observado no aparece mejora al subir SNR.

Sí aparece una asimetría fuerte en la generación: `extra-dimensions` tiene
mediana de SNR observado 0,83 en train y 0,92 en validation, frente a SNR
objetivo mediano cercano a 18–20 (razón mediana 0,043–0,045). El código de
esa inyección aplica una atenuación dependiente de `distance_mpc` sin un
parámetro explícito de escala de referencia. Esto puede hacer que la clase
sea casi indistinguible del ruido; no conviene cambiar la fórmula sin
especificar y justificar físicamente esa escala.

Los arrays guardados no contienen señales limpias de strain. Para el
experimento de banco de plantillas se regeneraron en WSL con PyCBC las
plantillas requeridas a partir de los parámetros de train; esto no añadió
plantillas de validation ni consultó test. No se midió una curva física nueva
de SNR. Añadí al generador
`X_global_template_{train,val,test}`: features con una plantilla GR media
ajustada solo con referencias GR de train, con leave-pair-out para train, sin
restar el control GR exacto de cada pareja. En el conjunto regenerado, los tres
modelos evaluados con `global_template` quedaron en **7,69 %** de balanced
accuracy tanto en CV agrupada como en validación; el IC bootstrap colapsa en
7,69 % y la permutación da `p = 1,0`. El análisis no leyó ningún array de test.
En contraste, el diagnóstico de diferencia contra el GR exacto emparejado dio
89,64 % en validación (IC bootstrap por parejas 87,44–91,68 %; `p = 0,0002`,
mínimo resoluble con 5.000 permutaciones). Ese resultado sigue siendo oráculo:
confirma que cancelar ruido emparejado revela señal, no que la feature sea
utilizable con un evento real. Resultados completos en
[`accuracy_gap_projection_global_template.json`](./accuracy_gap_projection_global_template.json).

También se ejecutó una ablación separada que normaliza cada teoría a su propio
SNR objetivo. No mejoró ninguna entrada de evento único: PCA, features físicas
y `global_template` quedaron en **7,69 %** para los tres modelos (CV agrupada y
validación; `p = 1,0`). La diferencia contra el control GR emparejado siguió
siendo informativa (Extra Trees: 91,84 % con features físicas y 80,38 % con
`global_template`), pero es el mismo diagnóstico oráculo y no una mejora
desplegable. Los p-valores de estas exploraciones son no ajustados. El cambio
de normalización altera la amplitud física y solo sirve para aislar
morfología; no debe interpretarse como el rendimiento de una población real.
Resultados en
[`accuracy_gap_projection_equal_snr.json`](./accuracy_gap_projection_equal_snr.json).

Como cribado adicional de modelos sobre esa misma validación, se probaron un
MLP (`128,64` neuronas, early stopping) y LDA con shrinkage, también con un
SVC RBF sobre 12 componentes LDA. En PCA y `global_template`, todos dieron
7,69 % tanto en CV agrupada de train como en validación. Por tanto, el bloqueo
no se resuelve simplemente sustituyendo los tres baselines iniciales por una
red densa o una proyección supervisada. Este cribado es exploratorio y no
incluye el test.

Como experimento físico adicional, se generaron 16 plantillas por teoría
usando solo train, seleccionadas a lo largo de los cuantiles de masa chirp.
La primera implementación comparaba strain crudo contra plantillas blanqueadas;
esa puntuación era inválida y no se conserva como resultado. Se corrigió para
blanquear consulta y plantillas con la PSD del segmento y se regeneró el
experimento. Con la puntuación corregida, la validación quedó otra vez en
**7,69 %** (IC bootstrap por pareja 7,69–7,69 %, permutación `p = 1,0`); las
predicciones colapsaron en dos clases. El test no se leyó. Este prototipo no
valida una búsqueda física completa con marginalización de parámetros y
tiempo. Resultado vigente en
[`accuracy_gap_projection_matched_filter_whitened.json`](./accuracy_gap_projection_matched_filter_whitened.json).

### Control positivo ppE con ruido coloreado

Tras corregir SNR y escala de ruido, ejecuté varios clasificadores con 200
parejas train/100 validation. Con fuente fija (masa total 30 M☉, distancia
300 Mpc, SNR 20–30) y `delta_phi_100hz = 0,01`, Logistic Regression sobre
strain blanqueado obtuvo **70,5 %** de balanced accuracy (IC bootstrap por
pareja 65,5–75,5 %; permutación `p = 0,0002`). Extra Trees obtuvo 61,0 %.
Con PCA de 128 componentes y el escalado existente, Logistic Regression bajó
a 54,0 % y Extra Trees quedó en 50,0 %. En este control, evitar PCA mejora la
accuracy y es una alternativa concreta que merece validación independiente.

| Control (balanced accuracy) | Strain: Logistic | Strain: Extra Trees | PCA: Logistic | PCA: Extra Trees |
|---|---:|---:|---:|---:|
| Fuente fija, `delta_phi=0,01` | **70,5 %** | 61,0 % | 54,0 % | 50,0 % |
| Rangos nominales y fuente variable | 51,0 % | 49,5 % | 50,5 % | 49,5 % |
| Stress, fuente fija, `delta_phi=0,1` | 100,0 % | 99,5 % | 96,0 % | 77,5 % |

El segundo renglón usa `delta_phi=0,001–0,01`, SNR 8–30, masas 10–80 M☉ y
distancias 100–1000 Mpc; no muestra mejora significativa en el clasificador
de evento único. Un diagnóstico de likelihood con **parámetros fuente y
`delta_phi` exactos de validation** dio 58,0 % (IC 54,5–61,5 %; `p=0,0002`),
pero es un oráculo no desplegable. Apunta a que una búsqueda GR/ppE que
marginalice los parámetros de fuente es el siguiente candidato físico; no
demuestra aún que esa solución generalice.

El stress test usa `delta_phi=0,1`, diez veces el extremo superior nominal:
prueba que, con señal grande y fuente fija, el strain bruto puede clasificarse
casi perfectamente; no es una mejora válida para el rango físico nominal. Los
resultados de selección de modelos sobre validation son exploratorios y los
p-valores no están corregidos por comparaciones múltiples.

Todos estos controles siguen usando ruido gaussiano sintético coloreado con
la PSD estimada de **un solo** segmento O3b: no reproducen glitches ni la
no-gaussianidad de strain real. No se leyeron arrays de test ni se ejecutó IBM.
Resultados reproducibles:
[`ppe_pipeline_calibrated_fixed_source_20261114.json`](./ppe_pipeline_calibrated_fixed_source_20261114.json),
[`ppe_pipeline_calibrated_nominal_20261114.json`](./ppe_pipeline_calibrated_nominal_20261114.json) y
[`ppe_pipeline_calibrated_stress_20261114.json`](./ppe_pipeline_calibrated_stress_20261114.json).

## Proyección estadística

### Tamaño de test

Para un test balanceado de 13 clases, una aproximación binomial optimista
(eventos independientes, azar `p0 = 1/13`, contraste unilateral `alpha = 0,05`)
da:

| Test por clase | N total | Accuracy mínima para rechazar azar | Potencia si la accuracy real es 10 % |
|---:|---:|---:|---:|
| 10 | 130 | 12,31 % | 22,7 % |
| 30 | 390 | 10,26 % | 45,8 % |
| 50 | 650 | 9,54 % | 67,1 % |
| 75 | 975 | 9,23 % | ≥80 % |
| 100 | 1.300 | 9,00 % | 89,5 % |
| 300 | 3.900 | 8,44 % | 99,96 % |

Esta potencia es para accuracy global y es optimista si hay correlación entre
eventos, segmentos compartidos o selección del modelo con el mismo conjunto.
El estudio final debe bloquear por evento/segmento, usar el test una sola vez
y reportar accuracy balanceada, IC por bootstrap de grupos y permutación
preespecificada. Recomiendo como mínimo 100 eventos independientes por clase;
300 por clase permite intervalos más estables y análisis por clase.

Con 1 muestra por clase no puede estimarse de forma útil la variación por
clase. El test independiente de 100 por clase del baseline sí tiene tamaño
razonable, pero da exactamente azar; más ejemplos no arreglan una
representación sin señal. Incluso el 6 % del TFM con N=100 es compatible con
azar: bajo el modelo binomial uniforme, la prueba exacta unilateral de estar
por debajo de azar da `p = 0,343`.

### Más shots

El error estándar máximo de una probabilidad medida en un qubit es
`0,5 / sqrt(shots)`: 6,25 puntos porcentuales con 64 shots; 2,21 pp con 512;
0,78 pp con 4.096; y 0,39 pp con 16.384. Aumentar shots reduce ruido de
medición; **no** cambia la información de clase ni garantiza subir accuracy.

### Más qubits y más tiempo

No existe aquí una relación empírica accuracy-qubits/tiempo: solo hay pilotos
pequeños y no una matriz de escalado con réplicas independientes y test final.
Así que no sería estadísticamente serio extrapolar la accuracy al pasar de 12
a 16/20/32 qubits. Si las features permanecen sin señal de clase, la expectativa
es seguir alrededor de 7,69 %, incluso con más cómputo. Una proyección positiva
solo sería defendible después de mostrar una curva reproducible en varios seeds
y eventos independientes, frente a un baseline clásico equivalente.

## Qué hace falta para validar una mejora

### Reformulación prioritaria: GR frente a una desviación ppE -1PN

Se mantiene la pregunta física: **¿distingue el strain una predicción GR de
una desviación de fase compatible con física más allá de GR?** Para hacerla
contrastable, se reduce la taxonomía de 13 etiquetas a dos hipótesis: GR y una
única deformación fenomenológica ppE a -1PN (`pn-deformation`, índice 4).
La tarea tiene azar balanceado del 50 %, pero no pretende clasificar todas las
teorías ni detectar señal frente a ruido.

El generador produce ambas hipótesis con los mismos parámetros de fuente
cuando se usa PyCBC, el mismo segmento de ruido y SNR de referencia GR. El
clasificador recibe cada evento por separado: no se le entregan diferencias
contra el GR contrafactual, masa chirp verdadera ni plantillas GR emparejadas.
Los segmentos GPS son disjuntos entre splits y PCA/escalado se ajustan solo
con train. Código:
[`generate_gr_vs_ppe_dataset`](../../src/infrastructure/ligo_noise_injection_adapter.py)
y [`prepare_gr_vs_ppe_dataset.py`](../../scripts/prepare_gr_vs_ppe_dataset.py).

**Límite de interpretación:** ppE es una parametrización de desviaciones,
no una teoría completa. El inyector usa `delta_phi_hat` entre 0,001 y 0,01 y
una ley de fase \(f^{-7/3}\); el propio código advierte que no es el
coeficiente ppE publicado sin conversión. Un resultado positivo sería
evidencia de que el modelo discrimina esta familia de inyecciones simuladas
bajo las condiciones ensayadas, no un descubrimiento de nueva física ni
evidencia contra GR en datos observacionales.

La reducción binaria se cribó también en los arrays físicos ya generados,
sin abrir los arrays de test: Extra Trees sobre las features PCA de GR frente
a ppE obtuvo balanced accuracy **50,00 %** en 98 eventos de validation (49
parejas), IC bootstrap por pareja **50,00–50,00 %**, permutación por pareja
`p = 1,0`. Logistic Regression, SVC RBF, LDA-shrinkage y Extra Trees sobre
`global_template` también quedaron en 50,00 %. Por tanto, reducir las clases
por sí solo **no** ha producido una mejora demostrada. Estos resultados no
validan la nueva generación ni justifican abrir el test.

La nueva generación está cubierta por una prueba unitaria de strain sintético
que verifica pares, balance y separación de splits, no rendimiento científico.
Para regenerar el conjunto (usar únicamente rangos off-source y un inventario
revisados):

```bash
python scripts/prepare_gr_vs_ppe_dataset.py \
  --gps-ranges data/o3b/windows/gps_ranges_global_template.json \
  --detector H1 \
  --n-train-pairs 499 \
  --n-val-pairs 49 \
  --n-test-pairs 100 \
  --snr-min 8 --snr-max 30 \
  --output-dir results/scaling_study/dataset_gr_vs_ppe
```

Para validar la reformulación:

1. Fijar antes de medir un baseline clásico y compararlo con VQC usando las
   mismas features, splits y semillas.
2. Elegir configuración solo con train/validation; reportar balanced
   accuracy, ROC-AUC, matriz de confusión e IC/permutación agrupados por
   segmento.
3. Informar rendimiento por SNR e intensidad ppE; no seleccionar el intervalo
   de deformación por el resultado del test.
4. Repetir con varios seeds y datos de strain/segmentos independientes.
5. Abrir test una sola vez después de congelar pipeline e hipótesis; solo
   entonces evaluar IBM con backend, shots y mitigación registrados.

No se envió un trabajo remoto a IBM Quantum en esta sesión porque
`qiskit_ibm_runtime` no está instalado en el entorno activo. El resultado de
hardware citado arriba procede del TFM, no de una nueva ejecución.
La regeneración sí se ejecutó en WSL Ubuntu 24.04, con PyCBC 2.9.0 y LALSuite
8.0.1. El primer intento detectó incompatibilidad entre PyCBC y Astropy 8; al
fijar Astropy 6.1.7, NumPy 2.2.6 y setuptools 80.9.0, se verificó la
generación real IMRPhenomD.

Se generó `dataset_global_template_20261112` (6.487 train, 637 validation,
1.300 test). El test usa 100 segmentos nuevos, sin solaparse con el holdout
previo y separados de este por al menos 12 segundos. Los rangos provienen del
inventario local O3b; el manifiesto los etiqueta como candidatos pendientes
de revisión de vetos/calidad, así que no debe presentarse como prueba física
definitiva. La selección/evaluación en validación no encontró una configuración
prometedora para `global_template`; por ello, el test nuevo permanece cerrado.
Se generó además `dataset_global_template_equal_snr_20261112`, con los mismos
splits y normalización separada `per_waveform`. Su validación tampoco encontró
una feature de evento único por encima del azar; el test permanece cerrado.
Esta ablación iguala el SNR de cada teoría para aislar morfología, no es una
normalización física por defecto ni debe mezclarse con `reference_gr`.
También se generó `dataset_matched_filter_20261112`; su banco usa 16 parejas
de train por clase y solo produce scores para validation. La accuracy resultó
azar, así que tampoco se abrió su test.

## Reproducibilidad

El diagnóstico está en [`scripts/analyze_accuracy_gap.py`](../../scripts/analyze_accuracy_gap.py)
y la prueba de test fija en
[`scripts/evaluate_single_event_holdout.py`](../../scripts/evaluate_single_event_holdout.py).
La instalación aislada y regeneración se reproducen desde WSL, en la raíz del
repositorio:

```bash
python3 -m venv ~/.venvs/qnim-framework
~/.venvs/qnim-framework/bin/python -m pip install \
  "pycbc==2.9.0" "astropy==6.1.7" "numpy==2.2.6" \
  "setuptools==80.9.0" scikit-learn h5py
~/.venvs/qnim-framework/bin/python scripts/prepare_noise_injection_dataset.py \
  --gps-ranges data/o3b/windows/gps_ranges_global_template.json \
  --detector H1 \
  --inventory data/o3b/events/O3b_4KHZ_R1_H1.json \
  --frame-cache data/o3b/events \
  --output-dir results/scaling_study/dataset_global_template_20261112 \
  --n-per-class 499 --n-val-per-class 49 --n-test-per-class 100 \
  --n-components 128 --n-pca-fit 128 --snr-min 8 --snr-max 30 \
  --seed 20261112
~/.venvs/qnim-framework/bin/python scripts/analyze_accuracy_gap.py \
  --dataset-dir results/scaling_study/dataset_global_template_20261112 \
  --output results/scaling_study/accuracy_gap_projection_global_template.json \
  --bootstrap-iterations 5000 --permutation-iterations 5000 \
  --seed 20261112

# Ablación morfológica a SNR igualado (mantiene separado el dataset de referencia)
~/.venvs/qnim-framework/bin/python scripts/prepare_noise_injection_dataset.py \
  --gps-ranges data/o3b/windows/gps_ranges_global_template.json \
  --detector H1 \
  --inventory data/o3b/events/O3b_4KHZ_R1_H1.json \
  --frame-cache data/o3b/events \
  --output-dir results/scaling_study/dataset_global_template_equal_snr_20261112 \
  --n-per-class 499 --n-val-per-class 49 --n-test-per-class 100 \
  --n-components 128 --n-pca-fit 128 --snr-min 8 --snr-max 30 \
  --seed 20261112 --snr-normalization per_waveform
~/.venvs/qnim-framework/bin/python scripts/analyze_accuracy_gap.py \
  --dataset-dir results/scaling_study/dataset_global_template_equal_snr_20261112 \
  --output results/scaling_study/accuracy_gap_projection_equal_snr.json \
  --bootstrap-iterations 5000 --permutation-iterations 5000 \
  --seed 20261112

# Banco de plantillas por teoría, ajustado solo con parejas de train
~/.venvs/qnim-framework/bin/python scripts/prepare_noise_injection_dataset.py \
  --gps-ranges data/o3b/windows/gps_ranges_global_template.json \
  --detector H1 \
  --inventory data/o3b/events/O3b_4KHZ_R1_H1.json \
  --frame-cache data/o3b/events \
  --output-dir results/scaling_study/dataset_matched_filter_20261112 \
  --n-per-class 499 --n-val-per-class 49 --n-test-per-class 100 \
  --n-components 128 --n-pca-fit 128 --snr-min 8 --snr-max 30 \
  --seed 20261112 --matched-filter-bank-size 16
~/.venvs/qnim-framework/bin/python scripts/analyze_accuracy_gap.py \
  --dataset-dir results/scaling_study/dataset_matched_filter_20261112 \
  --output results/scaling_study/accuracy_gap_projection_matched_filter.json \
  --bootstrap-iterations 5000 --permutation-iterations 5000 \
  --seed 20261112
```

La prueba final no se debe abrir para elegir features o modelos. Solo si la
validación del baseline `global_template` da un resultado prometedor, ejecutar
una única vez:

```bash
~/.venvs/qnim-framework/bin/python scripts/evaluate_single_event_holdout.py \
  --dataset-dir results/scaling_study/dataset_global_template_20261112 \
  --feature-source global_template \
  --output results/scaling_study/final_test_global_template.json
```

El diagnóstico anterior del dataset histórico se reproduce desde Windows con:

```powershell
.\.venv312\Scripts\python.exe scripts\analyze_accuracy_gap.py `
  --dataset-dir results\scaling_study\dataset_merger_centered_300 `
  --output results\scaling_study\accuracy_gap_projection.json
.\.venv312\Scripts\python.exe scripts\evaluate_single_event_holdout.py `
  --dataset-dir results\scaling_study\dataset_physical128 `
  --output results\scaling_study\final_test_physical128.json
```

## Reformulación poblacional en curso

La pregunta de trabajo pasa de clasificar eventos aislados a contrastar una
deformación ppE compartida por una población:

> ¿Existe evidencia de un coeficiente ppE -1PN común distinto de cero tras
> marginalizar los parámetros de fuente de cada evento, y cómo cambia la
> sensibilidad al añadir bandas de fase y eventos?

Para eventos independientes, la likelihood poblacional es el producto de las
likelihoods por evento (suma de log-likelihoods). La hipótesis nula es
`beta = 0`; la alternativa integra `beta` sobre un prior continuo predeclarado.
Las métricas principales serán el factor de Bayes y el intervalo posterior de
`beta`; accuracy queda como métrica secundaria de los controles de inyección.

El primer componente implementado es
[`infer_shared_beta`](../../src/application/population_ppe_inference.py), que
combina una matriz evento × beta de log-likelihoods ya marginalizadas. Sus
pruebas confirman la acumulación de evidencia y la contracción del posterior
en un caso sintético. **Todavía no** calcula likelihoods desde strain ni
marginaliza parámetros fuente; no es un resultado físico ni VQC. La siguiente
fase debe implementar esa likelihood a partir de residuos de fase multibanda,
validarla con inyecciones y ruido off-source disjunto, y solo después comparar
VQC con matched filtering y baselines clásicos. El test permanece cerrado.

El control ppE corregido se reproduce en WSL (ruido sintético con escala
absoluta de PSD de una muestra local; no se consulta test):

```bash
~/.venvs/qnim-framework/bin/python scripts/diagnose_ppe_pipeline.py \
  --train-pairs 200 --val-pairs 100 --n-components 128 \
  --snr-min 20 --snr-max 30 \
  --delta-phi-min 0.01 --delta-phi-max 0.01 \
  --mass-min 30 --mass-max 30 \
  --distance-min 300 --distance-max 300 \
  --bootstrap-iterations 5000 --permutation-iterations 5000 \
  --seed 20261114 \
  --output results/scaling_study/ppe_pipeline_calibrated_fixed_source_20261114.json

# Rango nominal, incluyendo variación de parámetros fuente
~/.venvs/qnim-framework/bin/python scripts/diagnose_ppe_pipeline.py \
  --train-pairs 200 --val-pairs 100 --n-components 128 \
  --snr-min 8 --snr-max 30 \
  --delta-phi-min 0.001 --delta-phi-max 0.01 \
  --mass-min 10 --mass-max 80 \
  --distance-min 100 --distance-max 1000 \
  --bootstrap-iterations 5000 --permutation-iterations 5000 \
  --seed 20261114 \
  --output results/scaling_study/ppe_pipeline_calibrated_nominal_20261114.json

# Stress test exploratorio: delta_phi=0.1; no es el rango nominal.
~/.venvs/qnim-framework/bin/python scripts/diagnose_ppe_pipeline.py \
  --train-pairs 200 --val-pairs 100 --n-components 128 \
  --snr-min 20 --snr-max 30 \
  --delta-phi-min 0.1 --delta-phi-max 0.1 \
  --mass-min 30 --mass-max 30 \
  --distance-min 300 --distance-max 300 \
  --bootstrap-iterations 5000 --permutation-iterations 5000 \
  --seed 20261114 \
  --output results/scaling_study/ppe_pipeline_calibrated_stress_20261114.json
```

Las métricas de validación son exploratorias: el dataset se inspeccionó durante
el diagnóstico y la transformación emparejada es un control oráculo. No deben
presentarse como una mejora confirmada ni como una predicción de hardware.
