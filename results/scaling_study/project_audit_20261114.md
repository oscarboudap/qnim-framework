# Auditoría del problema de clasificación QNIM

**Fecha:** 14-11-2026  
**Pregunta auditada:** clasificar, a partir de una señal de onda gravitacional,
cuál de 13 etiquetas/modelos la generó; demostrar rendimiento por encima de
azar (`1/13 = 7,69 %`) y evaluar si un VQC aporta mejora.

## Conclusión ejecutiva

No hay un cambio de feature map, VQC o hiperparámetros que pueda garantizar
una accuracy superior al azar en señales físicas independientes. La prueba
directa más útil de esta auditoría fue un matched-filter por clase con datos y
plantillas blanqueados correctamente: quedó en **7,69 %** en validación,
bootstrap por parejas 7,69–7,69 %, `p = 1,0`. El test no se leyó.

Sí se midió **100 %** en un control sintético rápido (13 clases, 120 train y
60 validation por clase, ExtraTrees 300), pero el generador inserta adrede una
marca espectral por etiqueta. Ese control comprueba que un clasificador puede
aprender clases codificadas en las features; no es evidencia sobre ondas
gravitacionales ni mejora válida del TFM.

El principal bloqueo no parece ser el número de qubits: el proyecto mezcla
varios problemas/datasets no equivalentes, etiquetas teóricas mal definidas,
normalizaciones que eliminan señales de clase y al menos una ruta con
información verdadera del simulador disponible en inferencia. Afinar el VQC
antes de corregir esto puede aumentar la puntuación de un proxy sin mejorar la
clasificación física.

## Problema previsto y cómo lo implementa el código

El TFM formula una clasificación multiclase de 13 modelos/teorías sobre
señales de ondas gravitacionales. El objetivo del notebook de escalabilidad
añade comparar dimensiones PCA y qubits, evitar fallback falso de IBM, y
validar estadísticamente contra azar. El TFM reporta 6 % en IBM Fez frente a
7,7 % de azar, y 83,2 % en simulador; son resultados de hardware y simulación,
respectivamente, no una misma medida.

Hay tres rutas que no deben presentarse como una única evaluación:

1. **`SSTGAdapter` (ruta por defecto):** onda sintética y un marcador espectral
   deliberado por clase. El propio código califica este marcador como fuerte y
   artificial. Ejecución reproducida en esta auditoría: 100 % en validación
   con un clasificador clásico. Sirve de smoke test del pipeline.
2. **`PhysicsSSTGAdapter` / `LIGOPCAAdapter`:** señales PyCBC y transformaciones
   de teorías, sin el mismo ruido O3b añadido a los datos del clasificador.
   Sus distribuciones y preprocesamientos no son equivalentes a la ruta de
   inyección sobre ruido.
3. **`LIGONoiseInjectionAdapter`:** inyecta teorías en segmentos O3b disjuntos
   entre train/validation/test. Cada pareja de 13 clases comparte fuente,
   segmento de ruido y SNR objetivo. En este camino se obtuvieron resultados
   de evento único en azar con PCA y plantilla global. La diferencia respecto
   al control GR exacto emparejado es alta, pero ese control es oráculo: no
   está disponible para un evento real.

La ruta de ruido es la más próxima al objetivo de evento único, aunque sus
rangos O3b aún están marcados como candidatos pendientes de revisión de
calidad/vetos. No es todavía una certificación con datos GWOSC finales.

## Hallazgos priorizados

### P0 — La ruta principal por defecto resuelve una tarea sintética distinta

[`generate_results.py`](../../scripts/generate_results.py#L89) elige
`SSTGAdapter` salvo que se solicite explícitamente `--physics-generator` o
`--ligo-pca`. El adaptador codifica la clase con una marca espectral artificial
([`sstg_adapter.py`](../../src/infrastructure/sstg_adapter.py#L197)); con esa
marca se midió 100 % de validación.
Informar esa cifra como accuracy de identificación de teorías físicas sería
incorrecto. El CLI muestra un aviso en el texto del generador, pero no bloquea
ni etiqueta la métrica final como control artificial.

**Acción:** mantener el marcador solo como prueba de humo, etiquetar su salida
como sintética/no física y exigir selección explícita de un generador para
resultados científicos. Esto evita una mejora aparente, no produce por sí
mismo una mejora física.

### P1 — La ruta `PhysicsSSTGAdapter` usa la masa chirp verdadera y mezcla
validation en el escalado

El adaptador pasa `all_mc` (masa chirp verdadera generada por el simulador) a
`DephasingFeatureExtractor.transform`
([`physics_sstg_adapter.py`](../../src/infrastructure/physics_sstg_adapter.py#L246));
la masa elige qué plantilla GR usar.
En un evento real esa masa debe estimarse a partir del strain, no leerse del
generador. Además, `col_mean` y `col_std` se calculan sobre `features` antes
de separar train y validation
([`physics_sstg_adapter.py`](../../src/infrastructure/physics_sstg_adapter.py#L248)),
por lo que validation participa en el preprocesamiento.

**Acción:** ajustar el escalador solo con train y reemplazar la masa verdadera
por una estimación inferible del strain (o por una plantilla global/familia
de plantillas marginalizada). La accuracy de esta ruta debe medirse de nuevo;
las puntuaciones antiguas no son una medida sin fuga.

### P1 — La clase “standard-siren” no es una teoría alternativa identificable
con las features normalizadas

En los inyectores, `standard-siren` solo multiplica la amplitud por
`H0_ref/H0_true`
([`stochastic_signal_generator.py`](../../src/infrastructure/stochastic_signal_generator.py#L75)).
Las rutas `LIGOPCAAdapter` y `PhysicsSSTGAdapter` normalizan cada waveform por
norma ([`dephasing_features.py`](../../src/infrastructure/dephasing_features.py#L69)),
borrando exactamente un factor escalar global. La
normalización `per_waveform` hace lo mismo por construcción. Además, una
medida de sirena estándar de `H0` requiere información de distancia/redshift
que no está codificada como una etiqueta de morfología de strain.

**Acción:** separar la estimación cosmológica de `H0` de la clasificación de
teorías, o definir explícitamente qué observación externa hace identificable
esa etiqueta. No se debe cambiar la normalización física a ciegas: la escala
de amplitud también es degenerada con distancia e inclinación.

### P1 — La taxonomía no es coherente entre generadores

El índice 9 se llama `liv-alpha2` en `SSTGAdapter`, `LIGOPCAAdapter` y en la
tabla de nombres de `generate_results.py`, pero `PhysicsSSTGAdapter` y
`LIGONoiseInjectionAdapter` inyectan y etiquetan `liv-alpha1.5`. El propio
generador físico documenta que alpha=2 se sustituyó por alpha=1.5. Por tanto,
un mismo número de clase no representa la misma hipótesis en todos los
experimentos.

**Acción:** definir una taxonomía y parámetros canónicos únicos; versionar
datasets existentes porque no se pueden renombrar retrospectivamente sus
etiquetas sin cambiar su significado.

### P1 — Varias inyecciones se declaran como aproximaciones o stress tests

El código documenta LIV alpha=1.5/4 como surrogate sin integral cosmológica,
GUP como ley de fase de stress-test no calibrada, y Chern–Simons con una
amplificación `K` sobre la cota. Las 13 clases, por tanto, no son 13
distribuciones observacionales calibradas y mutuamente identificables.
También se observó en la generación ADD un SNR mediano mucho menor que el
objetivo, por la atenuación dependiente de distancia.

**Acción:** validar dimensionalidad, límites observacionales y potencia de
cada inyector con tests de waveform y posterior predictivo; especificar
priors/parametrizaciones antes de producir datos de clasificación. No elevar
parámetros arbitrariamente para hacer más fácil el clasificador.

### P2 — El primer resultado del prototipo matched-filter era inválido

En el primer prototipo se comparaba el strain crudo de validation con
plantillas blanqueadas. Se corrigió para ponderar consulta y plantillas con
la PSD de cada segmento, se añadió una prueba sintética de selección de la
plantilla exacta y se regeneró el conjunto. La validación corregida siguió en
7,69 %, `p = 1,0`; el test continúa sin leerse. El resultado previo del
prototipo debe considerarse sustituido.

**Acción:** tratar el resultado corregido como evidencia negativa para este
banco pequeño; la siguiente variante requeriría marginalización formal sobre
masa, fase, tiempo, inclinación y otros parámetros, y una nueva validación
reservada, no más ajuste sobre el mismo validation.

## Resultados reproducidos en esta auditoría

| Ruta / método | Evaluación | Resultado | Interpretación |
|---|---:|---:|---|
| SSTG con marcador, ExtraTrees 300 | 120 train + 60 validation/clase | 100,00 % | Control artificial; no física |
| Features de evento único `global_template` | CV agrupada y validation | 7,69 % | Azar; p=1 en validation |
| Matched-filter, 16 plantillas/clase, PSD correcto | 49 parejas/validation | 7,69 %; IC 7,69–7,69 %; p=1 | No supera azar |
| Diferencia contra GR de la misma pareja | validation | Hasta 89,64 % | Diagnóstico oracle, no desplegable |
| Holdout independiente anterior | 100 ejemplos/clase | 7,69 % | Baseline ExtraTrees en azar |

El JSON del matched filter marca `test_arrays_read: false`. Los resultados de
CV/validation exploratorios se han reutilizado durante esta auditoría; no
deben usarse como si fueran un holdout final. Para una afirmación final hace
falta un dataset de test no consultado y split por segmento/fuente.

## Plan de acción para una mejora física demostrable

1. Congelar la pregunta: clasificar hipótesis físicas identificables, no
   “13 nombres”; fijar priors y separar la tarea de sirena estándar.
2. Corregir la fuga de `Mc` verdadera y el escalado con validation.
3. Unificar los nombres y los inyectores; validar cada waveform limpio y su
   SNR medido con PSD antes de añadir ruido.
4. Generar nuevos train/validation/test con splits por segmento y fuente,
   parámetros físicos muestreados de forma comparable y calidad O3b revisada.
5. Medir primero matched filters/métodos clásicos sobre train/validation.
   Seleccionar el pipeline sin tocar test.
6. Solo si un baseline clásico supera azar de forma reproducible, comparar
   VQC y feature maps en las mismas features, splits y semillas; ejecutar
   hardware al final.

No hay una modificación del VQC que garantice > azar sin señal identificable.
La forma de garantizar un número mayor en este código sería introducir una
etiqueta codificada artificialmente (como el marcador SSTG), filtrar clases
con test, o usar parámetros verdaderos no disponibles al inferir. Esas
opciones invalidan la afirmación física. La ganancia legítima requiere
corregir el problema de datos/definición y demostrarla con una prueba nueva.

## Archivos de evidencia y validación

- [`accuracy_gap_projection.md`](./accuracy_gap_projection.md): análisis
  estadístico y resultados previos de la ruta con ruido.
- [`accuracy_gap_projection_matched_filter_whitened.json`](./accuracy_gap_projection_matched_filter_whitened.json):
  matched filter corregido; train/validation solamente.
- [`accuracy_gap_projection_global_template.json`](./accuracy_gap_projection_global_template.json):
  resultado de features global-template.
- [`scripts/generate_results.py`](../../scripts/generate_results.py): entrada
  principal y selector del generador por defecto.
- [`src/infrastructure/sstg_adapter.py`](../../src/infrastructure/sstg_adapter.py):
  generador marcador sintético.
- [`src/infrastructure/physics_sstg_adapter.py`](../../src/infrastructure/physics_sstg_adapter.py):
  generación física, selección con `Mc` real y escalado actual.
- [`src/infrastructure/stochastic_signal_generator.py`](../../src/infrastructure/stochastic_signal_generator.py):
  aproximaciones de inyectores y definición LIV.

Pruebas focalizadas: **18 passed** en Windows para inyección, análisis y
métricas, y **7 passed** en WSL para el adaptador PyCBC. Compilación Python y
diagnósticos Pylance de los archivos tocados: sin errores.
