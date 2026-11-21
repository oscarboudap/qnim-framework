# Auditoría final de código — QNIM_IBM_QPU_PILOT_V1

## Alcance y restricciones

Auditoría estática, inspección de rutas y pruebas unitarias con mocks/sentinels.
No se consultó la cuenta IBM; no se enviaron jobs; no se construyeron circuitos;
no se ejecutaron Sampler, Aer o StatevectorSampler; no hubo entrenamiento ni
acceso a `X_test`/`y_test`. La comprobación de provenance verificó hashes de los
artefactos train/validation/PCA y del código generador, sin cargar arrays.

## 1. Integridad del repositorio y procedencia

Los siguientes cambios de auditoría se aplicaron sobre trabajo previo sin
revertirlo:

| Archivo | Procedencia de los cambios en esta auditoría |
|---|---|
| `src/infrastructure/qnim_ibm_qpu_pilot.py` | Módulo previo del piloto; se completaron el ledger compartido, límites acumulados y registro de evaluaciones incompletas. |
| `scripts/run_qnim_ibm_qpu_pilot.py` | Runner previo; ahora comprueba el presupuesto restante en el ledger global antes del preentrenamiento local. |
| `src/infrastructure/qiskit_vqc_trainer.py` | Archivo con cambios preexistentes mezclados; esta auditoría añadió validación de contexto/presupuesto y persistencia de parámetros, pérdida y métricas. |
| `scripts/generate_results.py` | Corrección de esta auditoría: la ruta de desarrollo desde disco ya no carga los arrays de test. |
| `test/unit/test_qnim_ibm_qpu_pilot.py` | Pruebas previas del piloto ampliadas para gate de envío, evaluación pendiente y límites indirectos. |
| `test/unit/test_experiment_config.py` | Archivo con cambios preexistentes mezclados; solo se ajustó el fixture de una ruta hardware mocked para que no se identifique como el protocolo real. |
| `results/scaling_study/qnim_ibm_qpu_pilot_v1/qnim_ibm_pilot_final_code_audit.md` | Informe de esta auditoría. |

El worktree ya contenía modificaciones rastreadas de scaling/QNIM y artefactos
históricos no rastreados. Se conservaron. Esta auditoría no modificó `protocol.json`,
los resultados/protocolos V1, V2 o V3, ni los archivos de datos históricos.

`git diff --check` sin configuración de fin de línea marca los CR finales como
espacios; `core.autocrlf=true`. La comprobación apropiada
`git -c core.whitespace=cr-at-eol diff --check` terminó con código 0. No se
detectaron otros errores de whitespace con esa configuración.

## 2. Aislamiento del test

- `PilotSplits` expone únicamente `X_train`, `y_train`, `X_val` y `y_val`; el
  cargador usa una lista explícita de archivos de train/validation.
- La orquestación ordinaria de `QiskitVQCTrainer` valida con `X_val/y_val`.
  La única lectura de `dataset.X_test/y_test` en el entrenador está dentro de
  `evaluate_test_on_ibm`, después de comprobar
  `authorize_test_access is True`.
- La ruta `scripts/generate_results.py --from-disk` cargaba test aunque no lo
  evaluaba. Se documentó el defecto antes de corregirlo; ahora mantiene esos
  campos en `None` y solo carga train/validation.
- Los tests sentinel hacen que cualquier lectura de propiedades `X_test` o
  `y_test` falle. Cubren entrenamiento/orquestación, validación y el gate de
  evaluación final.
- No se encontró una llamada implícita a evaluación final en el flujo normal
  `mode="ibm"`.

**Límite fuera del piloto:** `scripts/evaluate_single_event_holdout.py` y
`scripts/diagnose_qnim_classical_baselines.py` tienen flujos explícitos que
leen/evalúan test. No se llaman desde el runner aislado del piloto y no se
ejecutaron.

## 3. Auditoría de presupuesto

El runner usa ahora un ledger común en
`results/scaling_study/qnim_ibm_qpu_pilot_v1/jobs`, no un subdirectorio nuevo por
run, y verifica el saldo estimado antes del preentrenamiento. El lock exclusivo
serializa la comprobación y creación de intenciones. Cada intención vuelve a
comprobar los topes acumulados de jobs y shots, incluso para llamadas directas
al sampler.

Para entradas identificadas como `QNIM_IBM_QPU_PILOT_V1`, el contexto del
sampler y el ledger comprueban además:

- el gate `submission_enabled` del protocolo;
- SPSA en pasos protocolizados, batch/shots protocolizados y roles ordenados
  `plus`, `minus` y referencia;
- el máximo acumulado de cinco pasos SPSA y 65 ejemplos de validación;
- PUBs/circuitos compatibles con los ejemplos y el factor ZNE declarado.

Un job ambiguo o una evaluación sin pérdida/métricas persistidas queda en estado
no resuelto (`submission_unknown`, `completion_unknown`,
`completed_unhashed` o `evaluation_incomplete`) y bloquea reintentos. Jobs,
PUBs y shots ya intentados se siguen contabilizando tras fallos o cancelaciones.
Los tests mocked cubren límites acumulados, bloqueos de reintento y rechazo de
excesos por rutas indirectas. No se hizo una prueba de envío real.

El estimador fuente del protocolo es: 401 llamadas objetivo locales, 403 runs
locales, 3.264 circuitos/PUBs y 916.992 shots locales; hardware: 11 jobs SPSA
(incluida una referencia), 1 job de validación, 121 circuitos/PUBs y 46.592
shots. QPU-seconds y resource units permanecen desconocidos y no se estiman.

## 4. Persistencia y reconstrucción

- La intención atómica se escribe antes de `SamplerV2.run`; el ID se guarda
  inmediatamente tras obtenerlo. Si el envío o la respuesta son ambiguos, no
  hay reenvío automático.
- El resultado crudo serializado se persiste en un sidecar y su SHA-256 queda
  en el registro, junto con timestamps, job ID, PUBs, shots y métricas IBM si
  están disponibles.
- Cada evaluación SPSA registra rol, paso, vector efectivo y hash de
  parámetros, pérdida, índices de muestra, seed y constantes del optimizador.
  Los vectores plus/minus permiten reconstruir la perturbación y el gradiente;
  la inicialización local queda en checkpoint.
- Validación persiste métricas, etiquetas y predicciones. Si la conversión a
  métrica falla después del job, el registro queda bloqueado como
  `evaluation_incomplete`.
- Checkpoints usan reemplazo atómico y verifican hash de metadata, archivo de
  parámetros y vector.

## 5. Identidad algorítmica y alcance científico

La identidad declarada es **preentrenamiento local QNSPSA-EML-Feynman +
fine-tuning IBM mediante `HardwareSPSAOptimizer`**. No es QNSPSA-EML-Feynman
remoto. La evaluación hardware del piloto usa exclusivamente validation.

La provenance verificada identifica inyecciones simuladas PyCBC
`IMRPhenomPv2`, `strict_physics=true`, `fallback_allowed=false`, PCA/escalado
fit en train y ausencia de test. Los hashes de artefactos y código fuente
coinciden con el protocolo. No obstante, no existe ledger de parámetros por
evento (`event_level_parameter_ledger_present=false`) y el dataset no incluye
ruido de detector ni eventos observacionales. Por ello no se certifica como
evidencia de discriminación astrofísica observacional ni de que las inyecciones
representen diferencias físicas reales. No se afirma ninguna mejora de
accuracy aún no medida.

## 6. Rutas IBM no pertenecientes al piloto

`src/infrastructure/ibm_quantum_adapter.py` contiene una ruta de envío directa,
separada del ledger/gate del piloto, controlada por `USE_REAL_HARDWARE`.
`test/integration/test_ibm_submission.py` puede llegar a esa ruta y enviar un
job si se habilita el entorno. Ninguno se ejecutó. No deben usarse para este
piloto; la protección `submission_enabled=false` es específica del protocolo
aislado y no constituye una prohibición global de todos los adaptadores del
repositorio.

## 7. Preflight IBM y estado del protocolo

No se autenticó ni consultó IBM. El único chequeo de paquetes local indicó:

| Elemento | Resultado |
|---|---|
| Qiskit instalado | 2.3.1 |
| IBM Runtime instalado | No |
| Versión IBM Runtime declarada | 0.46.1 |
| Compatibilidad Runtime | No verificada |
| Cuenta, región, backends, layout, cola, plan/cuotas | No consultados; desconocidos |
| `submission_enabled` | `false` |

La identidad, hashes y restricciones permanecen aisladas en
`qnim_ibm_qpu_pilot_v1/protocol.json`. No se instaló ni actualizó ningún paquete
ni se cambió el protocolo para habilitar envíos.

## 8. Regresión y validación segura

Pasaron **41 tests** en módulos/rutas seleccionados:

- Todos los tests de `test/unit/test_qnim_ibm_qpu_pilot.py`.
- Todos los tests de `test/unit/test_hardware_spsa_optimizer.py`.
- Todos los tests de `test/unit/test_ibm_token_configuration.py`.
- 12 tests seleccionados de `test/unit/test_experiment_config.py`, incluidos
  los flujos con trainer/runtime falsos.

También pasó `py_compile` sobre los módulos y scripts cambiados. No se ejecutó
el patrón general de pytest ni tests que construyen circuitos, ejecutan
simuladores, lanzan entrenamiento real o contactan IBM. Los tests de integración
IBM permanecen sin ejecutar.

## 9. Defectos corregidos y riesgos restantes

Corregidos: ledger reiniciable por run, contabilidad indirecta incompleta,
trazabilidad insuficiente de pérdidas/parámetros/resultados, posible continuación
tras evaluación incompleta, bypass del gate desde métodos trainer/sampler
marcados como piloto y carga innecesaria de test desde la CLI genérica.

Riesgos pendientes:

1. Falta validar en un entorno aislado la compatibilidad con IBM Runtime 0.46.1,
   autenticación, backend, layout conectado, calibración, plan y cuotas.
2. No hay evidencia de independencia/randomness del hardware; seeds distintos
   no prueban independencia.
3. Falta provenance de parámetros por evento para auditar injection-by-injection.
   El dataset solo es apto para el alcance declarado de inyecciones simuladas.
4. Los adaptadores IBM legacy descritos arriba pueden enviar jobs fuera del
   presupuesto del piloto.
5. El ledger es conservador ante fallos: un lock obsoleto o una evaluación
   ambigua requiere reconciliación manual; no debe borrarse para reintentar.

## 10. Condiciones antes del primer job

1. Autorizar explícitamente la consulta de metadatos IBM y verificar credenciales
   sin imprimir secretos.
2. Instalar/probar Runtime 0.46.1 en un entorno separado, sin tocar `.venv312`.
3. Confirmar backend operativo, conectividad/layout de ocho qubits, calibración,
   cola y recursos de cuenta disponibles.
4. Resolver y documentar el modelo de randomness, el alcance científico y el
   presupuesto de 16 jobs/60.000 shots; no aumentar los topes implícitamente.
5. Decidir si se exige ledger de parámetros por evento antes de aceptar el
   dataset; mantener explícito que no son datos observacionales ni evidencia de
   física real.
6. Mantener el test sellado durante piloto y habilitar cualquier evaluación
   final solo mediante la operación separada y su autorización explícita.
7. Obtener autorización del responsable antes de cambiar
   `submission_enabled=false`; revisar que el ledger compartido no tenga
   intenciones irresueltas.
8. No usar la ruta legacy `IBMQuantumAdapter` para ejecutar este protocolo.

## 11. Decisión final

- **Implementación y diseño:** suficientes para continuar con preflight local
  sin jobs; las correcciones focalizadas pasaron las pruebas seguras.
- **Consulta IBM en esta auditoría:** **NO-GO**. Estaba expresamente fuera de
  alcance y Runtime no está instalado.
- **Ejecución QPU:** **NO-GO**. `submission_enabled` sigue en `false`; cuenta,
  backend, compatibilidad Runtime, plan/recursos y randomness no están
  verificados ni autorizados.

La siguiente decisión del responsable debe ser si autoriza una consulta IBM
solo de metadatos, sin envío. La ejecución del piloto requiere resolver primero
las condiciones anteriores.
