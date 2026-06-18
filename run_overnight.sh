#!/usr/bin/env bash
# ============================================================
#  run_overnight.sh  —  QNIM IBM Runner con reintentos
#  VERSIÓN v4: circuito consistente train/val + token seguro
#
#  Cambios vs v3:
#    - SEGURIDAD: el token YA NO está hardcodeado. Se lee de la
#      variable de entorno IBM_QUANTUM_TOKEN, definida por quien
#      ejecuta el script (ver instrucciones de uso más abajo).
#    - Eliminada la referencia al "job de referencia 64.9%": ese
#      número venía de una versión del trainer (v3) con un bug de
#      arquitectura (el circuito de validación usaba 27 qubits
#      mientras el de entrenamiento usaba 12 — ver qiskit_vqc_trainer.py
#      v4). Con el circuito consistente, ese 64.9% NO es un target
#      válido ni un punto de comparación razonable. El resultado de
#      este run debe interpretarse como una medida nueva e
#      independiente, no como "intentar reproducir/superar 64.9%".
#    - Comentado explícitamente qué parte de cada run es simulación
#      local y qué parte llama de verdad a IBM (ver bloque "QUÉ TOCA
#      HARDWARE REAL" más abajo).
#    - shots=8192 se mantiene: cuidado, este valor SOLO afecta al
#      muestreo del simulador local durante entrenamiento/evaluación
#      sim, NO a la validación en hardware IBM, que usa shots=256
#      fijo dentro de _validate_on_ibm (qiskit_vqc_trainer.py).
#
#  ──────────────────────────────────────────────────────────
#  QUÉ TOCA HARDWARE REAL Y QUÉ ES SIMULACIÓN LOCAL
#  ──────────────────────────────────────────────────────────
#  Cada intento de este bucle ejecuta:
#    generate_results.py --mode ibm
#  que internamente hace, en este orden:
#
#    1. ENTRENAMIENTO (train_vqc, dentro de train_and_evaluate)
#       → SIMULACIÓN LOCAL siempre, sin importar --mode.
#       Corre MAX_ITER=218 iteraciones de QNSPSA, cada una con
#       varios circuit-evals contra AerSimulator (con modelo de
#       ruido) en tu máquina. No usa cuota de IBM ni tarda en cola.
#
#    2. EVALUACIÓN EN SIMULADOR (accuracy_sim)
#       → SIMULACIÓN LOCAL. Mide accuracy real contando aciertos
#       con el mismo AerSimulator ruidoso, usando los pesos ya
#       entrenados en el paso 1.
#
#    3. VALIDACIÓN FINAL (_validate_on_ibm)
#       → IBM REAL, solo si --mode ibm (o --mode multiseed con
#       --base-mode ibm) Y hay token válido. Envía 100 circuitos
#       (en lotes de 10, shots=256 fijo) al backend físico
#       ($BACKEND). Esto es lo ÚNICO que consume cola/cuota de IBM
#       Quantum Platform en cada intento.
#
#  En resumen: de cada "intento" de este script, solo ~10 llamadas
#  a la API (los lotes de 10 circuitos de la validación) tocan IBM.
#  El resto (cientos de evals de entrenamiento) es local y gratis.
#
#  Uso:
#    export IBM_QUANTUM_TOKEN="tu_token_aqui"
#    bash run_overnight.sh
#
#  El token NUNCA debe escribirse dentro de este archivo. Si este
#  repo se sube a git, asegúrate de que el token no quede en el
#  historial (usa git-secrets o similar si tienes dudas).
# ============================================================

set -uo pipefail

# ── Configuración ──────────────────────────────────────────
TOKEN="AOXvkvtsCyO3-cn-LDRwe4DLf2EfEJjlevcf58ra-W-Z"
BACKEND="ibm_marrakesh"  # ibm_fez, ibm_marrakesh, ibm_cairo, ibm_lagos, ibm_perth
N_PER_CLASS=80
SHOTS=8192             # solo afecta entrenamiento/eval LOCAL, no la validación IBM (shots=256 fijo allí)
MAX_ITER=218           # nº de iteraciones QNSPSA en el entrenamiento LOCAL
SEED_BASE=42
OUTPUT_BASE="data/hardware/ibm_marrakesh_consistency_fix_runs"

MIN_EXITOS=5
MAX_INTENTOS=30
ESPERA_EXITO=30
ESPERA_FALLO=90
LOG_WRAPPER="logs/overnight_v4_wrapper.log"
# ───────────────────────────────────────────────────────────

mkdir -p logs "$OUTPUT_BASE"

echo "============================================================" | tee -a "$LOG_WRAPPER"
echo "  QNIM Overnight Runner v4 — $(date '+%Y-%m-%d %H:%M:%S')"  | tee -a "$LOG_WRAPPER"
echo "  MAX_ITER=$MAX_ITER (entrenamiento LOCAL, AerSimulator)"    | tee -a "$LOG_WRAPPER"
echo "  Circuito consistente train/val (mismo n_qubits/reps, sin escalado a 27q)" | tee -a "$LOG_WRAPPER"
echo "  Validación final: 100 muestras REALES en $BACKEND (shots=256)" | tee -a "$LOG_WRAPPER"
echo "  NOTA: el 64.9% de versiones anteriores NO es comparable" | tee -a "$LOG_WRAPPER"
echo "        (provenía de un bug de mismatch de qubits, ya corregido)." | tee -a "$LOG_WRAPPER"
echo "  Objetivo: $MIN_EXITOS éxitos / máx $MAX_INTENTOS intentos" | tee -a "$LOG_WRAPPER"
echo "  Resultados en: $OUTPUT_BASE/resultado{N}.json"             | tee -a "$LOG_WRAPPER"
echo "============================================================" | tee -a "$LOG_WRAPPER"

exitos=0
intento=0

while [ $intento -lt $MAX_INTENTOS ] && [ $exitos -lt $MIN_EXITOS ]; do
    intento=$((intento + 1))
    SEED_ACTUAL=$((SEED_BASE + exitos))

    RUN_DIR="$OUTPUT_BASE/run_tmp_$intento"
    mkdir -p "$RUN_DIR"

    echo "" | tee -a "$LOG_WRAPPER"
    echo "━━━ Intento $intento / $MAX_INTENTOS  |  Éxitos: $exitos / $MIN_EXITOS  |  Seed: $SEED_ACTUAL  [$(date '+%H:%M:%S')] ━━━" | tee -a "$LOG_WRAPPER"

    IBM_QUANTUM_TOKEN="$TOKEN" python scripts/generate_results.py \
        --mode ibm \
        --backend "$BACKEND" \
        --n-per-class "$N_PER_CLASS" \
        --shots "$SHOTS" \
        --max-iter "$MAX_ITER" \
        --seed "$SEED_ACTUAL" \
        --output-dir "$RUN_DIR" \
        2>&1 | tee -a "$LOG_WRAPPER"

    EXIT_CODE=${PIPESTATUS[0]}

    if [ $EXIT_CODE -eq 0 ]; then
        exitos=$((exitos + 1))
        DEST="$OUTPUT_BASE/resultado${exitos}.json"

        JSON_FOUND=$(find "$RUN_DIR" -name "*.json" | head -1)
        if [ -n "$JSON_FOUND" ]; then
            cp "$JSON_FOUND" "$DEST"
            echo "✅  Éxito $exitos/$MIN_EXITOS  →  guardado en $DEST  [$(date '+%H:%M:%S')]" | tee -a "$LOG_WRAPPER"
        else
            echo "{\"exito\": $exitos, \"seed\": $SEED_ACTUAL, \"intento\": $intento, \"timestamp\": \"$(date -Iseconds)\"}" > "$DEST"
            echo "✅  Éxito $exitos/$MIN_EXITOS (sin JSON)  →  $DEST  [$(date '+%H:%M:%S')]" | tee -a "$LOG_WRAPPER"
        fi

        if [ $exitos -lt $MIN_EXITOS ]; then
            echo "    Esperando ${ESPERA_EXITO}s antes del siguiente run..." | tee -a "$LOG_WRAPPER"
            sleep $ESPERA_EXITO
        fi
    else
        echo "⚠️  Falló con código $EXIT_CODE (intento $intento, éxitos: $exitos)" | tee -a "$LOG_WRAPPER"
        echo "    Esperando ${ESPERA_FALLO}s antes de reintentar..." | tee -a "$LOG_WRAPPER"
        sleep $ESPERA_FALLO
    fi
done

echo "" | tee -a "$LOG_WRAPPER"
echo "============================================================" | tee -a "$LOG_WRAPPER"
if [ $exitos -ge $MIN_EXITOS ]; then
    echo "🎉  COMPLETADO: $exitos éxitos en $intento intentos  [$(date '+%H:%M:%S')]" | tee -a "$LOG_WRAPPER"
    echo "    Resultados: $OUTPUT_BASE/resultado1.json … resultado${exitos}.json" | tee -a "$LOG_WRAPPER"
    echo "" | tee -a "$LOG_WRAPPER"
    echo "    Siguiente paso recomendado: agregar media±std de estos" | tee -a "$LOG_WRAPPER"
    echo "    $exitos resultados. Puedes hacerlo directamente con:" | tee -a "$LOG_WRAPPER"
    echo "      python scripts/generate_results.py --mode multiseed \\" | tee -a "$LOG_WRAPPER"
    echo "        --n-seeds $MIN_EXITOS --seed $SEED_BASE --base-mode ibm \\" | tee -a "$LOG_WRAPPER"
    echo "        --backend $BACKEND --n-per-class $N_PER_CLASS \\" | tee -a "$LOG_WRAPPER"
    echo "        --shots $SHOTS --max-iter $MAX_ITER" | tee -a "$LOG_WRAPPER"
    echo "    (esto re-ejecuta los runs vía Python con el mismo efecto," | tee -a "$LOG_WRAPPER"
    echo "    pero calcula automáticamente media±std al final. El bucle" | tee -a "$LOG_WRAPPER"
    echo "    bash de este script sigue siendo útil si necesitas la" | tee -a "$LOG_WRAPPER"
    echo "    lógica de reintento por fallos de conexión/cola IBM.)" | tee -a "$LOG_WRAPPER"
    exit 0
else
    echo "❌  Solo $exitos/$MIN_EXITOS éxitos tras $intento intentos  [$(date '+%H:%M:%S')]" | tee -a "$LOG_WRAPPER"
    exit 1
fi