#!/usr/bin/env bash
# ============================================================
#  run_overnight.sh  —  QNIM IBM Runner con reintentos
#  VERSIÓN v3: noise-aware training + 218 épocas
#
#  Cambios vs v2:
#    - MAX_ITER=218 (igual que el job de referencia d8neqmb2d42s73cdpbeg)
#    - Trainer con noise-aware (AerSimulator + noise model ibm_marrakech)
#    - PATIENCE=20 para no parar antes de las 218 épocas
#    - n_hw=100 muestras en validación IBM (fix del bug n_hw=8)
#  Uso:  bash run_overnight.sh
# ============================================================

# ── Configuración ──────────────────────────────────────────
TOKEN="AOXvkvtsCyO3-cn-LDRwe4DLf2EfEJjlevcf58ra-W-Z"
BACKEND="ibm_marrakesh"  # ibm_fez, ibm_marrakech, ibm_cairo, ibm_lagos, ibm_perth
N_PER_CLASS=80
SHOTS=8192
MAX_ITER=218          # = job de referencia (64.9%)
SEED_BASE=42
OUTPUT_BASE="data/hardware/ibm_marrakesh_noise_aware_runs"

MIN_EXITOS=5
MAX_INTENTOS=30
ESPERA_EXITO=30
ESPERA_FALLO=90
LOG_WRAPPER="logs/overnight_v3_wrapper.log"
# ───────────────────────────────────────────────────────────

mkdir -p logs "$OUTPUT_BASE"

echo "============================================================" | tee -a "$LOG_WRAPPER"
echo "  QNIM Overnight Runner v3 — $(date '+%Y-%m-%d %H:%M:%S')"  | tee -a "$LOG_WRAPPER"
echo "  MAX_ITER=$MAX_ITER  (job ref: 218)"                        | tee -a "$LOG_WRAPPER"
echo "  Noise-aware training: AerSimulator + ibm_marrakech noise model"  | tee -a "$LOG_WRAPPER"
echo "  n_hw=100 muestras validación IBM"                          | tee -a "$LOG_WRAPPER"
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
    exit 0
else
    echo "❌  Solo $exitos/$MIN_EXITOS éxitos tras $intento intentos  [$(date '+%H:%M:%S')]" | tee -a "$LOG_WRAPPER"
    exit 1
fi