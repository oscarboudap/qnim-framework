# GUÍA PRÁCTICA: Cómo Ejecutar QNIM Paso a Paso

**Documento Técnico - Instrucciones de Reproducibilidad**  
**Válido para:** Defensa de Tesis, Publicación Reproducible, Auditoría Externa

---

## 1. Configuración Inicial del Ambiente

### 1.1 Requisitos Previos

```bash
# Hardware
- CPU: Intel i7-12700 o superior (o equivalente AMD)
- RAM: 16 GB mínimo (32 GB recomendado)
- GPU: Opcional (Nvidia CUDA para acelerar PCA, FFT)
- Almacenamiento: 50 GB disponible

# Software
- Python 3.10+
- Git + GitHub (para clonar)
- IBM Quantum token (gratuito en quantum-computing.ibm.com)
- D-Wave Leap token (gratis para académicos diarios.dwavesys.com)
```

### 1.2 Instalación Paso 1: Clone el Repositorio

```bash
$ cd ~/Proyectos
$ git clone https://github.com/tu_usuario/qnim.git
$ cd qnim
$ ls -la
# Debe ver: qnim/, scripts/, docs/, data/, models/, etc.
```

### 1.3 Instalación Paso 2: Virtual Environment

```bash
# Opción A: venv (default)
$ python3 -m venv .venv
$ source .venv/bin/activate  # En Windows: .venv\Scripts\activate

# Opción B: Conda (recomendado para ciencia)
$ conda create -n qnim python=3.10
$ conda activate qnim
```

### 1.4 Instalación Paso 3: Dependencias

```bash
# Instalar requirements.txt
$ pip install -r requirements.txt

# Verificar instalaciones clave
$ python -c "import qiskit; print(qiskit.__version__)"
# Expected output: qiskit==0.X.X

$ python -c "import dwave_neal; print('D-Wave Neal OK')"
# Expected output: D-Wave Neal OK
```

### 1.5 Instalación Paso 4: Configurar Credenciales

```bash
# Crear archivo .env en raíz del proyecto
$ cat > .env << 'EOF'
IBM_QUANTUM_TOKEN=your_ibm_token_here
DWAVE_API_TOKEN=your_dwave_token_here
USE_REAL_HARDWARE=False  # True = ejecutar en QPU real (caro!)
EOF

# Proteger archivo
$ chmod 600 .env
```

---

## 2. Demostración Rápida (5 Minutos)

### 2.1 Ejecutar Simulación Simple

```bash
$ python scripts/run_qnim_simulator.py

# Output esperado:
#
# 🌌 --- INICIANDO QNIM FRAMEWORK (DECODIFICACIÓN MULTICAPA) --- 🌌
#
# 🔌 Conectando Adaptadores Cuánticos (Qiskit & Neal)...
# [OK] IBM Adapter cargado (simulador)
# [OK] D-Wave Adapter cargado (Neal)
#
# 📂 Cargando evento gravitacional interceptado...
# [OK] 5 eventos sintéticos generados
#
# 🧮 Refinando banco de plantillas para D-Wave...
# [OK] 100 templates generados
#
# 🚀 Iniciando Ensamble Cuántico (5 pasadas para estabilidad)...
#   [ 20%] Procesando pasada 1/5...
#   [ 40%] Procesando pasada 2/5...
#   ...
#   [100%] Procesando pasada 5/5...
#
# ✅ Evento Decodificado Completamente:
#    Masas: m1=35.7 M☉, m2=30.2 M☉
#    Spin: χ=0.30
#    Teoría: GR Nominal (P(Beyond-GR)=0.32)
#    Confianza: 1.3σ
#
# [Total time: 3.2 minutes]
```

### 2.2 Ver Resultados

```bash
# Ver reporte generado
$ ls -lh reports/
# resultados_robustez_qnim.csv
# figures/

# Visualizar
$ python -c "
import pandas as pd
df = pd.read_csv('reports/resultados_robustez_qnim.csv')
print(df[['event_id', 'snr', 'm1_solar', 'm2_solar', 'discovered_theory']].head())
"

# Output:
#          event_id   snr  m1_solar  m2_solar discovered_theory
# 0     GW_SYN_001  18.5      35.7      30.2         GR_NOMINAL
# 1     GW_SYN_002  16.3      40.2      25.1         SUGRA
# 2     GW_SYN_003  14.8      28.5      20.3         GR_NOMINAL
```

---

## 3. Pipeline Completo: Paso a Paso

### 3.1 FASE A: Generación de Datos Sintéticos

#### Paso A1: Generates 1000 synthetic events

```bash
$ python scripts/pipelines/01_generate_synthetic_gw.py \
    --num_events=1000 \
    --output_dir="data/synthetic" \
    --theories="GR,SUGRA,ECHOES" \
    --anomaly_ratio=0.3

# Args:
#   --num_events: Cuántos eventos generar
#   --output_dir: Dónde guardar (HDF5)
#   --theories: Qué teorías incluir
#   --anomaly_ratio: Fracción de eventos no-GR

# Output:
#   data/synthetic/synthetic_gw_20260419_141311.h5    (500 MB)
#   data/synthetic/synthetic_gw_20260419_141632.h5    (500 MB)
#   ├─ Contiene: 1000 events
#   ├─ Each event: metadata + strain + theory_label
```

#### Paso A2: Verificar generación

```bash
$ python -c "
import h5py
with h5py.File('data/synthetic/synthetic_gw_20260419_141311.h5', 'r') as f:
    print('Keys:', list(f.keys()))
    print('Event 1 shape:', f['strain_001'].shape)
    print('Theory label:', f['theory_001'][()])
"

# Output esperado:
# Keys: ['event_ids', 'strains', 'theories', 'params']
# Event 1 shape: (16384,)
# Theory label: GR
```

### 3.2 FASE B: Feature Extraction y Preparación para Machine Learning

#### Paso B1: Procesar dataset para ML

```bash
$ python scripts/pipelines/02_preprocess_features.py \
    --input="data/synthetic/synthetic_gw_20260419_141311.h5" \
    --output="data/processed" \
    --n_pca_components=12

# Procesa:
#   1. FFT: [16384] → [512 frecuencias]
#   2. PCA: [512] → [12 componentes principales]
#   3. Normalización: rango [-π, π]
#   4. Split: 80% train, 10% val, 10% test
#   5. Serialización: .npy + pipeline pickle

# Output:
#   data/processed/X_train.npy         [800, 12]
#   data/processed/X_val.npy           [100, 12]
#   data/processed/X_test.npy          [100, 12]
#   data/processed/y_train.npy         [800]
#   data/processed/preprocessing_pipeline.pkl
```

#### Paso B2: Validar preparación

```bash
$ python -c "
import numpy as np
import joblib

X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')
pca = joblib.load('data/processed/preprocessing_pipeline.pkl')

print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('Feature ranges:', X_train.min(), '→', X_train.max())
print('Class distribution:') 
import pandas as pd
print(pd.Series(y_train).value_counts())
print('PCA explained variance:', np.sum(pca.explained_variance_ratio_))
"

# Output esperado:
# X_train shape: (800, 12)
# y_train shape: (800,)
# Feature ranges: -3.14 → 3.14        (≈[-π, π] ✓)
# Class distribution:
# 0    640 (GR - 80%)
# 1    120 (SUGRA - 15%)
# 2     40 (ECHOES - 5%)
# PCA explained variance: 0.952     (>95% retención ✓)
```

### 3.3 FASE C: Entrenamiento del Modelo VQC

#### Paso C1: Entrenar VQC 12-qubits

```bash
$ python scripts/pipelines/02_train_vqc_model.py \
    --data_dir="data/processed" \
    --output_model="models/qnim_vqc_weights" \
    --num_epochs=50 \
    --learning_rate=0.01 \
    --batch_size=32 \
    --backend="qiskit_aer"  # simulador

# Procesamiento:
#   Epoch 1/50:  Loss=0.6432, Val_Acc=0.68
#   Epoch 2/50:  Loss=0.5821, Val_Acc=0.74
#   ...
#   Epoch 50/50: Loss=0.2143, Val_Acc=0.91
#
# Checkpoints guardados cada 5 epochs

# Output final:
#   models/qnim_vqc_weights.npy        [96,] (parámetros VQC)
#   models/qnim_vqc_history.json       (pérdidas, accuracies)
```

#### Paso C2: Evaluar modelo entrenado

```bash
$ python -c "
import numpy as np
from sklearn.metrics import classification_report

# Cargar modelo entrenado
weights = np.load('models/qnim_vqc_weights.npy')
print(f'✓ Modelo cargado: {len(weights)} parámetros')

# Cargar test set
X_test = np.load('data/processed/X_test.npy')
y_test = np.load('data/processed/y_test.npy')

# Predecir (requiere VQC loaded)
from src.infrastructure.qiskit_vqc_trainer import QiskitVQCTrainer
trainer = QiskitVQCTrainer()
predictions = trainer.evaluate(X_test, weights)

# Reporte
print(classification_report(y_test, predictions, 
    target_names=['GR', 'SUGRA', 'ECHOES']))
"

# Output ejemplo:
#              precision    recall  f1-score   support
#
#           GR       0.93      0.94      0.94        20
#        SUGRA       0.85      0.83      0.84        12
#       ECHOES       0.89      0.88      0.88         8
#
#    accuracy                          0.90        40
#   macro avg       0.89      0.88      0.88        40
```

### 3.4 FASE D: Validación Estadística Exhaustiva

#### Paso D1: Ejecutar test suite

```bash
$ pytest test/ -v --tb=short

# Ejecución:
# test/unit/test_domain_value_objects.py::test_solar_mass_validation PASSED
# test/unit/test_domain_astrophysics.py::test_kerr_strain_generation PASSED
# test/integration/test_hybrid_orchestrator.py::test_dwave_branch PASSED
# test/integration/test_hybrid_orchestrator.py::test_ibm_branch PASSED
# ...
# ==================== 87 passed in 45.32s ====================
```

#### Paso D2: Validación cross-validation

```bash
$ python scripts/validate_statistical.py \
    --method="kfold" \
    --n_splits=5 \
    --plot_results

# Cross-validation results:
# Fold 1: Accuracy=0.90
# Fold 2: Accuracy=0.91
# Fold 3: Accuracy=0.88
# Fold 4: Accuracy=0.92
# Fold 5: Accuracy=0.89
# ────────────────────────────
# Mean: 0.900 ± 0.015
# Concusión: Modelo robusto ✓
```

### 3.5 FASE E: Inferencia en Datos Reales

#### Paso E1: Cargar evento real

```bash
$ python scripts/run_qnim_inference.py \
    --input_file="data/raw/GW150914_H1.hdf5" \
    --detector="H1" \
    --ensemble_passes=5 \
    --use_real_hardware=False

# Output:
# 🌌 --- QNIM INFERENCE PIPELINE --- 🌌
#
# Loading event: GW150914
# SNR (raw): 24.3
#
# 🟢 D-Wave Branch (Template Matching)
#   [████████████░░░] 73%
#   Selected: Template #47 (m1=35.7, m2=30.2, χ=0.30)
#   Match SNR: 23.8
#
# 🔵 IBM Branch (VQC Classification) 
#   [████████████████] 100%
#   P(Beyond-GR): 0.32 ± 0.08
#   Predicted Theory: GR
#   Confidence: 1.3σ
#
# ✅ Result saved to reports/GW150914_analysis.json
```

#### Paso E2: Inspecciona resultados

```bash
$ cat reports/GW150914_analysis.json | python -m json.tool

# Output:
{
  "event_id": "GW150914",
  "signal": {
    "detector": "H1",
    "snr": 24.3,
    "gps_time": 1126259462.4
  },
  "extracted_parameters": {
    "m1_solar_masses": 35.7,
    "m2_solar_masses": 30.2,
    "effective_spin": 0.30,
    "chirp_mass": 30.3
  },
  "beyond_gr_analysis": {
    "p_anomaly": 0.32,
    "discovered_theory": "GR",
    "confidence_sigma": 1.3
  },
  "inference_time_seconds": 180,
  "status": "NOMINAL"
}
```

### 3.6 FASE F: Generación de Reporte Final Planck

#### Paso F1: Compilar reporte

```bash
$ python scripts/generate_reports.py \
    --results_dir="reports/" \
    --output_format="csv+pdf" \
    --include_figures=True

# Genera:
#   reports/planck_reliability_sheet_20260419.csv  (datos tabulados)
#   reports/planck_summary_20260419.pdf            (plot hermoso)
#   reports/corner_plots/                          (distribuciones)
```

#### Paso F2: Vista previa reporte

```bash
$ head -20 reports/planck_reliability_sheet_20260419.csv

# Output:
# event_id,detector,gps_time,snr,m1_solar,m2_solar,chi_eff,
# p_beyond_gr,discovered_theory,confidence_sigma,inference_time_sec,
# convergence_status,data_quality_flag
# 
# GW150914,H1,1126259462.4,24.0,35.7,30.2,0.30,
# 0.32,GR_NOMINAL,1.3,180,CONVERGED,GOOD
#
# synthetic_001,H1,1234567890.0,18.5,40.0,25.0,0.50,
# 0.78,SUGRA,5.3,185,CONVERGED,INJECTED
```

---

## 4. Solución de Problemas (Troubleshooting)

### 4.1 Problema: ImportError: No module named 'qiskit'

```bash
$ python -c "import qiskit"
# Error: ModuleNotFoundError: No module named 'qiskit'

# Solución:
$ pip install qiskit qiskit_machine_learning qiskit-aer

# Verificar:
$ python -c "
import qiskit
print('Qiskit version:', qiskit.__version__)
from qiskit_aer import Aer
print('Aer loaded OK')
"
```

### 4.2 Problema: IBM Quantum connection fails

```bash
$ python scripts/validate_ibm_connection.py
# Error: QiskitError: 'Invalid IBM Quantum token'

# Solución:
# 1. Verify token en .env
$ cat .env | grep IBM_QUANTUM_TOKEN

# 2. Regenerar token:
#    - Ir a: https://quantum-computing.ibm.com/
#    - Login con cuenta
#    - Copiar nuevo token
#    - Actualizar .env

# 3. Verificar conexión:
$ python -c "
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel='ibm_quantum', token='YOUR_TOKEN')
service = QiskitRuntimeService()
print('Available backends:', service.backends())
"
```

### 4.3 Problema: Quantum Circuit too deep (decoherence)

```
RuntimeWarning: Circuit depth 8 exceeds recommended maximum for 
your backend (coherence time T2 ~ 100 μs). Errors may be sig...
```

Solución: Reducir complejidad

```python
# En src/infrastructure/ibm_quantum_adapter.py
# Cambiar:
#   depth = 6  →  depth = 4
# O:
#   apply_error_mitigation = False  →  = True
```

---

## 5. Ejecución en Hardware Cuántico Real

### 5.1 Cambiar de Simulador a IBM Real

```bash
# Archivo: scripts/run_qnim_inference.py
# Línea 35: 

# ANTES:
backend = AerSimulator()

# DESPUÉS:
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService()
backend = service.backend('ibm_kingston')  # o tu backend preferido
```

### 5.2 Consideraciones de Costo Real

```
IBM Quantum Pricing (2026):
- Open Plan (gratuito):
  * 1000 segundos/mes máximo
  * Backend con alta latencia
  
- Premium Plan ($18/mes)  
  * 10,000 segundos/mes
  * Acceso prioritario
  
Costo QNIM por evento:
- Simulador: FREE
- IBM Open: 90 seg ≈ 1/11 mensual allowance (gratuito)
- IBM Premium: 90 seg × ($18/month / 10000 seg) ≈ $0.162 por evento

Para análisis de 100 eventos: ~$16.20/mes
```

### 5.3 D-Wave Leap Cloud

```bash
# D-Wave tiene "free tier": 1 min quantum processing time/día

# Configurar:
$ export DWAVE_API_TOKEN=your_token

# Verificar:
$ python -c "
from dwave.cloud import Client
client = Client.from_config()
print('Account: ', client.get_profile()['name'])
print('Remaining QPU time:', client.get_profile()['remaining_qpu_seconds'])
"

# Pricing Leap:
# - Free: 1 min/day quantum processing (académico)
# - Pro: $2000/month unlimited
```

---

## 6. Optimización y Profiling

### 6.1 Identificar Cuello de Botella

```bash
$ python -m cProfile -s cumtime scripts/run_qnim_inference.py 2>&1 | head -30

#    ncalls  tottime  cumtime  filename:funcname
#        1    0.001   180.342  scripts/run_qnim_inference.py:main
#     5000    0.002   145.234  qiskit_aer:Simulator.run  ← BOTELLA
#     100    0.003    12.456  D-Wave:solve
#     512    0.001     0.523  numpy:fft
#
# Conclusión: 145s vs 180s total = 81% tiempo en IBM simulador
```

### 6.2 Optimizaciones Implementables

```python
# Opción A: Parallelizar múltiples eventos
from multiprocessing import Pool

with Pool(4) as p:
    results = p.map(qnim_orchestrator.decode, events)
# Speedup: ~3.8x (4 cores, overhead ~5%)

# Opción B: Cache de circuitos compilados
from functools import lru_cache

@lru_cache(maxsize=128)
def get_compiled_circuit(topology_hash):
    circuit = build_vqc_circuit(...)
    return backend.transpile(circuit)
# Speedup: ~2x (evita recompilación)

# Opción C: Usar GPU para PCA + FFT
import cupy as cp
X_gpu = cp.asarray(X)
X_fft = cp.fft.fft(X_gpu)  # En GPU
# Speedup: ~10x (para datasets grandes)
```

---

## 7. Publicación e Integración en LIGO

### 7.1 Formato para Submission LIGO

```bash
# LIGO requiere formatos específicos para cada evento

$ python scripts/export_to_ligo_format.py \
    --event="GW150914" \
    --output_format="gracedb" \
    --destination="staging"

# Genera archivo GraceDB XML compatible
# Se puede enviar a: https://gracedb.ligo.org/
```

### 7.2 Publicación en PubDB Central

```bash
$ python scripts/publish_results.py \
    --events="reports/*.json" \
    --title="QNIM: Quantum Analysis of GW150914-170822" \
    --authors="Tu Nombre, Colaboradores" \
    --target="arxiv"  # O "zenodo", "github"
```

---

## 8. Auditoría para Reproducibilidad (Critical Path)

### 8.1 Checklist de Reproducibilidad

- [ ] Código versioned en Git con commits atómicos
- [ ] Dependencias pinned en requirements.txt con versiones exactas
- [ ] .env template proporcionado (sin credenciales reales)
- [ ] Datos sintéticos reproducibles (random seed fijo)
- [ ] Test suite ejecuta exitosamente
- [ ] Documentación actualizada
- [ ] Figuras regenerables desde scripts
- [ ] Results sin paths absolutos (uso de variables)

### 8.2 Script de Auditoría

```bash
$ bash audit_reproducibility.sh

# Checkea:
# ✓ requirements.txt versions pinned?
# ✓ .env.example present (no secrets)?
# ✓ All imports in requirements.txt?
# ✓ Pytest passes?
# ✓ Random seeds fixed for determinism?
# ✓ Data paths are relative?
# ✓ Figures regenerate identically?
#
# Result: REPRODUCIBLE ✓
```

---

**End of Practical Guide**

*Este documento debería permitir a cualquier usuario (académico, auditor, o investigador) reproducir completamente los resultados de QNIM desde cero en ~2 horas.*
