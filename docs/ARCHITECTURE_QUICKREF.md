# 📋 QUICK REFERENCE: MAPEO DE ARCHIVOS IMPORTANTES

## 🔵 DOMAIN LAYER - Archivos Clave

### `src/domain/astrophysics/`

```
entities/gravitational_wave.py
├── GWSignal                      [Clase] Representa una onda gravitacional
│   ├── h_plus: NDArray           Polarización + de la onda
│   ├── h_cross: NDArray          Polarización × de la onda
│   ├── detector: DetectorType    H1 (Hanford), L1 (Livingston), V1 (Virgo)
│   ├── gps_time: GPSTime         Timestamp GPS (precisión nanosegundos)
│   ├── theory: TheoryFamily      RG, QUANTUM, MODIFIED_GRAVITY
│   └── validate_physics()        Valida invariantes del dominio
└── QuantumDecodedEvent           [Clase] Evento después de análisis

value_objects/theory_family.py
├── TheoryFamily (Enum)
│   ├── RG                        General Relativity (baseline)
│   ├── QUANTUM                   Quantum corrections
│   └── MODIFIED_GRAVITY          Alternative theories
└── Immutable: una vez creado, no cambia

value_objects/detector_type.py
├── DetectorType (Enum)
│   ├── H1                        LIGO Hanford
│   ├── L1                        LIGO Livingston
│   └── V1                        Virgo
└── Usada para tracking de fuente

sstg/simple_generator.py
└── SimpleSyntheticGWGenerator    [Clase] Generador de ondas sintéticas
    ├── generate_event()          Crea evento único (GR, QUANTUM, MOD_GRAVITY)
    ├── generate_batch()          Crea múltiples eventos
    └── Usa providers para física

sstg/providers/
├── kerr_provider.py              Spacetime de Kerr (rotación BH)
├── scalarTensor_provider.py      Gravedad con campo escalar
└── lqg_provider.py               Loop Quantum Gravity
```

**USO**: Nunca importes directamente en aplicación, usa a través de puertos (application/ports.py)

---

### `src/domain/quantum/`

```
entities/quantum_state.py
└── QuantumState                  [Clase] Estado cuántico abstracto
    ├── amplitudes: Dict          {|000>: 0.5, |111>: 0.5, ...}
    ├── probabilities()           Calcula amplitudes al cuadrado
    └── to_bloch_sphere()         Proyecta a esfera de Bloch

services/vqc_classifier.py
└── VQCClassifier                 [Clase] Clasificador VQC (sin Qiskit)
    ├── classify(features)        Ejecuta VQC lógicamente
    └── Retorna probabilidades sin detalles de qubits
```

**USO**: Define qué hacemos, no cómo lo hacemos

---

## 🟢 APPLICATION LAYER - Archivos Clave

### `src/application/ports.py`

```
IDataLoaderPort                  [Interfaz abstracta]
├── load_dataset()               Carga datos (sin especificar HDF5 vs Parquet)
└── Implementación: storage/hdf5_exporter.py

IStoragePort                     [Interfaz abstracta]
├── save()                       Guarda resultados
└── load()                       Carga resultados

IQuantumOptimizerPort            [Interfaz abstracta]
├── optimize(problem)            Optimiza (sin especificar D-Wave vs Neal)
└── Implementación: neal_annealer_adapter.py, dwave_cloud_adapter.py

IQuantumMLTrainerPort            [Interfaz abstracta]
├── train()                      Entrena modelo cuántico
└── Implementación: qiskit_vqc_trainer.py

IPreprocessingPort               [Interfaz abstracta]
├── transform()                  Preprocesa datos
└── Implementación: sklearn_preprocessing_adapter.py

ISyntheticDataGeneratorPort      [Interfaz abstracta]
├── generate_batch()             Genera datos sintéticos
└── Implementación: sstg_adapter.py
```

**CLAVE**: Estos puertos son los "contratos" entre capas. Los scripts los implementan en infrastructure/.

---

### `src/application/dto.py`

```
ClassicParametersExtracted       [Dataclass] Parámetros clásicos extraídos
├── m1_solar_masses              Masa 1 en masas solares
├── m2_solar_masses              Masa 2 en masas solares
├── chirp_mass                   Masa de chirp (observable)
├── effective_spin               Spin efectivo
├── distance_mpc                 Distancia en Megaparsecs
└── snr                          Signal-to-Noise Ratio

BeyondGRSignature                [Dataclass] Firma de nueva física
├── anomaly_magnitude            Qué tan fuerte la anomalía
├── theory_probability           P(beyond-GR)
└── confidence_sigma             Significancia en σ

InferenceResult                  [Dataclass] Resultado de predicción
├── event_id                     ID del evento
├── predicted_theory             RG, QUANTUM, MODIFIED_GRAVITY
├── probabilities: Dict          Probabilidades de cada clase
├── classic_params               Parámetros extraídos
├── beyond_gr_signature          Firma de nueva física
└── timestamp_utc                Cuándo se hizo la predicción

TrainingMetrics                  [Dataclass] Métricas de entrenamiento
├── training_accuracy            Accuracy en set de entrenamiento
├── test_accuracy                Accuracy en set de test
├── precision_per_class: Dict    Precisión por clase
├── recall_per_class: Dict       Recall por clase
└── confusion_matrix             Matriz de confusión
```

**USO**: Garantizan type safety entre capas

---

### `src/application/hybrid_orchestrator.py`

```
HybridInferenceOrchestrator      [Clase] Orquestador principal
├── __init__(quantum_opt, ml_trainer, data_loader, storage)
│   └── Inyección de dependencias (puertos, no implementaciones)
├── execute_inference()          Ejecuta inferencia completa
│   ├── 1. Prepara datos
│   ├── 2. Rama D-Wave (optimización)
│   ├── 3. Rama IBM (VQC)
│   ├── 4. Agregación de resultados
│   └── 5. Retorna InferenceResult
└── El script phase4_inference.py lo usa
```

**CLAVE**: No tiene detalles de Qiskit, D-Wave, sklearn - todo abstraído en puertos

---

## 🔴 INFRASTRUCTURE LAYER - Archivos Clave

### `src/infrastructure/storage/`

```
hdf5_exporter.py                 [Implementación de IStoragePort]
└── HDF5Exporter
    ├── save(data, path)         Usa h5py para guardar a disco
    ├── load(path)               Lee archivos .h5
    └── Traducción: InferenceResult ↔ HDF5 bytes

quantum_dataloader.py            [Implementación de IDataLoaderPort]
└── QuantumDatasetLoader
    ├── load_from_h5()           Lee datos GW de archivos .h5
    ├── load_gw150914()          Carga evento real LIGO
    └── Traducción: HDF5 ↔ NDArray
```

**USO**: Aquí SÓLO hay detalles de librerías externas (h5py, numpy, etc.)

---

### `src/infrastructure/`

```
ibm_quantum_adapter.py           [Implementación de IQuantumOptimizerPort]
└── IBMQuantumAdapter
    ├── __init__(token)          Se conecta a IBM Quantum
    ├── optimize(problem)        Envía a IBM, espera resultado
    └── Traducción: QUBO ↔ IBM Quantum

neal_annealer_adapter.py         [Implementación de IQuantumOptimizerPort]
└── NealSimulatedAnnealerAdapter
    ├── optimize(problem)        Usa neal (D-Wave simulado)
    └── Traducción: QUBO ↔ neal

qiskit_vqc_trainer.py            [Implementación de IQuantumMLTrainerPort]
└── QiskitVQCTrainer
    ├── train(X, y)              Entrena VQC usando Qiskit
    ├── predict(X)               Predice usando Qiskit
    └── Traducción: features ↔ Qiskit circuits

sklearn_preprocessing_adapter.py [Implementación de IPreprocessingPort]
└── SklearnPreprocessor
    ├── fit(X)                   Ajusta PCA, normalización
    ├── transform(X)             Preprocesa datos
    └── Traducción: raw signals ↔ features

matplotlib_metrics_reporter.py   [Implementación de IMetricsReporterPort]
└── MatplotlibMetricsReporter
    ├── plot_confusion_matrix()  Dibuja matriz de confusión
    ├── plot_roc_curve()         Dibuja curva ROC
    └── Traducción: números ↔ gráficos
```

**REGLA DE ORO**: Infrastructure PUEDE importar sklearn, qiskit, h5py, etc.  
Pero NUNCA debería contener lógica de negocio (eso va en domain/)

---

## 📝 SCRIPTS LAYER - Archivos de Ejecución

### `scripts/phase1_diagnostics.py`

```
Propósito: Valida que todo el sistema funciona

Pasos:
1. Importa SimpleSyntheticGWGenerator (domain/)
   └─ Verifica que los imports funcionan

2. Crea 1 evento sintético (test)
   └─ Verifica que la física funciona

3. Guarda a HDF5 (test)
   └─ Verifica que la I/O funciona

4. Carga configuración (CLI)
   └─ Verifica que la config funciona

Output: Console output, estado OK/ERROR
```

**Ejecutar**: `python scripts/phase1_diagnostics.py`

---

### `scripts/phase3_train.py`

```
Propósito: Entrenar modelo clasificador

Pasos:
1. Carga dataset sintético (infrastructure/storage/hdf5_exporter)
   └─ 500 eventos × 16384 muestras

2. Codifica etiquetas (domain/value_objects/theory_family)
   └─ RG → 0, QUANTUM → 1, MODIFIED_GRAVITY → 2

3. Preprocesa con PCA (infrastructure/sklearn_preprocessing_adapter)
   └─ 16384 features → 64 features

4. Divide train/test (80/20)

5. Entrena Random Forest (sklearn)
   └─ 100 árboles, profundidad máxima 20

6. Evalúa (application/dto/TrainingMetrics)
   ├─ Training accuracy: 94.5%
   ├─ Test accuracy: 84.0%
   └─ Classification report por clase

7. Guarda modelos (infrastructure/storage/)
   ├─ models/qnim_theory_classifier.pkl
   ├─ models/qnim_pca.pkl
   └─ models/qnim_label_encoder.pkl

Output: 
✓ Modelos guardados
✓ Métricas mostradas en console
```

**Ejecutar**: `python scripts/phase3_train.py`

---

### `scripts/phase4_inference.py`

```
Propósito: Analizar GW150914 con modelo entrenado

Pasos:
1. Carga modelo entrenado (infrastructure/storage/)
   ├─ classifier.pkl
   ├─ pca.pkl
   └─ label_encoder.pkl

2. Carga datos LIGO reales (infrastructure/storage/quantum_dataloader)
   ├─ H-H1_LOSC_4_V2-1126259446-32.hdf5 (H1 = Hanford)
   ├─ L-L1_LOSC_4_V2-1126259446-32.hdf5 (L1 = Livingston)
   └─ 131,072 muestras c/u

3. Preprocesa H1 (infrastructure/sklearn_preprocessing_adapter)
   ├─ Toma primeros 16384 samples
   ├─ Aplica PCA
   └─ Normaliza

4. Predice teoría (application/hybrid_orchestrator)
   ├─ classifier.predict() → clase
   ├─ classifier.predict_proba() → probabilidades
   └─ Retorna InferenceResult (dto)

5. Decodifica predicción (domain/value_objects/)
   └─ 0 → RG, 1 → QUANTUM, etc.

6. Muestra resultados (presentation/)
   ├─ Teoría predicha: RG
   ├─ Probabilidades: RG 84.3%, QUANTUM 8.3%, MOD_GRAVITY 7.4%
   └─ Confidence: 84.3%

7. Guarda resultados (infrastructure/storage/)
   └─ reports/gw150914_analysis_results.json

Output:
✓ Predicción: General Relativity
✓ Confianza: 84.3%
✓ JSON guardado
```

**Ejecutar**: `python scripts/phase4_inference.py`

---

### `scripts/phase5_final.py`

```
Propósito: Generar reportes finales

Pasos:
1. Recopila resultados de fases anteriores
   ├─ metrics.json (phase3)
   ├─ gw150914_analysis_results.json (phase4)
   └─ dataset info

2. Genera reportes JSON (infrastructure/reporting/)
   ├─ project_completion_status.json
   ├─ performance_metrics.json
   ├─ dataset_summary.json
   ├─ gw150914_analysis.json
   └─ project_summary.txt

3. Imprime resumen a console (presentation/)
   ├─ Accuracy final: 84%
   ├─ GW150914 resultado: RG
   ├─ Todos archivos generados
   └─ Status: READY FOR DEPLOYMENT

Output:
✓ 5 archivos JSON/TXT
✓ Console summary
```

**Ejecutar**: `python scripts/phase5_final.py`

---

### `scripts/pipelines/00_full_pipeline.py`

```
Propósito: Orquestador maestro que ejecuta fase 1 → 2 → 3

Pasos:
1. PHASE 1: Generar datos sintéticos
   └─ Llama a 01_generate_synthetic_gw.py
   └─ Output: data/synthetic/massive_dataset/synthetic_gw_*.h5 (500 eventos)

2. PHASE 2: Entrenar modelo
   └─ Llama a 02_train_vqc_model.py (o phase3_train.py simplificado)
   └─ Output: models/qnim_theory_classifier.pkl, pca.pkl, etc.

3. PHASE 3: Validación estadística
   └─ Llama a 03_validate_exhaustive.py
   └─ Output: validation_results.json

Maneja dependencias entre fases:
✓ No ejecuta fase 2 si fase 1 falló
✓ Guarda checkpoints
✓ Reporta tiempo total
```

**Ejecutar**: `python scripts/pipelines/00_full_pipeline.py`

---

## 🎯 MAPEO RÁPIDO: "¿DÓNDE ESTÁ X?"

| Pregunta | Respuesta | Archivo |
|----------|-----------|---------|
| ¿Qué es una onda GW? | Clase GWSignal | `src/domain/astrophysics/entities/gravitational_wave.py` |
| ¿Qué teorías existen? | Enum TheoryFamily | `src/domain/astrophysics/value_objects/theory_family.py` |
| ¿Cómo genero datos sintéticos? | SimpleSyntheticGWGenerator | `src/domain/astrophysics/sstg/simple_generator.py` |
| ¿Cuáles son los contratos? | Puertos (interfaces) | `src/application/ports.py` |
| ¿Qué formato tienen los datos? | DTOs | `src/application/dto.py` |
| ¿Cómo entreno el modelo? | ModelTrainingUseCase | `src/application/model_training_service.py` |
| ¿Cómo conecto a IBM? | IBMQuantumAdapter | `src/infrastructure/ibm_quantum_adapter.py` |
| ¿Cómo conecto a D-Wave? | NealSimulatedAnnealerAdapter | `src/infrastructure/neal_annealer_adapter.py` |
| ¿Cómo leo HDF5? | QuantumDatasetLoader | `src/infrastructure/storage/quantum_dataloader.py` |
| ¿Cómo guardo resultados? | HDF5Exporter | `src/infrastructure/storage/hdf5_exporter.py` |
| ¿Cuándo uso cada script? | Ver abajo | `scripts/phase*.py` |

---

## 🔄 FLUJO RECOMENDADO DE LECTURA

**Si quieres entender el proyecto completamente:**

```
1. COMIENZA AQUÍ (este documento)
   └─ Te da un mapa rápido

2. LEE: src/application/ports.py
   └─ Los "contratos" del sistema

3. LEE: src/application/dto.py
   └─ Formato de datos entre capas

4. LEE: src/domain/astrophysics/entities/gravitational_wave.py
   └─ Entidades de dominio

5. LEE: scripts/phase3_train.py
   └─ Ejemplo concreto de flujo

6. LEE: ARCHITECTURE_EXPLAINED.md
   └─ Explicación profunda
```

---

## 🚀 EJECUCIÓN TÍPICA

```bash
# 1. Validar que todo funciona
python scripts/phase1_diagnostics.py

# 2. Generar dataset sintético (si no existe)
python scripts/pipelines/01_generate_synthetic_gw.py

# 3. Entrenar modelo
python scripts/phase3_train.py

# 4. Predecir en GW150914
python scripts/phase4_inference.py

# 5. Generar reportes
python scripts/phase5_final.py

# O todo junto:
python scripts/pipelines/00_full_pipeline.py
```

---

**Última actualización**: 10 de Mayo de 2026
