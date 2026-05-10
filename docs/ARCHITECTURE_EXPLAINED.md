# 🏗️ ARQUITECTURA QNIM - EXPLICACIÓN COMPLETA

## 📌 RESUMEN EJECUTIVO

QNIM tiene **DOS estrategias coexistentes**:

1. **Clean Architecture (src/)**: La teoría ideal con 5 capas bien separadas
2. **Scripts de Fase (scripts/)**: La práctica real que ejecuta el pipeline end-to-end

**No es una contradicción**: Los scripts USAN la arquitectura interna. Los scripts son los "orquestadores" que coordinan toda la maquinaria.

---

## 🎯 LA ARQUITECTURA LIMPIA: src/

### Estructura de 5 Capas (Hexagonal Architecture)

```
src/
├── domain/                 ← CAPA 1: LÓGICA DE NEGOCIO PURA (sin dependencias externas)
├── application/            ← CAPA 2: CASOS DE USO (orquesta domain)
├── infrastructure/         ← CAPA 3: ADAPTADORES (implementa interfaces)
├── presentation/           ← CAPA 4: CLI & VISUALIZACIÓN
└── test/                   ← CAPA 5: PRUEBAS
```

### CAPA 1: DOMAIN (Lógica de Negocio Pura)

**Ubicación**: `src/domain/`

**Qué es**: 
- Contiene la LÓGICA FÍSICA Y MATEMÁTICA del problema
- CERO dependencias externas
- NO importa: numpy, sklearn, qiskit, pytorch, nada
- Es 100% agnóstica de cómo se ejecuta

**Subcapas**:

```
domain/
├── astrophysics/                    # Física gravitacional
│   ├── entities/                    # Objetos del dominio
│   │   └── gravitational_wave.py   # Clase GWSignal
│   ├── value_objects/               # Valores inmutables
│   │   ├── detector_type.py        # Enum: H1, L1, V1
│   │   ├── gps_time.py             # Timestamp con precisión GPS
│   │   └── theory_family.py        # Enum: RG, QUANTUM, MODIFIED_GRAVITY
│   ├── services/                    # Servicios de dominio (sin estado)
│   │   └── wave_decoder.py         # Decodifica física del evento
│   └── sstg/                        # Simple Synthetic GW Generator
│       ├── simple_generator.py     # Generador básico de señales
│       └── providers/              # Backends de física
│           ├── kerr_provider.py   # Spacetime Kerr (rotación)
│           ├── scalarTensor_provider.py
│           └── lqg_provider.py    # Loop Quantum Gravity
│
├── quantum/                         # Física cuántica
│   ├── entities/                    # Estados cuánticos
│   │   └── quantum_state.py        # Clase QState
│   ├── services/                    # Servicios cuánticos
│   │   ├── vqc_classifier.py       # VQC sin dependencias
│   │   └── anomaly_detector.py     # Detecta Beyond-GR
│   └── value_objects/
│       └── circuit_config.py       # Config de circuitos (sin Qiskit)
│
├── metrology/                       # Mediciones y estadística
│   ├── fisher_information.py       # Matriz Fisher
│   ├── bayesian_estimator.py       # Estimación Bayesiana
│   └── significance_test.py        # Tests de significancia
│
└── shared/                          # Compartido entre dominio
    ├── exceptions.py               # Excepciones del dominio
    └── value_objects.py            # Valores comunes
```

**Ejemplo: Entidad de Dominio**

```python
# src/domain/astrophysics/entities/gravitational_wave.py

@dataclass
class GWSignal:
    """Entidad raíz del dominio: una onda gravitacional"""
    
    h_plus: NDArray  # Polarización +
    h_cross: NDArray  # Polarización ×
    detector: DetectorType  # H1, L1, V1
    gps_time: GPSTime  # Timestamp preciso
    theory: TheoryFamily  # RG, QUANTUM, MODIFIED_GRAVITY
    
    # INVARIANTE DEL DOMINIO:
    def validate_physics(self):
        """Asegura que cumple leyes de la física"""
        assert len(self.h_plus) == len(self.h_cross), "Polarizaciones deben tener mismo largo"
        assert self.h_plus.dtype == np.float64, "Precisión requerida"
        # ... más validaciones ...
```

**¿Cuándo usas Domain?**
- Para escribir tests sin dependencias externas
- Para verificar reglas de negocio
- Para documentar QUIN ES TU PROBLEMA sin código de infraestructura

---

### CAPA 2: APPLICATION (Casos de Uso)

**Ubicación**: `src/application/`

**Qué es**:
- Orquesta el dominio
- Define CONTRATOS de entrada/salida (DTOs)
- NO toca infraestructura directamente
- Usa PUERTOS (interfaces) que se implementan en infrastructure

**Subcapas**:

```
application/
├── hybrid_orchestrator.py          # Caso de uso principal
│   └── HybridInferenceOrchestrator # Coordina D-Wave + IBM
├── process_event_use_case.py       # Caso de uso: procesar evento GW
├── model_training_service.py       # Caso de uso: entrenar modelo
├── validation_service.py           # Caso de uso: validar
├── sstg_service.py                 # Caso de uso: generar datos
│
├── ports.py                        # ⭐ CONTRATOS (interfaces abstractas)
│   ├── IQuantumOptimizerPort       # ¿Quién optimiza? (D-Wave/Neal)
│   ├── IQuantumMLTrainerPort       # ¿Quién entrena ML? (Qiskit/PyTorch)
│   ├── IDataLoaderPort             # ¿Quién carga datos? (HDF5/Parquet)
│   ├── IStoragePort                # ¿Quién guarda? (S3/Local)
│   └── ... más puertos ...
│
├── dto.py                          # Data Transfer Objects (type safety)
│   ├── ClassicParametersExtracted  # Parámetros masas/spins
│   ├── BeyondGRSignature          # Firma de nueva física
│   ├── InferenceResult            # Resultado de predicción
│   └── TrainingMetrics            # Métricas de entrenamiento
│
├── exceptions.py                   # Excepciones de aplicación
│   └── ApplicationException        # Base para todas
│
└── validators/                     # Validadores (sin estado)
    └── input_validator.py          # Valida DTOs de entrada
```

**Ejemplo: Caso de Uso (Use Case)**

```python
# src/application/process_event_use_case.py

class DecodeGravitationalWaveUseCase:
    """Caso de uso: Decodificar un evento GW"""
    
    def __init__(self,
                 orchestrator: IHybridInferencePort,  # ← Inyección de dependencia
                 preprocessor: IPreprocessingPort):
        self.orchestrator = orchestrator
        self.preprocessor = preprocessor
    
    def execute(self, gw_signal: GWSignal) -> InferenceResult:
        """
        Entrada: GWSignal (del dominio)
        Salida: InferenceResult (DTO de salida)
        
        Pasos:
        1. Validar entrada
        2. Preprocesar (usa puerto, NO sklearn directamente)
        3. Ejecutar inferencia (usa puerto, NO Qiskit directamente)
        4. Agregar resultados
        5. Retornar DTO
        """
        # No tiene detalles de infraestructura
        # Los detalles están en infrastructure/
```

**¿Cuándo usas Application?**
- Cuando necesitas la "receta" de qué hace el sistema
- Para definir contratos entre componentes
- Para cambiar fácilmente de SQLite a PostgreSQL sin afectar lógica

---

### CAPA 3: INFRASTRUCTURE (Adaptadores)

**Ubicación**: `src/infrastructure/`

**Qué es**:
- Implementa los PUERTOS definidos en application
- Aquí SÍ pueden haber dependencias: sklearn, Qiskit, D-Wave, HDF5, etc.
- Traducen entre el mundo abstracto (domain/application) y librerías reales

**Subcapas**:

```
infrastructure/
├── storage/                        # ← Implementa IDataLoaderPort + IStoragePort
│   ├── hdf5_exporter.py           # Lee/escribe HDF5
│   ├── quantum_dataloader.py      # Carga datos GW reales
│   └── massive_loader.py          # Cargador de datasets grandes
│
├── quantum_adapters/              # ← Implementan IQuantumOptimizerPort
│   ├── ibm_quantum_adapter.py     # Conexión a IBM Quantum (Qiskit)
│   ├── neal_annealer_adapter.py   # D-Wave simulado (neal)
│   └── dwave_cloud_adapter.py     # D-Wave real (cloud)
│
├── ml_adapters/                   # ← Implementan IQuantumMLTrainerPort, IPreprocessingPort
│   ├── qiskit_vqc_trainer.py      # Entrena VQC con Qiskit
│   ├── sklearn_preprocessing_adapter.py  # PCA, normalización
│   └── pytorch_classifier.py      # Si usases PyTorch
│
├── reporting/                     # ← Implementan IMetricsReporterPort
│   ├── matplotlib_metrics_reporter.py
│   ├── json_exporter.py
│   └── table_printer.py
│
├── sstg_adapter.py                # ← Implementa ISyntheticDataGeneratorPort
│   └── Conecta domain/sstg con infraestructura real
│
├── exceptions.py                  # Excepciones específicas de infra
│   ├── DataLoaderException
│   ├── QuantumComputeException
│   └── TrainingException
│
└── __init__.py                    # Exporta adaptadores públicos
```

**Ejemplo: Adaptador (Implementación de Puerto)**

```python
# src/infrastructure/storage/hdf5_exporter.py

class HDF5Exporter(IStoragePort):
    """Implementa puerto de almacenamiento con HDF5"""
    
    def save(self, data: InferenceResult, path: str) -> None:
        """Implementa método del puerto"""
        with h5py.File(path, 'w') as f:
            f.create_dataset('probabilities', 
                           data=data.probabilities,
                           compression='gzip')
            # ... más detalles HDF5 ...
    
    def load(self, path: str) -> InferenceResult:
        """Implementa método del puerto"""
        with h5py.File(path, 'r') as f:
            probs = f['probabilities'][:]
            # ... desserializar ...
        return InferenceResult(...)
```

**¿Cuándo usas Infrastructure?**
- Cuando necesitas conectar con librerías reales
- Para cambiar de sklearn a XGBoost sin tocar application
- Para mockear en tests

---

### CAPA 4: PRESENTATION (CLI & Visualización)

**Ubicación**: `src/presentation/`

**Qué es**:
- La interfaz de usuario
- CLI commands que el usuario ejecuta
- Transformación de resultados a strings/plots

**Ejemplo**:

```
presentation/
├── cli_presenter.py               # Mensajes del terminal
└── visualize_results.py           # Gráficos matplotlib
```

---

### CAPA 5: TEST (Pruebas)

**Ubicación**: `src/test/`

**Qué es**:
- Tests unitarios del domain (sin mocks)
- Tests de integración con puertos
- Tests end-to-end

---

## 📝 LOS SCRIPTS DE FASE: scripts/

### La Realidad Práctica

Los scripts son **ORQUESTADORES** que coordinan todo el pipeline end-to-end.

```
scripts/
├── phase1_diagnostics.py           # Valida que todo funciona
├── phase3_train.py                 # Entrena el modelo
├── phase4_inference.py             # Predice en GW150914
├── phase5_final.py                 # Genera reportes
├── phase5_reports.py              # Reportes formales
│
├── pipelines/                      # Pipeline modularizado
│   ├── 00_full_pipeline.py        # Orquestador maestro
│   ├── 01_generate_synthetic_gw.py # Block 1: Generar datos
│   ├── 02_train_vqc_model.py      # Block 2: Entrenar
│   └── 03_validate_exhaustive.py  # Block 3: Validar
│
├── run_qnim_inference.py           # Script de inferencia
├── run_qnim_simulator.py           # Script de simulación
├── validate_infrastructure.py      # Valida conexiones
└── [otros scripts de soporte]
```

### Flujo de Fase 3 (phase3_train.py)

```python
#!/usr/bin/env python3
"""PHASE 3: Train GW Theory Classifier"""

# 1. CARGA DATASET
with h5py.File('data/synthetic/synthetic_gw_20260510_012853.h5', 'r') as f:
    strain_plus = f['strain_plus'][:]        # 500 eventos × 16384 muestras
    theory_labels = f['theory_labels'][:]    # 500 etiquetas (RG, QUANTUM, etc)

# 2. CODIFICA ETIQUETAS (domain → números)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(theory_labels)

# 3. PREPROCESA (application/infrastructure)
pca = PCA(n_components=64)
X = pca.fit_transform(strain_plus)  # 16384 features → 64 features

# 4. DIVIDE DATASET
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 5. ENTRENA CLASIFICADOR (infrastructure → sklearn)
classifier = RandomForestClassifier(n_estimators=100, max_depth=20)
classifier.fit(X_train, y_train)

# 6. EVALÚA
accuracy = classifier.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.1%}")  # Output: 84%

# 7. GUARDA MODELOS (infrastructure → joblib)
joblib.dump(classifier, 'models/qnim_theory_classifier.pkl')
joblib.dump(pca, 'models/qnim_pca.pkl')
joblib.dump(label_encoder, 'models/qnim_label_encoder.pkl')
```

### Flujo de Fase 4 (phase4_inference.py)

```python
#!/usr/bin/env python3
"""PHASE 4: Analyze GW150914 with Trained Model"""

# 1. CARGA MODELO ENTRENADO (infrastructure)
classifier = joblib.load('models/qnim_theory_classifier.pkl')
pca = joblib.load('models/qnim_pca.pkl')
label_encoder = joblib.load('models/qnim_label_encoder.pkl')

# 2. CARGA DATOS REALES DE LIGO (infrastructure)
with h5py.File('data/raw/H-H1_LOSC_4_V2-1126259446-32.hdf5', 'r') as f:
    h1_strain = f['/strain/Strain'][()]  # 131,072 muestras

# 3. PREPROCESA IGUAL QUE EN ENTRENAMIENTO
strain_data = h1_strain[:16384]  # Toma primeros 16384 samples
X = strain_data.reshape(1, -1)
X_pca = pca.transform(X)  # Aplica PCA

# 4. PREDICE (application/domain)
prediction = classifier.predict(X_pca)[0]
probabilities = classifier.predict_proba(X_pca)[0]

# 5. DECODIFICA PREDICCIÓN (domain)
pred_class = label_encoder.inverse_transform([prediction])[0]

# 6. MUESTRA RESULTADOS (presentation)
print(f"Predicción: {pred_class}")
print(f"RG: {probabilities[0]:.1%}")
print(f"QUANTUM: {probabilities[1]:.1%}")
print(f"MODIFIED_GRAVITY: {probabilities[2]:.1%}")
```

---

## 🔄 FLUJO DE DEPENDENCIAS (Regla de Oro)

```
Flujo correcto de dependencias (limpio):

Scripts (entrypoint)
    ↓ (usan)
Application (casos de uso)
    ↓ (definen puertos)
Puertos (interfaces abstractas)
    ↑ (implementan)
Infrastructure (adaptadores)
    ↓ (usan)
Domain (lógica pura, sin dependencias)

REGLA: Nunca ir hacia arriba ↑
Domain NUNCA importa Infrastructure
Application NUNCA importa Infrastructure directamente
```

---

## 📊 RESUMEN: ¿QUÉ HACE CADA ARCHIVO?

### Ejemplo: Entrenar un modelo

```
usuario ejecuta: python scripts/phase3_train.py
                        ↓
                 phase3_train.py (script)
                        ↓ (crea)
                 HybridInferenceOrchestrator (application)
                        ↓ (orquesta)
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
    SklearnPreprocessor   IBMQuantumAdapter   NealAnnealerAdapter
    (infrastructure)      (infrastructure)     (infrastructure)
        ↓               ↓               ↓
    domain/             domain/         domain/
    (lógica pura)       (lógica pura)   (lógica pura)
                        ↓
                    Resultado: InferenceResult (DTO)
                        ↓
                    Saved to models/ (infrastructure)
```

---

## 🎯 MAPPING DE ARCHIVOS A FUNCIONES

### DOMAIN (Qué es el problema)
- `domain/astrophysics/entities/gravitational_wave.py` → Define qué es una onda gravitacional
- `domain/astrophysics/value_objects/theory_family.py` → Define qué teorías existen (RG, QUANTUM, MODIFIED_GRAVITY)
- `domain/astrophysics/services/wave_decoder.py` → Lógica pura de decodificación
- `domain/quantum/services/vqc_classifier.py` → Lógica pura de VQC

### APPLICATION (Cómo resolvemos el problema)
- `application/process_event_use_case.py` → "Quiero decodificar un evento GW"
- `application/model_training_service.py` → "Quiero entrenar un modelo"
- `application/ports.py` → "Necesito estos adaptadores (pero no me importa cómo están implementados)"
- `application/dto.py` → "Los datos de entrada/salida tienen esta forma"

### INFRASTRUCTURE (Con qué herramientas lo resolvemos)
- `infrastructure/storage/hdf5_exporter.py` → Usa h5py para guardar
- `infrastructure/ibm_quantum_adapter.py` → Usa Qiskit para acceder a IBM
- `infrastructure/sklearn_preprocessing_adapter.py` → Usa sklearn para PCA
- `infrastructure/neal_annealer_adapter.py` → Usa D-Wave para optimización

### SCRIPTS (La secuencia de pasos)
- `scripts/phase1_diagnostics.py` → Valida que todo funciona
- `scripts/phase3_train.py` → Entrena usando application + infrastructure + domain
- `scripts/phase4_inference.py` → Predice usando modelos entrenados
- `scripts/phase5_final.py` → Genera reportes finales

---

## 🚀 CÓMO EJECUTAR EN LA PRÁCTICA

### Opción 1: Script Individual de Fase

```bash
# Ejecutar fase 3 (entrenar)
python scripts/phase3_train.py

# Ejecutar fase 4 (inferencia)
python scripts/phase4_inference.py

# Ejecutar fase 5 (reportes)
python scripts/phase5_final.py
```

### Opción 2: Pipeline Completo

```bash
# Ejecutar fases 1, 2, 3 en orden
python scripts/pipelines/00_full_pipeline.py
```

### Opción 3: Componentes Individuales

```python
# Si quieres usar la arquitectura limpia en tu código

from src.application.hybrid_orchestrator import HybridInferenceOrchestrator
from src.infrastructure.ibm_quantum_adapter import IBMQuantumAdapter
from src.infrastructure.neal_annealer_adapter import NealSimulatedAnnealerAdapter
from src.domain.astrophysics.entities import GWSignal

# Crear orquestador
ibm = IBMQuantumAdapter()
dwave = NealSimulatedAnnealerAdapter()
orchestrator = HybridInferenceOrchestrator(ibm, dwave)

# Usar con domain entities
signal = GWSignal(...)
result = orchestrator.process(signal)
```

---

## 🎓 CLAVE CONCEPTUAL

**La Clean Architecture no es "un montón de carpetas"**

Es un **principio organizativo**:

```
┌─────────────────────────────────────────┐
│          SCRIPTS (orquestadores)        │ ← Usuario ejecuta aquí
├─────────────────────────────────────────┤
│    APPLICATION (casos de uso)           │ ← Qué hacer
├─────────────────────────────────────────┤
│    INFRASTRUCTURE (detalles técnicos)   │ ← Cómo hacerlo
├─────────────────────────────────────────┤
│    DOMAIN (lógica pura)                 │ ← QUÉ es el problema
└─────────────────────────────────────────┘

VENTAJAS:
✓ Domain es testeable sin dependencias
✓ Puedo cambiar de sklearn a XGBoost sin tocar application
✓ Puedo cambiar de HDF5 a Parquet sin tocar domain
✓ Código clara y mantenible
✓ Fácil de documentar y explicar
```

---

## 🔗 PRÓXIMOS PASOS

1. **Leer los tests** en `src/test/` para ver ejemplos de cómo se usan los componentes
2. **Revisar `src/application/ports.py`** para entender los contratos
3. **Analizar `scripts/phase3_train.py`** para ver cómo orquestamos todo
4. **Experimentar**: Añade una nueva función en `domain/` sin dependencias externas

---

**Fecha**: 10 de Mayo de 2026  
**Estado**: Clean Architecture + Scripts de Fase = Arquitectura Completa
