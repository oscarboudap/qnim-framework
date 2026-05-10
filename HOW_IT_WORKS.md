# QNIM Framework — Cómo Funciona el Código

**Quantum Neural Inference for Multiphysics (QNIM)**  
TFM — Óscar Boullosa Dapena, UNIR 2026

---

## 1. Visión General

QNIM es un framework que combina computación cuántica y física de ondas gravitacionales para **clasificar la teoría de gravedad que generó una señal GW** detectada por LIGO/Virgo. Clasifica entre 10 teorías (Relatividad General + 9 alternativas more-exotic).

El pipeline completo se ejecuta con un único script:

```bash
python scripts/generate_results.py --mode fallback   # siempre funciona (sin dependencias externas)
python scripts/generate_results.py --mode sim        # simulador Qiskit Aer (~10 min)
python scripts/generate_results.py --mode ibm        # hardware IBM ibm_fez real
```

---

## 2. Arquitectura Clean

El proyecto sigue **Clean Architecture** con 4 capas concéntricas. Las dependencias solo fluyen hacia dentro (nunca hacia fuera):

```
┌─────────────────────────────────────────────────────────────┐
│  ENTRY POINT   scripts/generate_results.py                  │
│  (Composition Root: ensambla adaptadores e inyecta deps)    │
├─────────────────────────────────────────────────────────────┤
│  PRESENTATION / CLI  src/presentation/ · src/cli/           │
│  (Mappers DTO, configuración de presentación, logging)      │
├─────────────────────────────────────────────────────────────┤
│  INFRASTRUCTURE      src/infrastructure/                    │
│  (Adaptadores concretos: IBM, D-Wave, matplotlib, HDF5)     │
├─────────────────────────────────────────────────────────────┤
│  APPLICATION         src/application/                       │
│  (Casos de uso, puertos abstractos, DTOs)                   │
├─────────────────────────────────────────────────────────────┤
│  DOMAIN              src/domain/                            │
│  (Entidades, Value Objects, Servicios de dominio puros)     │
└─────────────────────────────────────────────────────────────┘
```

**Regla de oro:** el `domain/` no importa nada de `infrastructure/`. El `application/` no importa nada de `infrastructure/`. Toda la física pura vive en `domain/`.

---

## 3. Flujo de Ejecución Detallado

```
scripts/generate_results.py   ← Composition Root
│
│  1. Lee args CLI (--mode, --n-qubits, --shots, …)
│  2. Instancia adaptadores de infraestructura
│  3. Inyecta adaptadores en el caso de uso
│  4. Llama use_case.execute()
│  5. Imprime resumen final
│
└─→ GenerateExperimentResultsUseCase.execute()
      │   (src/application/use_cases/generate_experiment_results_use_case.py)
      │
      ├─→ [Paso 1] Generación de datos sintéticos
      │     Puerto: ISyntheticDataGeneratorPort
      │     Adaptador: SSTGAdapter (src/infrastructure/sstg_adapter.py)
      │       └─→ genera strain sintético para 10 clases de teorías
      │            usando distribuciones astrofísicas realistas
      │
      ├─→ [Paso 2] Extracción de parámetros físicos (D-Wave QUBO)
      │     Puerto: IQuantumOptimizerPort
      │     Adaptador: NealSimulatedAnnealerAdapter (src/infrastructure/neal_annealer_adapter.py)
      │       └─→ Template matching QUBO sobre el dataset
      │            → m1, m2, χ_eff via recocido cuántico simulado
      │
      ├─→ [Paso 3] Clasificación cuántica (VQC en IBM)
      │     Puerto: IQuantumMLTrainerPort
      │     Adaptador: QiskitVQCTrainer (src/infrastructure/qiskit_vqc_trainer.py)
      │       ├─→ --mode fallback: valores calibrados del TFM (sin Qiskit)
      │       ├─→ --mode sim: Qiskit Aer StatevectorSampler
      │       └─→ --mode ibm: IBM ibm_fez real (156 qubits Heron r1)
      │
      ├─→ [Paso 4] Análisis estadístico (QFI vs CFI, GW150914)
      │     Puerto: IBayesianEstimatorPort
      │     Adaptador: StatisticalAnalysisService (src/infrastructure/statistical_analysis_service.py)
      │       ├─→ QFI via parameter shift rule sobre pesos VQC
      │       ├─→ CFI via bootstrap SVC clásico
      │       └─→ Re-análisis GW150914 + H₀ via sirena estándar
      │
      └─→ [Paso 5] Reporting (figuras, JSON, CSV, LaTeX)
            Puerto: IResultsReporterPort
            Adaptador: MatplotlibResultsReporter (src/infrastructure/reporting/)
              └─→ 7 figuras PNG + full_results.json + results_summary.csv
```

---

## 4. Capa Domain (`src/domain/`)

La física pura. **Cero imports de infraestructura.**

### 4.1 Astrophysics (`src/domain/astrophysics/`)

| Archivo | Rol |
|---|---|
| `entities.py` | Entidades: `QuantumDecodedEvent`, `GWSignal`, `BayesianEvidenceCalculator` |
| `value_objects.py` | Value Objects inmutables: `TheoryFamily`, `SolarMass`, `GPSTime`, `DetectorType`, `Spin`, … |
| `domain_services.py` | Servicios stateless: `AstrophysicalCalculus` (chirp mass, luminosity distance, …) |
| `layers.py` | Las 7 capas de información de una señal GW (Capa 1–7) |
| `repositories.py` | Interfaces de repositorio: `GravitationalWaveRepository` |
| `theory_signatures.py` | Discriminadores bayesianos por teoría (`ScalarTensorAnalyzer`, …) |

#### SSTG — Synthetic Signal Template Generator (`src/domain/astrophysics/sstg/`)

| Archivo | Rol |
|---|---|
| `engine.py` | `StochasticGravityEngine` — muestrea parámetros con distribuciones astrofísicas reales (IMF Salpeter, distribución en volumen) |
| `constraints.py` | `PhysicalConstraints` — valida que los parámetros sean físicamente válidos |
| `generator.py` | Orquestador SSTG completo: llama a PyCBC + inyectores Capas 4–7 |
| `simple_generator.py` | Generador analítico de fallback (sin PyCBC) |
| `providers/kerr_provider.py` | Proveedor de waveforms Kerr (GR puro) |
| `injectors/layer1_scalar_tensor_complete.py` | Inyector Capa 1: acoplamiento escalar-tensorial |
| `injectors/layer3_adscft_transport_complete.py` | Inyector Capa 3: transporte AdS/CFT |
| `injectors/layer4_quantum_foam_complete.py` | Inyector Capa 4: espuma cuántica de Wheeler |
| `injectors/layer5_beyond_gr_complete.py` | Inyector Capa 5: dispersión gravitón masivo (dRGT), dimensiones extra (ADD), Brans-Dicke, LIV |
| `injectors/layer6_horizon_topology_complete.py` | Inyector Capa 6: ecos fuzzball, cuantización área LQG, memoria gravitacional, soft hair |
| `injectors/layer7_quantum_corrections_complete.py` | Inyector Capa 7: correcciones cuánticas de Planck |
| `injectors/layer_zeta_regularization_complete.py` | Regularización zeta de Riemann para divergencias UV |

### 4.2 Quantum (`src/domain/quantum/`)

| Archivo | Rol |
|---|---|
| `interfaces.py` | Puertos de dominio cuántico: `IQuantumAnnealer`, `IGateBasedQuantumComputer` |
| `entities.py` | `VQCTopology`, `QuantumCircuit` |
| `value_objects.py` | `QUBOProblem`, `AnnealingResult`, `QuantumCircuitConfig` |
| `vqc_architecture.py` | Arquitectura del VQC: `ChebyshevFeatureMap`, `EfficientSU2Ansatz` |
| `qubo_formulator.py` | Formula problemas QUBO para template matching de parámetros GW |

### 4.3 Metrology (`src/domain/metrology/`)

| Archivo | Rol |
|---|---|
| `value_objects.py` | `HubbleConstant`, `FisherMatrix`, `PowerSpectralDensity`, `NoHairViolationResult` |
| `fisher_matrix_calculator.py` | Cálculo de matriz de Fisher (cotas Cramér-Rao) — versión con Value Objects |
| `fisher_matrix_new.py` | Cálculo Fisher (versión nueva, con `FisherResult` dataclass) — usada en el pipeline activo |
| `hubble_metrology.py` | `HubbleCosmologyCalculator` — H₀ via sirenas estándar |
| `multipole_validator.py` | `NoHairTheoremAnalyzer` — test del teorema de no-cabello |
| `planck_error_bounds.py` | `QuantumGravitySignificanceCalculator` — significancia de nueva física |
| `bootstrap_resampler.py` | `BootstrapResampler` — remuestreo bootstrap para incertidumbres |

### 4.4 Shared (`src/domain/shared/`)

| Archivo | Rol |
|---|---|
| `exceptions.py` | Jerarquía de excepciones de dominio |
| `zeta_regularization_rigorous.py` | Regularización zeta rigurosa (análisis teórico) |

---

## 5. Capa Application (`src/application/`)

**Casos de uso y puertos abstractos.** Cero imports de infraestructura (matplotlib, qiskit, scipy, etc.).

### Caso de Uso Principal

**`use_cases/generate_experiment_results_use_case.py`**  
`GenerateExperimentResultsUseCase` — orquesta el pipeline completo del TFM. Recibe los 5 adaptadores vía constructor (inyección de dependencias) y los llama en orden.

**`ExperimentConfig`** — DTO inmutable (`frozen=True`) con toda la configuración del experimento. Valida que `n_qubits ≤ 27` (límite práctico de IBM ibm_fez sin QEC).

### Puertos (Interfaces)

**`ports/results_reporter_port.py`** define:

| Puerto | Implementación |
|---|---|
| `IResultsReporterPort` | `MatplotlibResultsReporter` |
| `ISyntheticDataGeneratorPort` | `SSTGAdapter` |
| `IQuantumOptimizerPort` | `NealSimulatedAnnealerAdapter` |
| `IQuantumMLTrainerPort` | `QiskitVQCTrainer` |
| `IBayesianEstimatorPort` | `StatisticalAnalysisService` |

**`ports/__init__.py`** exporta también: `IDataLoaderPort`, `IStoragePort`, `IMetricsReporterPort`, `IQuantumMLTrainerPort`.

### DTOs de Resultados

Viven en `ports/results_reporter_port.py`:
- `FullExperimentResultDTO` — resultado completo del experimento
- `VQCTrainingResultDTO` — métricas del VQC (loss history, accuracy sim/IBM, confusion matrix)
- `QFIAdvantageDTO` — resultado QFI vs CFI por parámetro físico
- `GW150914ReanalysisDTO` — re-análisis del evento GW150914

### Otros Servicios de Application

| Archivo | Rol |
|---|---|
| `process_event_use_case.py` | `DecodeGravitationalWaveUseCase` — decodifica un evento GW individual |
| `hybrid_orchestrator.py` | `HybridInferenceOrchestrator` — coordina IBM + D-Wave |
| `model_training_service.py` | `ModelTrainingUseCase` — entrena el VQC |
| `sstg_service.py` | `SyntheticDataGenerationUseCase` — genera dataset sintético |
| `statistical_validation_service.py` | `StatisticalValidationService` — tests estadísticos |
| `validation_service.py` | `ModelValidationUseCase` — valida el modelo entrenado |
| `mcmc_benchmarking.py` | Comparativa MCMC clásico vs VQC |
| `dto.py` | DTOs del pipeline de inferencia |

---

## 6. Capa Infrastructure (`src/infrastructure/`)

**Adaptadores concretos.** Aquí viven todas las dependencias externas.

| Archivo | Adaptador | Puerto que implementa |
|---|---|---|
| `sstg_adapter.py` | `SSTGAdapter` | `ISyntheticDataGeneratorPort` |
| `neal_annealer_adapter.py` | `NealSimulatedAnnealerAdapter` | `IQuantumAnnealer` / `IQuantumOptimizerPort` |
| `qiskit_vqc_trainer.py` | `QiskitVQCTrainer` | `IQuantumMLTrainerPort` |
| `ibm_quantum_adapter.py` | `IBMQuantumAdapter` | `IGateBasedQuantumComputer` |
| `statistical_analysis_service.py` | `StatisticalAnalysisService` | `IBayesianEstimatorPort` |
| `sklearn_preprocessing_adapter.py` | `SklearnPreprocessingAdapter` | normalización de features |
| `matplotlib_metrics_reporter.py` | `MatplotlibMetricsReporter` | `IMetricsReporterPort` |
| `exceptions.py` | `ReportingException`, `TrainingException` | — |
| `reporting/matplotlib_results_reporter.py` | `MatplotlibResultsReporter` | `IResultsReporterPort` |
| `storage/hdf5_exporter.py` | `HDF5Exporter` | persistencia de datasets |
| `storage/massive_loader.py` | `MassiveDataLoader` | carga masiva de HDF5 |
| `storage/quantum_dataloader.py` | `QuantumDatasetLoader` | carga de datos para IBM |
| `ibm_quantum_results_collector.py` | `IBMQuantumResultsCollector` | recolección de métricas IBM |
| `visualization_engine.py` | `VisualizationEngine` | generación de figuras (7 tipos) |
| `matricula_vectors.py` | utilidades | dataset físicamente honesto + benchmark |

---

## 7. Capa Presentation / CLI (`src/presentation/`, `src/cli/`)

### Presentation (`src/presentation/`)

| Archivo | Rol |
|---|---|
| `cli_presenter.py` | Formatea y muestra resultados en consola |
| `configuration.py` | Configuración de la presentación |
| `dto_mappers.py` | Convierte DTOs de dominio a formatos de presentación |
| `validation_visualizations.py` | Visualizaciones de validación estadística |
| `visualize_results.py` | Orquesta la visualización de resultados |

### CLI (`src/cli/`)

| Archivo | Rol |
|---|---|
| `script_config.py` | Configuración de los scripts CLI |
| `script_container.py` | Contenedor de dependencias para scripts |
| `script_exceptions.py` | Excepciones específicas de CLI |
| `script_logging.py` | Configuración centralizada de logging |

---

## 8. Scripts (`scripts/`)

### Scripts Activos (Pipeline Principal)

| Script | Función |
|---|---|
| `generate_results.py` | **Punto de entrada principal.** Composition root del TFM. |
| `generate_reports.py` | Genera informes PDF/HTML desde JSON de resultados |
| `generate_corner_plot.py` | Corner plot del posterior (m1, m2, χ_eff, d_L) |
| `plot_results.py` | Genera figuras adicionales desde resultados existentes |
| `monitor_generation.py` | Monitoriza el progreso de generación de datos |
| `statistical_validation_sweep.py` | Sweep de validación estadística sobre parámetros |
| `validate_dataset.py` | Valida el dataset sintético generado |
| `validate_ibm_connection.py` | Verifica conexión con IBM Quantum |
| `validate_infrastructure.py` | Verifica que todas las dependencias estén disponibles |
| `validate_statistical.py` | Ejecuta suite de tests estadísticos |
| `verify_h5.py` | Verifica integridad de archivos HDF5 |

### Pipelines Secuenciales (`scripts/pipelines/`)

| Script | Función |
|---|---|
| `00_full_pipeline.py` | Pipeline completo secuencial |
| `01_generate_synthetic_gw.py` | Paso 1: genera dataset sintético GW |
| `02_train_vqc_model.py` | Paso 2: entrena el VQC |
| `02_train_vqc_model_simplified.py` | Paso 2: versión simplificada de entrenamiento |
| `03_validate_exhaustive.py` | Paso 3: validación exhaustiva estadística |

---

## 9. Tests (`test/`)

```
test/
├── conftest.py              ← fixtures de pytest compartidos
├── sanity_check.py          ← test rápido de integridad del proyecto
├── unit/                    ← tests unitarios (sin I/O externo)
│   ├── test_layer1_scalar_tensor.py
│   ├── test_layer2_zeta_regularization.py
│   ├── test_layer3_adscft_transport.py
│   ├── test_layer4_quantum_foam.py
│   ├── test_layer5_beyond_gr.py
│   ├── test_layer6_horizon_topology.py
│   ├── test_layer7_quantum_corrections.py
│   ├── test_mcmc_benchmarking.py
│   ├── test_physics_layers.py
│   ├── test_zeta_regularization_rigorous.py
│   ├── test_statistical_validation_service.py
│   ├── test_synthetic_generation.py
│   └── test_visualization_scripts.py
└── integration/             ← tests de integración (pueden conectar a IBM)
    ├── test_generator_integration.py
    ├── test_gw_analysis.py
    ├── test_hybrid_inference.py
    ├── test_quantum_pipeline.py
    ├── test_run_real_ibm.py
    └── test_ibm_submission.py
```

---

## 10. Modos de Ejecución

### `--mode fallback` (por defecto)
- No requiere Qiskit, D-Wave ni IBM
- Usa valores calibrados del TFM hardcoded en `_FallbackXxx` classes dentro de `generate_results.py`
- Produce todas las figuras y reportes en segundos
- Útil para: desarrollo, CI/CD, demo rápida

### `--mode sim`
- Requiere: `qiskit`, `qiskit-machine-learning`, `qiskit-algorithms`
- Usa `StatevectorSampler` de Qiskit (simulador local, sin ruido)
- El VQC se entrena de verdad: ~10 min con `--n-qubits 12 --max-iter 100`
- Produce resultados reales con accuracy >91%

### `--mode ibm`
- Requiere: `qiskit-ibm-runtime`, `IBM_QUANTUM_TOKEN` en `.env`
- Conecta a `ibm_fez` (156 qubits, Heron r1, 2026)
- Límite práctico: `--n-qubits 27` (sin ZNE), `--n-qubits 50` (con `--use-zne`)
- Los shots se envían en jobs a la cola de IBM

### `--mode figures`
- Solo regenera las 7 figuras desde `reports/full_results.json` existente
- No reentrena el VQC
- Útil para ajustar estilos de figuras sin re-ejecutar

---

## 11. Outputs Generados

```
reports/
├── full_results.json          ← Todos los resultados en JSON
├── results_summary.csv        ← Tabla de métricas principal
├── performance_metrics.json   ← Métricas de rendimiento
├── gw150914_analysis.json     ← Re-análisis GW150914
└── figures/
    ├── fig1_convergence.png       ← Curvas pérdida + accuracy VQC
    ├── fig2_confusion_matrix.png  ← Matriz de confusión 10×10
    ├── fig3_qfi_cfi.png           ← QFI vs CFI (ventaja cuántica)
    ├── fig4_accuracy_snr.png      ← Accuracy vs SNR: QNIM vs ResNet
    ├── fig5_barren_plateaus.png   ← Var[∂L/∂θ] vs n_qubits
    ├── fig6_gw150914.png          ← Re-análisis GW150914 + Bayes factors
    └── fig7_dashboard.png         ← Dashboard global IBM Quantum

logs/
└── generate_results.log       ← Log completo de la ejecución

models/
├── qnim_vqc_weights.npy       ← Pesos del VQC entrenado
└── qnim_test_weights_36.npy   ← Pesos de test (36 params, n=12 qubits)
```

---

## 12. Dependencias Clave

| Dependencia | Uso | Capa |
|---|---|---|
| `qiskit` | Circuitos cuánticos, transpilación | Infrastructure |
| `qiskit-machine-learning` | Clase `VQC`, `ZZFeatureMap`, `RealAmplitudes` | Infrastructure |
| `qiskit-algorithms` | Optimizador `SPSA` | Infrastructure |
| `qiskit-ibm-runtime` | Conexión IBM Quantum, `SamplerV2` | Infrastructure |
| `neal` | Simulador D-Wave (recocido cuántico) | Infrastructure |
| `matplotlib` | Generación de figuras | Infrastructure |
| `numpy` | Álgebra lineal, arrays | Infrastructure / Domain |
| `scipy` | Estadística, optimización | Infrastructure |
| `pycbc` | Waveforms GW (opcional, SSTG completo) | Domain |
| `h5py` | Lectura/escritura HDF5 (datos LIGO) | Infrastructure |

---

## 13. Diagrama de Dependencias entre Módulos

```
scripts/generate_results.py
    ↓ imports
src/application/use_cases/generate_experiment_results_use_case.py
    ↓ imports (puertos abstractos)
src/application/ports/results_reporter_port.py
    ↑ implementado por
src/infrastructure/
    ├── sstg_adapter.py              ← genera datos
    ├── neal_annealer_adapter.py     ← D-Wave QUBO
    ├── qiskit_vqc_trainer.py        ← VQC cuántico
    ├── statistical_analysis_service.py ← QFI/CFI, GW150914
    └── reporting/matplotlib_results_reporter.py ← 7 figuras

src/infrastructure/ → src/domain/ (value objects, interfaces)
src/application/   → src/domain/ (value objects, interfaces)
src/domain/        → nadie (dependencias cero hacia afuera)
```

---

## 14. Principios de Diseño Aplicados

| Principio | Aplicación |
|---|---|
| **Clean Architecture** | 4 capas con dependencias hacia dentro |
| **Hexagonal Architecture** | Puertos en application/, adaptadores en infrastructure/ |
| **Dependency Injection** | Adaptadores inyectados en el constructor del caso de uso |
| **DDD (Domain-Driven Design)** | Entidades, Value Objects, Servicios de dominio, Repositorios |
| **Single Responsibility** | Cada clase = una responsabilidad |
| **Open/Closed** | Nuevo backend IBM → solo nuevo adaptador, sin tocar application/ |
| **Interface Segregation** | Puertos específicos por capacidad (no un puerto gigante) |
| **Hot-Swap** | `NealSimulatedAnnealerAdapter` ↔ D-Wave QPU real sin cambiar application/ |

---

*Generado automáticamente — QNIM Framework TFM 2026*
