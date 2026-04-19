# Auditoría DDD: Capa Infrastructure
**Fecha:** Abril 2026 | **Estado:** 40% DDD Compliant → 95% (post-refactor target)  
**Objetivo:** Verificar que infrastructure implementa correctamente los puertos definidos en application

---

## Hallazgos Críticos

### 1. ❌ CRÍTICO: Puertos Faltantes (Implementation Gap)

**Puertos Definidos en Application:**
- ✅ IDataLoaderPort → quantum_dataloader.py [IMPLEMENTADO]
- ✅ IStoragePort → hdf5_exporter.py [IMPLEMENTADO]
- ❌ IQuantumMLTrainerPort → [FALTA] (VQC training)
- ❌ IPreprocessingPort → [FALTA] (sklearn pipeline)
- ❌ IMetricsReporterPort → [FALTA] (visualization)
- ❌ ISyntheticDataGeneratorPort → [FALTA] (SSTG)
- ✅ IQuantumOptimizerPort → neal_annealer_adapter.py [IMPLEMENTADO]
- ❌ ITrainingProgressObserver → [FALTA] (progress callbacks)

**Impacto:**
- Application espera @abstractmethod training() de IQuantumMLTrainerPort
- Infrastructure no puede satisfacer contrato
- ModelTrainingUseCase bloqueado (sin trainer disponible)
- SSTG no tiene adaptador de infraestructura

---

### 2. ❌ CRÍTICO: Return Types Inconsistentes

**Problema 1: hdf5_exporter.py**
```python
def save_batch(self, dataset):  # ❌ No tipado
    # Parámetro sin validación
    # No hay documentación de estructura esperada
    return str(folder_path)  # ✅ OK, pero sin VO
```

**Problema 2: massive_loader.py**
```python
def load_and_preprocess(self, n_components=12, fixed_length=16384):  # ❌ Sin tipos
    # Returns: (X_final, encoded_labels, pca)
    # Tuple de 3 valores - Poco estructurado
    # No hay VO de aplicación
    return X_final, self.encoder.transform(y_raw), pca
```

**Problem 3: quantum_dataloader.py**
```python
def prepare_for_quantum(self, file_path: str, is_real_data: bool = False) -> np.ndarray:
    # ✅ Tipado (buen patrón)
    # Pero retorna np.ndarray crudo, no VO
    return self._format_length(...)
```

**Solución DDD:**
- Todos los returns deben ser VOs tipados (desde application/dto.py)
- hdf5_exporter.save_batch() → SyntheticDatasetInfo
- massive_loader.load_and_preprocess() → Tuple[np.ndarray, np.ndarray, ...] o custom VO
- quantum_dataloader.prepare_for_quantum() → np.ndarray OK (datos brutos)

---

### 3. ❌ ALTO: Responsabilidades Mezcladas

**Problema: massive_loader.py**
```python
class MassiveDatasetLoader:
    def load_and_preprocess(self, ...):
        # 1. Cargador de datos (Infrastructure) ✅
        strain = h5['strain'][:]
        
        # 2. Whitening clásico (Infrastructure) ✅
        strain_whitened = (strain - mean) / std
        
        # 3. PCA (sklearn - Infrastructure) ✅
        pca = PCA(n_components=12)
        X_compressed = pca.fit_transform(X_np)
        
        # 4. Mapeo a ángulos cuánticos (Preprocessing - Infrastructure) ✅
        X_final = (X_compressed - min) / (max - min) * np.pi
        
        # PERO: Retorna (X_final, labels, pca)
        # ¿Quién usa esto? ¿Es un puerto de application?
```

**Problema:** 
- No hay invocación desde application layer
- MassiveDatasetLoader es standalone, no implementa ningún IxxxPort
- Parece dead code (no integrado con hexagonal architecture)

**Solución DDD:**
- Mover load_and_preprocess lógica a IPreprocessingPort
- O crear separación clara:
  - IDataLoaderPort.load_batch() → raw X, y
  - IPreprocessingPort.fit_transform() → compressed X

---

### 4. ❌ ALTO: Sin Implementación de Entrenamiento Cuántico

**Problema: No hay implementación de IQuantumMLTrainerPort**

```python
# application/ports.py define:
class IQuantumMLTrainerPort(ABC):
    def train_vqc(self, X_train, y_train, num_qubits, ...):
        """Entrena VQC"""
        raise NotImplementedError

# infrastructure/ NO tiene adaptador para esto
# ❌ No hay QiskitVQCTrainerAdapter
# ❌ ModelTrainingUseCase.execute() falla en self.trainer.train_vqc()
```

**Impacto:**
- ModelTrainingUseCase no puede entrenar (bloqueado)
- Versión anterior tenía Qiskit importado directamente
- Ahora debe estar en un adaptador

**Solución:**
Crear `qiskit_vqc_trainer.py` que implemente IQuantumMLTrainerPort

---

### 5. ❌ ALTO: Sin Implementación de Reportes

**Problema: No hay implementación de IMetricsReporterPort**

```python
# application/ports.py define:
class IMetricsReporterPort(ABC):
    def report_confusion_matrix(self, TP, TN, FP, FN, output_path):
        raise NotImplementedError

# infrastructure/ NO tiene adaptador para esto
# ❌ No hay MatplotlibMetricsReporter
# ❌ ValidationService no puede reportar
```

**Solución:**
Crear `matplotlib_reporter.py` que implemente IMetricsReporterPort

---

### 6. ❌ ALTO: Sin Implementación de Síntesis de Datos

**Problema: No hay implementación de ISyntheticDataGeneratorPort**

```python
# application/ports.py define:
class ISyntheticDataGeneratorPort(ABC):
    def synthesize_event(self, m1, m2, distance, theory_family):
        raise NotImplementedError

# infrastructure/ NO tiene adaptador
# ❌ SyntheticDataGenerationUseCase no puede generar
# ❌ SSTG sigue en domain/astrophysics (violación identificada)
```

**Solución:**
1. Crear `sstg_adapter.py` en infrastructure/
2. Mover src/domain/astrophysics/sstg → src/infrastructure/sstg/
3. Implementar ISyntheticDataGeneratorPort

---

### 7. ❌ ALTO: Domain Interfaces siendo Usadas Directamente

**Problema: Infrastructure importa interfaces de domain**

```python
# ibm_quantum_adapter.py:
from src.domain.quantum.interfaces import IGateBasedQuantumComputer  # ❌
from src.domain.quantum.entities import VQCTopology  # ❌

# neal_annealer_adapter.py:
from src.domain.quantum.interfaces import IQuantumAnnealer  # ❌
from src.domain.quantum.entities import AnnealingResult  # ❌
```

**¿Es esto un problema?**
- NO en este caso (domain interfaces son correctas)
- Pero hay inconsistencia: 
  - Quantum adapters usan domain interfaces (IGateBasedQuantumComputer)
  - Storage adapters deberían usar domain interfaces también
  - Pero no existen puertos de domain para data loader (crean en application)

**Patrón correcto:**
```
Domain:           Interfaces (IQuantumAnnealer, IGateBasedQuantumComputer)
                           ↑
Infrastructure:   Adapters implementando domain interfaces
                           ↑
Application:      Define nuevos puertos (IDataLoaderPort, IStoragePort, etc.)
                  que NO están en domain
```

**Solución:** Documentar claramente:
- Quantum interfaces = Domain (interfaces computacionales)
- Data interfaces = Application (interfaces de orquestación)
- Esto NO es violación, es separación de concerns

---

### 8. ❌ MEDIO: Missing Error Handling

**Problema: Sin custom exceptions**

```python
# hdf5_exporter.py
def save_batch(self, dataset):
    # ❌ Si dataset es None, crash silencioso
    # ❌ Si Path no tiene permisos, excepción genérica
    # ❌ Sin validación de estructura de dataset
    
# quantum_dataloader.py
def prepare_for_quantum(self, file_path):
    # ✅ Hay FileNotFoundError pero no es custom
    # ❌ Si file_path no es .h5, falla genérica
```

**Solución:**
Crear custom exceptions en infrastructure:

```python
class InfrastructureException(Exception):
    """Base para exceptions de infrastructure"""
    pass

class DataLoaderException(InfrastructureException):
    """Error al cargar datos"""
    pass

class StorageException(InfrastructureException):
    """Error al guardar datos"""
    pass

class QuantumComputeException(InfrastructureException):
    """Error en computación cuántica"""
    pass
```

---

### 9. ❌ MEDIO: __init__.py Vacío

```python
# src/infrastructure/__init__.py
# (vacío)

# src/infrastructure/storage/__init__.py
# (vacío)
```

**Impacto:**
- Consumidores no saben qué importar
- IDE sin autocompletion
- Violación: Package encapsulation

**Solución:**
```python
# __init__.py
from src.infrastructure.storage.hdf5_exporter import HDF5Exporter
from src.infrastructure.storage.quantum_dataloader import QuantumDatasetLoader
from src.infrastructure.ibm_quantum_adapter import IBMQuantumAdapter
from src.infrastructure.neal_annealer_adapter import NealSimulatedAnnealerAdapter

__all__ = [
    "HDF5Exporter",
    "QuantumDatasetLoader",
    "IBMQuantumAdapter",
    "NealSimulatedAnnealerAdapter",
]
```

---

### 10. ❌ MEDIO: Documentation Gap

**Problema:**
- Sin docstrings extensivos
- Sin ejemplos de uso
- Sin tipo hints completos

```python
# Antes:
def save_batch(self, dataset):  # ❌ No documentado

# Después:
def save_batch(self, dataset: List[Dict[str, Any]]) -> str:
    """
    Guarda un lote de eventos sintéticos en archivos HDF5.
    
    Args:
        dataset: Lista de dicts con estructura:
            {
                "strain": np.ndarray [16384],
                "label": str,
                "metadata": {"m1": float, "m2": float, "distance": float, "theory": str}
            }
    
    Returns:
        str: Ruta del directorio creado (ej: 'data/synthetic/20260419-141715/')
    
    Raises:
        StorageException: Si falla la escritura
        ValueError: Si estructura de dataset es inválida
    """
```

---

## Scorecard Pre-Refactor

| Principio DDD | Score | Evidencia |
|---|---|---|
| **Port Implementation** | 40% | 3 de 8 puertos implementados |
| **Return Type Safety** | 50% | Mezcla np.ndarray, str, tuple |
| **Error Handling** | 20% | Sin custom exceptions |
| **Separation of Concerns** | 50% | massive_loader mezcla concerns |
| **Type Hints** | 60% | Parcialmente tipado |
| **Documentation** | 30% | Docstrings incompletos |
| **Module Exports** | 0% | __init__.py vacío |
| **Integration** | 30% | massive_loader desacoplado |

**PROMEDIO ACTUAL:** 40% DDD Compliant

---

## Acciones de Refactor (TODO)

### 1. ✅ Crear IQuantumMLTrainerPort Implementation
   - `qiskit_vqc_trainer.py` (200 LOC)
   - Implements train_vqc(), save_weights(), load_weights()
   - Uses Qiskit internally (hidden from application)

### 2. ✅ Crear IPreprocessingPort Implementation
   - `sklearn_preprocessing_adapter.py` (150 LOC)
   - Wraps sklearn Pipeline (StandardScaler + PCA + MinMax)
   - Implements fit_transform(), transform(), save(), load()

### 3. ✅ Crear IMetricsReporterPort Implementation
   - `matplotlib_metrics_reporter.py` (120 LOC)
   - Implements report_confusion_matrix(), report_inference_trace()
   - Uses matplotlib, seaborn internally

### 4. ✅ Mover SSTG a Infrastructure
   - Create `src/infrastructure/sstg/` (directory)
   - Move `src/domain/astrophysics/sstg/` → here
   - Implement ISyntheticDataGeneratorPort adapter

### 5. ✅ Refactor Adapters (Existing)
   - Add type hints (HDF5Exporter, quantum_dataloader)
   - Add error handling (custom exceptions)
   - Add comprehensive docstrings

### 6. ✅ Create Infrastructure Exceptions
   - `exceptions.py` (5 custom exception classes)

### 7. ✅ Update __init__.py Files
   - `src/infrastructure/__init__.py` (exports)
   - `src/infrastructure/storage/__init__.py` (exports)

### 8. ✅ Create Integration Tests
   - Verify each adapter satisfies its port contract

---

## Post-Refactor Scorecard Target

| Principio DDD | Target | Cómo |
|---|---|---|
| **Port Implementation** | 100% | All 8 ports implemented |
| **Return Type Safety** | 95% | All returns typed or VOs |
| **Error Handling** | 90% | Custom exceptions throughout |
| **Separation of Concerns** | 95% | Each adapter = one concern |
| **Type Hints** | 100% | 100% type coverage |
| **Documentation** | 95% | Full docstrings everywhere |
| **Module Exports** | 95% | Clear __all__ definitions |
| **Integration** | 95% | All ports integrated |

**TARGET POST-REFACTOR:** 95% DDD Compliant ✨

---

## Architecture After Refactoring

```
CLEAN HEXAGONAL BOUNDARY:

                    Domain Layer (Pure Business Logic)
                    ├── AstrophysicalCalculus
                    ├── NoHairTheoremAnalyzer
                    └── QuantumEventInferenceService
                           ↑
                    (depends on)
                           │
        Application Layer (Use Cases & Orchestration)
        ├── DecodeGravitationalWaveUseCase
        ├── ModelTrainingUseCase
        ├── ModelValidationUseCase
        └── SyntheticDataGenerationUseCase
                           ↑
                    (depends on)
                           │
        Port Interfaces (Abstract Contracts)
        ├── IDataLoaderPort (loads LIGO/synthetic)
        ├── IStoragePort (HDF5 persistence)
        ├── IQuantumMLTrainerPort (Qiskit inside)
        ├── IPreprocessingPort (sklearn inside)
        ├── IMetricsReporterPort (matplotlib inside)
        (+ 3 more ports)
                           ↑
                    (implemented by)
                           │
        Infrastructure Layer (Adapters - Concrete Implementations)
        ├── QuantumDatasetLoader (implements IDataLoaderPort)
        ├── HDF5Exporter (implements IStoragePort)
        ├── QiskitVQCTrainer (implements IQuantumMLTrainerPort) ✨ NEW
        ├── SklearnPreprocessor (implements IPreprocessingPort) ✨ NEW
        ├── MatplotlibMetricsReporter (implements IMetricsReporterPort) ✨ NEW
        ├── IBMQuantumAdapter (implements domain IGateBasedQuantumComputer)
        ├── NealSimulatedAnnealerAdapter (implements domain IQuantumAnnealer)
        └── SSTGAdapter (implements ISyntheticDataGeneratorPort) ✨ NEW

BOUNDARY RULE:
- Inside (domain + application + ports) = Framework agnostic
- Outside (infrastructure) = Framework specific (Qiskit, sklearn, matplotlib)
- Ports = Language agnostic contracts
```

---

## Summary: 8 Files to Create/Modify

### CREATE (NEW):
1. `src/infrastructure/qiskit_vqc_trainer.py` (200 LOC)
   - QiskitVQCTrainer implements IQuantumMLTrainerPort

2. `src/infrastructure/sklearn_preprocessing_adapter.py` (150 LOC)
   - SklearnPreprocessor implements IPreprocessingPort

3. `src/infrastructure/matplotlib_metrics_reporter.py` (120 LOC)
   - MatplotlibMetricsReporter implements IMetricsReporterPort

4. `src/infrastructure/sstg_adapter.py` (100 LOC)
   - SSTGAdapter implements ISyntheticDataGeneratorPort

5. `src/infrastructure/exceptions.py` (50 LOC)
   - Custom exceptions

6. `src/infrastructure/sstg/` (MOVE from domain)
   - Move entire src/domain/astrophysics/sstg/ → here

### MODIFY (REFACTOR):
7. `src/infrastructure/__init__.py` (50 LOC)
   - Export all adapters + exceptions

8. `src/infrastructure/storage/__init__.py` (30 LOC)
   - Export storage adapters

### ALSO MODIFY (Type hints + docs):
- `src/infrastructure/storage/hdf5_exporter.py`
- `src/infrastructure/storage/quantum_dataloader.py`
- `src/infrastructure/ibm_quantum_adapter.py`
- `src/infrastructure/neal_annealer_adapter.py`

---

## Why This Matters (Clean Architecture Enforced)

**BEFORE (Broken Hexagon):**
```
Application imports sklearn, Qiskit, matplotlib 😱
  ↓
Infrastructure half-implemented, half-missing 😱
  ↓
MassiveDatasetLoader standalone, not integrated 😱
  ↓
No error handling, no contracts 😱
```

**AFTER (Clean Hexagon):**
```
Application: Pure business logic, zero framework knowledge ✅
  ↑ (depends on)
  │
Ports: Clear contracts (abstract interfaces) ✅
  ↑ (implemented by)
  │
Infrastructure: All framework code hidden here ✅
  - Qiskit inside QiskitVQCTrainer
  - sklearn inside SklearnPreprocessor
  - matplotlib inside MatplotlibMetricsReporter
  ↓
If Qiskit version changes: Update ONE adapter ✅
If need TensorFlow instead: Create TFVQCTrainer adapter ✅
```

---

## Estimated Effort

| Task | Hours | Notes |
|---|---|---|
| Create IQuantumMLTrainerPort impl | 1.5 | VQC training wrapper |
| Create IPreprocessingPort impl | 1 | sklearn wrapper |
| Create IMetricsReporterPort impl | 1 | matplotlib wrapper |
| Move SSTG + create adapter | 1.5 | Relocation + integration |
| Create exceptions | 0.5 | 5 exception classes |
| Refactor existing adapters | 1 | Type hints + docs |
| Update __init__.py files | 0.5 | Exports + documentation |
| Integration testing | 2 | Verify port contracts |
| **Total** | **9 hours** | Parallel work possible |

---

**STATUS: Ready for Refactoring ✨**

Next step: Implement all 4 missing adapters + refactor existing
