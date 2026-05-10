# 📊 VISUAL REFERENCE: CAPAS vs PAQUETES PYTHON

## EL GRAN DIAGRAMA

```
┌────────────────────────────────────────────────────────────────────┐
│                       SCRIPTS (ENTRADA DEL USUARIO)               │
│  phase1_diagnostics.py | phase3_train.py | phase4_inference.py   │
│                                                                    │
│  Importan: application/ + infrastructure/                         │
│  Usa paquetes: h5py, sklearn, numpy                              │
└────────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────────┐
│              APPLICATION LAYER (CASOS DE USO)                     │
│  hybrid_orchestrator.py | process_event_use_case.py              │
│                                                                    │
│  Define PUERTOS (interfaces abstractas) que NO toca paquetes     │
│  Importa: domain/ + puertos                                       │
│  NO importa: h5py, sklearn, qiskit, etc (lo hace infrastructure) │
└────────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────────┐
│           INFRASTRUCTURE LAYER (ADAPTADORES CONCRETOS)            │
│  storage/ | quantum_adapters/ | ml_adapters/ | reporting/       │
│                                                                    │
│  AQUÍ SÍ importan todos los paquetes externos:                   │
│  ✓ h5py (HDF5)           ✓ sklearn (ML clásico)                 │
│  ✓ qiskit (IBM Quantum)  ✓ dwave (D-Wave)                       │
│  ✓ joblib (serialización)  ✓ matplotlib (plots)                 │
│                                                                    │
│  Implementan los puertos de application/                          │
└────────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────────┐
│         DOMAIN LAYER (LÓGICA PURA SIN DEPENDENCIAS)              │
│  astrophysics/ | quantum/ | metrology/                           │
│                                                                    │
│  ZERO paquetes externos importados directamente                  │
│  Solo: dataclasses, enum, typing, numpy (tipos, no ejecutable)  │
│                                                                    │
│  Entidades, Value Objects, Servicios de dominio                  │
└────────────────────────────────────────────────────────────────────┘
```

---

## 🔍 ¿QUÉ IMPORTA CADA CARPETA?

### ✅ SÍ PUEDE IMPORTAR

```python
# scripts/phase3_train.py
import h5py                                    ← External
from sklearn.ensemble import RandomForestClassifier  ← External
from sklearn.decomposition import PCA          ← External
import joblib                                  ← External

from src.application.model_training_service import ModelTrainingUseCase  ← Internal
from src.infrastructure.storage.hdf5_exporter import HDF5Exporter      ← Internal
from src.infrastructure.sklearn_preprocessing_adapter import SklearnPreprocessor  ← Internal
```

### ❌ NO DEBE IMPORTAR

```python
# src/application/hybrid_orchestrator.py
from src.application.ports import IQuantumOptimizerPort  ← OK (interface)

# ❌ MALO: No debería importar implementaciones concretas
from src.infrastructure.sklearn_preprocessing_adapter import SklearnPreprocessor
# ¿Por qué? Porque rompe la abstracción. Recibe la implementación en __init__

# ✅ BUENO: Recibe interface
def __init__(self, preprocessor: IPreprocessingPort):
    self.preprocessor = preprocessor  # No sabe si es sklearn, pytorch, etc.
```

### ❌ NUNCA DEBERÍA IMPORTAR

```python
# src/domain/astrophysics/entities/gravitational_wave.py

# ❌ NUNCA:
import h5py           # ← External
import sklearn        # ← External
import torch          # ← External
import qiskit         # ← External

# ✅ SOLO:
from dataclasses import dataclass
from typing import Optional
import numpy as np    # OK solo para tipos, no para lógica
```

---

## 📦 TABLA DE PAQUETES POR CAPA

| Paquete | Infrastructure | Application | Domain | Scripts |
|---------|-----------------|-------------|--------|---------|
| `h5py` | ✅ (HDF5Exporter) | ❌ | ❌ | ✅ (directo) |
| `sklearn` | ✅ (SklearnAdapter) | ❌ | ❌ | ✅ (directo) |
| `qiskit` | ✅ (QiskitAdapter) | ❌ | ❌ | ✅ (directo) |
| `dwave` | ✅ (DWaveAdapter) | ❌ | ❌ | ✅ (directo) |
| `joblib` | ✅ (serialización) | ❌ | ❌ | ✅ (directo) |
| `matplotlib` | ✅ (reportes) | ❌ | ❌ | ✅ (directo) |
| `numpy` | ✅ | ⚠️ (DTOs) | ⚠️ (tipos) | ✅ |
| `typing` | ✅ | ✅ | ✅ | ✅ |
| `dataclasses` | ✅ | ✅ | ✅ | ✅ |

**Leyenda**: ✅ OK | ⚠️ Solo para tipos | ❌ No debe

---

## 🎯 EL FLUJO CONCRETO: Fase 3 (Entrenamiento)

```python
# =====================================================
# USUARIO EJECUTA
# =====================================================
$ python scripts/phase3_train.py

# =====================================================
# scripts/phase3_train.py (SCRIPT - puede importar todo)
# =====================================================

import h5py                                          # ← Aquí OK
from sklearn.ensemble import RandomForestClassifier # ← Aquí OK
from sklearn.decomposition import PCA               # ← Aquí OK
from src.infrastructure.storage.hdf5_exporter import HDF5Exporter

# PASO 1: Cargar datos usando adapter
loader = HDF5Exporter()  # Implementa IStoragePort
dataset = loader.load('data/synthetic/synthetic_gw_20260510_012853.h5')

# PASO 2: Extraer arrays de domain entities
strain_plus = dataset.strain_plus        # De GWSignal
theory_labels = dataset.theory_labels    # De TheoryFamily

# PASO 3: Entrenar (sklearn aquí es OK, es un script)
pca = PCA(n_components=64)
X = pca.fit_transform(strain_plus)

classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train)

# PASO 4: Guardar (back to adapter)
exporter = HDF5Exporter()
exporter.save(classifier, 'models/qnim_theory_classifier.pkl')
```

vs

```python
# =====================================================
# application/model_training_service.py (USE CASE)
# =====================================================

# NO PUEDE IMPORTAR paquetes externos directamente

from src.application.ports import (
    IPreprocessingPort,      # Interface
    IQuantumMLTrainerPort,   # Interface
    IStoragePort             # Interface
)
from src.application.dto import TrainingMetrics
from src.domain.astrophysics.entities import GWSignal

class ModelTrainingUseCase:
    def __init__(self,
                 preprocessor: IPreprocessingPort,      # ← Interface, no impl
                 trainer: IQuantumMLTrainerPort,        # ← Interface, no impl
                 storage: IStoragePort):                # ← Interface, no impl
        self.preprocessor = preprocessor
        self.trainer = trainer
        self.storage = storage
    
    def execute(self, signal: GWSignal) -> TrainingMetrics:
        # Usa interfaces, no sabe si backend es sklearn, pytorch, etc.
        X = self.preprocessor.transform(signal)
        metrics = self.trainer.train(X, signal.theory)
        self.storage.save(metrics)
        return metrics
        
        # ¿Cómo se usa? En un script:
        # ├─ preprocessor = SklearnPreprocessor()  (implementa IPreprocessingPort)
        # ├─ trainer = QiskitVQCTrainer()           (implementa IQuantumMLTrainerPort)
        # ├─ storage = HDF5Exporter()               (implementa IStoragePort)
        # └─ use_case = ModelTrainingUseCase(preprocessor, trainer, storage)
```

---

## 🔄 LA MAGIA DE LOS PUERTOS

### Sin Puertos (❌ ACOPLAMIENTO)

```python
# application/orchestrator.py
from sklearn.preprocessing import StandardScaler       # ← Acoplado a sklearn
from src.infrastructure.sklearn_preprocessing_adapter import SklearnPreprocessor

class Orchestrator:
    def process(self, data):
        scaler = StandardScaler()  # ← Cambiarlo a pytorch=reescribir application
        scaled = scaler.fit_transform(data)
        return scaled
```

Si quieres cambiar a PyTorch preprocessing, ¡tienes que editar application! ❌

### Con Puertos (✅ DESACOPLAMIENTO)

```python
# application/ports.py
class IPreprocessingPort:
    def transform(self, data):
        pass

# application/orchestrator.py
class Orchestrator:
    def __init__(self, preprocessor: IPreprocessingPort):  # ← Inyección
        self.preprocessor = preprocessor
    
    def process(self, data):
        scaled = self.preprocessor.transform(data)  # ← No sabe SI es sklearn, pytorch
        return scaled

# infrastructure/sklearn_preprocessing_adapter.py
class SklearnPreprocessor(IPreprocessingPort):
    def transform(self, data):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        return scaler.fit_transform(data)

# infrastructure/pytorch_preprocessing_adapter.py
class PyTorchPreprocessor(IPreprocessingPort):
    def transform(self, data):
        import torch
        # Implementación con torch
        return processed_data

# scripts/phase3_train.py
# Solo cambias esta línea:
preprocessor = PyTorchPreprocessor()  # Antes: SklearnPreprocessor()
use_case = ModelTrainingUseCase(preprocessor, ...)
# ✓ No modificaste application/
```

---

## 🏗️ CAPAS EN LA PRÁCTICA

### DOMAIN: Puro Razonamiento

```python
# src/domain/astrophysics/services/wave_decoder.py
# NO TIENE NINGUN IMPORT EXTERNO

class WaveDecoder:
    """Lógica pura de decodificación física"""
    
    def extract_chirp_mass(self, h_plus, h_cross):
        """Calcula chirp mass de las polarizaciones"""
        # Solo matemática
        amplitude = (h_plus**2 + h_cross**2)**0.5
        frequency_derivative = # ... cálculo matemático ...
        m_chirp = # ... fórmula física ...
        return m_chirp
    
    def is_consistent_with_gr(self, parameters):
        """Verifica si es consistente con GR"""
        # Solo lógica de dominio
        expected_ringdown = # ... fórmula ...
        measured_ringdown = parameters.ringdown_frequency
        return abs(expected - measured) < TOLERANCE
```

**Testeable sin dependencias**:
```python
def test_extract_chirp_mass():
    decoder = WaveDecoder()  # ✓ No necesita h5py, sklearn, nada
    mass = decoder.extract_chirp_mass([...], [...])
    assert mass > 0
```

### APPLICATION: Orquestación

```python
# src/application/hybrid_orchestrator.py
# NO IMPORTA LIBRARIAS EXTERNAS, SOLO PUERTOS

class HybridInferenceOrchestrator:
    def __init__(self,
                 d_wave: IQuantumOptimizerPort,      # ← Puerto
                 ibm: IQuantumMLTrainerPort,         # ← Puerto
                 storage: IStoragePort):             # ← Puerto
        self.d_wave = d_wave
        self.ibm = ibm
        self.storage = storage
    
    def execute_inference(self, signal: GWSignal) -> InferenceResult:
        """Orquesta ambas ramas cuánticas"""
        # Rama D-Wave
        d_wave_result = self.d_wave.optimize(signal)
        
        # Rama IBM
        ibm_result = self.ibm.classify(signal)
        
        # Agregación
        final_result = self._aggregate(d_wave_result, ibm_result)
        
        # Guardado
        self.storage.save(final_result)
        
        return final_result
```

**¿Por qué no tiene h5py aquí?** Porque la orquestación no debería saber cómo se guardan los datos. Lo delega al puerto `IStoragePort`.

### INFRASTRUCTURE: Implementaciones Concretas

```python
# src/infrastructure/sklearn_preprocessing_adapter.py
# AQUI SÍ IMPORTA sklearn, numpy, etc.

from sklearn.preprocessing import StandardScaler
import numpy as np

class SklearnPreprocessor(IPreprocessingPort):
    def __init__(self):
        self.scaler = StandardScaler()
    
    def transform(self, data):
        """Implementación concreta con sklearn"""
        return self.scaler.fit_transform(data)
```

**¿Por qué aquí?** Porque el detalles de cómo se escala es un detalle de IMPLEMENTACIÓN.

---

## 🔍 DEBUGGING: ¿Dónde está mi paquete?

**Pregunta**: "¿Por qué no puedo hacer `import sklearn` en application/?"

**Respuesta**: Porque rompe la abstracción. Si lo haces:

```python
# ❌ MALO: application/orchestrator.py
import sklearn

class Orchestrator:
    def process(self, data):
        scaler = sklearn.preprocessing.StandardScaler()
        return scaler.fit_transform(data)
```

Entonces:
1. Si quieres cambiar de sklearn a pytorch, ¡debes reescribir application/!
2. Application se vuelve acoplada a sklearn
3. No puedes testear sin sklearn instalado
4. Es más difícil de mantener

**La solución**: Usa puertos

```python
# ✅ BUENO: application/orchestrator.py
class Orchestrator:
    def __init__(self, preprocessor: IPreprocessingPort):
        self.preprocessor = preprocessor
    
    def process(self, data):
        return self.preprocessor.transform(data)

# infrastructure/sklearn_preprocessing_adapter.py
import sklearn  # ← OK, está en infrastructure

class SklearnPreprocessor(IPreprocessingPort):
    def transform(self, data):
        return sklearn.preprocessing.StandardScaler().fit_transform(data)
```

---

## 📋 CHECKLIST: ¿Estoy en la capa correcta?

### Estoy escribiendo en `domain/`?

```
❌ Hago "import sklearn"?
❌ Hago "import h5py"?
❌ Hago "import qiskit"?
❌ Uso detalles de base de datos?
❌ Tengo que saber qué versión de librería se usa?

✅ Solo matemática?
✅ Solo dataclasses?
✅ Solo typing?
```

### Estoy escribiendo en `application/`?

```
✅ Defino casos de uso?
✅ Uso puertos (IPreprocessingPort)?
✅ Combino múltiples servicios?
✅ Retorno DTOs?

❌ Importo sklearn?
❌ Importo h5py?
❌ Importo qiskit?
❌ Callo directamente a librerías?
```

### Estoy escribiendo en `infrastructure/`?

```
✅ Implemento un puerto?
✅ Importo sklearn, h5py, qiskit?
✅ Hago la "traducción" entre dominios?
✅ Manejo excepciones específicas de librería?

❌ Tengo lógica de negocio?
❌ Dependo de un caso de uso específico?
❌ Defino nuevos puertos aquí?
```

---

## 🎯 RESUMEN FINAL

| Aspecto | Domain | Application | Infrastructure | Scripts |
|---------|--------|-------------|-----------------|---------|
| **Propósito** | Lógica pura | Casos de uso | Detalles técnicos | Entry points |
| **Paquetes externos** | ❌ NO | ❌ NO | ✅ SÍ | ✅ SÍ |
| **Complejidad** | Alta (física) | Media (orquestación) | Baja (traducción) | Baja (llamadas) |
| **Testeable sin mocks** | ✅ SÍ | ⚠️ Con puertos | ❌ NO | ❌ NO |
| **Cambiable** | 🔒 No (física) | ✅ Fácil (puertos) | ✅ Fácil (swappable) | ✅ Fácil |
| **Quién lo usa** | Application | Scripts | Application | Usuarios |

---

**Versión**: 1.0  
**Fecha**: 10 de Mayo de 2026  
**Autor**: Clean Architecture Analysis
