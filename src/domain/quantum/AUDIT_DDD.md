# AUDITORÍA DDD: domain/quantum

**Fecha**: 19 Apr 2026  
**Revisor**: DDD Architecture Auditor  
**Estado**: ⚠️ CRÍTICO - Múltiples violaciones DDD encontradas

---

## 📋 ANÁLISIS POR ARCHIVO

### 1. `vqc_architecture.py` - ⚠️ DUPLICACIÓN + INCONSISTENCIA

**Problema 1: VQCTopology duplicada**
```python
# EN ESTE ARCHIVO:
@dataclass
class VQCTopology:
    num_qubits: int
    num_features: int
    ansatz_reps: int
    ...
    def get_parameter_count(self) -> int:
        ...

# TAMBIÉN EN entities.py:
@dataclass
class VQCTopology:
    num_qubits: int
    feature_map_reps: int
    ansatz_reps: int
    entanglement_strategy: str
```
- ❌ DOS definiciones diferentes de la MISMA clase
- ❌ Nombres de campos desincronizados (num_features vs feature_map_reps)
- ❌ Una tiene método, otra no
- ❌ Una tiene offset de entanglement, otra no

**Solución DDD**: Consolidar en UNA sola definición en `value_objects.py`

**Problema 2: Cálculo de parámetros sin documentación**
```python
def get_parameter_count(self) -> int:
    # Para RealAmplitudes es num_qubits * (reps + 1)
    return self.num_qubits * (self.ansatz_reps + 1)
```
- ⚠️ Asume "RealAmplitudes" (coupling específico)
- ⚠️ Fórmula puede variar por ansatz type
- ✅ Debería ser parametrizable


### 2. `entities.py` - ✅ PARCIALMENTE OK

**Problema 1: Falta validación**
```python
@dataclass
class QUBOProblem:
    linear_terms: Dict[int, float]
    quadratic_terms: Dict[Tuple[int, int], float]
    offset: float = 0.0
```
- ❌ No frozen=True (mutable)
- ❌ Sin validación __post_init__
- ❌ No verifica que claves sean índices válidos

**Problema 2: AnnealingResult mezcla responsabilidades**
```python
@dataclass
class AnnealingResult:
    best_state: Dict[int, int]
    lowest_energy: float
    num_occurrences: int
    is_ground_state_confident: bool
```
- ⚠️ ¿"num_occurrences" es propiedad de resultado o del sampling?
- ⚠️ ¿"is_ground_state_confident" es computed o input?
- ✅ Debería tener propiedades claramente separadas

**Problema 3: VQCTopology duplicada (CRÍTICO)**
- ❌ Misma clase definida en 2 archivos con campos diferentes


### 3. `interfaces.py` - ⚠️ INCOMPLETO

**Problema 1: Falta de documentación**
```python
class IQuantumAnnealer(ABC):
    """
    Puerto para el Recocido Cuántico (D-Wave).
    ...
    """
    @abstractmethod
    def sample_qubo(self, Q: dict, num_reads: int) -> dict:
        pass
```
- ⚠️ Type hints incompletos (Q: dict debería ser QUBO: QUBOProblem)
- ⚠️ Falta descripción detallada de parámetros
- ⚠️ Falta información de retorno

**Problema 2: Falta métodos**
```python
class IQuantumAnnealer(ABC):
    # ❌ ¿Qué pasa con timing, errores, calibración?
    # ❌ ¿Cómo se configura el embedding?
    # ❌ ¿Cómo se manejan los chain breaks?
```

**Solved por DDD**: Más métodos abstractos documentados


### 4. `qubo_formulator.py` - ✅ BIEN (PERO...)

**OK:**
- ✅ Stateless (@classmethod)
- ✅ Lógica pura (sin I/O)
- ✅ Algoritmo correcto (QUBO con penalizaciones)

**Problemas:**
```python
@classmethod
def build_formulation(cls, target_signal: np.ndarray, templates: List[dict]) -> QUBOProblem:
    """
    ...
    """
    penalty_weight = 1000.0  # ❌ MAGIC NUMBER
```

1. ❌ `penalty_weight = 1000.0`: Magic number sin justificación
   - ⚠️ Debería ser parametrizable
   - ⚠️ O calculado dinámicamente según el problema

2. ❌ `templates: List[dict]`: Type hint vago
   - ✅ Debería ser `List[TemplateSignal]` (value object)

3. ❌ Sin validación de inputs
   - ❌ ¿Qué pasa si target_signal está vacía?
   - ❌ ¿Qué pasa si templates es []?

---

## 🔄 INTEGRACIÓN CON CAPAS ASTROPHYSICS

**Problema**: ¿Donde encaja domain/quantum en las 7 capas?

Hipótesis:
- Capas 1-5 (Signal → BeyondGR): Análisis clásico + inferencia Bayesian
- **Capa 7 (DeepQuantum)**: Aquí entra la circuitería cuántica
  - VQCTopology: Estructura del ansatz
  - QUBO: Optimización cuántica
  - AnnealingResult: Salida de QPU

**Problema DDD**: domain/quantum NO ESTÁ INTEGRADO CON CAPAS
- ❌ No hay referencia desde DeepQuantumManifold
- ❌ No hay value objects compartidos
- ❌ Está aislado (violación de Bounded Context)

---

## ✅ ARQUITECTURA CLEAN RECOMENDADA

```
domain/quantum/
├── value_objects.py          ← NUEVO: Consolidar VQCTopology + QUBO
│   ├── VQCTopology (definitiva)
│   ├── QUBOProblem (frozen)
│   ├── AnnealingResult (frozen)
│   ├── TemplateSignal (VO)
│   └── QuantumCircuitConfig (VO)
│
├── interfaces.py             ← REFACTOR: +docstrings, type hints
│   ├── IQuantumAnnealer
│   └── IGateBasedQuantumComputer
│
├── qubo_services.py          ← RENAME: qubo_formulator → qubo_services
│   ├── TemplateMatchingQUBO (stateless)
│   ├── PenaltyCalculator (factorizado)
│   └── QUBOValidator
│
├── entities.py               ← DEPRECATE (mover a value_objects.py)
├── vqc_architecture.py       ← DEPRECATE (mover a value_objects.py)
└── __init__.py               ← PUBLIC API (20+ exports)
```

---

## 🎯 ACCIONES REQUERIDAS (Prioridad)

### CRÍTICO (hacer ahora)
1. [ ] Consolidar VQCTopology
   - Una definición única: quitar de vqc_architecture.py
   - Colocar en value_objects.py

2. [ ] Crear value_objects.py
   - VQCTopology (definitiva, frozen)
   - QUBOProblem (frozen, validado)
   - AnnealingResult (frozen, propiedades)
   - TemplateSignal (nuevo VO)
   - QuantumCircuitConfig (nuevo VO)

3. [ ] Refactorizar entities.py
   - Mover contenido a value_objects.py
   - Dejar archivo como DEPRECATED

4. [ ] Refactorizar qubo_formulator.py
   - Renombrar a qubo_services.py
   - Parametrizar penalty_weight
   - Usar TemplateSignal VO

5. [ ] Mejorar interfaces.py
   - Docstrings detallados
   - Type hints correctos (QUBOProblem en lugar de dict)
   - Más métodos abstractos (e.g., get_embedding_time)

### IMPORTANTE (antes de merge)
6. [ ] Validación __post_init__ en todos los VOs
7. [ ] Conectar domain/quantum ↔ domain/astrophysics/layers.py
   - Agregar campos a DeepQuantumManifold
8. [ ] Actualizar __init__.py con 20+ exports

### FUTURO
9. [ ] Tests unitarios
10. [ ] Concretizar interfaces (DWaveAdapter, QiskitAdapter)


# ============================================================================
# PUNTUACIÓN DDD ACTUAL
# ============================================================================

| Criterio | Score | Status |
|----------|-------|--------|
| Stateless services | 80% | ⚠️ qubo_formulator OK |
| Value objects | 40% | ❌ Sin frozen, sin validación |
| Type safety | 50% | ⚠️ dict en lugar de tipos |
| Immutability | 20% | ❌ No frozen |
| Separation of concerns | 60% | ⚠️ VQCTopology duplicada |
| **OVERALL** | **50%** | ⚠️ **DEBAJO DEL ESTÁNDAR** |

**Requerimiento**: ≥ 80% para pasar auditoría

---

## 📊 ESFUERZO ESTIMADO

- Consolidar VQCTopology: 1 hora
- Crear value_objects.py: 2 horas
- Refactorizar interfaces + services: 2 horas
- Tests unitarios: 2 horas
- Documentación: 1 hora
- **Total**: ~8 horas (1 día)

---

## CONCLUSIÓN

**Status**: ⚠️ **NO LISTO PARA PRODUCCIÓN**

Los conceptos cuánticos tienen lógica correcta pero violan DDD:
- ❌ VQCTopology duplicada (CRÍTICO)
- ❌ Sin frozen=True (mutable)
- ❌ Sin validación
- ❌ Type hints vagos (dict en lugar de tipos)
- ❌ Magic numbers sin parametrización

**Recomendación**: Refactorizar antes de merge a main.

---

## DIAGRAMA: Dependencias Cuánticas

```
domain/astrophysics/layers.py
    └─ DeepQuantumManifold
        ├─ ads_cft: AdSCFTDuality
        ├─ quantum_corrections: QuantumCorrectionsMetric
        ├─ lorentz_violation: LorentzViolation
        ├─ discovery_significance: QuantumGravitySignificance
        └─ ❌ FALTA: quantum_circuit_topology: VQCTopology ❌
            └─ Debería venir de domain/quantum/value_objects.py


domain/quantum/value_objects.py (NUEVO)
    ├─ VQCTopology: Ansatz configuration (reps, entanglement, etc)
    ├─ QUBOProblem: H = H_cost + H_penalty
    ├─ AnnealingResult: Salida de QPU
    └─ TemplateSignal: Plantilla para matching


domain/quantum/qubo_services.py
    ├─ TemplateMatchingQUBO: Construye QUBO
    ├─ PenaltyCalculator: Calcula penalizaciones
    └─ QUBOValidator: Valida matrix


domain/quantum/interfaces.py
    ├─ IQuantumAnnealer: D-Wave interface
    └─ IGateBasedQuantumComputer: IBM/Qiskit interface
        (Implementadas en infrastructure/)
```

