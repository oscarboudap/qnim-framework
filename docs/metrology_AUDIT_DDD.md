# AUDITORÍA DDD: domain/metrology

**Fecha**: 19 Apr 2026  
**Revisor**: DDD Architecture Auditor  
**Estado**: ⚠️ CRÍTICO - Múltiples violaciones DDD encontradas

---

## 📋 ANÁLISIS POR ARCHIVO

### 1. `fisher_matrix_calculator.py` - ❌ VIOLACIÓN CRÍTICA

**Problema 1: Stateful (NO ES DOMINIO PURO)**
```python
def __init__(self, psd_array: np.ndarray, delta_f: float):
    self.psd = psd_array  # ← ESTADO MUTABLE
    self.delta_f = delta_f  # ← ESTADO MUTABLE
```
- ❌ Servicios de dominio deben ser stateless
- ❌ Cada instancia mantiene datos
- ❌ No es reutilizable en múltiples contextos
- ❌ Difícil de testear (setup dependiente)

**Problema 2: Falta de value objects**
```python
# Debería recibir:
class PSD(ValueObject):
    frequency_bins: np.ndarray
    delta_f: float
    
# En lugar de:
def __init__(self, psd_array: np.ndarray, delta_f: float)
```

**Solución DDD**:
```python
class FisherMatrixCalculator:  # Stateless domain service
    @staticmethod
    def compute_inner_product(h1_f: np.ndarray, h2_f: np.ndarray, 
                             psd: np.ndarray, delta_f: float) -> float:
        """Pure function: no state, deterministic."""
    
    @staticmethod
    def build_fisher_matrix(h_f, derivatives, psd, delta_f):
        """Pure function: no state."""
```

---

### 2. `hubble_metrology.py` - ✅ PARCIALMENTE OK

**Problema**: Mezcla de responsabilidades
```python
class HubbleMetrology:
    @classmethod
    def infer_hubble_constant(cls, luminosity_distance_mpc: float, 
                             redshift_z: float) -> float:
        """Este es un servicio de dominio que debería ser stateless"""
```

**Aspectos OK**:
- ✅ No state (`@classmethod`)
- ✅ Pure computation
- ✅ Lógica clara

**Problema**: Debería ser un value object OR documentar como domain service
```python
# OPCIÓN 1: Value Object (si representa un concepto)
@dataclass(frozen=True)
class HubbleConstant(ValueObject):
    value_km_s_mpc: float
    
    @staticmethod
    def from_gravitational_wave(luminosity_distance_mpc: float, 
                               redshift_z: float) -> 'HubbleConstant':
        """Factory method"""

# OPCIÓN 2: Domain Service stateless (si es solo cálculo)
class HubbleCosmologyCalculator:
    @staticmethod
    def infer_hubble_constant(...):
        pass
```

---

### 3. `multipole_validator.py` - ⚠️ PARCIAL

**Problema 1: Stateful validator**
```python
def __init__(self, tolerance_threshold: float = 0.05):
    self.tolerance_threshold = tolerance_threshold  # ← ESTADO
```
- ❌ Should be injected as ValueObject
- ❌ Or passed as parameter to method

**Problema 2: Mezcla conceptos**
```python
def evaluate_no_hair_theorem(self, classical_mass, classical_spin, 
                            quantum_anomaly_confidence):
    is_violated = delta_q_deviation > self.tolerance_threshold
    inferred_theory = TheoryFamily.LOOP_QUANTUM_GRAVITY if is_violated...
```
- ❌ "validator" (boolean) + "inferrer" (theory selection) mezclados
- ✅ Debería ser: Validator → AnalysisResult (value object)

**Solución DDD**:
```python
@dataclass(frozen=True)
class NoHairViolationResult(ValueObject):
    expected_kerr_q: float
    measured_delta_q: float
    is_violated: bool
    inferred_theory: TheoryFamily
    violation_magnitude: float

class NoHairTheoremValidator:
    @staticmethod
    def evaluate(classical_mass: float, classical_spin: float, 
                quantum_anomaly_confidence: float,
                tolerance_threshold: float) -> NoHairViolationResult:
        """Pure function returning rich value object"""
```

---

### 4. `planck_error_bounds.py` - ⚠️ PARCIAL

**OK**:
- ✅ Stateless (`@classmethod`)
- ✅ Constants properly defined
- ✅ Pure computation

**Problema 1**: Return dict (should be value object)
```python
# ❌ Retorna dict (no tipado)
return {
    "sigma": round(sigma_value, 2),
    "p_value": p_value,
    "is_discovery": is_discovery,
    "conclusion": "..."
}

# ✅ Debería ser:
@dataclass(frozen=True)
class DiscoverySignificance(ValueObject):
    sigma: float
    p_value: float
    is_discovery: bool
    conclusion: str
```

**Problema 2**: Nombre genérico
- ❌ "PlanckErrorBounds" sugiere bounds, pero calcula significance
- ✅ Debería ser: "QuantumGravitySignificanceCalculator"

---

## 🔄 INTEGRACIÓN CON CAPAS EXISTENTES

### Capa 4 (CosmologicalEvolution) - HubbleMetrology
✅ OK: Calcula H0 desde luminosity distance + redshift
⚠️ TO-DO: Conectar con Capa 4 value objects

### Capa 6 (HorizonQuantumTopology) - MultipoleValidator
⚠️ PROBLEMA: No existe NoHairTheorem en layers.py
- Debería agregar a `HorizonQuantumTopology`:
```python
@dataclass
class NoHairViolationAnalysis:
    is_kerr_violated: bool
    deviation_magnitude: float
    inferred_beyond_gr_theory: Optional[TheoryFamily]
```

### Capa 7 (DeepQuantumManifold) - Planck bounds
⚠️ PROBLEMA: No existe LorentzViolation en layers.py
- Debería agregar a `DeepQuantumManifold`:
```python
@dataclass
class QuantumDiscoverySignificance:
    sigma_value: float
    p_value: float
    is_discovery: bool
```

---

## ✅ ARQUITECTURA CLEAN RECOMENDADA

```
domain/metrology/
├── value_objects.py          ← NUEVO: Rich types (PSD, HubbleConstant, etc)
├── metrology_services.py     ← NUEVO: Stateless calculators
│   ├── FisherMatrixCalculator (static methods)
│   ├── HubbleCosmologyCalculator (static methods)
│   ├── NoHairTheoremAnalyzer (static methods)
│   └── QuantumGravitySignificanceCalculator (static methods)
├── entities.py               ← Metrology domain events/facts
├── repositories.py           ← Persistence abstractions
├── __init__.py               ← Public API (60+ exports)
└── validators.py             ← DEPRECATE fisher_matrix_calculator.py
```

---

## 🎯 ACCIONES REQUERIDAS (Prioridad)

### CRÍTICO (hacer ahora)
1. [ ] Crear `value_objects.py` con tipos ricos
   - `PSD(frequency_bins, delta_f)` frozen
   - `HubbleConstant(value_km_s_mpc)` frozen
   - `NoHairViolationResult(...)` frozen
   - `DiscoverySignificance(...)` frozen

2. [ ] Refactorizar `fisher_matrix_calculator.py` → stateless
   - Mover `__init__` parameters a method arguments
   - Usar `@staticmethod` en lugar de instance methods

3. [ ] Refactorizar `hubble_metrology.py`
   - Cambiar nombre a `HubbleCosmologyCalculator`
   - Documentar como domain service

4. [ ] Refactorizar `multipole_validator.py`
   - Stateless: `tolerance_threshold` como parámetro
   - Retornar `NoHairViolationResult` en lugar de dict

5. [ ] Refactorizar `planck_error_bounds.py`
   - Cambiar nombre a `QuantumGravitySignificanceCalculator`
   - Retornar `DiscoverySignificance` value object

### IMPORTANTE (antes de merge)
6. [ ] Actualizar `layers.py`:
   - Agregar NoHairViolationAnalysis a `HorizonQuantumTopology`
   - Agregar QuantumDiscoverySignificance a `DeepQuantumManifold`

7. [ ] Conectar servicios con capas
   - Metrology services usan layers.py value objects

8. [ ] Actualizar `__init__.py`
   - Exportar 40+ value objects + services

9. [ ] Tests unitarios
   - Test cada servicio con fixtures (PSD, masses, etc)

### FUTURO
10. [ ] Repositorio: MeasurementAnalysisRepository
11. [ ] Use cases de aplicación
12. [ ] Integración con SSTG para synthetic data

---

## 📊 PUNTUACIÓN DDD ACTUAL

| Criterio | Score | Status |
|----------|-------|--------|
| Stateless services | 50% | ⚠️ Fisher es stateful |
| Value objects | 20% | ❌ Retorna dicts |
| Type safety | 60% | ⚠️ Mixed (class + dict) |
| Immutability | 80% | ✅ Los que existen son OK |
| Separation of concerns | 40% | ⚠️ Mezcla responsabilidades |
| **OVERALL** | **50%** | ⚠️ **DEBAJO DEL ESTÁNDAR** |

**Requerimiento**: ≥ 80% para pasar auditoría

---

## 🚀 ESFUERZO ESTIMADO

- Value objects: 2 horas
- Refactorizar 4 archivos: 3 horas
- Tests unitarios: 2 horas
- Documentación: 1 hora
- **Total**: ~8 horas (1 día de trabajo)

---

## CONCLUSIÓN

**Status**: ⚠️ **NO LISTO PARA PRODUCCIÓN**

Los validadores/calculadores tienen lógica correcta pero violan principios DDD:
- ❌ No son completamente stateless
- ❌ Retornan tipos primitivos (dict, float) en lugar de value objects
- ❌ Necesitan inyección de dependencias

**Recomendación**: Refactorizar antes de merge a main.

