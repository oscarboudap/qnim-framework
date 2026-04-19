"""
ARQUITECTURA DOMAIN-DRIVEN DESIGN (DDD)

Guía de estructura y principios aplicados a la capa de dominio de astrofísica.
"""

# ============================================================================
# PRINCIPIOS DDD APLICADOS
# ============================================================================

## 1. LIMITACIÓN CLARA DE DOMINIO (Bounded Context)

   DOMINIO = "Astrofísica de Ondas Gravitacionales"
   
   INCLUYE:
   ✅ Física pura de GW (cálculos, constrains, validaciones)
   ✅ Estructuras de datos (Value Objects, Entities)
   ✅ Relaciones entre conceptos (Aggregate Root)
   ✅ Reglas de negocio (Energy Conditions, Cosmic Censorship)
   
   EXCLUYE (Infraestructura):
   ❌ Síntesis de datos (SSTG generators)
   ❌ Persistencia de bases datos
   ❌ APIs, REST, presentación
   ❌ Frameworks de I/O
   ❌ Detalles de computación (threading, etc)


## 2. LENGUAJE UBICUO (Ubiquitous Language)

   Las clases y métodos usan VOCABULARIO ASTROFÍSICO EXACTO:
   
   ✅ `QuantumDecodedEvent` - no "DataPoint" o "Sample"
   ✅ `SignalMathematicalStructure` - no "SignalData"
   ✅ `IntrinsicGeometry` - no "BinaryParams"
   ✅ `CosmologicalEvolution` - no "DistanceData"
   ✅ `BeyondGRSignatures` - no "AnomalyFlags"
   ✅ `HorizonQuantumTopology` - no "RingdownAnalysis"
   
   Cada término corresponde a UN CONCEPTO FÍSICO ESPECÍFICO.
   No hay ambigüedad entre especialistas de GW.


## 3. VALUE OBJECTS (Inmutables, Validados)

   Ubicación: value_objects.py
   
   Características:
   - frozen=True: Inmutables una vez creados
   - Validación post-init: Rangos físicos forzados
   - Métodos de utilidad: relative_error(), signal_to_noise()
   - Encapsulan incertidumbre: value + sigma
   
   Ejemplos:
   
   ```python
   class SolarMass(Measurement):
       def __post_init__(self):
           super().__post_init__()
           if self.value <= 0:
               raise ValueError("Masa estelar no física")
   
   class Spin(Measurement):
       def __post_init__(self):
           super().__post_init__()
           if not (-1e-6 <= self.value <= 1 + 1e-6):
               raise ValueError(f"Spin fuera de rango")
   
   class Redshift(Measurement):
       def __post_init__(self):
           super().__post_init__()
           if self.value < -1 + 1e-6:
               raise ValueError("Redshift no-físico")
   ```
   
   PRINCIPIO: "Es imposible crear un Value Object inválido."


## 4. CAPAS FÍSICAS (Layers)

   Ubicación: layers.py
   
   Organización: 7 capas anidadas de información
   
   ```
   Capa 1: SignalMathematicalStructure   (15 dims)
           ↑ observables matemáticos puros
   
   Capa 2: IntrinsicGeometry            (10 dims)
           ↑ parámetros del sistema binario
   
   Capa 3: AstrophysicalEnvironment      (6 dims)
           ↑ contexto: disco, B-field, axiones
   
   Capa 4: CosmologicalEvolution        (8 dims)
           ↑ propagación cosmológica
   
   Capa 5: BeyondGRSignatures           (40+ dims)
           ↑ nueva física (5 categorías teóricas)
   
   Capa 6: HorizonQuantumTopology       (8 dims)
           ↑ estructura del horizonte
   
   Capa 7: DeepQuantumManifold          (6 dims)
           ↑ física cuántica profunda
   ```
   
   Cada capa es:
   - Independiente conceptualmente
   - Pero correlacionada a través de evento raíz
   - Puede estar ausente (Optional en entity)
   - Tiene observables con incertidumbre


## 5. ENTIDADES Y AGGREGATES

   Ubicación: entities.py
   
   Raíz del Agregado:
   
   ```python
   @dataclass
   class QuantumDecodedEvent:
       """ENTIDAD RAÍZ (Aggregate Root)"""
       event_id: str                    # Identidad única
       detector_network: Set[str]       # Contexto
       snr_total: SignalToNoise         # Calidad medición
       
       # Las 7 capas
       signal_math: SignalMathematicalStructure
       geometry: IntrinsicGeometry
       environment: Optional[AstrophysicalEnvironment]
       cosmology: Optional[CosmologicalEvolution]
       beyond_gr: Optional[BeyondGRSignatures]
       horizon_topology: Optional[HorizonQuantumTopology]
       deep_quantum: Optional[DeepQuantumManifold]
   ```
   
   Invariantes:
   - event_id único (no duplicados)
   - Todas las capas se infieren COHERENTEMENTE
   - Si una capa cambia, se crea NUEVO evento (inmutabilidad)
   - frozen=True asegura inmodificabilidad


## 6. ESPECIFICACIONES (Specification Pattern)

   Ubicación: repositories.py
   
   Para consultas type-safe sin SQL strings:
   
   ```python
   # DDD: Consultas como objetos
   spec = CompositeSpecification([
       MinimumSNRSpecification(min_snr=20),
       BeyondGRConfidenceSpecification(min_confidence=0.8),
       TheoryFamilySpecification({
           TheoryFamily.BRANS_DICKE,
           TheoryFamily.ECO_GRAVASTAR
       })
   ], logic="AND")
   
   events = repository.find_all_matching(spec)
   ```
   
   Ventajas:
   - Type-safe
   - Reutilizable
   - Agnóstico a base datos
   - Testeable


## 7. INTERFACES DE DOMINIO (No implementaciones)

   Ubicación: entities.py (como ABC)
   
   ```python
   class BayesianEvidenceCalculator(ABC):
       @abstractmethod
       def compute_log_evidence(
           self, event: QuantumDecodedEvent, theory: TheoryFamily
       ) -> Tuple[float, float]:
           pass
   
   class TheoryDiscriminator(ABC):
       @abstractmethod
       def bayes_factor(
           self, event: QuantumDecodedEvent, 
           theory_a: TheoryFamily, theory_b: TheoryFamily
       ) -> float:
           pass
   
   class LayerAnalyzer(ABC):
       @abstractmethod
       def extract_observables(self, event: QuantumDecodedEvent) -> Dict:
           pass
   ```
   
   PRINCIPIO: "El dominio define Qkirk se necesita, la infraestructura HOW se implementa."


## 8. SERVICIOS DE DOMINIO (Lógica reutilizable)

   Ubicación: domain_services.py
   
   Características:
   - Stateless (sin estado mutable)
   - Reutilizables en distintos contextos
   - Lógica pura (sin I/O)
   - Operan sobre Value Objects y Entities
   
   Ejemplos:
   
   ```python
   class AstrophysicalCalculus:
       @staticmethod
       def chirp_mass(m1: float, m2: float) -> float:
           """ℳ_c = (m₁m₂)^(3/5) / (m₁+m₂)^(1/5)"""
           return ((m1 * m2) ** 0.6) / ((m1 + m2) ** 0.2)
       
       @staticmethod
       def echo_delay_quantum_gravity(m_total_solar: float) -> float:
           """Δt ~ 2 r_* ln(M / ℓ_Planck)"""
           ...
   
   class PhysicalConstraintValidator:
       @staticmethod
       def check_energy_conditions(m1, m2, distance) -> Tuple[bool, str]:
           """Valida restricciones de GR"""
           ...
       
       @staticmethod
       def validate_cosmic_censorship(m_total, spin) -> Tuple[bool, str]:
           """Censura cósmica: a ≤ M"""
           ...
   ```
   
   PRINCIPIO: "Evita lógica en entidades; extrae a servicios reutilizables."


## 9. FACTORIES (Construcción colaborada)

   Ubicación: entities.py
   
   ```python
   class QuantumDecodedEventFactory:
       @staticmethod
       def create_from_raw_strain(
           event_id: str,
           detector_network: Set[str],
           snr_total: SignalToNoise,
           # ... data raw
       ) -> QuantumDecodedEvent:
           """Factory garantiza que evento es válido (invariantes)"""
           # 1. Valida inputs
           # 2. Construye capas coherentemente
           # 3. Retorna evento garantizado válido
           pass
   ```
   
   PRINCIPIO: "No construyas AGGREGATES directamente; usa factories."


## 10. REPOSITORIOS (Interfaces abstractas)

   Ubicación: repositories.py
   
   ```python
   class GravitationalWaveRepository(ABC):
       @abstractmethod
       def add(self, event: QuantumDecodedEvent) -> None:
           pass
       
       @abstractmethod
       def find_by_id(self, event_id: str) -> Optional[QuantumDecodedEvent]:
           pass
       
       @abstractmethod
       def find_all_matching(self, spec: GWEventSpecification) -> List:
           pass
   ```
   
   PRINCIPIO: "Abstrae persistencia; dominio no sabe de SQL/MongoDB."


## 11. SERVICIO DE DOMINIO ORQUESTADOR

   Ubicación: entities.py
   
   ```python
   @dataclass
   class MultiLayerInferenceService:
       """Coordina inferencia coherente a través de 7 capas"""
       evidence_calculator: BayesianEvidenceCalculator
       discriminator: TheoryDiscriminator
       layer_analyzers: Dict[int, LayerAnalyzer]
       
       def infer_all_layers(self, event: QuantumDecodedEvent):
           """Inferencia bayesiana multi-modelo"""
           # Calcula posterior sobre teorías
           # Respeta correlaciones
           # Retorna P(theory | data)
   ```
   
   PRINCIPIO: "Servicios de dominio orquestan operaciones complejas."


# ============================================================================
# CONTRAPOSICIÓN: QUÉ NO VA EN DOMINIO
# ============================================================================

❌ **EN DOMINIO:**
```python
# INCORRECTO: síntesis de datos en dominio
class SignalGenerator:
    def generate_waveform_imr():
        ...

# INCORRECTO: persistencia en dominio
class EventRepository:
    def query_sql():
        ...

# INCORRECTO: HTTP/REST en dominio
def api_endpoint_get_events():
    ...

# INCORRECTO: logging/debugging en dominio
logger.info("Processing event...")
```

✅ **EN INFRAESTRUCTURA:**
```python
# src/infrastructure/sstg/generator.py
class SignalGenerator:
    def generate_waveform_imr():  # ← Síntesis
        ...

# src/infrastructure/repositories/
class SQLRepository(GravitationalWaveRepository):
    def find_by_id(self, id):  # ← Implementación SQL
        ...

# src/presentation/api.py
def get_events():  # ← HTTP
    ...

# General logging
logger.info("...")
```


# ============================================================================
# ESTRUCTURA DE DIRECTORIOS ACTUAL (CORRECTA)
# ============================================================================

```
src/domain/astrophysics/           ← DOMINIO PURO
├── __init__.py                    (exportar)
├── value_objects.py              (tipos + validación)
├── layers.py                     (7 capas)
├── entities.py                   (raíz agregado + interfaces)
├── theory_signatures.py          (discriminadores bayesianos)
├── repositories.py               (interfaces persistencia)
├── domain_services.py            (lógica reutilizable)
├── converters.py                 (adaptadores legacy)
├── ARCHITECTURE.md               (guía física)
├── DDD_ARCHITECTURE.md           (este archivo)
└── sstg/                         ❌ DEBERÍA SER INFRAESTRUCTURA


src/application/                   ← USE CASES
├── __init__.py
├── infer_layers_use_case.py      (coordinador)
└── ...

src/infrastructure/                ← IMPLEMENTACIÓN
├── sstg/                         ✅ Generador sintético
│   ├── engine.py
│   ├── generator.py
│   ├── injectors/
│   └── providers/
├── repositories/                 ✅ Backends concretos
│   ├── memory_repository.py
│   ├── sql_repository.py
│   └── mongodb_repository.py
├── adapters/                     ✅ Adaptadores
└── storage/
```

**IMPORTANTE**: SSTG debe moverse a infraestructura. Ver ARCHITECTURE.md.


# ============================================================================
# VERIFICACIÓN DE LIMPIEZA DDD
# ============================================================================

Use las siguientes preguntas para auditar:

1. ¿Hay imports de frameworks (SQLAlchemy, requests, etc) en domain/*?
   → Si SÍ: ❌ INCORRECTO
   → Si NO: ✅ CORRECTO

2. ¿Todas las clases en domain/* son puras (sin I/O)?
   → Si SÍ: ✅ CORRECTO
   → Si NO: ❌ INCORRECTO

3. ¿Value Objects son frozen y validados?
   → Si SÍ: ✅ CORRECTO
   → Si NO: ❌ INCORRECTO

4. ¿Hay lógica de síntesis de datos en domain/*?
   → Si SÍ: ❌ INCORRECTO (debe ir a infraestructura)
   → Si NO: ✅ CORRECTO

5. ¿Las interfaces (ABC) no tienen implementaciones?
   → Si SÍ: ✅ CORRECTO
   → Si NO: ❌ INCORRECTO

6. ¿El dominio es agnóstico a cómo se persisten datos?
   → Si SÍ: ✅ CORRECTO (Repositories son interfaces)
   → Si NO: ❌ INCORRECTO

7. ¿Se usan Specifications para consultas?
   → Si SÍ: ✅ CORRECTO
   → Si NO: ⚠️ MEJORA POSIBLE (pero no crítico)

8. ¿Enums en lugar de strings mágicos?
   → Si SÍ: ✅ CORRECTO
   → Si NO: ❌ INCORRECTO

STATE ACTUAL:
- ✅ value_objects.py: CORRECTO
- ✅ layers.py: CORRECTO
- ✅ entities.py: CORRECTO
- ✅ theory_signatures.py: CORRECTO
- ✅ repositories.py: CORRECTO
- ✅ domain_services.py: CORRECTO (NUEVO)
- ⚠️ converters.py: LEGACY (delegando a domain_services)
- ⚠️ sstg/*: DEBERÍA ESTAR EN INFRAESTRUCTURA

ACCIÓN: Mover SSTG a src/infrastructure/sstg/ antes de merge a main.
"""
