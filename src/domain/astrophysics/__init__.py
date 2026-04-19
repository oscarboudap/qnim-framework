"""
Dominio de Astrofísica de Ondas Gravitacionales.

Exporta la arquitectura de 7 capas con más de 40 dimensiones de observables
para inferencia bayesiana multi-modelo de teorías de gravedad.

ESTRUCTURA DDD (Domain-Driven Design):
- value_objects: Medidas inmutables con incertidumbre
- layers: 7 capas de información física
- entities: Raíces de agregado (QuantumDecodedEvent)
- theory_signatures: Discriminadores bayesianos
- repositories: Interfaces persistencia (abstractas)
- domain_services: Lógica pura de astrofísica (reutilizable)
- converters: Adaptadores (compatibilidad hacia atrás)
"""

# ============================================================================
# VALUE OBJECTS (Tipos de datos de dominio)
# ============================================================================
from .value_objects import (
    # Enumeraciones
    DetectorType,
    TheoryFamily,
    QuantumProperties,
    
    # Physical Quantities con incertidumbres
    PhysicalQuantity,
    Measurement,
    SolarMass,
    FrequencyHz,
    DistanceMPC,
    GPSTime,
    Spin,
    Polarization,
    Inclination,
    TidalDeformability,
    QuadrupoleMoment,
    WaveAmplitude,
    Redshift,
    GravitonMass,
    Eccentricity,
    SignalToNoise,
)

# ============================================================================
# CAPAS FÍSICAS (Layers)
# ============================================================================
from .layers import (
    # Capa 1
    FrequencyEvolution,
    PolarizationContent,
    MultipolarDecomposition,
    DetectorCoherence,
    SignalMathematicalStructure,
    
    # Capa 2
    ChirpMassParameters,
    SpinConfiguration,
    OrbitalDynamics,
    MatterProperties,
    NoHairTheorem,
    IntrinsicGeometry,
    
    # Capa 3
    AccretionDiskEnvironment,
    MagneticFieldEnvironment,
    AxionSuperradianceCloud,
    AstrophysicalEnvironment,
    
    # Capa 4
    StandardSirenCosmology,
    StochasticGWBackground,
    FundamentalConstantViolations,
    CosmologicalEvolution,
    
    # Capa 5 (Beyond GR)
    ScalarTensorSignatures,
    ModifiedGravityDispersion,
    ExtraDimensionSignatures,
    ExoticCompactObjectSignatures,
    AxionSuperradianceSignatures,
    BeyondGRSignatures,
    
    # Capa 6
    EchoSpectroscopy,
    BMSSoftHairStructure,
    GravitationalMemory,
    HorizonQuantumTopology,
    
    # Capa 7
    AdSCFTDuality,
    QuantumCorrectionsMetric,
    LorentzViolation,
    DeepQuantumManifold,
)

# ============================================================================
# ENTIDADES Y AGREGADOS
# ============================================================================
from .entities import (
    # Raíz del agregado
    QuantumDecodedEvent,
    
    # Especificaciones
    GWEventSpecification,
    
    # Interfaces de dominio
    BayesianEvidenceCalculator,
    TheoryDiscriminator,
    LayerAnalyzer,
    
    # Servicio de dominio (orquestador)
    MultiLayerInferenceService,
    
    # Factory
    QuantumDecodedEventFactory,
)

# ============================================================================
# SERVICIOS DE DOMINIO (Lógica pura reutilizable)
# ============================================================================
from .domain_services import (
    # Constantes y calculadoras
    PhysicalConstants,
    AstrophysicalCalculus,
    
    # Validadores
    PhysicalConstraintValidator,
    
    # Calculadoras cosmológicas
    CosmologicalPropagation,
    
    # Métricas de significancia
    LayerSignificanceCalculator,
)

# ============================================================================
# FIRMAS TEÓRICAS Y DISCRIMINADORES
# ============================================================================
from .theory_signatures import (
    # Analizadores por teoría
    ScalarTensorAnalyzer,
    ModifiedGravityAnalyzer,
    ExtraDimensionAnalyzer,
    ECOAnalyzer,
    AxionSuperradianceAnalyzer,
    
    # Calculador bayesiano
    BayesianMultiModelCalculator,
    
    # Discriminador de teorías
    BayesianTheoryDiscriminator,
    
    # Tabla de firmas
    TheorySignatureLibrary,
)

# ============================================================================
# REPOSITORIOS Y PERSISTENCIA
# ============================================================================
from .repositories import (
    # Interfaces
    GravitationalWaveRepository,
    UnitOfWork,
    LayerRepository,
    
    # Especificaciones
    EventSpecification,
    MinimumSNRSpecification,
    BeyondGRConfidenceSpecification,
    HasEchoesSpecification,
    TheoryFamilySpecification,
    CompositeSpecification,
    
    # Factory
    RepositoryFactory,
)

# ============================================================================
# ADAPTADORES Y CONVERTIDORES (Legacy compatibility)
# ============================================================================
from .converters import AstrophysicsMath

# ============================================================================
# EXPORTACIONES PÚBLICAS
# ============================================================================

__all__ = [
    # Value Objects
    "DetectorType",
    "TheoryFamily",
    "QuantumProperties",
    "PhysicalQuantity",
    "Measurement",
    "SolarMass",
    "FrequencyHz",
    "DistanceMPC",
    "GPSTime",
    "Spin",
    "Polarization",
    "Inclination",
    "TidalDeformability",
    "QuadrupoleMoment",
    "WaveAmplitude",
    "Redshift",
    "GravitonMass",
    "Eccentricity",
    "SignalToNoise",
    
    # Capas
    "FrequencyEvolution",
    "PolarizationContent",
    "MultipolarDecomposition",
    "DetectorCoherence",
    "SignalMathematicalStructure",
    "ChirpMassParameters",
    "SpinConfiguration",
    "OrbitalDynamics",
    "MatterProperties",
    "NoHairTheorem",
    "IntrinsicGeometry",
    "AccretionDiskEnvironment",
    "MagneticFieldEnvironment",
    "AxionSuperradianceCloud",
    "AstrophysicalEnvironment",
    "StandardSirenCosmology",
    "StochasticGWBackground",
    "FundamentalConstantViolations",
    "CosmologicalEvolution",
    "ScalarTensorSignatures",
    "ModifiedGravityDispersion",
    "ExtraDimensionSignatures",
    "ExoticCompactObjectSignatures",
    "AxionSuperradianceSignatures",
    "BeyondGRSignatures",
    "EchoSpectroscopy",
    "BMSSoftHairStructure",
    "GravitationalMemory",
    "HorizonQuantumTopology",
    "AdSCFTDuality",
    "QuantumCorrectionsMetric",
    "LorentzViolation",
    "DeepQuantumManifold",
    
    # Entidades
    "QuantumDecodedEvent",
    "GWEventSpecification",
    "BayesianEvidenceCalculator",
    "TheoryDiscriminator",
    "LayerAnalyzer",
    "MultiLayerInferenceService",
    "QuantumDecodedEventFactory",
    
    # Servicios de Dominio
    "PhysicalConstants",
    "AstrophysicalCalculus",
    "PhysicalConstraintValidator",
    "CosmologicalPropagation",
    "LayerSignificanceCalculator",
    
    # Firmas
    "ScalarTensorAnalyzer",
    "ModifiedGravityAnalyzer",
    "ExtraDimensionAnalyzer",
    "ECOAnalyzer",
    "AxionSuperradianceAnalyzer",
    "BayesianMultiModelCalculator",
    "BayesianTheoryDiscriminator",
    "TheorySignatureLibrary",
    
    # Repositorios
    "GravitationalWaveRepository",
    "UnitOfWork",
    "LayerRepository",
    "EventSpecification",
    "MinimumSNRSpecification",
    "BeyondGRConfidenceSpecification",
    "HasEchoesSpecification",
    "TheoryFamilySpecification",
    "CompositeSpecification",
    "RepositoryFactory",
    
    # Adaptadores/Legacy
    "AstrophysicsMath",
]


