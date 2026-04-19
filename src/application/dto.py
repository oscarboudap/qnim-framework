"""
Application Layer: Data Transfer Objects (DTOs) & Value Objects
================================================================

Encapsulan los valores de entrada/salida entre capas, garantizando
type safety y documentación explícita de contratos.

Todos son frozen=True (immutable), con validación en __post_init__.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np


# ============================================================================
# INFERENCE PIPELINE RESULTS
# ============================================================================

@dataclass(frozen=True)
class ClassicParametersExtracted:
    """
    Resultado de la rama D-Wave: parámetros clásicos extraídos del template matching.
    
    Corresponde a Capa 2 (Geometría Intrínseca): m1, m2, espín, etc.
    
    Attributes:
        m1_solar_masses: Masa del objeto primario [masas solares]
        m2_solar_masses: Masa del objeto secundario [masas solares]
        chirp_mass_solar_masses: Masa chirp = (m1*m2)^(3/5) / (m1+m2)^(1/5)
        effective_spin: Espín efectivo χ_eff = (m1*s1 + m2*s2) / (m1+m2)
        template_match_snr: SNR del template matching (confianza)
        selected_template_index: Índice en la lista de templates
    """
    m1_solar_masses: float
    m2_solar_masses: float
    chirp_mass_solar_masses: float
    effective_spin: float
    template_match_snr: float
    selected_template_index: int
    
    def __post_init__(self):
        # Validaciones
        assert self.m1_solar_masses > 0, "m1 debe ser positivo"
        assert self.m2_solar_masses > 0, "m2 debe ser positivo"
        assert self.chirp_mass_solar_masses > 0, "chirp_mass debe ser positivo"
        assert -1 <= self.effective_spin <= 1, "χ_eff ∈ [-1, 1]"
        assert self.template_match_snr >= 0, "SNR ≥ 0"
        assert self.selected_template_index >= 0, "Índice ≥ 0"


@dataclass(frozen=True)
class BeyondGRSignature:
    """Firma de Física Más Allá de GR (Capa 5)."""
    dipolar_emission_strength: float  # Violación dipolar (scalar-tensor)
    graviton_mass_ev: float  # Masa del gravitón [eV]
    kk_dimensions_detected: int  # Dimensiones Kaluza-Klein detectadas
    
    def __post_init__(self):
        assert self.dipolar_emission_strength >= 0, "Dipolar ≥ 0"
        assert self.graviton_mass_ev >= 0, "Graviton mass ≥ 0"
        assert 0 <= self.kk_dimensions_detected <= 10, "KK dimensions ∈ [0, 10]"


@dataclass(frozen=True)
class QuantumTopologySignature:
    """Firma de Topología Cuántica del Horizonte (Capa 6)."""
    no_hair_delta_q: float  # Violación de teorema sin-cabello (|Q_meas - Q_kerr|)
    horizon_reflectivity: float  # Reflectividad de ondas de horizonte [0, 1]
    echo_delay_milliseconds: Optional[float] = None  # Delay de ecos (si presentes)
    
    def __post_init__(self):
        assert 0 <= self.no_hair_delta_q <= 1, "ΔQ ∈ [0, 1]"
        assert 0 <= self.horizon_reflectivity <= 1, "Reflectivity ∈ [0, 1]"
        if self.echo_delay_milliseconds is not None:
            assert self.echo_delay_milliseconds > 0, "Echo delay > 0"


@dataclass(frozen=True)
class DeepQuantumManifoldSignature:
    """Firma de Manifold Cuántico Profundo (Capa 7)."""
    ads_cft_dual_error: float  # Error relativo a conjetura AdS/CFT [0, 1]
    discovered_theory_family: str  # Familia de teoría: "GR", "LQG", "ScalarTensor", etc.
    discovery_confidence_sigma: float  # Significancia estadística [σ]
    
    def __post_init__(self):
        assert 0 <= self.ads_cft_dual_error <= 1, "AdS/CFT error ∈ [0, 1]"
        assert self.discovered_theory_family in [
            "GENERAL_RELATIVITY", "LOOP_QUANTUM_GRAVITY", "SCALAR_TENSOR",
            "MODIFIED_GRAVITY", "EXTRA_DIMENSIONS", "ECO", "AXION_SUPERRADIANCE"
        ], "Unknown theory family"
        assert self.discovery_confidence_sigma >= 0, "Confidence ≥ 0σ"


@dataclass(frozen=True)
class ClassificationResult:
    """
    Resultado de la rama IBM (VQC): clasificación de teoría + firmas multivariante.
    
    Encapsula todos los resultados cuánticos en un objeto tipado.
    """
    beyond_gr: BeyondGRSignature
    quantum_topology: QuantumTopologySignature
    deep_manifold: DeepQuantumManifoldSignature
    raw_probability_anomaly: float  # P(anomalía) = probs[0][1]
    
    def __post_init__(self):
        assert 0 <= self.raw_probability_anomaly <= 1, "Probability ∈ [0, 1]"


@dataclass(frozen=True)
class InferenceResult:
    """
    Resultado completo de la pipeline de inferencia (7 capas).
    
    Integra:
    - Capa 2: Parámetros clásicos (D-Wave)
    - Capas 5-7: Firmas cuánticas (IBM)
    - Metología: Auditoría No-Cabello
    - Integración: Estado de agregado modificado
    """
    event_id: str
    classic_parameters: ClassicParametersExtracted
    classification: ClassificationResult
    no_hair_violation_detected: bool
    overall_theory_verdict: str  # Teoría ganadora
    processing_timestamp_gps: float
    snr_final: float


# ============================================================================
# TRAINING & VALIDATION
# ============================================================================

@dataclass(frozen=True)
class TrainingMetrics:
    """Métricas de entrenamiento del modelo cuántico."""
    final_training_loss: float
    final_validation_accuracy: float
    num_iterations_completed: int
    estimated_time_seconds: float
    model_checkpoint_path: str


@dataclass(frozen=True)
class ConfusionMatrixData:
    """Datos brutos para matriz de confusión (sin visualización)."""
    true_positives: int  # TP: Detectadas anomalías reales
    true_negatives: int  # TN: Detector GR correcto
    false_positives: int  # FP: Falsa alarma
    false_negatives: int  # FN: Anomalía no detectada
    
    @property
    def accuracy(self) -> float:
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        if total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / total
    
    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        if denom == 0:
            return 0.0
        return self.true_positives / denom
    
    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        if denom == 0:
            return 0.0
        return self.true_positives / denom


@dataclass(frozen=True)
class SyntheticDatasetInfo:
    """Información sobre dataset sintético generado."""
    num_events_rg: int  # Eventos de Relatividad General
    num_events_lqg: int  # Eventos de Gravedad Cuántica
    total_events: int
    output_directory: str
    golden_events_count: int  # Eventos con SNR > umbral


# ============================================================================
# CONFIGURATION / THRESHOLDS
# ============================================================================

@dataclass(frozen=True)
class ClassificationThresholds:
    """
    Parámetros numéricos de la pipeline de clasificación.
    
    Estos se inyectan en los servicios, evitando magic numbers dispersos.
    """
    # D-Wave branch
    template_matching_min_snr: float = 8.0  # Umbral SNR mínimo para match válido
    dwave_num_reads: int = 100  # Lecturas del recocedor
    
    # IBM branch
    quantum_anomaly_threshold_prob: float = 0.5  # P(anomalía) > 0.5 = detectada
    
    # Coupling constants (física teórica)
    dipolar_coupling_factor: float = 0.05  # a * P_anom
    graviton_mass_factor: float = 1e-23  # b * P_anom [eV]
    kk_dimension_scaling: float = 2.0  # c * P_anom
    horizon_reflectivity_coupling: float = 0.3  # d * P_anom
    
    # No-Hair Theorem audit
    no_hair_tolerance: float = 0.05  # ΔQ_tol
    echo_delay_coupling: float = 50.0  # ms per unit anomaly
    
    # AdS/CFT
    ads_cft_coupling: float = 0.01  # e * P_anom
    discovery_confidence_base: float = 2.0  # σ_base
    discovery_confidence_scale: float = 6.0  # σ_scale * P_anom


# ============================================================================
# EXCEPTIONS
# ============================================================================

class ApplicationException(Exception):
    """Base para exceptions de application layer."""
    pass


class PortNotAvailableException(ApplicationException):
    """Puerto de infraestructura no configurado."""
    pass


class InvalidInputException(ApplicationException):
    """Parámetros de entrada inválidos."""
    pass


class InferenceFailedException(ApplicationException):
    """Pipeline de inferencia falló."""
    pass


class TrainingFailedException(ApplicationException):
    """Entrenamiento del modelo falló."""
    pass
