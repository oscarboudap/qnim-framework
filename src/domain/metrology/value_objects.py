"""
Value Objects para la capa de metrología.

Estos objetos representan conceptos metrológicos inmutables y validados.
Son tipos ricos que reemplazan primitivos (dict, float, list).

Nivel DDD: Postdoctoral
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from src.domain.astrophysics.value_objects import TheoryFamily


# ============================================================================
# CAPA 4: COSMOLOGÍA
# ============================================================================

@dataclass(frozen=True)
class HubbleConstant:
    """
    Constante de Hubble inferida de una onda gravitacional.
    
    Attributes:
        value_km_s_mpc: Valor en km/s/Mpc (rango: 60-80 para Universo actual)
        redshift: Redshift medido del evento
        luminosity_distance_mpc: Distancia de luminosidad en Mpc
        upper_bound_km_s_mpc: Cota superior (intervalo de credibilidad 68%)
        lower_bound_km_s_mpc: Cota inferior (intervalo de credibilidad 68%)
    """
    value_km_s_mpc: float
    redshift: float
    luminosity_distance_mpc: float
    upper_bound_km_s_mpc: float
    lower_bound_km_s_mpc: float
    
    def __post_init__(self):
        if self.value_km_s_mpc <= 0:
            raise ValueError("H0 debe ser positiva")
        if self.redshift < 0:
            raise ValueError("Redshift no puede ser negativo")
        if self.luminosity_distance_mpc <= 0:
            raise ValueError("Distancia de luminosidad debe ser positiva")
        if not (self.lower_bound_km_s_mpc <= self.value_km_s_mpc <= 
                self.upper_bound_km_s_mpc):
            raise ValueError("Valor H0 fuera del intervalo de error")
    
    @property
    def relative_uncertainty(self) -> float:
        """Incertidumbre relativa (σ/H0)."""
        sigma = (self.upper_bound_km_s_mpc - self.lower_bound_km_s_mpc) / 2
        return sigma / self.value_km_s_mpc
    
    @property
    def is_tension_with_planck(self) -> bool:
        """¿Hay tensión con medida de Planck (67.4 km/s/Mpc)?"""
        return self.lower_bound_km_s_mpc > 70.0  # ~3-sigma tension


# ============================================================================
# CAPA 6: TOPOLOGÍA DEL HORIZONTE
# ============================================================================

@dataclass(frozen=True)
class NoHairViolationResult:
    """
    Resultado de validar el Teorema de No-Cabello de Kerr.
    
    En Relatividad General, el multipolo de cuadrupolo Q debe satisfacer:
    Q = -a^2 * M^3  (relación de Kerr exacta)
    
    Si hay anomalía cuántica o nueva física, Q desviará de esta predicción.
    
    Attributes:
        expected_kerr_q: Predicción exacta de RG (Q_Kerr = -a^2 * M^3)
        measured_delta_q: Desviación medida del modelo QNN
        is_violated: ¿Se viola el teorema? (δQ > tolerancia)
        violation_magnitude: Magnitud de la violación en unidades de σ_ruido
        inferred_theory: Teoría compatible con la violación detectada
        confidence: Confianza de la predicción (0-1)
    """
    expected_kerr_q: float
    measured_delta_q: float
    is_violated: bool
    violation_magnitude: float
    inferred_theory: TheoryFamily
    confidence: float
    
    def __post_init__(self):
        if not (0 <= self.confidence <= 1):
            raise ValueError("Confidence debe estar en [0, 1]")
        if self.violation_magnitude < 0:
            raise ValueError("Magnitud de violación debe ser positiva")
    
    @property
    def discovery_sigma(self) -> float:
        """Equivalente en σ (5-sigma = descubrimiento)."""
        return self.violation_magnitude if self.is_violated else 0.0


# ============================================================================
# CAPA 7: FÍSICA CUÁNTICA PROFUNDA
# ============================================================================

@dataclass(frozen=True)
class QuantumGravitySignificance:
    """
    Significancia estadística de desviaciones de Relatividad General.
    
    Metrología cuántica en el contexto de ondas gravitacionales para
    detectar nueva física en la escala de Planck.
    
    Attributes:
        sigma_value: Desviación en unidades de σ (Z-score)
        p_value: Valor-p bilateral (Gauss)
        is_discovery: ¿5-sigma o más? (criterio de descubrimiento)
        conclusion: Interpretación física legible
        discovery_threshold_sigma: Umbral usado (normalmente 5.0)
    """
    sigma_value: float
    p_value: float
    is_discovery: bool
    conclusion: str
    discovery_threshold_sigma: float = 5.0
    
    def __post_init__(self):
        if self.sigma_value < 0:
            raise ValueError("Sigma debe ser positiva")
        if not (0 <= self.p_value <= 1):
            raise ValueError("p-value debe estar en [0, 1]")
        if self.discovery_threshold_sigma <= 0:
            raise ValueError("Umbral debe ser positivo")
    
    @property
    def is_3sigma(self) -> bool:
        """¿Evidencia moderada (3-sigma)?"""
        return 3.0 <= self.sigma_value < 5.0
    
    @property
    def is_beyond_gr(self) -> bool:
        """¿Evidencia de nueva física?"""
        return self.is_discovery or self.is_3sigma


# ============================================================================
# INSTRUMENTACIÓN: POWER SPECTRAL DENSITY
# ============================================================================

@dataclass(frozen=True)
class PowerSpectralDensity:
    """
    Power Spectral Density (PSD) del detector de ondas gravitacionales.
    
    Representa el ruido del detector en el dominio de frecuencias.
    Necesario para calcular productos escalares en el espacio de Hilbert.
    
    Attributes:
        frequency_bins: Array 1D de frecuencias (Hz)
        psd_values: Array 1D de PSD (Hz^-1)
        delta_f: Resolución de frecuencia (Hz)
        detector_name: Nombre del detector (H1, L1, V1)
    """
    frequency_bins: np.ndarray
    psd_values: np.ndarray
    delta_f: float
    detector_name: str
    
    def __post_init__(self):
        if len(self.frequency_bins) != len(self.psd_values):
            raise ValueError("frequency_bins y psd_values deben tener la misma longitud")
        if self.delta_f <= 0:
            raise ValueError("delta_f debe ser positiva")
        if np.any(self.psd_values <= 0):
            raise ValueError("PSD values deben ser positivas (ruido no negativo)")
        if self.detector_name not in ["H1", "L1", "V1", "K1"]:
            raise ValueError(f"Detector desconocido: {self.detector_name}")


# ============================================================================
# METROLOGÍA: MATRIZ DE FISHER
# ============================================================================

@dataclass(frozen=True)
class FisherMatrix:
    """
    Matriz de Información de Fisher.
    
    Γ_ij = ⟨∂_i h | ∂_j h⟩ con producto escalar ponderado por ruido.
    
    Propiedades:
    - Simétrica (Hermitiana)
    - Inversa = cota inferior de Cramér-Rao
    - Diagonal: errores independientes de parámetros
    - Determinante: volumen de elipse de error
    
    Attributes:
        matrix: Array 2D NumPy, simétrico
        parameter_names: Nombres de parámetros (filas/columnas)
        snr: Signal-to-Noise Ratio del evento
    """
    matrix: np.ndarray
    parameter_names: tuple
    snr: float
    
    def __post_init__(self):
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Matriz debe ser cuadrada")
        if len(self.parameter_names) != self.matrix.shape[0]:
            raise ValueError("Número de parámetros no coincide con matriz")
        if self.snr < 0:
            raise ValueError("SNR debe ser positiva")
        # Verificar simetría
        if not np.allclose(self.matrix, self.matrix.T):
            raise ValueError("Matriz debe ser simétrica")
    
    @property
    def cramer_rao_bounds(self) -> dict:
        """
        Cotas inferiores de Cramér-Rao (√(F^-1)_ii).
        
        Returns:
            Dict mapping parameter name → lower bound on uncertainty
        """
        try:
            cov = np.linalg.inv(self.matrix)
            bounds = np.sqrt(np.diag(cov))
            return dict(zip(self.parameter_names, bounds))
        except np.linalg.LinAlgError:
            return {}
    
    @property
    def eigvals_condition_number(self) -> float:
        """Número de condición (λ_max / λ_min)."""
        eigvals = np.linalg.eigvalsh(self.matrix)
        eigvals = eigvals[eigvals > 1e-10]  # Ignorar eigenvalores negativos/nulos
        if len(eigvals) < 2:
            return 1.0
        return np.max(eigvals) / np.min(eigvals)
