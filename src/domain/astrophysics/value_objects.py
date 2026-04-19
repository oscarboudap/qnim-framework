from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Dict
import numpy as np
from abc import ABC, abstractmethod

# ============================================================================
# ENUMERACIONES DE FÍSICA
# ============================================================================

class DetectorType(Enum):
    H1 = "LIGO_Hanford"
    L1 = "LIGO_Livingston"
    V1 = "Virgo"
    K1 = "KAGRA"
    LISA = "LISA_Space"
    ET = "Einstein_Telescope"
    CE = "Cosmic_Explorer"


class TheoryFamily(Enum):
    """Las 7 familias teóricas principales más referencias internas."""
    # GR estándar
    KERR_VACUUM = "GR_Kerr_Vacuum"
    KERR_NEWMAN = "GR_Kerr_Newman_Charged"
    
    # Gravedad escalar-tensorial (Capa 5.1)
    BRANS_DICKE = "Brans_Dicke_Dipolar"
    HORNDESKI = "Horndeski_DHOST_Scalar"
    
    # Gravedad modificada (Capa 5.2)
    F_R_GRAVITY = "f_R_Modified_Gravity"
    MASSIVE_GRAVITY = "dRGT_Massive_Graviton"
    
    # Dimensiones extra (Capa 5.3)
    KALUZA_KLEIN = "ADD_Kaluza_Klein"
    RANDALL_SUNDRUM = "RS_Warped_Extra_Dim"
    
    # Objetos compactos exóticos (Capa 5.4)
    GRAVASTAR = "ECO_Gravastar"
    BOSON_STAR = "ECO_Boson_Star"
    STRING_FUZZBALL = "ECO_String_Fuzzball"
    PLANCK_STAR = "ECO_Loop_Quantum_Gravity"
    
    # Fenómenos cuánticos (Capa 5.5)
    AXION_CLOUD = "Quantum_Axion_Superradiance"
    
    # Espuma cuántica (Capa 4 - Wheeler)
    QUANTUM_FOAM = "Wheeler_Quantum_Spacetime_Foam"
    
    # Otros (Capa 6-7)
    BMS_SOFT_HAIR = "BMS_Soft_Hair_Symmetry"
    QUANTUM_CORRECTED = "Quantum_Corrected_Metric"
    LORENTZ_VIOLATING = "SME_Lorentz_Violation"


class QuantumProperties(Enum):
    """Categorías de modificaciones cuánticas."""
    HAWKING_RADIATION = "Hawking_Efecto_Evaporación"
    PLANCK_SCALE_CORRECTION = "Corrección_Escala_Planck"
    ENTANGLEMENT_ENTROPY = "Entropía_de_Entanglement"
    FIREWALL_REMNANT = "Firewall_y_Remanentes"


# ============================================================================
# VALUE OBJECTS: CANTIDADES FÍSICAS CON INCERTIDUMBRES
# ============================================================================

class PhysicalQuantity(ABC):
    """Clase base para cantidades físicas medidas con incertidumbre."""
    
    @abstractmethod
    def relative_error(self) -> float:
        """Retorna el error relativo: sigma/value."""
        pass
    
    @abstractmethod
    def signal_to_noise(self) -> float:
        """Retorna SNR: value / sigma."""
        pass


@dataclass(frozen=True)
class Measurement(PhysicalQuantity):
    """Cantidad medida con valor, incertidumbre y correlaciones."""
    value: float
    sigma: float  # Desviación estándar (1-sigma)
    covariance_tags: Tuple[str, ...] = ()  # Tags para correlaciones posteriores
    
    def __post_init__(self):
        if self.sigma < 0:
            raise ValueError("σ debe ser no-negativa")
        if np.isnan(self.value) or np.isnan(self.sigma):
            raise ValueError("Valores NaN en medición")
    
    def relative_error(self) -> float:
        return self.sigma / abs(self.value) if self.value != 0 else np.inf
    
    def signal_to_noise(self) -> float:
        return abs(self.value) / self.sigma if self.sigma > 0 else np.inf
    
    def interval_credible_68(self) -> Tuple[float, float]:
        """Intervalo de credibilidad al 68% (1σ)."""
        return (self.value - self.sigma, self.value + self.sigma)
    
    def interval_credible_95(self) -> Tuple[float, float]:
        """Intervalo de credibilidad al 95% (2σ)."""
        return (self.value - 2*self.sigma, self.value + 2*self.sigma)


@dataclass(frozen=True)
class SolarMass(Measurement):
    """Masa del sistema en masas solares."""
    
    def __post_init__(self):
        super().__post_init__()
        if self.value <= 0:
            raise ValueError("Masa estelar no física")


@dataclass(frozen=True)
class FrequencyHz(Measurement):
    """Frecuencia gravitacional en Hz."""
    
    def __post_init__(self):
        super().__post_init__()
        if self.value <= 0:
            raise ValueError("Frecuencia negativa no física")


@dataclass(frozen=True)
class DistanceMPC(Measurement):
    """Distancia de luminosidad en megaparsecs."""
    
    def __post_init__(self):
        super().__post_init__()
        if self.value <= 0:
            raise ValueError("Distancia negativa")
    
    def to_meters(self) -> Measurement:
        """Conversión a metros."""
        mpc_to_m = 3.086e22
        return Measurement(
            value=self.value * mpc_to_m,
            sigma=self.sigma * mpc_to_m
        )


@dataclass(frozen=True)
class GPSTime(Measurement):
    """Tiempo GPS en segundos (GPSTIME)."""
    
    def __post_init__(self):
        super().__post_init__()
        if self.value < 0:
            raise ValueError("Tiempo GPS negativo")


@dataclass(frozen=True)
class Spin(Measurement):
    """Parámetro de spin adimensional (0 ≤ χ ≤ 1)."""
    
    def __post_init__(self):
        super().__post_init__()
        if not (-1e-6 <= self.value <= 1 + 1e-6):
            raise ValueError(f"Spin fuera de rango: {self.value}")


@dataclass(frozen=True)
class Polarization(Measurement):
    """Ángulo de polarización en radianes."""
    
    def __post_init__(self):
        super().__post_init__()
        if not (0 <= self.value < 2*np.pi):
            raise ValueError(f"Ángulo de polarización fuera de rango")


@dataclass(frozen=True)
class Inclination(Measurement):
    """Ángulo de inclinación en radianes."""
    
    def __post_init__(self):
        super().__post_init__()
        if not (0 <= self.value <= np.pi):
            raise ValueError(f"Inclinación fuera de rango")


@dataclass(frozen=True)
class TidalDeformability(Measurement):
    """Deformabilidad de marea Λ (adimensional) - solo neutron stars."""
    
    def __post_init__(self):
        super().__post_init__()
        # Λ típicamente 0-5000 para NS
        if self.value < 0:
            raise ValueError("Λ debe ser ≥0")


@dataclass(frozen=True)
class QuadrupoleMoment(Measurement):
    """Momento cuadrupolar Q (teorema de no-pelo)."""
    
    def __post_init__(self):
        super().__post_init__()
        # Para Kerr exacto: Q = -J²/M, típicamente Q < 0


@dataclass(frozen=True)
class WaveAmplitude(Measurement):
    """Amplitud de la onda h (strain adimensional)."""
    
    def __post_init__(self):
        super().__post_init__()
        if self.value < 0:
            raise ValueError("Amplitud debe ser positiva")


@dataclass(frozen=True)
class Redshift(Measurement):
    """Redshift cosmológico z."""
    
    def __post_init__(self):
        super().__post_init__()
        if self.value < -1 + 1e-6:
            raise ValueError("Redshift no-físico z < -1")


@dataclass(frozen=True)
class GravitonMass(Measurement):
    """Masa del gravitón en eV/c² (para gravedad masiva)."""
    
    def __post_init__(self):
        super().__post_init__()
        # Cotas empíricas: m_g < 1.27 × 10^-23 eV/c²


@dataclass(frozen=True)
class Eccentricity(Measurement):
    """Excentricidad orbital (0 ≤ e < 1)."""
    
    def __post_init__(self):
        super().__post_init__()
        if not (0 <= self.value < 1):
            raise ValueError(f"Excentricidad fuera de rango: {self.value}")


@dataclass(frozen=True)
class SignalToNoise(Measurement):
    """SNR integrado de evento GW."""
    
    def __post_init__(self):
        super().__post_init__()
        if self.value < 0:
            raise ValueError("SNR negativo")