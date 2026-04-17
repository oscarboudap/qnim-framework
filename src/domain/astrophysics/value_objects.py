# src/domain/astrophysics/value_objects.py
from dataclasses import dataclass
from enum import Enum

class DetectorType(Enum):
    H1 = "LIGO_Hanford"
    L1 = "LIGO_Livingston"
    V1 = "Virgo"
    K1 = "KAGRA"
    LISA = "LISA_Space" # Preparado para la 3ª Generación (Escalabilidad TFM)
    ET = "Einstein_Telescope"

class TheoryFamily(Enum):
    """Clasificación de las anomalías físicas (Capa 5 y 6)"""
    GENERAL_RELATIVITY = "Kerr_Vacuum"
    LOOP_QUANTUM_GRAVITY = "LQG_Area_Quantization"
    STRING_FUZZBALL = "String_Theory_Fuzzball"
    SCALAR_TENSOR = "Horndeski_DHOST"
    ECO_GRAVASTAR = "Exotic_Compact_Object"

@dataclass(frozen=True)
class GPSTime:
    """Representa un tiempo exacto en el calendario GPS de LIGO."""
    value: float
    def __post_init__(self):
        if self.value < 0:
            raise ValueError("El tiempo GPS no puede ser negativo.")

@dataclass(frozen=True)
class SolarMass:
    """Encapsula la masa para evitar errores de unidades."""
    value: float
    def __post_init__(self):
        if self.value <= 0:
            raise ValueError("La masa debe ser estrictamente positiva.")