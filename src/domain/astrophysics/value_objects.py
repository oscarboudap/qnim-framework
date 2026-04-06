# src/domain/astrophysics/value_objects.py
from dataclasses import dataclass
from enum import Enum

class DetectorType(Enum):
    H1 = "Hanford"
    L1 = "Livingston"
    V1 = "Virgo"

@dataclass(frozen=True)
class GPSTime:
    """Representa un tiempo exacto en el calendario GPS de LIGO."""
    value: float

    def __post_init__(self):
        if self.value < 0:
            raise ValueError("El tiempo GPS no puede ser negativo.")