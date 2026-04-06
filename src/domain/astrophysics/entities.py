# src/domain/astrophysics/entities.py
from dataclasses import dataclass
import numpy as np
from .value_objects import DetectorType, GPSTime

@dataclass(frozen=True)
class GWSignal:
    """Entidad raíz del dominio de Astrofísica."""
    strain: np.ndarray
    detector: DetectorType
    sample_rate: int
    gps_start: GPSTime

    @property
    def duration(self) -> float:
        return len(self.strain) / self.sample_rate

    @property
    def time_vector(self) -> np.ndarray:
        return np.linspace(0, self.duration, len(self.strain))

    def __repr__(self):
        return f"<GWSignal {self.detector.name} | {self.duration:.2f}s | {self.sample_rate}Hz>"