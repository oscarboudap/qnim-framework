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
    
@dataclass
class PlanckEventTechnicalSheet:
    # A. Módulo de Estructura (Kerr vs Exotic)
    kerr_violation_parameter: float  # Desviación del momento cuadrupolar (No-Hair)
    tidal_deformability: float       # Lambda (Estrellas de Neutrones)
    
    # B. Módulo de Cosmología
    luminosity_distance: float       # Sirena estándar
    hubble_contribution: float       # H0 inference
    
    # C. Módulo de Gravedad Cuántica
    lqg_confidence: float            # Discretización R y rebote epsilon
    string_fuzzball_echoes: float    # Parámetro de reflectividad de la superficie
    
    # D. Robustez
    snr_instrumental: float          # SNR real post-whitening
    sigma_confidence: float          # Nivel de significancia (1-sigma a 5-sigma)