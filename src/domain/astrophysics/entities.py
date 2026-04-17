# src/domain/astrophysics/entities.py
from dataclasses import dataclass
import numpy as np
from .value_objects import DetectorType, GPSTime
from .layers import IntrinsicGeometry, CosmologicalFootprint, BeyondGRDeviations, QuantumHorizonTopology
from typing import Optional
@dataclass(frozen=True)
class GWSignal:
    """Capa 1: La señal pura detectada por el interferómetro."""
    strain: np.ndarray
    detector: DetectorType
    sample_rate: int
    gps_start: GPSTime
    snr_instrumental: float # SNR post-whitening

    @property
    def duration(self) -> float:
        return len(self.strain) / self.sample_rate

    @property
    def time_vector(self) -> np.ndarray:
        return np.linspace(0, self.duration, len(self.strain))

@dataclass
class QuantumDecodedEvent:
    """
    ENTIDAD RAÍZ (Aggregate Root).
    Representa el resultado final de la Inferencia Cuántica QNIM.
    """
    event_id: str
    signal: GWSignal # Capa 1
    
    # Capas de Realidad (Inicialmente vacías, se llenan por el Orquestador Híbrido)
    geometry: Optional[IntrinsicGeometry] = None          # Llenado por D-Wave
    cosmology: Optional[CosmologicalFootprint] = None     # Llenado por cálculos clásicos
    beyond_gr: Optional[BeyondGRDeviations] = None        # Llenado por VQC (12-27 cúbits)
    topology: Optional[QuantumHorizonTopology] = None     # Llenado por VQC (12-27 cúbits)

    def is_standard_general_relativity(self) -> bool:
        """Regla de Negocio del Dominio: Comprueba la invulnerabilidad de Kerr."""
        if not self.topology: return True
        return self.topology.detected_theory == TheoryFamily.GENERAL_RELATIVITY

    def __repr__(self):
        theory = self.topology.detected_theory.value if self.topology else "Processing..."
        return f"<QuantumEvent {self.event_id} | SNR: {self.signal.snr_instrumental:.1f} | Theory: {theory}>"