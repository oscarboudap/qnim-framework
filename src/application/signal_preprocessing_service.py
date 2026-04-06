import numpy as np
from src.domain.astrophysics.entities import GWSignal

class SignalPreprocessingService:
    def clean_signal(self, signal: GWSignal) -> GWSignal:
        """Normalización y filtrado básico para el embedding cuántico."""
        # Normalizamos a rango [-1, 1] y aplicamos un recorte de seguridad
        norm_strain = signal.strain / np.max(np.abs(signal.strain))
        return GWSignal(norm_strain, signal.detector, signal.sample_rate, signal.gps_start)