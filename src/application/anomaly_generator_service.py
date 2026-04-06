import numpy as np
from src.domain.astrophysics.entities import GWSignal

class AnomalyGeneratorService:
    def apply_quantum_drift(self, signal: GWSignal, epsilon: float = 0.1) -> GWSignal:
        peak_idx = np.argmax(np.abs(signal.strain))
        corrected_strain = np.copy(signal.strain)
        t_post = np.arange(len(signal.strain) - peak_idx)
        # Efecto de gravedad cuántica: eco post-merger
        echo = epsilon * np.sin(0.5 * t_post) * np.exp(-t_post / 500)
        corrected_strain[peak_idx:] += echo
        return GWSignal(corrected_strain, signal.detector, signal.sample_rate, signal.gps_start)