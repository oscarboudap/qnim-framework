import numpy as np
from src.domain.astrophysics.entities import GWSignal

class AnomalyGeneratorService:
    def apply_quantum_drift(self, signal: GWSignal, epsilon: float = 0.1) -> GWSignal:
        """
        Aplica correcciones de Loop Quantum Gravity (Punto 2.2.2.1 de la Memoria).
        Simula la granularidad del espacio-tiempo en la fase de ringdown.
        """
        peak_idx = np.argmax(np.abs(signal.strain))
        corrected_strain = np.copy(signal.strain)
        
        # Simulación de efectos de gravedad cuántica post-merger (ecos de Planck)
        t_post = np.arange(len(signal.strain) - peak_idx)
        # El drift se manifiesta como una perturbación en la frecuencia de decaimiento
        echo = epsilon * np.sin(0.5 * t_post) * np.exp(-t_post / 500)
        corrected_strain[peak_idx:] += echo
        
        return GWSignal(
            strain=corrected_strain, 
            detector=signal.detector, 
            sample_rate=signal.sample_rate, 
            gps_start=signal.gps_start
        )

    def inject_fuzzball_structure(self, signal: GWSignal, multipole_shift: float = 0.2) -> GWSignal:
        """Simula la alteración del momento cuadrupolar (Teoría de Cuerdas)."""
        distorted_strain = signal.strain * (1 + multipole_shift * np.cos(np.linspace(0, 10, len(signal.strain))))
        return GWSignal(distorted_strain, signal.detector, signal.sample_rate, signal.gps_start)