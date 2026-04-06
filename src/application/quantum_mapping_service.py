import numpy as np
from src.domain.astrophysics.entities import GWSignal

class QuantumMappingService:
    """Implementa IQuantumEmbedder según el diagrama de clases."""
    def __init__(self, target_qubits: int = 8):
        self.target_qubits = target_qubits

    def prepare_for_embedding(self, signal: GWSignal) -> np.ndarray:
        """
        Mapea amplitudes de la señal a ángulos de rotación para las puertas cuánticas.
        Usa una ventana del signal para ajustarse al número de qubits.
        """
        data = signal.strain
        # Tomamos los últimos N puntos (fase de ringdown donde está la anomalía)
        window = data[-self.target_qubits:] 
        
        # Normalización a [0, pi] para rotaciones Ry/Rz
        angles = np.interp(window, (window.min(), window.max()), (0, np.pi))
        return angles