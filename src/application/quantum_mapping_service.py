import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from src.domain.astrophysics.entities import GWSignal

class QuantumMappingService:
    def __init__(self, target_qubits: int = 8, reps: int = 1):
        self.target_qubits = target_qubits
        self.reps = reps
        # Definimos el mapa de características con entrelazamiento lineal
        # Esto mimetiza la propagación temporal de la onda gravitacional
        self._feature_map = ZZFeatureMap(
            feature_dimension=target_qubits, 
            reps=reps, 
            entanglement='linear'
        )

    def prepare_for_embedding(self, signal: GWSignal) -> np.ndarray:
        """
        Extrae los momentos clave del ringdown y los prepara para el mapa ZZ.
        Foco: Resolución espectral de la escala de Planck (Anexo C).
        """
        strain = signal.strain
        # Foco en el 'Merger-Ringdown': los últimos N puntos de la señal blanqueada
        # Aquí es donde la curvatura es máxima y los efectos cuánticos emergen
        window = strain[-self.target_qubits:]
        
        # Normalización estadística para evitar la saturación de las puertas ZZ
        # Escalamos a [0, 2*pi] para aprovechar todo el círculo de Bloch
        angles = np.interp(window, (window.min(), window.max()), (0, 2 * np.pi))
        
        return angles

    def get_feature_map(self):
        return self._feature_map
    def prepare_multitask_embedding(self, signal: GWSignal):
        # Aseguramos que tomamos target_qubits puntos de cada zona
        inspiral = signal.strain[:self.target_qubits]
        ringdown = signal.strain[-self.target_qubits:]
        
        combined = np.concatenate([inspiral, ringdown])
        # Re-escalado a rango de fase cuántica [0, 2pi]
        return np.interp(combined, (combined.min(), combined.max()), (0, 2*np.pi))