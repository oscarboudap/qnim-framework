import numpy as np
from src.domain.astrophysics.entities import GWSignal

class QuantumMappingService:
    def __init__(self, target_qubits: int = 4):
        self.target_qubits = target_qubits

    def prepare_for_embedding(self, signal: GWSignal) -> np.ndarray:
        """
        Localiza el evento y extrae los puntos para los qubits.
        Normaliza a [0, pi] para la rotación de puertas cuánticas.
        """
        # Buscamos el pico de amplitud (el choque de los agujeros negros)
        # En una señal blanqueada, el pico es muy claro
        peak_idx = np.argmax(np.abs(signal.strain))
        
        # Tomamos una ventana alrededor del pico
        half_qubits = self.target_qubits // 2
        chunk = signal.strain[peak_idx - half_qubits : peak_idx + half_qubits]
        
        # Si el recorte falla por bordes, forzamos el tamaño
        if len(chunk) != self.target_qubits:
            chunk = signal.strain[:self.target_qubits]

        # Escalado para la Esfera de Bloch (Imprescindible para Qiskit)
        # Mapeamos el rango de la señal a [0, np.pi]
        min_val, max_val = np.min(chunk), np.max(chunk)
        normalized = (chunk - min_val) / (max_val - min_val) * np.pi
        
        return normalized