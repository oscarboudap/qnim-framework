import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes

class QuantumBridge:
    """
    El puente entre la astrofísica clásica y el espacio de Hilbert.
    Transforma el strain de la GW en un estado cuántico entrelazado.
    """

    def __init__(self, n_qubits=8):
        self.n_qubits = n_qubits

    def encode_waveform(self, strain_data):
        """
        Realiza el Angle Encoding de la señal.
        Mapeamos la fase de la onda a rotaciones Ry y Rz.
        """
        # 1. Normalización de la señal para el rango [0, pi]
        normalized_strain = self._normalize_for_quantum(strain_data)
        
        # 2. Creación del circuito de Feature Map
        # Usamos ZZFeatureMap para capturar correlaciones no lineales (entrelazamiento)
        # Esto es clave para detectar "Ecos" de cuerdas o "Jitter" de LQG
        circuit = ZZFeatureMap(feature_dimension=self.n_qubits, reps=2, entanglement='linear')
        
        # 3. Binding de los datos (inyectamos la onda en el circuito)
        # Tomamos una muestra de la onda que encaje con el número de cúbits
        params = normalized_strain[:self.n_qubits]
        bound_circuit = circuit.assign_parameters(params)
        
        return bound_circuit

    def _normalize_for_quantum(self, data):
        """Escala la señal para que sea un ángulo válido en la esfera de Bloch."""
        return np.pi * (data - np.min(data)) / (np.max(data) - np.min(data))