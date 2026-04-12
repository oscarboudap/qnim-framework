import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation

class QuantumBridge:
    """
    El Corazón Cuántico de QNIM.
    Realiza el mapeo directo de la señal de GW al Espacio de Hilbert
    usando Amplitude Encoding (Cuántica Pura).
    """

    def __init__(self, n_qubits=6):
        self.n_qubits = n_qubits
        self.feature_dimension = 2**n_qubits

    def build_quantum_state(self, normalized_strain):
        """
        Transforma el vector de strain en un circuito de preparación de estado.
        """
        if len(normalized_strain) != self.feature_dimension:
            raise ValueError(f"La señal debe tener exactamente {self.feature_dimension} puntos.")

        # Creamos el circuito base
        qc = QuantumCircuit(self.n_qubits)
        
        # 1. Amplitude Encoding (State Preparation)
        # Esto 'comprime' la señal completa en la amplitud de probabilidad de los cúbits
        state_prep = StatePreparation(normalized_strain)
        qc.append(state_prep, range(self.n_qubits))
        
        # 2. Barrera de coherencia (Para separar carga de procesamiento)
        qc.barrier()
        
        return qc

    def get_info(self):
        return {
            "method": "Amplitude Encoding",
            "qubits_used": self.n_qubits,
            "data_points_mapped": self.feature_dimension
        }