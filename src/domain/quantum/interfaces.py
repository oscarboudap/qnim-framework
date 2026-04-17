# src/domain/quantum/interfaces.py
from abc import ABC, abstractmethod
import numpy as np

class IQuantumAnnealer(ABC):
    """
    Puerto para el Recocido Cuántico (D-Wave).
    Garantiza el diseño Hot-Swap: puede inyectarse un simulador local o la QPU real.
    """
    @abstractmethod
    def sample_qubo(self, Q: dict, num_reads: int) -> dict:
        pass

class IGateBasedQuantumComputer(ABC):
    """
    Puerto para Computación basada en Puertas (IBM/Qiskit).
    """
    @abstractmethod
    def execute_circuit(self, circuit_data, backend_name: str) -> np.ndarray:
        pass