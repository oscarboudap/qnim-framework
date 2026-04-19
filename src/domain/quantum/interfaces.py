"""Interfaces (portos) para hardware cuántico.

Nível Postdoctoral: Define contratos entre dominio e infraestructura.
Permite inyección de dependencias para hot-swap entre simuladores y QPU real.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import numpy as np

from .value_objects import QUBOProblem, AnnealingResult, QuantumCircuitConfig


class IQuantumAnnealer(ABC):
    """
    Puerto para Adiabatic Quantum Computing / Quantum Annealing (D-Wave).
    
    Contrato: Dado un problema QUBO, retorna la solución aproximada
    encontrada por el procesador cuántico.
    
    Hot-Swap: Puede inyectarse simulador local (Neal) o QPU real (D-Wave).
    """
    
    @abstractmethod
    def sample_qubo(self, problem: QUBOProblem, num_reads: int,
                   chain_strength: float = 2.0) -> AnnealingResult:
        """
        Resuelve un problema QUBO mediante recocido cuántico.
        
        Args:
            problem: QUBOProblem (H = h_i + J_ij)
            num_reads: Número de ejecuciones independientes (default: 1000)
            chain_strength: Fuerza de cadena (D-Wave, default: 2.0 × max|J|)
            
        Returns:
            AnnealingResult con best_state, lowest_energy, num_occurrences
            
        Raises:
            ValueError: Si problem.num_qubits > capacidad del hardware
        """
        pass
    
    @abstractmethod
    def get_embedding_time(self, num_qubits: int) -> float:
        """
        Tiempo estimado para embedding (D-Wave necesita chains).
        
        Args:
            num_qubits: Número de qubits lógicos
            
        Returns:
            Tiempo estimado (microsegundos)
        """
        pass
    
    @abstractmethod
    def get_native_graph_topology(self) -> Dict[int, list]:
        """
        Retorna la topología nativa del hardware (qubit adjacency).
        
        Returns:
            Dict{ qubit: [vecinos] }
        """
        pass


class IGateBasedQuantumComputer(ABC):
    """
    Puerto para Quantum Gate Computing basado en puertas unitarias (IBM/Qiskit).
    
    Contrato: Construir y ejecutar circuitos cuánticos genéricos.
    Capacidad: Variational Quantum Classifiers (VQC), VQE, QAOA.
    """
    
    @abstractmethod
    def execute_circuit(self, circuit_data: dict, config: QuantumCircuitConfig) -> np.ndarray:
        """
        Ejecuta un circuito cuántico y retorna resultados.
        
        Args:
            circuit_data: Serialización del circuito (Qiskit QuantumCircuit o equivalente)
            config: QuantumCircuitConfig (backend, shots, mitigación, etc)
            
        Returns:
            Array 1D de counts (distribución de resultados)
            
        Raises:
            ConnectionError: Si no puede conectar al backend
        """
        pass
    
    @abstractmethod
    def get_backend_properties(self) -> Dict[str, float]:
        """
        Propiedades del backend (ruido, tiempos de puerta, T_coherence).
        
        Returns:
            Dict con {
                'num_qubits': int,
                't1_microseconds': float,
                't2_microseconds': float,
                'gate_fidelity': float,
                'readout_error': float,
            }
        """
        pass