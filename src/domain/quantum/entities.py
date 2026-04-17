# src/domain/quantum/entities.py
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class QUBOProblem:
    """Representación matemática pura de un Quadratic Unconstrained Binary Optimization."""
    linear_terms: Dict[int, float]
    quadratic_terms: Dict[Tuple[int, int], float]
    offset: float = 0.0

@dataclass
class AnnealingResult:
    """Resultado devuelto por la QPU de D-Wave o el simulador Neal."""
    best_state: Dict[int, int]
    lowest_energy: float
    num_occurrences: int
    is_ground_state_confident: bool

@dataclass
class VQCTopology:
    """Definición teórica de la red neuronal cuántica."""
    num_qubits: int
    feature_map_reps: int
    ansatz_reps: int
    entanglement_strategy: str # Ej: 'linear', 'full'