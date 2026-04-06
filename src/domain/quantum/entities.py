# src/domain/quantum/entities.py
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass(frozen=True)
class QuantumConfiguration:
    """Configuración del hardware/simulador cuántico."""
    n_qubits: int
    reps: int = 3
    entanglement: str = 'linear'

@dataclass(frozen=True)
class InferenceResult:
    """Resultado puro de una clasificación cuántica."""
    probabilities: Dict[int, float]
    predicted_class: int
    metadata: Dict[str, any] = field(default_factory=dict)