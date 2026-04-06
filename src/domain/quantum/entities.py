from dataclasses import dataclass
from typing import Dict

@dataclass
class QuantumConfiguration:
    n_qubits: int
    reps: int
    optimization_level: int

@dataclass
class InferenceResult:
    probabilities: Dict[int, float]
    predicted_class: int
    metadata: Dict