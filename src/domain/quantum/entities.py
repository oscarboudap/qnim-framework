# src/domain/quantum/entities.py
from dataclasses import dataclass, field
from typing import List, Dict, Any

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

@dataclass
class PlanckInferenceResult:
    # Módulo 1: Naturaleza del objeto (Kerr vs Exótico)
    is_exotic_candidate: bool
    no_hair_deviation: float          # delta Q (Momento Cuadrupolar)
    tidal_deformability: float        # Lambda (Estrellas de Neutrones)
    
    # Módulo 2: Gravedad Cuántica y Propagación
    lqg_evidence: float               # Probabilidad Bayesiana Amortizada LQG
    string_fuzzball_evidence: float    # Probabilidad Cuerdas (ecos de superficie)
    lorentz_violation_alpha: float    # Dispersión (v vs c)
    
    # Módulo 3: Cosmología
    luminosity_distance_mpc: float
    estimated_h0: float
    
    # Metadatos de Confianza
    snr_observed: float
    sigma_level: float                # Significancia estadística (Standard 5-Sigma)
    metadata: Dict[str, Any] = field(default_factory=dict)