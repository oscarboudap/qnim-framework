"""
Value Objects para el dominio cuántico (domain/quantum).

Estos objetos representan conceptos cuánticos inmutables y validados.
Encapsulan la configuración de circuitos cuánticos, resultados de reco cocido,
y problemas de optimización.

Nivel DDD: Postdoctoral
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
import numpy as np


# ============================================================================
# CONFIGURACIÓN CUÁNTICA: VQC (Variational Quantum Classifier)
# ============================================================================

@dataclass(frozen=True)
class VQCTopology:
    """
    Topología de la Red Neuronal Cuántica Variacional (VQC).
    
    Define la estructura del ansatz del circuito cuántico que actuará
    como clasificador para distinguir poblaciones de eventos GW.
    
    Attributes:
        num_qubits: Número de qubits en el procesador (típicamente 2-10)
        num_features: Dimensionalidad del espacio de features classicales
                     (después de PCA de las señales)
        feature_map_reps: Repeticiones del mapa de features (encoding)
        ansatz_reps: Profundidad del ansatz (repeticiones del circuito variable)
        entanglement_type: Estrategia de entrelazamiento: 'linear', 'circular', 'full'
    """
    num_qubits: int
    num_features: int
    feature_map_reps: int
    ansatz_reps: int
    entanglement_type: str = "circular"
    
    def __post_init__(self):
        if self.num_qubits <= 0 or self.num_qubits > 20:
            raise ValueError("num_qubits debe estar en [1, 20]")
        if self.num_features <= 0 or self.num_features > 100:
            raise ValueError("num_features debe estar en [1, 100]")
        if self.feature_map_reps < 0 or self.feature_map_reps > 10:
            raise ValueError("feature_map_reps debe estar en [0, 10]")
        if self.ansatz_reps <= 0 or self.ansatz_reps > 20:
            raise ValueError("ansatz_reps debe estar en [1, 20]")
        if self.entanglement_type not in ["linear", "circular", "full"]:
            raise ValueError(f"entanglement_type no reconocido: {self.entanglement_type}")
    
    @property
    def total_parameters(self) -> int:
        """
        Calcula el número total de parámetros (θ) entrenables.
        
        Depende del tipo de ansatz. Para RealAmplitudes:
            num_θ = num_qubits × (ansatz_reps + 1)
        
        Para Two-Local (con entanglement):
            num_θ = 2 × num_qubits × (ansatz_reps + 1)
        """
        # Asumiendo RealAmplitudes estándar
        return self.num_qubits * (self.ansatz_reps + 1)
    
    @property
    def circuit_depth(self) -> int:
        """Profundidad aproximada del circuito transmonado."""
        # Profundidad = feature_map_reps + ansatz_reps
        return self.feature_map_reps + self.ansatz_reps
    
    @property
    def is_shallow_circuit(self) -> bool:
        """¿Es QAOA/VQE shallow (< 5 capas)? Más robusto al ruido."""
        return self.circuit_depth < 5


# ============================================================================
# OPTIMIZACIÓN CUÁNTICA: QUBO
# ============================================================================

@dataclass(frozen=True)
class QUBOProblem:
    """
    Problema de Optimización Cuadrática No Restringida Binaria (QUBO).
    
    Hamiltoniano: H = Σ_i h_i x_i + Σ_{i<j} J_{ij} x_i x_j
    
    Donde:
    - x_i ∈ {0, 1}: variables binarias
    - h_i: términos lineales
    - J_{ij}: términos de acoplamiento cuadrático
    - Mínimo de H ↔ mejor solución
    
    Attributes:
        linear_terms: Dict{ qubit_index: coefficient }
        quadratic_terms: Dict{ (i, j): coupling } con i < j
        offset: Energía offset (Etotal = H + offset)
        num_qubits: Número de variables binarias
    """
    linear_terms: Dict[int, float]
    quadratic_terms: Dict[Tuple[int, int], float]
    offset: float = 0.0
    
    def __post_init__(self):
        # Validar que todos los índices sean válidos
        all_indices = set(self.linear_terms.keys())
        for (i, j) in self.quadratic_terms.keys():
            if i >= j:
                raise ValueError(f"Índices cuadráticos deben cumplir i < j, recibido: ({i}, {j})")
            all_indices.update([i, j])
        
        if len(all_indices) == 0:
            raise ValueError("QUBO vacío: sin términos lineales ni cuadráticos")
        
        # Verificar que coeficientes sean números finitos
        for i, coeff in self.linear_terms.items():
            if not np.isfinite(coeff):
                raise ValueError(f"Coeficiente no finito en índice {i}: {coeff}")
        
        for (i, j), coeff in self.quadratic_terms.items():
            if not np.isfinite(coeff):
                raise ValueError(f"Coeficiente no finito en acoplamiento ({i},{j}): {coeff}")
    
    @property
    def num_qubits(self) -> int:
        """Número de qubits en el problema."""
        all_indices = set(self.linear_terms.keys())
        for (i, j) in self.quadratic_terms.keys():
            all_indices.update([i, j])
        return max(all_indices) + 1 if all_indices else 0
    
    @property
    def num_interactions(self) -> int:
        """Número de acoplamientos cuadrático (términos J_{ij})."""
        return len(self.quadratic_terms)
    
    @property
    def sparsity(self) -> float:
        """Sparsitud: qué fracción de posibles acoplamientos están presentes."""
        max_interactions = self.num_qubits * (self.num_qubits - 1) // 2
        if max_interactions == 0:
            return 0.0
        return self.num_interactions / max_interactions


# ============================================================================
# RESULTADOS DE RECOCIDO CUÁNTICO
# ============================================================================

@dataclass(frozen=True)
class AnnealingResult:
    """
    Resultado devuelto por un recocidor cuántico (D-Wave) o simulador.
    
    Attributes:
        best_state: Dict{ qubit: value } con solución óptima encontrada
        lowest_energy: Energía mínima alcanzada (E_ground o cerca)
        num_occurrences: Frecuencia de esta solución (de 1 a num_reads)
        timing_qpu_microseconds: Tiempo de ejecución QPU (microsegundos)
        is_ground_state_confident: ¿Es probablemente el ground state verdadero?
        chain_break_fraction: Fracción de chains rotos (0-1, 0=perfecto)
    """
    best_state: Dict[int, int]
    lowest_energy: float
    num_occurrences: int
    timing_qpu_microseconds: float = 0.0
    is_ground_state_confident: bool = False
    chain_break_fraction: float = 0.0
    
    def __post_init__(self):
        if len(self.best_state) == 0:
            raise ValueError("best_state no puede estar vacío")
        
        # Verificar que valores sean binarios
        for qubit, value in self.best_state.items():
            if value not in [0, 1]:
                raise ValueError(f"Valor de qubit {qubit} debe ser 0 ó 1, recibido: {value}")
        
        if self.num_occurrences <= 0:
            raise ValueError("num_occurrences debe ser positivo")
        
        if not np.isfinite(self.lowest_energy):
            raise ValueError("lowest_energy debe ser finita")
        
        if not (0 <= self.chain_break_fraction <= 1):
            raise ValueError("chain_break_fraction debe estar en [0, 1]")
    
    @property
    def num_qubits_active(self) -> int:
        """Número de qubits que participan en la solución."""
        return len(self.best_state)
    
    @property
    def hamming_weight(self) -> int:
        """Número de qubits a estado 1 (Hamming weight)."""
        return sum(self.best_state.values())
    
    @property
    def reliability_score(self) -> float:
        """
        Puntuación de confiabilidad de resultado (0-1).
        Combina: confianza en ground state + ausencia de chain breaks.
        """
        gs_factor = 0.8 if self.is_ground_state_confident else 0.3
        chain_penalty = 1.0 - self.chain_break_fraction
        return gs_factor * chain_penalty


# ============================================================================
# PLANTILLAS DE SEÑALES
# ============================================================================

@dataclass(frozen=True)
class TemplateSignal:
    """
    Plantilla de forma de onda para template matching.
    
    Una plantilla es un patrón conocido de onda gravitacional
    que se compara con la señal observada para detección/clasificación.
    
    Attributes:
        template_id: Identificador único
        strain_data: Array 1D de amplitudes (en tiempo o frecuencia)
        theory_family: TheoryFamily que genera esta plantilla
        parameters: Dict con parámetros generativos (M_c, a, etc)
        normalization: Norma L2 de la plantilla (para inner product)
    """
    template_id: str
    strain_data: np.ndarray
    theory_family: str  # ej: "GR", "ScalarTensor", "LoopQuantumGravity"
    parameters: Dict[str, float]
    normalization: float
    
    def __post_init__(self):
        if len(self.strain_data) == 0:
            raise ValueError("strain_data no puede estar vacío")
        
        if not np.all(np.isfinite(self.strain_data)):
            raise ValueError("strain_data contiene NaN o infinito")
        
        if self.normalization <= 0:
            raise ValueError("Normalización debe ser positiva")


# ============================================================================
# CONFIGURACIÓN DE CIRCUITO CUÁNTICO
# ============================================================================

@dataclass(frozen=True)
class QuantumCircuitConfig:
    """
    Configuración técnica para ejecutar un circuito cuántico.
    
    Parámetros que especifican cómo ejecutar el VQC:
    - Backend (simulador, QPU real)
    - Número de shots
    - Mitigación de errores
    - Parámetros de transpilación
    
    Attributes:
        backend_name: 'simulator', 'real_ibm', 'real_dwave', 'neal_simulator'
        num_shots: Número de ejecuciones (default: 1024)
        error_mitigation: Tipo de mitigación ('none', 'zne', 'pec')
        optimization_level: Transpiler optimization (0-3)
        seed_estimator: Semilla para reproducibilidad
    """
    backend_name: str
    num_shots: int = 1024
    error_mitigation: str = "none"
    optimization_level: int = 2
    seed_estimator: Optional[int] = None
    
    def __post_init__(self):
        if self.backend_name not in ["simulator", "real_ibm", "real_dwave", "neal_simulator"]:
            raise ValueError(f"Backend desconocido: {self.backend_name}")
        
        if self.num_shots < 100 or self.num_shots > 100000:
            raise ValueError("num_shots debe estar en [100, 100000]")
        
        if self.error_mitigation not in ["none", "zne", "pec"]:
            raise ValueError(f"Error mitigation desconocida: {self.error_mitigation}")
        
        if self.optimization_level not in [0, 1, 2, 3]:
            raise ValueError("optimization_level debe estar en [0, 3]")
    
    @property
    def is_real_hardware(self) -> bool:
        """¿Se ejecuta en hardware real?"""
        return "real" in self.backend_name.lower()
    
    @property
    def is_simulator(self) -> bool:
        """¿Se ejecuta en simulador?"""
        return "simulator" in self.backend_name.lower() or "neal" in self.backend_name.lower()
