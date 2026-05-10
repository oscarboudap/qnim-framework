"""
Application Layer: Port Interfaces (Hexagonal Architecture)
===========================================================

Define contratos abstractos para adaptadores de infraestructura.

La application layer DEPENDE DE ESTAS INTERFACES, no de implementaciones concretas.
La infraestructura las implementa (D-Wave, Qiskit, HDF5, etc.).

Patrón Hexagonal: inside boundary = domain + application + ports
                 outside boundary = infrastructure adapter implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np


# ============================================================================
# DATA ACCESS PORTS
# ============================================================================

class IDataLoaderPort(ABC):
    """
    Puerto para cargar datos de eventos gravitacionales.
    
    Abstrae:
    - Lectura HDF5 / LIGO data
    - Preprocesamiento (normalización)
    - Inyección en pipeline cuántico
    """
    
    @abstractmethod
    def prepare_for_quantum(self, file_path: str) -> np.ndarray:
        """
        Carga y prepara una señal para la rama IBM (VQC).
        
        Args:
            file_path: Ruta a archivo HDF5
            
        Returns:
            Array 1D con la señal [16384] elementos, normalizada en [-π, π]
            
        Raises:
            FileNotFoundError: Si archivo no existe
            ValueError: Si formato es inválido
        """
        raise NotImplementedError


class IStoragePort(ABC):
    """
    Puerto para persistencia de datos generados (dataset sintético).
    """
    
    @abstractmethod
    def save_batch(self, samples: List[Dict]) -> str:
        """
        Guarda un lote de eventos sintéticos en disco.
        
        Args:
            samples: Lista de {"strain": np.ndarray, "label": str, "metadata": dict}
            
        Returns:
            Ruta del directorio creado (ej: 'data/synthetic/20260419-141715/')
        """
        raise NotImplementedError
    
    @abstractmethod
    def load_batch(self, batch_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carga un lote previousmente guardado.
        
        Returns:
            (X, y) arrays de características y labels
        """
        raise NotImplementedError


# ============================================================================
# QUANTUM OPTIMIZATION PORTS
# ============================================================================

class IQuantumOptimizerPort(ABC):
    """
    Adaptador genérico para optimizadores cuánticos (D-Wave, anealadores).
    
    La application llama este puerto, que delega a infraestructura específica.
    """
    
    @abstractmethod
    def solve_qubo(self, 
                   linear_terms: Dict[int, float],
                   quadratic_terms: Dict[Tuple[int, int], float],
                   num_reads: int = 100) -> Dict[int, int]:
        """
        Resuelve un problema QUBO usando el backend cuántico.
        
        Args:
            linear_terms: {variable_idx: coeff}
            quadratic_terms: {(i, j): coeff}
            num_reads: Número de lecturas/muestras
            
        Returns:
            Solution dict: {variable_idx: value (0 o 1)}
            
        Raises:
            ValueError: Si QUBO es inválido
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_annealing_timing(self) -> float:
        """Retorna tiempo de recocido en microsegundos."""
        raise NotImplementedError


# ============================================================================
# QUANTUM ML TRAINING PORTS
# ============================================================================

class IQuantumMLTrainerPort(ABC):
    """
    Adaptador para entrenar modelos cuánticos (VQC, QNN).
    
    Abstrae:
    - Feature mapping
    - Ansatz parametrizado
    - Optimización
    - File I/O de pesos
    """
    
    @abstractmethod
    def train_vqc(self,
                  X_train: np.ndarray,
                  y_train: np.ndarray,
                  num_qubits: int,
                  max_iterations: int = 100,
                  optimizer_name: str = "SPSA") -> Dict[str, any]:
        """
        Entrena un Variational Quantum Classifier.
        
        Args:
            X_train: Features [n_samples, n_features]
            y_train: Labels [n_samples, n_classes]
            num_qubits: Qubits disponibles
            max_iterations: Iteraciones del optimizador
            optimizer_name: "SPSA", "COBYLA", "ADAM"
            
        Returns:
            {
                "weights": np.ndarray,
                "training_loss": float,
                "validation_accuracy": float,
                "iterations": int,
                "execution_time_seconds": float
            }
        """
        raise NotImplementedError
    
    @abstractmethod
    def save_weights(self, weights: np.ndarray, path: str) -> None:
        """Guarda pesos del modelo en disco."""
        raise NotImplementedError
    
    @abstractmethod
    def load_weights(self, path: str) -> np.ndarray:
        """Carga pesos previamente guardados."""
        raise NotImplementedError


class IPreprocessingPort(ABC):
    """
    Adaptador para pipeline clásico (StandardScaler, PCA, Mapper).
    """
    
    @abstractmethod
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Ajusta y transforma datos de entrada."""
        raise NotImplementedError
    
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforma datos usando ajuste previo."""
        raise NotImplementedError
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Guarda el pipeline (escalador + PCA) en disco."""
        raise NotImplementedError
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Carga pipeline previamente guardado."""
        raise NotImplementedError


# ============================================================================
# REPORTING / VISUALIZATION PORTS
# ============================================================================

class IMetricsReporterPort(ABC):
    """
    Adaptador para reportes (abstracta visualización de presentation layer).
    
    Application retorna datos tipados (ConfusionMatrixData).
    Presentation los visualiza (con matplotlib, plotly, etc.).
    """
    
    @abstractmethod
    def report_confusion_matrix(self,
                               true_positives: int,
                               true_negatives: int,
                               false_positives: int,
                               false_negatives: int,
                               output_path: Optional[str] = None) -> None:
        """
        Genera reporte de matriz de confusión.
        
        Args:
            Conteos TP/TN/FP/FN
            output_path: Donde guardar (si None, solo en memoria)
        """
        raise NotImplementedError
    
    @abstractmethod
    def report_inference_trace(self,
                              event_id: str,
                              classic_params: Dict,
                              quantum_results: Dict,
                              execution_time_seconds: float) -> None:
        """Genera reporte de inferencia (trazabilidad)."""
        raise NotImplementedError


# ============================================================================
# SYNTHETIC DATA GENERATION PORTS
# ============================================================================

class ISyntheticDataGeneratorPort(ABC):
    """
    Adaptador para conectar SSTG (debe estar en infrastructure, no domain).
    """
    
    @abstractmethod
    def synthesize_event(self,
                        m1_solar_masses: float,
                        m2_solar_masses: float,
                        distance_mpc: float,
                        theory_family: str) -> np.ndarray:
        """
        Sintetiza una onda gravitacional bajo una teoría específica.
        
        Args:
            m1_solar_masses: Masa primaria [M_sol]
            m2_solar_masses: Masa secundaria [M_sol]
            distance_mpc: Distancia comóvil [Mpc]
            theory_family: "GR", "LQG", etc.
            
        Returns:
            Strain array [16384] elementos
        """
        raise NotImplementedError


# ============================================================================
# EVENT REPOSITORY PORT
# ============================================================================

class IQuantumDecodedEventRepository(ABC):
    """
    Repository para Aggregate Root: QuantumDecodedEvent.
    
    Gestiona persistencia de eventos con todas sus capas (1-7).
    """
    
    @abstractmethod
    def save(self, event) -> None:
        """Persiste un evento completo."""
        raise NotImplementedError
    
    @abstractmethod
    def find_by_id(self, event_id: str):
        """Recupera evento por ID."""
        raise NotImplementedError
    
    @abstractmethod
    def find_by_theory_verdict(self, theory: str) -> List:
        """Query: Encuentra eventos que concluyeron con teoría X."""
        raise NotImplementedError


# ============================================================================
# PROGRESS TRACKING PORT
# ============================================================================

class ITrainingProgressObserver(ABC):
    """
    Observer para trackear progreso de entrenamiento.
    
    Desacopla application de UI/logging specifics.
    """
    
    @abstractmethod
    def on_iteration_complete(self,
                             iteration: int,
                             total_iterations: int,
                             current_loss: float) -> None:
        """Callback cuando una iteración termina."""
        raise NotImplementedError
    
    @abstractmethod
    def on_training_complete(self, final_metrics: Dict) -> None:
        """Callback cuando el entrenamiento termina."""
        raise NotImplementedError
