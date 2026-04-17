import numpy as np
import joblib
from src.domain.astrophysics.value_objects import TheoryFamily
from src.domain.quantum.interfaces import IQuantumAnnealer, IGateBasedQuantumComputer
from src.domain.quantum.qubo_formulator import TemplateMatchingQUBO
from src.domain.quantum.vqc_architecture import QNIMQuantumCircuit

class HybridInferenceOrchestrator:
    """
    Servicio de Aplicación que coordina las dos ramas cuánticas (IBM y D-Wave).
    Recibe puertos abstractos (interfaces), por lo que cumple Inversión de Dependencias (SOLID).
    """
    
    def __init__(self, ibm_backend: IGateBasedQuantumComputer, dwave_backend: IQuantumAnnealer):
        self.ibm = ibm_backend
        self.dwave = dwave_backend
        self.topology = QNIMQuantumCircuit.get_standard_topology()

    def execute_dwave_branch(self, target_signal: np.ndarray, search_space_templates: list) -> dict:
        """
        Rama de Recocido Cuántico (Capa 2: Geometría Intrínseca).
        Busca los parámetros clásicos (Masa, Espín) minimizando la energía de un QUBO.
        """
        # 1. El Dominio formula las matemáticas
        qubo_problem = TemplateMatchingQUBO.build_formulation(target_signal, search_space_templates)
        
        # 2. La Infraestructura (D-Wave) lo resuelve
        result = self.dwave.sample_qubo(Q=qubo_problem.quadratic_terms, num_reads=100)
        
        # 3. Decodificamos el resultado
        best_idx = max(result.best_state, key=result.best_state.get)
        best_match = search_space_templates[best_idx]
        
        return best_match['params'] # Devuelve {m1, m2, spin}

    def execute_ibm_branch(self, compressed_features: np.ndarray) -> dict:
        """
        Rama Basada en Puertas (Capas 5 y 6: Anomalías).
        Inyecta las 12 características en el circuito VQC para detectar física más allá de GR.
        """
        # La ejecución del circuito la delega a la infraestructura (QiskitRuntime o Simulador)
        prediction_probabilities = self.ibm.execute_circuit(
            circuit_topology=self.topology, 
            features=compressed_features
        )
        
        # Decodificación (0: RG, 1: Anomalía)
        anomaly_prob = float(prediction_probabilities[0][1])
        
        return {
            "detected_theory": TheoryFamily.LOOP_QUANTUM_GRAVITY if anomaly_prob > 0.5 else TheoryFamily.GENERAL_RELATIVITY,
            "anomaly_confidence": anomaly_prob
        }