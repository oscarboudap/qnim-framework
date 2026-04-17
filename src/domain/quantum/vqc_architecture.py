# src/domain/quantum/vqc_architecture.py
from src.domain.quantum.entities import VQCTopology

class QNIMQuantumCircuit:
    """
    Definición del Ansatz y Feature Map del clasificador híbrido.
    Centraliza las decisiones topológicas para justificar la mitigación 
    de Barren Plateaus en arquitecturas de 12 a 27 cúbits.
    """
    
    @staticmethod
    def get_standard_topology() -> VQCTopology:
        """
        Devuelve la topología validada experimentalmente en el TFM.
        12 cúbits es el equilibrio óptimo entre expresividad del espacio 
        de Hilbert y evitación de ruido NISQ excesivo.
        """
        return VQCTopology(
            num_qubits=12,
            feature_map_reps=1,  # ZZFeatureMap repeticiones
            ansatz_reps=2,       # RealAmplitudes repeticiones
            entanglement_strategy='linear' # Mitiga la profundidad del circuito
        )