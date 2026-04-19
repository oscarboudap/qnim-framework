import neal
from src.domain.quantum.interfaces import IQuantumAnnealer
from src.domain.quantum.entities import AnnealingResult

class NealSimulatedAnnealerAdapter(IQuantumAnnealer):
    """
    Adaptador de Infraestructura para el Recocido Cuántico.
    Usa el simulador local de D-Wave (Neal). 
    Diseño Hot-Swap: Para usar la QPU real, solo habría que cambiar el 'SimulatedAnnealingSampler' 
    por 'DWaveSampler(token=...)' en este único archivo.
    """
    
    def __init__(self):
        """Inicializa el sampler de Neal"""
        self.sampler = neal.SimulatedAnnealingSampler()
    
    def sample_qubo(self, Q: dict, num_reads: int = 100) -> AnnealingResult:
        # Ejecutamos la búsqueda del estado fundamental (Ground State)
        response = self.sampler.sample_qubo(Q, num_reads=num_reads)
        
        # Extraemos el mejor resultado
        best_sample = response.first.sample
        lowest_energy = response.first.energy
        occurrences = response.first.num_occurrences
        
        # Si el estado fundamental se ha encontrado muchas veces, tenemos alta confianza
        is_confident = occurrences >= (num_reads * 0.1)
        
        return AnnealingResult(
            best_state=best_sample,
            lowest_energy=lowest_energy,
            num_occurrences=occurrences,
            is_ground_state_confident=is_confident
        )
    
    def get_embedding_time(self, num_qubits: int) -> float:
        """
        Tiempo estimado para embedding en simulador (siempre rápido).
        
        Args:
            num_qubits: Número de qubits lógicos
            
        Returns:
            Tiempo estimado en microsegundos
        """
        # Simulador local: tiempo negligible, solo modelamos overhead
        return 10.0 + (num_qubits * 0.5)  # ~10-100 microsegundos
    
    def get_native_graph_topology(self) -> dict:
        """
        Retorna topología nativa (simulador no tiene restricciones).
        
        Returns:
            Dict completamente conectado para 8 qubits (como simplificación)
        """
        # Simulador: asumimos conectividad completa
        num_qubits = 8
        topology = {}
        for i in range(num_qubits):
            topology[i] = [j for j in range(num_qubits) if i != j]
        return topology