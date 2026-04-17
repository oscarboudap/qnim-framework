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
    
    def sample_qubo(self, Q: dict, num_reads: int = 100) -> AnnealingResult:
        # Inicializamos el sampler de D-Wave
        sampler = neal.SimulatedAnnealingSampler()
        
        # Ejecutamos la búsqueda del estado fundamental (Ground State)
        response = sampler.sample_qubo(Q, num_reads=num_reads)
        
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