from pathlib import Path
from src.domain.astrophysics.sstg.engine import QuantumUniverseEngine
from src.domain.astrophysics.value_objects import TheoryFamily

class SyntheticSignalGenerationService:
    """
    Caso de Uso: Generar un dataset masivo de ondas gravitacionales.
    Orquesta el motor del dominio y la persistencia en disco.
    """
    def __init__(self, exporter_port):
        self.engine = QuantumUniverseEngine(sample_rate=4096)
        self.exporter = exporter_port # Puerto hacia la infraestructura (HDF5)

    def generate_balanced_dataset(self, total_events: int = 200) -> str:
        """
        Genera un dataset 50% Relatividad General y 50% Gravedad Cuántica.
        Asegura eventos Golden (alta SNR) para el entrenamiento cuántico.
        """
        samples = []
        half = total_events // 2
        
        for i in range(total_events):
            # 1. Definir los parámetros base (Golden events: distancias cortas)
            m1, m2 = 35.0, 30.0 # Masas estelares típicas
            distance = 400.0    # Alta SNR
            
            # 2. Elegir la teoría (50/50 balance)
            target_theory = TheoryFamily.GENERAL_RELATIVITY if i < half else TheoryFamily.LOOP_QUANTUM_GRAVITY
            
            # 3. El Dominio sintetiza la onda cruzando las 7 capas
            strain = self.engine.synthesize_event(m1, m2, distance, target_theory)
            
            # 4. Empaquetar para guardar
            samples.append({
                "strain": strain,
                "label": target_theory.value,
                "metadata": {"m1": m1, "m2": m2, "distance": distance}
            })
            
        # 5. La infraestructura guarda los archivos
        output_path = self.exporter.save_batch(samples)
        return output_path