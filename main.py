import os
import sys
from dotenv import load_dotenv

# Añadimos la raíz al path para que reconozca el paquete 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

load_dotenv()

from src.infrastructure.h5_gw_repository import H5GWRepository
from src.infrastructure.qiskit_classifier import QiskitClassifier
from src.application.process_event_use_case import ProcessEventUseCase
from src.application.signal_preprocessing_service import SignalPreprocessingService
from src.application.quantum_mapping_service import QuantumMappingService
from src.application.anomaly_generator_service import AnomalyGeneratorService
from src.presentation.cli_presenter import CLIPresenter

def main():
    CLIPresenter.show_welcome()
    
    # Inyección de dependencias (Infrastructure)
    repo = H5GWRepository(base_path=".")
    classifier = QiskitClassifier(n_qubits=8, iterations=5)
    
    # Inyección de dependencias (Application Services)
    preprocessor = SignalPreprocessingService()
    mapper = QuantumMappingService(target_qubits=8)
    anomaly_gen = AnomalyGeneratorService()
    
    # Orquestador del Caso de Uso
    use_case = ProcessEventUseCase(
        repository=repo,
        classifier=classifier,
        preprocessor=preprocessor,
        mapper=mapper,
        anomaly_generator=anomaly_gen
    )

    try:
        print("[INFO] Analizando evento sintético con ruido de hardware real...")
        res_rg, res_lqg = use_case.execute_comparison(detector_name="H1", epsilon=0.1)
        
        print("\n--- RESULTADOS ---")
        print(f"Prob. Relatividad General: {res_rg.probabilities[0]:.4f}")
        print(f"Prob. Gravedad Cuántica:   {res_lqg.probabilities[0]:.4f}")
        print(f"Desviación Detectada (Δ):  {abs(res_rg.probabilities[0] - res_lqg.probabilities[0]):.4f}")
        
    except Exception as e:
        print(f"❌ Error en el sistema: {e}")

if __name__ == "__main__":
    main()