import os, sys, pandas as pd
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
load_dotenv()

from src.infrastructure.h5_gw_repository import H5GWRepository
from src.infrastructure.qiskit_classifier import QiskitClassifier
from src.application.process_event_use_case import ProcessEventUseCase
from src.application.signal_preprocessing_service import SignalPreprocessingService
from src.application.quantum_mapping_service import QuantumMappingService
from src.application.anomaly_generator_service import AnomalyGeneratorService

def run_sweep():
    repo = H5GWRepository(".")
    classifier = QiskitClassifier(n_qubits=8, iterations=10)
    use_case = ProcessEventUseCase(repo, classifier, SignalPreprocessingService(), 
                                   QuantumMappingService(8), AnomalyGeneratorService())
    
    epsilons = [0.0, 0.05, 0.1, 0.2, 0.5]
    results = []
    
    for eps in epsilons:
        _, res_lqg = use_case.execute_comparison("H1", epsilon=eps)
        results.append({"epsilon": eps, "prob": res_lqg.probabilities[0], "std": res_lqg.metadata["std_dev"]})
        print(f"Epsilon {eps} analizado.")
    
    pd.DataFrame(results).to_csv("resultados_robustez.csv", index=False)
    print("✅ Barrido completado. CSV generado.")

if __name__ == "__main__":
    run_sweep()