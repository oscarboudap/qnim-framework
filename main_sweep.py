import os
import sys
import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
load_dotenv()

from src.infrastructure.h5_gw_repository import H5GWRepository
from src.infrastructure.qiskit_classifier import QiskitClassifier
from src.application.process_event_use_case import ProcessEventUseCase
from src.application.signal_preprocessing_service import SignalPreprocessingService
from src.application.quantum_mapping_service import QuantumMappingService
from src.application.anomaly_generator_service import AnomalyGeneratorService

def run_robustness_sweep():
    print("🔬 [QNIM-LAB] Iniciando Barrido de Sensibilidad Cuántica (Simulación con Ruido)...")
    
    # 1. Configuración: Forzamos Local para el barrido extenso
    base_path = os.path.dirname(os.path.abspath(__file__))
    repo = H5GWRepository(base_path=base_path)
    
    # Usamos el nuevo simulador con ruido de Kingston (8 qubits)
    classifier = QiskitClassifier(n_qubits=8, iterations=10) # 10 iteraciones por punto para estadística
    
    preprocessor = SignalPreprocessingService()
    mapper = QuantumMappingService(target_qubits=8)
    anomaly_gen = AnomalyGeneratorService()
    
    use_case = ProcessEventUseCase(repo, classifier, preprocessor, mapper, anomaly_gen)
    
    # 2. Definimos el rango de anomalías (Epsilon de 0.0 a 0.5)
    epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    results = []

    print(f"{'Epsilon':<10} | {'Confianza LQG':<15} | {'Std Dev':<10} | {'Delta (vs RG)':<10}")
    print("-" * 55)

    # Obtenemos la referencia de RG (Epsilon = 0) una sola vez
    res_rg, _ = use_case.execute_comparison("H1")
    prob_rg = res_rg.probabilities[0]

    for eps in epsilons:
        # Ejecutamos el caso de uso con el epsilon específico
        # Nota: Deberías modificar execute_comparison para aceptar epsilon, o hacerlo manual:
        raw_signal = repo.get_signal_by_detector("H1")
        clean_signal = preprocessor.clean_signal(raw_signal)
        
        # Generamos la anomalía con el epsilon del bucle
        anomalous_signal = anomaly_gen.apply_quantum_drift(clean_signal, epsilon=eps)
        
        data_lqg = mapper.prepare_for_embedding(anomalous_signal)
        res_lqg = classifier.predict(data_lqg)
        
        prob_lqg = res_lqg.probabilities[0]
        std_lqg = res_lqg.metadata.get("std_dev", 0)
        delta = prob_lqg - prob_rg
        
        results.append({
            "epsilon": eps,
            "prob_lqg": prob_lqg,
            "std_dev": std_lqg,
            "delta": delta
        })
        
        print(f"{eps:<10.2f} | {prob_lqg:<15.4f} | {std_lqg:<10.4f} | {delta:<10.4f}")

    # 3. Exportamos a CSV para gráficas en la memoria
    df = pd.DataFrame(results)
    df.to_csv("resultados_robustez_qnim.csv", index=False)
    print("\n✅ Barrido completado. Datos guardados en 'resultados_robustez_qnim.csv'")

if __name__ == "__main__":
    run_robustness_sweep()