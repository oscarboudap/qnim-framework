import numpy as np
import pandas as pd
import os
import joblib
from pathlib import Path
import sys
import os
from pathlib import Path

# Añadimos la raíz del proyecto al path de búsqueda de Python
root_path = str(Path(__file__).parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)
from src.domain.metrology.planck_error_bounds import PlanckErrorBounds
from src.infrastructure.ibm_quantum_adapter import IBMQuantumAdapter
from src.infrastructure.neal_annealer_adapter import NealSimulatedAnnealerAdapter
from src.application.hybrid_orchestrator import HybridInferenceOrchestrator
from src.infrastructure.storage.quantum_dataloader import QuantumDatasetLoader

def inject_noise(strain, target_snr):
    """
    Inyecta ruido blanco gaussiano para alcanzar un SNR específico.
    Matemáticamente: SNR = Power(Signal) / Power(Noise)
    """
    signal_power = np.sqrt(np.mean(strain**2))
    noise_volts = signal_power / target_snr
    noise = np.random.normal(0, noise_volts, len(strain))
    return strain + noise

def run_monte_carlo_validation():
    print("🧪 Iniciando Validación Estadística de Monte Carlo (QNIM)...")
    
    # 1. Setup de Infraestructura
    weights_path = "models/qnim_vqc_weights.npy"
    pipeline_path = "models/qnim_preprocessing_pipeline.pkl"
    
    ibm_adapter = IBMQuantumAdapter(weights_path)
    dwave_adapter = NealSimulatedAnnealerAdapter()
    orchestrator = HybridInferenceOrchestrator(ibm_adapter, dwave_adapter)
    loader = QuantumDatasetLoader(target_samples=16384)
    compressor = joblib.load(pipeline_path)

    # 2. Parámetros del Experimento
    # Probamos SNRs desde 5 (muy ruidoso) hasta 25 (muy limpio)
    noise_levels = [5, 10, 15, 20, 25]
    iterations_per_level = 10 # Para un TFM real, sube esto a 50 o 100 si tienes tiempo
    
    # Cargamos una pequeña muestra de test (10 eventos de RG y 10 de LQG)
    data_dir = Path("data/synthetic")
    latest_batch = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("2026")])[-1]
    all_files = list(latest_batch.glob("*.h5"))
    test_files = all_files[:iterations_per_level] + all_files[-iterations_per_level:]

    results = []

    for snr in noise_levels:
        print(f"\n📡 Evaluando robustez a SNR = {snr}")
        for i, file_path in enumerate(test_files):
            # Carga y degradación
            strain_pure = loader.prepare_for_quantum(str(file_path))
            noisy_strain = inject_noise(strain_pure, target_snr=snr)
            
            # Compresión Dimensional (PCA)
            features = compressor.transform([noisy_strain])
            
            # Inferencia Cuántica (Rama IBM)
            prediction = orchestrator.execute_ibm_branch(features)
            conf = prediction['anomaly_confidence']
            
            # Validación de Significancia (Dominio)
            # El ruido de fondo es inversamente proporcional al SNR
            stats = PlanckErrorBounds.calculate_discovery_significance(
                background_noise_level=(1.0/snr), 
                quantum_signal_strength=conf
            )
            
            # Etiqueta real (asumimos que la primera mitad son RG y la segunda LQG)
            true_label = "Kerr_RG" if i < iterations_per_level else "LQG_Quantum"
            
            results.append({
                "Iteration": i,
                "True_Label": true_label,
                "SNR": snr,
                "Anomaly_Confidence": conf,
                "Z_Score_Sigma": stats['sigma'],
                "P_Value": stats['p_value'],
                "Is_Discovery": stats['is_discovery']
            })
            print(f"  └─ Evento {i+1}/{len(test_files)}: {stats['sigma']} Sigma", end="\r")

    # 3. Exportación de Datos
    os.makedirs("reports", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("reports/resultados_robustez_qnim.csv", index=False)
    print("\n\n✅ Prueba de Monte Carlo completada.")
    print(f"📊 Promedio de Sigma a SNR 20: {df[df['SNR']==20]['Z_Score_Sigma'].mean():.2f}")

if __name__ == "__main__":
    run_monte_carlo_validation()