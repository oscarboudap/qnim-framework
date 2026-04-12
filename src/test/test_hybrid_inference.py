import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 1. Configuración del Path para reconocer 'src'
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import h5py
from src.infrastructure.storage.quantum_dataloader import QuantumDatasetLoader
from src.application.hybrid_orchestrator import HybridInferenceOrchestrator

def run_hybrid_diagnostic():
    load_dotenv()  # Carga IBM_QUANTUM_TOKEN, IBM_BACKEND_NAME, etc.
    
    print("🚀 --- QNIM: INICIANDO DIAGNÓSTICO CUÁNTICO HÍBRIDO ---")
    
    # --- CONFIGURACIÓN DE HARDWARE ---
    token = os.getenv("IBM_QUANTUM_TOKEN")
    backend = os.getenv("IBM_BACKEND_NAME", "ibm_kingston")
    # Si en tu .env pones USE_REAL_HARDWARE=True, irá a la QPU real. 
    # Si es False, usará el modelo de ruido en local.
    use_real = os.getenv("USE_REAL_HARDWARE", "False").lower() == "true"

    # 2. Instanciamos el orquestador con los nuevos argumentos sincronizados
    # Nota: No pasamos dwave_token porque usaremos simulación local Neal
    orchestrator = HybridInferenceOrchestrator(
        ibm_token=token, 
        ibm_backend=backend,
        use_real_ibm=use_real
    )

    loader = QuantumDatasetLoader(target_samples=64)

    # 3. Selección de un evento ciego del Bloque 1
    data_dir = project_root / "data" / "synthetic"
    last_batch = sorted([f for f in data_dir.iterdir() if f.is_dir()])[-1]
    sample_file = list(last_batch.glob("*.h5"))[0]
    
    print(f"📂 Cargando evento ciego: {sample_file.name}")

    # 4. Preprocesamiento
    quantum_ready_data = loader.prepare_for_quantum(str(sample_file))
    
    with h5py.File(sample_file, 'r') as f:
        # Recuperamos 'm1' y 'spin' (o 'a_star') para el espacio de fases
        true_meta = {
            "m1": f.attrs.get("true_m1", 35.0),
            "spin": f.attrs.get("true_spin", 0.0) or f.attrs.get("true_a_star", 0.0),
            "theory": f.attrs["true_theory"].decode('utf-8') if isinstance(f.attrs["true_theory"], bytes) else f.attrs["true_theory"]
        }

    # 5. EJECUCIÓN DEL ORQUESTADOR
    diagnosis = orchestrator.run_full_diagnosis(
        {'quantum_signal': quantum_ready_data, 'metadata': true_meta}
    )

    # 6. REPORTE DE RESULTADOS
    print("\n" + "="*45)
    print("📊 REPORTE DE INFERENCIA QNIM")
    print("="*45)
    print(f"🔍 CLASIFICACIÓN (IBM VQC):")
    print(f"   > Predicción: {diagnosis['theory']}")
    print(f"   > Confianza:  {diagnosis['confidence']*100:.2f}%")
    print(f"   > Realidad:    {true_meta['theory']}")
    
    print("-" * 45)
    print(f"📏 REFINAMIENTO (D-WAVE QUBO):")
    print(f"   > Masa Estimada: {diagnosis['best_fit_mass']:.2f} M_sun")
    print(f"   > Masa Real:     {true_meta['m1']:.2f} M_sun")
    print(f"   > Precisión:     {diagnosis['dwave_precision']}")
    print("="*45)

if __name__ == "__main__":
    run_hybrid_diagnostic()