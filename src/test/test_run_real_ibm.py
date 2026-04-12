import os
import sys
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.application.hybrid_orchestrator import HybridInferenceOrchestrator
from src.infrastructure.storage.quantum_dataloader import QuantumDatasetLoader

def run_real_hardware_analysis():
    load_dotenv()
    token = os.getenv("IBM_QUANTUM_TOKEN")
    backend = os.getenv("IBM_BACKEND_NAME", "ibm_kingston")
    
    print("📢 --- QNIM: LANZAMIENTO A HARDWARE REAL (ZERO-SHOT TEST) ---")
    print(f"📡 Backend seleccionado: {backend}")

    # 1. Inicializamos el Orquestador en modo REAL
    orchestrator = HybridInferenceOrchestrator(
        ibm_token=token, 
        ibm_backend=backend,
        use_real_ibm=True 
    )

    # 2. Carga dinámica del evento
    loader = QuantumDatasetLoader(target_samples=64)
    data_dir = project_root / "data" / "synthetic"
    
    try:
        all_batches = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("2026")])
        latest_batch = all_batches[-1]
        sample_file = list(latest_batch.glob("*.h5"))[0]
        print(f"📂 Batch detectado: {latest_batch.name}")
    except Exception as e:
        print(f"❌ Error al localizar datos: {e}")
        return

    quantum_data = loader.prepare_for_quantum(str(sample_file))

    event_context = {
        'id': f'GW-REAL-{latest_batch.name[-6:]}',
        'quantum_signal': quantum_data,
        'metadata': {'m1': 50.36, 'spin': 0.0} 
    }

    # 3. EJECUCIÓN (Aquí actuará el Hot-Swap del Orquestador)
    print("\n🛰️  Iniciando orquestación híbrida...")
    diagnosis = orchestrator.run_full_diagnosis(event_context)

    # 4. REPORTE FINAL
    print("\n" + "⭐"*45)
    print("✅ RESULTADO OBTENIDO DESDE HARDWARE REAL")
    print("⭐"*45)
    print(f"Teoría Predicha: {diagnosis['theory']}")
    print(f"Confianza QPU:   {diagnosis.get('confidence', 0)*100:.2f}%")
    print(f"Masa (D-Wave):   {diagnosis['best_fit_mass']:.2f} M_sun")
    print(f"Backend:         {backend}")
    print("="*45)

if __name__ == "__main__":
    run_real_hardware_analysis()