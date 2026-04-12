import os
import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from src.application.hybrid_orchestrator import HybridInferenceOrchestrator
from src.application.validation_service import ValidationService
from src.infrastructure.storage.quantum_dataloader import QuantumDatasetLoader

def generate_full_tfm_report():
    print("📊 --- GENERADOR DE REPORTES QNIM ---")
    
    orchestrator = HybridInferenceOrchestrator(use_real_ibm=False)
    
    print("🧠 Configurando topología de 12 cúbits y cargando modelo...")
    fmap = ZZFeatureMap(12, reps=1)
    weights_path = "models/qnim_vqc_weights.npy"
    
    if os.path.exists(weights_path):
        loaded_weights = np.load(weights_path)
        orchestrator.vqc_engine.initialize_classifier(fmap, initial_weights=loaded_weights)
        
        # --- EL TRUCO (DUMMY FIT) ---
        # Engañamos a Qiskit para que active su bandera interna de "modelo entrenado"
        fake_X = np.zeros((1, 12))
        fake_y = np.array([[1, 0]])
        orchestrator.vqc_engine.vqc.fit(fake_X, fake_y)
        # ---------------------------
        
    else:
        print("❌ Error: No se encontró el cerebro entrenado en models/.")
        return
    
    # Datos de prueba (Test Set dummy)
    test_X = np.random.normal(0, 1, (10, 12)) 
    test_y = np.array([[1,0]]*5 + [[0,1]]*5)   # 5 RG, 5 LQG
    
    os.makedirs("reports/figures", exist_ok=True)
    validator = ValidationService(orchestrator)
    report = validator.run_confusion_assessment(test_X, test_y)
    
    print("\n📝 RESUMEN DE PRECISIÓN:")
    print(report)

if __name__ == "__main__":
    generate_full_tfm_report()