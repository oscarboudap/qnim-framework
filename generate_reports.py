import os
import numpy as np
import joblib
import random
from pathlib import Path
from qiskit.circuit.library import ZZFeatureMap
from src.application.hybrid_orchestrator import HybridInferenceOrchestrator
from src.application.validation_service import ValidationService
from src.infrastructure.storage.quantum_dataloader import QuantumDatasetLoader

def generate_full_tfm_report():
    print("📊 --- GENERADOR DE REPORTES QNIM (CON PIPELINE) ---")
    
    orchestrator = HybridInferenceOrchestrator(use_real_ibm=False)
    
    print("🧠 Configurando topología de 12 cúbits y cargando modelos...")
    fmap = ZZFeatureMap(12, reps=1)
    weights_path = "models/qnim_vqc_weights.npy"
    pipeline_path = "models/qnim_preprocessing_pipeline.pkl"
    
    # 1. CARGAMOS PESOS Y PIPELINE
    if os.path.exists(weights_path) and os.path.exists(pipeline_path):
        loaded_weights = np.load(weights_path)
        orchestrator.vqc_engine.initialize_classifier(fmap, initial_weights=loaded_weights)
        quantum_pipeline = joblib.load(pipeline_path)
        
        # --- EL TRUCO (DUMMY FIT) ---
        fake_X = np.zeros((1, 12))
        fake_y = np.array([[1, 0]])
        orchestrator.vqc_engine.vqc.fit(fake_X, fake_y)
        # ---------------------------
        
    else:
        print("❌ Error: Faltan los pesos o el modelo Pipeline en la carpeta models/.")
        return
    
    # 2. CARGAMOS DATOS DE PRUEBA REALES
    print("📂 Cargando datos de prueba reales para la validación...")
    loader = QuantumDatasetLoader(target_samples=16384) # Cargamos la onda entera
    data_dir = Path("data/synthetic")
    
    try:
        latest_batch = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("2026")])[-1]
    except IndexError:
        print("❌ Error: No hay datos sintéticos.")
        return
        
    test_X_raw, test_y = [], []
    
    # Mezclamos todos los archivos
    all_files = list(latest_batch.glob("*.h5"))
    random.seed(42) # Semilla fija para que tus gráficas sean reproducibles
    random.shuffle(all_files)
    
    # Extraemos 20 eventos mixtos del dataset para evaluar
    for file in all_files[:20]:
        test_X_raw.append(loader.prepare_for_quantum(str(file)))
        
        # --- EL TRUCO DEL NOMBRE DE ARCHIVO ---
        # file.stem saca "event_00150", split('_')[1] saca "00150", int() lo hace 150
        file_num = int(file.stem.split('_')[1])
        
        if file_num < 100:
            test_y.append([1, 0]) # RG (Kerr)
        else:
            test_y.append([0, 1]) # LQG (Anomalía Cuántica)
        # --------------------------------------
                
    test_X_raw = np.array(test_X_raw)
    test_y = np.array(test_y)
    
    # 3. PREPROCESAMIENTO CUÁNTICO
    print(f"🗜️ Aplicando Pipeline Cuántico a {len(test_X_raw)} datos de prueba...")
    test_X = quantum_pipeline.transform(test_X_raw)
    
    # 4. VALIDACIÓN Y MATRIZ DE CONFUSIÓN
    os.makedirs("reports/figures", exist_ok=True)
    validator = ValidationService(orchestrator)
    report = validator.run_confusion_assessment(test_X, test_y)
    
    print("\n📝 RESUMEN DE PRECISIÓN:")
    print(report)

if __name__ == "__main__":
    generate_full_tfm_report()