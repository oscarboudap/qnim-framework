import os
import numpy as np
from src.application.hybrid_orchestrator import HybridInferenceOrchestrator
from src.application.validation_service import ValidationService
from src.infrastructure.storage.quantum_dataloader import QuantumDatasetLoader

def generate_full_tfm_report():
    print("📊 --- GENERADOR DE REPORTES QNIM ---")
    
    # 1. Cargar el orquestador con el cerebro ya entrenado
    orchestrator = HybridInferenceOrchestrator(use_real_ibm=False)
    
    # 2. Cargar datos de test (usaremos los mismos para el ejemplo)
    loader = QuantumDatasetLoader(target_samples=64)
    # Aquí cargarías una carpeta diferente de 'test' si la tuvieras
    test_X = np.random.normal(0, 1, (10, 12)) # Dummy test data
    test_y = np.array([[1,0]]*5 + [[0,1]]*5)   # 5 RG, 5 Anomalías
    
    # 3. Generar Matriz de Confusión
    os.makedirs("reports/figures", exist_ok=True)
    validator = ValidationService(orchestrator)
    report = validator.run_confusion_assessment(test_X, test_y)
    
    print("\n📝 RESUMEN DE PRECISIÓN:")
    print(report)

if __name__ == "__main__":
    generate_full_tfm_report()