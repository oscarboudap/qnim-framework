#!/usr/bin/env python3
"""
Presentation Layer: Generate Validation Reports

Orquesta la generación de reportes usando ValidationService (Application).
No hay lógica de negocio aquí - solo coordinación.
"""

import os
import sys
import numpy as np
import joblib
import random
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path so imports work from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

# Application: ValidationService
from src.application.validation_service import ValidationService

# Infrastructure: Adaptadores
from src.infrastructure.ibm_quantum_adapter import IBMQuantumAdapter
from src.infrastructure.neal_annealer_adapter import NealSimulatedAnnealerAdapter
from src.infrastructure.storage.quantum_dataloader import QuantumDatasetLoader

# Application: Orquestador
from src.application.hybrid_orchestrator import HybridInferenceOrchestrator


def main():
    print("="*70)
    print("📊 GENERADOR DE REPORTES QNIM")
    print("="*70)
    print()
    
    load_dotenv()
    os.makedirs("reports/validation", exist_ok=True)
    
    # 1. VERIFICAR MODELOS (Infrastructure prerequisite)
    print("📦 Verificando modelos preentrenados...")
    weights_path = "models/qnim_vqc_weights.npy"
    pipeline_path = "models/qnim_preprocessing_pipeline.pkl"
    
    if not os.path.exists(weights_path) or not os.path.exists(pipeline_path):
        print(f"   ❌ Faltan modelos en {weights_path} o {pipeline_path}")
        print("   Ejecuta primero: python3 train_complete.py")
        return 1
    
    print(f"   ✅ Modelos encontrados")
    print()
    
    # 2. CARGAR ADAPTADORES (Infrastructure layer)
    print("🔌 Inicializando adaptadores...")
    try:
        ibm_adapter = IBMQuantumAdapter(weights_path)
        dwave_adapter = NealSimulatedAnnealerAdapter()
        print("   ✅ Adaptadores listos")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 1
    
    print()
    
    # 3. CREAR ORQUESTADOR (Application layer)
    print("🧮 Creando orquestador híbrido...")
    orchestrator = HybridInferenceOrchestrator(ibm_adapter, dwave_adapter)
    print("   ✅ Orquestador listo")
    print()
    
    # 4. CREAR VALIDATOR (Application layer)
    print("✓ Inicializando ValidationService...")
    validator = ValidationService(orchestrator)
    print()
    
    # 5. CARGAR DATOS DE PRUEBA (Infrastructure)
    print("📂 Cargando datos de prueba...")
    loader = QuantumDatasetLoader(target_samples=16384)
    data_dir = Path("data/synthetic")
    
    try:
        latest_batch = sorted([d for d in data_dir.iterdir() 
                              if d.is_dir() and d.name.startswith("202")])[-1]
    except IndexError:
        print("   ❌ No hay datos sintéticos en data/synthetic/")
        return 1
    
    # Cargar 20 archivos aleatorios
    all_files = list(latest_batch.glob("*.h5"))
    random.seed(42)
    random.shuffle(all_files)
    
    test_X_raw, test_y = [], []
    for file in all_files[:20]:
        test_X_raw.append(loader.prepare_for_quantum(str(file)))
        
        # Labels generados por convenio de nombres de archivo
        file_num = int(file.stem.split('_')[1])
        test_y.append([1, 0] if file_num < 100 else [0, 1])
    
    test_X_raw = np.array(test_X_raw)
    test_y = np.array(test_y)
    
    print(f"   ✅ {len(test_X_raw)} eventos cargados")
    print()
    
    # 6. PREPROCESAMIENTO (Infrastructure)
    print("🗜️  Aplicando pipeline cuántico...")
    pipeline = joblib.load(pipeline_path)
    test_X = pipeline.transform(test_X_raw)
    print(f"   ✅ Transformados a {test_X.shape[1]} dimensiones")
    print()
    
    # 7. EJECUTAR VALIDACIÓN (Application use case)
    print("🔍 Ejecutando validación...")
    report = validator.run_confusion_assessment(test_X, test_y)
    
    print()
    print("📝 RESULTADOS:")
    print("-"*70)
    print(report)
    print("-"*70)
    print()
    
    print("="*70)
    print("✅ REPORTES GENERADOS")
    print("="*70)
    print()
    print("Archivos guardados en:")
    print("  • reports/validation/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())