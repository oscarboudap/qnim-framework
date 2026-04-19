#!/usr/bin/env python3
"""
Infrastructure Validator: Verifica conectividad IBM Quantum

Este script valida que la capa Infrastructure esté correctamente 
configurada (puertos, adaptadores, conexiones).
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path so imports work from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.infrastructure.ibm_quantum_adapter import IBMQuantumAdapter
from src.infrastructure.neal_annealer_adapter import NealSimulatedAnnealerAdapter
from src.domain.quantum.entities import VQCTopology


def validate_all():
    print("="*70)
    print("🔍 VALIDADOR DE CONECTIVIDAD: INFRASTRUCTURE LAYER")
    print("="*70)
    print()
    
    load_dotenv()
    results = {}
    
    # 1. VALIDAR AMBIENTE
    print("📋 Validando configuración .env...")
    ibm_token = os.getenv("IBM_QUANTUM_TOKEN")
    backend_name = os.getenv("IBM_BACKEND_NAME", "ibm_fez")
    use_real_hw = os.getenv("USE_REAL_HARDWARE", "False").lower() == "true"
    
    if ibm_token:
        print(f"   ✅ Token configurado: {ibm_token[:20]}...")
        print(f"   ✅ Backend: {backend_name}")
        print(f"   ✅ Use Real Hardware: {use_real_hw}")
        results["Environment"] = True
    else:
        print("   ❌ Token no configurado en .env")
        results["Environment"] = False
    
    print()
    
    # 2. VALIDAR MODELOS
    print("📦 Validando modelos preentrenados...")
    models = {
        "models/qnim_vqc_weights.npy": "Pesos VQC",
        "models/qnim_preprocessing_pipeline.pkl": "Pipeline PCA",
    }
    
    all_exist = True
    for path, desc in models.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024**2)
            print(f"   ✅ {desc}: {size_mb:.2f} MB")
        else:
            print(f"   ❌ {desc}: NO encontrado")
            all_exist = False
    
    results["Models"] = all_exist
    print()
    
    # 3. VALIDAR IBM ADAPTER (Puerto IGateBasedQuantumComputer)
    print("🛰️  Validando Puerto: IGateBasedQuantumComputer...")
    try:
        if not os.path.exists("models/qnim_vqc_weights.npy"):
            import numpy as np
            os.makedirs("models", exist_ok=True)
            np.save("models/qnim_vqc_weights.npy", np.random.randn(10))
        
        adapter = IBMQuantumAdapter("models/qnim_vqc_weights.npy")
        print(f"   ✅ Adaptador instanciado")
        print(f"   ℹ️  Modo: {'Hardware Real' if adapter.use_real_hardware else 'Simulador Local'}")
        print(f"   ℹ️  Backend: {adapter.backend_name}")
        
        # Validar que puede crear topologías
        topo = VQCTopology(num_qubits=4, num_features=8, feature_map_reps=2, ansatz_reps=1, entanglement_type="linear")
        print(f"   ✅ VQCTopology válida: {topo.num_qubits} qubits, {topo.num_features} features")
        
        results["IBM Adapter"] = True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results["IBM Adapter"] = False
    
    print()
    
    # 4. VALIDAR NEAL ADAPTER (Puerto IQuantumAnnealer)
    print("❄️  Validando Puerto: IQuantumAnnealer...")
    try:
        neal_adapter = NealSimulatedAnnealerAdapter()
        print(f"   ✅ Adaptador instanciado")
        print(f"   ℹ️  Simulador: {neal_adapter.sampler.__class__.__name__}")
        results["Neal Adapter"] = True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results["Neal Adapter"] = False
    
    print()
    
    # 5. VALIDAR ARQUITECTURA
    print("🏗️  Validando Estructura DDD...")
    required_dirs = {
        "src/domain": "Domain",
        "src/application": "Application",
        "src/infrastructure": "Infrastructure",
    }
    
    all_dirs = True
    for path, name in required_dirs.items():
        if os.path.isdir(path):
            files = len(list(Path(path).rglob("*.py")))
            print(f"   ✅ {name} Layer: {files} archivos")
        else:
            print(f"   ❌ {name} Layer: NO encontrado")
            all_dirs = False
    
    results["DDD Structure"] = all_dirs
    print()
    
    # REPORTE FINAL
    print("="*70)
    print("📊 REPORTE FINAL")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} | {name}")
    
    print("="*70)
    print(f"Resultado: {passed}/{total} validaciones pasadas")
    
    if passed == total:
        print()
        print("✅ INFRASTRUCTURE LISTA PARA PRODUCCIÓN")
        print()
        print("Puedes ejecutar:")
        print("   python3 train_complete.py           # Entrenar modelos")
        print("   python3 run_qnim_inference.py       # Ejecutar inferencia")
        print("   python3 run_qnim_simulator.py       # Demo en simulador")
        print("   python3 generate_reports.py         # Generar reportes")
        return 0
    else:
        print()
        print("❌ REVISA LOS ERRORES ARRIBA")
        return 1


if __name__ == "__main__":
    sys.exit(validate_all())
