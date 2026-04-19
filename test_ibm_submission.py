#!/usr/bin/env python3
"""
Quick Test: IBM Real Hardware Submission
- Tests if USE_REAL_HARDWARE=True triggers IBM connection
- Doesn't require lengthy VQC training
- Minimal setup just to test the flag behavior
"""
import os
import sys
import numpy as np
from dotenv import load_dotenv

from src.infrastructure.ibm_quantum_adapter import IBMQuantumAdapter
from src.domain.quantum.entities import VQCTopology

def main():
    load_dotenv()
    
    print("=" * 70)
    print("🧪 TEST: IBM REAL HARDWARE SUBMISSION")
    print("=" * 70)
    print()
    
    # Config
    use_real = os.getenv("USE_REAL_HARDWARE", "False") == "True"
    backend = os.getenv("IBM_BACKEND_NAME", "ibm_fez")
    
    print(f"⚙️  Configuración:")
    print(f"   USE_REAL_HARDWARE: {use_real}")
    print(f"   Backend: {backend}")
    print()
    
    # Create weights matching 12 qubits * 3 depth = 36 params
    os.makedirs("models", exist_ok=True)
    weights_path = "models/qnim_test_weights_36.npy"
    weights = np.random.randn(36) * 0.01
    np.save(weights_path, weights)
    
    print(f"📦 Pesos generados: {weights.shape[0]} parámetros")
    print()
    
    # Create topology matching weights
    topology = VQCTopology(
        num_qubits=12,
        num_features=12,
        feature_map_reps=2,
        ansatz_reps=2,
        entanglement_type="linear"
    )
    print(f"🔬 Topología: {topology.num_qubits} qubits → {topology.total_parameters} params")
    assert topology.total_parameters == 36, f"Mismatch: {topology.total_parameters} != 36"
    print()
    
    # Initialize adapter
    print("🔌 Inicializando IBMQuantumAdapter...")
    try:
        adapter = IBMQuantumAdapter(weights_path)
        print(f"   ✅ Adaptador instanciado")
        print(f"   Modo: {'🛰️  HARDWARE REAL' if adapter.use_real_hardware else '🔬 SIMULADOR LOCAL'}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 1
    
    print()
    
    # Create test data
    print("📊 Generando datos de prueba...")
    features = np.random.randn(12) * 0.1
    features = features / (np.linalg.norm(features) + 1e-10)
    print(f"   ✅ Features: {features.shape}")
    print()
    
    # Test execution
    print("🚀 Ejecutando circuito...")
    if use_real:
        print("   ⚠️  USE_REAL_HARDWARE=True - Intentando conexión a IBM...")
    else:
        print("   ℹ️  USE_REAL_HARDWARE=False - Usando simulador local")
    print()
    
    try:
        result = adapter.execute_circuit(topology, features.reshape(1, -1))
        print(f"✅ ÉXITO: Circuito ejecutado")
        print(f"   Resultado: {result}")
        print()
        print("📊 ANÁLISIS:")
        if use_real:
            print("   Plan OPEN permitió ejecución en hardware real (inesperado)")
        else:
            print("   Simulador local funcionó correctamente")
        return 0
        
    except RuntimeError as e:
        error_msg = str(e)
        print(f"ℹ️  INFO: {error_msg}")
        print()
        print("📊 ANÁLISIS:")
        
        if "not authorized to run a session when using the open plan" in error_msg.lower() or "free plan" in error_msg.lower():
            print("   ✅ ESPERADO: Plan OPEN rechazó job en hardware real")
            print("   ✅ USE_REAL_HARDWARE=True funcionó correctamente")
            print("   ✅ Sistema intenta conectar a IBM (credenciales validadas)")
            print()
            print("🎯 CONCLUSIÓN:")
            print("   • Sistema está preparado para hardware real")
            print("   • Para ejecutar en hardware necesitas plan de pago")
            print("   • Con USE_REAL_HARDWARE=False usa simulador local")
            return 0
        else:
            print(f"   ❌ Error inesperado: {error_msg}")
            return 1
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ ERROR: {error_msg}")
        print()
        print("📊 ANÁLISIS:")
        
        # Check expected errors
        if "channel" in error_msg.lower() or "token" in error_msg.lower():
            print("   ⚠️  Problema de autenticación/canal")
            print("   → Verifica credenciales IBM en .env")
            return 1
            
        else:
            print("   ❌ Error inesperado durante ejecución")
            import traceback
            traceback.print_exc()
            return 1

if __name__ == "__main__":
    sys.exit(main())
