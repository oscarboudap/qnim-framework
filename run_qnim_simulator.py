#!/usr/bin/env python3
"""
Presentation Layer: Demo de QNIM en Simulador Local

Ejecuta VQC en simulador Qiskit Aer (Compatible con plan gratuito)
Demuestra architecture DDD completo.
"""

import sys
import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Infrastructure: Simulator
from src.infrastructure.ibm_quantum_adapter import IBMQuantumAdapter
from src.domain.quantum.entities import VQCTopology


def main():
    print("="*70)
    print("🚀 QNIM - SIMULADOR LOCAL (PLAN GRATUITO)")
    print("="*70)
    print()
    
    load_dotenv()
    
    # 1. Setup (Infrastructure)
    print("📦 Verificando dependencias...")
    try:
        import qiskit
        print(f"   ✅ Qiskit {qiskit.__version__}")
    except ImportError:
        print("   ❌ Qiskit no instalado")
        return 1
    
    try:
        from qiskit_aer import AerSimulator
        print("   ✅ Qiskit Aer")
    except ImportError:
        print("   ❌ Qiskit Aer no instalado")
        return 1
    
    print()
    
    # 2. Datos sintéticos (Domain logic)
    print("📊 Generando ondas gravitacionales sintéticas...")
    t = np.linspace(0, 1, 1024)
    phase = 2 * np.pi * (35 * t + (250 - 35) * t**2 / 2)
    strain_data = np.sin(phase) + np.random.randn(len(phase)) * 0.1
    
    print(f"   ✅ {len(strain_data)} samples")
    print(f"   📈 Amplitud: [{strain_data.min():.3f}, {strain_data.max():.3f}]")
    print()
    
    # 3. Crear adapter IBM (Infrastructure - usa flag del .env)
    print("🔌 Inicializando IBMQuantumAdapter (respeta .env)...")
    try:
        # Use actual pretrained weights path
        weights_path = "models/qnim_vqc_weights.npy"
        if not os.path.exists(weights_path):
            print(f"   ⚠️  {weights_path} no encontrado, creando pesos de prueba...")
            os.makedirs("models", exist_ok=True)
            # 10 parameters = 5 qubits * (1 rep + 1), or 10 qubits * 1
            np.save(weights_path, np.random.randn(10) * 0.1)
        
        adapter = IBMQuantumAdapter(weights_path)
        modo = "🛰️  HARDWARE REAL" if adapter.use_real_hardware else "🔬 SIMULADOR LOCAL"
        print(f"   ✅ Adaptador: {modo} (pesos: {adapter.loaded_weights.shape[0]} params)")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print()
    
    # 4. Crear VQC (Domain entity) - Must match trained weights!
    print("🔧 VQC Topology (matching trained weights)...")
    # 36 parameters → 12 qubits * 3 depth (12 * (2+1) = 36)
    topology = VQCTopology(
        num_qubits=12,
        num_features=12,
        feature_map_reps=2,
        ansatz_reps=2,
        entanglement_type="linear"
    )
    print(f"   ✅ {topology.num_qubits} qubits | Depth: {topology.ansatz_reps} | Params: {topology.total_parameters}")
    
    # 5. Normalizar data (Domain logic)
    print("⚙️  Normalizando datos...")
    # Extract features matching circuit input size (num_qubits)
    features = strain_data[:topology.num_qubits]
    features = features / (np.linalg.norm(features) + 1e-10)
    print(f"   ✅ Features normalizados ({len(features)} dims para {topology.num_qubits} qubits)")
    print()
    
    # 6. Ejecutar (Infrastructure: adapter decides local vs remote by .env)
    print("🚀 Ejecutando circuito VQC...")
    try:
        result = adapter.execute_circuit(topology, features.reshape(1, -1))
        print(f"   ✅ Resultado: {result}")
    except Exception as e:
        print(f"   ❌ Error durante ejecución: {e}")
        print("   (Es normal si el plan es 'open' y probaste USE_REAL_HARDWARE=True)")
        return 1
    
    print()
    print("="*70)
    print("✅ DEMO COMPLETADA")
    print("="*70)
    print()
    print("Configuración:")
    print(f"  • USE_REAL_HARDWARE: {adapter.use_real_hardware}")
    print(f"  • Backend: {adapter.backend_name}")
    print(f"  • Plan: FREE (simulador local es ilimitado)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
