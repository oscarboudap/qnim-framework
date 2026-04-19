#!/usr/bin/env python3
"""
VALIDADOR DE CONECTIVIDAD: Verifica que la infrastructure esté correctamente configurada

Esta herramienta de validación integra con la arquitectura DDD:
- Valida adaptadores cuánticos (IBM Quantum Adapter)
- Verifica configuración .env
- Comprueba que los modelos preentrenados existan
- Usa los mismos puertos que DecodeGravitationalWaveUseCase

Resultado: Infrastructure layer correctamente inyectado y funcional
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path so imports work from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# IMPORTACIONES DDD
# ============================================================================

from src.infrastructure.ibm_quantum_adapter import IBMQuantumAdapter
from src.infrastructure.neal_annealer_adapter import NealSimulatedAnnealerAdapter
from src.domain.quantum.entities import VQCTopology


# ============================================================================
# VALIDADORES
# ============================================================================

def validate_env_configuration() -> bool:
    """Valida que las variables de .env estén correctamente configuradas."""
    print("📋 Validando configuración .env...")
    
    load_dotenv()
    
    required_vars = {
        "IBM_QUANTUM_TOKEN": "Token para autenticación IBM Quantum",
        "IBM_BACKEND_NAME": "Backend IBM Quantum (ibm_fez, ibm_kingston, etc)",
        "USE_REAL_HARDWARE": "Flag para usar hardware real (False/True)",
    }
    
    all_valid = True
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            if var == "IBM_QUANTUM_TOKEN":
                # No mostrar el token completo
                displayed = value[:20] + "..." if len(value) > 20 else value
            else:
                displayed = value
            
            print(f"   ✅ {var:<20} = {displayed}")
        else:
            print(f"   ❌ {var:<20} = NO CONFIGURADA")
            all_valid = False
    
    return all_valid


def validate_models_exist() -> bool:
    """Valida que los modelos preentrenados existan."""
    print("\n📦 Validando modelos preentrenados...")
    
    models = {
        "models/qnim_vqc_weights.npy": "Pesos VQC entrenados",
        "models/qnim_preprocessing_pipeline.pkl": "Pipeline de preprocesamiento",
    }
    
    all_exist = True
    for path, description in models.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024**2)
            print(f"   ✅ {path:<50} ({size_mb:.2f} MB)")
        else:
            print(f"   ❌ {path:<50} NO ENCONTRADO")
            all_exist = False
    
    return all_exist


def validate_ibm_adapter() -> bool:
    """
    Valida el adaptador IBM Quantum (Infrastructure layer).
    
    Este es el puerto IGateBasedQuantumComputer que se inyecta en
    HybridInferenceOrchestrator → DecodeGravitationalWaveUseCase
    """
    print("\n🛰️  Validando Adaptador IBM Quantum (Infrastructure Layer)...")
    
    try:
        # Verificar que el archivo de pesos existe (prerrequisito)
        if not os.path.exists("models/qnim_vqc_weights.npy"):
            print("   ⚠️  Pesos no encontrados - usando ruta dummy para validación")
            weights_path = "models/dummy_weights.npy"
            # Crear un dummy por esta validación
            import numpy as np
            os.makedirs("models", exist_ok=True)
            np.save(weights_path, np.random.randn(10))
        else:
            weights_path = "models/qnim_vqc_weights.npy"
        
        # Instanciar el adaptador (inyección de dependencias)
        adapter = IBMQuantumAdapter(weights_path)
        print(f"   ✅ Adaptador instanciado correctamente")
        
        # Modo de ejecución
        modo = "🛰️  HARDWARE REAL" if adapter.use_real_hardware else "🔬 SIMULADOR LOCAL"
        print(f"   ℹ️  Modo: {modo}")
        print(f"   ℹ️  Backend configurado: {adapter.backend_name}")
        
        # Validar que puede crear topologías VQC
        topology = VQCTopology(
            num_qubits=4,
            num_features=8,
            feature_map_reps=2,
            ansatz_reps=3,
            entanglement_type="linear"
        )
        print(f"   ✅ VQCTopology creada: {topology.num_qubits} qubits")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error al instanciar adaptador: {str(e)}")
        return False


def validate_neal_adapter() -> bool:
    """
    Valida el adaptador D-Wave Neal (Infrastructure layer).
    
    Este es el puerto IQuantumAnnealer que se inyecta en
    HybridInferenceOrchestrator
    """
    print("\n❄️  Validando Adaptador D-Wave Neal (Infrastructure Layer)...")
    
    try:
        adapter = NealSimulatedAnnealerAdapter()
        print(f"   ✅ Adaptador instanciado correctamente")
        print(f"   ℹ️  Simulador: {adapter.sampler.__class__.__name__}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error al instanciar adaptador: {str(e)}")
        return False


def validate_ddd_structure() -> bool:
    """Valida que la arquitectura DDD existe y está estructura correctamente."""
    print("\n🏗️  Validando Arquitectura DDD...")
    
    required_dirs = {
        "src/domain": "Domain Layer",
        "src/application": "Application Layer",
        "src/infrastructure": "Infrastructure Layer",
        "src/presentation": "Presentation Layer",
    }
    
    all_exist = True
    for path, name in required_dirs.items():
        if os.path.isdir(path):
            file_count = len(list(Path(path).rglob("*.py")))
            print(f"   ✅ {name:<20} ({file_count} archivos Python)")
        else:
            print(f"   ❌ {name:<20} NO ENCONTRADO")
            all_exist = False
    
    return all_exist


# ============================================================================
# FUNCIÓN PRINCIPAL DE VALIDACIÓN
# ============================================================================

def validate_all() -> int:
    """
    Ejecuta todas las validaciones e imprime reporte.
    
    Returns:
        0 si todo OK, 1 si hay errores
    """
    print("="*70)
    print("🔍 VALIDADOR DE INFRAESTRUCTURA QNIM (DDD-Integrated)")
    print("="*70)
    print()
    
    results = {}
    
    # Ejecutar validaciones
    results["DDD Structure"] = validate_ddd_structure()
    results["Environment Configuration"] = validate_env_configuration()
    results["Models"] = validate_models_exist()
    results["Neal Adapter"] = validate_neal_adapter()
    results["IBM Adapter"] = validate_ibm_adapter()
    
    # Reporte final
    print("\n" + "="*70)
    print("📊 REPORTE DE VALIDACIÓN")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} | {name}")
    
    print("="*70)
    print(f"Resultado: {passed}/{total} validaciones pasadas")
    
    if passed == total:
        print("✅ INFRAESTRUCTURA LISTA PARA EJECUCIÓN")
        print("\nPuedes ejecutar:")
        print("   python3 run_qnim_inference.py  # Ejecutar inferencia completa")
        print("   python3 train.py                # Entrenar modelos")
        return 0
    else:
        print("❌ ERRORES DE CONFIGURACIÓN - Revisa los mensajes arriba")
        return 1


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    sys.exit(validate_all())
