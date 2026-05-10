import os
import sys
from pathlib import Path

# Forzar la raíz del proyecto en el PATH
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.infrastructure.storage.quantum_dataloader import QuantumDatasetLoader
from src.infrastructure.quantum_bridge import QuantumBridge

def run_pipeline_test():
    print("🧪 Iniciando Test del Motor de Preprocesamiento (Bloque 2)...")
    
    # 1. Localizar el último dataset generado
    data_dir = project_root / "data" / "synthetic"
    if not data_dir.exists():
        print("❌ No se encontró la carpeta de datos. Genera datos primero.")
        return

    all_folders = sorted([f for f in data_dir.iterdir() if f.is_dir()])
    if not all_folders:
        print("❌ No hay datasets disponibles.")
        return

    sample_file = list(all_folders[-1].glob("*.h5"))[0]
    print(f"📂 Procesando archivo: {sample_file.name}")

    # 2. Preprocesar (Carga -> Whitening -> Bandpass -> 64 samples)
    loader = QuantumDatasetLoader(target_samples=64)
    quantum_data = loader.prepare_for_quantum(str(sample_file))
    print(f"✅ Señal preparada. Norma unitaria: {np.linalg.norm(quantum_data):.4f}")

    # 3. Quantum Bridge (Amplitude Encoding)
    bridge = QuantumBridge(n_qubits=6)
    circuit = bridge.build_quantum_state(quantum_data)

    print("\n🚀 Circuito de Amplitude Encoding Generado (Cuántica Pura):")
    print(circuit.draw(output='text'))
    
    # Justificación para la memoria
    print(f"\n💡 Info: Mapeados {len(quantum_data)} puntos en {bridge.n_qubits} cúbits.")

if __name__ == "__main__":
    run_pipeline_test()