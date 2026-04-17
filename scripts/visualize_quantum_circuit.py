import os
import sys
from pathlib import Path

# 1. Configuración agresiva del Path para WSL
# Buscamos la carpeta que contiene 'src'
current_file = Path(__file__).resolve()
project_root = current_file.parent
if (project_root / 'src').exists():
    sys.path.insert(0, str(project_root))
else:
    # Si se lanza desde subcarpetas, buscamos recursivamente hacia arriba
    for parent in current_file.parents:
        if (parent / 'src').exists():
            sys.path.insert(0, str(parent))
            project_root = parent
            break

# 2. Imports locales
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation, RealAmplitudes
from qiskit.circuit.library import real_amplitudes
# Importamos tus módulos verificados
from src.application.sstg_service import SSTGService
from src.infrastructure.storage.quantum_dataloader import QuantumDatasetLoader

def generate_and_visualize():
    print("🔬 Iniciando Pipeline de Visualización QNIM...")
    
    # --- BLOQUE 1: GENERACIÓN DE DATOS (Física Teórica) ---
    print("\n--- Bloque 1: Generación de Datos Sintéticos ---")
    service = SSTGService()
    # Generamos un evento de RG para tener una onda limpia
    event = service.generate_training_sample()
    print(f"✅ Evento generado. Teoría: {event['label']}")
    print(f"✅ Distancia: {event['metadata']['distance']:.2f} Mpc")

    # --- BLOQUE 2: PREPROCESAMIENTO (Mapeo Cuántico Pure) ---
    print("\n--- Bloque 2: Preprocesamiento y Mapeo ---")
    # Limpiamos la señal y extraemos 64 samples (para 6 cúbits)
    loader = QuantumDatasetLoader(target_samples=64)
    quantum_data = loader.prepare_for_quantum_event(event)
    print(f"✅ Señal normalizada (Norma: {np.linalg.norm(quantum_data):.2f})")

    # --- BLOQUE 2: CONSTRUCCIÓN DEL CIRCUITO (Amplitude Encoding) ---
    n_qubits = 6
    print(f"\n--- Creando Circuito de {n_qubits} Cúbits (Amplitude Encoding) ---")
    qc = QuantumCircuit(n_qubits)
    
    # Carga de datos pura: StatePreparation
    state_prep = StatePreparation(quantum_data)
    qc.append(state_prep, range(n_qubits))
    qc.barrier() # Separación visual
    
    # --- BLOQUE 3 (AVANCE): AÑADIMOS EL ANSATZ (Capa de Aprendizaje) ---
    # Para que el dibujo sea completo, añadimos la capa de RealAmplitudes
    print("--- Añadiendo Ansatz de RealAmplitudes (Capa Entrenable) ---")
    ansatz = real_amplitudes(num_qubits=n_qubits, reps=2, entanglement='linear')
    full_circuit = qc.compose(ansatz)
    full_circuit.measure_all()

    # --- VISUALIZACIÓN PROFESIONAL ---
    print("\n--- Generando Dibujo del Circuito (Matplotlib) ---")
    
    # Usamos el backend de Matplotlib 'agg' que no requiere display
    plt.switch_backend('Agg')
    
    # Dibujamos el circuito completo
    fig = full_circuit.draw(output='mpl', style='iqx', scale=1.0)
    
    # Guardamos la imagen
    output_image = "qnim_quantum_circuit.png"
    fig.savefig(output_image, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"🎉 ¡Éxito! El dibujo del circuito se ha guardado en: {os.path.abspath(output_image)}")
    print("💡 Abre este archivo desde Windows para ver cómo la onda gravitacional se ha convertido en cúbits.")

if __name__ == "__main__":
    generate_and_visualize()