import os
import sys
from pathlib import Path

# 1. Forzar la raíz del proyecto en el PATH
# Buscamos la carpeta que contiene 'src'
current_dir = Path(__file__).resolve().parent # src/test
project_root = current_dir.parent.parent      # raíz del proyecto
sys.path.insert(0, str(project_root))

# 2. Ahora realizamos los imports
import numpy as np
import h5py
import matplotlib.pyplot as plt

try:
    from src.application.sstg_service import SSTGService
    from src.infrastructure.storage.hdf5_exporter import HDF5Exporter
    print("✅ Módulos cargados correctamente.")
except ModuleNotFoundError as e:
    print(f"❌ Error de importación: {e}")
    print(f"Ruta de búsqueda actual: {sys.path[0]}")
    sys.exit(1)

# ... resto del código ...
def run_audit():
    print("🔬 Iniciando Auditoría del Generador de Universos QNIM...")
    
    # 1. Instanciamos servicios
    service = SSTGService()
    exporter = HDF5Exporter()
    
    # 2. Generamos una pequeña muestra estadística (Población Realista)
    n_test = 5
    print(f"Generating {n_test} synthetic events with realistic priors...")
    samples = [service.generate_training_sample() for _ in range(n_test)]
    
    # 3. Exportamos a HDF5 (Creará la carpeta con timestamp)
    output_path = exporter.save_batch(samples)
    print(f"✅ Archivos exportados a: {output_path}")

    # 4. PRUEBA DE "CAJA NEGRA" (Simulando al QML)
    print("\n--- Verificación de Blindaje de Datos ---")
    test_file = list(Path(output_path).glob("*.h5"))[0]
    
    with h5py.File(test_file, "r") as f:
        # El QML solo debe ver esto
        strain = f["strain"][:]
        
        # El QML NO debe ver esto (solo nosotros para validar)
        label = f.attrs["target_label"]
        dist = f.attrs["true_distance"]
        
        print(f"Archivo auditado: {test_file.name}")
        print(f"Etiqueta oculta detectada: {label}")
        print(f"Longitud del strain: {len(strain)} samples")
        
        if len(strain) > 0 and not np.isnan(strain).any():
            print("✅ Señal física válida (sin NaNs).")
        else:
            print("❌ Error: Señal corrupta o vacía.")

    # 5. INSPECCIÓN VISUAL DE DIFERENCIAS TEÓRICAS
    visualize_comparison(service)

def visualize_comparison(service):
    """Genera dos teorías opuestas para ver si la física se inyecta correctamente."""
    # Cambiamos 'generate_blind_challenge' por 'generate_training_sample'
    # O bien, si quieres forzar la categoría, asegúrate de que el método lo permita
    
    print("📊 Generando comparativa visual...")
    
    # Si tu SSTGService.generate_training_sample no acepta 'category', 
    # simplemente genera dos muestras aleatorias:
    event_1 = service.generate_training_sample()
    event_2 = service.generate_training_sample()
    
    plt.figure(figsize=(12, 6))
    plt.plot(event_1['strain'][:1000], label=f"Evento A: {event_1['label']}", alpha=0.7)
    plt.plot(event_2['strain'][:1000], label=f"Evento B: {event_2['label']}", alpha=0.7)
    
    plt.title("Comparativa de Firmas Sintéticas QNIM")
    plt.xlabel("Samples")
    plt.ylabel("Strain (h)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Si estás en WSL sin entorno gráfico, guarda la imagen:
    plt.savefig('audit_comparison.png')
    print("✅ Gráfico guardado en 'audit_comparison.png'")
    # plt.show() # Descomenta si tienes X-Server

if __name__ == "__main__":
    run_audit()