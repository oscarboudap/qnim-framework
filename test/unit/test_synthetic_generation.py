import os
import sys
from pathlib import Path

# 1. Forzar la raíz del proyecto en el PATH
current_dir = Path(__file__).resolve().parent 
project_root = current_dir.parent.parent      
sys.path.insert(0, str(project_root))

import numpy as np
import h5py
import matplotlib.pyplot as plt

try:
    from src.application.sstg_service import SSTGService
    from src.infrastructure.storage.hdf5_exporter import HDF5Exporter
    print("✅ Módulos cargados correctamente.")
except ModuleNotFoundError as e:
    print(f"❌ Error de importación: {e}")
    sys.exit(1)

def run_audit():
    print("🔬 Iniciando Auditoría con Filtro de SNR para Gravedad Cuántica...")
    
    service = SSTGService()
    exporter = HDF5Exporter()
    
    n_test = 200
    samples = []
    
    print(f"🌌 Generando {n_test} eventos. Estrategia: Búsqueda de alta fidelidad para LQG.")

    for i in range(n_test):
        # Decidimos la teoría para esta muestra
        target_theory = "LQG" if i >= (n_test // 2) else "RG"
        
        # --- BUCLE DE CALIDAD DE SEÑAL ---
        # Si es LQG, generamos hasta encontrar una muestra con SNR aceptable (distancia corta)
        while True:
            # Asumimos que generate_training_sample acepta parámetros de control
            # O que genera metadatos que podemos validar
            sample = service.generate_training_sample() 
            
            # Si es RG, la aceptamos siempre (para que el modelo vea RG en todas sus formas)
            if target_theory == "RG":
                samples.append(sample)
                break
            
            # Si es LQG, verificamos que la distancia sea corta (SNR alta)
            # Ajusta el umbral según la lógica de tu SSTGService (ej. d < 500 Mpc)
            distancia = sample.get('metadata', {}).get('distance', 1000)
            if distancia < 600: 
                # Etiquetamos explícitamente como anomalía para el entrenamiento
                sample['label'] = "LQG" 
                samples.append(sample)
                break
            # Si no es buena, el bucle 'while' repite la generación
            
        print(f"\r⏳ [SSTG] Evento {i+1}/{n_test} sintetizado ({target_theory})... ", end="", flush=True)
        
    print("\n✅ Dataset balanceado y curado por SNR generado con éxito.")
    
    output_path = exporter.save_batch(samples)
    print(f"✅ Archivos exportados a: {output_path}")

    # Inspección visual de la última muestra de LQG generada
    visualize_comparison(service, samples[-1])

def visualize_comparison(service, last_sample):
    print("📊 Generando comparativa visual del evento curado...")
    
    # Generamos un RG puro para comparar contra la última anomalía
    rg_event = service.generate_training_sample() # Por defecto suele ser Kerr
    
    plt.figure(figsize=(12, 6))
    plt.plot(rg_event['strain'][:2000], label="Referencia: RG (Kerr)", alpha=0.5, color='gray')
    plt.plot(last_sample['strain'][:2000], label=f"Target: {last_sample['label']} (Alta SNR)", alpha=0.9)
    
    plt.title("Firma de Gravedad Cuántica vs Relatividad General (Señal Curada)")
    plt.xlabel("Samples")
    plt.ylabel("Strain (h)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('audit_comparison_high_snr.png')
    print("✅ Gráfico de alta fidelidad guardado en 'audit_comparison_high_snr.png'")

if __name__ == "__main__":
    run_audit()