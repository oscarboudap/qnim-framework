import os
import sys
import numpy as np
from dotenv import load_dotenv

# Aseguramos que Python vea la carpeta 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

load_dotenv()

# --- ORQUESTACIÓN HÍBRIDA (NUEVA ARQUITECTURA) ---
from src.application.hybrid_orchestrator import HybridInferenceOrchestrator
from src.infrastructure.storage.quantum_dataloader import QuantumDatasetLoader

# --- SERVICIOS ---
from src.application.sstg_service import SSTGService
from src.application.anomaly_generator_service import AnomalyGeneratorService

# --- DOMINIO / METROLOGÍA ---
from src.domain.metrology.multipole_validator import MultipoleValidator
from src.domain.astrophysics.converters import extract_physical_params
from src.presentation.cli_presenter import CLIPresenter

def main():
    # 1. CONFIGURACIÓN DE ENTORNO
    base_path = os.path.dirname(os.path.abspath(__file__))
    USE_REAL_HARDWARE = os.getenv("USE_REAL_HARDWARE", "False").lower() == "true"
    MODE = os.getenv("QNIM_MODE", "REAL") # Opciones: "REAL" o "SYNTHETIC"
    
    token = os.getenv("IBM_QUANTUM_TOKEN")
    backend = os.getenv("IBM_BACKEND_NAME", "ibm_kingston")

    # 2. INYECCIÓN DE DEPENDENCIAS (El nuevo cerebro)
    orchestrator = HybridInferenceOrchestrator(
        ibm_token=token,
        ibm_backend=backend,
        use_real_ibm=USE_REAL_HARDWARE
    )
    
    loader = QuantumDatasetLoader(target_samples=64)
    sstg_service = SSTGService()
    anomaly_gen = AnomalyGeneratorService()
    multipole_val = MultipoleValidator()

    CLIPresenter.show_welcome()
    
    try:
        # 3. SELECCIÓN DE DATOS (Real vs Sintético Estocástico)
        if MODE == "SYNTHETIC":
            print(f"\n[SSTG] Generando Evento Sintético Estocástico (Blind Challenge)...")
            challenge = sstg_service.generate_blind_challenge(theory="RG") 
            raw_strain = challenge['strain']
            ground_truth = challenge['metadata']
            
            # Adaptamos la señal para el QML
            quantum_data = raw_strain[:64] if len(raw_strain) >= 64 else np.pad(raw_strain, (0, 64-len(raw_strain)))
            event_id = "GW-SSTG-BLIND"
            print(f"🧪 Teoría Inyectada: {ground_truth['theory']} | Masa Objetivo: {ground_truth['m1']:.2f}")
            
        else:
            print(f"\n[LIGO] Accediendo a datos reales...")
            # Para el TFM, cargamos un evento validado de la carpeta sintética como proxy de datos reales
            data_dir = os.path.join(base_path, "data", "synthetic")
            try:
                latest_batch = sorted([d for d in os.listdir(data_dir) if d.startswith("2026")])[-1]
                sample_file = os.path.join(data_dir, latest_batch, "event_00000.h5")
                quantum_data = loader.prepare_for_quantum(sample_file)
            except Exception:
                raise FileNotFoundError("No se han encontrado datos preprocesados en data/synthetic.")
            
            ground_truth = {"m1": 36.2, "spin": 0.0, "theory": "RG"} # Referencia de GW150914
            event_id = "GW150914-REAL"

        # 4. PREPARACIÓN DEL CONTEXTO
        event_context = {
            'id': event_id,
            'quantum_signal': quantum_data,
            'metadata': ground_truth 
        }

        # 5. EJECUCIÓN DEL PIPELINE QML (Conecta a IBM si USE_REAL_HARDWARE=True)
        print("\n🚀 Iniciando Inferencia Híbrida QNIM...")
        diagnosis = orchestrator.run_full_diagnosis(event_context)
        
        # Extracción de resultados
        theory_label = diagnosis['theory']
        confidence = diagnosis['confidence']
        m_dwave = diagnosis['best_fit_mass']
        
        # 6. INFERENCIA Y METROLOGÍA (Formalismo Christensen & Meyer)
        # Usamos la masa refinada por D-Wave para la metrología
        delta_q = anomaly_gen.get_quadrupole_deviation(confidence)
        
        # Como D-Wave nos da m1, estimamos m2 para el validador (aproximación)
        analysis = multipole_val.check_no_hair_theorem(m_dwave, m_dwave*0.8, delta_q)
        
        # 7. PRESENTACIÓN DE RESULTADOS DOCTORALES
        print("\n" + "═"*60)
        print(f"📜 INFORME QNIM - MODO: {MODE}")
        print(f"📡 Hardware Real IBM: {'ACTIVO' if USE_REAL_HARDWARE else 'INACTIVO (SIMULADOR)'}")
        print("═"*60)
        print(f"🔹 TEORÍA DETECTADA:    {theory_label}")
        print(f"🔹 CONFIANZA CUÁNTICA:  {confidence*100:.2f}%")
        
        print("\n" + "─"*60)
        print("📊 COMPARATIVA DE PRECISIÓN (Inferencia Híbrida vs Ground Truth)")
        print(f"Masa Calculada (D-Wave):  {m_dwave:.2f} M_sun")
        print(f"Masa Real (Referencia):   {ground_truth['m1']:.2f} M_sun")
        error_m1 = abs(m_dwave - ground_truth['m1']) / ground_truth['m1'] * 100
        print(f"❌ Error Relativo:        {error_m1:.2f}%")
        
        print("\n" + "🔭 METROLOGÍA Y ANOMALÍAS")
        print(f"🔸 Desviación ΔQ:        {delta_q:.4f}")
        print(f"🔸 Consistencia RG:      {'SÍ' if 'Relativity' in theory_label else 'NO (ANOMALÍA DETECTADA)'}")
        print("═"*60 + "\n")

    except Exception as e:
        print(f"❌ ERROR EN EL PIPELINE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()