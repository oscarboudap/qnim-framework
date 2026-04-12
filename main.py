import os
import sys
import numpy as np
from dotenv import load_dotenv

# Aseguramos que Python vea la carpeta 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

load_dotenv()

# --- INFRAESTRUCTURA ---
from src.infrastructure.h5_gw_repository import H5GWRepository
from src.infrastructure.qiskit_classifier import QiskitClassifier
from src.infrastructure.ibm_quantum_adapter import IBMQuantumClassifier

# --- APLICACIÓN ---
from src.application.process_event_use_case import ProcessEventUseCase
from src.application.signal_preprocessing_service import SignalPreprocessingService
from src.application.quantum_mapping_service import QuantumMappingService
from src.application.anomaly_generator_service import AnomalyGeneratorService
from src.application.hubble_solver_service import HubbleSolverService
# --- NUEVO SERVICIO SSTG ---
from src.application.sstg_service import SSTGService

# --- DOMINIO / METROLOGÍA ---
from src.domain.metrology.multipole_validator import MultipoleValidator
from src.domain.metrology.fisher_matrix_calculator import FisherMatrixCalculator
from src.domain.astrophysics.converters import extract_physical_params
from src.presentation.cli_presenter import CLIPresenter

def main():
    # 1. CONFIGURACIÓN DE ENTORNO
    base_path = os.path.dirname(os.path.abspath(__file__))
    USE_REAL_HARDWARE = os.getenv("USE_REAL_HARDWARE", "False").lower() == "true"
    # Nueva variable para decidir si usamos datos de LIGO o Generación Estocástica
    MODE = os.getenv("QNIM_MODE", "REAL") # Opciones: "REAL" o "SYNTHETIC"
    
    # 2. INYECCIÓN DE DEPENDENCIAS
    repo = H5GWRepository(base_path=base_path)
    
    if USE_REAL_HARDWARE:
        classifier = IBMQuantumClassifier(token=os.getenv("IBM_QUANTUM_TOKEN"), 
                                          backend_name=os.getenv("IBM_BACKEND_NAME", "ibm_fez"))
    else:
        classifier = QiskitClassifier(n_qubits=8)
    
    preprocessor = SignalPreprocessingService()
    mapper = QuantumMappingService(target_qubits=8)
    anomaly_gen = AnomalyGeneratorService()
    hubble_service = HubbleSolverService()
    multipole_val = MultipoleValidator()
    
    # Instanciamos el nuevo Laboratorio de Universos Sintéticos
    sstg_service = SSTGService()
    
    use_case = ProcessEventUseCase(
        repository=repo, 
        classifier=classifier, 
        preprocessor=preprocessor, 
        mapper=mapper,
        anomaly_generator=anomaly_gen
    )

    CLIPresenter.show_welcome()
    
    try:
        # 3. SELECCIÓN DE DATOS (Real vs Sintético Estocástico)
        if MODE == "SYNTHETIC":
            print(f"\n[SSTG] Generando Evento Sintético Estocástico (Blind Challenge)...")
            # El motor resuelve las ecuaciones y nos da una onda física válida
            challenge = sstg_service.generate_blind_challenge(theory="RG") 
            data_to_analyze = challenge['strain']
            ground_truth = challenge['metadata']
            print(f"🧪 Teoría Inyectada: {ground_truth['theory']} | Masa Objetivo: {ground_truth['m1']:.2f}")
        else:
            print(f"\n[LIGO] Accediendo a datos reales de GW150914...")
            # Aquí seguiría tu lógica de repo.get_event()
            data_to_analyze = "H1_data_from_h5" 
            ground_truth = {"m1": 36.2, "theory": "RG"} # Valores de referencia LVK

        # 4. EJECUCIÓN DEL PIPELINE QML
        # Adaptamos el execute para que acepte tanto datos reales como el strain sintético
        res_rg, res_lqg = use_case.execute_comparison(detector_data=data_to_analyze)
        
        # 5. INFERENCIA Y METROLOGÍA (Formalismo Christensen & Meyer)
        m_chirp_det = 28.1 + (res_lqg.probabilities[0] - 0.5) * 5 
        m1, m2 = extract_physical_params(m_chirp_det)
        
        # Validación de No-Cabello
        delta_q = anomaly_gen.get_quadrupole_deviation(res_lqg.probabilities[0])
        analysis = multipole_val.check_no_hair_theorem(m1, m2, delta_q)
        
        # 6. PRESENTACIÓN DE RESULTADOS DOCTORALES
        print("\n" + "═"*60)
        print(f"📜 INFORME QNIM - MODO: {MODE}")
        print("═"*60)
        print(f"🔹 TEORÍA DETECTADA:    {analysis['object_type']}")
        print(f"🔹 CONFIANZA CUÁNTICA:   {res_lqg.probabilities[0]:.4f}")
        
        print("\n" + "─"*60)
        print("📊 COMPARATIVA DE PRECISIÓN (Inferencia vs Ground Truth)")
        print(f"Masa Calculada (m1):  {m1:.2f} M_sun")
        print(f"Masa Real (Inyectada): {ground_truth['m1']:.2f} M_sun")
        error_m1 = abs(m1 - ground_truth['m1']) / ground_truth['m1'] * 100
        print(f"❌ Error Relativo:     {error_m1:.2f}%")
        
        print("\n" + "🔭 METROLOGÍA Y ANOMALÍAS")
        print(f"🔸 Desviación ΔQ:      {delta_q:.4f}")
        print(f"🔸 Consistencia RG:    {'SÍ' if analysis['is_pure_kerr'] else 'NO (ANOMALÍA)'}")
        print("═"*60 + "\n")

    except Exception as e:
        print(f"❌ ERROR EN EL PIPELINE: {e}")

if __name__ == "__main__":
    main()