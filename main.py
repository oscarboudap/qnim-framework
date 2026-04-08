import os
import sys
import numpy as np
from dotenv import load_dotenv

# Aseguramos que Python vea la carpeta 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Cargamos variables de entorno (Token, Backend, etc.)
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

# --- METROLOGÍA (NUEVO) ---
from src.domain.metrology.multipole_validator import MultipoleValidator
from src.domain.metrology.fisher_matrix_calculator import FisherMatrixCalculator
from src.presentation.cli_presenter import CLIPresenter

def main():
    # 1. CONFIGURACIÓN DE ENTORNO
    base_path = os.path.dirname(os.path.abspath(__file__))
    USE_REAL_HARDWARE = os.getenv("USE_REAL_HARDWARE", "False").lower() == "true"
    IBM_TOKEN = os.getenv("IBM_QUANTUM_TOKEN")
    BACKEND_NAME = os.getenv("IBM_BACKEND_NAME", "ibm_fez")
    
    # 2. INYECCIÓN DE DEPENDENCIAS
    repo = H5GWRepository(base_path=base_path)
    
    if USE_REAL_HARDWARE:
        if not IBM_TOKEN:
            print("❌ ERROR: No se encontró IBM_QUANTUM_TOKEN")
            sys.exit(1)
        classifier = IBMQuantumClassifier(token=IBM_TOKEN, backend_name=BACKEND_NAME)
    else:
        print("💻 [QNIM] Iniciando Simulación Cuántica Local (FakeSherbrooke Noise Model).")
        classifier = QiskitClassifier(n_qubits=8)
    
    # Instanciamos servicios con rigor físico
    preprocessor = SignalPreprocessingService()
    mapper = QuantumMappingService(target_qubits=8)
    anomaly_gen = AnomalyGeneratorService()
    hubble_service = HubbleSolverService()
    multipole_val = MultipoleValidator()
    
    # Construimos el caso de uso
    use_case = ProcessEventUseCase(
        repository=repo, 
        classifier=classifier, 
        preprocessor=preprocessor, 
        mapper=mapper,
        anomaly_generator=anomaly_gen
    )

    # 3. EJECUCIÓN DEL PIPELINE CIENTÍFICO
    CLIPresenter.show_welcome()
    
    try:
        print(f"\n[INFO] Accediendo a datos de LIGO y calculando Auditoría Geométrica...")
        
        # Ejecutamos comparación entre Einstein (RG) y Teorías Alternativas
        res_rg, res_lqg = use_case.execute_comparison(detector_name="H1")
        
        # --- 4. EXTRACCIÓN DE FICHA TÉCNICA (Punto 6.2) ---
        # Simulamos una distancia de 100 Mpc para el evento GW150914 (Sirena Estándar)
        dist_mpc = 100.0
        snr_est = 20.0
        
        # Inferencia de Hubble (Métrica FLRW)
        z_approx = (dist_mpc * 70) / 299792.458 
        h0_val = hubble_service.infer_h0(dist_mpc, z_approx)
        
        # Validación de No-Cabello (Multipolos de Kerr)
        # Usamos la probabilidad del modelo anómalo para ver cuánto 'cabello' detecta
        delta_q = anomaly_gen.get_quadrupole_deviation(res_lqg.probabilities[0])
        hair_analysis = multipole_val.check_no_hair_theorem(mass=30, spin=0.7, observed_m2=delta_q)
        
        # Matriz de Fisher (Límites de Precisión)
        fisher = FisherMatrixCalculator(snr=snr_est)
        precision = fisher.calculate_precision_bounds({"mass": 30})

        # --- 5. PRESENTACIÓN DE RESULTADOS ---
        print("\n" + "═"*60)
        print("📜 FICHA TÉCNICA DE PLANCK - INFERENCIA MULTIFÍSICA")
        print("═"*60)
        
        print(f"🔹 NATURALEZA DEL OBJETO: {hair_analysis['object_type']}")
        print(f"🔹 CONFIANZA CUÁNTICA:    {res_lqg.probabilities[0]:.4f}")
        print(f"🔹 DESVIACIÓN DE KERR (ΔQ): {delta_q:.4f} (Hair Parameter)")
        
        print("\n" + "─"*60)
        print("📊 METROLOGÍA COSMOLÓGICA (Sirena Estándar)")
        print(f"🔸 Constante de Hubble (H0): {h0_val:.2f} km/s/Mpc")
        print(f"🔸 Precisión de Masa (σM):  ±{precision['sigma_mass']:.4f} M_sun")
        print("─"*60)
        
        # Delta de discriminación
        delta_sens = abs(res_rg.probabilities[0] - res_lqg.probabilities[0])
        print(f"\n📈 SENSIBILIDAD DEL FRAMEWORK (Δ): {delta_sens:.4f}")
        
        if delta_q > 0.05:
            print("\n⚠️  DETECCIÓN ANÓMALA: La señal presenta ecos compatibles con Cuerdas (Fuzzball).")
        else:
            print("\n✅ CONSISTENCIA: La señal es compatible con Relatividad General de Einstein.")

        print("═"*60 + "\n")
        
    except Exception as e:
        print(f"❌ ERROR EN EL PIPELINE: {e}")

if __name__ == "__main__":
    main()