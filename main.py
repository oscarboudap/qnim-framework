import os
import sys
from dotenv import load_dotenv

# Aseguramos que Python vea la carpeta 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Cargamos variables de entorno desde el .env
load_dotenv()

from src.infrastructure.h5_gw_repository import H5GWRepository
from src.infrastructure.qiskit_classifier import QiskitClassifier
from src.infrastructure.ibm_quantum_adapter import IBMQuantumClassifier
from src.application.process_event_use_case import ProcessEventUseCase
from src.application.signal_preprocessing_service import SignalPreprocessingService
from src.application.quantum_mapping_service import QuantumMappingService
from src.application.anomaly_generator_service import AnomalyGeneratorService # Importante añadir esta
from src.presentation.cli_presenter import CLIPresenter

def main():
    # 1. Configuración de Rutas y Entorno (Punto 3.2: Gestión de Infraestructura)
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Leemos configuración del .env
    USE_REAL_HARDWARE = os.getenv("USE_REAL_HARDWARE", "False").lower() == "true"
    IBM_TOKEN = os.getenv("IBM_QUANTUM_TOKEN")
    # Usamos ibm_fez que es el que te ha funcionado antes
    BACKEND_NAME = os.getenv("IBM_BACKEND_NAME", "ibm_fez")
    
    # 2. INYECCIÓN DE DEPENDENCIAS (Arquitectura Limpia)
    repo = H5GWRepository(base_path=base_path)
    
    # Selección del motor de inferencia (Local vs Hardware Real)
    if USE_REAL_HARDWARE:
        if not IBM_TOKEN:
            print("❌ ERROR: No se encontró IBM_QUANTUM_TOKEN en el archivo .env")
            sys.exit(1)
        # El adaptador ahora es autónomo con el token
        classifier = IBMQuantumClassifier(token=IBM_TOKEN, backend_name=BACKEND_NAME)
    else:
        print("💻 [QNIM] Iniciando Simulación Cuántica Local (Aer Simulator).")
        classifier = QiskitClassifier(n_qubits=8)
    
    # Instanciamos los servicios de aplicación
    preprocessor = SignalPreprocessingService()
    mapper = QuantumMappingService(target_qubits=8)
    anomaly_gen = AnomalyGeneratorService() # <--- Creamos la instancia aquí
    
    # Construimos el caso de uso inyectando todas las dependencias requeridas
    use_case = ProcessEventUseCase(
        repository=repo, 
        classifier=classifier, 
        preprocessor=preprocessor, 
        mapper=mapper,
        anomaly_generator=anomaly_gen # <--- Inyectamos el generador para evitar el error anterior
    )

    # 3. EJECUCIÓN DEL PIPELINE
    CLIPresenter.show_welcome()
    
    try:
        print(f"\n[INFO] Accediendo a datos de LIGO (H1) y preparando circuitos...")
        
        # Este método ahora enviará dos Jobs a la QPU de IBM si USE_REAL_HARDWARE es True
        res_rg, res_lqg = use_case.execute_comparison(detector_name="H1")
        
        # --- PRESENTACIÓN DE RESULTADOS ---
        print("\n" + "="*50)
        print("🔍 RESULTADOS DE LA INFERENCIA QNIM")
        print("="*50)
        
        # Probabilidades de la Hipótesis 0 (Relatividad General)
        print(f"Hipótesis RG (Realidad):   {res_rg.probabilities[0]:.4f} Confianza")
        print(f"Hipótesis LQG (Anomalía): {res_lqg.probabilities[0]:.4f} Confianza")
        
        # Delta de discriminación (Punto 5.2 de la Memoria)
        delta = abs(res_rg.probabilities[0] - res_lqg.probabilities[0])
        print(f"\n📈 Sensibilidad del modelo (Δ): {delta:.4f}")
        
        if USE_REAL_HARDWARE:
            print(f"\n[METADATA] IBM Job ID (RG): {res_rg.metadata.get('job_id')}")
            print(f"[METADATA] Backend: {res_rg.metadata.get('backend')}")

        print("="*50 + "\n")
        
    except Exception as e:
        # Aquí capturaremos cualquier error de red o de lógica del pipeline
        print(f"❌ ERROR EN EL PIPELINE: {e}")

if __name__ == "__main__":
    main()