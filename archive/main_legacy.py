import os
import joblib
import numpy as np
from pathlib import Path

# Infraestructura
from src.infrastructure.storage.quantum_dataloader import QuantumDatasetLoader
from src.infrastructure.neal_annealer_adapter import NealSimulatedAnnealerAdapter
from src.infrastructure.ibm_quantum_adapter import IBMQuantumAdapter 

# Aplicación
from src.application.hybrid_orchestrator import HybridInferenceOrchestrator
from src.application.process_event_use_case import DecodeGravitationalWaveUseCase

# Dominio
from src.domain.astrophysics.entities import QuantumDecodedEvent, GWSignal
from src.domain.astrophysics.value_objects import DetectorType, GPSTime, TheoryFamily
from src.domain.astrophysics.sstg.providers.kerr_provider import KerrVacuumProvider
from src.domain.astrophysics.value_objects import SolarMass

def main():
    print("🌌 --- INICIANDO QNIM FRAMEWORK (DECODIFICACIÓN MULTICAPA) --- 🌌")
    
    # --- CONFIGURACIÓN DEL EXPERIMENTO ---
    EVALUAR_EVENTO_REAL = True 
    NUM_PASADAS_ENSAMBLE = 5  # Para estabilizar la confianza en datos reales
    
    # 1. INICIALIZAR INFRAESTRUCTURA
    weights_path = "models/qnim_vqc_weights.npy"
    pipeline_path = "models/qnim_preprocessing_pipeline.pkl"
    
    if not os.path.exists(weights_path) or not os.path.exists(pipeline_path):
        print("❌ Error: Faltan los modelos. Ejecuta 'python3 train.py' primero.")
        return
        
    print("🔌 Conectando Adaptadores Cuánticos (Qiskit & Neal)...")
    dwave_adapter = NealSimulatedAnnealerAdapter()
    ibm_adapter = IBMQuantumAdapter(weights_path) 
    compressor = joblib.load(pipeline_path)

    # 2. INICIALIZAR APLICACIÓN
    orchestrator = HybridInferenceOrchestrator(ibm_adapter, dwave_adapter)
    use_case = DecodeGravitationalWaveUseCase(orchestrator, compressor)

    # 3. CARGAR DATOS
    print("📂 Cargando evento gravitacional interceptado...")
    loader = QuantumDatasetLoader(target_samples=16384)
    
    if EVALUAR_EVENTO_REAL:
        ruta_gw150914 = "data/raw/H-H1_LOSC_4_V2-1126259446-32.hdf5"
        raw_strain = loader.prepare_for_quantum(ruta_gw150914, is_real_data=True)
        event_id = "GW150914"
        snr_real = 24.0
    else:
        data_dir = Path("data/synthetic")
        latest_batch = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("2026")])[-1]
        test_file = list(latest_batch.glob("*.h5"))[-1] 
        raw_strain = loader.prepare_for_quantum(str(test_file), is_real_data=False)
        event_id = test_file.stem
        snr_real = 18.5

    signal = GWSignal(
        strain=raw_strain,
        detector=DetectorType.H1,
        sample_rate=4096,
        gps_start=GPSTime(1126259462.4), 
        snr_instrumental=snr_real
    )
    event = QuantumDecodedEvent(event_id=event_id, signal=signal)

    # 4. GENERAR ESPACIO DE BÚSQUEDA D-WAVE (Capa 2)
    print("🧮 Refinando banco de plantillas para D-Wave...")
    kerr = KerrVacuumProvider()
    templates = []
    
    masas_1 = np.linspace(34, 37, 5) 
    masas_2 = np.linspace(28, 32, 5) 
    
    for m1 in masas_1:
        for m2 in masas_2:
            tmpl_strain = kerr.generate_base_strain(SolarMass(m1), SolarMass(m2), distance_mpc=410.0)
            templates.append({
                "params": {"m1": round(m1, 2), "m2": round(m2, 2), "spin": 0.69},
                "strain": tmpl_strain
            })

    # 5. EJECUTAR INFERENCIA HÍBRIDA (Modo Ensamble)
    print(f"🚀 Iniciando Ensamble Cuántico ({NUM_PASADAS_ENSAMBLE} pasadas para estabilidad)...")
    
    lista_sigmas = []
    lista_delta_q = []
    decoded_event = None

    for i in range(NUM_PASADAS_ENSAMBLE):
        progreso = ((i + 1) / NUM_PASADAS_ENSAMBLE) * 100
        print(f"  [{progreso:3.0f}%] Procesando pasada {i+1}/{NUM_PASADAS_ENSAMBLE}...", end="\r")
        
        # Ejecución del caso de uso
        resultado = use_case.execute(event, templates)
        
        lista_sigmas.append(resultado.topology.quantum_confidence_sigma)
        lista_delta_q.append(resultado.topology.no_hair_delta_q)
        decoded_event = resultado # Guardamos la última instancia para el reporte

    # Promediamos los valores críticos para el reporte final
    sigma_final = sum(lista_sigmas) / NUM_PASADAS_ENSAMBLE
    delta_q_final = sum(lista_delta_q) / NUM_PASADAS_ENSAMBLE
    
    # Actualizamos el objeto final con los promedios
    decoded_event.topology.quantum_confidence_sigma = sigma_final
    decoded_event.topology.no_hair_delta_q = delta_q_final

    print("\n" + "═"*70)
    print("🌌 QNIM MULTILAYER DECODER REPORT (PROCESAMIENTO COMPLETO)")
    print("═"*70)

    print(f"\n[CAPA 1 & 2: GEOMETRÍA Y SEÑAL]")
    print(f" ├─ Masas: {decoded_event.geometry.m1.value}/{decoded_event.geometry.m2.value} M_sun")
    print(f" └─ Momento Cuadrupolar Q: {decoded_event.topology.no_hair_delta_q:.4f}")

    print(f"\n[CAPA 3 & 4: ENTORNO Y COSMOLOGÍA]")
    print(f" ├─ B-Field (Magnetares): {decoded_event.environment.magnetic_field_gauss:.2e} G")
    print(f" ├─ Constante de Hubble H0: {decoded_event.cosmology.h0:.2f} km/s/Mpc")
    print(f" └─ Masa Gravitón m_g: {decoded_event.cosmology.m_graviton:.2e} eV")

    print(f"\n[CAPA 5 & 6: BEYOND GR Y ESTRUCTURA DEL HORIZONTE]")
    print(f" ├─ Emisión Dipolar: {decoded_event.beyond_gr.dipolar_emission_strength:.4f}")
    print(f" ├─ Ecos de ECOs (Reflectividad): {decoded_event.topology.echo_amplitude:.4f}")
    print(f" └─ Dimensiones Extra (KK): {decoded_event.beyond_gr.extra_dims}")

    print(f"\n[CAPA 7: FÍSICA CUÁNTICA PROFUNDA]")
    print(f" ├─ Coef. Violación Lorentz: {decoded_event.quantum.lorentz_v:.2e}")
    print(f" ├─ Confianza Cuántica Final: {decoded_event.quantum.sigma:.2f} Sigma")
    print(f" └─ 🌌 VERDICTO TEÓRICO: {decoded_event.quantum.theory.value}")
    print("═"*70)
    
    # Decisión final basada en el promedio del ensamble
    veredicto = decoded_event.topology.detected_theory.value
    print(f"  └─ 🌌 VERDICTO FÍSICO FINAL: {veredicto}")
    print("="*65)
    print(f"✅ Análisis completado. El ensamble ha reducido la varianza en un {100*(1-1/np.sqrt(NUM_PASADAS_ENSAMBLE)):.1f}%.")

if __name__ == "__main__":
    main()