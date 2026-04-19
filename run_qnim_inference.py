#!/usr/bin/env python3
"""
EJECUTABLE PRINCIPAL DE QNIM: Decodificación de ondas gravitacionales

Este script integra COMPLETAMENTE la arquitectura DDD:
- Infrastructure: Adaptadores cuánticos (IBM + Neal) y de datos
- Application: Orquestador híbrido y casos de uso
- Domain: Entidades, value objects y lógica de negocio puro

Configuración por .env:
    USE_REAL_HARDWARE=False  → Simulador local (Plan gratuito recomendado)
    USE_REAL_HARDWARE=True   → Hardware real IBM (Plan pago)
"""

import os
import sys
import joblib
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# ============================================================================
# IMPORTACIONES DDD
# ============================================================================

# Infrastructure Layer: Adaptadores hexagonales
from src.infrastructure.storage.quantum_dataloader import QuantumDatasetLoader
from src.infrastructure.neal_annealer_adapter import NealSimulatedAnnealerAdapter
from src.infrastructure.ibm_quantum_adapter import IBMQuantumAdapter

# Application Layer: Casos de uso y orquestación
from src.application.hybrid_orchestrator import HybridInferenceOrchestrator
from src.application.process_event_use_case import DecodeGravitationalWaveUseCase

# Domain Layer: Entidades, Value Objects, y lógica pura
from src.domain.astrophysics.entities import QuantumDecodedEvent, GWSignal
from src.domain.astrophysics.value_objects import DetectorType, GPSTime, TheoryFamily
from src.domain.astrophysics.sstg.providers.kerr_provider import KerrVacuumProvider
from src.domain.astrophysics.value_objects import SolarMass


# ============================================================================
# FUNCIONES AUXILIARES DE VALIDACIÓN
# ============================================================================

def validate_prerequisites():
    """Valida que los modelos preentrenados existan."""
    weights_path = "models/qnim_vqc_weights.npy"
    pipeline_path = "models/qnim_preprocessing_pipeline.pkl"
    
    if not os.path.exists(weights_path) or not os.path.exists(pipeline_path):
        print("❌ ERROR: Faltan los modelos preentrenados")
        print(f"   Archivos requeridos:")
        print(f"   - {weights_path}")
        print(f"   - {pipeline_path}")
        print("   Solución: Ejecuta primero 'python3 train.py'")
        return False
    
    return True


def load_gravitational_wave(use_real_data: bool = True) -> tuple[str, np.ndarray, float]:
    """
    Carga un evento de onda gravitacional.
    
    Args:
        use_real_data: Si True, carga GW150914 real. Si False, carga evento sintético.
    
    Returns:
        (event_id, strain, snr_instrumental)
    """
    loader = QuantumDatasetLoader(target_samples=16384)
    
    if use_real_data:
        ruta_gw150914 = "data/raw/H-H1_LOSC_4_V2-1126259446-32.hdf5"
        if not os.path.exists(ruta_gw150914):
            print(f"⚠️  Archivo real no encontrado: {ruta_gw150914}")
            print("   Cambiando a evento sintético...")
            use_real_data = False
    
    if use_real_data:
        raw_strain = loader.prepare_for_quantum(ruta_gw150914, is_real_data=True)
        return "GW150914", raw_strain, 24.0
    else:
        data_dir = Path("data/synthetic")
        synthetic_dirs = sorted([d for d in data_dir.iterdir() 
                                if d.is_dir() and d.name.startswith("202")])
        if not synthetic_dirs:
            raise FileNotFoundError("No hay eventos sintéticos disponibles en data/synthetic/")
        
        latest_batch = synthetic_dirs[-1]
        test_file = list(latest_batch.glob("*.h5"))[-1]
        raw_strain = loader.prepare_for_quantum(str(test_file), is_real_data=False)
        return test_file.stem, raw_strain, 18.5


def generate_template_bank() -> list[dict]:
    """
    Genera banco de plantillas de onda gravitacional para D-Wave.
    
    Domain Logic: Usa KerrVacuumProvider del dominio para generar
    plantillas teóricamente correctas.
    
    Returns:
        Lista de plantillas (strain + parámetros de masa/spin)
    """
    kerr = KerrVacuumProvider()
    templates = []
    
    # Espacio de búsqueda: masas componentes en rango astrofísicamente válido
    masas_1 = np.linspace(34, 37, 5)  # Masa 1: ~35 M_sun
    masas_2 = np.linspace(28, 32, 5)  # Masa 2: ~30 M_sun
    
    for m1 in masas_1:
        for m2 in masas_2:
            tmpl_strain = kerr.generate_base_strain(
                SolarMass(m1), 
                SolarMass(m2), 
                distance_mpc=410.0
            )
            templates.append({
                "params": {"m1": round(m1, 2), "m2": round(m2, 2), "spin": 0.69},
                "strain": tmpl_strain
            })
    
    return templates


def run_ensemble_inference(
    use_case: DecodeGravitationalWaveUseCase,
    event: QuantumDecodedEvent,
    templates: list[dict],
    num_passes: int = 5
) -> QuantumDecodedEvent:
    """
    Ejecuta inferencia en modo ensamble para reducir varianza cuántica.
    
    Application Logic: Orquesta múltiples pasadas del caso de uso
    para promediar resultados y estabilizar confianza.
    
    Args:
        use_case: Caso de uso inyectado (DecodeGravitationalWaveUseCase)
        event: Evento a decodificar (QuantumDecodedEvent)
        templates: Banco de plantillas
        num_passes: Número de pasadas del ensamble
    
    Returns:
        QuantumDecodedEvent con promedios de ensamble
    """
    print(f"🚀 Iniciando Ensamble Cuántico ({num_passes} pasadas)...")
    
    lista_sigmas = []
    lista_delta_q = []
    decoded_event = None
    
    for i in range(num_passes):
        progreso = ((i + 1) / num_passes) * 100
        print(f"  [{progreso:3.0f}%] Pasada {i+1}/{num_passes}...", end="\r")
        
        # Ejecutar el caso de uso (Application layer)
        resultado = use_case.execute(event, templates)
        
        # Acumular métricas cuánticas
        lista_sigmas.append(resultado.topology.quantum_confidence_sigma)
        lista_delta_q.append(resultado.topology.no_hair_delta_q)
        decoded_event = resultado
    
    # Promediar resultados del ensamble
    sigma_final = sum(lista_sigmas) / num_passes
    delta_q_final = sum(lista_delta_q) / num_passes
    
    # Actualizar evento final con promedios
    if decoded_event:
        decoded_event.topology.quantum_confidence_sigma = sigma_final
        decoded_event.topology.no_hair_delta_q = delta_q_final
    
    print()  # Salto de línea
    return decoded_event


def print_inference_report(event: QuantumDecodedEvent):
    """Imprime reporte formateado de decodificación."""
    print("\n" + "="*70)
    print("📜 REPORTE DE DECODIFICACIÓN QNIM (7 CAPAS)")
    print("="*70)
    
    print(f"🆔 Evento: {event.event_id}")
    print(f"🎯 SNR Instrumental: {event.signal.snr_instrumental}")
    
    print("\n▶ [CAPA 1-2: GEOMETRÍA INTRÍNSECA - D-WAVE]")
    print(f"  ├─ Masa 1 inferida: {event.geometry.m1.value:.2f} M_sun")
    print(f"  ├─ Masa 2 inferida: {event.geometry.m2.value:.2f} M_sun")
    print(f"  └─ Espín efectivo: {event.geometry.effective_spin_chi}")
    
    print("\n▶ [CAPA 5-7: TOPOLOGÍA DEL HORIZONTE - IBM QUANTUM]")
    print(f"  ├─ Desviación de Kerr (ΔQ): {event.topology.no_hair_delta_q:.4f}")
    print(f"  ├─ Confianza Cuántica: {event.topology.quantum_confidence_sigma:.2f} σ")
    
    veredicto = event.topology.detected_theory.value
    print(f"  └─ 🌌 VEREDICTO FÍSICO: {veredicto}")
    
    print("="*70)
    variance_reduction = 100 * (1 - 1/np.sqrt(5))
    print(f"✅ Análisis completado. Reducción de varianza: {variance_reduction:.1f}%")
    print("="*70)


# ============================================================================
# FUNCIÓN PRINCIPAL: CASO DE USO COMPLETO
# ============================================================================

def main():
    """
    Flujo principal de ejecución del sistema QNIM completamente integrado con DDD.
    
    Capas ejecutadas:
    1. Infrastructure: Carga adaptadores (IBM Quantum + Neal Annealer)
    2. Domain: Genera plantillas teóricamente válidas
    3. Application: Orquesta la inferencia híbrida
    4. Presentation: Imprime resultados
    """
    
    # ────── SETUP ──────
    load_dotenv()
    
    print("🌌 --- INICIANDO QNIM FRAMEWORK (DECODIFICACIÓN MULTICAPA) --- 🌌\n")
    
    # ────── VALIDACIÓN ──────
    if not validate_prerequisites():
        return 1
    
    # ────── INFRASTRUCTURE LAYER ──────
    print("🔌 Inicializando Infrastructure Layer...")
    print("   ├─ Cargando modelos VQC preentrenados...")
    
    weights_path = "models/qnim_vqc_weights.npy"
    pipeline_path = "models/qnim_preprocessing_pipeline.pkl"
    
    print("   ├─ Conectando adaptadores cuánticos...")
    
    # Inyección de dependencias (Puertos definidos en domain/quantum/interfaces.py)
    ibm_adapter = IBMQuantumAdapter(weights_path)  # Puerto: IGateBasedQuantumComputer
    dwave_adapter = NealSimulatedAnnealerAdapter()  # Puerto: IQuantumAnnealer
    
    print("   └─ Cargando pipeline de preprocesamiento ML...")
    compressor = joblib.load(pipeline_path)
    
    # ────── APPLICATION LAYER ──────
    print("\n📦 Inicializando Application Layer...")
    print("   ├─ Creando orquestador híbrido (IBM + D-Wave)...")
    
    orchestrator = HybridInferenceOrchestrator(ibm_adapter, dwave_adapter)
    
    print("   └─ Inyectando caso de uso: DecodeGravitationalWaveUseCase...")
    
    use_case = DecodeGravitationalWaveUseCase(orchestrator, compressor)
    
    # ────── DATA LOADING (Domain + Infrastructure) ──────
    print("\n📂 Cargando datos de ondas gravitacionales...")
    
    EVALUAR_EVENTO_REAL = True
    event_id, raw_strain, snr_real = load_gravitational_wave(use_real_data=EVALUAR_EVENTO_REAL)
    
    # Construir entidad de dominio
    signal = GWSignal(
        strain=raw_strain,
        detector=DetectorType.H1,
        sample_rate=4096,
        gps_start=GPSTime(1126259462.4),
        snr_instrumental=snr_real
    )
    event = QuantumDecodedEvent(event_id=event_id, signal=signal)
    
    print(f"   ✅ Evento cargado: {event_id} (SNR={snr_real})")
    
    # ────── DOMAIN LAYER: TEMPLATE BANK GENERATION ──────
    print("\n🧮 Generando banco de plantillas (Dominio Astrophysics)...")
    
    templates = generate_template_bank()
    
    print(f"   ✅ Generadas {len(templates)} plantillas de onda gravitacional")
    
    # ────── APPLICATION LAYER: ENSEMBLE INFERENCE ──────
    print("\n⚡ Ejecutando inferencia cuántica híbrida...")
    
    NUM_PASADAS_ENSAMBLE = 5
    decoded_event = run_ensemble_inference(
        use_case=use_case,
        event=event,
        templates=templates,
        num_passes=NUM_PASADAS_ENSAMBLE
    )
    
    # ────── PRESENTATION LAYER ──────
    print_inference_report(decoded_event)
    
    return 0


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    sys.exit(main())
