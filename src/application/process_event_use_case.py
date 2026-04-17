import numpy as np
from src.domain.astrophysics.entities import QuantumDecodedEvent, GWSignal
from src.domain.astrophysics.layers import IntrinsicGeometry, QuantumHorizonTopology
from src.domain.astrophysics.value_objects import SolarMass
from src.domain.metrology.multipole_validator import MultipoleValidator
from src.application.hybrid_orchestrator import HybridInferenceOrchestrator

class DecodeGravitationalWaveUseCase:
    """
    Caso de Uso Principal: "Decodificar las Capas de Realidad de una Onda Gravitacional".
    Aquí es donde las 7 Capas de tu Marco Teórico cobran vida.
    """
    
    def __init__(self, orchestrator: HybridInferenceOrchestrator, pipeline_compressor):
        self.orchestrator = orchestrator
        self.compressor = pipeline_compressor # El StandardScaler+PCA+MinMax
        self.validator = MultipoleValidator(tolerance_threshold=0.05)

    def execute(self, event: QuantumDecodedEvent, search_space_templates: list) -> QuantumDecodedEvent:
        print(f"\n🚀 Iniciando Inferencia Cuántica QNIM para evento: {event.event_id}")
        
        # --- CAPA 2: GEOMETRÍA INTRÍNSECA (D-WAVE) ---
        print("🟢 Ejecutando Rama D-Wave (Recocido Cuántico)...")
        classical_params = self.orchestrator.execute_dwave_branch(event.signal.strain, search_space_templates)
        
        event.geometry = IntrinsicGeometry(
            m1=SolarMass(classical_params['m1']),
            m2=SolarMass(classical_params['m2']),
            chirp_mass=SolarMass(classical_params.get('m_chirp', 30.0)),
            effective_spin_chi=classical_params['spin']
        )

        # --- PREPARACIÓN DIMENSIONAL ---
        # Comprimimos los 16384 puntos a 12 características topológicas
        features = self.compressor.transform([event.signal.strain])

        # --- CAPAS 5 y 6: FÍSICA MÁS ALLÁ DE GR (IBM) ---
        print("🔵 Ejecutando Rama IBM (VQC 12-Cúbits)...")
        quantum_results = self.orchestrator.execute_ibm_branch(features)
        
        # --- EL JUEZ (METROLOGÍA) ---
        print("⚖️ Auditando Teorema de No-Cabello...")
        audit_report = self.validator.evaluate_no_hair_theorem(
            classical_mass=classical_params['m1'], 
            classical_spin=classical_params['spin'], 
            quantum_anomaly_confidence=quantum_results['anomaly_confidence']
        )
        
        event.topology = QuantumHorizonTopology(
            no_hair_delta_q=audit_report['measured_delta_Q'],
            horizon_reflectivity=0.3 if audit_report['no_hair_violation'] else 0.0,
            detected_theory=quantum_results['detected_theory'],
            quantum_confidence_sigma=quantum_results['anomaly_confidence'] * 5.0 # Mapeo a sigma
        )
        
        print(f"✅ Decodificación completada. Veredicto: {audit_report['inferred_theory'].value}")
        return event