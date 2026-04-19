"""
Application Service: Primary Use Case - Decode GW Event
========================================================

Pipeline de 7 capas: Transforma evento bruto en evento codificado cuánticamente.

Coordina:
- Capa 1-4: Preparación de datos
- Rama D-Wave (Capa 2): Extrae parámetros clásicos
- Rama IBM (Capas 5-7): Clasifica teoría + computa firmas cuánticas
- Auditoría: No-Hair Theorem validation

IMPORTANTE: Retorna NUEVO evento (immutable), no muta input.
"""

from dataclasses import replace
from typing import List, Optional

import numpy as np

from src.domain.astrophysics.entities import QuantumDecodedEvent
from src.domain.astrophysics.value_objects import SolarMass, Spin
from src.domain.quantum.value_objects import TemplateSignal
from src.domain.metrology.value_objects import NoHairViolationResult
from src.application.hybrid_orchestrator import HybridInferenceOrchestrator
from src.application.dto import (
    InferenceResult,
    ClassificationThresholds,
    InferenceFailedException,
)
from src.application.ports import IPreprocessingPort


class DecodeGravitationalWaveUseCase:
    """
    Caso de Uso: "Decodificar las Capas de Realidad de una Onda Gravitacional".
    
    Coordinaciónlimpia de:
    1. Rama clásica (D-Wave): m1, m2, χ_eff
    2. Rama cuántica (IBM): Teoría + firmas
    3. Metrología: Auditoría No-Cabello
    4. Aggregación: Evento con 7 capas completo
    
    RETORNA: Nuevo evento (input no es mutado).
    """
    
    def __init__(self,
                 orchestrator: HybridInferenceOrchestrator,
                 preprocessing: IPreprocessingPort,
                 thresholds: ClassificationThresholds):
        """
        Args:
            orchestrator: Coordina 2 ramas cuánticas (inyectadas de infraestructura)
            preprocessing: Pipeline clásico (PCA, escalador) vía puerto
            thresholds: Parámetros de clasificación
        """
        self.orchestrator = orchestrator
        self.preprocessor = preprocessing
        self.thresholds = thresholds
    
    def execute(self,
                event: QuantumDecodedEvent,
                templates: List[TemplateSignal]) -> InferenceResult:
        """
        Ejecuta la pipeline de decodificación cuántica de 7 capas.
        
        Args:
            event: Evento gravitacional bruto (QuantumDecodedEvent)
            templates: Lista de template signals para matching
            
        Returns:
            InferenceResult: Objeto tipado con resultados (NEW event también)
            
        Raises:
            InferenceFailedException: Si la pipeline falla
            ValueError: Si templates vacío
        """
        try:
            if not templates:
                raise ValueError("Templates list cannot be empty")
            
            print(f"\n🚀 Iniciando Decodificación Cuántica QNIM: {event.event_id}")
            
            # ============================================================
            # RAMA 1: D-WAVE (CAPA 2: GEOMETRÍA INTRÍNSECA)
            # ============================================================
            print("🟢 Rama D-Wave: Extrayendo parámetros clásicos...")
            
            classic_params = self.orchestrator.execute_dwave_branch(
                target_signal=event.signal.strain,
                templates=templates
            )
            
            # Crear nueva capa de geometría (inmutable)
            from src.domain.astrophysics.layers import IntrinsicGeometry
            new_geometry = IntrinsicGeometry(
                m1=SolarMass(classic_params.m1_solar_masses),
                m2=SolarMass(classic_params.m2_solar_masses),
                chirp_mass=SolarMass(classic_params.chirp_mass_solar_masses),
                effective_spin_chi=Spin(classic_params.effective_spin)
            )
            
            # ============================================================
            # PREPARACIÓN: DIMENSIONAL COMPRESSION
            # ============================================================
            print("🔧 Preparación: Comprimiendo 16384 → 12 dimensiones...")
            
            # Use preprocessing port (no sklearn imports aquí)
            features = self.preprocessor.transform(
                np.array([event.signal.strain])  # Shape: [1, 16384]
            )  # Retorna: [1, 12]
            
            # ============================================================
            # RAMA 2: IBM (CAPAS 5-7: FÍSICA MÁS ALLÁ DE GR)
            # ============================================================
            print("🔵 Rama IBM: Ejecutando VQC 12-Cúbits...")
            
            classification = self.orchestrator.execute_ibm_branch(
                compressed_features=features,
                thresholds=self.thresholds
            )
            
            # ============================================================
            # METROLOGÍA: AUDITORÍA NO-CABELLO
            # ============================================================
            print("⚖️ Metrología: Auditando Teorema de No-Cabello...")
            
            # Importar servicio metrológico  (domain, no application logic)
            from src.domain.metrology.no_hair_audit import NoHairTheoremAnalyzer
            
            no_hair_result: NoHairViolationResult = (
                NoHairTheoremAnalyzer.evaluate_no_hair_theorem(
                    expected_kerr_q=new_geometry.effective_spin_chi.value * 0.5,
                    measured_delta_q=classification.quantum_topology.no_hair_delta_q,
                    confidence=classification.deep_manifold.discovery_confidence_sigma,
                    tolerance_threshold=self.thresholds.no_hair_tolerance
                )
            )
            
            # ============================================================
            # RECONSTRUCCIÓN: 7 CAPAS COMPLETAS
            # ============================================================
            print("🔨 Reconstruyendo evento con 7 capas completas...")
            
            # Crear evento modificado (immutable pattern: nuevo objeto)
            from src.domain.astrophysics.layers import QuantumHorizonTopology
            from src.domain.astrophysics.layers import Capa7DeepQuantumManifold
            
            new_topology = QuantumHorizonTopology(
                no_hair_delta_q=no_hair_result.measured_delta_q,
                horizon_reflectivity=(
                    classification.quantum_topology.horizon_reflectivity
                ),
                echo_delay_ms=classification.quantum_topology.echo_delay_milliseconds,
                detected_theory_family=classification.deep_manifold.discovered_theory_family,
                quantum_confidence_sigma=classification.deep_manifold.discovery_confidence_sigma
            )
            
            new_deep_manifold = Capa7DeepQuantumManifold(
                ads_cft_dual_error=classification.deep_manifold.ads_cft_dual_error,
                theory_verdict=classification.deep_manifold.discovered_theory_family,
                discovery_significance=classification.deep_manifold.discovery_confidence_sigma
            )
            
            # Create new event object (NOT MUTABLE input)
            modified_event = replace(
                event,
                geometry=new_geometry,
                topology=new_topology,
                deep_manifold=new_deep_manifold
            )
            
            # ============================================================
            # RESULTADO FINAL
            # ============================================================
            veredicto = classification.deep_manifold.discovered_theory_family
            print(f"✅ Decodificación completada. Veredicto: {veredicto}")
            
            return InferenceResult(
                event_id=event.event_id,
                classic_parameters=classic_params,
                classification=classification,
                no_hair_violation_detected=no_hair_result.is_violated,
                overall_theory_verdict=veredicto,
                processing_timestamp_gps=event.signal.gps_time.value,
                snr_final=classic_params.template_match_snr
            )
        
        except Exception as e:
            raise InferenceFailedException(
                f"Inference pipeline failed for event {event.event_id}: {str(e)}"
            )