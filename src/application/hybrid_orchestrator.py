from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from src.domain.astrophysics.value_objects import TheoryFamily
from src.domain.quantum.interfaces import IQuantumAnnealer, IGateBasedQuantumComputer
from src.domain.quantum.qubo_formulator import TemplateMatchingQUBO
from src.domain.quantum.value_objects import VQCTopology, TemplateSignal
from src.application.dto import (
    ClassicParametersExtracted,
    BeyondGRSignature,
    QuantumTopologySignature,
    DeepQuantumManifoldSignature,
    ClassificationResult,
    ClassificationThresholds,
)


class HybridInferenceOrchestrator:
    """
    Orquestador Híbrido de Inferencia Cuántica.
    
    Pipeline de 2 ramas:
    1. D-Wave Branch: Template matching → parámetros clásicos (Capa 2)
    2. IBM Branch: VQC 12-Cúbits → firmas cuánticas (Capas 5-7)
    
    Stateless según DDD. Inyecta thresholds para evitar magic numbers.
    """
    
    def __init__(self,
                 ibm_backend: IGateBasedQuantumComputer,
                 dwave_backend: IQuantumAnnealer,
                 thresholds: ClassificationThresholds):
        """
        Args:
            ibm_backend: Interfaz al procesador cuántico (Qiskit, simulador, etc.)
            dwave_backend: Interfaz al recocedor (Leap, simulador, etc.)
            thresholds: Parámetros numéricos de clasificación
        """
        self.ibm = ibm_backend
        self.dwave = dwave_backend
        self.thresholds = thresholds
        self.topology = VQCTopology.get_standard_topology()

    def execute_dwave_branch(self,
                            target_signal: np.ndarray,
                            templates: List['TemplateSignal']) -> ClassicParametersExtracted:
        """
        Rama D-Wave: Template Matching via QUBO Optimization.
        
        Capa 2 (Geometría Intrínseca): Extrae (m1, m2, χ_eff) minimizando mismatch.
        
        Args:
            target_signal: Strain data [16384] elementos
            templates: List[TemplateSignal] con parámetros conocidos
            
        Returns:
            ClassicParametersExtracted: Parámetros con validación y SNR
            
        Raises:
            ValueError: Si no hay templates válidos
        """
        if not templates:
            raise ValueError("Templates list is empty")
        
        # 1. Domain: Formular el problema de optimización
        qubo_problem = TemplateMatchingQUBO.build_formulation(
            target_signal=target_signal,
            templates=templates,
            penalty_weight=None  # Auto-calcula
        )
        
        # 2. Infrastructure: Resolver QUBO
        solution = self.dwave.solve_qubo(
            linear_terms=qubo_problem.linear_terms,
            quadratic_terms=qubo_problem.quadratic_terms,
            num_reads=self.thresholds.dwave_num_reads
        )
        
        # 3. Extract best match (argmin energy)
        best_idx = max(range(len(templates)), 
                      key=lambda i: solution.get(i, 0))
        best_template = templates[best_idx]
        
        # 4. Retornar encapsulado en VO (type-safe)
        return ClassicParametersExtracted(
            m1_solar_masses=best_template.parameters['m1'],
            m2_solar_masses=best_template.parameters['m2'],
            chirp_mass_solar_masses=best_template.parameters.get(
                'chirp_mass',
                (best_template.parameters['m1'] * best_template.parameters['m2']) ** (3/5) /
                (best_template.parameters['m1'] + best_template.parameters['m2']) ** (1/5)
            ),
            effective_spin=best_template.parameters.get('spin', 0.0),
            template_match_snr=best_template.parameters.get('snr', self.thresholds.template_matching_min_snr),
            selected_template_index=best_idx
        )

    def execute_ibm_branch(self, compressed_features: np.ndarray,
                          thresholds: ClassificationThresholds) -> ClassificationResult:
        """
        Rama IBM: VQC 12-Cúbits → Clasificación de Teoría.
        
        Capas 5-7: Extrae firmas Beyond-GR, No-Cabello, AdS/CFT.
        
        Args:
            compressed_features: Array [1, 12] (PCA features en rango [-π, π])
            thresholds: Parámetros de acoplamiento físico
            
        Returns:
            ClassificationResult: Tipado, con todas las firmas cuánticas
        """
        # 1. Execute circuit on quantum backend
        probs = self.ibm.execute_circuit(self.topology, compressed_features)
        
        # 2. Extract anomaly probability from output state
        # probs = [[P(|0>), P(|1>)], ...] for each measurement
        p_anomaly = float(probs[0][1]) if len(probs) > 0 else 0.0
        
        # Ensure bounds
        p_anomaly = max(0.0, min(1.0, p_anomaly))
        
        # 3. Compute Capa 5: Beyond-GR Signatures
        beyond_gr = BeyondGRSignature(
            dipolar_emission_strength=p_anomaly * thresholds.dipolar_coupling_factor,
            graviton_mass_ev=p_anomaly * thresholds.graviton_mass_factor,
            kk_dimensions_detected=int(
                p_anomaly * thresholds.kk_dimension_scaling
            ) if p_anomaly > 0.5 else 0
        )
        
        # 4. Compute Capa 6: Quantum Topology (No-Hair)
        quantum_topo = QuantumTopologySignature(
            no_hair_delta_q=p_anomaly * 0.2,  # Normalized violation
            horizon_reflectivity=(
                p_anomaly * thresholds.horizon_reflectivity_coupling
                if p_anomaly > 0.7 else 0.0
            ),
            echo_delay_milliseconds=(
                p_anomaly * thresholds.echo_delay_coupling
                if p_anomaly > 0.5 else None
            )
        )
        
        # 5. Compute Capa 7: Deep Quantum Manifold
        theory_verdict = self._decide_theory(p_anomaly)
        deep_manifold = DeepQuantumManifoldSignature(
            ads_cft_dual_error=p_anomaly * thresholds.ads_cft_coupling,
            discovered_theory_family=theory_verdict,
            discovery_confidence_sigma=(
                thresholds.discovery_confidence_base +
                p_anomaly * thresholds.discovery_confidence_scale
            )
        )
        
        # 6. Encapsulate in typed VO
        return ClassificationResult(
            beyond_gr=beyond_gr,
            quantum_topology=quantum_topo,
            deep_manifold=deep_manifold,
            raw_probability_anomaly=p_anomaly
        )
    
    @staticmethod
    def _decide_theory(p_anomaly: float) -> str:
        """
        Mapea anomaly probability a familia de teoría detectada.
        
        Esta es una heurística - la teoría real requiere otros datos.
        """
        if p_anomaly < 0.25:
            return "GENERAL_RELATIVITY"
        elif p_anomaly < 0.45:
            return "SCALAR_TENSOR"
        elif p_anomaly < 0.65:
            return "MODIFIED_GRAVITY"
        elif p_anomaly < 0.80:
            return "EXTRA_DIMENSIONS"
        elif p_anomaly < 0.95:
            return "ECO"
        else:
            return "LOOP_QUANTUM_GRAVITY"