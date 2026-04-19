"""
ENTIDADES DE DOMINIO: Agregados, Root Entities, y Servicios de Dominio.

La anatomía completa de un evento GW decodificado en sus 7 capas,
con inferencia bayesiana multi-modelo y cálculo de evidencias.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from abc import ABC, abstractmethod
import numpy as np

from .layers import (
    SignalMathematicalStructure, IntrinsicGeometry, AstrophysicalEnvironment,
    CosmologicalEvolution, BeyondGRSignatures, HorizonQuantumTopology,
    DeepQuantumManifold
)
from .value_objects import (
    TheoryFamily, Measurement, SignalToNoise, SolarMass, Redshift
)


# ============================================================================
# RAÍZ DEL AGREGADO: Evento GW Decodificado
# ============================================================================

@dataclass
class QuantumDecodedEvent:
    """
    ENTIDAD RAÍZ (Aggregate Root).
    
    Representa la decodificación completa y multicapa de un evento GW.
    Encapsula toda la información astrofísica extraíble de h(t).
    
    Invariantes:
    - event_id único dentro del detector-network
    - snr_total ≥ 8 para estar en catálogo público
    - todas las 7 capas se infieren juntas respetando correlaciones
    """
    
    # Identidad del evento
    event_id: str
    detector_network: Set[str]  # {"H1", "L1", "V1", ...}
    snr_total: SignalToNoise
    
    # Las 7 capas de información
    signal_math: SignalMathematicalStructure
    geometry: IntrinsicGeometry
    environment: Optional[AstrophysicalEnvironment] = None
    cosmology: Optional[CosmologicalEvolution] = None
    beyond_gr: Optional[BeyondGRSignatures] = None
    horizon_topology: Optional[HorizonQuantumTopology] = None
    deep_quantum: Optional[DeepQuantumManifold] = None
    
    # Timestamps
    gps_time_created: float = 0
    last_inference_update: float = 0
    
    # Metadata de inferencia
    inference_version: str = "7-layer-postdoc-v1"
    data_quality_flags: Dict[str, bool] = field(default_factory=dict)
    
    def get_full_evidence_vector(self) -> np.ndarray:
        """
        Extrae el vector de evidencia bayesiana de 40+ dimensiones.
        
        Estructura:
        [signal_math: 15 dims] 
        [geometry: 10 dims] 
        [environment: 6 dims] 
        [cosmology: 8 dims] 
        [beyond_gr: 40+ dims] 
        [horizon: 8 dims] 
        [deep_quantum: 6 dims]
        
        Returns:
            np.ndarray de forma (N_dims, 2) donde N_dims es la dimensión
            y cada fila es [value, sigma] del observable.
        """
        dimensions = []
        
        # ===== CAPA 1: Signal Mathematical (15 dims)
        if self.signal_math:
            dimensions.append(("phase_value", self.signal_math.instantaneous_phase))
            dimensions.append(("f_0", self.signal_math.phase_rate.f_0))
            dimensions.append(("f_dot", self.signal_math.phase_rate.f_dot))
            
            # Multipoles
            h_plus = self.signal_math.multipoles.h_plus
            h_cross = self.signal_math.multipoles.h_cross
            dimensions.append(("h_plus_value", h_plus))
            dimensions.append(("h_cross_value", h_cross))
            
            # Polarization
            pol = self.signal_math.polarization
            dimensions.append(("ellipticity", pol.ellipticity_e))
            dimensions.append(("psi_angle", pol.polarization_angle_psi))
            dimensions.append(("iota_angle", pol.inclination_iota))
            
            # Extra polarizations (if detected)
            if self.signal_math.multipoles.h_breathing:
                dimensions.append(("h_breathing", self.signal_math.multipoles.h_breathing))
            if self.signal_math.multipoles.h_longitudinal:
                dimensions.append(("h_longitudinal", self.signal_math.multipoles.h_longitudinal))
        
        # ===== CAPA 2: Intrinsic Geometry (10 dims)
        if self.geometry:
            masses = self.geometry.masses
            dimensions.append(("M_chirp", masses.chirp_mass))
            dimensions.append(("M_total", masses.total_mass()))
            dimensions.append(("m1", masses.m1))
            dimensions.append(("m2", masses.m2))
            dimensions.append(("eta", masses.symmetric_mass_ratio))
            
            spins = self.geometry.spins
            dimensions.append(("chi_eff", spins.chi_eff))
            dimensions.append(("chi_p", spins.chi_p))
            
            orbit = self.geometry.orbit
            dimensions.append(("eccentricity", orbit.eccentricity))
            
            if self.geometry.matter and self.geometry.matter.tidal_deformability_1:
                dimensions.append(("Lambda_1", self.geometry.matter.tidal_deformability_1))
        
        # ===== CAPA 3: Environment (6 dims)
        if self.environment:
            if self.environment.accretion_disk:
                disk = self.environment.accretion_disk
                dimensions.append(("disk_mass_fraction", disk.disk_mass_fraction))
                dimensions.append(("phase_dephasing", disk.phase_dephasing_accumulated))
            
            if self.environment.axion_cloud:
                axion = self.environment.axion_cloud
                if axion.axion_mass_ev:
                    dimensions.append(("m_axion", axion.axion_mass_ev))
                if axion.extraction_efficiency.value > 0:
                    dimensions.append(("axion_efficiency", axion.extraction_efficiency))
        
        # ===== CAPA 4: Cosmology (8 dims)
        if self.cosmology:
            siren = self.cosmology.siren
            dimensions.append(("d_L", siren.luminosity_distance_dl))
            dimensions.append(("z_redshift", siren.redshift_z))
            
            if siren.hubble_constant_inference:
                dimensions.append(("H0_inferred", siren.hubble_constant_inference))
            
            sgwb = self.cosmology.stochastic_background
            dimensions.append(("Omega_GW_index", sgwb.spectrum_index))
        
        # ===== CAPA 5: Beyond GR (40+ dims - el corazón)
        if self.beyond_gr:
            bgr = self.beyond_gr
            
            # Scalar-tensor (5 dims)
            if bgr.scalar_tensor:
                st = bgr.scalar_tensor
                if st.dipolar_emission_flux_phi:
                    dimensions.append(("dipolar_phi", st.dipolar_emission_flux_phi))
                if st.gw_speed_deviation:
                    dimensions.append(("v_gw_deviation", st.gw_speed_deviation))
                if st.breathing_mode_amplitude:
                    dimensions.append(("h_breathing_scalar_tensor", st.breathing_mode_amplitude))
            
            # Modified gravity (6 dims)
            if bgr.modified_gravity_dispersion:
                mg = bgr.modified_gravity_dispersion
                if mg.massive_graviton_mass:
                    dimensions.append(("m_graviton", mg.massive_graviton_mass))
                if mg.group_delay_phase_shift:
                    dimensions.append(("group_delay", mg.group_delay_phase_shift))
            
            # Extra dimensions (5 dims)
            if bgr.extra_dimensions:
                ed = bgr.extra_dimensions
                if ed.energy_leakage_exponent_n:
                    dimensions.append(("KK_exponent_n", Measurement(float(ed.energy_leakage_exponent_n), 0.1)))
                if ed.bulk_amplitude_reduction:
                    dimensions.append(("bulk_reduction", ed.bulk_amplitude_reduction))
            
            # ECOs (8 dims)
            if bgr.exotic_objects:
                eco = bgr.exotic_objects
                if eco.echo_time_delay_dt:
                    dimensions.append(("echo_dt", eco.echo_time_delay_dt))
                if eco.echo_reflectivity_R:
                    dimensions.append(("echo_R", eco.echo_reflectivity_R))
                if eco.ringdown_decay_exponential:
                    dimensions.append(("ringdown_tau", eco.ringdown_decay_exponential))
            
            # Axions (5 dims)
            if bgr.axion_superradiance:
                axion = bgr.axion_superradiance
                if axion.axion_mass_constraint:
                    dimensions.append(("axion_mass_constraint", axion.axion_mass_constraint))
                if axion.qnm_frequency_shift:
                    dimensions.append(("QNM_shift_from_axion", axion.qnm_frequency_shift))
            
            dimensions.append(("beyond_gr_confidence", bgr.beyond_gr_confidence))
        
        # ===== CAPA 6: Horizon Topology (8 dims)
        if self.horizon_topology:
            ht = self.horizon_topology
            
            if ht.echo_spectroscopy:
                echo = ht.echo_spectroscopy
                dimensions.append(("num_echoes", Measurement(float(len(echo.echo_pattern)), 0.5)))
            
            if ht.bms_structure:
                bms = ht.bms_structure
                if bms.supertranslation_charge_q:
                    dimensions.append(("BMS_charge_q", bms.supertranslation_charge_q))
            
            if ht.gravitational_memory and ht.gravitational_memory.memory_step_dc_offset:
                dimensions.append(("memory_step", ht.gravitational_memory.memory_step_dc_offset))
        
        # ===== CAPA 7: Deep Quantum (6 dims)
        if self.deep_quantum:
            dq = self.deep_quantum
            
            if dq.ads_cft and dq.ads_cft.anomalous_dimension_deviation:
                dimensions.append(("CFT_anomaly_dim", dq.ads_cft.anomalous_dimension_deviation))
            
            if dq.quantum_corrections and dq.quantum_corrections.renormalized_stress_tensor:
                dimensions.append(("renorm_stress_tensor", dq.quantum_corrections.renormalized_stress_tensor))
            
            if dq.lorentz_violation and dq.lorentz_violation.birefringence_time_delay:
                dimensions.append(("lorentz_birefringence", dq.lorentz_violation.birefringence_time_delay))
            
            dimensions.append(("quantum_significance", dq.quantum_significance))
        
        # Convierte a array: [N_observables, 2] donde columna 0 es valor, columna 1 es sigma
        evidence_array = np.zeros((len(dimensions), 2), dtype=float)
        for i, (name, measurement) in enumerate(dimensions):
            evidence_array[i, 0] = measurement.value
            evidence_array[i, 1] = measurement.sigma
        
        return evidence_array
    
    @property
    def inferred_theory(self) -> TheoryFamily:
        """Teoría más probable según Beyond-GR signatures."""
        if self.beyond_gr and self.beyond_gr.preferred_theory_family:
            return self.beyond_gr.preferred_theory_family
        return TheoryFamily.KERR_VACUUM
    
    def num_layers_populated(self) -> int:
        """Retorna cuántas de las 7 capas están pobladas."""
        count = 2  # signal_math y geometry siempre presentes
        if self.environment: count += 1
        if self.cosmology: count += 1
        if self.beyond_gr: count += 1
        if self.horizon_topology: count += 1
        if self.deep_quantum: count += 1
        return count
    
    def completeness_score(self) -> float:
        """Retorna 0-1 indicando cuán completo está el análisis."""
        return self.num_layers_populated() / 7.0


# ============================================================================
# ESPECIFICADOR DE FILTRO: para consultas de repositorio
# ============================================================================

@dataclass
class GWEventSpecification:
    """
    Criterios de búsqueda para eventos GW en repositorio.
    """
    min_snr: Optional[float] = None
    min_chirp_mass: Optional[float] = None
    max_chirp_mass: Optional[float] = None
    tidal_deformability_required: bool = False
    beyond_gr_confidence_min: Optional[float] = None
    theory_families: Optional[Set[TheoryFamily]] = None
    has_echoes: bool = False


# ============================================================================
# INTERFACES DE DOMINIO: Acuerdos de cálculo
# ============================================================================

class BayesianEvidenceCalculator(ABC):
    """
    Interfaz para calcular evidencia bayesiana de una teoría
    dado un evento GW.
    """
    
    @abstractmethod
    def compute_log_evidence(
        self, 
        event: QuantumDecodedEvent,
        theory: TheoryFamily
    ) -> Tuple[float, float]:
        """
        Args:
            event: Evento GW con todas sus capas
            theory: Teoría a testear
        
        Returns:
            (log_Z, sigma_log_Z): evidencia integrada y su incertidumbre
        """
        pass
    
    @abstractmethod
    def compute_likelihood(
        self, 
        event: QuantumDecodedEvent,
        theory: TheoryFamily
    ) -> float:
        """
        Likelihoood L(data | theory) normalizado a [0,1].
        """
        pass


class TheoryDiscriminator(ABC):
    """
    Interfaz para discriminar entre pares de teorías.
    """
    
    @abstractmethod
    def bayes_factor(
        self, 
        event: QuantumDecodedEvent,
        theory_a: TheoryFamily,
        theory_b: TheoryFamily
    ) -> float:
        """
        B_AB = Z_A / Z_B (razon de evidencias).
        """
        pass


class LayerAnalyzer(ABC):
    """
    Interfaz para analizar una capa específica.
    """
    
    @abstractmethod
    def extract_observables(self, event: QuantumDecodedEvent) -> Dict[str, Measurement]:
        """Extrae los observables relevantes de la capa."""
        pass
    
    @abstractmethod
    def compute_layer_significance(self, event: QuantumDecodedEvent) -> float:
        """Retorna SNR-like del contenido de la capa (0-1)."""
        pass


# ============================================================================
# SERVICIO DE DOMINIO: Inferencia Multicapa
# ============================================================================

@dataclass
class MultiLayerInferenceService:
    """
    SERVICIO DE DOMINIO.
    
    Coordina la inferencia coherente a través de las 7 capas.
    Respeta correlaciones, propaga incertidumbres, y produce
    un posterior multi-modelo.
    """
    
    evidence_calculator: BayesianEvidenceCalculator
    discriminator: TheoryDiscriminator
    layer_analyzers: Dict[int, LayerAnalyzer]  # {1-7: analyzer}
    
    def infer_all_layers(self, event: QuantumDecodedEvent) -> Dict[str, float]:
        """
        Realiza inferencia coherente de todas las capas.
        
        Returns:
            Dict con posterior probabilities sobre teorías:
            {theory_name: posterior_probability}
        """
        # Calcula evidencia para cada teoría
        log_evidences = {}
        for theory in TheoryFamily:
            log_z, sigma = self.evidence_calculator.compute_log_evidence(event, theory)
            log_evidences[theory.name] = log_z
        
        # Convierte a probabilidades normalizadas
        log_z_array = np.array(list(log_evidences.values()))
        log_z_max = np.max(log_z_array)
        
        # Evita overflow: exp(log_z - log_z_max)
        relative_evidences = np.exp(log_z_array - log_z_max)
        posterior = relative_evidences / np.sum(relative_evidences)
        
        result = {}
        for i, theory_name in enumerate(log_evidences.keys()):
            result[theory_name] = float(posterior[i])
        
        return result
    
    def assess_layer_quality(self, event: QuantumDecodedEvent, layer_number: int) -> float:
        """
        Retorna calidad (0-1) de la capa especificada.
        """
        if layer_number not in self.layer_analyzers:
            return 0.0
        
        return self.layer_analyzers[layer_number].compute_layer_significance(event)


# ============================================================================
# FACTORY: Constructor de eventos coherentes
# ============================================================================

class QuantumDecodedEventFactory:
    """
    FACTORY para crear eventos bien-formados que respetan invariantes.
    """
    
    @staticmethod
    def create_from_raw_strain(
        event_id: str,
        detector_network: Set[str],
        snr_total: SignalToNoise,
        # ... raw h(t) data ...
    ) -> QuantumDecodedEvent:
        """Factory method que construye todas las 7 capas coherentemente."""
        # Placeholder: en infraestructura habría adaptador a datos reales
        pass