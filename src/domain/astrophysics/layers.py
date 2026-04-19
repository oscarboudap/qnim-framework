"""
CAPAS FÍSICAS DE LA INFORMACIÓN GRAVITACIONAL.

Un evento GW codifica 7 capas anidadas de realidad física.
Cada capa es un dominio de observables con incertidumbres y correlaciones.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Set
from .value_objects import (
    TheoryFamily, SolarMass, FrequencyHz, DistanceMPC, GPSTime,
    Spin, Polarization, Inclination, TidalDeformability, QuadrupoleMoment,
    WaveAmplitude, Redshift, GravitonMass, Eccentricity, SignalToNoise,
    Measurement, QuantumProperties
)

# Importar value objects de dominio/metrology
from src.domain.metrology.value_objects import (
    NoHairViolationResult, QuantumGravitySignificance
)


# ============================================================================
# CAPA 1: LA SEÑAL COMO OBJETO MATEMÁTICO
# ============================================================================

@dataclass(frozen=True)
class FrequencyEvolution:
    """Seguimiento de la evolución temporal de la frecuencia."""
    f_0: FrequencyHz                    # Frecuencia de entrada
    f_t: FrequencyHz                    # Frecuencia instantánea
    f_dot: Measurement                  # Chirp rate df/dt
    f_dot_dot: Optional[Measurement] = None  # Segunda derivada


@dataclass(frozen=True)
class PolarizationContent:
    """Propiedades de polarización de la onda GW."""
    ellipticity_e: Measurement          # Elipticidad (0 ≤ e ≤ 1)
    polarization_angle_psi: Polarization  # Ángulo de pol. (0 ≤ ψ < π)
    inclination_iota: Inclination       # Inclinación orbital (0 ≤ ι ≤ π)
    eccentricity_vector: Tuple[float, float] = (0, 0)  # Componentes en X,Y


@dataclass(frozen=True)
class MultipolarDecomposition:
    """Descomposición de la onda en modos de polarización."""
    h_plus: WaveAmplitude               # Polarización +
    h_cross: WaveAmplitude              # Polarización ×
    h_breathing: Optional[WaveAmplitude] = None  # Modo escalar (no-GR)
    h_longitudinal: Optional[WaveAmplitude] = None  # Modo escalar long.
    h_vector_x: Optional[WaveAmplitude] = None  # Modo vectorial X
    h_vector_y: Optional[WaveAmplitude] = None  # Modo vectorial Y
    
    def num_polarizations(self) -> int:
        """Cuenta polarizaciones detectadas (2 en GR, hasta 6 en no-GR)."""
        count = 2  # h_+ y h_×
        if self.h_breathing and self.h_breathing.value > 1e-25: count += 1
        if self.h_longitudinal and self.h_longitudinal.value > 1e-25: count += 1
        if self.h_vector_x and self.h_vector_x.value > 1e-25: count += 1
        if self.h_vector_y and self.h_vector_y.value > 1e-25: count += 1
        return count


@dataclass(frozen=True)
class DetectorCoherence:
    """Coherencia de fase y amplitud entre detectores."""
    h1_l1_coherence: Dict[str, Measurement]  # {t, f, SNR}
    h1_v1_coherence: Dict[str, Measurement]  
    l1_v1_coherence: Dict[str, Measurement]
    triangulation_error: Measurement    # Error de localización en el cielo (sr)


@dataclass(frozen=True)
class SignalMathematicalStructure:  # CAPA 1 REVISADA
    """
    La señal h(t) codifica su propia estructura topológica.
    Antes de interpretar físicamente, extraemos pure mathematical observables.
    """
    # Core strain data
    instantaneous_phase: Measurement    # φ(t) - fase instantánea
    phase_rate: FrequencyEvolution      # dφ/dt = 2πf(t)
    
    # Multipolar content
    polarization: PolarizationContent
    multipoles: MultipolarDecomposition
    
    # Sky localization
    detector_coherence: DetectorCoherence
    
    # Timing and causality
    event_time_gps: GPSTime
    signal_duration_s: Measurement      # Duración en segundos
    
    # Spectral properties
    frequency_resolution: Measurement   # Δf en Hz
    phase_accuracy: Measurement         # Incertidumbre fase (radianes)


# ============================================================================
# CAPA 2: GEOMETRÍA INTRÍNSECA DEL SISTEMA FUENTE
# ============================================================================

@dataclass(frozen=True)
class ChirpMassParameters:
    """Masa de chirp: el observable más robusto de GW."""
    chirp_mass: SolarMass               # M_c
    symmetric_mass_ratio: Measurement   # η = m₁m₂/(m₁+m₂)²
    m1: SolarMass                       # Masa primaria
    m2: SolarMass                       # Masa secundaria
    
    def total_mass(self) -> SolarMass:
        """M_tot = m₁ + m₂."""
        return SolarMass(
            value=self.m1.value + self.m2.value,
            sigma=np.sqrt(self.m1.sigma**2 + self.m2.sigma**2)
        )


@dataclass(frozen=True)
class SpinConfiguration:
    """Configuración de espines del sistema binario."""
    chi_eff: Spin                       # Espín efecto (alineado)
    chi_p: Spin                         # Espín precesante (perpendicular)
    s1_mag: Optional[Spin] = None       # Magnitud del espín del BH1
    s2_mag: Optional[Spin] = None       # Magnitud del espín del BH2
    precession_frequency: Optional[Measurement] = None  # Frecuencia de precesión


@dataclass(frozen=True)
class OrbitalDynamics:
    """Evolución dinámica de la órbita binaria."""
    eccentricity: Eccentricity          # e(t) - excentricidad
    orbital_frequency: FrequencyHz      # f_orb
    harmonic_content: Dict[float, Measurement]  # f_orb, 2f_orb, 3f_orb...
    

@dataclass(frozen=True)
class MatterProperties:
    """Propiedades de materia denso (solo NS)."""
    tidal_deformability_1: Optional[TidalDeformability] = None
    tidal_deformability_2: Optional[TidalDeformability] = None
    tidal_love_numbers: Dict[str, Measurement] = field(default_factory=dict)


@dataclass(frozen=True)
class NoHairTheorem:
    """Violaciones del teorema de no-pelo (nueva física)."""
    quadrupole_moment_q: QuadrupoleMoment  # Q - Q_Kerr
    higher_multipoles: Dict[str, Measurement]  # M3, S3, M4...
    metric_deviations: Dict[str, Measurement]  # δg_μν


@dataclass(frozen=True)
class IntrinsicGeometry:  # CAPA 2 REVISADA
    """
    Manifold 7-dimensional de parámetros astrofísicos.
    Define la "semilla" del sistema previo a radiación GW.
    """
    # Masas y masa de chirp
    masses: ChirpMassParameters
    
    # Spines precesantes
    spins: SpinConfiguration
    
    # Órbitales dinámicas
    orbit: OrbitalDynamics
    
    # Violaciones de no-pelo (testeo de GR)
    no_hair: NoHairTheorem
    
    # Materia (NS)
    matter: Optional[MatterProperties] = None


# ============================================================================
# CAPA 3: ENTORNO ASTROFÍSICO
# ============================================================================

@dataclass(frozen=True)
class AccretionDiskEnvironment:
    """Disco de gas circumbinario y fricción dinámica."""
    disk_mass_fraction: Measurement     # M_disk / M_binary
    disk_viscosity_alpha: Measurement   # α de Shakura-Sunyaev
    phase_dephasing_accumulated: Measurement  # Δ φ por fricción
    mean_motion_resonance_order: Optional[int] = None  # n:m resonancia


@dataclass(frozen=True)
class MagneticFieldEnvironment:
    """Campo magnético intenso (magnetares, NS)."""
    b_field_strength_gauss: Optional[Measurement] = None  # B ~ 10^15 G
    alfven_mode_hybridization: Measurement = field(default_factory=lambda: Measurement(0, 0))
    qnm_splitting: Dict[str, Measurement] = field(default_factory=dict)  # Pares de freq


@dataclass(frozen=True)
class AxionSuperradianceCloud:
    """Nubes bosonicas por superradiancia de Penrose."""
    axion_mass_ev: Optional[Measurement] = None  # m_a en eV/c²
    cloud_growth_timescale: Optional[Measurement] = None  # τ en segundos
    monochromatic_gw_frequency: Optional[FrequencyHz] = None  # f ~ m_a*c²/h
    extraction_efficiency: Measurement = field(default_factory=lambda: Measurement(0, 0))


@dataclass(frozen=True)
class AstrophysicalEnvironment:  # CAPA 3 REVISADA
    """
    Contexto astrofísico rico donde vive el sistema binario.
    De aquí emergen firmas de formación y evolución.
    """
    # Disco de acreción
    accretion_disk: Optional[AccretionDiskEnvironment] = None
    
    # Campo magnético
    magnetic_field: Optional[MagneticFieldEnvironment] = None
    
    # Nubes exóticas
    axion_cloud: Optional[AxionSuperradianceCloud] = None
    
    # Indicador general de "riqueza" ambiental
    environmental_complexity_flag: bool = False


# ============================================================================
# CAPA 4: COSMOLOGÍA Y FÍSICA GLOBAL
# ============================================================================

@dataclass(frozen=True)
class StandardSirenCosmology:
    """Sirenas estándar: dL sin escalera de distancias."""
    luminosity_distance_dl: DistanceMPC  # d_L medido de amplitud
    redshift_z: Redshift                # Redshift cosmológico
    hubble_constant_inference: Optional[Measurement] = None  # H₀
    posterior_hubble: Dict[str, float] = field(default_factory=dict)  # {H0_best, H0_sigma}


@dataclass(frozen=True)
class StochasticGWBackground:
    """Fondo estocástico de ondas gravitacionales."""
    omega_gw_spectrum: Dict[float, Measurement]  # Ω_GW(f) vs frecuencia
    spectrum_index: Measurement         # Índice espectral α
    energy_density_fraction: Measurement  # Ω_gw integral
    sgwb_interpretation: Dict[str, str]  # {source: "BBH+BNS+strings+inflation"}


@dataclass(frozen=True)
class FundamentalConstantViolations:
    """Variación de constantes fundamentales con redshift."""
    graviton_mass_limit: Optional[GravitonMass] = None  # m_g(z)
    speed_of_gw: Optional[Measurement] = None  # c_gw/c
    fine_structure_variation: Optional[Measurement] = None  # Δα/α


@dataclass(frozen=True)
class CosmologicalEvolution:  # CAPA 4 REVISADA
    """
    Información cosmológica y de larga escala.
    Link a energía oscura, materia oscura, y contenido del universo.
    """
    # Sirena estándar (dL, z)
    siren: StandardSirenCosmology
    
    # Fondo estocástico (SGWB)
    stochastic_background: StochasticGWBackground
    
    # Constantes fundamentales
    fundamental_constants: FundamentalConstantViolations
    
    # Topología global
    universe_is_compact: bool = False
    multiple_images_evidence: Optional[Dict[str, Measurement]] = None


# ============================================================================
# CAPA 5: FÍSICA MÁS ALLÁ DEL GR (40+ DIMENSIONES)
# ============================================================================

@dataclass(frozen=True)
class ScalarTensorSignatures:
    """Firmas de teorías escalar-tensorial (Brans-Dicke, Horndeski)."""
    dipolar_emission_flux_phi: Optional[Measurement] = None  # Emisión dipolar
    scalar_charge_difference: Optional[Measurement] = None  # s₁ - s₂
    brans_dicke_omega_bd: Optional[Measurement] = None  # ω_BD
    breathing_mode_amplitude: Optional[WaveAmplitude] = None
    gw_speed_deviation: Optional[Measurement] = None  # c_gw²/c² - 1
    g5_coupling: Optional[Measurement] = None  # G₅(φ,X) ≠ 0


@dataclass(frozen=True)
class ModifiedGravityDispersion:
    """Dispersión de grupo en gravedad f(R) y masiva."""
    massive_graviton_mass: Optional[GravitonMass] = None  # m_g
    group_delay_phase_shift: Optional[Measurement] = None  # δΨ(f)
    low_freq_deceleration: Optional[Measurement] = None  # Chirp invertido
    dispersion_relation_deviation: Optional[Dict[float, Measurement]] = None


@dataclass(frozen=True)
class ExtraDimensionSignatures:
    """Firmas de dimensiones extra (Kaluza-Klein, Randall-Sundrum)."""
    energy_leakage_exponent_n: Optional[int] = None  # h ~ d_L^{-(1+n/2)}
    kk_resonance_frequencies: Optional[Dict[float, Measurement]] = None  # f_KK
    bulk_amplitude_reduction: Optional[Measurement] = None


@dataclass(frozen=True)
class ExoticCompactObjectSignatures:
    """Firmas de ECOs: ecos, reflectividad, timing."""
    echo_time_delay_dt: Optional[Measurement] = None  # Δt ~ ln(M/ℓ_P)
    echo_reflectivity_R: Optional[Measurement] = None  # ℜ
    echo_phase_reference: Optional[Measurement] = None  # φ_ref
    ringdown_decay_exponential: Optional[Measurement] = None
    num_detectable_echoes: Optional[int] = None


@dataclass(frozen=True)
class AxionSuperradianceSignatures:
    """Firmas de nubes de axiones por superradiancia."""
    axion_mass_constraint: Optional[GravitonMass] = None  # m_axion
    monochromatic_cw_signal: Optional[Measurement] = None  # Amplitud CW
    qnm_frequency_shift: Optional[Measurement] = None  # δf_QNM
    cloud_spin_down_rate: Optional[Measurement] = None  # dχ/dt


@dataclass(frozen=True)
class BeyondGRSignatures:  # CAPA 5 REVISADA (40+ dimensiones)
    """
    Vector de evidencia bayesiana de nuevas física.
    Cada componente es un observable medible con incertidumbre.
    """
    # 5.1: Teorías escalar-tensorial
    scalar_tensor: Optional[ScalarTensorSignatures] = None
    
    # 5.2: Gravedad f(R) y masiva
    modified_gravity_dispersion: Optional[ModifiedGravityDispersion] = None
    
    # 5.3: Dimensiones extra
    extra_dimensions: Optional[ExtraDimensionSignatures] = None
    
    # 5.4: Objetos compactos exóticos
    exotic_objects: Optional[ExoticCompactObjectSignatures] = None
    
    # 5.5: Superradiancia de axiones
    axion_superradiance: Optional[AxionSuperradianceSignatures] = None
    
    # Indicador de presencia de nueva física
    beyond_gr_confidence: Measurement = field(default_factory=lambda: Measurement(0, 0))
    preferred_theory_family: Optional[TheoryFamily] = None


# ============================================================================
# CAPA 6: ESTRUCTURA DEL HORIZONTE Y TOPOLOGÍA
# ============================================================================

@dataclass(frozen=True)
class EchoSpectroscopy:
    """Ecos como espectroscopía del horizonte."""
    echo_pattern: Dict[int, Measurement]  # {n_echo: amplitude}
    echo_frequencies: Dict[float, Measurement]  # Espectro de frecuencias
    horizon_structure_type: Optional[str] = None  # "LQG" / "String" / "Gravastar"


@dataclass(frozen=True)
class BMSSoftHairStructure:
    """Simetrías BMS y soft hair del horizonte."""
    supertranslation_charge_q: Optional[Measurement] = None
    superrotation_charge_m: Optional[Measurement] = None
    soft_graviton_correlations: Optional[Dict[float, Measurement]] = None


@dataclass(frozen=True)
class GravitationalMemory:
    """Efecto de memoria de Christodoulou."""
    memory_step_dc_offset: Optional[Measurement] = None
    nonlinear_memory_component: Optional[Measurement] = None  # ν-emission sig
    permanent_displacement: Measurement = field(default_factory=lambda: Measurement(0, 0))


@dataclass(frozen=True)
class HorizonQuantumTopology:  # CAPA 6 REVISADA
    """
    Física en r → r⁺: la frontera del conocimiento.
    Estructura fina del horizonte y topología global.
    """
    # Ecos como ventana al horizonte
    echo_spectroscopy: Optional[EchoSpectroscopy] = None
    
    # BMS y simetrías infinito-dimensionales
    bms_structure: Optional[BMSSoftHairStructure] = None
    
    # Memoria gravitacional (Christodoulou)
    gravitational_memory: Optional[GravitationalMemory] = None
    
    # Test del Teorema de No-Cabello (Metrología Capa 6)
    no_hair_violation: Optional[NoHairViolationResult] = None
    
    # Topología global del universo
    is_spatially_compact: bool = False
    torus_periodicity: Optional[Dict[str, Measurement]] = None


# ============================================================================
# CAPA 7: FÍSICA CUÁNTICA PROFUNDA
# ============================================================================

@dataclass(frozen=True)
class AdSCFTDuality:
    """Física holográfica: QNMs ↔ polos del propagador CFT."""
    qnm_frequencies: Dict[str, Measurement]  # {(l,m,n): f}
    cft_operator_dimension: Optional[Measurement] = None
    anomalous_dimension_deviation: Optional[Measurement] = None


@dataclass(frozen=True)
class QuantumCorrectionsMetric:
    """Correcciones de un-loop a la métrica de Kerr."""
    renormalized_stress_tensor: Optional[Measurement] = None  # ⟨T_μν⟩
    hawking_radiation_backreaction: Optional[Measurement] = None
    planck_scale_corrections: Optional[Dict[str, Measurement]] = None  # O(ℏ) effects


@dataclass(frozen=True)
class LorentzViolation:
    """Violações del grupo de Lorentz (SME de Kostelecký)."""
    sme_coefficients: Dict[str, Measurement]  # {sbar_μν: value}
    birefringence_time_delay: Optional[Measurement] = None  # Δt(h+ vs h×)
    phase_offset_between_polarizations: Optional[Measurement] = None


@dataclass(frozen=True)
class DeepQuantumManifold:  # CAPA 7 REVISADA
    """
    La frontera de lo computable cuánticamente.
    Dualidades holográficas, correcciones cuánticas, violaciones de simetría.
    """
    # Dualidad AdS/CFT
    ads_cft: Optional[AdSCFTDuality] = None
    
    # Correcciones cuánticas O(ℏ)
    quantum_corrections: Optional[QuantumCorrectionsMetric] = None
    
    # Violaciones de Lorentz y SME
    lorentz_violation: Optional[LorentzViolation] = None
    
    # Propiedades cuánticas generales
    quantum_properties: Set[QuantumProperties] = field(default_factory=set)
    
    # Significancia estadística de nueva física (Metrología Capa 7)
    discovery_significance: Optional[QuantumGravitySignificance] = None
    
    # Confianza estadística (legacy)
    quantum_significance: Measurement = field(default_factory=lambda: Measurement(0, 0))


# ============================================================================
# HELPER: Numpy import para cálculos
# ============================================================================
import numpy as np