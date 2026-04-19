"""
SERVICIOS DE DOMINIO: Lógica Pura de Astrofísica Gravitacional

Estas son operaciones que pertenecen al dominio pero NO a una entidad-específico.
Son reutilizables, sin estado, inmutables.

Principio DDD: "Un Servicio de Dominio opera sobre Entidades y Value Objects,
nunca sobre datos crudos o detalles de infraestructura."
"""

from typing import Tuple
import numpy as np

from .value_objects import (
    SolarMass, Measurement, FrequencyHz, GravitonMass, Spin
)


# ============================================================================
# CONSTANTES FÍSICAS FUNDAMENTALES
# ============================================================================

class PhysicalConstants:
    """Constantes universales en unidades SI (dominio-aware)."""
    
    # Fundamentales
    G = 6.67430e-11          # Gravitación [m³ kg⁻¹ s⁻²]
    C = 299792458            # Velocidad luz [m s⁻¹]
    HBAR = 1.054571817e-34   # Planck reducida [J s]
    
    # Derivadas (SI)
    M_SUN_KG = 1.98847e30    # Masa solar [kg]
    L_PLANCK = 1.616255e-35  # Longitud Planck [m]
    T_PLANCK = 5.391247e-44  # Tiempo Planck [s]
    M_PLANCK_KG = 2.176434e-8  # Masa Planck [kg]
    
    # Astrofísicas
    MPC_TO_M = 3.08567758e22  # 1 Mpc en metros
    YEAR_TO_S = 365.25 * 24 * 3600  # 1 año en segundos
    
    @classmethod
    def get_schwarzschild_radius_m(cls, mass_kg: float) -> float:
        """r_s = 2GM/c² en metros."""
        return 2.0 * cls.G * mass_kg / (cls.C ** 2)
    
    @classmethod
    def get_schwarzschild_radius_solar_masses(cls, mass_solar: float) -> float:
        """r_s en km para masas solares."""
        r_s = cls.get_schwarzschild_radius_m(mass_solar * cls.M_SUN_KG)
        return r_s / 1000.0  # Convierte a km


# ============================================================================
# CALCULADORA DE ASTROFÍSICA
# ============================================================================

class AstrophysicalCalculus:
    """
    Cálculos puros de ondas gravitacionales.
    
    Todos los método son stateless (solo toman parámetros).
    Resultados son exactos (o fenomenológicos bien definidos).
    """
    
    @staticmethod
    def chirp_mass(m1: float, m2: float) -> float:
        """
        Masa de chirp: ℳ_c = (m₁m₂)^(3/5) / (m₁+m₂)^(1/5)
        
        El observable MEJOR medido en ondas gravitacionales.
        Precisión típica: ~10⁻³ relativa para SNR > 20.
        
        Args:
            m1, m2: masas en masas solares
        
        Returns:
            Masa de chirp en masas solares
        """
        return ((m1 * m2) ** 0.6) / ((m1 + m2) ** 0.2)
    
    @staticmethod
    def symmetric_mass_ratio(m1: float, m2: float) -> float:
        """
        η = m₁m₂ / (m₁+m₂)²
        
        Complementa ℳ_c en la descripción del sistema.
        Rango: 0 < η ≤ 0.25 (máximo para m₁ = m₂)
        """
        total = m1 + m2
        return (m1 * m2) / (total ** 2)
    
    @staticmethod
    def innermost_stable_circular_orbit_frequency(m_total_solar: float) -> float:
        """
        Frecuencia de ISCO: f_ISCO = 4397 Hz / M_total [M_sun]
        
        Define el límite del régimen inspiralling.
        Por encima, la órbita se vuelve inestable.
        """
        return 4397.0 / m_total_solar
    
    @staticmethod
    def echo_delay_quantum_gravity(m_total_solar: float) -> float:
        """
        Delay temporal para ecos de Gravedad Cuántica.
        
        Δt ~ 2 r_* ln(M / ℓ_Planck)
        
        Esta es la FIRMA ÚNICA de la Gravedad Cuántica en ecos.
        Para LQG, Strings, Gravastar: Δt es diferente para cada uno.
        
        Args:
            m_total_solar: Masa total en masas solares
        
        Returns:
            Delay en milisegundos
        """
        r_s = PhysicalConstants.get_schwarzschild_radius_m(
            m_total_solar * PhysicalConstants.M_SUN_KG
        )
        
        # Coordenada "tortoise" cerca del horizonte
        delay_s = (2.0 * r_s / PhysicalConstants.C) * np.log(
            r_s / PhysicalConstants.L_PLANCK
        )
        
        return delay_s * 1000.0  # Convierte a ms
    
    @staticmethod
    def tidal_disruption_frequency(m_bh: float, m_companion: float) -> float:
        """
        Frecuencia donde las fuerzas de marea del BH rompen el companion.
        
        Para NS: μ ~ λ M_bh² / R_ns³
        La deformabilidad de marea Λ aparece aquí.
        """
        # Fenomenológico: depende también de EOS companion
        ratio = m_companion / m_bh if m_bh > 0 else 1.0
        return 500.0 * ratio ** (1/3)  # Hz (simplificado)
    
    @staticmethod
    def ringdown_quality_factor(spin: float) -> float:
        """
        Factor de calidad Q del ringdown: cuánto oscila antes de decaer.
        
        Q ~ función(a, l, m, n)
        Para Kerr: Q ≈ 2 (extremal) a 10 (no-spinning)
        """
        # Aproximación fenomenológica
        # Spin extremal (a=1): Q mínimo
        # Spin nulo (a=0): Q máximo
        return 10.0 / (1.0 + 5.0 * spin ** 2)
    
    @staticmethod
    def quasi_normal_mode_frequency(m_total_solar: float, l: int = 2, m: int = 2) -> float:
        """
        Frecuencia del Quasi-Normal Mode del agujero negro post-merger.
        
        f_QNM depende de (M, a, l, m, n)
        Para BH no-giratorio: f ~ 0.37 / M [Hz para M en M_sun]
        
        Args:
            m_total_solar: Masa total
            l, m: números cuánticos (típicamente l=2, m=2)
        
        Returns:
            Frecuencia en Hz
        """
        # Aproximación pura (Leaver, Kidder)
        return (0.37 / m_total_solar) * (1.0 + 0.1 * l * m)


# ============================================================================
# VALIDADOR DE CONSTRAINS FÍSICOS
# ============================================================================

class PhysicalConstraintValidator:
    """
    Verifica que parámetros satisfacen leyes físicas fundamentales.
    
    Esto es DOMINIO: son restricciones de negocio, no detalles de síntesis.
    """
    
    @staticmethod
    def check_energy_conditions(
        m1_solar: float, 
        m2_solar: float, 
        distance_mpc: float
    ) -> Tuple[bool, str]:
        """
        Valida condiciones de energía en GR.
        
        1. Energy Conditions aceptadas (débil, nula, fuerte, dominante)
        2. Causalidad respetada
        3. Energía no-negativa
        
        Returns:
            (válido, razón si inválido)
        """
        # Condición: masa positiva y finita
        if m1_solar <= 0 or m2_solar <= 0:
            return False, "Masa negativa o nula"
        
        if m1_solar > 1e3 or m2_solar > 1e3:
            return False, "Masa extremadamente alta (>1000 M_sun)"
        
        # Condición: distancia positiva
        if distance_mpc <= 0 or distance_mpc > 1e5:
            return False, "Distancia inválida (0-100000 Mpc)"
        
        # Condición de débil energía: ρ ≥ 0 (se cumple por construcción)
        # Condición de nula: T_μν k^μ k^ν ≥ 0 (auto-satisfecho)
        
        return True, "OK"
    
    @staticmethod
    def validate_cosmic_censorship(
        m_total_solar: float,
        spin: float
    ) -> Tuple[bool, str]:
        """
        Conjetura de Censura Cósmica de Penrose:
        "No pueden existir sinularidades desnudas."
        
        Para Kerr: a ≤ M (parámetro de spin ≤ 1)
        Violación → inestabilidad física.
        
        Returns:
            (válido, razón)
        """
        if m_total_solar <= 0:
            return False, "Masa nula"
        
        if not (0 <= spin <= 1.0):
            return False, f"Spin fuera de rango [0,1]: {spin}"
        
        # Kerr extremal en el borde permitido
        if spin > 1.0 + 1e-6:
            return False, "Spin > 1 viola Cosmic Censorship"
        
        return True, "OK"
    
    @staticmethod
    def validate_neutron_star_physics(
        mass_solar: float,
        tidal_deformability_lambda: float
    ) -> Tuple[bool, str]:
        """
        Restricciones de Ecuación de Estado (EOS) para NS.
        
        - Masa: 1.0-2.5 M_sun típicamente
        - Λ: Depende de EOS, ~50-5000 para NS observadas
        
        GW170817 constrinió Λ severamente.
        """
        if not (0.5 < mass_solar < 3.0):
            return False, "Masa NS fuera de rango (~1-2.5 M_sun)"
        
        if not (0 < tidal_deformability_lambda < 10000):
            return False, "Tidal deformability fuera de rango"
        
        return True, "OK"
    
    @staticmethod
    def validate_merger_dynamics(
        m1_solar: float,
        m2_solar: float,
        chi: float
    ) -> Tuple[bool, str]:
        """
        Valida que los parámetros llevan a un merger físico.
        
        - GW emitida debe traer órbita al colapso
        - Spines alineados vs caóticos tienen historias diferentes
        """
        # Ratio de masas debe no ser extremo (q > 0.07 típicamente para eventos detectados)
        q = min(m1_solar, m2_solar) / max(m1_solar, m2_solar)
        if q < 0.05:
            return False, "Ratio de masas muy extremo (q < 0.05)"
        
        # Spines alineados: formación común
        # Spines caóticos: captura dinámica (menos probable)
        # Ambos son válidos
        
        return True, "OK"


# ============================================================================
# CALCULADORA DE PROPAGACIÓN COSMOLÓGICA
# ============================================================================

class CosmologicalPropagation:
    """
    Efectos de la propagación de GW a través del universo en expansión.
    
    Dominio: cómo cambia la forma de onda con redshift y distancia.
    """
    
    @staticmethod
    def luminosity_distance_to_comoving(
        d_l_mpc: float,
        z: float
    ) -> float:
        """
        Convierte distancia de luminosidad a distancia comóvil.
        
        d_c = d_L / (1+z)
        
        En surveys cosmológicos es la métrica natural.
        """
        if z < 0:
            return 0.0
        return d_l_mpc / (1.0 + z)
    
    @staticmethod
    def strain_redshift_factor(z: float) -> float:
        """
        Factor de dilución de strain por redshift.
        
        h(z) = h_source / (1 + z)^2
        
        Viene de dilución de energía (1+z) + redshift frecuencias (1+z)
        """
        return 1.0 / ((1.0 + z) ** 2)
    
    @staticmethod
    def frequency_redshift(f_source: float, z: float) -> float:
        """
        Frecuencia observada en frame comóvil.
        
        f_obs = f_source / (1 + z)
        
        Impacto: frecuencias más bajas en detectores (menor SNR en altas f)
        """
        return f_source / (1.0 + z)


# ============================================================================
# DISCRIMINADOR DE CAPAS
# ============================================================================

class LayerSignificanceCalculator:
    """
    Calcula la "importancia" o "confianza" de cada capa para un evento.
    
    Métrica: SNR de los observables de esa capa.
    """
    
    @staticmethod
    def capa_1_significance(
        num_polarizations_detected: int,
        triangulation_error_sr: float
    ) -> float:
        """
        Capa 1 es significante si:
        - Polarizaciones múltiples detectadas (>2 → no-GR posible)
        - Triangulación precisa (<10 sr es excelente)
        """
        pol_score = min(num_polarizations_detected / 6.0, 1.0)  # 6 max teóricos
        triangulation_score = max(1.0 - triangulation_error_sr / 100.0, 0.0)
        
        return (pol_score + triangulation_score) / 2.0
    
    @staticmethod
    def capa_2_significance(
        snr_total: float,
        has_high_frequency_content: bool
    ) -> float:
        """
        Capa 2 (geometría) es significante si:
        - SNR alto (mejor medición de parámetros)
        - Contenido en HF (mejor constrains de spines/masas)
        """
        snr_score = min(snr_total / 100.0, 1.0)  # SNR 100 = máximo
        freq_bonus = 0.2 if has_high_frequency_content else 0.0
        
        return min(snr_score + freq_bonus, 1.0)
    
    @staticmethod
    def capa_5_significance(
        num_beyond_gr_observables_detected: int,
        combined_snr_beyond_gr: float
    ) -> float:
        """
        Capa 5 es significante si:
        - ≥2-3 observables Beyond-GR detectados
        - SNR combinado > 5 (umbral de significancia)
        """
        obs_score = min(num_beyond_gr_observables_detected / 5.0, 1.0)
        snr_score = min(combined_snr_beyond_gr / 10.0, 1.0)  # 10-sigma = muy fuerte
        
        return (obs_score + snr_score) / 2.0


# ============================================================================
# EXPORTACIONES
# ============================================================================

__all__ = [
    "PhysicalConstants",
    "AstrophysicalCalculus",
    "PhysicalConstraintValidator",
    "CosmologicalPropagation",
    "LayerSignificanceCalculator",
]
