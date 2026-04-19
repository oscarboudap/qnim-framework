"""
Domain: CAPA 5 - Beyond-GR Physics Injection
=============================================

Inyecta firmas de teorías alternativas a GR:

1. **Brans-Dicke Scalar-Tensor Theory**
   - Emisión dipolar extra (reducción de masa orbital)
   - Polarización escalar (+ tensor estándar)
   - Número de campos Brans-Dicke ω

2. **Theories with Extra Dimensions (ADD/RS)**
   - Escape de energía al 'bulk'
   - Atenuación anómala d_L^{-(2+n/2)}
   - n = número de dimensiones extra

3. **Massive Graviton (dRGT Gravity)**
   - Dispersión dependiente de frecuencia
   - Retraso temporal anómalo
   - Masa del gravitón m_g

4. **Modified Dispersion Relations**
   - Lorentz violation (LV)
   - Velocidad de onda no constante
   - Efectos en coalescencia

Basado en:
- Berti, Cardoso, Gualtieri (2015) - Beyond-GR tests with GW
- Mirshekari, Yunes, Will (2012) - Compact Binary Tests of GR
- Yunes & Siemonato (2013) - Modified Dispersion
"""

import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass
from scipy import signal


@dataclass
class BeyondGRParams:
    """Parámetros de inyección beyond-GR."""
    theory: str  # "Brans-Dicke", "ADD", "dRGT", "Lorentz", "none"
    omega_bd: float = 100.0  # Brans-Dicke parameter (ω)
    extra_dims: int = 0  # ADD: número de dimensiones
    graviton_mass: float = 0.0  # dRGT: masa del gravitón [eV]
    lorentz_violation: float = 0.0  # LV: parámetro de violación
    mode: str = "injection"  # "injection" o "phase_shift"


class Layer5BeyondGRInjector:
    """
    Inyector riguroso de física beyond-GR en ondas gravitacionales.
    
    Implementa modificaciones a la fase, amplitud y polarización de ondas GW.
    """
    
    @staticmethod
    def inject_brans_dicke_dipolar(
        h_plus: np.ndarray,
        h_cross: np.ndarray,
        omega_param: float = 100.0,
        total_mass_msun: float = 60.0,
        fs: int = 16384
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inyecta radiación dipolar de Brans-Dicke.
        
        La teoría escalar-tensorial predice:
        - Energía radiada más rápidamente que GR
        - Polarización escalar adicional h_s
        - Corrección: Ψ(f) = Ψ_GR(f) - (3/8) * (omega_param)^{-1} * f^{-1/3}
        
        Args:
            h_plus: Onda + de GR (baseline)
            h_cross: Onda × de GR (baseline)
            omega_param: Parámetro de Brans-Dicke (típicamente 100-500)
            total_mass_msun: Masa total en masas solares
            fs: Frecuencia de sampleo
            
        Returns:
            (h_plus_BD, h_cross_BD) - Ondas modificadas con efectos BD
        """
        # Conversión a frecuencia
        freqs = np.fft.rfftfreq(len(h_plus), d=1/fs)
        freqs = np.maximum(freqs, 10)  # Evitar singularidades
        
        # Transformada de Fourier
        Hp = np.fft.rfft(h_plus)
        Hc = np.fft.rfft(h_cross)
        
        # Modificación de fase por radiación dipolar (Yunes & Siemonato)
        # ΔΨ = -(3/(8π)) * (1 + ω_BD)^{-1} * (π M f)^{-5/3}
        mass_kg = total_mass_msun * 1.989e30
        m_meters = mass_kg * 6.674e-11 / (3e8) ** 2  # GM/c^2
        
        phase_correction = -(3 / (8 * np.pi)) * (1 + omega_param) ** (-1) * \
                          (np.pi * m_meters * freqs) ** (-5 / 3)
        
        phase_factor = np.exp(1j * phase_correction)
        Hp *= phase_factor
        Hc *= phase_factor
        
        # Amplitud reducida (energía escapa como dipolar radiation)
        amplitude_factor = 1.0 - (1 + omega_param) ** (-1) * 0.1
        Hp *= amplitude_factor
        Hc *= amplitude_factor
        
        # Convertir de vuelta
        h_plus_bd = np.fft.irfft(Hp, n=len(h_plus))
        h_cross_bd = np.fft.irfft(Hc, n=len(h_cross))
        
        return h_plus_bd, h_cross_bd
    
    @staticmethod
    def inject_scalar_polarization(
        h_plus: np.ndarray,
        h_cross: np.ndarray,
        amplitude_fraction: float = 0.05,
        fs: int = 16384
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Inyecta polarización escalar extra (en teorías escalar-tensoriales).
        
        En GR solo hay 2 polarizaciones (+ y ×).
        En BD hay 4 (incluye escalar y pseudoescalar).
        
        Args:
            h_plus, h_cross: Polarizaciones tensorialiales
            amplitude_fraction: Amplitud de escalar como fracción de tensor
            fs: Frecuencia de sampleo
            
        Returns:
            (h_plus, h_cross, h_scalar) - Incluida nueva polarización
        """
        # La polarización escalar se comporta como antialiased high-pass filter
        # respecto al tensor
        h_scalar = signal.sosfilt(
            signal.butter(4, 50, 'high', fs=fs, output='sos'),
            h_plus  # Usa el contenido de h+ como base
        )
        
        # Normalizar
        if np.max(np.abs(h_scalar)) > 0:
            h_scalar = h_scalar / np.max(np.abs(h_scalar)) * amplitude_fraction * np.max(np.abs(h_plus))
        
        return h_plus, h_cross, h_scalar
    
    @staticmethod
    def inject_extra_dimensions_leakage(
        h_plus: np.ndarray,
        h_cross: np.ndarray,
        n_extra_dims: int = 1,
        distance_mpc: float = 400.0,
        fs: int = 16384
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Teoría ADD (Arkani-Hamed, Dimopoulos, Dvali):
        Energía escapa a dimensiones extra → amplitud anomalous vs distancia.
        
        Predicción: h ~ d_L^{-(2+n/2)} en lugar de h ~ d_L^{-1}
        
        Args:
            h_plus, h_cross: Ondas baseline
            n_extra_dims: Número de dimensiones extra (típicamente 2-6)
            distance_mpc: Distancia de luminosidad (Mpc)
            fs: Frecuencia de sampleo
            
        Returns:
            (h_plus_add, h_cross_add) - Con efecto ADD
        """
        # Factor de atenuación por escape al bulk
        # Normalizar a 4D (n=0)
        attenuation_gr = 1.0  # d_L^{-1}
        attenuation_add = 1.0 / ((distance_mpc + 1) ** (n_extra_dims / 2))
        
        # Factor de corrección
        correction_factor = attenuation_add / (attenuation_gr + 1e-10)
        
        # En frecuencia, el efecto es predominante en altas frecuencias
        freqs = np.fft.rfftfreq(len(h_plus), d=1/fs)
        
        # Frequency-dependent suppression: LOW frequencies menos afectadas
        freq_weight = 1.0 + 0.5 * n_extra_dims * (freqs / np.max(freqs)) ** 2
        freq_weight = np.minimum(freq_weight, 1.0 / correction_factor)
        
        # Aplicar en Fourier
        Hp = np.fft.rfft(h_plus) * freq_weight * correction_factor
        Hc = np.fft.rfft(h_cross) * freq_weight * correction_factor
        
        h_plus_add = np.fft.irfft(Hp, n=len(h_plus))
        h_cross_add = np.fft.irfft(Hc, n=len(h_cross))
        
        return h_plus_add, h_cross_add
    
    @staticmethod
    def inject_massive_graviton_dispersion(
        h_plus: np.ndarray,
        h_cross: np.ndarray,
        graviton_mass_hmevcmsq: float = 1e-22,  # [eV/c²]
        total_mass_msun: float = 60.0,
        distance_mpc: float = 400.0,
        fs: int = 16384
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gravitón masivo (dRGT - de Rham, Gabadadze, Tolley).
        
        El gravitón tiene relación de dispersión: ω² = c²k² + (m_g*c²/ℏ)²
        
        Efectos observables:
        - Diferentes velocidades de grupo/fase por frecuencia
        - Retraso temporal anómalo: Δt ~ (m_g*c²)² / (2E)
        - Cambio de fase acumulado en la propagación
        
        Args:
            h_plus, h_cross: Ondas baseline
            graviton_mass_hmevcmsq: Masa del gravitón [eV/c²]
            total_mass_msun: Masa total del sistema [M_sun]
            distance_mpc: Distancia de luminosidad [Mpc]
            fs: Frecuencia de sampleo
            
        Returns:
            (h_plus_mg, h_cross_mg) - Con dispersión de gravitón masivo
        """
        # Constantes de conversión
        c = 3e8  # m/s
        G = 6.674e-11  # m³/(kg·s²)
        Mpc_to_m = 3.086e22
        hbar = 1.055e-34  # J·s
        eV_to_J = 1.602e-19
        
        m_g_SI = graviton_mass_hmevcmsq * eV_to_J / (c ** 2)  # En kg
        distance_m = distance_mpc * Mpc_to_m
        mass_kg = total_mass_msun * 1.989e30
        
        # Frecuencias
        freqs = np.fft.rfftfreq(len(h_plus), d=1/fs)
        freqs = np.maximum(freqs, 1)  # Evitar singularidad
        
        # Energía típica de la onda: E ~ ℏω
        # Retraso por masa: Δt(f) ~ (m_g * c²)² / (ℏ ω) = (m_g * c²)² / (h * f)
        # Desfase: Δφ(f) = 2π * f * Δt = 2π * (m_g * c²)² / h
        
        phase_lag = (2 * np.pi * (m_g_SI * c ** 2) ** 2) / (6.626e-34 * freqs)
        
        # Transformadas de Fourier
        Hp = np.fft.rfft(h_plus)
        Hc = np.fft.rfft(h_cross)
        
        # Aplicar dispersión
        phase_factor = np.exp(-1j * phase_lag)
        Hp_dispersed = Hp * phase_factor
        Hc_dispersed = Hc * phase_factor
        
        # También aplicar atenuación (radiación de modo masivo decae más rápido)
        amplitude_factor = 1.0 - 0.05 * graviton_mass_hmevcmsq * 1e22
        Hp_dispersed *= amplitude_factor
        Hc_dispersed *= amplitude_factor
        
        # Convertir de vuelta
        h_plus_mg = np.fft.irfft(Hp_dispersed, n=len(h_plus))
        h_cross_mg = np.fft.irfft(Hc_dispersed, n=len(h_cross))
        
        return h_plus_mg, h_cross_mg
    
    @staticmethod
    def inject_lorentz_violation(
        h_plus: np.ndarray,
        h_cross: np.ndarray,
        violation_parameter: float = 1e-15,  # Parámetro s de Myers-Pospelov
        fs: int = 16384
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Violación de Lorentz (Quantum Gravity Phenomenology).
        
        Gravedad cuántica a escala Planck modifica la geometría a escalas astrofísicas.
        Efectos: birefringencia, rotación de polarización, dispersión anómala.
        
        Args:
            h_plus, h_cross: Polarizaciones
            violation_parameter: s ~ (E_Planck / E_photon)
            fs: Frecuencia de sampleo
            
        Returns:
            (h_plus_lv, h_cross_lv) - Con violación de Lorentz
        """
        # Birefringencia: La velocidad difiere entre + y ×
        # Δv/c ~ s * (f / f_Planck)
        
        freqs = np.fft.rfftfreq(len(h_plus), d=1/fs)
        f_planck = 5.39e44  # Frecuencia de Planck (Hz)
        
        # Velocidad relativa
        relative_velocity_diff = violation_parameter * (freqs / f_planck)
        
        # Retraso temporal: Δt = d * Δv / v²
        delay_times = relative_velocity_diff / (3e8) * 1e-3  # En segundos
        
        # Aplicar retrasos como phase shifts
        Hp = np.fft.rfft(h_plus)
        Hc = np.fft.rfft(h_cross)
        
        phase_plus = np.exp(-2j * np.pi * freqs * delay_times)
        phase_cross = np.exp(2j * np.pi * freqs * delay_times)  # Opuesto
        
        Hp *= phase_plus
        Hc *= phase_cross
        
        # Rotación de polarización
        rotation_angle = violation_parameter * 0.1  # radianes
        Hp_rot = Hp * np.cos(rotation_angle) - Hc * np.sin(rotation_angle)
        Hc_rot = Hp * np.sin(rotation_angle) + Hc * np.cos(rotation_angle)
        
        h_plus_lv = np.fft.irfft(Hp_rot, n=len(h_plus))
        h_cross_lv = np.fft.irfft(Hc_rot, n=len(h_cross))
        
        return h_plus_lv, h_cross_lv
    
    @staticmethod
    def apply_beyond_gr_physics(
        h_plus: np.ndarray,
        h_cross: np.ndarray,
        beyond_gr_params: BeyondGRParams,
        total_mass_msun: float = 60.0,
        distance_mpc: float = 400.0,
        fs: int = 16384
    ) -> Dict[str, np.ndarray]:
        """
        Inyector principal: aplica el modelo beyond-GR seleccionado.
        
        Args:
            h_plus, h_cross: Ondas baseline (GR)
            beyond_gr_params: Parámetros de teoría
            total_mass_msun: Masa total
            distance_mpc: Distancia
            fs: Frecuencia de sampleo
            
        Returns:
            Dict con ondas modificadas y polarizaciones extras si aplica
        """
        result = {
            "h_plus": h_plus.copy(),
            "h_cross": h_cross.copy()
        }
        
        if beyond_gr_params.theory == "none":
            return result
        
        elif beyond_gr_params.theory == "Brans-Dicke":
            h_p, h_c = Layer5BeyondGRInjector.inject_brans_dicke_dipolar(
                h_plus, h_cross,
                omega_param=beyond_gr_params.omega_bd,
                total_mass_msun=total_mass_msun,
                fs=fs
            )
            result["h_plus"] = h_p
            result["h_cross"] = h_c
            
            # Agregar polarización escalar
            _, _, h_s = Layer5BeyondGRInjector.inject_scalar_polarization(
                h_p, h_c, amplitude_fraction=0.05, fs=fs
            )
            result["h_scalar"] = h_s
        
        elif beyond_gr_params.theory == "ADD":
            h_p, h_c = Layer5BeyondGRInjector.inject_extra_dimensions_leakage(
                h_plus, h_cross,
                n_extra_dims=beyond_gr_params.extra_dims,
                distance_mpc=distance_mpc,
                fs=fs
            )
            result["h_plus"] = h_p
            result["h_cross"] = h_c
            result["n_extra_dims"] = beyond_gr_params.extra_dims
        
        elif beyond_gr_params.theory == "dRGT":
            h_p, h_c = Layer5BeyondGRInjector.inject_massive_graviton_dispersion(
                h_plus, h_cross,
                graviton_mass_hmevcmsq=beyond_gr_params.graviton_mass,
                total_mass_msun=total_mass_msun,
                distance_mpc=distance_mpc,
                fs=fs
            )
            result["h_plus"] = h_p
            result["h_cross"] = h_c
            result["m_graviton"] = beyond_gr_params.graviton_mass
        
        elif beyond_gr_params.theory == "Lorentz":
            h_p, h_c = Layer5BeyondGRInjector.inject_lorentz_violation(
                h_plus, h_cross,
                violation_parameter=beyond_gr_params.lorentz_violation,
                fs=fs
            )
            result["h_plus"] = h_p
            result["h_cross"] = h_c
            result["lv_param"] = beyond_gr_params.lorentz_violation
        
        return result
