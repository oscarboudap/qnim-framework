"""
Capa 7: Correcciones Cuánticas y Gravitación Clásica
SSTG Physics Injection - Quantum Corrections Layer

Modelos físicos:
1. Evaporación de Hawking (pérdida de masa con tiempo)
2. Efectos AdS/CFT (dualidad gauge/gravedad)
3. Paradoja del firewall (AMPS)
4. Radiación de Hawking (backreaction)
5. Entanglement efectos en GW

Referencias:
- Hawking (1974): Black Hole Evaporation
- Maldacena (1997): AdS/CFT correspondence
- Almheiri et al (2013): AMPS firewall paradox
- Unruh (1976): Hawking radiation in curved spacetime
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
from scipy import special as scipy_special
from scipy.fft import fft, ifft, fftfreq, rfft, irfft, rfftfreq
from scipy import integrate
import warnings

warnings.filterwarnings('ignore')

# Constantes físicas [SI]
C_LIGHT = 299792458.0  # m/s
G_CONST = 6.67430e-11  # m³/kg/s²
HBAR = 1.054571817e-34  # J·s
KBOLTZMANN = 1.380649e-23  # J/K
M_SUN = 1.98841e30  # kg
M_PLANCK = np.sqrt(HBAR * C_LIGHT / G_CONST)  # kg
L_PLANCK = np.sqrt(HBAR * G_CONST / C_LIGHT**3)  # m
T_PLANCK = L_PLANCK / C_LIGHT  # s


@dataclass
class QuantumCorrectionParams:
    """Parámetros para inyección de correcciones cuánticas"""
    theory: str  # "Hawking", "AdS-CFT", "Firewall", "None"
    hawking_temperature: float = 0.0  # Temperatura Hawking (K)
    evaporation_rate: float = 0.0  # Tasa de evaporación (-dM/dt en kg/s)
    ads_cft_coupling: float = 0.0  # Constante de acoplamiento AdS/CFT
    entanglement_entropy: float = 0.0  # Escala de entrelazamiento (bits)
    backreaction_strength: float = 0.0  # Intensidad de backreaction
    mode: str = "injection"


class Layer7QuantumCorrectionsInjector:
    """
    Inyector de correcciones cuánticas para más allá de GR.
    
    Implementa:
    - Evaporación de Hawking (masa decreciente)
    - Radiación de Hawking (espectro térmico)
    - AdS/CFT plasma caliente (disipación)
    - AMPS firewall (entanglement wedge)
    - Backreaction de radiación
    """
    
    @staticmethod
    def compute_hawking_temperature(mass_kg: float) -> float:
        """
        Calcula temperatura de Hawking para BH de masa dada.
        
        Fórmula: T_H = (ℏ c³)/(8π k_B G M)
        
        Args:
            mass_kg: Masa del BH en kg
        
        Returns:
            Temperatura Hawking en Kelvin
        """
        numerator = HBAR * C_LIGHT**3
        denominator = 8 * np.pi * KBOLTZMANN * G_CONST * mass_kg
        return numerator / denominator
    
    @staticmethod
    def compute_hawking_luminosity(mass_kg: float) -> float:
        """
        Calcula potencia radiada (luminosidad Hawking).
        
        Fórmula: L_H = (ℏ c⁶)/(15360 π G² M²)
        
        Args:
            mass_kg: Masa BH en kg
        
        Returns:
            Luminosidad Hawking en Watts
        """
        numerator = HBAR * C_LIGHT**6
        denominator = 15360 * np.pi * (G_CONST**2) * (mass_kg**2)
        return numerator / denominator
    
    @staticmethod
    def compute_evaporation_lifetime(mass_kg: float) -> float:
        """
        Calcula tiempo de vida de evaporación completa.
        
        Fórmula: t_evap = (M₀³)/(3 M_p⁴) × (M_p / M_GUT)⁴ × t_Planck
        
        Para BH macroscópicos (M >> M_Planck):
        t_evap ≈ 2.1 × 10⁶⁷ (M / M_sun)³ años
        
        Args:
            mass_kg: Masa BH inicial en kg
        
        Returns:
            Tiempo de evaporación en segundos
        """
        # Construcción: t_evap ~ M³ / M_p⁴ × t_Planck
        t_evap = (mass_kg**3) / (M_PLANCK**4) * T_PLANCK
        return t_evap
    
    @staticmethod
    def inject_hawking_evaporation(
        h_plus: np.ndarray,
        h_cross: np.ndarray,
        fs: float = 16384.0,
        mass_initial: float = 10.0,
        evaporation_rate: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Inyecta efecto de evaporación de Hawking.
        
        Física: La masa del BH decrece lentamente, lo que modifica
        la frecuencia de cuasinormal modes.
        
        d f_QNM / d M = -f_QNM / M (cambio relativo)
        
        Args:
            h_plus, h_cross: Strain GR baseline
            fs: Sampling frequency
            mass_initial: Masa inicial en masas solares
            evaporation_rate: Tasa de cambio de masa (masas_solares/segundo)
        
        Returns:
            h_plus_evap, h_cross_evap, metadata
        """
        mass_kg = mass_initial * M_SUN
        total_time = len(h_plus) / fs
        
        # Temperatura y luminosidad
        t_hawking = Layer7QuantumCorrectionsInjector.compute_hawking_temperature(mass_kg)
        l_hawking = Layer7QuantumCorrectionsInjector.compute_hawking_luminosity(mass_kg)
        lifetime_evap = Layer7QuantumCorrectionsInjector.compute_evaporation_lifetime(mass_kg)
        
        h_plus_out = h_plus.copy()
        h_cross_out = h_cross.copy()
        
        time_array = np.arange(len(h_plus)) / fs
        
        # Masa como función del tiempo: M(t) = M₀ - dM/dt × t
        if evaporation_rate > 0:
            mass_evaporating = mass_initial - evaporation_rate * time_array
            mass_evaporating = np.maximum(mass_evaporating, mass_initial * 0.9)
        else:
            mass_evaporating = np.ones_like(time_array) * mass_initial
        
        # Cambio en frecuencia cuasinormal
        mass_ratio = mass_evaporating / mass_initial
        frequency_shift = mass_ratio
        
        # Aplicar corrección de frecuencia en ringdown
        from scipy import signal as sig
        analytic_signal = sig.hilbert(h_plus)
        amplitude_envelope = np.abs(analytic_signal)
        
        amplitude_normalized = amplitude_envelope / (np.max(amplitude_envelope) + 1e-10)
        ringdown_start = np.where(amplitude_normalized < 0.1)[0]
        if len(ringdown_start) > 0:
            ringdown_idx = ringdown_start[0]
        else:
            ringdown_idx = int(0.8 * len(h_plus))
        
        if ringdown_idx < len(h_plus):
            phase_hplus = np.unwrap(np.angle(analytic_signal))
            phase_shift = np.cumsum(frequency_shift[:ringdown_idx]) / fs
            phase_shift = np.pad(phase_shift, (0, len(h_plus) - len(phase_shift)))
            
            h_plus_out = amplitude_envelope * np.cos(phase_hplus - phase_shift)
        
        metadata = {
            "hawking_temperature_K": t_hawking,
            "hawking_luminosity_W": l_hawking,
            "evaporation_lifetime_s": lifetime_evap,
            "mass_final": mass_evaporating[-1],
            "total_mass_lost_kg": (mass_initial - mass_evaporating[-1]) * M_SUN
        }
        
        return h_plus_out, h_cross_out, metadata
    
    @staticmethod
    def inject_hawking_radiation_spectrum(
        h_plus: np.ndarray,
        h_cross: np.ndarray,
        fs: float = 16384.0,
        mass: float = 10.0,
        temperature_scale: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inyecta radiación térmica de Hawking como ruido adicional.
        
        Física: El espectro de radiación es Planckiano (térmico).
        
        N(f) = 8π h f³/c³ × 1/(exp(hf/kT) - 1)
        
        Args:
            h_plus, h_cross: Strain baseline
            fs: Sampling frequency
            mass: Masa BH en masas solares
            temperature_scale: Factor que multiplica T_Hawking
        
        Returns:
            h_plus_rad, h_cross_rad: Con ruido Hawking agregado
        """
        mass_kg = mass * M_SUN
        t_hawking = Layer7QuantumCorrectionsInjector.compute_hawking_temperature(mass_kg)
        
        t_hawking_effective = t_hawking * temperature_scale
        
        n_samples = len(h_plus)
        
        freqs = rfftfreq(n_samples, 1.0 / fs)
        
        exponent = HBAR * freqs / (KBOLTZMANN * t_hawking_effective)
        exponent = np.minimum(exponent, 100)
        # Avoid division by zero
        exponent = np.maximum(exponent, 0.1)
        thermal_occupation = 1.0 / (np.exp(exponent) - 1 + 1e-10)
        
        # Generate white noise and transform
        noise_white = np.random.normal(0, 1, n_samples)
        noise_rfft = rfft(noise_white)
        
        # Apply thermal spectrum
        noise_rfft *= np.sqrt(thermal_occupation)
        
        # Transform back to real space
        hawking_radiation = irfft(noise_rfft, n=n_samples)
        
        if np.max(np.abs(hawking_radiation)) > 0:
            hawking_radiation = hawking_radiation / np.max(np.abs(hawking_radiation)) * 0.05
        
        h_plus_rad = h_plus + hawking_radiation
        h_cross_rad = h_cross + 0.3 * hawking_radiation
        
        return h_plus_rad, h_cross_rad
    
    @staticmethod
    def inject_ads_cft_viscosity(
        h_plus: np.ndarray,
        h_cross: np.ndarray,
        fs: float = 16384.0,
        coupling_strength: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inyecta efectos AdS/CFT: viscosidad y difusión en plasma dual.
        
        Física: Por dualidad AdS/CFT, fluctuaciones gravitacionales
        en AdS corresponden a correladores 2-punto en CFT fuertemente acoplado.
        
        La viscosidad introduce atenuación: Green's function con amortiguamiento.
        
        Args:
            h_plus, h_cross: Strain baseline
            fs: Sampling frequency
            coupling_strength: Parámetro de acoplamiento AdS/CFT
        
        Returns:
            h_plus_adscft, h_cross_adscft: Con disipación AdS/CFT
        """
        n_samples = len(h_plus)
        time_array = np.arange(n_samples) / fs
        
        tau_dissipation = 1.0 / (coupling_strength + 1e-10)
        dissipation_envelope = np.exp(-time_array / tau_dissipation)
        
        h_plus_diss = h_plus * dissipation_envelope
        h_cross_diss = h_cross * dissipation_envelope
        
        from scipy import signal as sig
        
        cutoff_freq = (fs / 2) * (1.0 - coupling_strength)
        nyquist = fs / 2
        if cutoff_freq <= 0:
            cutoff_freq = nyquist * 0.5
        
        normalized_cutoff = cutoff_freq / nyquist
        normalized_cutoff = np.minimum(normalized_cutoff, 0.99)
        
        b, a = sig.butter(3, normalized_cutoff)
        h_plus_adscft = sig.filtfilt(b, a, h_plus_diss)
        h_cross_adscft = sig.filtfilt(b, a, h_cross_diss)
        
        return h_plus_adscft, h_cross_adscft
    
    @staticmethod
    def inject_entanglement_wedge_effects(
        h_plus: np.ndarray,
        h_cross: np.ndarray,
        fs: float = 16384.0,
        entanglement_entropy: float = 100.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inyecta efectos de cuña de entrelazamiento (entanglement wedge).
        
        Args:
            h_plus, h_cross: Strain baseline
            fs: Sampling frequency
            entanglement_entropy: Entropía de entrelazamiento (bits)
        
        Returns:
            h_plus_ent, h_cross_ent: Con efectos de entrelazamiento
        """
        n_samples = len(h_plus)
        
        info_loss_bits = entanglement_entropy
        info_loss_energy_scale = info_loss_bits * 0.001
        
        noise_white = np.random.normal(0, 1, n_samples)
        
        fft_noise = fft(noise_white)
        freqs = fftfreq(n_samples, 1.0 / fs)
        
        f_safe = np.abs(freqs) + 1e-6
        pink_filter = 1.0 / np.sqrt(f_safe)
        pink_filter = pink_filter / np.max(np.abs(pink_filter))
        
        fft_pink = fft_noise * pink_filter
        pink_noise = np.real(ifft(fft_pink))
        
        pink_noise = pink_noise / (np.std(pink_noise) + 1e-10)
        pink_noise = pink_noise * info_loss_energy_scale * 0.01
        
        h_plus_ent = h_plus + pink_noise * 0.3
        h_cross_ent = h_cross + pink_noise * 0.2
        
        return h_plus_ent, h_cross_ent
    
    @staticmethod
    def apply_quantum_corrections(
        h_plus: np.ndarray,
        h_cross: np.ndarray,
        params: QuantumCorrectionParams,
        mass: float = 10.0,
        fs: float = 16384.0
    ) -> Dict[str, any]:
        """
        Dispatcher principal para inyección de correcciones cuánticas.
        
        Args:
            h_plus, h_cross: Strain GR baseline
            params: QuantumCorrectionParams con configuración
            mass: Masa BH en masas solares
            fs: Sampling frequency
        
        Returns:
            Dict con {"h_plus", "h_cross", "physics_applied", "metadata"}
        """
        
        result = {
            "h_plus": h_plus.copy(),
            "h_cross": h_cross.copy(),
            "physics_applied": [],
            "metadata": {},
            "description": f"Quantum corrections: {params.theory}"
        }
        
        if params.theory.lower() == "hawking":
            h_plus_out, h_cross_out, metadata = Layer7QuantumCorrectionsInjector.inject_hawking_evaporation(
                h_plus, h_cross,
                fs=fs,
                mass_initial=mass,
                evaporation_rate=params.evaporation_rate or 0.0
            )
            result["h_plus"] = h_plus_out
            result["h_cross"] = h_cross_out
            result["metadata"] = metadata
            result["physics_applied"].append("Hawking evaporation")
            
            h_plus_out, h_cross_out = Layer7QuantumCorrectionsInjector.inject_hawking_radiation_spectrum(
                h_plus_out, h_cross_out,
                fs=fs,
                mass=mass,
                temperature_scale=10.0
            )
            result["h_plus"] = h_plus_out
            result["h_cross"] = h_cross_out
            result["physics_applied"].append("Hawking radiation spectrum")
            
        elif params.theory.lower() == "ads-cft" or params.theory.lower() == "adscft":
            h_plus_out, h_cross_out = Layer7QuantumCorrectionsInjector.inject_ads_cft_viscosity(
                h_plus, h_cross,
                fs=fs,
                coupling_strength=params.ads_cft_coupling or 0.05
            )
            result["h_plus"] = h_plus_out
            result["h_cross"] = h_cross_out
            result["physics_applied"].append("AdS/CFT viscosity")
            
            if params.entanglement_entropy > 0:
                h_plus_out, h_cross_out = Layer7QuantumCorrectionsInjector.inject_entanglement_wedge_effects(
                    h_plus_out, h_cross_out,
                    fs=fs,
                    entanglement_entropy=params.entanglement_entropy
                )
                result["h_plus"] = h_plus_out
                result["h_cross"] = h_cross_out
                result["physics_applied"].append("Entanglement wedge effects")
            
        elif params.theory.lower() == "firewall":
            h_plus_out, h_cross_out = Layer7QuantumCorrectionsInjector.inject_hawking_radiation_spectrum(
                h_plus, h_cross,
                fs=fs,
                mass=mass,
                temperature_scale=50.0
            )
            result["h_plus"] = h_plus_out
            result["h_cross"] = h_cross_out
            result["physics_applied"].append("Hawking radiation (firewall)")
            
        else:
            h_plus_out, h_cross_out = Layer7QuantumCorrectionsInjector.inject_hawking_radiation_spectrum(
                h_plus, h_cross,
                fs=fs,
                mass=mass,
                temperature_scale=1.0
            )
            result["physics_applied"].append("Subtle Hawking radiation")
            
            result["h_plus"] = h_plus_out
            result["h_cross"] = h_cross_out
        
        result["description"] = f"Quantum corrections: {params.theory} | Effects: {', '.join(result['physics_applied'])}"
        
        return result
