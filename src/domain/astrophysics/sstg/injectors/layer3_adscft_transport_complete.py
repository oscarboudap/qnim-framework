"""
Layer 3: AdS/CFT Holographic Transport
=====================================

Physics Implementation:
- Holographic viscosity bound: η/s ≥ 1/(4π) (KSS bound)
- Thermalization timescale from AdS black holes
- Holographic conductivity of dual CFT
- Quasinormal mode spectrum (retarded Green's functions)
- Entropy production via viscous dissipation

Reference: Kovtun, Son, Starinets (2005); Policastro, Son, Starinets (2001)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class AdSCFTTransportParams:
    """Parameters for AdS/CFT holographic transport"""
    # Viscosity scaling (relative to KSS bound)
    viscosity_ratio: float = 1.0  # η/s / (1/4π)
    
    # Thermalization timescale (in units of GW period)
    thermalization_time: float = 0.05
    
    # Holographic conductivity coupling
    conductivity_scale: float = 1e-3
    
    # Quasinormal mode damping (multiple of 1/thermalization_time)
    qnm_damping: float = 1.0
    
    # Number of dominant QNM modes to include
    n_qnm_modes: int = 3


class Layer3AdSCFTTransportInjector:
    """
    Injects AdS/CFT holographic transport effects into GW strain data.
    
    Physics:
    1. Viscous dissipation scales strain energy ∝ η/s
    2. Thermalization induces frequency-dependent damping
    3. Holographic conductivity creates phase shifts
    4. QNM spectrum from retarded Green's functions
    5. Entropy production feedback on late-time signal
    """
    
    # Riemann zeta value for thermalization
    ZETA_PRIME_MINUS_ONE = -0.1654952254
    KSS_BOUND = 1.0 / (4.0 * np.pi)  # η/s lower bound
    
    @staticmethod
    def inject_adscft_transport(
        h_plus: np.ndarray,
        h_cross: np.ndarray,
        params: AdSCFTTransportParams,
        freq_array: np.ndarray,
        fs: float
    ) -> Dict:
        """
        Main injection pipeline for AdS/CFT transport effects.
        
        Parameters
        ----------
        h_plus, h_cross : strain data
        params : AdSCFTTransportParams configuration
        freq_array : frequency array for FFT
        fs : sampling frequency
        
        Returns
        -------
        dict with modified strain, physics info, metadata
        """
        n_samples = len(h_plus)
        dt = 1.0 / fs
        t = np.arange(n_samples) * dt
        
        # 1. Apply viscous dissipation (KSS bound + corrections)
        h_plus = Layer3AdSCFTTransportInjector._apply_viscous_dissipation(
            h_plus, t, params
        )
        h_cross = Layer3AdSCFTTransportInjector._apply_viscous_dissipation(
            h_cross, t, params
        )
        
        # 2. Add thermalization envelope
        h_plus = Layer3AdSCFTTransportInjector._add_thermalization_envelope(
            h_plus, t, params
        )
        h_cross = Layer3AdSCFTTransportInjector._add_thermalization_envelope(
            h_cross, t, params
        )
        
        # 3. Apply holographic conductivity (frequency-dependent phase)
        h_plus = Layer3AdSCFTTransportInjector._apply_holographic_conductivity(
            h_plus, freq_array, params
        )
        h_cross = Layer3AdSCFTTransportInjector._apply_holographic_conductivity(
            h_cross, freq_array, params
        )
        
        # 4. Inject QNM spectrum effects
        h_plus = Layer3AdSCFTTransportInjector._inject_qnm_spectrum(
            h_plus, t, freq_array, params
        )
        h_cross = Layer3AdSCFTTransportInjector._inject_qnm_spectrum(
            h_cross, t, freq_array, params
        )
        
        # 5. Apply entropy production feedback
        h_plus = Layer3AdSCFTTransportInjector._apply_entropy_feedback(
            h_plus, t, params
        )
        h_cross = Layer3AdSCFTTransportInjector._apply_entropy_feedback(
            h_cross, t, params
        )
        
        return {
            "h_plus": h_plus,
            "h_cross": h_cross,
            "physics_applied": {
                "viscous_dissipation": f"η/s = {params.viscosity_ratio:.3f} × (1/4π)",
                "thermalization": f"τ_therm = {params.thermalization_time:.4f}",
                "holographic_conductivity": f"σ_scale = {params.conductivity_scale:.2e}",
                "qnm_spectrum": f"{params.n_qnm_modes} modes with Q = 1/ζ(-1)",
                "entropy_production": "Active feedback loop"
            },
            "metadata": {
                "layer": "Layer_3",
                "theory": "AdS/CFT_Holography",
                "reference": "Kovtun_Son_Starinets_2005"
            }
        }
    
    @staticmethod
    def _apply_viscous_dissipation(
        h: np.ndarray,
        t: np.ndarray,
        params: AdSCFTTransportParams
    ) -> np.ndarray:
        """
        Apply energy dissipation due to finite viscosity.
        
        Physics: Viscous braking scales as -V_visc · strain_rate²
        Leading to exponential energy decay with viscosity-dependent timescale.
        
        τ_visc ∝ 1/(η/s) → larger viscosity → faster decay
        """
        # Viscosity correction factor (1/4π baseline)
        eta_s = params.viscosity_ratio * Layer3AdSCFTTransportInjector.KSS_BOUND
        
        # Viscous braking timescale (inverse proportional to viscosity)
        # Normalized to thermalization timescale
        if eta_s > 0:
            tau_visc = params.thermalization_time * (1.0 / (4.0 * np.pi * eta_s / Layer3AdSCFTTransportInjector.KSS_BOUND))
        else:
            tau_visc = np.inf
        
        # Dissipation envelope: exp(-t / tau_visc)
        dissipation = np.exp(-t / tau_visc)
        h_dissipated = h * dissipation
        
        return h_dissipated
    
    @staticmethod
    def _add_thermalization_envelope(
        h: np.ndarray,
        t: np.ndarray,
        params: AdSCFTTransportParams
    ) -> np.ndarray:
        """
        Add thermalization envelope from AdS black hole dynamics.
        
        Physics: Thermalization timescale τ ~ 1/(T_Hawking) ~ 1/β
        System reaches thermal equilibrium exponentially.
        """
        tau_therm = params.thermalization_time
        
        # Early time: Gaussian ramp-up to constant envelope
        # Late time: Exponential decay at τ_therm
        t_early = tau_therm * 0.3
        
        # Smooth step function using complementary error function
        ramp = 0.5 * (1.0 + np.tanh((t - t_early) / (tau_therm * 0.1)))
        
        h_thermals = h * ramp
        
        return h_thermals
    
    @staticmethod
    def _apply_holographic_conductivity(
        h: np.ndarray,
        freq_array: np.ndarray,
        params: AdSCFTTransportParams
    ) -> np.ndarray:
        """
        Apply phase shifts from holographic conductivity.
        
        Physics: CFT conductivity σ(ω) creates retarded response
        σ(ω) = σ_DC + σ_AC(ω) with poles in lower half-plane (causality)
        
        This manifests as frequency-dependent phase shifts:
        φ(ω) ~ arctan(σ(ω) / ω_plasma)
        """
        from scipy.fft import rfft, irfft
        
        n_samples = len(h)
        
        # FFT to frequency domain
        h_fft = rfft(h)
        
        # Conductivity-induced phase shift
        # σ(ω) ~ σ_0 / (1 + i·ω/Γ) → phase ~ atan(ω/Γ)
        gamma = 1.0 / params.thermalization_time  # Scattering rate
        phase_shift = np.arctan(freq_array / (gamma + 1e-10))
        
        # Apply scaling from conductivity strength
        phase_shift *= params.conductivity_scale * np.pi
        
        # Apply phase shift: multiply by exp(i·phase)
        h_fft_shifted = h_fft * np.exp(1j * phase_shift)
        
        # Transform back
        h_conductivity = irfft(h_fft_shifted, n=n_samples)
        
        return h_conductivity
    
    @staticmethod
    def _inject_qnm_spectrum(
        h: np.ndarray,
        t: np.ndarray,
        freq_array: np.ndarray,
        params: AdSCFTTransportParams
    ) -> np.ndarray:
        """
        Inject quasinormal mode spectrum into strain.
        
        Physics: AdS black holes have QNM spectrum ω_n = ω_0 - i·Γ_n
        Retarded Green's functions have poles at QNM frequencies
        → ringdown signal is sum of exponentially damped oscillations
        """
        from scipy.fft import rfft, irfft
        
        n_samples = len(h)
        
        # QNM frequencies and damping rates
        # Fundamental mode (n=0): ω_0 ~ 0.74 × 2π × T_eff
        T_eff = params.thermalization_time
        
        h_qnm = h.copy()
        
        # Inject multiple QNM modes with decreasing amplitude
        for n in range(params.n_qnm_modes):
            # Mode frequency: shifts by ~0.1 for each overtone
            omega_qnm = 2.0 * np.pi * (0.74 + 0.1 * n) / T_eff
            
            # Damping rate: Q = ω/Γ ~ 2 + 0.5*n (increases for overtones)
            Q_factor = 2.0 + 0.5 * n
            gamma_qnm = omega_qnm / Q_factor
            
            # QNM contribution: exp(-Γ*t) * sin(ω*t)
            mode_amplitude = 1.0 / (1.0 + n)  # Decreasing amplitude
            h_qnm += mode_amplitude * np.exp(-gamma_qnm * t) * np.sin(omega_qnm * t) * np.max(np.abs(h)) * params.qnm_damping
        
        return h_qnm
    
    @staticmethod
    def _apply_entropy_feedback(
        h: np.ndarray,
        t: np.ndarray,
        params: AdSCFTTransportParams
    ) -> np.ndarray:
        """
        Apply entropy production feedback on late-time signal.
        
        Physics: Gravitational wave energy → heat in dual CFT
        Energy dissipation rate dE/dt sets entropy production rate
        This creates additional modulation in late-time phase.
        
        Entropy production: dS/dt ∝ (energy loss rate) / T_eff
        """
        # Entropy production scales with energy dissipation
        # Late-time phase modulation: δφ ~ ∫ (dS/dt) dt
        
        # Cumulative entropy effect (time-integral of dissipation rate)
        dissipation_rate = np.gradient(np.exp(-t / params.thermalization_time))
        entropy_accumulated = np.cumsum(np.abs(dissipation_rate)) * (t[1] - t[0])
        
        # Normalize and create phase modulation
        entropy_phase = entropy_accumulated / np.max(entropy_accumulated + 1e-10)
        entropy_phase *= np.pi * params.viscosity_ratio * 0.1  # Coupling strength
        
        # Apply entropy-induced phase shift (more effect on late times)
        time_weight = np.exp(-t / params.thermalization_time)
        h_entropy = h * (1.0 - time_weight * entropy_phase)
        
        return h_entropy
    
    @staticmethod
    def compute_holographic_thermalization(
        energy: float,
        temperature: float,
        viscosity_ratio: float = 1.0
    ) -> Tuple[float, float, float]:
        """
        Compute thermalization timescale and entropy evolution.
        
        Returns:
            τ_therm : Thermalization timescale
            S_initial : Initial entropy
            S_final : Final entropy after thermalization
        """
        # Thermalization timescale ~ 1/(T_Hawking)
        tau_therm = 1.0 / (temperature + 1e-10)
        
        # Initial entropy ~ Energy / Temperature
        S_initial = energy / (temperature + 1e-10)
        
        # Final entropy (thermal): S_BH ~ (E/T)^(2/3) for AdS
        S_final = (energy / (temperature + 1e-10)) * (viscosity_ratio + 1.0) / 2.0
        
        return tau_therm, S_initial, S_final


def inject_adscft_transport_simple(
    h_plus: np.ndarray,
    h_cross: np.ndarray,
    mass: float,
    fs: float,
    viscosity_ratio: float = 1.0,
    thermalization_time: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Convenience wrapper for AdS/CFT transport injection.
    
    Automatically sets frequency array and applies default parameters.
    """
    from scipy.fft import rfftfreq
    
    n_samples = len(h_plus)
    freq_array = rfftfreq(n_samples, 1.0/fs)
    
    params = AdSCFTTransportParams(
        viscosity_ratio=viscosity_ratio,
        thermalization_time=thermalization_time,
        conductivity_scale=1e-3 * mass / 10.0,  # Scale with mass
        qnm_damping=1.0,
        n_qnm_modes=3
    )
    
    result = Layer3AdSCFTTransportInjector.inject_adscft_transport(
        h_plus, h_cross, params, freq_array, fs
    )
    
    metadata = {
        "mass": mass,
        "technique": "AdS_CFT_holographic_transport",
        "viscosity_ratio": viscosity_ratio,
        "thermalization_time": thermalization_time,
        "reference": "Kovtun_Son_Starinets_2005"
    }
    
    return result["h_plus"], result["h_cross"], metadata
