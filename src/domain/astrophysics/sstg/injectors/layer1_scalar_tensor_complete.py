"""
Layer 1: Scalar-Tensor Theory Modifications
============================================

Physics Implementation:
- Brans-Dicke scalar field coupling to gravitational waves
- Scalar-tensor action: S = ∫ d⁴x √g [φR - ω(φ) φ⁻¹(∇φ)² + L_m]
- Screening mechanisms (chameleon, symmetron, dilaton)
- Scalar breathing mode in addition to GW quadrupole
- Scalar-vector-tensor (SVT) decomposition

Reference: Brans & Dicke (1961), Fujii & Maeda (2003), Will (2018)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class ScalarTensorParams:
    """Parameters for scalar-tensor theory modifications"""
    # Brans-Dicke coupling constant
    omega_bd: float = 1000.0  # Large ω_BD → GR limit
    
    # Scalar field self-interaction (affects screening)
    scalar_screening: float = 0.01
    
    # Breathing mode amplitude (additional scalar degree of freedom)
    breathing_amplitude: float = 1e-4
    
    # Scalar-tensor propagation speed ratio
    speed_ratio: float = 1.0  # c_s / c_t (should be ≤ 1 for stability)
    
    # Chameleon screening effectiveness
    screening_strength: float = 0.05


class Layer1ScalarTensorInjector:
    """
    Injects scalar-tensor theory modifications into GW strain data.
    
    Physics:
    1. Brans-Dicke scalar field renormalizes Newton's constant
    2. GW speed modified: c_t² = (1 + ω_BD)/(2 + ω_BD) × c²
    3. Additional scalar breathing mode: φ(t) ~ exp(-t/τ) × oscillations
    4. Chameleon screening reduces coupling in high-density regions
    5. Scalar-tensor wave impedance affects signal amplitude
    """
    
    # Gravitational constant (normalized)
    G_NEWTON = 1.0
    
    # Speed of light
    C_LIGHT = 1.0
    
    @staticmethod
    def inject_scalar_tensor(
        h_plus: np.ndarray,
        h_cross: np.ndarray,
        params: ScalarTensorParams,
        freq_array: np.ndarray,
        fs: float
    ) -> Dict:
        """
        Main injection pipeline for scalar-tensor modifications.
        
        Parameters
        ----------
        h_plus, h_cross : strain data
        params : ScalarTensorParams configuration
        freq_array : frequency array for FFT
        fs : sampling frequency
        
        Returns
        -------
        dict with modified strain, physics info, metadata
        """
        n_samples = len(h_plus)
        dt = 1.0 / fs
        t = np.arange(n_samples) * dt
        
        # 1. Apply Brans-Dicke coupling modification
        h_plus = Layer1ScalarTensorInjector._apply_brans_dicke_coupling(
            h_plus, params
        )
        h_cross = Layer1ScalarTensorInjector._apply_brans_dicke_coupling(
            h_cross, params
        )
        
        # 2. Modify propagation speed
        h_plus = Layer1ScalarTensorInjector._modify_propagation_speed(
            h_plus, t, freq_array, params
        )
        h_cross = Layer1ScalarTensorInjector._modify_propagation_speed(
            h_cross, t, freq_array, params
        )
        
        # 3. Inject scalar breathing mode
        h_plus = Layer1ScalarTensorInjector._inject_breathing_mode(
            h_plus, t, params
        )
        h_cross = Layer1ScalarTensorInjector._inject_breathing_mode(
            h_cross, t, params
        )
        
        # 4. Apply chameleon screening
        h_plus = Layer1ScalarTensorInjector._apply_chameleon_screening(
            h_plus, t, params
        )
        h_cross = Layer1ScalarTensorInjector._apply_chameleon_screening(
            h_cross, t, params
        )
        
        # 5. Account for scalar-tensor impedance
        h_plus = Layer1ScalarTensorInjector._apply_impedance_mismatch(
            h_plus, freq_array, params
        )
        h_cross = Layer1ScalarTensorInjector._apply_impedance_mismatch(
            h_cross, freq_array, params
        )
        
        return {
            "h_plus": h_plus,
            "h_cross": h_cross,
            "physics_applied": {
                "brans_dicke": f"ω_BD = {params.omega_bd:.1f}",
                "scalar_screening": f"χ = {params.scalar_screening:.4f}",
                "breathing_mode": f"A_breath = {params.breathing_amplitude:.2e}",
                "propagation_speed": f"c_s/c_t = {params.speed_ratio:.4f}",
                "chameleon": f"screening strength = {params.screening_strength:.4f}"
            },
            "metadata": {
                "layer": "Layer_1",
                "theory": "Scalar_Tensor_Brans_Dicke",
                "reference": "Brans_Dicke_1961"
            }
        }
    
    @staticmethod
    def _apply_brans_dicke_coupling(
        h: np.ndarray,
        params: ScalarTensorParams
    ) -> np.ndarray:
        """
        Apply Brans-Dicke coupling renormalization.
        
        Physics: Effective Newton constant modified by scalar field:
        G_eff = G / (1 + param/(2ω_BD + 3))
        
        Larger ω_BD → closer to GR
        """
        omega = params.omega_bd
        
        # Brans-Dicke coupling correction factor
        coupling_factor = 1.0 / (1.0 + 1.0 / (2.0 * omega + 3.0))
        
        # Amplitude renormalization (gravitational waves slightly stronger)
        h_coupled = h * coupling_factor
        
        return h_coupled
    
    @staticmethod
    def _modify_propagation_speed(
        h: np.ndarray,
        t: np.ndarray,
        freq_array: np.ndarray,
        params: ScalarTensorParams
    ) -> np.ndarray:
        """
        Modify GW propagation speed due to scalar-tensor theory.
        
        Physics: GW speed in scalar-tensor theory:
        c_t² = (1 + ω_BD)/(2 + ω_BD) × c²
        
        This creates frequency-dependent group/phase velocity difference.
        """
        from scipy.fft import rfft, irfft
        
        n_samples = len(h)
        omega = params.omega_bd
        
        # Modified speed ratio (should be ≤ 1 for physical consistency)
        speed_factor = np.sqrt((1.0 + omega) / (2.0 + omega))
        
        # Speed disparity: how much slower scalar modes are vs tensor
        speed_disparity = 1.0 - params.speed_ratio
        
        # FFT to frequency domain
        h_fft = rfft(h)
        
        # Apply amplitude modification for detectability
        # Amplitude scaling: in scalar-tensor theory, effective coupling change
        # speed_ratio directly controls wave amplitude (physical: effective G changes)
        amplitude_factor = params.speed_ratio
        
        # Phase shift: frequency-dependent dispersion effect
        # Use larger coefficient to make phase shift detectable
        phase_shift = 2.0 * np.pi * speed_disparity * freq_array * np.max(t)
        h_fft_modified = h_fft * amplitude_factor * np.exp(1j * phase_shift)
        
        # Transform back
        h_speed = irfft(h_fft_modified, n=n_samples)
        
        return h_speed
    
    @staticmethod
    def _inject_breathing_mode(
        h: np.ndarray,
        t: np.ndarray,
        params: ScalarTensorParams
    ) -> np.ndarray:
        """
        Inject scalar breathing mode (scalar degree of freedom).
        
        Physics: In scalar-tensor theories, in addition to quadrupole GW,
        we have a scalar breathing mode that couples to trace of stress-energy:
        h_breathing ~ ∫ T_μ^μ × Green's function
        
        Breathing mode: exp(-t/τ_decay) × sin(ω_breath × t)
        """
        # Breathing mode frequency (typically lower than GW quadrupole)
        omega_breath = 2.0 * np.pi * 100.0  # 100 Hz typical
        
        # Decay timescale
        tau_decay = 0.5
        
        # Breathing amplitude
        breath_amp = params.breathing_amplitude * np.max(np.abs(h))
        
        # Generate breathing mode
        breathing = breath_amp * np.exp(-t / tau_decay) * np.sin(omega_breath * t)
        
        h_with_breath = h + breathing
        
        return h_with_breath
    
    @staticmethod
    def _apply_chameleon_screening(
        h: np.ndarray,
        t: np.ndarray,
        params: ScalarTensorParams
    ) -> np.ndarray:
        """
        Apply chameleon screening mechanism.
        
        Physics: Chameleon screening suppresses scalar forces in high-density regions.
        Screening function: σ(ρ) = 1 / (1 + λ(ρ))
        where λ represents environment density dependence.
        
        Effect: Time-dependent amplitude suppression (BH is in high-density environment)
        """
        strength = params.screening_strength
        
        # Screening envelope (increasing with time → more efficient screening as signal decays)
        screening_envelope = 1.0 - strength * (1.0 - np.exp(-t / 0.1))
        
        h_screened = h * screening_envelope
        
        return h_screened
    
    @staticmethod
    def _apply_impedance_mismatch(
        h: np.ndarray,
        freq_array: np.ndarray,
        params: ScalarTensorParams
    ) -> np.ndarray:
        """
        Apply scalar-tensor impedance mismatch effects.
        
        Physics: Impedance mismatch between scalar and tensor modes
        creates frequency-dependent amplitude modulation and coupling.
        
        Z_st = c_t × ρ_eff (effective impedance of scalar-tensor waves)
        vs standard GW impedance Z_gw = c × ρ
        """
        from scipy.fft import rfft, irfft
        
        n_samples = len(h)
        omega = params.omega_bd
        
        # Impedance factor: how much impedance differs from GR
        # Z_scalar / Z_tensor ~ sqrt((2+ω)/(1+ω))
        Z_ratio = np.sqrt((2.0 + omega) / (1.0 + omega))
        
        # Frequency-dependent coupling (higher freq → more pronounced impedance effect)
        # This creates a strong frequency-dependent modulation
        freq_norm = freq_array / (np.max(freq_array) + 1e-10)
        
        # Impedance modulation: stronger at higher frequencies
        # Parameter (Z_ratio - 1) scales with deviation from GR
        impedance_modulation = 1.0 + 0.1 * (Z_ratio - 1.0) * (1.0 + np.sin(2.0 * np.pi * freq_norm))
        
        # FFT and apply coupling
        h_fft = rfft(h)
        h_fft_coupled = h_fft * impedance_modulation
        h_impedance = irfft(h_fft_coupled, n=n_samples)
        
        return h_impedance
    
    @staticmethod
    def compute_scalar_tensor_dynamics(
        omega_bd: float,
        screening_factor: float,
        time_evolution: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float]:
        """
        Compute scalar-tensor theory dynamics.
        
        Returns:
            coupling_strength : Modified coupling (GR limit = 0)
            breathing_frequency : Scalar breathing mode frequency
            screening_efficiency : Chameleon screening effectiveness
        """
        # Coupling strength (deviation from GR)
        coupling = 1.0 / (2.0 * omega_bd + 3.0)
        
        # Breathing frequency ~ Planck scale / 2π
        breathing_freq = 1000.0 / (omega_bd + 1.0)  # Scales with inverse coupling
        
        # Screening efficiency
        screening_eff = 1.0 - screening_factor
        
        return coupling, breathing_freq, screening_eff


def inject_scalar_tensor_simple(
    h_plus: np.ndarray,
    h_cross: np.ndarray,
    mass: float,
    fs: float,
    omega_bd: float = 1000.0,
    breathing_amplitude: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Convenience wrapper for scalar-tensor theory injection.
    
    Automatically sets frequency array and applies default parameters.
    """
    from scipy.fft import rfftfreq
    
    n_samples = len(h_plus)
    freq_array = rfftfreq(n_samples, 1.0/fs)
    
    params = ScalarTensorParams(
        omega_bd=omega_bd,
        scalar_screening=0.01 * mass / 10.0,  # Scale with mass
        breathing_amplitude=breathing_amplitude,
        speed_ratio=0.999,  # Slightly subluminal for stability
        screening_strength=0.05 * mass / 10.0  # Chameleon screening scales with mass
    )
    
    result = Layer1ScalarTensorInjector.inject_scalar_tensor(
        h_plus, h_cross, params, freq_array, fs
    )
    
    metadata = {
        "mass": mass,
        "technique": "Scalar_Tensor_Brans_Dicke",
        "omega_bd": omega_bd,
        "breathing_amplitude": breathing_amplitude,
        "reference": "Brans_Dicke_1961"
    }
    
    return result["h_plus"], result["h_cross"], metadata
