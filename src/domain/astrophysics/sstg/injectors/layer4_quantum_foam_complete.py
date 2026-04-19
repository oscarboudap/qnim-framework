"""
Layer 4 - Quantum Foam (Wheeler Spacetime Foam)

Implementa fluctuaciones topológicas del vacío cuántico que generan:
- Modified Dispersion Relations (MDR)
- Stochastic phase diffusion
- Wheeler vacuum fluctuations
- Coherence loss (apparent unitarity violation)
- Ruido interferométrico

Referencias:
- t Hooft: "Quantum Spacetime"
- Garay et al: "Spacetime as a quantum many-body system"
- Hawking: "Quantum gravity and black holes"
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List
from scipy import optimize, interpolate
from scipy.fft import rfft, irfft, rfftfreq
from scipy.signal.windows import hann


@dataclass
class QuantumFoamParams:
    """Parámetros de Wheeler quantum foam"""
    
    # MDR (Modified Dispersion Relation)
    mdr_exponent: float = 2.0  # n en p² = (E/c)²[1 + (E/E_P)^n]
    mdr_strength: float = 0.01  # Amplitud de corrección MDR
    
    # Stochastic phase diffusion
    diffusion_coefficient: float = 1e-3  # D_0 en fase
    coherence_timescale: float = 0.1  # τ_deco en segundos
    
    # Wheeler vacuum fluctuations
    vacuum_energy_density: float = 1e-9  # ρ_vac en GeV/cm³ (very small)
    fluctuation_amplitude: float = 0.01  # A en perturbative regime
    
    # Dissipative coupling
    dissipation_rate: float = 0.05  # τ_dissip⁻¹
    quantum_number: int = 2  # l quantum number


class Layer4QuantumFoamInjector:
    """
    Quantum Foam injection según Wheeler spacetime discreteness.
    
    Implementa:
    - Modified Dispersion Relations (MDR) p² = F(E, E_P)
    - Stochastic phase modulation
    - Vacuum fluctuation noise
    - Decoherence effects
    """
    
    # Constantes físicas
    PLANCK_ENERGY = 1.956e9  # GeV (E_P = sqrt(ℏc/G) / natural units)
    PLANCK_TIME = 5.39e-44   # segundos (t_P)
    PLANCK_LENGTH = 1.616e-35  # metros (ℓ_P)
    HBAR = 1.054571817e-34   # J·s
    C_LIGHT = 299792458      # m/s
    
    @staticmethod
    def inject_quantum_foam(
        h_plus: np.ndarray,
        h_cross: np.ndarray,
        params: QuantumFoamParams,
        freq_array: np.ndarray,
        fs: int = 4096
    ) -> Dict[str, np.ndarray]:
        """
        Inyecta efectos de quantum foam en la señal de onda gravitacional.
        
        Parámetros
        ----------
        h_plus, h_cross : ndarray
            Componentes polarización original
        params : QuantumFoamParams
            Parámetros del foam
        freq_array : ndarray
            Frecuencias positivas (Hz)
        fs : int
            Sampling frequency
            
        Retorna
        -------
        Dict con h_plus_foam, h_cross_foam, physics_applied, metadata
        """
        
        n_samples = len(h_plus)
        
        # 1. Apply Modified Dispersion Relation (MDR)
        # p²(f) = (2πf)²/c² * [1 + (E/E_P)^n] donde E = hf
        h_plus_mdr = Layer4QuantumFoamInjector._apply_mdr(
            h_plus, freq_array, fs, params
        )
        h_cross_mdr = Layer4QuantumFoamInjector._apply_mdr(
            h_cross, freq_array, fs, params
        )
        
        # 2. Apply stochastic phase diffusion
        h_plus_foam = Layer4QuantumFoamInjector._apply_phase_diffusion(
            h_plus_mdr, n_samples, params, fs
        )
        h_cross_foam = Layer4QuantumFoamInjector._apply_phase_diffusion(
            h_cross_mdr, n_samples, params, fs
        )
        
        # 3. Add Wheeler vacuum fluctuations (noise)
        h_plus_foam = Layer4QuantumFoamInjector._add_vacuum_noise(
            h_plus_foam, freq_array, params
        )
        h_cross_foam = Layer4QuantumFoamInjector._add_vacuum_noise(
            h_cross_foam, freq_array, params
        )
        
        # 4. Apply coherence loss (decoherence)
        deco_envelope = Layer4QuantumFoamInjector._compute_decoherence_envelope(
            n_samples, params, 1.0/fs
        )
        h_plus_foam *= deco_envelope
        h_cross_foam *= deco_envelope
        
        # Metadata
        physics_applied = [
            "mdr_dispersion",
            "phase_diffusion",
            "vacuum_fluctuations",
            "decoherence"
        ]
        
        metadata = {
            "foam_type": "wheeler_spacetime",
            "mdr_exponent": params.mdr_exponent,
            "mdr_strength": params.mdr_strength,
            "coherence_timescale": params.coherence_timescale,
            "diffusion_coefficient": params.diffusion_coefficient,
            "planck_order_corrections": True,
            "linearized_approximation": params.mdr_strength < 0.1
        }
        
        return {
            "h_plus": h_plus_foam,
            "h_cross": h_cross_foam,
            "physics_applied": physics_applied,
            "metadata": metadata
        }
    
    @staticmethod
    def _apply_mdr(
        signal_component: np.ndarray,
        freq_array: np.ndarray,
        fs: int,
        params: QuantumFoamParams
    ) -> np.ndarray:
        """
        Apply Modified Dispersion Relation:
        E(k) → E(k)[1 + (E/E_P)^n]
        
        En el dominio de frecuencias:
        ω(f) = 2πf/c * [1 + (hf/E_P)^n]
        
        Efecto: fase modificada, dispersión subluminal
        """
        
        n_samples = len(signal_component)
        dt = 1.0 / fs
        
        # Transformada de Fourier
        signal_fft = rfft(signal_component)
        freqs_rfft = rfftfreq(n_samples, dt)
        freqs_rfft = np.abs(freqs_rfft)  # Solo positivas
        
        # Calcular energía por modo
        # E = h * f (donde h es Planck)
        h_planck = 4.1357e-15  # eV·s
        energy_ev = h_planck * freqs_rfft
        
        # Calcular factor MDR
        # modification = (E/E_P)^n
        energy_ratio = energy_ev / (Layer4QuantumFoamInjector.PLANCK_ENERGY * 1e9)
        
        # Evitar divergencias en modos muy bajos
        energy_ratio = np.maximum(energy_ratio, 1e-12)
        
        mdr_factor = 1.0 + params.mdr_strength * (energy_ratio ** params.mdr_exponent)
        
        # Aplicar corrección en Fourier (como corrección de fase)
        phase_correction = 2 * np.pi * (mdr_factor - 1.0)
        signal_fft *= np.exp(1j * phase_correction)
        
        # Antitransformada
        signal_mdr = irfft(signal_fft, n=n_samples)
        
        return signal_mdr
    
    @staticmethod
    def _apply_phase_diffusion(
        signal_component: np.ndarray,
        n_samples: int,
        params: QuantumFoamParams,
        fs: int
    ) -> np.ndarray:
        """
        Stochastic phase diffusion:
        ϕ(t) = ϕ_0 + ∫ √D dW(t)
        
        Donde D = D_0 (E/E_P) y dW es noise Gaussiano
        """
        
        dt = 1.0 / fs
        
        # Generar Wiener process (random walk)
        dW = np.random.normal(0, np.sqrt(dt), n_samples)
        
        # Integrated diffusion
        phase_random = np.cumsum(dW)
        
        # Scaling con energía típica (usar RMS de strain como proxy)
        energy_proxy = np.sqrt(np.mean(signal_component**2))
        if energy_proxy > 0:
            diffusion_scaled = params.diffusion_coefficient * energy_proxy
        else:
            diffusion_scaled = params.diffusion_coefficient
        
        # Apply phase modulation
        phase_modulation = diffusion_scaled * phase_random
        
        signal_phase_diff = signal_component * np.exp(1j * phase_modulation)
        
        # Tomar parte real
        signal_phase_diff = np.real(signal_phase_diff)
        
        return signal_phase_diff
    
    @staticmethod
    def _add_vacuum_noise(
        signal_component: np.ndarray,
        freq_array: np.ndarray,
        params: QuantumFoamParams
    ) -> np.ndarray:
        """
        Add Wheeler vacuum fluctuation noise.
        
        Spectrum ~ f^(-4/3) para frecuencias altas (AdS/CFT inspired)
        """
        
        n_samples = len(signal_component)
        
        # Generate colored noise with 1/f^(4/3) power spectrum
        # Procedimiento: generar white noise, transformar a frecuencia,
        # aplicar filtro 1/f^(4/3), transformar back
        
        # White noise
        white_noise = np.random.normal(0, 1, n_samples)
        
        # FFT
        noise_fft = rfft(white_noise)
        freqs = rfftfreq(n_samples, 1.0/4096)  # Hardcoded fs para spectrum
        freqs = np.maximum(freqs, 1e-10)  # Avoid division by zero
        
        # Apply 1/f^(4/3) filter
        spectral_filter = 1.0 / (freqs ** (4.0/3.0))
        spectral_filter[0] = 0  # Remove DC component
        
        # Normalize
        spectral_filter /= np.sqrt(np.mean(spectral_filter**2))
        
        noise_fft *= spectral_filter
        
        # IFFT
        foam_noise = irfft(noise_fft, n=n_samples)
        
        # Scale by vacuum energy density
        foam_noise *= params.fluctuation_amplitude
        
        return signal_component + foam_noise
    
    @staticmethod
    def _compute_decoherence_envelope(
        n_samples: int,
        params: QuantumFoamParams,
        dt: float
    ) -> np.ndarray:
        """
        Compute decoherence envelope:
        V(t) = exp(-t/τ_deco)
        
        Efectivamente decae la amplitud de coherencia
        """
        
        time_array = np.arange(n_samples) * dt
        tau_deco = params.coherence_timescale
        
        # Exponential decay envelope
        envelope = np.exp(-time_array / tau_deco)
        
        # Smooth to avoid sharp edges
        # Use Hann window to taper start/end
        hann_window = hann(n_samples)
        envelope *= hann_window
        
        return envelope
    
    @staticmethod
    def compute_mdr_dispersion_relation(
        frequency: float,
        mdr_exponent: float = 2.0,
        mdr_strength: float = 0.01
    ) -> Tuple[float, float]:
        """
        Compute phase velocity and group velocity including MDR.
        
        Retorna (v_phase, v_group) - ambas < c para foam
        """
        
        h_planck = 4.1357e-15
        E = h_planck * frequency
        energy_ratio = E / (Layer4QuantumFoamInjector.PLANCK_ENERGY * 1e9)
        
        # MDR factor (must be > 1 to slow down light)
        mdr_factor = 1.0 + mdr_strength * (energy_ratio ** mdr_exponent)
        
        # c_light with correction (slower due to foam)
        c_light = 299792458  # m/s
        v_phase = c_light / mdr_factor  # Subluminal!
        
        # Group velocity (derivative)
        if mdr_strength > 0:
            dE_df = h_planck
            numerator = mdr_strength * mdr_exponent * (energy_ratio ** (mdr_exponent - 1))
            denominator = Layer4QuantumFoamInjector.PLANCK_ENERGY * 1e9
            d_mdr = numerator / denominator
            
            # v_g = d(ω)/d(k) where ω = v_phase * k
            v_group = v_phase - frequency * (c_light / (mdr_factor**2)) * d_mdr * dE_df
        else:
            v_group = v_phase
        
        return v_phase, v_group


# Funciones de conveniencia para integración

def inject_quantum_foam_simple(
    h_plus: np.ndarray,
    h_cross: np.ndarray,
    mass: float,
    fs: int = 4096,
    mdr_strength: float = 0.01
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Wrapper simple para inyección de quantum foam.
    
    Usa parámetros por defecto optimizados para detección.
    """
    
    params = QuantumFoamParams(
        mdr_strength=mdr_strength,
        mdr_exponent=2.0,
        diffusion_coefficient=1e-3,
        coherence_timescale=max(0.1, 0.5 / mass)  # Scaling con masa
    )
    
    # Compute FFT to get frequency array
    n_samples = len(h_plus)
    freq_array = rfftfreq(n_samples, 1.0/fs)
    freq_array = np.abs(freq_array)
    
    result = Layer4QuantumFoamInjector.inject_quantum_foam(
        h_plus, h_cross, params, freq_array, fs
    )
    
    return result["h_plus"], result["h_cross"], result["metadata"]
