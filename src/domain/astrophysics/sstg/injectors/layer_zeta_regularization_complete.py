"""
Layer 2 - Zeta Regularization (Formal Entropy Calculation)

Implementa regularización zeta para cálculo de entropía formal en sistemas cuánticos.
Basado en: Hawking (1974), Wald (1993), Bekenstein-Hawking entropy

Física:
-------
La regularización zeta es una técnica matemática para regularizar series divergentes:
ζ(s) = Σ n^(-s)  [Riemann zeta function]

Para cuantización de campos cerca del horizonte:
- Entropía: S ~ -ζ'(-1) + correcciones logarítmicas
- Energía: E_vac = Σ ℏω_n → regularizada con ζ(s)
- Determinantes: det Δ ~ exp(ζ'(0))

En gravedad cuántica:
- Entropía Bekenstein-Hawking: S_BH = A/(4l_P²)
- Correcciones zeta: δS = -ln(2π) + correcciones logarítmicas
- Apareció en Cardy anomaly, entanglement entropy

Implementación inyecta:
1. Correcciones de entropía (entropy source term)
2. Ruido cuántico de punto cero (zero-point quantum noise)
3. Modificación de tensor energía-momento
4. Efectos de regularización en amplitud de ondas
"""

import numpy as np
from dataclasses import dataclass
from scipy.special import zeta, spence, polygamma
from scipy.fft import rfft, irfft


@dataclass
class ZetaRegularizationParams:
    """Parámetros de regularización zeta"""
    entropy_scale: float = 0.01  # Escala típica de corrección de entropía (%)
    zeta_power: float = 2.0  # Potencia de divergencia (típicamente 2)
    zero_point_strength: float = 1e-3  # Amplitud de ruido de punto cero
    logarithmic_correction: float = 1e-4  # Corrección logarítmica (Cardy-like)
    cutoff_frequency: float = 1e3  # Frecuencia de corte UV (Hz)


class Layer2ZetaRegularizationInjector:
    """
    Inyector de regularización zeta para gravitational waveforms
    """
    
    @staticmethod
    def inject_zeta_regularization(
        h_plus: np.ndarray,
        h_cross: np.ndarray,
        params: ZetaRegularizationParams,
        freq_array: np.ndarray,
        fs: float,
    ) -> dict:
        """
        Inyecta efectos de regularización zeta en waveform
        
        Parameters
        ----------
        h_plus, h_cross : ndarray
            Polarizaciones GW originales
        params : ZetaRegularizationParams
            Parámetros de regularización
        freq_array : ndarray
            Array de frecuencias en Fourier
        fs : float
            Sampling frequency (Hz)
            
        Returns
        -------
        dict : {h_plus, h_cross, physics_applied, metadata}
        """
        n_samples = len(h_plus)
        dt = 1.0 / fs
        
        # 1. Calcular corrección de entropía (modifica amplitud)
        h_plus_entropy = Layer2ZetaRegularizationInjector._apply_entropy_correction(
            h_plus, params, fs
        )
        h_cross_entropy = Layer2ZetaRegularizationInjector._apply_entropy_correction(
            h_cross, params, fs
        )
        
        # 2. Inyectar ruido cuántico de punto cero
        h_plus_zp = Layer2ZetaRegularizationInjector._add_zero_point_noise(
            h_plus_entropy, n_samples, params, fs
        )
        h_cross_zp = Layer2ZetaRegularizationInjector._add_zero_point_noise(
            h_cross_entropy, n_samples, params, fs
        )
        
        # 3. Aplicar corrección logarítmica (Cardy anomaly-like)
        h_plus_log = Layer2ZetaRegularizationInjector._apply_logarithmic_correction(
            h_plus_zp, params, fs
        )
        h_cross_log = Layer2ZetaRegularizationInjector._apply_logarithmic_correction(
            h_cross_zp, params, fs
        )
        
        # 4. Computar factor de regularización (zeta cutoff effect)
        zeta_factor = Layer2ZetaRegularizationInjector._compute_zeta_factor(
            freq_array, params
        )
        
        # 5. Aplicar suavizado espectral
        h_plus_smooth = Layer2ZetaRegularizationInjector._apply_spectral_smoothing(
            h_plus_log, zeta_factor
        )
        h_cross_smooth = Layer2ZetaRegularizationInjector._apply_spectral_smoothing(
            h_cross_log, zeta_factor
        )
        
        return {
            "h_plus": h_plus_smooth,
            "h_cross": h_cross_smooth,
            "physics_applied": [
                "entropy_correction",
                "zero_point_noise",
                "logarithmic_correction",
                "zeta_spectral_smoothing"
            ],
            "metadata": {
                "entropy_scale": params.entropy_scale,
                "zeta_power": params.zeta_power,
                "zero_point_strength": params.zero_point_strength,
                "technique": "Riemann_zeta_regularization"
            }
        }
    
    @staticmethod
    def _apply_entropy_correction(h: np.ndarray, params: ZetaRegularizationParams, fs: float) -> np.ndarray:
        """
        Aplica corrección de entropía formal
        
        S_reg = -ζ'(-1) + ln(2π) + ...
        
        Physics: Entropía renormalizada en teoría cuántica de campos
        """
        n_samples = len(h)
        t = np.arange(n_samples) / fs
        
        # Calcular ζ'(-1) numérica (derivada de zeta en -1)
        # Valor conocido: ζ'(-1) ≈ -0.1654952...
        zeta_prime_minus_one = -0.1654952254  
        entropy_shift = -zeta_prime_minus_one + np.log(2*np.pi)
        
        # Aplicar como modulación lenta (envelope correction)
        # Corrección adicional: ln(S_0) donde S_0 ~ área/l_P²
        entropy_envelope = 1.0 + params.entropy_scale * entropy_shift / 10.0
        
        # Modular señal con envelope suave
        modulation = entropy_envelope + 0.01 * np.sin(2*np.pi*0.5*t)  # Término oscilante pequeño
        
        return h * modulation
    
    @staticmethod
    def _add_zero_point_noise(h: np.ndarray, n_samples: int, params: ZetaRegularizationParams, fs: float) -> np.ndarray:
        """
        Inyecta ruido cuántico de punto cero regularizado
        
        E_vac = Σ ℏω_n / 2  → Se suma sobre modos regularizados
        
        Physics: Energía de punto cero diverge sin regularización
        """
        # Ruido Gaussiano coloreado (∝ 1/f**0.5 - espectro de punto cero)
        white_noise = np.random.normal(0, 1, n_samples)
        
        # FFT para colorear
        fft_noise = rfft(white_noise)
        freq_array = np.fft.rfftfreq(n_samples, 1.0/fs)
        
        # Espectro de punto cero: S(f) ∝ f^(-(zeta_power-1)/2)
        # Típicamente zeta_power=2 → S(f) ∝ f^(-0.5)
        power = (params.zeta_power - 1.0) / 2.0
        spectrum = 1.0 / np.maximum(freq_array, 1.0)**power
        spectrum[0] = spectrum[1]  # Evitar singularidad en f=0
        
        # Aplicar espectro
        fft_colored = fft_noise * np.sqrt(spectrum) / np.sqrt(np.mean(spectrum))
        colored_noise = irfft(fft_colored, n_samples)
        
        # Normalizar amplitud
        amplitude = params.zero_point_strength * np.std(h)
        amplitude = np.maximum(amplitude, 1e-30)  # Evitar NaN
        
        return h + (amplitude * colored_noise / np.std(colored_noise + 1e-30))
    
    @staticmethod
    def _apply_logarithmic_correction(h: np.ndarray, params: ZetaRegularizationParams, fs: float) -> np.ndarray:
        """
        Aplica corrección logarítmica (inspirada en Cardy anomaly)
        
        δS ~ (c/6) * ln(N) donde c es central charge, N modos
        
        Physics: Entanglement entropy tiene corrección logarítmica
        """
        n_samples = len(h)
        t = np.arange(n_samples) / fs
        
        # Corrección que crece lentamente con el número de modos
        # Usar logaritmo temporal como proxy para número de modos
        time_log = np.log(np.maximum(t + 1e-6, 1e-6))  # Evitar log(0)
        
        # Normalizar
        time_log_norm = time_log / np.max(np.abs(time_log) + 1e-30)
        
        # Modular señal con corrección logarítmica suave
        log_correction = 1.0 + params.logarithmic_correction * time_log_norm
        
        return h * log_correction
    
    @staticmethod
    def _compute_zeta_factor(freq_array: np.ndarray, params: ZetaRegularizationParams) -> np.ndarray:
        """
        Calcula factor de regularización zeta como función de frecuencia
        
        ζ_factor(f) = [1 + (f/f_cutoff)^(-power)] ^ (-0.5)
        
        Physics: Cutoff UV regulariza divergencias
        """
        f_cutoff = params.cutoff_frequency
        f_safe = np.maximum(freq_array, 1.0)  # Evitar singularidades
        
        # Función suave de transición
        ratio = f_safe / f_cutoff
        
        # Factor zeta: regularización típica
        zeta_factor = 1.0 / np.sqrt(1.0 + ratio**params.zeta_power + 1e-30)
        
        return zeta_factor
    
    @staticmethod
    def _apply_spectral_smoothing(h: np.ndarray, zeta_factor: np.ndarray) -> np.ndarray:
        """
        Aplica suavizado espectral con factor zeta
        """
        # FFT
        fft_h = rfft(h)
        
        # Aplicar factor zeta en Fourier
        fft_smoothed = fft_h * zeta_factor
        
        # IFFT
        h_smoothed = irfft(fft_smoothed, len(h))
        
        return h_smoothed
    
    @staticmethod
    def compute_renormalized_entropy(
        area: float,
        mass: float = 10.0,
        include_corrections: bool = True
    ) -> tuple:
        """
        Calcula entropía renormalizada usando regularización zeta
        
        S_BH = A / (4 l_P^2)  [Bekenstein-Hawking]
        S_reg = S_BH + δS_zeta + δS_log
        
        Parameters
        ----------
        area : float
            Área del horizonte (en l_P^2)
        mass : float
            Masa del agujero negro (en m_P)
        include_corrections : bool
            Incluir correcciones de regularización
            
        Returns
        -------
        S_BH, S_zeta, S_total
        """
        # Entropía Bekenstein-Hawking
        S_BH = area / 4.0
        
        if not include_corrections:
            return S_BH, 0.0, S_BH
        
        # Corrección zeta (∝ ζ'(-1))
        zeta_correction = -0.1654952 * np.log(mass)  # Log correction from zeta function
        
        # Corrección logarítmica (Cardy anomaly-like)
        log_correction = 0.393 * np.log(area)  # Central charge effect
        
        S_zeta = zeta_correction + log_correction
        S_total = S_BH + S_zeta
        
        return S_BH, S_zeta, S_total


def inject_zeta_regularization_simple(
    h_plus: np.ndarray,
    h_cross: np.ndarray,
    mass: float = 10.0,
    fs: float = 4096.0,
    entropy_scale: float = 0.01,
) -> tuple:
    """
    Wrapper simple para inyección de regularización zeta
    
    Parameters
    ----------
    h_plus, h_cross : ndarray
        Polarizaciones
    mass : float
        Masa en m_P
    fs : float
        Frecuencia de muestreo
    entropy_scale : float
        Escala de corrección de entropía
        
    Returns
    -------
    h_plus_zeta, h_cross_zeta, metadata : tuple
    """
    n_samples = len(h_plus)
    freq_array = np.fft.rfftfreq(n_samples, 1.0/fs)
    
    # Parámetros predeterminados escalados por masa
    params = ZetaRegularizationParams(
        entropy_scale=entropy_scale,
        zeta_power=2.0,
        zero_point_strength=1e-3 / (1.0 + mass/10.0),  # Escala con masa
        logarithmic_correction=1e-4 * np.log(mass),
        cutoff_frequency=1e3 * (mass / 10.0)**0.5  # Escala UV
    )
    
    result = Layer2ZetaRegularizationInjector.inject_zeta_regularization(
        h_plus, h_cross, params, freq_array * (1 + mass/100.0), fs
    )
    
    metadata = {
        "mass": mass,
        "entropy_scale": entropy_scale,
        "technique": "Riemann_zeta_regularization",
        "physics_type": "formal_entropy",
        "corrections_applied": result["physics_applied"]
    }
    
    return result["h_plus"], result["h_cross"], metadata
