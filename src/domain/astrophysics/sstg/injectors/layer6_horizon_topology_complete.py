"""
Capa 6: Topología del Horizonte de Eventos
SSTG Physics Injection - Beyond-GR Horizon Effects

Modelos físicos:
1. Ecos de Fuzzball (Estructura de cuerda en horizonte)
2. Objetos Compactos Efectivos (ECO) - Echo fenomenología
3. Cuantización de LQG (Gravedad cuántica de bucles)
4. Memoria gravitacional (Efecto de cola de radiación gravitatoria)
5. Ringdown modificado en horizonte no-clásico

Referencias:
- Cardoso, Pani (2017): ECOs y testing de horizonte
- Addazi, et al (2021): LQG y ondas gravitatorias
- Blaut, et al (2015): Memoria gravitacional observacionales
- Almheiri et al (2013): AMPS firewall paradox
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from scipy import signal as scipy_signal
from scipy.fft import fft, ifft, fftfreq
import warnings

warnings.filterwarnings('ignore')

# Constantes físicas [SI]
C_LIGHT = 299792458.0  # m/s
G_CONST = 6.67430e-11  # m³/kg/s²
M_SUN = 1.98841e30  # kg
PC_TO_M = 3.0857e16  # 1 parsec en metros
MPC_TO_M = PC_TO_M * 1e6  # 1 Mpc en metros


@dataclass
class HorizonTopologyParams:
    """Parámetros para inyección de topología de horizonte"""
    theory: str  # "ECO", "LQG", "Firewall", "Fuzzball", "None"
    echo_delay: float = 0.0  # Retraso de eco en segundos
    echo_amplitude: float = 0.1  # Amplitud relativa del eco
    n_echoes: int = 3  # Número de ecos a generar
    lqg_area_quantum: float = 0.0  # Cuantización de área (LQG parameter)
    firewall_strength: float = 0.0  # Intensidad de cortina de fuego
    memory_amplitude: float = 0.0  # Amplitud de efecto de memoria
    ringdown_decay: float = 0.0  # Factor de amortiguamiento modificado
    mode: str = "injection"  # "injection" o "substitution"


class Layer6HorizonTopologyInjector:
    """
    Inyector de topología del horizonte para más allá de GR.
    
    Implementa:
    - Ecos con delays apropiados (2 travesías del agujero negro)
    - Cuantización de LQG con espectro discreto de frecuencias
    - Memoria gravitacional (integral de H_+ H_×)
    - Modificación de ringdown (Q modificado)
    """
    
    @staticmethod
    def inject_eco_echoes(
        h_plus: np.ndarray,
        h_cross: np.ndarray,
        echo_delay: float,
        echo_amplitude: float,
        n_echoes: int = 3,
        fs: float = 16384.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inyecta ecos de ECO (Effective Compact Object).
        
        Física: Ondas rebotando en interior del horizonte
        Fórmula de delay: τ_echo = 2M(π/2 - arcsin(b/2M))
        donde b es el parámetro de impacto del fotón
        
        Args:
            h_plus, h_cross: Strain de GR baseline
            echo_delay: Retraso del primer eco (segundos)
            echo_amplitude: Amplitud relativa (típicamente 0.1-0.3)
            n_echoes: Número de ecos (usualmente 2-4)
            fs: Sampling frequency
        
        Returns:
            h_plus_eco, h_cross_eco: Waveforms con ecos inyectados
        """
        h_plus_eco = h_plus.copy()
        h_cross_eco = h_cross.copy()
        
        delay_samples = int(echo_delay * fs)
        total_samples = len(h_plus)
        
        # Generar ecos sucesivos con amortiguamiento
        for i in range(1, n_echoes + 1):
            # Retraso: cada eco llega 2× después del anterior
            current_delay = delay_samples * i
            if current_delay >= total_samples:
                continue
            
            # Amplitud decay: A_n = A_0 * r^n con 0 < r < 1
            # En ECOs, r ~ 0.9-0.99 (muy poco amortiguamiento)
            decay_factor = echo_amplitude * (0.95 ** (i - 1))
            
            # Copiar y pegar la señal original con delay
            if current_delay + len(h_plus) <= total_samples:
                h_plus_eco[current_delay:current_delay + len(h_plus)] += decay_factor * h_plus
                h_cross_eco[current_delay:current_delay + len(h_cross)] += decay_factor * h_cross
            else:
                # Truncar si supera límite
                available = total_samples - current_delay
                h_plus_eco[current_delay:] += decay_factor * h_plus[:available]
                h_cross_eco[current_delay:] += decay_factor * h_cross[:available]
        
        return h_plus_eco, h_cross_eco
    
    @staticmethod
    def inject_lqg_discrete_spectrum(
        h_plus: np.ndarray,
        h_cross: np.ndarray,
        area_quantum: float,
        fs: float = 16384.0,
        mass: float = 10.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inyecta efectos de cuantización de LQG (Gravedad cuántica de bucles).
        
        Física: Horizonte tiene espectro de área discreto
        A_n = 4π ℓ_p² sqrt(n(n+1)) con n = 1,2,3,...
        donde ℓ_p es la longitud de Planck
        
        Esto modifica el espectro de cuasinormal modes (QNM).
        
        Args:
            h_plus, h_cross: Strain GR baseline
            area_quantum: Parámetro de cuantización LQG (0.1-1.0)
            fs: Sampling frequency
            mass: Masa del BH en masas solares
        
        Returns:
            h_plus_lqg, h_cross_lqg: Waveforms con espectro modificado
        """
        n_samples = len(h_plus)
        
        # Parámetro de Planck adimensional
        l_planck_adimenional = area_quantum
        
        # Frecuencias cuasinormal para Kerr (baseline, Schwarzschild simplificado)
        # f_QNM ~ M_BH^{-1} * (Q ± sqrt(Q² - 4))/(2π)
        # Para spin neutro: f_QNM ~ 0.36 M_BH^{-1}, Q ~ 9
        
        mass_seconds = mass * M_SUN * G_CONST / (C_LIGHT ** 3)  # M en segundos
        f_qnm_base = 0.36 / mass_seconds  # Frecuencia cuasinormal GR
        
        # En LQG, los QNM se modifican por el espectro discreto
        # Aproximación: Shift en Q por cuantización
        q_factor_gw = 9.0  # Q factor for fundamental (2,2) mode Schwarzschild
        q_factor_lqg = q_factor_gw * (1.0 - l_planck_adimenional)
        
        # FFT para transformar a frecuencias
        h_plus_fft = fft(h_plus)
        h_cross_fft = fft(h_cross)
        freqs = fftfreq(n_samples, 1.0 / fs)
        
        # Aplicar filtro que enfatiza modos discretos LQG
        # Crear peine de frecuencias (discrete levels)
        lqg_mode_spacing = 0.5 / mass_seconds  # Spacing entre modos
        
        # Construcción de filtro paso-banda centrado en modos LQG
        filter_response = np.ones(len(freqs))
        for n in range(1, 5):  # Primeros 5 modos LQG
            mode_freq = n * lqg_mode_spacing
            # Ancho de banda: Δf = f_QNM / (2π Q)
            bandwidth = mode_freq / (2 * np.pi * q_factor_lqg)
            # Gaussiano centrado en modo
            filter_response += 0.5 * np.exp(-(freqs - mode_freq)**2 / (bandwidth**2))
        
        # Normalizar
        filter_response = filter_response / np.max(filter_response)
        
        # Aplicar filtro
        h_plus_fft_filtered = h_plus_fft * filter_response
        h_cross_fft_filtered = h_cross_fft * filter_response
        
        # Transformada inversa
        h_plus_lqg = np.real(ifft(h_plus_fft_filtered))
        h_cross_lqg = np.real(ifft(h_cross_fft_filtered))
        
        return h_plus_lqg, h_cross_lqg
    
    @staticmethod
    def inject_gravitational_memory(
        h_plus: np.ndarray,
        h_cross: np.ndarray,
        fs: float = 16384.0,
        amplitude: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inyecta efecto de memoria gravitacional.
        
        Física: La radiación gravitatoria tiene una "cola" (tail) que 
        produce desplazamiento permanente del espacio-tiempo.
        
        El strain tiene componente:
        h_mem(t) = (2/π) ∫₀^∞ df/f |h̃(f)|² sin(2πft + φ(f))
        
        Esta es la memoria de onda gravitatoria (Christodoulou, Zel'dovich-Polnarev)
        
        Args:
            h_plus, h_cross: Strain GR baseline
            fs: Sampling frequency
            amplitude: Factor de amplitud de memoria (típicamente 0.01-0.1)
        
        Returns:
            h_plus_mem, h_cross_mem: Waveforms con memoria inyectada
        """
        n_samples = len(h_plus)
        time_array = np.arange(n_samples) / fs
        
        # Cálculo de memoria: integral de h_{ij} h_{ij}
        # Fórmula simplificada: memoria ~ amplitude * cumulative_integral(h² dt)
        
        # Normalizar strains
        h_plus_norm = h_plus / (np.max(np.abs(h_plus)) + 1e-10)
        h_cross_norm = h_cross / (np.max(np.abs(h_cross)) + 1e-10)
        
        # Energy: E ~ h₊² + h_×²
        energy_flux = h_plus_norm**2 + h_cross_norm**2
        
        # Memoria acumulada (integral): cambio permanente
        memory_signal = amplitude * np.cumsum(energy_flux) / fs
        
        # Normalizar memoria a rango razonable [-1, 1]
        memory_signal = memory_signal / (np.max(np.abs(memory_signal)) + 1e-10)
        memory_signal = memory_signal * (amplitude / 10.0)  # Re-escalar
        
        # Agregar memoria (típicamente a final de la señal)
        h_plus_mem = h_plus + memory_signal
        h_cross_mem = h_cross + memory_signal * 0.3  # Acoplamiento más débil en h_×
        
        return h_plus_mem, h_cross_mem
    
    @staticmethod
    def inject_fuzzball_echoes(
        h_plus: np.ndarray,
        h_cross: np.ndarray,
        fs: float = 16384.0,
        amplitude: float = 0.15,
        mass: float = 10.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inyecta estructura de fuzzball (geometría regular de string theory).
        
        Física: En string theory, el horizonte clásico se reemplaza por
        un estado fuzzy (fuzzball) sin singularidad.
        
        Los ecos de fuzzball tienen:
        - Retraso: τ ~ 2M (viaje de ida-vuelta por interior suave)
        - Amplitud: A ~ 0.1-0.3 (sin divergencias)
        - Número: Múltiples reflexiones en estructura fuzzy
        
        Args:
            h_plus, h_cross: Strain GR baseline
            fs: Sampling frequency
            amplitude: Amplitud de eco fuzzball
            mass: Masa BH en masas solares
        
        Returns:
            h_plus_fb, h_cross_fb: Waveforms con estructura fuzzy
        """
        mass_seconds = mass * M_SUN * G_CONST / (C_LIGHT ** 3)
        
        # Light-crossing time ~ 2M
        light_cross = 2 * mass_seconds
        delay_samples = int(light_cross * fs)
        
        h_plus_fb = h_plus.copy()
        h_cross_fb = h_cross.copy()
        
        # Generar serie de ecos fuzzy
        # Característica: cuerpo amortiguado rápidamente (estructura suave)
        n_echoes_fuzzy = 4
        echo_decay_rate = 0.8  # Decaimiento rápido
        
        for n in range(1, n_echoes_fuzzy + 1):
            delay = delay_samples * n
            if delay >= len(h_plus):
                break
            
            # Amplitud de fuzzball: decae más rápido que ECO
            amp_n = amplitude * (echo_decay_rate ** n)
            
            # Convolucionar con filtro PASO-BAJO para suavizar (física fuzzy)
            # Filtro Butterworth de orden bajo
            cutoff_freq = 200 + 50 * n  # Reduce frecuencias altas en ecos
            nyquist = fs / 2
            normalized_cutoff = cutoff_freq / nyquist
            if normalized_cutoff >= 1.0:
                normalized_cutoff = 0.99
            
            b, a = scipy_signal.butter(2, normalized_cutoff)
            h_plus_filtered = scipy_signal.filtfilt(b, a, h_plus)
            h_cross_filtered = scipy_signal.filtfilt(b, a, h_cross)
            
            # Inyectar echo
            if delay + len(h_plus) <= len(h_plus_fb):
                h_plus_fb[delay:delay + len(h_plus)] += amp_n * h_plus_filtered
                h_cross_fb[delay:delay + len(h_cross)] += amp_n * h_cross_filtered
            else:
                available = len(h_plus_fb) - delay
                h_plus_fb[delay:] += amp_n * h_plus_filtered[:available]
                h_cross_fb[delay:] += amp_n * h_cross_filtered[:available]
        
        return h_plus_fb, h_cross_fb
    
    @staticmethod
    def inject_modified_ringdown(
        h_plus: np.ndarray,
        h_cross: np.ndarray,
        fs: float = 16384.0,
        q_modification: float = 0.1,
        mass: float = 10.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inyecta modificación del ringdown por topología de horizonte alterada.
        
        Física: En horizonte cuántico/modificado:
        - Q_factor se modifica: Q_GR → Q_new = Q_GR * (1 - ε_LQG)
        - Tiempo de decay: τ_decay = Q/(π f_QNM)
        
        Observables: Ringdown decae más rápido si Q decrece
        
        Args:
            h_plus, h_cross: Strain GR baseline
            fs: Sampling frequency
            q_modification: Factor de modificación (0.1-0.5)
            mass: Masa BH en masas solares
        
        Returns:
            h_plus_rd, h_cross_rd: Waveforms con ringdown modificado
        """
        mass_seconds = mass * M_SUN * G_CONST / (C_LIGHT ** 3)
        
        # Identificar región de ringdown (última parte)
        # Usar umbral de amplitud para encontrar onset
        amplitude_envelope = np.abs(scipy_signal.hilbert(h_plus))
        max_amplitude = np.max(amplitude_envelope)
        threshold = max_amplitude * 0.1  # 10% del máximo
        
        # Encontrar índice donde comienza ringdown (después del merger)
        ringdown_start = np.where(amplitude_envelope < threshold)[0]
        if len(ringdown_start) > 0:
            ringdown_idx = ringdown_start[0]
        else:
            ringdown_idx = int(0.8 * len(h_plus))  # Default al 80%
        
        h_plus_rd = h_plus.copy()
        h_cross_rd = h_cross.copy()
        
        # Extraer región de ringdown
        ringdown_array = np.arange(ringdown_idx, len(h_plus))
        ringdown_duration = len(ringdown_array) / fs
        
        # Q factor en GR: ~9 para modo (2,2) de Schwarzschild
        q_factor_gr = 9.0
        q_factor_modified = q_factor_gr * (1.0 - q_modification)
        
        # Tasa de decay: α = π f_QNM / Q
        f_qnm = 0.36 / mass_seconds
        decay_rate_gr = np.pi * f_qnm / q_factor_gr
        decay_rate_modified = np.pi * f_qnm / q_factor_modified
        
        # Aplicar exponential decay modificado a la región de ringdown
        time_ringdown = np.arange(len(ringdown_array)) / fs
        decay_envelope_gr = np.exp(-decay_rate_gr * time_ringdown)
        decay_envelope_modified = np.exp(-decay_rate_modified * time_ringdown)
        
        # Factor de corrección local
        correction_factor = decay_envelope_modified / (decay_envelope_gr + 1e-10)
        
        # Aplicar corrección (amplificar amortiguamiento)
        h_plus_rd[ringdown_idx:] *= correction_factor
        h_cross_rd[ringdown_idx:] *= correction_factor
        
        return h_plus_rd, h_cross_rd
    
    @staticmethod
    def apply_horizon_topology(
        h_plus: np.ndarray,
        h_cross: np.ndarray,
        params: HorizonTopologyParams,
        mass: float = 10.0,
        fs: float = 16384.0
    ) -> Dict[str, any]:
        """
        Dispatcher principal para inyección de topología de horizonte.
        
        Selecciona el modelo físico según params.theory y aplica inyecciones.
        
        Args:
            h_plus, h_cross: Strain GR baseline
            params: HorizonTopologyParams con configuración
            mass: Masa BH en masas solares
            fs: Sampling frequency
        
        Returns:
            Dict con claves: {"h_plus", "h_cross", "h_scalar" (opcional),
                             "description", "physics_applied"}
        """
        
        result = {
            "h_plus": h_plus.copy(),
            "h_cross": h_cross.copy(),
            "physics_applied": [],
            "description": f"Horizon topology: {params.theory}"
        }
        
        if params.theory.lower() == "eco":
            # Echo Compact Object
            h_plus_out, h_cross_out = Layer6HorizonTopologyInjector.inject_eco_echoes(
                h_plus, h_cross,
                echo_delay=params.echo_delay or 0.0001,
                echo_amplitude=params.echo_amplitude or 0.15,
                n_echoes=params.n_echoes or 3,
                fs=fs
            )
            result["h_plus"] = h_plus_out
            result["h_cross"] = h_cross_out
            result["physics_applied"].append("ECO echoes")
            
        elif params.theory.lower() == "lqg":
            # Loop Quantum Gravity
            h_plus_out, h_cross_out = Layer6HorizonTopologyInjector.inject_lqg_discrete_spectrum(
                h_plus, h_cross,
                area_quantum=params.lqg_area_quantum or 0.5,
                fs=fs,
                mass=mass
            )
            result["h_plus"] = h_plus_out
            result["h_cross"] = h_cross_out
            result["physics_applied"].append("LQG discrete spectrum")
            
        elif params.theory.lower() == "fuzzball":
            # String theory fuzzy ball
            h_plus_out, h_cross_out = Layer6HorizonTopologyInjector.inject_fuzzball_echoes(
                h_plus, h_cross,
                fs=fs,
                amplitude=params.echo_amplitude or 0.15,
                mass=mass
            )
            result["h_plus"] = h_plus_out
            result["h_cross"] = h_cross_out
            result["physics_applied"].append("Fuzzball echoes")
            
        elif params.theory.lower() == "firewall":
            # Almheiri-Marolf-Polchinski-Sully firewall
            # Firewall = horizonte "quemado" que emite radiación
            h_plus_out = h_plus.copy()
            h_cross_out = h_cross.copy()
            
            # Agregar ruido de firewall (radiación extra)
            if params.firewall_strength > 0:
                firewall_noise = np.random.normal(0, params.firewall_strength, len(h_plus))
                h_plus_out = h_plus + 0.7 * firewall_noise
                h_cross_out = h_cross + 0.5 * firewall_noise
            
            result["h_plus"] = h_plus_out
            result["h_cross"] = h_cross_out
            result["physics_applied"].append("Firewall radiation")
            
        else:
            # "none" o default: aplicar todos los efectos suaves
            h_plus_out = h_plus.copy()
            h_cross_out = h_cross.copy()
            
            # Memoria gravitacional (efecto universal)
            if params.memory_amplitude > 0:
                h_plus_out, h_cross_out = Layer6HorizonTopologyInjector.inject_gravitational_memory(
                    h_plus_out, h_cross_out,
                    fs=fs,
                    amplitude=params.memory_amplitude
                )
                result["physics_applied"].append("Gravitational memory")
            
            # Ringdown modificado (suave)
            if params.ringdown_decay > 0:
                h_plus_out, h_cross_out = Layer6HorizonTopologyInjector.inject_modified_ringdown(
                    h_plus_out, h_cross_out,
                    fs=fs,
                    q_modification=params.ringdown_decay,
                    mass=mass
                )
                result["physics_applied"].append("Modified ringdown")
            
            result["h_plus"] = h_plus_out
            result["h_cross"] = h_cross_out
        
        result["description"] = f"Horizon topology: {params.theory} | Effects: {', '.join(result['physics_applied'])}"
        
        return result
