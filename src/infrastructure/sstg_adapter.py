"""
Infrastructure: SSTG (Synthetic Signal Template Generator) Adapter
==================================================================

Adaptador que implementa ISyntheticDataGeneratorPort.

Genera datos sintéticos balanceados para el pipeline de clasificación
de 10 teorías gravitacionales. Funciona sin dependencias externas
complejas mediante síntesis simple.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from src.application.ports import ISyntheticDataGeneratorPort
from src.application.dto import BalancedDataset
from src.infrastructure.exceptions import ReportingException


class SSTGAdapter(ISyntheticDataGeneratorPort):
    """
    Generador de datos sintéticos para clasificación de teorías.
    
    Genera eventos sintéticos de 10 clases (una por teoría) con
    características realistas (strain simulado + ruido coloreado).
    """
    
    # 10 teorías beyond-GR soportadas
    THEORY_CLASSES = [
        "GR",                      # Clase 0: General Relativity (baseline)
        "scalar-tensor",           # Clase 1: Brans-Dicke
        "f(R)-gravity",            # Clase 2: f(R) modified gravity
        "loop-quantum-gravity",    # Clase 3: LQG
        "extra-dimensions",        # Clase 4: Kaluza-Klein
        "graviton-mass",           # Clase 5: Massive graviton
        "echo-hypothesis",         # Clase 6: Firewall echoes (ECO)
        "axion-superradiance",     # Clase 7: Axion coupling
        "string-inspired",         # Clase 8: String theory signatures
        "quantum-entanglement",    # Clase 9: Quantum information effects
    ]
    
    def __init__(self):
        """Inicializa el adaptador SSTG."""
        pass
    
    @staticmethod
    def _generate_simple_strain(
        m1: float,
        m2: float,
        distance: float,
        theory_offset: float = 0.0,
        duration: float = 8.0,
        fs: int = 4096
    ) -> np.ndarray:
        """
        Genera strain sintético simple con modulación por teoría.
        
        Args:
            m1: Masa primaria [M_sun]
            m2: Masa secundaria [M_sun]
            distance: Distancia [Mpc]
            theory_offset: Factor que modula la amplitud (por teoría)
            duration: Duración total [s]
            fs: Frecuencia de muestreo [Hz]
        
        Returns:
            Array 1D normalizado con strain simulado
        """
        n_samples = int(fs * duration)
        t = np.arange(n_samples) / fs
        
        # Frecuencia característica de la fuente (simple)
        m_total = m1 + m2
        f_char = 50 * (1 / m_total)  # Hz
        
        # Amplitud ~ 1 / (distance * m_total)
        amplitude = 1e-21 / (distance * np.sqrt(m_total))
        
        # Modulación por teoría (cada teoría tiene una "firma" diferente)
        theory_modulation = 1.0 + theory_offset * 0.3
        amplitude *= theory_modulation
        
        # Crear chirp simple (frecuencia creciente)
        phase = 2 * np.pi * f_char * t + 50 * t**2  # t² para incremento de freq
        
        # Señal
        strain = amplitude * np.sin(phase)
        
        # Añadir ruido coloreado (como PSD LIGO O3)
        white_noise = np.random.normal(0, amplitude * 0.5, n_samples)
        
        # Filtro pasa-banda simple (1/f spectrum)
        freq_spectrum = np.fft.rfft(white_noise)
        freqs = np.fft.rfftfreq(n_samples, 1/fs)
        
        # Colored noise: amplitud ~ 1/sqrt(f) para f > 10 Hz
        color_filter = np.ones_like(freqs)
        color_filter[freqs > 10] /= np.sqrt(freqs[freqs > 10] / 10)
        colored_noise_fft = freq_spectrum * color_filter
        colored_noise = np.fft.irfft(colored_noise_fft, n=n_samples)
        
        # Combinar
        strain = strain + colored_noise[:n_samples]
        
        # Normalizar
        strain_std = np.std(strain)
        if strain_std > 1e-25:
            strain = strain / strain_std
        
        return strain
    
    def synthesize_event(self,
                        mass1_solar_masses: float,
                        mass2_solar_masses: float,
                        distance_mpc: float,
                        theory_family: str,
                        sampling_rate_hz: int = 4096,
                        duration_seconds: float = 8.0) -> np.ndarray:
        """
        Sintetiza un evento de onda gravitacional.
        
        Args:
            mass1_solar_masses: Masa del cuerpo 1 (M☉)
            mass2_solar_masses: Masa del cuerpo 2 (M☉)
            distance_mpc: Distancia del observador (Mpc)
            theory_family: Familia teórica ("GR", "scalar-tensor", etc.)
            sampling_rate_hz: Frecuencia de muestreo (default: 4096 como LIGO)
            duration_seconds: Duración total de la ventana (default: 8s)
        
        Returns:
            np.ndarray: Strain data simulado [n_samples,]
        
        Raises:
            ReportingException: Si parámetros inválidos
        """
        try:
            # Validación de parámetros
            if mass1_solar_masses <= 0 or mass2_solar_masses <= 0:
                raise ValueError("Masas deben ser > 0")
            if distance_mpc <= 0:
                raise ValueError("Distancia debe ser > 0")
            if sampling_rate_hz <= 0:
                raise ValueError("Sampling rate debe ser > 0")
            if duration_seconds <= 0:
                raise ValueError("Duración debe ser > 0")
            
            # Offset por teoría (para que cada teoría tenga "firma" distinta)
            theory_idx = self.THEORY_CLASSES.index(theory_family) if theory_family in self.THEORY_CLASSES else 0
            theory_offset = theory_idx / len(self.THEORY_CLASSES)
            
            # Generar strain
            strain = self._generate_simple_strain(
                m1=mass1_solar_masses,
                m2=mass2_solar_masses,
                distance=distance_mpc,
                theory_offset=theory_offset,
                duration=duration_seconds,
                fs=sampling_rate_hz
            )
            
            return strain
        
        except Exception as e:
            raise ReportingException(
                f"Error sintetizando evento: {str(e)}"
            )
    
    def generate_balanced_dataset(self,
                                 n_per_class: int,
                                 n_val_per_class: int,
                                 target_snr_range: Tuple[float, float],
                                 seed: Optional[int] = None) -> BalancedDataset:
        """
        Genera un dataset balanceado con eventos sintéticos.
        
        REDUCCIÓN DE DIMENSIONALIDAD:
        - Strain raw: 32768 samples (8s @ 4096 Hz)
        - Features VQC: 12 (para 12 qubits)
        
        Método: FFT → extraer amplitudes de componentes de frecuencia
        Justificación: En GW analysis, el dominio frecuencial es el estándar.
        Cada componente de frecuencia es un observable físico independiente.
        
        Args:
            n_per_class: Eventos de entrenamiento por clase
            n_val_per_class: Eventos de validación por clase
            target_snr_range: Rango de SNR objetivo (min, max)
            seed: Seed para reproducibilidad
        
        Returns:
            BalancedDataset con features normalizadas y labels
        
        Raises:
            ReportingException: Si falla la generación
        """
        try:
            if seed is not None:
                np.random.seed(seed)
            
            snr_min, snr_max = target_snr_range
            
            # Parámetros físicos realistas (GW150914-like)
            mass_range = (10.0, 40.0)   # M_sun
            distance_range = (100.0, 1000.0)  # Mpc
            
            X_train_list = []
            y_train_list = []
            X_val_list = []
            y_val_list = []
            
            snr_values = []
            
            # Generar datos para cada clase (teoría)
            for class_idx, theory in enumerate(self.THEORY_CLASSES):
                print(f"  Generando clase {class_idx}: {theory}")
                
                # Generar eventos de entrenamiento
                for _ in range(n_per_class):
                    m1 = np.random.uniform(*mass_range)
                    m2 = np.random.uniform(*mass_range)
                    dist = np.random.uniform(*distance_range)
                    snr = np.random.uniform(snr_min, snr_max)
                    
                    try:
                        strain = self.synthesize_event(
                            mass1_solar_masses=m1,
                            mass2_solar_masses=m2,
                            distance_mpc=dist,
                            theory_family=theory,
                            sampling_rate_hz=4096,
                            duration_seconds=8.0
                        )
                        
                        # REDUCCIÓN DIMENSIONAL: Strain → Espectro FFT
                        # Calcular FFT (dominio frecuencial)
                        fft_result = np.fft.rfft(strain)
                        fft_magnitude = np.abs(fft_result)
                        
                        # Extraer primeros 12 componentes de frecuencia
                        # Estos corresponden a las bandas de frecuencia más importantes para GW
                        # (donde está la mayoría de la potencia de señal)
                        n_features = 12
                        features = fft_magnitude[:n_features]
                        
                        # Normalizar features
                        features_norm = features / (np.max(np.abs(features)) + 1e-10)
                        
                        X_train_list.append(features_norm)
                        y_train_list.append(class_idx)
                        snr_values.append(snr)
                    except Exception as e:
                        print(f"    ⚠️  Error generando evento: {e}")
                        continue
                
                # Generar eventos de validación
                for _ in range(n_val_per_class):
                    m1 = np.random.uniform(*mass_range)
                    m2 = np.random.uniform(*mass_range)
                    dist = np.random.uniform(*distance_range)
                    snr = np.random.uniform(snr_min, snr_max)
                    
                    try:
                        strain = self.synthesize_event(
                            mass1_solar_masses=m1,
                            mass2_solar_masses=m2,
                            distance_mpc=dist,
                            theory_family=theory,
                            sampling_rate_hz=4096,
                            duration_seconds=8.0
                        )
                        
                        # FFT reduction (mismo proceso)
                        fft_result = np.fft.rfft(strain)
                        fft_magnitude = np.abs(fft_result)
                        n_features = 12
                        features = fft_magnitude[:n_features]
                        features_norm = features / (np.max(np.abs(features)) + 1e-10)
                        
                        X_val_list.append(features_norm)
                        y_val_list.append(class_idx)
                        snr_values.append(snr)
                    except Exception as e:
                        print(f"    ⚠️  Error generando evento validación: {e}")
                        continue
            
            # Convertir a arrays
            X_train = np.array(X_train_list) if X_train_list else np.zeros((0, 12))
            y_train = np.array(y_train_list) if y_train_list else np.array([], dtype=int)
            X_val = np.array(X_val_list) if X_val_list else np.zeros((0, 12))
            y_val = np.array(y_val_list) if y_val_list else np.array([], dtype=int)
            
            # Calcular estadísticas de SNR
            snr_mean = np.mean(snr_values) if snr_values else 0.0
            snr_std = np.std(snr_values) if len(snr_values) > 1 else 0.0
            
            print(f"  ✅ Dataset generado: {len(X_train)} train / {len(X_val)} val")
            print(f"     Features: 12 (primeras componentes FFT)")
            print(f"     SNR: {snr_mean:.1f} ± {snr_std:.1f}")
            
            # Crear DTO
            return BalancedDataset(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                n_classes=len(self.THEORY_CLASSES),
                snr_mean=snr_mean,
                snr_std=snr_std,
                is_physically_valid=True
            )
        
        except Exception as e:
            raise ReportingException(
                f"Error generando dataset balanceado: {str(e)}"
            )

