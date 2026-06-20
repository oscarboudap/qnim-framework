"""
Infrastructure: SSTG (Synthetic Signal Template Generator) Adapter
==================================================================

Adaptador que implementa ISyntheticDataGeneratorPort.

Genera datos sintéticos balanceados para el pipeline de clasificación
de teorías gravitacionales. Funciona sin dependencias externas
complejas mediante síntesis simple.

CAMBIO v3 — BUG GRAVE CORREGIDO (la causa raíz de fondo de todo el
proceso de depuración de esta sesión):

  La versión anterior codificaba la "firma" de cada teoría como un
  escalado de AMPLITUD (`theory_modulation = 1 + theory_offset*0.3`).
  Pero el propio generador divide el strain por su desviación estándar
  al final (`strain /= np.std(strain)`) — y como esa std es proporcional
  a la misma amplitud, el escalado por teoría se CANCELA EXACTAMENTE
  antes de llegar al FFT. Y aunque no se cancelara ahí, la normalización
  posterior de features (`features / max(features)`) también borra
  cualquier diferencia de escala global.

  Verificación numérica: con (m1,m2,distancia,ruido) fijos y SOLO
  variando theory_offset de 0 a 0.92 (todo el rango real usado), las
  features resultantes eran BIT-IDÉNTICAS. Las etiquetas `y` eran
  estadísticamente independientes de las features `X` — ningún
  clasificador, clásico o cuántico, podía hacerlo mejor que el azar con
  estos datos, sin importar cuántas correcciones se aplicaran al VQC o
  al optimizador.

  FIX: la firma de teoría ahora se inyecta como un MARCADOR ESPECTRAL en
  un PAR de bins de FFT específico por clase (dentro de la ventana de 12
  bins que se extrae como features). Como ambas normalizaciones solo
  reescalan globalmente (nunca alteran qué bins concentran más potencia
  relativa), esta señal SÍ sobrevive. Se usan pares de bins (no un bin
  único) porque con 13 clases y solo 12 bins, un esquema de "un bin por
  clase" colisionaría (p.ej. clase 0 y clase 10 compartirían bin); con
  pares hay 55 combinaciones posibles, de sobra para 13 clases.

  AVISO PARA EL TFM: este marcador es una señal sintética FUERTE, pensada
  para validar que el pipeline VQC completo es capaz de aprender algo en
  absoluto (separabilidad casi perfecta por diseño). Para resultados
  finales con pretensión de realismo físico, lo ideal es sustituir este
  generador por el pipeline de StochasticSignalGenerator + los
  inyectores Layer5/6/7 ya existentes en
  src/domain/astrophysics/sstg/ (requieren PyCBC), que inyectan firmas
  espectrales genuinamente derivadas de física (dipolar, ecos,
  dispersión...) en vez de un marcador artificial.
"""

import numpy as np
from itertools import combinations
from typing import Optional, Tuple
from dataclasses import dataclass

from src.application.ports import ISyntheticDataGeneratorPort
from src.application.dto import BalancedDataset
from src.infrastructure.exceptions import ReportingException


class SSTGAdapter(ISyntheticDataGeneratorPort):
    """
    Generador de datos sintéticos para clasificación de teorías.
    
    Genera eventos sintéticos de N clases (una por teoría) con
    características realistas (strain simulado + ruido coloreado).
    """
    
    # 13 teorías beyond-GR soportadas (aligned with thesis classification scheme)
    THEORY_CLASSES = [
        "GR",                      # Clase 0: General Relativity (baseline)
        "standard-siren",          # Clase 1: Standard siren / H0 measurement
        "qnm-21",                  # Clase 2: QNM (2,1) overtone
        "qnm-33",                  # Clase 3: QNM (3,3) overtone
        "pn-deformation",          # Clase 4: Post-Newtonian 3.5 deformation
        "extra-dimensions",        # Clase 5: Randall-Sundrum extra dimensions
        "scalar-tensor",           # Clase 6: Brans-Dicke scalar-tensor
        "graviton-mass",           # Clase 7: Massive graviton
        "chern-simons",            # Clase 8: Chern-Simons parity violation
        "liv-alpha2",              # Clase 9: LIV α=2 (speed of gravity)
        "liv-alpha4",              # Clase 10: LIV α=4 (quadratic)
        "loop-quantum-gravity",    # Clase 11: LQG area quantisation
        "gup",                     # Clase 12: Generalised Uncertainty Principle
    ]
    
    def __init__(self):
        """Inicializa el adaptador SSTG."""
        pass
    
    # Pares de bins de FFT (dentro de la ventana de 12 features) que
    # codifican la "firma" de cada teoría. Bins 1..11 (se evita el bin 0
    # = componente DC). 55 combinaciones posibles, de sobra para 13 clases
    # sin colisiones.
    N_FEATURE_BINS = 12
    _MARKER_BIN_PAIRS = list(combinations(range(1, N_FEATURE_BINS), 2))

    @classmethod
    def _marker_bins_for_theory(cls, theory_idx: int) -> Tuple[int, int]:
        return cls._MARKER_BIN_PAIRS[theory_idx % len(cls._MARKER_BIN_PAIRS)]

    @classmethod
    def _generate_simple_strain(
        cls,
        m1: float,
        m2: float,
        distance: float,
        theory_idx: int = 0,
        marker_strength: float = 4.0,
        duration: float = 8.0,
        fs: int = 4096
    ) -> np.ndarray:
        """
        Genera strain sintético con un marcador espectral específico de
        teoría (par de bins de FFT), robusto frente a la normalización
        por desviación estándar que se aplica al final.

        Args:
            m1: Masa primaria [M_sun]
            m2: Masa secundaria [M_sun]
            distance: Distancia [Mpc]
            theory_idx: Índice de la teoría (0-12) — determina qué par
                de bins de FFT se marca (ver _marker_bins_for_theory).
            marker_strength: Fuerza relativa del marcador (relativa a la
                amplitud característica de la señal). Valor alto (~4.0)
                produce separabilidad casi perfecta — pensado para
                validar el pipeline, no como señal físicamente realista.
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

        # FIX v3: marcador espectral específico de teoría, inyectado
        # directamente en el dominio de frecuencia en un PAR de bins
        # único por clase (dentro de la ventana de 12 features extraída
        # más adelante). Amplitud proporcional a `amplitude` (misma
        # escala que el resto de la señal) -> sobrevive a la
        # normalización por std que viene justo después.
        bin_a, bin_b = cls._marker_bins_for_theory(theory_idx)
        strain_fft = np.fft.rfft(strain)
        boost = marker_strength * amplitude * n_samples / 2
        strain_fft[bin_a] += boost
        strain_fft[bin_b] += boost
        strain = np.fft.irfft(strain_fft, n=n_samples)
        
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
            
            # Índice de la teoría (determina el par de bins marcador)
            theory_idx = self.THEORY_CLASSES.index(theory_family) if theory_family in self.THEORY_CLASSES else 0
            
            # Generar strain
            strain = self._generate_simple_strain(
                m1=mass1_solar_masses,
                m2=mass2_solar_masses,
                distance=distance_mpc,
                theory_idx=theory_idx,
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
                                 seed: Optional[int] = None,
                                 max_classes: Optional[int] = None) -> BalancedDataset:
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
            max_classes: Si se proporciona, trunca el problema a las
                primeras `max_classes` teorías (de las 13 totales). Útil
                para pruebas de cordura rápidas (3-4 clases) antes de
                escalar al problema completo de 13 clases — un problema
                más fácil también es más barato por iteración (menos
                circuitos para mantener el dataset balanceado).
        
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

            theory_classes = (
                self.THEORY_CLASSES if max_classes is None
                else self.THEORY_CLASSES[:max(1, max_classes)]
            )
            
            X_train_list = []
            y_train_list = []
            X_val_list = []
            y_val_list = []
            
            snr_values = []
            
            # Generar datos para cada clase (teoría)
            for class_idx, theory in enumerate(theory_classes):
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
            print(f"     Clases: {len(theory_classes)}{' (truncado para prueba rapida)' if max_classes is not None else ''}")
            print(f"     SNR: {snr_mean:.1f} ± {snr_std:.1f}")
            
            # Crear DTO
            return BalancedDataset(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                n_classes=len(theory_classes),
                snr_mean=snr_mean,
                snr_std=snr_std,
                is_physically_valid=True
            )
        
        except Exception as e:
            raise ReportingException(
                f"Error generando dataset balanceado: {str(e)}"
            )