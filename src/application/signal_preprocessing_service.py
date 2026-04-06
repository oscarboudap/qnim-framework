import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from scipy.interpolate import interp1d
from src.domain.astrophysics.entities import GWSignal
from src.domain.astrophysics.value_objects import GPSTime

class SignalPreprocessingService:
    def __init__(self, lowcut: float = 35.0, highcut: float = 350.0):
        self.lowcut = lowcut
        self.highcut = highcut

    def clean_signal(self, signal: GWSignal) -> GWSignal:
        """Flujo completo de limpieza: Whitening -> Bandpass."""
        # 1. Whitening (Blanqueo)
        # Estimamos la densidad espectral de potencia (PSD)
        strain_f = np.fft.rfft(signal.strain)
        psd = np.abs(strain_f)**2
        # Suavizamos y normalizamos en frecuencia
        whitened_f = strain_f / np.sqrt(psd + 1e-20) 
        whitened_strain = np.fft.irfft(whitened_f, n=len(signal.strain))

        # 2. Bandpass Filter (Filtro de banda)
        clean_strain = self._apply_bandpass(whitened_strain, signal.sample_rate)

        # Retornamos una NUEVA entidad (Inmutabilidad)
        return GWSignal(
            strain=clean_strain,
            detector=signal.detector,
            sample_rate=signal.sample_rate,
            gps_start=signal.gps_start
        )

    def _apply_bandpass(self, data: np.ndarray, fs: int) -> np.ndarray:
        nyquist = 0.5 * fs
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, data)